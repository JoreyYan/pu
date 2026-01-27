import os
# —— 在任何 wandb/lightning 导入之前 —— 避免 fork 复制大内存
os.environ.setdefault("WANDB_START_METHOD", "thread")
# 可选：需要离线再打开
# os.environ.setdefault("WANDB_MODE", "offline")

import time
import torch
import hydra
import typing
from omegaconf import DictConfig, OmegaConf

# PyTorch Lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import DummyLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

# 你的工程模块
from data.datasets import PdbDataset
from data.protein_dataloader import ProteinData
from models.flow_module import FlowModule
from experiments import utils as eu

def is_rank0() -> bool:
    """DDP 场景下更稳的 rank0 判断；未初始化分布式时默认 True。"""
    # 优先 RANK，其次 LOCAL_RANK；都没有时视为 0
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision("high")

# Allow Hydra configs inside checkpoints when PyTorch uses safe deserialization.
try:
    from omegaconf.base import ContainerMetadata  # type: ignore
except ImportError:
    ContainerMetadata = None  # type: ignore

try:
    globals_to_allow = [DictConfig, typing.Any, dict, list, tuple]
    if ContainerMetadata is not None:
        globals_to_allow.append(ContainerMetadata)
    torch.serialization.add_safe_globals(globals_to_allow)
except AttributeError:
    pass

def set_global_seed(seed: int = 42):
    """设置全局随机种子（Python/NumPy/PyTorch/CUDA）。"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["GLOBAL_RANDOM_SEED"] = str(seed)
    print(f"[Seed] Global random seed set to {seed}")

class Experiment:
    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        self._task = self._data_cfg.task

        self._setup_dataset()
        self._datamodule: LightningDataModule = ProteinData(
            data_cfg=self._data_cfg,
            train_dataset=self._train_dataset,
            valid_dataset=self._valid_dataset,
        )

        self._train_device_ids = eu.get_available_device(self._exp_cfg.num_devices)
        log.info(f"Training with devices: {self._train_device_ids}")

        self._module: LightningModule = FlowModule(self._cfg)
        # 如果提供了 warm_start，我们只加载模型权重，不尝试恢复优化器/回调等（避免安全反序列化问题）
        if self._exp_cfg.get("warm_start", None):
            try:
                eu.load_partial_state_dict(
                    self._module,
                    self._exp_cfg.warm_start,
                    lightning_key="state_dict",
                    map_location="cpu",
                )
                log.info(f"Warm-started weights from {self._exp_cfg.warm_start}")
            except Exception as e:
                log.error(f"Warm-start failed: {e}")

    def _setup_dataset(self):
        if self._data_cfg.dataset == "scope":
            self._train_dataset, self._valid_dataset = eu.dataset_creation(
                ScopeDataset, self._cfg.scope_dataset, self._task
            )
        elif self._data_cfg.dataset == "pdb":
            self._train_dataset, self._valid_dataset = eu.dataset_creation(
                PdbDataset, self._cfg.pdb_dataset, self._task
            )
        else:
            raise ValueError(f"Unrecognized dataset {self._data_cfg.dataset}")

    def _build_logger(self):
        """仅在 rank0 构造 WandbLogger；其余 rank 返回 DummyLogger。"""
        if self._exp_cfg.debug:
            return None  # debug 下禁用 logger
        if is_rank0():
            # 你可以在 yaml 的 experiment.wandb 里配置 project/name 等
            # 这里加几个更稳的默认 settings
            wandb_kwargs = dict(self._exp_cfg.wandb)
            # 降低开销：不打包代码、不采系统指标（如需可在配置里覆盖）
            wandb_kwargs.setdefault("save_code", False)
            try:
                import wandb  # 局部导入，避免未用时引入
                wandb_kwargs.setdefault(
                    "settings",
                    wandb.Settings(start_method="thread", _disable_stats=True),
                )
            except Exception:
                pass
            logger = WandbLogger(
                **wandb_kwargs,
                log_model=False,  # 仅保留标量日志
            )
            return logger
        else:
            return DummyLogger()

    @rank_zero_only
    def _save_and_log_cfg(self, logger, ckpt_dir: str):
        """只在 rank0 保存配置并记录到 logger（不触碰 .experiment）。"""
        os.makedirs(ckpt_dir, exist_ok=True)
        cfg_path = os.path.join(ckpt_dir, "config.yaml")
        with open(cfg_path, "w") as f:
            OmegaConf.save(config=self._cfg, f=f.name)

        cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
        flat_cfg = dict(eu.flatten_dict(cfg_dict))
        if logger is not None:
            # 通过 Lightning API 写入超参，内部自带 rank-zero 保护
            logger.log_hyperparams(flat_cfg)

    def train(self):
        callbacks = []

        # Debug: 关多进程，便于排错 & 降低内存
        if self._exp_cfg.debug:
            log.info("Debug mode: disable W&B, set DataLoader.num_workers=0")
            self._data_cfg.loader.num_workers = 0

        # Logger（rank0: WandbLogger；其他: Dummy 或 None）
        logger = self._build_logger()

        # Checkpoint
        ckpt_dir = self._exp_cfg.checkpointer.dirpath
        log.info(f"Checkpoints saved to {ckpt_dir}")
        callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))

        # 仅 rank0 保存配置并通过 logger 记录
        self._save_and_log_cfg(logger, ckpt_dir)

        # Trainer
        trainer: Trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            # devices=self._train_device_ids,  # 如需手动指定
        )

        # 仅训练（不从 Lightning ckpt 全量恢复），避免 torch.load 的安全反序列化限制
        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
            ckpt_path=None,
        )

@hydra.main(version_base=None, config_path="../configs", config_name="/Train_esmsd.yaml")
def main(cfg: DictConfig):
    # 可选：warm-start 合并
    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), "config.yaml"
        )
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # 合并模型段配置（保持新字段）
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f"Loaded warm start config from {warm_start_cfg_path}")

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    set_global_seed(0)
    main()
