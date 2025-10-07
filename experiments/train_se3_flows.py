import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Pytorch lightning imports
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from data.datasets import ScopeDataset, PdbDataset
from data.protein_dataloader import ProteinData
from models.flow_module import FlowModule
from experiments import utils as eu
import wandb

def is_rank0():
    # 未初始化分布式时默认 True；已初始化时用 RANK/LOCAL_RANK
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
os.environ.setdefault("WANDB_START_METHOD", "thread")
log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')
import os
os.environ["WANDB_MODE"] = "online"

# 添加一个全局变量来存储seed
GLOBAL_SEED = None
def set_global_seed(seed=42):
    """设置全局随机种子并将其存储在环境变量中"""
    # 将种子存储在环境变量中
    os.environ['GLOBAL_RANDOM_SEED'] = str(seed)

    import random
    import numpy as np
    import torch

    # Set Python's random seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)

    # Set PyTorch's random seed
    torch.manual_seed(seed)

    # Set CUDA's random seed if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        # Additional settings for CUDA determinism
        # Note: This may impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Global random seed set to {GLOBAL_SEED}")
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
            valid_dataset=self._valid_dataset
        )
        self._train_device_ids = eu.get_available_device(self._exp_cfg.num_devices)
        log.info(f"Training with devices: {self._train_device_ids}")
        self._module: LightningModule = FlowModule(self._cfg)

    def _setup_dataset(self):
        if self._data_cfg.dataset == 'scope':
            self._train_dataset, self._valid_dataset = eu.dataset_creation(
                ScopeDataset, self._cfg.scope_dataset, self._task)
        elif self._data_cfg.dataset == 'pdb':
            self._train_dataset, self._valid_dataset = eu.dataset_creation(
                PdbDataset, self._cfg.pdb_dataset, self._task)
        else:
            raise ValueError(f'Unrecognized dataset {self._data_cfg.dataset}') 
        
    def train(self):
        callbacks = []
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
            # self._train_device_ids = [self._train_device_ids[0]]
            self._data_cfg.loader.num_workers = 0
        else:
            logger = WandbLogger(
                **self._exp_cfg.wandb,
                log_model=False
            )
            
            # Checkpoint directory.
            ckpt_dir = self._exp_cfg.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")
            
            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))
            
            # Save config only for main process.
            local_rank = os.environ.get('LOCAL_RANK', 0)
            if local_rank == 0:
                cfg_path = os.path.join(ckpt_dir, 'config.yaml')
                with open(cfg_path, 'w') as f:
                    OmegaConf.save(config=self._cfg, f=f.name)
                cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
                flat_cfg = dict(eu.flatten_dict(cfg_dict))
                if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                    logger.experiment.config.update(flat_cfg,allow_val_change=True)
        


        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            # gradient_clip_val=0.5,  # 裁剪梯度范数为1，可调整
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            # devices=self._train_device_ids,
        )
        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )


@hydra.main(version_base=None, config_path="../configs", config_name="Train_SH.yaml")
def main(cfg: DictConfig):

    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    set_global_seed(0)
    main()
