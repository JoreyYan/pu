import os
import time
import torch
import hydra
from omegaconf import DictConfig

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from data.datasets import ScopeDataset, PdbDataset,BaseDataset
from data.protein_dataloader import ProteinData
from models.flow_module import FlowModule
from experiments import utils as eu

torch.set_float32_matmul_precision('high')

# PyTorch 2.6: torch.load defaults to weights_only=True, which requires allowlisting
# any pickled custom globals. Lightning checkpoints may include OmegaConf DictConfig.
# Allowlist it to avoid UnpicklingError when loading checkpoints.
try:
    import torch.serialization as _ts
    from omegaconf import DictConfig as _OmegaDictConfig
    from omegaconf import ListConfig as _OmegaListConfig
    from omegaconf.base import ContainerMetadata as _OmegaContainerMetadata
    from typing import Any as _TypingAny
    _ts.add_safe_globals([_OmegaDictConfig, _OmegaListConfig, _OmegaContainerMetadata, _TypingAny])
except Exception:
    # If running with older PyTorch (<2.6) or environments without this API, ignore.
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


class InferenceExperiment:
    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._task = self._data_cfg.task

        self._predict_dataset = self._build_predict_dataset()
        self._prepare_inference_dir()

        self._datamodule: LightningDataModule = ProteinData(
            data_cfg=self._data_cfg,
            train_dataset=None,
            valid_dataset=self._predict_dataset,
            predict_dataset=self._predict_dataset,
        )

        self._module: LightningModule = self._load_module()
        self._module.eval()

    def _build_predict_dataset(self):
        if self._data_cfg.dataset == 'scope':
            _, eval_dataset = eu.dataset_creation(
                ScopeDataset, self._cfg.scope_dataset, self._task)
        elif self._data_cfg.dataset == 'pdb':
            _, eval_dataset = eu.dataset_creation(
                PdbDataset, self._cfg.pdb_dataset, self._task)
        elif self._data_cfg.dataset == 'val':
            eval_dataset = eu.dataset_creation(
                BaseDataset, self._cfg.val_dataset, self._task,_is_predict=True)

        else:
            raise ValueError(f'Unrecognized dataset {self._data_cfg.dataset}')
        return eval_dataset

    def _prepare_inference_dir(self):
        exp_cfg = self._cfg.experiment
        inference_dir = exp_cfg.get('inference_dir', None)
        if not inference_dir:
            timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
            inference_dir = os.path.join(os.getcwd(), 'inference_outputs', timestamp)
            exp_cfg.inference_dir = inference_dir
        os.makedirs(inference_dir, exist_ok=True)

    def _load_module(self) -> LightningModule:
        exp_cfg = self._cfg.experiment
        ckpt_path = exp_cfg.get('ckpt_path', None)
        if ckpt_path is None:
            ckpt_path = exp_cfg.get('warm_start', None)
        if ckpt_path:
            try:
                # First try the standard Lightning load (will use torch.load with weights_only=True on PyTorch>=2.6)
                return FlowModule.load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    cfg=self._cfg,
                )
            except Exception as e:
                # Fallback: explicitly load with weights_only=False (trusted local checkpoint), then load state_dict
                print(f"[Warn] load_from_checkpoint failed: {e}\n[Info] Falling back to manual torch.load(weights_only=False).")
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                model = FlowModule(self._cfg)
                state_dict = ckpt.get('state_dict', ckpt)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"[Info] State dict loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                if missing:
                    print(f"[Debug] Missing keys (first 10): {missing[:10]}")
                if unexpected:
                    print(f"[Debug] Unexpected keys (first 10): {unexpected[:10]}")
                return model
        return FlowModule(self._cfg)

    def infer(self):
        trainer = Trainer(
            **self._cfg.experiment.trainer,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=False,
        )

        trainer.predict(
            model=self._module,
            datamodule=self._datamodule,
            ckpt_path=None,
        )

@hydra.main(version_base=None, config_path="../configs", config_name="Infer_SH.yaml")
def main(cfg: DictConfig):
    exp = InferenceExperiment(cfg=cfg)
    exp.infer()

if __name__ == "__main__":
    set_global_seed(0)
    main()
