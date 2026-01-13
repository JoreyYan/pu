from typing import Any
import torch
import torch.nn.functional as F
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
import torch.distributed as dist
from pytorch_lightning import LightningModule
from analysis import metrics
from analysis import utils as au
# from models.flow_model import VAE,SHDecoder,SHframe_fbb
# from models.ff2 import FF2Model
from models import utils as mu
from data.interpolant import Interpolant
from data import utils as du
from data import all_atom
from data import so3_utils
from data import residue_constants
from experiments import utils as eu
from omegaconf import OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger
from chroma.layers.structure.backbone import FrameBuilder
from models.loss import type_ce_loss, type_top1_acc, compute_CE_perplexity, torsion_angle_loss, make_w_l, \
    std_lr_from_batch_masked, SideAtomsIGALoss_Final,BackboneGaussianAutoEncoderLoss
from models.shlossv2 import sh_loss_with_masks, sh_mse_loss
# from data.sh_density import sh_density_from_atom14_with_masks_clean
import torch.nn.functional as F
from models.loss import huber, pairwise_distance_loss, backbone_mse_loss
from openfold.np.residue_constants import restype_name_to_atom14_names
from models.AELOSS import seg_up_losses_angstrom,seg_ae_losses
from models.flow_model import SideAtomsFlowModel,SideAtomsIGAModel
from models.HGF_flow_model_structured import HierarchicalGaussianFieldModel
from models.shattetnion.ShDecoderSidechain import SHSidechainDecoder, DynamicKSidechainDecoder, assemble_atom14_with_CA, \
    SHGeoResHead
from openfold.utils import loss as openfold_loss
from openfold.utils import rigid_utils
import openfold.np.residue_constants as rc
from openfold.np.protein import from_prediction, to_pdb
from data.all_atom import atom14_to_atom37, make_new_atom14_resid
import data.all_atom as all_atom
import matplotlib
from experiments.utils import _load_submodule_from_ckpt, load_partial_state_dict
matplotlib.use('Agg')  # <--- 必须加这句，放在 import pyplot 之前
import matplotlib.pyplot as plt
import wandb

def compute_gamma(vq_loss, min_gamma=0.1, max_gamma=1.0, min_vq=3, max_vq=10):
    if vq_loss <= min_vq:
        return min_gamma
    elif vq_loss >= max_vq:
        return max_gamma
    else:
        # 线性插值
        ratio = (vq_loss - min_vq) / (max_vq - min_vq)
        gamma = min_gamma + ratio * (max_gamma - min_gamma)
        return gamma


def save_pdb(xyz, pdb_out="out.pdb"):
    pdb_out = pdb_out
    ATOMS = ["N", "CA", "C"]
    out = open(pdb_out, "w")
    k = 0
    a = 0
    for x, y, z in xyz:
        out.write(
            "ATOM  %5d  %2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
            % (k + 1, ATOMS[k % 3], "GLY", "A", a + 1, x, y, z, 1, 0)
        )
        k += 1
        if k % 3 == 0: a += 1
    out.close()


def save_4pdb(xyz, pdb_out="out.pdb"):
    pdb_out = pdb_out
    ATOMS = ["N", "CA", "C", "O"]
    out = open(pdb_out, "w")
    k = 0
    a = 0
    for x, y, z in xyz:
        out.write(
            "ATOM  %5d  %2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
            % (k + 1, ATOMS[k % 4], "GLY", "A", a + 1, x, y, z, 1, 0)
        )
        k += 1
        if k % 4 == 0: a += 1
    out.close()


class FlowModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._full_cfg = cfg
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant

        # # Set-up vector field prediction model
        # self._weights_path='/home/junyu/project/FoldFlow/ckpt/ff2_base.pth'
        # weights_pkl = du.read_pkl(
        #     self._weights_path, use_torch=True, map_location=self.device
        # )
        # deps = FF2Dependencies(cfg.ff2)
        # self.model = FF2Model.from_ckpt(weights_pkl, deps)

        # self.model=SHDecoder(cfg.model)




        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant, cfg.experiment.task, cfg.experiment.noise_scheme)

        # self.frames=FrameBuilder()

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self._val_ref_batch = None
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None
        self._inference_run_dir = None

        # self.SHDecoder=SHDecoder(cfg.model)

        # self.DynamicKSidechainDecoder=DynamicKSidechainDecoder(SHSidechainDecoder(**cfg.model.sh))

        # test_torision rec
        #self.model=SHGeoResHead( C=cfg.model.sh.C, L_max=cfg.model.sh.L_max, R_bins=cfg.model.sh.R_bins)
        # weight='/home/junyu/project/protein-frame-flow-u/experiments/ckpt/se3-fm_sh/hallucination_pdb_SHaapred/2025-08-24_21-04-49/last.ckpt'
        # _load_submodule_from_ckpt(self.PredHead, weight, lightning_key="state_dict", source_prefix=None)
        #
        # self.PredHead.requires_grad_(False)  # 冻结权重
        # self.PredHead.eval()  # 关掉 Dropout/BN 的统计更新

        self.model = HierarchicalGaussianFieldModel(cfg.model)
        #self.model =SideAtomsFlowModel_backup(cfg.model)
        # mw='/home/junyu/project/protein-frame-flow-u/experiments/ckpt/se3-fm_sh/pdb_seperated_Rm0_t0/2025-08-25_00-01-29/last.ckpt'
        # _load_submodule_from_ckpt(self.model, mw, lightning_key="state_dict", source_prefix=None)

        # Initialize IGA Loss
        # self.iga_loss_fn = BackboneGaussianAutoEncoderLoss()

    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self._exp_cfg.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self._exp_cfg.checkpointer.dirpath
            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir
    @torch.no_grad()
    def log_ca_distance_map_wandb(self, pred_ca, gt_ca, mask, step, log_key="train/ca_dist_map"):
        """
        pred_ca: [N, 3]
        gt_ca:   [N, 3]
        mask:    [N]
        """
        # 1. 截取有效残基 (去除 padding)
        valid_idx = mask.bool()
        p = pred_ca[valid_idx].detach().cpu()
        g = gt_ca[valid_idx].detach().cpu()

        # 2. 计算距离矩阵 (N_valid, N_valid)
        # cdist 计算成对欧氏距离
        d_pred = torch.cdist(p.unsqueeze(0), p.unsqueeze(0)).squeeze(0)
        d_gt = torch.cdist(g.unsqueeze(0), g.unsqueeze(0)).squeeze(0)

        # 3. 绘图
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Pred Map
        im0 = axes[0].imshow(d_pred.numpy(), origin='lower', cmap='viridis')
        axes[0].set_title(f"Prediction (Step {step})")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # GT Map
        im1 = axes[1].imshow(d_gt.numpy(), origin='lower', cmap='viridis')
        axes[1].set_title("Ground Truth")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # 4. Log 到 WandB
        # 注意：这里需要确保 logger 是 WandBLogger
        if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
            self.logger.experiment.log({log_key: wandb.Image(fig)}, step=step)

        # 5. 关闭图像防止内存泄漏
        plt.close(fig)
    def _get_inference_run_dir(self):
        base_dir = self.inference_dir
        if base_dir is None:
            base_dir = os.path.join(self.checkpoint_dir, 'inference')
            os.makedirs(base_dir, exist_ok=True)
        if self._inference_run_dir is None:
            run_name = getattr(self._exp_cfg.wandb, 'name', 'sh_infer')
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            step = getattr(self, 'global_step', 0)
            run_dir = os.path.join(base_dir, f"{run_name}_step{step}_{timestamp}")
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    os.makedirs(run_dir, exist_ok=True)
                dist.barrier()
            else:
                os.makedirs(run_dir, exist_ok=True)
            self._inference_run_dir = run_dir
        return self._inference_run_dir

    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def model_step_typePred(self, noisy_batch: Any):
        logits = self.model(noisy_batch['density'], noisy_batch['res_mask'], noisy_batch['aatype']
                            , noisy_batch['rotmats_1'], noisy_batch['trans_1'])
        loss = type_ce_loss(logits, noisy_batch['aatype'], node_mask=noisy_batch['res_mask'], label_smoothing=0.05)
        acc = type_top1_acc(logits, noisy_batch['aatype'])
        return {'aaloss': loss,
                'aa_acc': acc}

    def model_step_decoder(self, noisy_batch: Any):

        training_cfg = self._exp_cfg.training

        # self.SHDecoder(noisy_batch, noisy_batch['normalize_density'], noisy_batch['res_mask'],
        #                noisy_batch['aatype'],
        #                noisy_batch['rotmats_1'],
        #                noisy_batch['trans_1'])

        # Tau threshold: 设置为0关闭threshold
        tau = self._exp_cfg.training.get('sh_tau_threshold', 0.0)
        # if tau > 0:
        #     tau_mask = noisy_batch['density_mask'] * (torch.abs(noisy_batch['normalize_density']) > tau)
        #     sh = noisy_batch['normalize_density'] * tau_mask
        # else:
        #     # 不使用tau threshold，只用density_mask
        #     sh = noisy_batch['normalize_density'] * noisy_batch['density_mask']

        noisy_batch['r3_t']=noisy_batch['t']
       # noisy_batch['normalize_density']=sh
        logits, atoms14 = self.model(noisy_batch)

        # TYPELOSS
        typeloss, perplexity = compute_CE_perplexity(logits, noisy_batch['aatype'], mask=noisy_batch['res_mask'])
        acc = type_top1_acc(logits, noisy_batch['aatype'])

        # ANGELLOSS
        # chiloss=torsion_angle_loss(tor_pred,noisy_batch['torsion_angles'],noisy_batch['torsion_alt_angles'],noisy_batch['torsion_mask']).mean()

        # ANGEL TO FRAMES
        # identity=rigid_utils.Rigid.identity(noisy_batch['aatype'].shape,device=noisy_batch['aatype'].device)
        # rigid=all_atom.torsion_angles_to_frames(identity,tor_pred,noisy_batch['aatype'])
        # atoms14_pred_local=all_atom.frames_to_atom14_pos(rigid,noisy_batch['aatype'])*noisy_batch['atom14_gt_exists'][...,None]

        atoms14_pred_local = atoms14 * training_cfg.bb_atom_scale * noisy_batch['atom14_gt_exists'][..., None]

        # BACKBONE LOSS
        atoms14_local = noisy_batch['atoms14_local'] * noisy_batch['atom14_gt_exists'][..., None]

        local_mse_loss = backbone_mse_loss(atoms14_local, atoms14_pred_local, noisy_batch['atom14_gt_exists'],
                                           bb_atom_scale=training_cfg.bb_atom_loss_weight).mean()
        local_pair_loss = pairwise_distance_loss(atoms14_local,
                                                 atoms14_pred_local.clone(),
                                                 noisy_batch['atom14_gt_exists'], use_huber=False).mean()
        local_huber_loss = huber(atoms14_pred_local, atoms14_local, noisy_batch['atom14_gt_exists'])

        atomsloss = local_mse_loss + local_huber_loss + local_pair_loss
        loss = atomsloss * training_cfg.atom_loss_weight + typeloss * training_cfg.type_loss_weight  # +chiloss*training_cfg.chil_loss_weight

        return {
            'local_mse_loss': local_mse_loss.detach(),
            'local_pair_loss': local_pair_loss.detach(),
            'local_huber_loss': local_huber_loss.detach(),
            # 'chiloss':chiloss.detach(),
            'typeloss': typeloss.detach(),
            'loss': loss,
            'aa_acc': acc.detach(),
            'perplexity': perplexity
        }



    def model_step_shdiffusion(self, batch: Any):
        """
        SH 扩散训练：对 SH 加噪再解码，SNR 加权 + 额外 SH loss。
        """
        noisy_batch = self.interpolant.corrupt_batch(batch)
        batch['t'] = noisy_batch.get('t', batch.get('t'))
        if 'sh_t' not in noisy_batch:
            raise RuntimeError("Expected 'sh_t' in corrupted batch for shdiffusion task.")
        noisy_batch['r3_t']=noisy_batch['t']
        noisy_batch['normalize_density'] = noisy_batch['sh_t']
        training_cfg = self._exp_cfg.training

        tau = training_cfg.get('sh_tau_threshold', 0.0)
        if tau > 0:
            tau_mask = noisy_batch['density_mask'] * (torch.abs(noisy_batch['normalize_density']) > tau)
            sh = noisy_batch['normalize_density'] * tau_mask
        else:
            sh = noisy_batch['normalize_density'] * noisy_batch['density_mask']

        logits, atoms14 = self.model(noisy_batch)
        if not torch.isfinite(atoms14).all():
            raise ValueError("atoms14 contains NaN/Inf")

        typeloss, perplexity = compute_CE_perplexity(logits, batch['aatype'], mask=batch['res_mask'])
        acc = type_top1_acc(logits, batch['aatype'])

        atom_exists_mask = batch['atom14_gt_exists'][..., None]
        atoms14_pred_local = atoms14 * training_cfg.bb_atom_scale * atom_exists_mask
        atoms14_gt_local = batch['atoms14_local'] * atom_exists_mask
        atoms14_pred_local_no_snr = atoms14_pred_local.clone()
        atoms14_gt_local_no_snr = atoms14_gt_local.clone()

        snr = None
        if training_cfg.get('use_snr_weight', False) and 't' in noisy_batch:
            t_snr = noisy_batch['t'].to(atoms14_pred_local.dtype)
            eps = 1e-6
            t_clip = training_cfg.get('snr_t_clip', 0.99)
            snr = (t_snr.clamp(min=eps) / (1.0 - t_snr.clamp(max=t_clip) + eps)).sqrt()
            snr = snr[:, None, None]
            atoms14_pred_local = atoms14_pred_local * snr
            atoms14_gt_local = atoms14_gt_local * snr

        local_mse_loss = backbone_mse_loss(
            atoms14_gt_local, atoms14_pred_local, batch['atom14_gt_exists'],
            bb_atom_scale=training_cfg.bb_atom_loss_weight
        ).mean()
        local_pair_loss = pairwise_distance_loss(
            atoms14_gt_local, atoms14_pred_local.clone(), batch['atom14_gt_exists'], use_huber=False
        ).mean()
        local_huber_loss = huber(atoms14_pred_local, atoms14_gt_local, batch['atom14_gt_exists'])
        atomsloss = local_mse_loss + local_pair_loss + local_huber_loss
        per_example_atom_mse = (
            ((atoms14_pred_local_no_snr - atoms14_gt_local_no_snr) ** 2) * atom_exists_mask
        ).sum(dim=(1, 2, 3)) / atom_exists_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)

        # Track per-atom RMS errors (atoms 3-13) to monitor distal stability
        atom_sq_error = ((atoms14_pred_local_no_snr - atoms14_gt_local_no_snr) ** 2).sum(dim=-1)
        per_atom_side_mse = {}
        for atom_idx in range(3, 14):
            mask_atom = batch['atom14_gt_exists'][..., atom_idx]
            denom_atom = mask_atom.sum().clamp_min(1.0)
            mse_atom = (atom_sq_error[..., atom_idx] * mask_atom).sum() / denom_atom
            per_atom_side_mse[f'atom{atom_idx:02d}_mse'] = mse_atom.detach()

        if not torch.isfinite(atoms14_pred_local_no_snr).all():
            raise ValueError("atom14_pred contains NaN/Inf")

        clean_sh = batch.get('normalize_density', None)
        if clean_sh is not None:
            pred_sh, *_ = sh_density_from_atom14_with_masks_clean(
                atoms14_pred_local_no_snr,
                batch['atom14_element_idx'],
                batch['atom14_gt_exists'],
                L_max=self._model_cfg.sh.L_max,
                R_bins=self._model_cfg.sh.R_bins,

            )
            pred_sh = pred_sh / torch.sqrt(torch.tensor(4 * torch.pi))

            if clean_sh is not None:
                if not torch.isfinite(clean_sh).all():
                    raise ValueError("clean_sh contains NaN/Inf")

                if not torch.isfinite(pred_sh).all():
                    raise ValueError("pred_sh contains NaN/Inf")

            density_mask = batch['density_mask'].to(pred_sh.dtype).expand_as(pred_sh)
            sh_diff = torch.abs((pred_sh - clean_sh) * density_mask)
            sh_diff_flat = sh_diff.reshape(sh_diff.shape[0], -1)
            mask_flat = density_mask.reshape(density_mask.shape[0], -1)
            sh_loss_per_example = sh_diff_flat.sum(dim=-1) / mask_flat.sum(dim=-1).clamp_min(1.0)
            # if snr is not None:
            #     snr_scalar = snr.view(snr.shape[0], -1)[:, 0]
            #     sh_loss_per_example = sh_loss_per_example * snr_scalar
            sh_loss = sh_loss_per_example.mean()
        else:
            sh_loss = torch.tensor(0.0, device=atomsloss.device)


        sh_huber= huber(pred_sh, clean_sh,density_mask)
        sh_grad_norm = torch.tensor(0.0, device=atomsloss.device)
        if training_cfg.get('log_sh_grad_norm', False) and sh_huber.requires_grad:
            sh_grad = torch.autograd.grad(
                sh_huber,
                atoms14_pred_local_no_snr,
                retain_graph=True,
                allow_unused=False
            )[0]
            if sh_grad is not None:
                sh_grad_norm = torch.linalg.norm(sh_grad)

        loss = (
            atomsloss * training_cfg.atom_loss_weight
            + typeloss * training_cfg.type_loss_weight

        )



        aux = {
            'local_mse_loss': local_mse_loss.detach(),
            'local_pair_loss': local_pair_loss.detach(),
            'local_huber_loss': local_huber_loss.detach(),
            'typeloss': typeloss.detach(),
            'aa_acc': acc.detach(),
            'perplexity': perplexity,
            'sh_loss': sh_loss.detach(),
            'sh_huber': sh_huber.detach(),
             'sh_grad_norm': sh_grad_norm.detach(),
            'loss': loss,
            'atom_mse_per_example': per_example_atom_mse.detach(),
        }
        return aux

    def model_step_fbb(self, batch: Any, prob=None, return_outputs=False):
        """
        FBB training step using SideAtomsIGALoss_Final.
        Simplified logic: model forward -> loss calculation -> return metrics.

        Args:
            batch: input batch
            prob: corruption probability
            return_outputs: if True, also return model outputs (for validation saving)

        Returns:
            metrics: loss metrics dict
            (optional) outputs: dict with pred_atoms, logits, etc. (if return_outputs=True)
        """
        training_cfg = self._exp_cfg.training
        if self._exp_cfg.task in ('diffusion', 'shdiffusion'):
            raise RuntimeError('model_step_fbb is not intended for diffusion tasks')

        # Corrupt batch (add noise to sidechain)
        noisy_batch = self.interpolant.fbb_corrupt_batch(batch,prob)

        # Model forward
        outs,reg_down,downmetric = self.model(noisy_batch,step=self.global_step,total_steps=100000)



        # ==========================================================
        # [新增] WandB Visualization Logic
        # ==========================================================
        # 设置记录频率，例如每 100 step 记录一次
        # 必须确保只在 rank 0 记录 (如果不使用 DDP，self.global_rank 默认为 0)
        log_img_every = 500

        # 1. 获取 Batch 中第一个样本的数据
        # r_res: OffsetGaussianRigid
        pred_ca = outs["x_hat"][0]

        # # 获取 Pred CA (Translation)
        # if hasattr(r_pred, "get_trans"):
        #     pred_ca = r_pred.get_trans()[0]  # [N, 3]
        # else:
        #     pred_ca = r_pred.trans[0]

        # 获取 GT CA (必须是去中心化后的 noisy_batch['trans_1'])
        gt_ca = noisy_batch['trans_1'][0]  # [N, 3]

        # 获取 Mask
        # 注意：如果使用了 update_mask，最好只看 update 的部分，或者看全部 res_mask
        # 这里使用 res_mask 看整体结构
        mask = noisy_batch['res_mask'][0]  # [N]
        # [Debug Start]
        pred_mean = pred_ca.mean(dim=(0, 1))
        gt_mean = gt_ca.mean(dim=(0, 1))
        pred_abs_mean = pred_ca.abs().mean()
        gt_abs_mean = gt_ca.abs().mean()

        # print(f"\n[DEBUG CHECK]")
        # print(f"GT Center: {gt_mean.tolist()} (Should be near 0)")
        # print(f"Pred Center: {pred_mean.tolist()} (Should be near 0)")
        # print(f"GT Scale (Abs Mean): {gt_abs_mean.item():.2f} (Typical: 15-30)")
        # print(f"Pred Scale (Abs Mean): {pred_abs_mean.item():.2f}")

        if (self.global_step % log_img_every == 0) and (self.global_rank == 0) and isinstance(self.logger, WandbLogger):
            try:


                # 2. 调用绘图
                self.log_ca_distance_map_wandb(pred_ca, gt_ca, mask, self.global_step)
                self.logger.experiment.log({"train/Asoft_down_map": wandb.Image(outs["fig_dwon"])}, step=self.global_step)



            except Exception as e:
                print(f"[Warning] Failed to log distance map: {e}")
        # ==========================================================


        pred_ca = outs["x_hat"][0]

        # # 获取 Pred CA (Translation)
        # if hasattr(r_pred, "get_trans"):
        #     pred_ca = r_pred.get_trans()[0]  # [N, 3]
        # else:
        #     pred_ca = r_pred.trans[0]

        # 获取 GT CA (必须是去中心化后的 noisy_batch['trans_1'])
        gt_ca = noisy_batch['trans_1'][0]  # [N, 3]

        # 获取 Mask
        # 注意：如果使用了 update_mask，最好只看 update 的部分，或者看全部 res_mask
        # 这里使用 res_mask 看整体结构
        mask = noisy_batch['res_mask'][0]  # [N]



        # [Debug Start]
        pred_mean = pred_ca.mean(dim=(0, 1))
        gt_mean = gt_ca.mean(dim=(0, 1))
        pred_abs_mean = pred_ca.abs().mean()
        gt_abs_mean = gt_ca.abs().mean()

        ca_metric={
            "pred_mean":pred_mean,
            "gt_mean":gt_mean,
            "Pred Scale":pred_abs_mean,
            "GT Scale":gt_abs_mean,
        }


        # Calculate loss using IGA Loss
        metrics = seg_ae_losses(x_gt=noisy_batch['trans_1'],node_mask=noisy_batch['res_mask'],**outs)
        metrics = metrics |ca_metric|downmetric|reg_down

        metrics['loss']=metrics['loss']+reg_down["total"]

        if return_outputs:
            # 返回 metrics 和 outputs（用于 validation 保存结构）
            return metrics, outs, noisy_batch
        else:
            # 只返回 metrics（训练时）
            return metrics
    def model_step_allatoms(self, batch: Any, prob=None):

        training_cfg = self._exp_cfg.training


        noisy_batch = self.interpolant.allatoms_corrupt_batch(batch)



        # zero-like SC (scaled domain); backbone part not used but keep zeros for simplicity
        if 'atoms14_local_t' in noisy_batch:
            noisy_batch['atoms14_local_sc'] = torch.zeros_like(noisy_batch['atoms14_local_t'])
        else:
            noisy_batch['atoms14_local_sc'] = torch.zeros_like(noisy_batch['atoms14_local'])

        outs = self.model(noisy_batch)
        speed_vectors = outs['speed_vectors']  # [B,N,11,3]
        speed_pred = outs['speed_pred']  # [B,N,11]
        logits = outs.get('logits')

        loss_mask = noisy_batch['update_mask'] * noisy_batch['res_mask']

        target_vectors = noisy_batch['v_t'][..., 3:, :]
        target_speeds = torch.norm(target_vectors, dim=-1)

        if 'r3_t' in noisy_batch:
            r3_t = noisy_batch['r3_t'].to(target_vectors.dtype)
            eps = 1e-6
            t_clip = getattr(training_cfg, 'snr_t_clip', 0.99)
            snr_weight = (r3_t.clamp(min=eps) / (1.0 - r3_t.clamp(max=t_clip) + eps))[..., None]
            snr_weight = snr_weight.clamp(min=1.0,max=10.0)
            snr_weight=torch.sqrt(snr_weight)
            target_vectors = target_vectors * snr_weight[..., None]
            speed_vectors = speed_vectors * snr_weight[..., None]
            target_speeds = target_speeds * snr_weight
            speed_pred = speed_pred * snr_weight

        exists_mask = noisy_batch['atom14_gt_exists'][..., 3:].bool()
        atom_level_mask = (exists_mask & loss_mask.bool()[..., None]).to(speed_pred.dtype)

        mse_speed = F.mse_loss(speed_pred, target_speeds, reduction='none')
        denom = atom_level_mask.sum().clamp_min(1.0)
        speed_loss = (mse_speed * atom_level_mask).sum() / denom

        mae_speed = torch.abs(speed_pred - target_speeds)
        mae_speed = (mae_speed * atom_level_mask).sum() / denom

        vector_loss = F.mse_loss(speed_vectors, target_vectors, reduction='none')
        vector_loss = vector_loss.sum(dim=-1)
        vector_loss = (vector_loss * atom_level_mask).sum() / denom

        speed_weight = getattr(training_cfg, 'speed_loss_weight', 1.0)
        vector_weight = getattr(training_cfg, 'vector_loss_weight', 1.0)
        total_loss = speed_weight * speed_loss + vector_weight * vector_loss

        metrics = {
            'speed_loss': speed_loss.detach(),
            'vector_loss': vector_loss.detach(),
            'speed_mae': mae_speed.detach(),
            'loss': total_loss,
        }

        if logits is not None:
            typeloss, perplexity = compute_CE_perplexity(
                logits, noisy_batch['aatype'], mask=loss_mask
            )
            acc = type_top1_acc(logits, noisy_batch['aatype'], node_mask=loss_mask)
            metrics.update({
                'typeloss': typeloss.detach(),
                'aa_acc': acc.detach(),
                'perplexity': perplexity,
            })
            total_loss = total_loss + typeloss * training_cfg.type_loss_weight
            metrics['loss'] = total_loss

        metrics['speed_pred_norm'] = speed_pred.detach().mean()
        metrics['target_speed_norm'] = target_speeds.detach().mean()

        return metrics



    @torch.no_grad()
    def _debug_ce_self_checks(self, pred_logits: torch.Tensor, true_labels: torch.Tensor,
                              mask_ce: torch.Tensor | None, mask_acc: torch.Tensor | None,
                              node_mask_for_logits: torch.Tensor | None = None):
        B, N, C = pred_logits.shape
        logits = pred_logits.reshape(-1, C)
        targets = true_labels.reshape(-1)
        m_ce = (mask_ce.reshape(-1).bool() if mask_ce is not None else torch.ones_like(targets, dtype=torch.bool))
        m_acc = (mask_acc.reshape(-1).bool() if mask_acc is not None else m_ce)
        # 1) dynamic C
        # 2) same mask used
        only_ce = (m_ce & (~m_acc)).sum().item()
        only_acc = (m_acc & (~m_ce)).sum().item()
        # 3) ce mask subset of node_mask (if provided)
        subset_ok = True
        if node_mask_for_logits is not None:
            nm = node_mask_for_logits.reshape(-1).bool()
            subset_ok = bool((~m_ce | nm).all())
        # 4) CE weighted mean
        ce_vec = F.cross_entropy(logits, targets, reduction='none')
        denom = m_ce.float().sum().clamp_min(1.0)
        avg_ce = (ce_vec * m_ce.float()).sum() / denom
        ppl = torch.exp(avg_ce)
        # 7) mean p(true)
        probs = logits[m_acc].softmax(-1)
        p_true = probs.gather(1, targets[m_acc, None]).mean().item() if probs.numel() > 0 else float('nan')
        acc_val = type_top1_acc(pred_logits, true_labels, node_mask=mask_acc).item()
        msg = (
            f"[CE-DIAG] B={B} N={N} C={C} | only_ce={only_ce} only_acc={only_acc} subset_ok={subset_ok} "
            f"| CE={avg_ce.item():.4f} PPL={ppl.item():.2f} ACC={acc_val:.3f} mean_p(true)={p_true:.3f}"
        )
        try:
            self._print_logger.info(msg)
        except Exception:
            print(msg)

    def make_new_protein(self, aatype, result):

        chain_index = result['chain_index'].detach().cpu().numpy()

        # change atoms mask to idea one
        ideal_atom_mask = rc.STANDARD_ATOM_MASK[aatype]
        finial_atom_mask = ideal_atom_mask  # *proteins[0]['seqmask'][:,None]

        residx_atom37_to_atom14 = make_new_atom14_resid(aatype).cpu().numpy()
        proteins = {
            'aatype': aatype.detach().cpu().numpy(),
            'atom37_atom_exists': finial_atom_mask,
            'residx_atom37_to_atom14': residx_atom37_to_atom14,
            'final_14_positions': result['final_atom_positions'],
            'final_atom_mask': finial_atom_mask,
            'residue_index': result['residue_index'],
            'domain_name': result['domain_name'],

        }

        atom37 = atom14_to_atom37(proteins['final_14_positions'].squeeze(0), proteins)

        # ADD UNK atoms
        # atom37=atom37+(1-proteins[0]['atom37_atom_exists'][:,:,None])*proteins[0]['all_atom_positions']

        feats = {
            'aatype': aatype.detach().cpu().numpy(),
            'residue_index': result['residue_index'].detach().cpu().numpy(),
        }

        results = {'final_atom_positions': atom37,
                   'final_atom_mask': proteins['final_atom_mask']}

        design_protein = from_prediction(
            features=feats,
            result=results,
            chain_index=chain_index
        )
        path = '/home/junyu/project/binder_target/gcpr/preprocessed/select/fbb/'
        unrelaxed_output_path = path + 'design_' + str(proteins['domain_name'][0][0].detach().cpu().numpy())

        design_protein_str = to_pdb(design_protein)
        with open(unrelaxed_output_path + '_.pdb', 'w') as fp:
            fp.write(design_protein_str)

    @torch.no_grad()
    def sh_roundtrip_sanity(self,
                            atom14_gt_positions, atom14_gt_exists, atom14_elem_idx,
                            Rm, t,
                            L_max=8, R_bins=64, r_max=4.8, voxel=0.25, sigma_r=0.2,
                            restype_name_to_atom14_names=None, aatype=None
                            ):
        # 1) GT→局部
        rigid = du.create_rigid(Rm, t)
        xyz_local = rigid[..., None].invert_apply(atom14_gt_positions)  # [B,N,14,3]

        # 2) 局部→SH（你自己的 sh_density_from_atom14_with_masks）
        SH, *_ = sh_density_from_atom14_with_masks(
            xyz_local, atom14_elem_idx, atom14_gt_exists,
            L_max=L_max, R_bins=R_bins, r_max=r_max, sigma_r=sigma_r
        )  # [B,N,4,L+1,2L+1,R]

        # 3) SH 解码（oracle K & oracle aatype）
        base = SHSidechainDecoder(L_max, R_bins, r_max=r_max, voxel_size=voxel,
                                  min_peak_distance=0.7, topk_per_channel=[10, 6, 6, 2])
        dyn = DynamicKSidechainDecoder(base, include_backbone=True)
        out = dyn(SH, Rm, t, aatype=aatype)  # 用真实类型最稳

        # 4) 组装 atom14（CA 用 t）
        atom14_pred, atom14_exists_pred = assemble_atom14_with_CA(
            coords_global=out['coords_global'],  # 用 rigid.apply 的
            peaks_mask=out['peaks_mask'],
            scores=out['scores'],
            aatype_probs=F.one_hot(aatype, 20).float(),
            restype_name_to_atom14_names=restype_name_to_atom14_names,
            tpos=t
        )

        # 5) 回到局部再比 RMSD
        pred_local = rigid[..., None].invert_apply(atom14_pred)

        from models.utils import report_atom14_rmsd
        ATOM14_CANONICAL = [
            'N', 'CA', 'C', 'O',  # backbone
            'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2',  # Arg/芳香/侧链通用位
            'OG', 'SG', 'OH'  # Ser/Cys/Tyr 的侧链末端
        ]
        report_atom14_rmsd(pred_local, xyz_local, atom14_gt_exists, ATOM14_CANONICAL)
        return pred_local, xyz_local

    def model_step(self, noisy_batch: Any):

        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Empty batch encountered')
        num_batch, num_res = loss_mask.shape

        # # Ground truth labels
        # gt_trans_1 = noisy_batch['trans_1']
        # gt_rotmats_1 = noisy_batch['rotmats_1']
        # rotmats_t = noisy_batch['rotmats_t']
        # gt_rot_vf = so3_utils.calc_rot_vf(
        #     rotmats_t, gt_rotmats_1.type(torch.float32))
        # if torch.any(torch.isnan(gt_rot_vf)):
        #     raise ValueError('NaN encountered in gt_rot_vf')
        #
        #
        # gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3]

        # Model output predictions.
        _, gt_rot_u_t = self.model.flow_matcher.calc_rot_vectorfield(
            noisy_batch["rotmats_1"], noisy_batch["rotmats_t"], noisy_batch["t"]
        )

        model_output = self.model(noisy_batch)
        bb_mask = noisy_batch["res_mask"]
        flow_mask = 1 - noisy_batch["fixed_mask"]
        loss_mask = bb_mask * flow_mask
        batch_size, num_res = bb_mask.shape

        gt_trans_u_t = self._flow_matcher._r.vectorfield(
            noisy_batch["rotmats_1"], noisy_batch["rotmats_t"], noisy_batch["t"]
        )
        rot_vectorfield_scaling = batch["rot_vectorfield_scaling"]
        trans_vectorfield_scaling = batch["trans_vectorfield_scaling"]
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_v_t = model_output["rot_vectorfield"] * flow_mask[..., None, None]
        pred_trans_v_t = model_output["trans_vectorfield"] * flow_mask[..., None]

        # Translation vectorfield loss
        trans_vectorfield_mse = (gt_trans_u_t - pred_trans_v_t) ** 2 * loss_mask[
            ..., None
        ]
        trans_vectorfield_loss = torch.sum(
            trans_vectorfield_mse / trans_vectorfield_scaling[:, None, None] ** 2,
            dim=(-1, -2),
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        # Translation x0 loss
        gt_trans_x0 = batch["rigids_0"][..., 4:] * self._exp_cfg.training.coordinate_scaling
        pred_trans_x0 = model_output["rigids"][..., 4:] * self._exp_cfg.training.coordinate_scaling
        trans_x0_loss = torch.sum(
            (gt_trans_x0 - pred_trans_x0) ** 2 * loss_mask[..., None], dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        trans_loss = trans_vectorfield_loss * (
                batch["t"] > self._exp_cfg.training.trans_x0_threshold
        ) + trans_x0_loss * (batch["t"] <= self._exp_cfg.training.trans_x0_threshold)
        trans_loss *= self._exp_cfg.training.trans_loss_weight
        trans_loss *= 1  # int(self._fm_conf.flow_trans)

        # Rotation loss
        # gt_rot_u_t and pred_rot_v_t are matrices convert
        t_shape = batch["rot_t"].shape[0]
        rot_t = rearrange(batch["rot_t"], "t n c d -> (t n) c d", c=3, d=3).double()
        gt_rot_u_t = rearrange(gt_rot_u_t, "t n c d -> (t n) c d", c=3, d=3)
        pred_rot_v_t = rearrange(pred_rot_v_t, "t n c d -> (t n) c d", c=3, d=3)
        try:
            rot_t = rot_t.double()
            gt_at_id = pt_to_identity(rot_t, gt_rot_u_t)
            gt_rot_u_t = hat_inv(gt_at_id)
            pred_at_id = pt_to_identity(rot_t, pred_rot_v_t)
            pred_rot_v_t = hat_inv(pred_at_id)
        except ValueError as e:
            self._log.info(
                f"Skew symmetric error gt {((gt_at_id + gt_at_id.transpose(-1, -2)) ** 2).mean()} "
                f"pred {((pred_at_id + pred_at_id.transpose(-1, -2)) ** 2).mean()} Skipping rot loss"
            )
            gt_rot_u_t = torch.zeros_like(rot_t[..., 0])
            pred_rot_v_t = torch.zeros_like(rot_t[..., 0])

        gt_rot_u_t = rearrange(gt_rot_u_t, "(t n) c -> t n c", t=t_shape, c=3)
        pred_rot_v_t = rearrange(pred_rot_v_t, "(t n) c -> t n c", t=t_shape, c=3)

        if self._exp_cfg.training.separate_rot_loss:
            gt_rot_angle = torch.norm(gt_rot_u_t, dim=-1, keepdim=True)
            gt_rot_axis = gt_rot_u_t / (gt_rot_angle + 1e-6)

            pred_rot_angle = torch.norm(pred_rot_v_t, dim=-1, keepdim=True)
            pred_rot_axis = pred_rot_v_t / (pred_rot_angle + 1e-6)

            # Separate loss on the axis
            axis_loss = (gt_rot_axis - pred_rot_axis) ** 2 * loss_mask[..., None]
            axis_loss = torch.sum(axis_loss, dim=(-1, -2)) / (
                    loss_mask.sum(dim=-1) + 1e-10
            )

            # Separate loss on the angle
            angle_loss = (gt_rot_angle - pred_rot_angle) ** 2 * loss_mask[..., None]
            angle_loss = torch.sum(
                angle_loss / rot_vectorfield_scaling[:, None, None] ** 2, dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            angle_loss *= self._exp_cfg.training.rot_loss_weight
            angle_loss *= batch["t"] > self._exp_cfg.training.rot_loss_t_threshold
            rot_loss = angle_loss + axis_loss
        else:
            rot_mse = (gt_rot_u_t - pred_rot_v_t) ** 2 * loss_mask[..., None]
            rot_loss = torch.sum(
                rot_mse / rot_vectorfield_scaling[:, None, None] ** 2,
                dim=(-1, -2),
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            rot_loss *= self._exp_cfg.training.rot_loss_weight
            rot_loss *= batch["t"] > self._exp_cfg.training.rot_loss_t_threshold
        rot_loss *= 1  # int(self._fm_conf.flow_rot)

        # Backbone atom loss
        pred_atom37 = model_output["atom37"][:, :, :5]
        gt_rigids = ru.Rigid.from_tensor_7(batch["rigids_0"].type(torch.float32))
        gt_psi = batch["torsion_angles_sin_cos"][..., 2, :]
        gt_atom37, atom37_mask, _, _ = all_atom.compute_backbone(gt_rigids, gt_psi)
        gt_atom37 = gt_atom37[:, :, :5]
        atom37_mask = atom37_mask[:, :, :5]

        gt_atom37 = gt_atom37.to(pred_atom37.device)
        atom37_mask = atom37_mask.to(pred_atom37.device)
        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
        bb_atom_loss = torch.sum(
            (pred_atom37 - gt_atom37) ** 2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3),
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
        bb_atom_loss *= self._exp_cfg.training.bb_atom_loss_weight
        bb_atom_loss *= batch["t"] < self._exp_cfg.training.bb_atom_loss_t_filter
        bb_atom_loss *= self._exp_cfg.training.aux_loss_weight

        # Pairwise distance loss
        gt_flat_atoms = gt_atom37.reshape([batch_size, num_res * 5, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
        )
        pred_flat_atoms = pred_atom37.reshape([batch_size, num_res * 5, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
        )

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 5))
        flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res * 5])
        flat_res_mask = torch.tile(bb_mask[:, :, None], (1, 1, 5))
        flat_res_mask = flat_res_mask.reshape([batch_size, num_res * 5])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        # No loss on anything >6A
        proximity_mask = gt_pair_dists < 6
        pair_dist_mask = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask, dim=(1, 2)
        )
        dist_mat_loss /= torch.sum(pair_dist_mask, dim=(1, 2)) - num_res
        dist_mat_loss *= self._exp_cfg.training.dist_mat_loss_weight
        dist_mat_loss *= batch["t"] < self._exp_cfg.training.dist_mat_loss_t_filter
        dist_mat_loss *= self._exp_cfg.training.aux_loss_weight

        final_loss = rot_loss + trans_loss + bb_atom_loss + dist_mat_loss

        def normalize_loss(x):
            return x.sum() / (batch_loss_mask.sum() + 1e-10)

        aux_data = {
            "batch_train_loss": final_loss,
            "batch_rot_loss": rot_loss,
            "batch_trans_loss": trans_loss,
            "batch_bb_atom_loss": bb_atom_loss,
            "batch_dist_mat_loss": dist_mat_loss,
            "total_loss": normalize_loss(final_loss),
            "rot_loss": normalize_loss(rot_loss),
            "trans_loss": normalize_loss(trans_loss),
            "bb_atom_loss": normalize_loss(bb_atom_loss),
            "dist_mat_loss": normalize_loss(dist_mat_loss),
            "examples_per_step": torch.tensor(batch_size),
            "res_length": torch.mean(torch.sum(bb_mask, dim=-1)),
        }

        # Maintain a history of the past N number of steps.
        # Helpful for debugging.
        self._aux_data_history.append(
            {"aux_data": aux_data, "model_out": model_output, "batch": noisy_batch}
        )

        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)
        return normalize_loss(final_loss), aux_data

    def model_step_old(self, noisy_batch: Any):

        # 定期检查并重启死代码
        if self.global_step > 99 and (self.global_step % self.model.PCT.quantizer.restart == 0):
            print(f"Step {self.global_step}: Dead codes: {len(self.model.PCT.quantizer.dead_codes)}")
            self.model.PCT.quantizer.random_restart()

        # 定期重置usage统计（可选）
        if self.global_step > 99 and self.global_step % (self.model.PCT.quantizer.restart * 10) == 0:
            self.model.PCT.quantizer.reset_usage()

        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Empty batch encountered')
        num_batch, num_res = loss_mask.shape

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        rotmats_t = noisy_batch['rotmats_0']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        if torch.any(torch.isnan(gt_rot_vf)):
            raise ValueError('NaN encountered in gt_rot_vf')

        gt_bb_atoms = self.frames(gt_rotmats_1, gt_trans_1, noisy_batch['chain_idx'])
        # gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3]

        # Model output predictions.
        model_output, vq_loss, log_dict, z_q, downsampled_mask = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']

        # if torch.any(torch.isnan(rotmats_t)):
        #     raise ValueError('NaN encountered in rotmats_t')
        #
        # if torch.any(torch.isnan(pred_rotmats_1)):
        #     raise ValueError('NaN encountered in pred_rotmats_1')
        #
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        # if torch.any(torch.isnan(pred_rots_vf)):
        #     raise ValueError('NaN encountered in pred_rots_vf')

        # Backbone atom loss
        # pred_all_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)

        pred_all_atoms = self.frames(pred_rotmats_1, pred_trans_1, noisy_batch['chain_idx'])
        pred_bb_atoms = pred_all_atoms.clone()
        gt_bb_atoms *= training_cfg.bb_atom_scale
        pred_bb_atoms *= training_cfg.bb_atom_scale
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) * training_cfg.trans_scale
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        trans_loss = torch.clamp(trans_loss, max=5)

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf)
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # VQ-VAE应该用直接比较
        # rotation_vf_loss = so3_utils.calc_rot_vf(pred_rotmats_1, gt_rotmats_1)
        # rotation_vf_loss = training_cfg.rotation_loss_weights *torch.sum(rotation_vf_loss ** 2 * loss_mask[..., None], dim=(-1, -2))/ loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res * 4, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res * 4, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 4))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res * 4])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 4))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res * 4])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) + 1)

        se3_vf_loss = trans_loss
        auxiliary_loss = (
                bb_atom_loss * training_cfg.aux_loss_use_bb_loss
                + dist_mat_loss * training_cfg.aux_loss_use_pair_loss
        )

        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        auxiliary_loss = torch.clamp(auxiliary_loss, max=5)

        # fape
        fapeloss = mu.fbb_backbone_loss(pred_trans_1, pred_rotmats_1, gt_trans_1, gt_rotmats_1, loss_mask)

        se3_vf_loss += auxiliary_loss
        se3_vf_loss = se3_vf_loss + self._exp_cfg.training.vq_loss_weight * vq_loss + fapeloss
        if torch.any(torch.isnan(se3_vf_loss)):
            raise ValueError('NaN loss encountered')
        return {
            "trans_loss": trans_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss,
            "vq_loss": vq_loss,
            "fapeloss": fapeloss,
            "perplexity": log_dict['vq_perplexity'],
        }, pred_all_atoms.detach()

        def normalize_loss(x):
            return x.sum() / (batch_loss_mask.sum() + 1e-10)

        aux_data = {
            "batch_train_loss": final_loss,
            "batch_rot_loss": rot_loss,
            "batch_trans_loss": trans_loss,
            "batch_bb_atom_loss": bb_atom_loss,
            "batch_dist_mat_loss": dist_mat_loss,
            "total_loss": normalize_loss(final_loss),
            "rot_loss": normalize_loss(rot_loss),
            "trans_loss": normalize_loss(trans_loss),
            "bb_atom_loss": normalize_loss(bb_atom_loss),
            "dist_mat_loss": normalize_loss(dist_mat_loss),
            "examples_per_step": torch.tensor(batch_size),
            "res_length": torch.mean(torch.sum(bb_mask, dim=-1)),
        }

        # Maintain a history of the past N number of steps.
        # Helpful for debugging.
        self._aux_data_history.append(
            {"aux_data": aux_data, "model_out": model_output, "batch": noisy_batch}
        )

        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)
        return normalize_loss(final_loss), aux_data

    def model_step_ssq(self, noisy_batch: Any):
        """使用线性桥的训练函数，对齐 ml-simplefold 的训练目标"""
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Empty batch encountered')
        num_batch, num_res = loss_mask.shape

        # 从线性桥获取的目标向量场
        gt_trans_v = noisy_batch['trans_v']  # 平移向量场
        gt_rot_v = noisy_batch['rot_v']  # 旋转向量场

        # 当前噪声状态
        trans_t = noisy_batch['trans_t']
        rotmats_t = noisy_batch['rotmats_t']
        t = noisy_batch['t']

        # 真实状态
        trans_1 = noisy_batch['trans_1']
        rotmats_1 = noisy_batch['rotmats_1']

        # 模型预测
        model_output = self.model(noisy_batch)

        # 假设模型输出包含向量场预测，如果没有则需要添加
        if 'trans_vectorfield' in model_output:
            pred_trans_v = model_output['trans_vectorfield']
        else:
            # 如果没有，从预测的干净坐标计算向量场
            pred_trans_1 = model_output.get('pred_trans', trans_1)
            pred_trans_v = (pred_trans_1 - trans_t) / (1 - t[:, None, None] + 1e-8)

        if 'rot_vectorfield' in model_output:
            pred_rot_v = model_output['rot_vectorfield']
        else:
            # 如果没有，从预测的干净旋转计算向量场
            pred_rotmats_1 = model_output.get('pred_rotmats', rotmats_1)
            pred_rot_v = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)

        # 平移向量场损失
        trans_vectorfield_mse = (gt_trans_v - pred_trans_v) ** 2 * loss_mask[..., None]
        trans_vectorfield_loss = torch.sum(
            trans_vectorfield_mse,
            dim=(-1, -2),
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        # 旋转向量场损失
        rot_vectorfield_mse = (gt_rot_v - pred_rot_v) ** 2 * loss_mask[..., None]
        rot_vectorfield_loss = torch.sum(
            rot_vectorfield_mse,
            dim=(-1, -2),
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        # 总损失
        trans_loss = trans_vectorfield_loss * training_cfg.get('trans_loss_weight', 1.0)
        rot_loss = rot_vectorfield_loss * training_cfg.get('rot_loss_weight', 1.0)

        final_loss = trans_loss + rot_loss

        # 从预测的向量场重建干净坐标（用于验证）
        pred_trans_clean = trans_t + (1 - t[:, None, None]) * pred_trans_v

        # 辅助损失（可选）
        auxiliary_loss = torch.tensor(0.0, device=final_loss.device)

        # 如果有 backbone atom loss 的需求
        if hasattr(self, 'frames') and 'chain_idx' in noisy_batch:
            try:
                gt_backbone = self.frames(rotmats_1, trans_1, noisy_batch['chain_idx'])
                pred_backbone = self.frames(rotmats_t, pred_trans_clean, noisy_batch['chain_idx'])

                bb_atom_loss = torch.sum(
                    (gt_backbone - pred_backbone) ** 2 * loss_mask[..., None, None],
                    dim=(-1, -2, -3)
                ) / (loss_mask.sum(dim=-1) + 1e-10)

                auxiliary_loss += bb_atom_loss * training_cfg.get('bb_atom_loss_weight', 0.1)
            except Exception as e:
                print(f"Backbone loss calculation failed: {e}")

        final_loss += auxiliary_loss

        # 归一化损失
        def normalize_loss(x):
            return x.mean()

        aux_data = {
            "trans_loss": normalize_loss(trans_loss),
            "rot_loss": normalize_loss(rot_loss),
            "auxiliary_loss": normalize_loss(auxiliary_loss),
            "total_loss": normalize_loss(final_loss),
            "examples_per_step": torch.tensor(num_batch),
            "res_length": torch.mean(torch.sum(loss_mask, dim=-1)),
        }

        return normalize_loss(final_loss), aux_data

    def training_step_ssq(self, batch: Any, batch_idx: int):
        """使用线性桥的训练步骤"""
        self.interpolant.set_device(batch['res_mask'].device)

        # 使用新的线性桥加噪函数
        noisy_batch = self.interpolant.corrupt_batch_ssq(batch)

        # 使用新的训练函数
        loss, aux_data = self.model_step_ssq(noisy_batch)

        # 记录训练指标
        for key, value in aux_data.items():
            if isinstance(value, torch.Tensor):
                self.log(f"train/{key}", value, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        res_mask = batch['res_mask']
        csv_idx = batch['csv_idx']
        step_start_time = time.time()

        self.interpolant.set_device(batch['res_mask'].device)
        atom_mse_per_example = None

        # ====================================================================
        # 计算 loss（根据 task 类型）
        # ====================================================================
        save_samples = getattr(self._exp_cfg, 'save_val_samples', False)
        outs = None
        noisy_batch = None

        if self._exp_cfg.task == 'diffusion' or self._exp_cfg.task == 'shdiffusion':
            batch_losses = self.model_step_shdiffusion(batch)
            if self._exp_cfg.task == 'shdiffusion':
                atom_mse_per_example = batch_losses.pop('atom_mse_per_example', None)
        else:
            if self._exp_cfg.task == 'allatoms':
                batch_losses = self.model_step_shdiffusion(batch)
                if self._exp_cfg.task == 'shdiffusion':
                    atom_mse_per_example = batch_losses.pop('atom_mse_per_example', None)

            elif self._exp_cfg.task in ('fbb',):
                # 如果需要保存样本，则获取模型输出
                if save_samples:
                    batch_losses, outs, noisy_batch = self.model_step_fbb(batch, prob=0, return_outputs=True)
                else:
                    batch_losses,_ = self.model_step_fbb(batch, prob=0, return_outputs=False)

            elif self._exp_cfg.task in ('shfbb',):
                batch_losses = self.model_step_shfbb(batch, prob=1)

            elif self._exp_cfg.task in ('sh_to_atoms',):
                batch_losses = self.model_step_decoder(batch)

                # 计算per-atom RMSD统计
                with torch.no_grad():
                    # 使用与训练相同的tau配置
                    tau = self._exp_cfg.training.get('sh_tau_threshold', 0.0)
                    # if tau > 0:
                    #     tau_mask = batch['density_mask'] * (torch.abs(batch['normalize_density']) > tau)
                    #     sh = batch['normalize_density'] * tau_mask
                    # else:
                    #     sh = batch['normalize_density'] * batch['density_mask']
                    batch['r3_t'] = torch.ones_like(batch['res_mask'])
                    logits, atoms14_pred = self.model(batch)  #, node_mask=batch['res_mask']

                    atom14_gt = batch['atoms14_local']
                    atom14_mask = batch['atom14_gt_exists']

                    # 整体RMSD
                    diff = (atoms14_pred - atom14_gt) ** 2 * atom14_mask.unsqueeze(-1)
                    rmsd = torch.sqrt(diff.sum() / (atom14_mask.sum() + 1e-6))
                    batch_losses['atom14_rmsd'] = rmsd

                    # 按原子位置统计 (0-13: N, CA, C, O, CB, ...)
                    for atom_idx in range(14):
                        mask_i = atom14_mask[:, :, atom_idx]  # [B, N]
                        if mask_i.sum() > 0:
                            diff_i = (atoms14_pred[:, :, atom_idx] - atom14_gt[:, :, atom_idx]) ** 2  # [B, N, 3]
                            diff_i = diff_i * mask_i.unsqueeze(-1)  # [B, N, 3]
                            rmsd_i = torch.sqrt(diff_i.sum() / (mask_i.sum() + 1e-6))
                            batch_losses[f'atom{atom_idx}_rmsd'] = rmsd_i

                    # 侧链RMSD (atom 4-13)
                    sidechain_mask = atom14_mask.clone()
                    sidechain_mask[:, :, :4] = 0
                    if sidechain_mask.sum() > 0:
                        sc_diff = (atoms14_pred - atom14_gt) ** 2 * sidechain_mask.unsqueeze(-1)
                        sc_rmsd = torch.sqrt(sc_diff.sum() / (sidechain_mask.sum() + 1e-6))
                        batch_losses['sidechain_rmsd'] = sc_rmsd

            else:
                batch_losses = self.model_step_fbb_backup(batch, prob=1)
        if self._exp_cfg.task != 'shfbb_infer':

            num_batch = res_mask.shape[0]
            total_losses = {
                k: torch.mean(v) for k, v in batch_losses.items()
            }
            if self._exp_cfg.task == 'shdiffusion' and atom_mse_per_example is not None and 't' in batch:
                stratified_losses = mu.t_stratified_mean_loss(
                    batch['t'].detach().cpu(),
                    atom_mse_per_example.detach().cpu(),
                    loss_name='atom_mse'
                )
                for k, v in stratified_losses.items():
                    self._log_scalar(
                        f"valid/{k}", v, on_step=False, on_epoch=True, prog_bar=False, batch_size=num_batch
                    )
            for k, v in total_losses.items():
                self._log_scalar(
                    f"valid/{k}", v.detach().item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=num_batch)

            # Losses to track. Stratified across t.
            if self._exp_cfg.task == 'diffusion':
                for loss_name, loss_dict in batch_losses.items():

                    stratified_losses = mu.t_stratified_mean_loss(
                        batch['t'], loss_dict, loss_name=loss_name)
                    for k, v in stratified_losses.items():
                        self._log_scalar(
                            f"valid/{k}", v, on_step=False, on_epoch=True, prog_bar=False, batch_size=num_batch)
                    # Training throughput
                    scaffold_percent = torch.mean(batch['diffuse_mask'].float()).item()
                    self._log_scalar(
                        "valid/scaffolding_percent",
                        scaffold_percent, on_step=False, on_epoch=True, prog_bar=False, batch_size=num_batch)
                    motif_mask = 1 - batch['diffuse_mask'].float()
                    num_motif_res = torch.sum(motif_mask, dim=-1)
                    self._log_scalar(
                        "valid/motif_size",
                        torch.mean(num_motif_res).item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=num_batch)
                    self._log_scalar(
                        "valid/length", batch['res_mask'].shape[1], on_step=False, on_epoch=True, prog_bar=False, batch_size=num_batch)
                    self._log_scalar(
                        "valid/batch_size", num_batch, on_step=False, on_epoch=True, prog_bar=False)
                    step_time = time.time() - step_start_time
                    self._log_scalar(
                        "valid/examples_per_second", num_batch / step_time, on_step=False, on_epoch=True, prog_bar=False)
                    val_loss = total_losses['se3_vf_loss']
                    self._log_scalar(
                        "valid/loss", val_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=num_batch)
                num_batch, num_res = res_mask.shape

                gt_all_atoms = self.frames(batch['rotmats_1'], batch['trans_1'],
                                           batch['chain_idx']).detach().cpu().numpy()
                # gt__atoms = all_atom.transrot_to_atom37( list(zip(batch['trans_1'], batch['rotmats_1'])),batch['res_mask']).detach().cpu().numpy()

                samples = pred_bb_atoms.detach().cpu().numpy()
                batch_metrics = []
                for i in range(num_batch):
                    sample_dir = os.path.join(
                        self.checkpoint_dir,
                        f'sample_{csv_idx[i].item()}_idx_{batch_idx}_len_{num_res}'
                    )
                    os.makedirs(sample_dir, exist_ok=True)

                    # Write out sample to PDB file
                    final_pos = samples[i]
                    # saved_path = au.write_prot_to_pdb(
                    #     final_pos,
                    #     os.path.join(sample_dir, 'sample.pdb'),
                    #     no_indexing=True
                    # )

                    saved_path = os.path.join(sample_dir, 'sample.pdb')

                    save_4pdb(final_pos.reshape(-1, 3), os.path.join(sample_dir, 'sample.pdb'))
                    # save_4pdb(final_pos[..., [0,1,2,4], :].reshape(-1, 3), os.path.join(sample_dir, 'final_pos.pdb'))
                    save_4pdb(gt_all_atoms.reshape(-1, 3), os.path.join(sample_dir, 'backbone.pdb'))
                    save_4pdb(batch['backbone'].reshape(-1, 3).detach().cpu().numpy(),
                              os.path.join(sample_dir, 'backbone_4ATOMS.pdb'))

                    # _ = au.write_prot_to_pdb(
                    #     gt_all_atoms[i],
                    #     os.path.join(sample_dir, 'gt.pdb'),
                    #     no_indexing=True
                    # )

                    # _ = au.write_prot_to_pdb(
                    #     gt__atoms[i],
                    #     os.path.join(sample_dir, 'gt_transrot.pdb'),
                    #     no_indexing=True
                    # )

                    if isinstance(self.logger, WandbLogger):
                        self.validation_epoch_samples.append(
                            [saved_path, self.global_step, wandb.Molecule(saved_path)]
                        )

                    mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
                    ca_idx = residue_constants.atom_order['CA']
                    ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
                    batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

                batch_metrics = pd.DataFrame(batch_metrics)

                self.validation_epoch_metrics.append(batch_metrics)

            elif self._exp_cfg.task in ('aatype', 'SHdecode', 'shfbb', 'fbb', 'sh_to_atoms'):
                batch_losses_fixed = {}
                for k, v in batch_losses.items():
                    if hasattr(v, 'item'):  # 如果是tensor
                        batch_losses_fixed[k] = [v.item()]
                    else:  # 如果已经是标量
                        batch_losses_fixed[k] = [v]

                if 'speed_loss' in total_losses:
                    self._log_scalar(
                        'valid/speed_loss',
                        total_losses['speed_loss'],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        batch_size=num_batch,
                    )
                if 'speed_mae' in total_losses:
                    self._log_scalar(
                        'valid/speed_mae',
                        total_losses['speed_mae'],
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        batch_size=num_batch,
                    )
                self.validation_epoch_metrics.append(pd.DataFrame(batch_losses_fixed))

                # ====================================================================
                # 【新增】保存验证样本（PDB + FASTA）
                # ====================================================================
                if save_samples and outs is not None and noisy_batch is not None:
                    self._save_validation_samples(batch, outs, noisy_batch)

        if self._exp_cfg.task == 'shfbb':
            try:
                if self._val_ref_batch is not None:
                    self._export_val_sample(self._val_ref_batch)
            except Exception as exc:
                self._print_logger.warning(f"Failed to export validation sample: {exc}")
            finally:
                self._val_ref_batch = None
            self.validation_epoch_metrics.clear()

    def _save_validation_samples(self, batch, outs, noisy_batch):
        """保存验证样本的 PDB 和 FASTA 文件"""
        try:
            # 创建验证输出目录（每个 epoch 一个）
            val_output_dir = os.path.join(self.checkpoint_dir, f'val_samples_epoch{self.current_epoch}')
            os.makedirs(val_output_dir, exist_ok=True)

            # 从 outs 获取预测结果
            pred_atoms_local = outs['pred_atoms']  # [B, N, 10, 3] - IGA 输出局部坐标
            logits = outs['logits']  # [B, N, 21]

            # 组装完整的 14 原子
            backbone_local = batch['atoms14_local'][..., :4, :]  # [B, N, 4, 3]
            atoms14_local = torch.cat([backbone_local, pred_atoms_local], dim=-2)  # [B, N, 14, 3]

            # 转到全局坐标
            rigid = du.create_rigid(batch['rotmats_1'], batch['trans_1'])
            atoms14_global = rigid[..., None].apply(atoms14_local)

            # 计算序列指标
            aa_pred = logits.argmax(dim=-1)
            aa_true = batch['aatype']
            res_mask = batch['res_mask']
            recovery = type_top1_acc(logits, aa_true, node_mask=res_mask)
            _, perplexity = compute_CE_perplexity(logits, aa_true, mask=res_mask)

            # 获取样本 ID 和名称
            sample_ids = batch.get('csv_idx', torch.arange(res_mask.shape[0]))
            sample_ids = sample_ids.squeeze().tolist()
            sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids

            source_names = batch.get('source_name', None)
            idx_to_aa = {residue_constants.restype_order[aa]: aa for aa in residue_constants.restypes}

            # 保存每个样本
            for b, sample_id in enumerate(sample_ids):
                try:
                    sid = int(sample_id)
                except Exception:
                    sid = sample_id

                # 构建样本名称
                base_name = None
                if isinstance(source_names, list) and b < len(source_names) and source_names[b] is not None:
                    try:
                        base_name = str(source_names[b])
                    except Exception:
                        pass

                if base_name is not None:
                    base_name = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in base_name)

                tag = f"{sid:06d}" if isinstance(sid, int) else str(sid)
                if base_name:
                    tag = f"{base_name}_{tag}"

                # 创建样本目录
                sample_dir = os.path.join(val_output_dir, f'sample_{tag}')
                os.makedirs(sample_dir, exist_ok=True)

                # 保存 FASTA
                mask = res_mask[b].bool()
                aa_true_seq = ''.join([idx_to_aa.get(int(x), 'X') for x in aa_true[b][mask].tolist()])
                aa_pred_seq = ''.join([idx_to_aa.get(int(x), 'X') for x in aa_pred[b][mask].tolist()])

                fasta_path = os.path.join(sample_dir, 'sequence.fasta')
                with open(fasta_path, 'w') as f:
                    f.write(f'>original_{sample_id}\n{aa_true_seq}\n')
                    f.write(f'>predicted_{sample_id} recovery={float(recovery):.3f} perplexity={float(perplexity):.3f}\n{aa_pred_seq}\n')

                # 保存 PDB
                atom14_pred_np = atoms14_global[b].detach().cpu().numpy()
                pred_aatype = aa_pred[b].detach().long()
                residx_atom37_to_atom14 = make_new_atom14_resid(pred_aatype.cpu()).cpu()
                atom37_atom_exists = residue_constants.STANDARD_ATOM_MASK[pred_aatype.cpu().numpy()]

                protein_batch = {
                    'residx_atom37_to_atom14': residx_atom37_to_atom14,
                    'atom37_atom_exists': atom37_atom_exists,
                }

                atom37 = all_atom.atom14_to_atom37(torch.tensor(atom14_pred_np), protein_batch)
                aa_pred_np = aa_pred[b].detach().cpu().numpy()
                pdb_path = os.path.join(sample_dir, 'predicted.pdb')
                au.write_prot_to_pdb(atom37, pdb_path, aatype=aa_pred_np, no_indexing=True, overwrite=True)

                # 保存 diagnostics
                diag_path = os.path.join(sample_dir, 'diagnostics.txt')
                with open(diag_path, 'w', encoding='utf-8') as f:
                    f.write("=== Validation Sample ===\n")
                    f.write(f"Recovery: {float(recovery):.3f}\n")
                    f.write(f"Perplexity: {float(perplexity):.3f}\n")

            self._print_logger.info(f"Saved {len(sample_ids)} validation samples to {val_output_dir}")

        except Exception as exc:
            self._print_logger.warning(f"Failed to save validation samples: {exc}")
            import traceback
            traceback.print_exc()

    def on_validation_epoch_end(self):
        if self._exp_cfg.task != 'shfbb':
            return

    def _store_val_reference_batch(self, batch: dict[str, torch.Tensor]):
        ref_batch: dict[str, torch.Tensor] = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                ref_batch[key] = val.detach().cpu()
            else:
                ref_batch[key] = val
        self._val_ref_batch = ref_batch

    @torch.no_grad()
    def _export_val_sample(self, batch_cpu: dict[str, torch.Tensor]):
        device = next(self.model.parameters()).device
        batch = {}
        for key, val in batch_cpu.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(device)
            else:
                batch[key] = val

        self.interpolant.set_device(device)
        prepared = self.interpolant.fbb_prepare_batch(batch)
        sample_out = self.interpolant.fbb_sample_iterative(prepared, self.model)

        atoms14_global = sample_out['atoms14_global_final']
        logits = sample_out['logits_final']

        aa_source = batch.get('aatype')
        if logits is not None:
            aa_pred = logits.argmax(dim=-1)
        elif aa_source is not None:
            aa_pred = aa_source
        else:
            self._print_logger.warning('Skipping validation sample export: no logits or aatype found.')
            return

        sample_dir = os.path.join(self.checkpoint_dir, 'val_samples', f'epoch_{self.current_epoch:04d}')
        os.makedirs(sample_dir, exist_ok=True)

        num_samples = min(getattr(self._exp_cfg.training, 'val_sample_count', 1), atoms14_global.shape[0])
        for idx in range(num_samples):
            atom14 = atoms14_global[idx].detach().cpu().float()
            aatype = aa_pred[idx].detach().cpu()

            residx_atom37_to_atom14 = make_new_atom14_resid(aatype).cpu()
            atom37_atom_exists = residue_constants.STANDARD_ATOM_MASK[aatype.numpy()]

            protein_batch = {
                'residx_atom37_to_atom14': residx_atom37_to_atom14,
                'atom37_atom_exists': atom37_atom_exists,
            }

            atom37 = all_atom.atom14_to_atom37(atom14, protein_batch)

            tag = f'epoch{self.current_epoch:04d}_sample{idx:02d}'
            pdb_path = os.path.join(sample_dir, f'{tag}.pdb')
            au.write_prot_to_pdb(atom37, pdb_path, aatype=aatype.numpy(), no_indexing=True, overwrite=True)

            self._print_logger.info(f'Validation sample written to {pdb_path}')

    def _log_scalar(
            self,
            key,
            value,
            on_step=None,  # 从True改成None，让Lightning自动遵循log_every_n_steps
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
    ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()

        self.interpolant.set_device(batch['res_mask'].device)
        atom_mse_per_example = None
        if self._exp_cfg.task in ('diffusion', 'shdiffusion'):

            batch_losses = self.model_step_shdiffusion(batch)
            if self._exp_cfg.task == 'shdiffusion':
                atom_mse_per_example = batch_losses.pop('atom_mse_per_example', None)

        elif self._exp_cfg.task in ('allatoms'):

            batch_losses = self.model_step_allatoms(batch)

            atom_mse_per_example = batch_losses.pop('atom_mse_per_example', None)

        elif self._exp_cfg.task in ('fbb',):
            batch_losses = self.model_step_fbb(batch,prob=0)
        elif self._exp_cfg.task in ('shfbb',):
            batch_losses = self.model_step_shfbb(batch,prob=1)

        elif self._exp_cfg.task in ('sh_to_atoms',):
            batch_losses = self.model_step_decoder(batch)

        else:
            batch_losses = self.model_step_fbb_backup(batch)
        num_batch = batch['res_mask'].shape[0]
        if atom_mse_per_example is not None and 't' in batch:
            stratified_losses = mu.t_stratified_mean_loss(
                batch['t'].detach().cpu(),
                atom_mse_per_example.detach().cpu(),
                loss_name='atom_mse'
            )
            for k, v in stratified_losses.items():
                self._log_scalar(
                    f"train/{k}", v, prog_bar=False, batch_size=num_batch
                )

        total_losses = {
            k: torch.mean(v) for k, v in batch_losses.items()
        }

        # Losses to track. Stratified across t.
        #
        # for loss_name, loss_dict in batch_losses.items():
        #
        #     stratified_losses = mu.t_stratified_mean_loss(
        #         torch.tensor(0.001), loss_dict, loss_name=loss_name)
        #     for k,v in stratified_losses.items():
        #         self._log_scalar(
        #             f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        if self._exp_cfg.task == 'vae':
            for k, v in total_losses.items():
                self._log_scalar(
                    f"train/{k}", v, prog_bar=False, batch_size=num_batch)
            # Training throughput
            scaffold_percent = torch.mean(batch['diffuse_mask'].float()).item()
            self._log_scalar(
                "train/scaffolding_percent",
                scaffold_percent, prog_bar=False, batch_size=num_batch)
            motif_mask = 1 - batch['diffuse_mask'].float()
            num_motif_res = torch.sum(motif_mask, dim=-1)
            self._log_scalar(
                "train/motif_size",
                torch.mean(num_motif_res).item(), prog_bar=False, batch_size=num_batch)
            self._log_scalar(
                "train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
            self._log_scalar(
                "train/batch_size", num_batch, prog_bar=False)
            step_time = time.time() - step_start_time
            self._log_scalar(
                "train/examples_per_second", num_batch / step_time)
            train_loss = total_losses['se3_vf_loss']
            self._log_scalar(
                "train/loss", train_loss, batch_size=num_batch)
        elif self._exp_cfg.task == 'aatype':
            train_loss = total_losses['aaloss']
            self._log_scalar(
                "train/aaloss", train_loss, batch_size=num_batch)
            self._log_scalar(
                "train/aa_acc", total_losses['aa_acc'], batch_size=num_batch)
        elif self._exp_cfg.task in ('SHdecode', 'shfbb', 'fbb', 'sh_to_atoms', 'shdiffusion'):
            train_loss = total_losses['loss']

            # 移除手动频率控制，让Lightning根据log_every_n_steps自动控制
            if 'speed_loss' in total_losses:
                self._log_scalar(
                    "train/speed_loss",
                    total_losses['speed_loss'],
                    prog_bar=True,
                    batch_size=num_batch,
                )
            for k, v in total_losses.items():
                if k == 'speed_loss':
                    continue
                self._log_scalar(
                    f"train/{k}", float(v.detach().cpu().item()), prog_bar=True, batch_size=num_batch)

        return train_loss
    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )
    # def configure_optimizers(self):
    #     trainable = [p for p in self.parameters() if p.requires_grad]
    #
    #     optimizer=torch.optim.AdamW(
    #         params=trainable,
    #         **self._exp_cfg.optimizer
    #     )
    #
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer,
    #         mode='min',
    #         factor=0.7,  # 每次降为70% (更平缓)
    #         patience=3,  # 3次验证 = 6个epoch
    #         min_lr=1e-6,
    #         verbose=True,
    #         threshold=5e-5,  # 更小的阈值，更容易触发降lr
    #         cooldown=1  # 降lr后等2个epoch
    #     )
    #
    #     return {
    #         'optimizer': optimizer,
    #         'lr_scheduler': {
    #             'scheduler': scheduler,
    #             'monitor': 'valid/loss',  # 或你监控的其他指标
    #             'interval': 'epoch',
    #             'frequency': 1,
    #         }
    #     }

    def fbb_sample(
            self,
            batch: dict[str, torch.Tensor],
            diffusion_prob: float | None = 1,
            out_dir: str | None = None,
            use_sde: bool = False,
            sde_tau: float = 0.3,
            sde_w_cutoff: float = 0.99,
    ) -> dict[str, torch.Tensor]:
        """Run masked sidechain reconstruction for FBB inference.

        Args:
            batch: input features (same schema as training).
            diffusion_prob: optional mask prob override when calling interpolant.
            out_dir: optional directory to dump outputs (per batch element).
            use_sde: if True, use SimpleFold-style SDE sampling with noise; if False, use deterministic ODE
            sde_tau: temperature parameter for SDE sampling (default 0.3)
            sde_w_cutoff: diffusion coefficient cutoff for SDE sampling (default 0.99)

        Returns:
            Dictionary containing reconstructed side chains and related tensors.
        """

        device = batch['res_mask'].device
        self.interpolant.set_device(device)

        with torch.no_grad():
            prepared_batch = self.interpolant.fbb_corrupt_batch(batch,prob=1)
            # Choose sampling method
            if use_sde:
                # SimpleFold-style SDE sampling with noise
                sample_out = self.interpolant.fbb_sample_iterative_sde(
                    prepared_batch,
                    self.model,
                    tau=sde_tau,
                    w_cutoff=sde_w_cutoff,
                )
            else:
                # Deterministic ODE sampling (default)
                sample_out = self.interpolant.fbb_sample_iterative(
                    prepared_batch,
                    self.model,
                )
        # Calculate loss using IGA Loss
        metrics = self.iga_loss_fn(sample_out['out'],prepared_batch)
        logits = sample_out['logits_final']
        atoms14_local = sample_out['atoms14_local_final']

        # 用GT backbone替换predicted backbone（确保几何一致性）
        # if 'atoms14_local' in batch:
        #     atoms14_local = atoms14_local.clone()
        #     atoms14_local[..., :3, :] = batch['atoms14_local'][..., :3, :]
        #     sample_out['atoms14_local_final'] = atoms14_local
        #
        #     # ❗ 关键修复：重新计算global坐标（用修正后的local坐标）
        #     import openfold.utils.rigid_utils as ru
        #     rigid = ru.Rigid.from_tensor_7(batch['rigids_1'])
        #     atoms14_global = rigid[..., None].apply(atoms14_local)
        #     sample_out['atoms14_global_final'] = atoms14_global
        # else:
        #     atoms14_global = sample_out['atoms14_global_final']
        atoms14_global = sample_out['atoms14_global_final']

        diagnostics = sample_out.get('diagnostics', {})

        return {
            'logits': logits,
            'update_mask': prepared_batch['update_mask'],
            'atom14_local': atoms14_local,
            'atom14_global': atoms14_global,
            'diagnostics': diagnostics,
        }

    def predict_step(self, batch, batch_idx):
        if self._exp_cfg.task == 'shdiffusion':
            return self.predict_step_shdiffusion(batch, batch_idx)
        # del batch_idx  # lightning signature
        if self._exp_cfg.task not in ('shfbb','fbb', 'shfbb_infer'):
            raise RuntimeError('predict_step is only implemented for FBB tasks')

        sample_root = self._get_inference_run_dir()
        os.makedirs(sample_root, exist_ok=True)

        cfg_path = os.path.join(sample_root, 'config.yaml')
        if not os.path.exists(cfg_path):
            try:
                OmegaConf.save(self._full_cfg, cfg_path)
            except Exception as exc:
                self._print_logger.warning(f"Failed to dump config: {exc}")

        sample_ids = batch.get('csv_idx')
        if sample_ids is None:
            sample_ids = torch.arange(batch['res_mask'].shape[0])
        sample_ids = sample_ids.squeeze().tolist()
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids

        # Optional name mapping from CSV
        name_map = None
        try:
            import pandas as pd
            csv_path = getattr(self._data_cfg, 'csv_path', None)
            if csv_path and os.path.exists(csv_path):
                df_names = pd.read_csv(csv_path)
                col = 'pdb_name' if 'pdb_name' in df_names.columns else ('name' if 'name' in df_names.columns else None)
                if col is not None:
                    name_map = df_names[col].astype(str).tolist()
        except Exception:
            name_map = None

        # Prefer dataset-provided source_name if present (list[str])
        source_names = batch.get('source_name', None)

        result = self.fbb_sample(batch,use_sde=self._interpolant_cfg.sampling.do_sde)

        logits = result['logits']
        atom14_global = result['atom14_global']
        atom14_local = result['atom14_local']
        update_mask = result['update_mask']
        diagnostics = result.get('diagnostics', {})

        print(logits[0,0])
        aa_pred = logits.argmax(dim=-1)  # [B,N]
        aa_true = batch['aatype'] if 'aatype' in batch else aa_pred
        rec_mask = batch.get('res_mask', torch.ones_like(aa_pred))
        recovery = type_top1_acc(logits, aa_true, node_mask=rec_mask)
        _, perplexity = compute_CE_perplexity(logits, aa_true, mask=rec_mask)

        # 输出诊断信息到日志
        if diagnostics:
            diag_str = f"[DIAGNOSTICS] "
            if 'sidechain_rmsd' in diagnostics:
                diag_str += f"Sidechain_RMSD={diagnostics['sidechain_rmsd']:.4f}A  "
            if 'perplexity_with_pred_coords' in diagnostics:
                diag_str += f"PPL(pred_coords)={diagnostics['perplexity_with_pred_coords']:.3f}  "
            if 'perplexity_with_gt_coords' in diagnostics:
                diag_str += f"PPL(GT_coords)={diagnostics['perplexity_with_gt_coords']:.3f}  "
            if 'recovery_with_pred_coords' in diagnostics:
                diag_str += f"Recovery(pred_coords)={diagnostics['recovery_with_pred_coords']:.3f}  "
            if 'recovery_with_gt_coords' in diagnostics:
                diag_str += f"Recovery(GT_coords)={diagnostics['recovery_with_gt_coords']:.3f}"
            self._print_logger.info(diag_str)

        atom14_pred = atom14_global
        atom14_exists = batch['atom14_gt_exists'] if 'atom14_gt_exists' in batch else torch.ones_like(
            atom14_pred[..., 0])

        res_mask = batch['res_mask']

        idx_to_aa = {residue_constants.restype_order[aa]: aa for aa in residue_constants.restypes}

        predictions = []
        for b, sample_id in enumerate(sample_ids):
            try:
                sid = int(sample_id)
            except Exception:
                sid = sample_id
            base_name = None
            if isinstance(source_names, list) and b < len(source_names) and source_names[b] is not None:
                try:
                    base_name = str(source_names[b])
                except Exception:
                    base_name = None
            if base_name is None and name_map is not None and isinstance(sid, int) and 0 <= sid < len(name_map):
                base_name = name_map[sid]
            if base_name is not None:
                base_name = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in base_name)
            tag = f"{sid:06d}" if isinstance(sid, int) else str(sid)
            if base_name:
                tag = f"{base_name}_{tag}"
            mask = res_mask[b].bool()
            aa_true_seq = ''.join([idx_to_aa.get(int(x), 'X') for x in aa_true[b][mask].tolist()])
            aa_pred_seq = ''.join([idx_to_aa.get(int(x), 'X') for x in aa_pred[b][mask].tolist()])

            sample_dir = os.path.join(sample_root, f'sample_{tag}')
            os.makedirs(sample_dir, exist_ok=True)

            fasta_path = os.path.join(sample_dir, 'sequence.fasta')
            with open(fasta_path, 'w') as f:
                f.write(f'>original_{sample_id}\n')
                f.write(f'{aa_true_seq}\n')
                f.write(f'>predicted_{sample_id} recovery={float(recovery):.3f} perplexity={float(perplexity):.3f}\n')
                f.write(f'{aa_pred_seq}\n')

            # 保存诊断信息
            if diagnostics:
                diag_path = os.path.join(sample_dir, 'diagnostics.txt')
            with open(diag_path, 'w', encoding='utf-8') as f:
                    f.write("=== Coordinate Quality Diagnostics ===\n\n")
                    if 'sidechain_rmsd' in diagnostics:
                        f.write(f"Sidechain RMSD (vs GT): {diagnostics['sidechain_rmsd']:.4f} Angstrom\n\n")

                    f.write("=== Logits Quality: Predicted Coords vs GT Coords ===\n")
                    if 'perplexity_with_pred_coords' in diagnostics and 'perplexity_with_gt_coords' in diagnostics:
                        ppl_pred = diagnostics['perplexity_with_pred_coords']
                        ppl_gt = diagnostics['perplexity_with_gt_coords']
                        f.write(f"Perplexity with predicted coords: {ppl_pred:.3f}\n")
                        f.write(f"Perplexity with GT coords:        {ppl_gt:.3f}\n")
                        f.write(f"Perplexity degradation:           {ppl_pred - ppl_gt:.3f} ({((ppl_pred/ppl_gt - 1)*100):.1f}%)\n\n")

                    if 'recovery_with_pred_coords' in diagnostics and 'recovery_with_gt_coords' in diagnostics:
                        rec_pred = diagnostics['recovery_with_pred_coords']
                        rec_gt = diagnostics['recovery_with_gt_coords']
                        f.write(f"Recovery with predicted coords:   {rec_pred:.3f}\n")
                        f.write(f"Recovery with GT coords:          {rec_gt:.3f}\n")
                        f.write(f"Recovery degradation:             {rec_pred - rec_gt:.3f} ({((rec_pred/rec_gt - 1)*100):.1f}%)\n\n")

                    f.write("=== Interpretation ===\n")
                    f.write("- Sidechain RMSD: 坐标误差（越小越好）\n")
                    f.write("- PPL degradation: 因坐标误差导致的perplexity增加\n")
                    f.write("- 如果PPL(pred) >> PPL(GT)，说明logits质量下降是由坐标误差导致的\n")

            atom14_pred_np = atom14_pred[b].detach().cpu().numpy()

            pred_aatype = aa_pred[b].detach().long()
            residx_atom37_to_atom14 = make_new_atom14_resid(pred_aatype.cpu()).cpu()
            atom37_atom_exists = residue_constants.STANDARD_ATOM_MASK[pred_aatype.cpu().numpy()]

            protein_batch = {
                'residx_atom37_to_atom14': residx_atom37_to_atom14,
                'atom37_atom_exists': atom37_atom_exists,
            }

            atom37 = all_atom.atom14_to_atom37(
                torch.tensor(atom14_pred_np),
                protein_batch
            )

            aa_pred_np = aa_pred[b].detach().cpu().numpy()
            pdb_path = os.path.join(sample_dir, 'predicted.pdb')
            au.write_prot_to_pdb(atom37, pdb_path, aatype=aa_pred_np, no_indexing=True, overwrite=True)

            predictions.append({
                'sample_id': sample_id,
                'fasta_path': fasta_path,
                'pdb_path': pdb_path,
                'update_mask': update_mask[b].detach().cpu(),
            })

        return predictions

    def _save_structures(self, batch, batch_idx, sample_out, method_name='ode', base_dir=None):
        """
        保存结构文件的通用函数

        Args:
            batch: 输入batch
            batch_idx: batch索引
            sample_out: 采样输出（包含logits, atoms14等）
            method_name: 方法名称，用于创建子目录（'ode' 或 'sde'）
        """
        sample_root = base_dir if base_dir is not None else self._get_inference_run_dir()

        # 在根目录下创建方法特定的子目录
        method_dir = os.path.join(sample_root, method_name)
        os.makedirs(method_dir, exist_ok=True)

        # 保存配置（只保存一次）
        cfg_path = os.path.join(sample_root, 'config.yaml')
        if not os.path.exists(cfg_path):
            try:
                OmegaConf.save(self._full_cfg, cfg_path)
            except Exception as exc:
                self._print_logger.warning(f"Failed to dump config: {exc}")

        # 获取样本ID
        sample_ids = batch.get('csv_idx')
        if sample_ids is None:
            sample_ids = torch.arange(batch['res_mask'].shape[0])
        sample_ids = sample_ids.squeeze().tolist()
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids

        # 获取名称映射
        name_map = None
        try:
            import pandas as pd
            csv_path = getattr(self._data_cfg, 'csv_path', None)
            if csv_path and os.path.exists(csv_path):
                df_names = pd.read_csv(csv_path)
                col = 'pdb_name' if 'pdb_name' in df_names.columns else ('name' if 'name' in df_names.columns else None)
                if col is not None:
                    name_map = df_names[col].astype(str).tolist()
        except Exception:
            name_map = None

        source_names = batch.get('source_name', None)

        # 从sample_out提取结果
        logits = sample_out['logits_final']
        atom14_global = sample_out['atoms14_global_final']
        atom14_local = sample_out['atoms14_local_final']
        update_mask = sample_out['update_mask']
        diagnostics = sample_out.get('diagnostics', {})

        aa_pred = logits.argmax(dim=-1)
        aa_true = batch['aatype'] if 'aatype' in batch else aa_pred
        rec_mask = batch.get('res_mask', torch.ones_like(aa_pred))
        recovery = type_top1_acc(logits, aa_true, node_mask=rec_mask)
        _, perplexity = compute_CE_perplexity(logits, aa_true, mask=rec_mask)

        atom14_pred = atom14_global
        res_mask = batch['res_mask']
        idx_to_aa = {residue_constants.restype_order[aa]: aa for aa in residue_constants.restypes}

        predictions = []
        for b, sample_id in enumerate(sample_ids):
            try:
                sid = int(sample_id)
            except Exception:
                sid = sample_id

            base_name = None
            if isinstance(source_names, list) and b < len(source_names) and source_names[b] is not None:
                try:
                    base_name = str(source_names[b])
                except Exception:
                    base_name = None
            if base_name is None and name_map is not None and isinstance(sid, int) and 0 <= sid < len(name_map):
                base_name = name_map[sid]
            if base_name is not None:
                base_name = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in base_name)

            tag = f"{sid:06d}" if isinstance(sid, int) else str(sid)
            if base_name:
                tag = f"{base_name}_{tag}"

            mask = res_mask[b].bool()
            aa_true_seq = ''.join([idx_to_aa.get(int(x), 'X') for x in aa_true[b][mask].tolist()])
            aa_pred_seq = ''.join([idx_to_aa.get(int(x), 'X') for x in aa_pred[b][mask].tolist()])

            # 在方法目录下创建样本目录
            sample_dir = os.path.join(method_dir, f'sample_{tag}')
            os.makedirs(sample_dir, exist_ok=True)

            # 保存FASTA
            fasta_path = os.path.join(sample_dir, 'sequence.fasta')
            with open(fasta_path, 'w') as f:
                f.write(f'>original_{sample_id}\n')
                f.write(f'{aa_true_seq}\n')
                f.write(f'>predicted_{sample_id}_({method_name}) recovery={float(recovery):.3f} perplexity={float(perplexity):.3f}\n')
                f.write(f'{aa_pred_seq}\n')

            # 保存诊断信息
            if diagnostics:
                diag_path = os.path.join(sample_dir, 'diagnostics.txt')
            with open(diag_path, 'w', encoding='utf-8') as f:
                    f.write(f"=== {method_name.upper()} Sampling Results ===\n\n")
                    f.write("=== Coordinate Quality Diagnostics ===\n\n")
                    if 'sidechain_rmsd' in diagnostics:
                        f.write(f"Sidechain RMSD (vs GT): {diagnostics['sidechain_rmsd']:.4f} Angstrom\n\n")

                    f.write("=== Logits Quality: Predicted Coords vs GT Coords ===\n")
                    if 'perplexity_with_pred_coords' in diagnostics and 'perplexity_with_gt_coords' in diagnostics:
                        ppl_pred = diagnostics['perplexity_with_pred_coords']
                        ppl_gt = diagnostics['perplexity_with_gt_coords']
                        f.write(f"Perplexity with predicted coords: {ppl_pred:.3f}\n")
                        f.write(f"Perplexity with GT coords:        {ppl_gt:.3f}\n")
                        f.write(f"Perplexity degradation:           {ppl_pred - ppl_gt:.3f} ({((ppl_pred/ppl_gt - 1)*100):.1f}%)\n\n")

                    if 'recovery_with_pred_coords' in diagnostics and 'recovery_with_gt_coords' in diagnostics:
                        rec_pred = diagnostics['recovery_with_pred_coords']
                        rec_gt = diagnostics['recovery_with_gt_coords']
                        f.write(f"Recovery with predicted coords:   {rec_pred:.3f}\n")
                        f.write(f"Recovery with GT coords:          {rec_gt:.3f}\n")
                        f.write(f"Recovery degradation:             {rec_pred - rec_gt:.3f} ({((rec_pred/rec_gt - 1)*100):.1f}%)\n\n")

            # 保存PDB
            atom14_pred_np = atom14_pred[b].detach().cpu().numpy()
            pred_aatype = aa_pred[b].detach().long()
            residx_atom37_to_atom14 = make_new_atom14_resid(pred_aatype.cpu()).cpu()
            atom37_atom_exists = residue_constants.STANDARD_ATOM_MASK[pred_aatype.cpu().numpy()]

            protein_batch = {
                'residx_atom37_to_atom14': residx_atom37_to_atom14,
                'atom37_atom_exists': atom37_atom_exists,
            }

            atom37 = all_atom.atom14_to_atom37(
                torch.tensor(atom14_pred_np),
                protein_batch
            )

            aa_pred_np = aa_pred[b].detach().cpu().numpy()
            pdb_path = os.path.join(sample_dir, 'predicted.pdb')
            au.write_prot_to_pdb(atom37, pdb_path, aatype=aa_pred_np, no_indexing=True, overwrite=True)

            predictions.append({
                'sample_id': sample_id,
                'fasta_path': fasta_path,
                'pdb_path': pdb_path,
                'update_mask': update_mask[b].detach().cpu(),
                'method': method_name,
            })

        return predictions

    def predict_step_shdiffusion(self, batch, batch_idx):
        """SH diffusion decoder inference."""
        if self._exp_cfg.task != 'shdiffusion':
            raise RuntimeError('predict_step_shdiffusion only valid for shdiffusion task')

        sample_root = self.inference_dir if self.inference_dir is not None else None
        os.makedirs(sample_root, exist_ok=True)

        cfg_path = os.path.join(sample_root, 'config.yaml')
        if not os.path.exists(cfg_path):
            try:
                OmegaConf.save(self._full_cfg, cfg_path)
            except Exception as exc:
                self._print_logger.warning(f"Failed to dump config: {exc}")

        device = batch['res_mask'].device
        self.interpolant.set_device(device)

        sh_params = {
            'L_max': self._model_cfg.sh.L_max,
            'R_bins': self._model_cfg.sh.R_bins,
            'sigma_r': self._exp_cfg.training.get('sh_sigma_r', 0.25),
        }

        sample_root = self._get_inference_run_dir()
        sample_out = self.interpolant.sh_sample_iterative(
            batch,
            self.model,
            sh_params,
            num_timesteps=self._interpolant_cfg.sampling.num_timesteps,
            return_traj=True,
        )

        atoms14_local = sample_out['atoms14_local_final']
        rigid = du.create_rigid(batch['rotmats_1'], batch['trans_1'])
        atoms14_global = rigid[..., None].apply(atoms14_local)
        sample_out['atoms14_global_final'] = atoms14_global

        sample_out['update_mask'] = batch.get('res_mask', torch.ones_like(batch['res_mask']))

        diagnostics = sample_out.get('diagnostics', {})
        atom14_mask = batch.get('atom14_gt_exists')
        if atom14_mask is not None:
            diff = (atoms14_local - batch['atoms14_local']) ** 2
            diff = diff * atom14_mask[..., None]
            denom = atom14_mask.sum().clamp_min(1.0)
            diagnostics['atom14_rmsd'] = torch.sqrt(diff.sum() / denom).item()

            side_mask = atom14_mask[..., 3:]
            if side_mask.sum() > 0:
                side_diff = (atoms14_local[..., 3:, :] - batch['atoms14_local'][..., 3:, :]) ** 2
                side_diff = side_diff * side_mask[..., None]
                side_rmsd = torch.sqrt(side_diff.sum() / side_mask.sum().clamp_min(1.0))
                diagnostics['sidechain_rmsd'] = side_rmsd.item()

        logits = sample_out.get('logits_final')
        if logits is not None and 'aatype' in batch:
            aa_true = batch['aatype']
            rec_mask = batch.get('res_mask', torch.ones_like(aa_true))
            recovery = type_top1_acc(logits, aa_true, node_mask=rec_mask)
            _, perplexity = compute_CE_perplexity(logits, aa_true, mask=rec_mask)
            diagnostics['perplexity_with_pred_coords'] = float(perplexity)
            diagnostics['recovery_with_pred_coords'] = float(recovery)

        if 'trajectory' in sample_out:
            diagnostics['sh_steps'] = len(sample_out['trajectory'])

        sh_pred = sample_out.get('sh_pred_final')
        clean_sh = batch.get('normalize_density')
        density_mask = batch.get('density_mask')
        if sh_pred is not None and clean_sh is not None:
            diff = sh_pred - clean_sh
            if density_mask is not None:
                while density_mask.dim() < diff.dim():
                    density_mask = density_mask.unsqueeze(-1)
                diff = diff * density_mask
            diagnostics['sh_l2'] = float(torch.sqrt((diff ** 2).mean()).item())

        sample_out['diagnostics'] = diagnostics

        method_dir = os.path.join(sample_root, 'shdiff')
        os.makedirs(method_dir, exist_ok=True)

        log_line = None
        if diagnostics:
            diag_str = f"[SH Predict] (step={self.global_step}) "
            if 'atom14_rmsd' in diagnostics:
                diag_str += f"Atom14_RMSD={diagnostics['atom14_rmsd']:.4f}A  "
            if 'sidechain_rmsd' in diagnostics:
                diag_str += f"Sidechain_RMSD={diagnostics['sidechain_rmsd']:.4f}A  "
            if 'perplexity_with_pred_coords' in diagnostics:
                diag_str += f"PPL={diagnostics['perplexity_with_pred_coords']:.3f}  "
            if 'recovery_with_pred_coords' in diagnostics:
                diag_str += f"Recovery={diagnostics['recovery_with_pred_coords']:.3f}  "
            if 'sh_l2' in diagnostics:
                diag_str += f"SH_L2={diagnostics['sh_l2']:.4f}"
            diag_str += f"| saved_to={method_dir}"
            self._print_logger.info(diag_str)
            log_line = diag_str + "\n"

        predictions = self._save_structures(batch, batch_idx, sample_out, method_name='shdiff', base_dir=sample_root)

        if log_line is not None:
            log_path = os.path.join(method_dir, 'inference.log')
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(log_line)
            except Exception as exc:
                self._print_logger.warning(f"Failed to write SH inference log: {exc}")

        return predictions

    def predict_stepodesde(self, batch, batch_idx):
        """
        对比 ODE 和 SDE 两种采样方法的 predict_step
        自动运行两次推理并记录对比结果，同时保存两种方法的结果到不同子目录
        """
        if self._exp_cfg.task not in ('shfbb', 'shfbb_infer'):
            raise RuntimeError('predict_step is only implemented for FBB tasks')

        sample_root = self.inference_dir if self.inference_dir is not None else None
        os.makedirs(sample_root, exist_ok=True)

        # 初始化统计
        if not hasattr(self, '_ode_sde_stats'):
            self._ode_sde_stats = {
                'ode_rmsds': [],
                'sde_rmsds': [],
                'ode_ppls': [],
                'sde_ppls': [],
                'ode_recoveries': [],
                'sde_recoveries': [],
            }

        device = batch['res_mask'].device
        self.interpolant.set_device(device)

        # GT坐标
        gt_atoms14_local = batch['atoms14_local']  # [B, N, 14, 3]
        gt_sidechain = gt_atoms14_local[..., 3:, :]  # [B, N, 11, 3]
        side_exists = batch['atom14_gt_exists'][..., 3:]  # [B, N, 11]

        # 准备推理
        prepared = self.interpolant.fbb_prepare_batch(batch)

        # 运行 ODE 采样 (1步)
        with torch.no_grad():
            sample_out_ode = self.interpolant.fbb_sample_iterative(
                prepared,
                self.model,
                num_timesteps=1,  # 强制使用1步ODE
            )
            # 为了保存需要添加update_mask
            sample_out_ode['update_mask'] = prepared['update_mask']

            pred_sidechain_ode = sample_out_ode['atoms14_local_final'][..., 3:, :]
            diagnostics_ode = sample_out_ode.get('diagnostics', {})

            # 计算 ODE RMSD
            def compute_sidechain_rmsd(pred_atoms, gt_atoms, exists_mask):
                diff = (pred_atoms - gt_atoms) ** 2
                diff = diff.sum(dim=-1)
                rmsd_per_atom = torch.sqrt(diff + 1e-8)
                mask = exists_mask.float()
                num_atoms = mask.sum()
                if num_atoms > 0:
                    mean_rmsd = (rmsd_per_atom * mask).sum() / num_atoms
                else:
                    mean_rmsd = torch.tensor(0.0)
                return mean_rmsd.item()

            rmsd_ode = compute_sidechain_rmsd(
                pred_sidechain_ode[0],
                gt_sidechain[0],
                side_exists[0]
            )
            ppl_ode = diagnostics_ode.get('perplexity_with_pred_coords', None)
            recovery_ode = diagnostics_ode.get('recovery_with_pred_coords', None)

        # 重新准备 batch (避免状态污染)
        prepared = self.interpolant.fbb_prepare_batch(batch)

        # 运行 SDE 采样
        with torch.no_grad():
            sample_out_sde = self.interpolant.fbb_sample_iterative_sde(
                prepared,
                self.model,
                tau=0.3,
                w_cutoff=0.99,
            )
            # 为了保存需要添加update_mask
            sample_out_sde['update_mask'] = prepared['update_mask']

            pred_sidechain_sde = sample_out_sde['atoms14_local_final'][..., 3:, :]
            diagnostics_sde = sample_out_sde.get('diagnostics', {})

            # 计算 SDE RMSD
            rmsd_sde = compute_sidechain_rmsd(
                pred_sidechain_sde[0],
                gt_sidechain[0],
                side_exists[0]
            )
            ppl_sde = diagnostics_sde.get('perplexity_with_pred_coords', None)
            recovery_sde = diagnostics_sde.get('recovery_with_pred_coords', None)

        # 记录统计
        self._ode_sde_stats['ode_rmsds'].append(rmsd_ode)
        self._ode_sde_stats['sde_rmsds'].append(rmsd_sde)
        if ppl_ode is not None:
            self._ode_sde_stats['ode_ppls'].append(ppl_ode)
        if ppl_sde is not None:
            self._ode_sde_stats['sde_ppls'].append(ppl_sde)
        if recovery_ode is not None:
            self._ode_sde_stats['ode_recoveries'].append(recovery_ode)
        if recovery_sde is not None:
            self._ode_sde_stats['sde_recoveries'].append(recovery_sde)

        # 输出对比信息
        log_msg = (
            f"[ODE vs SDE] Sample {batch_idx}: "
            f"ODE_RMSD={rmsd_ode:.4f}A, SDE_RMSD={rmsd_sde:.4f}A, "
            f"Diff={(rmsd_sde - rmsd_ode):+.4f}A "
            f"({'SDE better' if rmsd_sde < rmsd_ode else 'ODE better'})"
        )
        if ppl_ode is not None and ppl_sde is not None:
            log_msg += f" | ODE_PPL={ppl_ode:.3f}, SDE_PPL={ppl_sde:.3f}"
        if recovery_ode is not None and recovery_sde is not None:
            log_msg += f" | ODE_Rec={recovery_ode:.3f}, SDE_Rec={recovery_sde:.3f}"
        self._print_logger.info(log_msg)

        # 每10个样本输出一次累积统计
        if (batch_idx + 1) % 10 == 0:
            ode_rmsds = self._ode_sde_stats['ode_rmsds']
            sde_rmsds = self._ode_sde_stats['sde_rmsds']
            ode_ppls = self._ode_sde_stats['ode_ppls']
            sde_ppls = self._ode_sde_stats['sde_ppls']
            ode_recoveries = self._ode_sde_stats['ode_recoveries']
            sde_recoveries = self._ode_sde_stats['sde_recoveries']

            stats_msg = (
                f"\n[ODE vs SDE Statistics] After {len(ode_rmsds)} samples:\n"
                f"  ODE avg RMSD: {np.mean(ode_rmsds):.4f} ± {np.std(ode_rmsds):.4f} Angstrom\n"
                f"  SDE avg RMSD: {np.mean(sde_rmsds):.4f} ± {np.std(sde_rmsds):.4f} Angstrom\n"
                f"  Difference: {(np.mean(sde_rmsds) - np.mean(ode_rmsds)):+.4f} Angstrom"
            )

            if len(ode_ppls) > 0 and len(sde_ppls) > 0:
                stats_msg += (
                    f"\n  ODE avg PPL: {np.mean(ode_ppls):.3f} ± {np.std(ode_ppls):.3f}\n"
                    f"  SDE avg PPL: {np.mean(sde_ppls):.3f} ± {np.std(sde_ppls):.3f}\n"
                    f"  Difference: {(np.mean(sde_ppls) - np.mean(ode_ppls)):+.3f}"
                )

            if len(ode_recoveries) > 0 and len(sde_recoveries) > 0:
                stats_msg += (
                    f"\n  ODE avg Recovery: {np.mean(ode_recoveries):.3f} ± {np.std(ode_recoveries):.3f}\n"
                    f"  SDE avg Recovery: {np.mean(sde_recoveries):.3f} ± {np.std(sde_recoveries):.3f}\n"
                    f"  Difference: {(np.mean(sde_recoveries) - np.mean(ode_recoveries)):+.3f}"
                )

            self._print_logger.info(stats_msg)

        # 保存ODE和SDE的结果到不同子目录
        self._print_logger.info(f"[Saving] Saving ODE results to {sample_root}/ode/")
        ode_predictions = self._save_structures(batch, batch_idx, sample_out_ode, method_name='ode')

        self._print_logger.info(f"[Saving] Saving SDE results to {sample_root}/sde/")
        sde_predictions = self._save_structures(batch, batch_idx, sample_out_sde, method_name='sde')

        # 返回ODE的predictions（保持兼容性）
        return ode_predictions
