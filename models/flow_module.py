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
from models.flow_model import VAE,SHDecoder,SHframe_fbb
from models.ff2 import FF2Model
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
from models.loss import type_ce_loss,type_top1_acc,compute_CE_perplexity,torsion_angle_loss,make_w_l,std_lr_from_batch_masked
from models.shlossv2 import sh_loss_with_masks,sh_mse_loss
from data.sh_density import sh_density_from_atom14_with_masks
import torch.nn.functional as F
from models.loss import huber,pairwise_distance_loss,backbone_mse_loss
from openfold.np.residue_constants import restype_name_to_atom14_names
from models.ff2flow.ff2_dependencies import FF2Dependencies
from models.flow_model import SideAtomsFlowModel,SideAtomsFlowModel_backup
from models.shattetnion.ShDecoderSidechain import  SHSidechainDecoder,DynamicKSidechainDecoder,assemble_atom14_with_CA,SHGeoResHead
from openfold.utils import loss as openfold_loss
from openfold.utils import rigid_utils
import openfold.np.residue_constants as rc
from openfold.np.protein import from_prediction, to_pdb
from data.all_atom import atom14_to_atom37, make_new_atom14_resid
import  data.all_atom  as all_atom
from experiments.utils import _load_submodule_from_ckpt,load_partial_state_dict
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
    pdb_out=pdb_out
    ATOMS = ["N","CA","C"]
    out = open(pdb_out,"w")
    k = 0
    a = 0
    for x,y,z in xyz:
        out.write(
            "ATOM  %5d  %2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
            % (k+1,ATOMS[k%3],"GLY","A",a+1,x,y,z,1,0)
        )
        k += 1
        if k % 3 == 0: a += 1
    out.close()

def save_4pdb(xyz, pdb_out="out.pdb"):
    pdb_out=pdb_out
    ATOMS = ["N","CA","C","O"]
    out = open(pdb_out,"w")
    k = 0
    a = 0
    for x,y,z in xyz:
        out.write(
            "ATOM  %5d  %2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
            % (k+1,ATOMS[k%4],"GLY","A",a+1,x,y,z,1,0)
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
        self.interpolant = Interpolant(cfg.interpolant,cfg.experiment.task,cfg.experiment.noise_scheme)

        # self.frames=FrameBuilder()

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None

        # self.SHDecoder=SHDecoder(cfg.model)


        # self.DynamicKSidechainDecoder=DynamicKSidechainDecoder(SHSidechainDecoder(**cfg.model.sh))

        # test_torision rec
        # self.PredHead=SHGeoResHead( C=cfg.model.sh.C, L_max=cfg.model.sh.L_max, R_bins=cfg.model.sh.R_bins)
        # weight='/home/junyu/project/protein-frame-flow-u/experiments/ckpt/se3-fm_sh/hallucination_pdb_SHaapred/2025-08-24_21-04-49/last.ckpt'
        # _load_submodule_from_ckpt(self.PredHead, weight, lightning_key="state_dict", source_prefix=None)
        #
        # self.PredHead.requires_grad_(False)  # 冻结权重
        # self.PredHead.eval()  # 关掉 Dropout/BN 的统计更新


        self.model=SideAtomsFlowModel_backup( cfg.model)
        # mw='/home/junyu/project/protein-frame-flow-u/experiments/ckpt/se3-fm_sh/pdb_seperated_Rm0_t0/2025-08-25_00-01-29/last.ckpt'
        # _load_submodule_from_ckpt(self.model, mw, lightning_key="state_dict", source_prefix=None)

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
        logits  = self.model(noisy_batch['density'],noisy_batch['res_mask'],noisy_batch['aatype']
                             ,noisy_batch['rotmats_1'],noisy_batch['trans_1'])
        loss = type_ce_loss(logits, noisy_batch['aatype'], node_mask=noisy_batch['res_mask'], label_smoothing=0.05)
        acc = type_top1_acc(logits, noisy_batch['aatype'])
        return {'aaloss':loss,
                'aa_acc':acc}

    def model_step_decoder(self, noisy_batch: Any):


        training_cfg = self._exp_cfg.training

        # self.SHDecoder(noisy_batch, noisy_batch['normalize_density'], noisy_batch['res_mask'],
        #                noisy_batch['aatype'],
        #                noisy_batch['rotmats_1'],
        #                noisy_batch['trans_1'])

        tau=0.2
        tau_mask = noisy_batch['density_mask'] * (torch.abs(noisy_batch['normalize_density']) > tau)
        sh=noisy_batch['normalize_density']*tau_mask

        logits,atoms14  = self.model(sh,node_mask=noisy_batch['res_mask'])

        # TYPELOSS
        typeloss,perplexity = compute_CE_perplexity(logits, noisy_batch['aatype'], mask=noisy_batch['res_mask'])
        acc = type_top1_acc(logits, noisy_batch['aatype'])

        # ANGELLOSS
        #chiloss=torsion_angle_loss(tor_pred,noisy_batch['torsion_angles'],noisy_batch['torsion_alt_angles'],noisy_batch['torsion_mask']).mean()

        # ANGEL TO FRAMES
        #identity=rigid_utils.Rigid.identity(noisy_batch['aatype'].shape,device=noisy_batch['aatype'].device)
        #rigid=all_atom.torsion_angles_to_frames(identity,tor_pred,noisy_batch['aatype'])
        #atoms14_pred_local=all_atom.frames_to_atom14_pos(rigid,noisy_batch['aatype'])*noisy_batch['atom14_gt_exists'][...,None]

        atoms14_pred_local=atoms14*training_cfg.bb_atom_scale*noisy_batch['atom14_gt_exists'][...,None]

        # BACKBONE LOSS
        atoms14_gt_local=noisy_batch['atoms14_local']*noisy_batch['atom14_gt_exists'][...,None]

        local_mse_loss = backbone_mse_loss(atoms14_gt_local, atoms14_pred_local, noisy_batch['atom14_gt_exists'], bb_atom_scale=training_cfg.bb_atom_loss_weight).mean()
        local_pair_loss = pairwise_distance_loss(atoms14_gt_local,
                                           atoms14_pred_local.clone() ,
                                           noisy_batch['atom14_gt_exists'], use_huber=False).mean()
        local_huber_loss=huber(atoms14_pred_local, atoms14_gt_local,noisy_batch['atom14_gt_exists'])

        atomsloss=local_mse_loss+local_huber_loss+local_pair_loss
        loss=atomsloss*training_cfg.atom_loss_weight+typeloss*training_cfg.type_loss_weight  #+chiloss*training_cfg.chil_loss_weight




        return {
            'local_mse_loss':local_mse_loss.detach(),
            'local_pair_loss':local_pair_loss.detach(),
            'local_huber_loss':local_huber_loss.detach(),
            # 'chiloss':chiloss.detach(),
           'typeloss':typeloss.detach(),
            'loss':loss,
           'aa_acc': acc.detach(),
           'perplexity':perplexity
        }

    def model_step_fbb(self, batch: Any,prob=None):


        training_cfg = self._exp_cfg.training
        if self._exp_cfg.task=='diffusion' or self._exp_cfg.task=='shdiffusion':
            noisy_batch = self.interpolant.corrupt_batch(batch)
        elif self._exp_cfg.task=='shfbb':
            if prob is not None:
                noisy_batch = self.interpolant.fbb_corrupt_batch(batch,prob)
            else:
                noisy_batch = self.interpolant.fbb_corrupt_batch(batch,1)
        if self._exp_cfg.task == 'shfbb_infer':

            noisy_batch = self.interpolant.fbb_corrupt_batch(batch,)


        # hit=0.9
        # lowt=0.1

        # zero-like SC (scaled domain); backbone part not used but keep zeros for simplicity
        noisy_batch['atoms14_local_sc'] = torch.zeros_like(noisy_batch['atoms14_local_t'])
        # Sidechain self-conditioning (teacher-forcing style)
        if 'atoms14_local_t' in noisy_batch:
            if torch.rand(()) > 0.5:
                with torch.no_grad():
                    draft_outs = self.model(noisy_batch)
                side_pred_scaled = draft_outs['side_atoms']  # [B,N,11,3] (scaled domain)
                bb_scaled = noisy_batch['atoms14_local_t'][..., :3, :]
                atoms14_local_sc = torch.cat([bb_scaled, side_pred_scaled], dim=-2)
                noisy_batch['atoms14_local_sc'] = atoms14_local_sc.detach()



        outs= self.model(noisy_batch )
        # logits_t_hi = outs['logits']
        # logits_t_lo= self.model(noisy_batch ,lowt)['logits']
        #
        # import torch, torch.nn.functional as F
        # d_l2 = (logits_t_hi - logits_t_lo).pow(2).mean().sqrt()
        # d_l1 = (logits_t_hi - logits_t_lo).abs().mean()
        # cos = torch.nn.functional.cosine_similarity(
        #     logits_t_hi.flatten(1), logits_t_lo.flatten(1), dim=1).mean()
        #
        # # 分布层面
        # p_hi = F.softmax(logits_t_hi, dim=-1)
        # p_lo = F.softmax(logits_t_lo, dim=-1)
        # kl = (p_hi * (p_hi.clamp_min(1e-8).log() - p_lo.clamp_min(1e-8).log())).sum(-1).mean()
        # print(d_l2.item(), d_l1.item(), cos.item(), kl.item())


        side_atoms = outs['side_atoms']  # [B,N,11,3]
        logits = outs['logits']

        # 掩码
        loss_mask = noisy_batch['update_mask'] * noisy_batch['res_mask']  # [B,N]


        # === 类型损失 ===
        typeloss, perplexity = compute_CE_perplexity(
            logits, noisy_batch['aatype'], mask=loss_mask
        )
        acc = type_top1_acc(logits, noisy_batch['aatype'], node_mask=loss_mask)

        # Optional CE/ACC diagnostics
        if getattr(self._exp_cfg.training, 'debug_ce_diag', False):
            self._debug_ce_self_checks(logits, noisy_batch['aatype'], loss_mask, loss_mask)

        # === 侧链坐标损失（坐标统一放回 Å 再算）===
        coord_scale = 8.0
        side_gt_local = noisy_batch['atoms14_local'][..., 3:, :]  # [B,N,11,3]
        # 保证掩码是 bool
        exists11_bool = noisy_batch['atom14_gt_exists'][..., 3:].bool()  # [B,N,11]
        loss_mask_bool = loss_mask.bool()  # [B,N]

        # 逻辑与
        atom_level_mask = exists11_bool & loss_mask_bool[..., None]  # [B,N,11]

        # 转回 float 参与 loss
        atom_level_mask = atom_level_mask.to(side_gt_local.dtype)

        # SNR-aware scaling (counteract high-noise dominance):
        # scale ~ 1 / (1 - t_clip). Larger weights for late timesteps.
        t_clip = getattr(training_cfg, 't_normalize_clip', 0.9)
        eps = 1e-6
        # 放大到 Å 单位
        side_gt_ang = side_gt_local * coord_scale
        side_pred_ang = side_atoms * coord_scale

        if 'r3_t' in noisy_batch:
            # r3_t shape [B,N]; make [B,N,1,1] for broadcasting over [B,N,11,3]
            r3_t = noisy_batch['r3_t'].to(side_gt_local.dtype)
            r3_norm_scale = 1.0 - torch.clamp(r3_t, max=t_clip)
            r3_norm_scale = torch.clamp(r3_norm_scale, min=eps)[..., None, None]
            snr_scale = (getattr(training_cfg, 'bb_atom_scale', 1.0)) / r3_norm_scale
            side_gt_scaled = side_gt_ang * snr_scale
            side_pred_scaled = side_pred_ang * snr_scale
        else:
            side_gt_scaled = side_gt_ang
            side_pred_scaled = side_pred_ang

        local_mse_loss = backbone_mse_loss(
            side_gt_scaled,
            side_pred_scaled,
            atom_level_mask,
            bb_atom_scale=training_cfg.bb_atom_loss_weight
        ).mean()

        local_pair_loss = pairwise_distance_loss(
            side_gt_scaled,
            side_pred_scaled.clone(),
            atom_level_mask,
            use_huber=False
        ).mean()

        local_huber_loss = huber(
            side_pred_scaled,
            side_gt_scaled,
            atom_level_mask
        ).mean()

        atomsloss = local_mse_loss + local_huber_loss + local_pair_loss

        # === 总损失 ===
        loss = (

                atomsloss * training_cfg.atom_loss_weight +
                typeloss * training_cfg.type_loss_weight
        )

        return {
            'local_mse_loss':local_mse_loss.detach(),
            'local_pair_loss':local_pair_loss.detach(),
            'local_huber_loss':local_huber_loss.detach(),
            'typeloss':typeloss.detach(),
            'loss':loss,
            'aa_acc': acc.detach(),
            'perplexity':perplexity
        }

    def model_step_fbb_backup(self, batch: Any, prob=None):
        """旧版 FBB step：无侧链自条件，坐标不做缩放乘回 8。"""
        training_cfg = self._exp_cfg.training
        if self._exp_cfg.task=='diffusion' or self._exp_cfg.task=='shdiffusion':
            noisy_batch = self.interpolant.corrupt_batch(batch)
        elif self._exp_cfg.task=='shfbb':
            noisy_batch = self.interpolant.fbb_corrupt_batch_backup(batch, prob if prob is not None else 1)
        if self._exp_cfg.task == 'shfbb_infer':
            noisy_batch = self.interpolant.fbb_corrupt_batch_backup(batch)

        outs = self.model(noisy_batch)
        side_atoms = outs['side_atoms']  # [B,N,11,3]
        logits = outs['logits']

        loss_mask = noisy_batch['update_mask'] * noisy_batch['res_mask']

        typeloss, perplexity = compute_CE_perplexity(
            logits, noisy_batch['aatype'], mask=loss_mask
        )
        acc = type_top1_acc(logits, noisy_batch['aatype'], node_mask=loss_mask)

        side_gt_local = noisy_batch['atoms14_local'][..., 3:, :]
        exists11_bool = noisy_batch['atom14_gt_exists'][..., 3:].bool()
        atom_level_mask = (exists11_bool & loss_mask.bool()[..., None]).to(side_gt_local.dtype)

        t_clip = getattr(training_cfg, 't_normalize_clip', 0.9)
        eps = 1e-6
        if 'r3_t' in noisy_batch:
            r3_t = noisy_batch['r3_t'].to(side_gt_local.dtype)
            r3_norm_scale = 1.0 - torch.clamp(r3_t, max=t_clip)
            r3_norm_scale = torch.clamp(r3_norm_scale, min=eps)[..., None, None]
            snr_scale = (getattr(training_cfg, 'bb_atom_scale', 1.0)) / r3_norm_scale
            side_gt_scaled = side_gt_local * snr_scale
            side_pred_scaled = side_atoms * snr_scale
        else:
            side_gt_scaled = side_gt_local
            side_pred_scaled = side_atoms

        local_mse_loss = backbone_mse_loss(
            side_gt_scaled,
            side_pred_scaled,
            atom_level_mask,
            bb_atom_scale=training_cfg.bb_atom_loss_weight
        ).mean()

        local_pair_loss = pairwise_distance_loss(
            side_gt_scaled,
            side_pred_scaled.clone(),
            atom_level_mask,
            use_huber=False
        ).mean()

        local_huber_loss = huber(
            side_pred_scaled,
            side_gt_scaled,
            atom_level_mask
        ).mean()

        atomsloss = local_mse_loss + local_huber_loss + local_pair_loss
        loss = (
            atomsloss * training_cfg.atom_loss_weight +
            typeloss * training_cfg.type_loss_weight
        )

        return {
            'local_mse_loss': local_mse_loss.detach(),
            'local_pair_loss': local_pair_loss.detach(),
            'local_huber_loss': local_huber_loss.detach(),
            'typeloss': typeloss.detach(),
            'loss': loss,
            'aa_acc': acc.detach(),
            'perplexity': perplexity,
        }

    @torch.no_grad()
    def _debug_ce_self_checks(self, pred_logits: torch.Tensor, true_labels: torch.Tensor,
                              mask_ce: torch.Tensor | None, mask_acc: torch.Tensor | None,
                              node_mask_for_logits: torch.Tensor | None = None):
        B, N, C = pred_logits.shape
        logits  = pred_logits.reshape(-1, C)
        targets = true_labels.reshape(-1)
        m_ce  = (mask_ce.reshape(-1).bool()  if mask_ce  is not None else torch.ones_like(targets, dtype=torch.bool))
        m_acc = (mask_acc.reshape(-1).bool() if mask_acc is not None else m_ce)
        # 1) dynamic C
        # 2) same mask used
        only_ce  = (m_ce & (~m_acc)).sum().item()
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
    def make_new_protein(self,aatype,result):



        chain_index = result['chain_index'].detach().cpu().numpy()

        # change atoms mask to idea one
        ideal_atom_mask=rc.STANDARD_ATOM_MASK[aatype]
        finial_atom_mask=ideal_atom_mask#*proteins[0]['seqmask'][:,None]

        residx_atom37_to_atom14=make_new_atom14_resid(aatype).cpu().numpy()
        proteins={
            'aatype':aatype.detach().cpu().numpy(),
            'atom37_atom_exists': finial_atom_mask,
            'residx_atom37_to_atom14': residx_atom37_to_atom14,
            'final_14_positions': result['final_atom_positions'],
            'final_atom_mask': finial_atom_mask,
            'residue_index':result['residue_index'],
            'domain_name':result['domain_name'],


        }


        atom37=atom14_to_atom37(proteins['final_14_positions'].squeeze(0),proteins)

        #ADD UNK atoms
        #atom37=atom37+(1-proteins[0]['atom37_atom_exists'][:,:,None])*proteins[0]['all_atom_positions']


        feats={
            'aatype':aatype.detach().cpu().numpy(),
            'residue_index':result['residue_index'].detach().cpu().numpy(),
        }

        results={'final_atom_positions': atom37,
             'final_atom_mask': proteins['final_atom_mask']}




        design_protein=from_prediction(
            features=feats,
            result=results,
            chain_index=chain_index
        )
        path='/home/junyu/project/binder_target/gcpr/preprocessed/select/fbb/'
        unrelaxed_output_path=path+'design_'+str(proteins['domain_name'][0][0].detach().cpu().numpy())

        design_protein_str=to_pdb(design_protein)
        with open(unrelaxed_output_path+'_.pdb', 'w') as fp:
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
        trans_loss *= 1 #int(self._fm_conf.flow_trans)

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
        rot_loss *= 1 #int(self._fm_conf.flow_rot)

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
        if self.global_step>99  and (self.global_step % self.model.PCT.quantizer.restart == 0):
            print(f"Step {self.global_step}: Dead codes: {len(self.model.PCT.quantizer.dead_codes)}")
            self.model.PCT.quantizer.random_restart()

        # 定期重置usage统计（可选）
        if self.global_step>99  and  self.global_step % (self.model.PCT.quantizer.restart * 10) == 0:
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

        gt_bb_atoms=self.frames(gt_rotmats_1,gt_trans_1,noisy_batch['chain_idx'])
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


        pred_all_atoms=self.frames(pred_rotmats_1,pred_trans_1,noisy_batch['chain_idx'])
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
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*4, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*4, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 4))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*4])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 4))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*4])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) + 1)

        se3_vf_loss = trans_loss
        auxiliary_loss = (
            bb_atom_loss * training_cfg.aux_loss_use_bb_loss
            + dist_mat_loss * training_cfg.aux_loss_use_pair_loss
        )

        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        auxiliary_loss = torch.clamp(auxiliary_loss, max=5)

        #fape
        fapeloss=mu.fbb_backbone_loss(pred_trans_1,pred_rotmats_1,gt_trans_1,gt_rotmats_1,loss_mask)


        se3_vf_loss += auxiliary_loss
        se3_vf_loss=se3_vf_loss+self._exp_cfg.training.vq_loss_weight*vq_loss+fapeloss
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
        },pred_all_atoms.detach()


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
        gt_rot_v = noisy_batch['rot_v']      # 旋转向量场
        
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
        if self._exp_cfg.task=='diffusion' or self._exp_cfg.task=='shdiffusion':
            noisy_batch = self.interpolant.corrupt_batch(batch)
            batch_losses = self.model_step(noisy_batch)
        else:
            batch_losses = self.model_step_fbb_backup(batch,prob=1)
        if self._exp_cfg.task!='shfbb_infer':

            num_batch = res_mask.shape[0]
            total_losses = {
                k: torch.mean(v) for k, v in batch_losses.items()
            }
            for k, v in total_losses.items():
                self._log_scalar(
                    f"val/{k}", v.detach().item(), prog_bar=False, batch_size=num_batch)

            # Losses to track. Stratified across t.
            if self._exp_cfg.task=='diffusion':
                for loss_name, loss_dict in batch_losses.items():

                    stratified_losses = mu.t_stratified_mean_loss(
                        batch['t'], loss_dict, loss_name=loss_name)
                    for k, v in stratified_losses.items():
                        self._log_scalar(
                            f"val/{k}", v, prog_bar=False, batch_size=num_batch)
                    # Training throughput
                    scaffold_percent = torch.mean(batch['diffuse_mask'].float()).item()
                    self._log_scalar(
                        "val/scaffolding_percent",
                        scaffold_percent, prog_bar=False, batch_size=num_batch)
                    motif_mask = 1 - batch['diffuse_mask'].float()
                    num_motif_res = torch.sum(motif_mask, dim=-1)
                    self._log_scalar(
                        "val/motif_size",
                        torch.mean(num_motif_res).item(), prog_bar=False, batch_size=num_batch)
                    self._log_scalar(
                        "val/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
                    self._log_scalar(
                        "val/batch_size", num_batch, prog_bar=False)
                    step_time = time.time() - step_start_time
                    self._log_scalar(
                        "val/examples_per_second", num_batch / step_time)
                    val_loss = total_losses['se3_vf_loss']
                    self._log_scalar(
                        "val/loss", val_loss, batch_size=num_batch)
                num_batch, num_res = res_mask.shape

                gt_all_atoms = self.frames(batch['rotmats_1'], batch['trans_1'], batch['chain_idx']).detach().cpu().numpy()
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

            elif self._exp_cfg.task == 'aatype' or self._exp_cfg.task == 'SHdecode'or self._exp_cfg.task == 'shfbb':
                batch_losses_fixed = {}
                for k, v in batch_losses.items():
                    if hasattr(v, 'item'):  # 如果是tensor
                        batch_losses_fixed[k] = [v.item()]
                    else:  # 如果已经是标量
                        batch_losses_fixed[k] = [v]

                self.validation_epoch_metrics.append(pd.DataFrame(batch_losses_fixed))







    def on_validation_epoch_end(self):
        # if len(self.validation_epoch_samples) > 0:
        #     self.logger.log_table(
        #         key='valid/samples',
        #         columns=["sample_path", "global_step", "Protein"],
        #         data=self.validation_epoch_samples)
        #     self.validation_epoch_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for metric_name,metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f'valid/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        print(self.validation_epoch_metrics)
        self.validation_epoch_metrics.clear()

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
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
        if self._exp_cfg.task == 'diffusion' or self._exp_cfg.task == 'shdiffusion':
            noisy_batch = self.interpolant.corrupt_batch(batch)

            # peak部分不计算梯度

        batch_losses = self.model_step_fbb_backup(batch)
        num_batch = batch['res_mask'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
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
                "train/aa_acc",  total_losses['aa_acc'], batch_size=num_batch)
        elif self._exp_cfg.task == 'SHdecode' or self._exp_cfg.task == 'shfbb':
            train_loss= total_losses['loss']

            if self.global_step % 1 == 0:
                for k, v in total_losses.items():
                    self._log_scalar(
                        f"train/{k}", float(v.detach().cpu().item()), prog_bar=True, batch_size=num_batch)


        return train_loss

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params=trainable,
            **self._exp_cfg.optimizer
        )

    def fbb_sample(
        self,
        batch: dict[str, torch.Tensor],
        diffusion_prob: float | None = 1,
        out_dir: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run masked sidechain reconstruction for FBB inference.

        Args:
            batch: input features (same schema as training).
            diffusion_prob: optional mask prob override when calling interpolant.
            out_dir: optional directory to dump outputs (per batch element).

        Returns:
            Dictionary containing reconstructed side chains and related tensors.
        """

        device = batch['res_mask'].device
        self.interpolant.set_device(device)

        with torch.no_grad():
            prepared_batch = self.interpolant.fbb_prepare_batch(batch)
            # Single-step sampling at t = min_t to match the starting bridge
            sample_out = self.interpolant.fbb_sample_iterative(
                prepared_batch,
                self.model,

            )

        logits = sample_out['logits_final']
        atoms14_local = sample_out['atoms14_local_final']
        atoms14_global = sample_out['atoms14_global_final']

        return {
            'logits': logits,
            'update_mask': prepared_batch['update_mask'],
            'atom14_local': atoms14_local,
            'atom14_global': atoms14_global,
        }

    def predict_step(self, batch, batch_idx):
        #del batch_idx  # lightning signature
        if self._exp_cfg.task not in ('shfbb', 'shfbb_infer'):
            raise RuntimeError('predict_step is only implemented for FBB tasks')

        sample_root = self.inference_dir if self.inference_dir is not None else None
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

        result = self.fbb_sample(batch)

        logits = result['logits']
        atom14_global = result['atom14_global']
        atom14_local = result['atom14_local']
        update_mask = result['update_mask']

        aa_pred = logits.argmax(dim=-1)  # [B,N]
        aa_true = batch['aatype'] if 'aatype' in batch else aa_pred
        rec_mask = batch.get('res_mask', torch.ones_like(aa_pred))
        recovery = type_top1_acc(logits, aa_true, node_mask=rec_mask)
        _, perplexity = compute_CE_perplexity(logits, aa_true, mask=rec_mask)

        atom14_pred = atom14_global
        atom14_exists = batch['atom14_gt_exists'] if 'atom14_gt_exists' in batch else torch.ones_like(atom14_pred[..., 0])

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
