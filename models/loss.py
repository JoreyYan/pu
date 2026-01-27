import torch
import torch.nn.functional as F
from openfold.utils import rigid_utils
from data import utils as du
from data import all_atom
from data import so3_utils
from openfold.np import residue_constants
from scipy.spatial.transform import Rotation
import numpy as np

def type_ce_loss(logits, targets, node_mask, label_smoothing=0.0):
    """
    logits: [B, N, 20]
    targets: [B, N]
    node_mask: [B, N]
    """
    loss = F.cross_entropy(
        logits.transpose(1, 2),
        targets,
        reduction='none',
        label_smoothing=label_smoothing
    )
    loss = torch.sum(loss * node_mask) / (torch.sum(node_mask) + 1e-10)
    return loss

def type_top1_acc(logits, targets, node_mask=None):
    """
    logits: [B, N, 20]
    targets: [B, N]
    node_mask: [B, N]
    """
    preds = torch.argmax(logits, dim=-1)
    acc = (preds == targets).float()
    if node_mask is not None:
        acc = acc * node_mask
        acc = torch.sum(acc) / (torch.sum(node_mask) + 1e-10)
    else:
        acc = torch.mean(acc)
    return acc

def compute_CE_perplexity(logits, targets, mask=None):
    """
    logits: [B, N, 20]
    targets: [B, N]
    mask: [B, N]
    """
    loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction='none')
    if mask is not None:
        loss = loss * mask
        loss = torch.sum(loss) / (torch.sum(mask) + 1e-10)
    else:
        loss = torch.mean(loss)
    perplexity = torch.exp(loss)
    return loss, perplexity

def torsion_angle_loss(pred_angles, gt_angles, gt_alt_angles, mask):
    """
    pred_angles: [B, N, 7, 2] (sin, cos)
    gt_angles: [B, N, 7, 2]
    gt_alt_angles: [B, N, 7, 2]
    mask: [B, N, 7]
    """
    # L2 loss on sin/cos
    l2_loss = torch.sum((pred_angles - gt_angles) ** 2, dim=-1)
    l2_alt_loss = torch.sum((pred_angles - gt_alt_angles) ** 2, dim=-1)
    loss = torch.minimum(l2_loss, l2_alt_loss)
    loss = torch.sum(loss * mask) / (torch.sum(mask) + 1e-10)
    return loss

def huber(pred, target, mask, delta=1.0):
    """
    pred: [B, N, ...]
    target: [B, N, ...]
    mask: [B, N, ...]
    """
    loss = F.huber_loss(pred, target, reduction='none', delta=delta)
    loss = torch.sum(loss * mask) / (torch.sum(mask) + 1e-10)
    return loss

def pairwise_distance_loss(pred, target, mask, use_huber=False):
    """
    pred: [B, N, 3]
    target: [B, N, 3]
    mask: [B, N]
    """
    # Pairwise distances
    pred_dists = torch.cdist(pred, pred)
    target_dists = torch.cdist(target, target)
    
    # Mask
    mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
    
    if use_huber:
        loss = F.huber_loss(pred_dists, target_dists, reduction='none')
    else:
        loss = (pred_dists - target_dists) ** 2
        
    loss = torch.sum(loss * mask_2d) / (torch.sum(mask_2d) + 1e-10)
    return loss

def backbone_mse_loss(pred, target, mask, bb_atom_scale=1.0):
    """
    pred: [B, N, 3]
    target: [B, N, 3]
    mask: [B, N]
    """
    loss = (pred - target) ** 2
    loss = torch.sum(loss * mask.unsqueeze(-1)) / (torch.sum(mask) + 1e-10)
    return loss * bb_atom_scale

def make_w_l(l, w_max=1.0):
    return w_max

def std_lr_from_batch_masked(batch, mask):
    return 1.0

class SideAtomsIGALoss_Final:
    def __init__(self):
        pass
    
    def __call__(self, model_out, batch):
        return {}

class BackboneGaussianAutoEncoderLoss:
    def __init__(self):
        pass
    
    def __call__(self, model_out, batch):
        return {}



def masked_mean(x, mask, dim=None, eps=1e-10):
    # mask broadcast to x
    num = (x * mask).sum(dim=dim)
    den = mask.sum(dim=dim).clamp_min(eps)
    return num / den



def LinearBridgeLoss(self, model_output, noisy_batch,cfg,frames):


        # ------------------
        # masks
        # ------------------
        res_mask = noisy_batch["res_mask"]                  # [B,N]
        diffuse_mask = noisy_batch["diffuse_mask"]          # [B,N]
        loss_mask = (res_mask * diffuse_mask).float()       # [B,N]
        if torch.any(loss_mask.sum(dim=-1) < 1):
            raise ValueError("Empty batch encountered")

        B, N = loss_mask.shape
        mask_v = loss_mask[..., None]                       # [B,N,1]

        # ------------------
        # noisy state at t
        # ------------------
        trans_t   = noisy_batch["trans_t"]                  # [B,N,3]
        rotmats_t = noisy_batch["rotmats_t"]                # [B,N,3,3]
        t         = noisy_batch["t"]                        # [B] or [B,1]
        if t.ndim == 2: t = t[:, 0]
        denom = (1.0 - t).clamp_min(getattr(cfg, "vf_denom_min", 1e-3))  # [B]
        denom_bt = denom[:, None, None]                     # [B,1,1]

        # ------------------
        # ground truth endpoint + (optional) gt vectorfields from bridge
        # ------------------
        trans_1   = noisy_batch["trans_1"]                  # [B,N,3]
        rotmats_1 = noisy_batch["rotmats_1"]                # [B,N,3,3]

        # linear-bridge provided gt vf (preferred if consistent)
        gt_trans_v = noisy_batch.get("trans_v", None)       # [B,N,3]
        gt_rot_v   = noisy_batch.get("rot_v", None)         # [B,N,3]

        # if not provided, derive from endpoint (self-consistent)
        if gt_trans_v is None:
            gt_trans_v = (trans_1 - trans_t) / denom_bt
        if gt_rot_v is None:
            gt_rot_v = self.so3_utils.calc_rot_vf(rotmats_t, rotmats_1)  # typically log(Rt^T R1)/(1-t)



        pred_trans_1   = model_output["pred_trans"]         # [B,N,3]
        pred_rotmats_1 = model_output["pred_rotmats"]       # [B,N,3,3]

        # ------------------
        # derive pred vf from endpoint (your preferred form)
        # ------------------
        pred_trans_v = (pred_trans_1 - trans_t) / denom_bt
        pred_rot_v   = self.so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)

        # ------------------
        # vf losses (R,t)
        # ------------------
        trans_v_mse = ((gt_trans_v - pred_trans_v) ** 2) * mask_v
        trans_v_loss_per_ex = trans_v_mse.sum(dim=(-1, -2)) / (loss_mask.sum(dim=-1).clamp_min(1e-10))

        rot_v_mse = ((gt_rot_v - pred_rot_v) ** 2) * mask_v
        rot_v_loss_per_ex = rot_v_mse.sum(dim=(-1, -2)) / (loss_mask.sum(dim=-1).clamp_min(1e-10))

        trans_loss = trans_v_loss_per_ex * getattr(cfg, "trans_loss_weight", 1.0)
        rot_loss   = rot_v_loss_per_ex   * getattr(cfg, "rot_loss_weight", 1.0)

        # ------------------
        # backbone loss (endpoint)
        # ------------------
        bb_atom_loss = torch.zeros((B,), device=trans_loss.device)
        if (frames is not None) and ("chain_idx" in noisy_batch):
            chain_idx = noisy_batch["chain_idx"]
            gt_bb = frames(rotmats_1,     trans_1,     chain_idx)  # [B,N,?,3]
            pr_bb = frames(pred_rotmats_1, pred_trans_1, chain_idx)
            # scale if you want
            bb_scale = getattr(cfg, "bb_atom_scale", 1.0)
            gt_bb = gt_bb * bb_scale
            pr_bb = pr_bb * bb_scale

            # mask broadcast: [B,N,1,1]
            bb_mask = loss_mask[..., None, None]
            bb_atom_loss = ((gt_bb - pr_bb) ** 2 * bb_mask).sum(dim=(-1,-2,-3)) / (bb_mask.sum(dim=(-1,-2)).clamp_min(1e-10))
            bb_atom_loss = bb_atom_loss * getattr(cfg, "bb_atom_loss_weight", 0.0)

        # ------------------
        # ellipsoid scaling + local mean losses (endpoint)
        # ------------------
        # You must map these keys to your actual model_output/noisy_batch keys
        # Example expected shapes:
        #   scaling: [B,N,3] (positive)  -> recommend compare in log-space
        #   local_mean: [B,N,3]
        ell_scale_loss = torch.zeros((B,), device=trans_loss.device)
        ell_mu_loss    = torch.zeros((B,), device=trans_loss.device)

        if ("ellipsoid_scaling" in model_output) and ("ellipsoid_scaling_1" in noisy_batch):
            pred_s = model_output["ellipsoid_scaling"]          # [B,N,3]
            gt_s   = noisy_batch["ellipsoid_scaling_1"]         # [B,N,3]
            # log-space for stability (avoid negative / scale explosion)
            pred_ls = torch.log(pred_s.clamp_min(1e-6))
            gt_ls   = torch.log(gt_s.clamp_min(1e-6))
            mse = ((pred_ls - gt_ls) ** 2) * mask_v
            ell_scale_loss = mse.sum(dim=(-1,-2)) / (loss_mask.sum(dim=-1).clamp_min(1e-10))
            ell_scale_loss = ell_scale_loss * getattr(cfg, "ellipsoid_scaling_loss_weight", 0.0)

        if ("ellipsoid_local_mean" in model_output) and ("ellipsoid_local_mean_1" in noisy_batch):
            pred_mu = model_output["ellipsoid_local_mean"]      # [B,N,3]
            gt_mu   = noisy_batch["ellipsoid_local_mean_1"]     # [B,N,3]
            mse = ((pred_mu - gt_mu) ** 2) * mask_v
            ell_mu_loss = mse.sum(dim=(-1,-2)) / (loss_mask.sum(dim=-1).clamp_min(1e-10))
            ell_mu_loss = ell_mu_loss * getattr(cfg, "ellipsoid_local_mean_loss_weight", 0.0)

        # ------------------
        # sequence loss (token CE)
        # ------------------
        seq_loss = torch.zeros((B,), device=trans_loss.device)
        if ("seq_logits" in model_output) and ("seq_1" in noisy_batch):
            # logits: [B,N,V], targets: [B,N]
            logits = model_output["seq_logits"]
            targets = noisy_batch["seq_1"].long()
            # CE per token
            ce = F.cross_entropy(logits.transpose(1,2), targets, reduction="none")  # [B,N]
            seq_loss = (ce * loss_mask).sum(dim=-1) / (loss_mask.sum(dim=-1).clamp_min(1e-10))
            seq_loss = seq_loss * getattr(cfg, "seq_loss_weight", 0.0)

        # ------------------
        # final
        # ------------------
        final_loss_per_ex = trans_loss + rot_loss + bb_atom_loss + ell_scale_loss + ell_mu_loss + seq_loss
        total_loss = final_loss_per_ex.mean()

        aux = {
            "total_loss": total_loss.detach(),
            "trans_vf_loss": trans_loss.mean().detach(),
            "rot_vf_loss": rot_loss.mean().detach(),
            "bb_atom_loss": bb_atom_loss.mean().detach(),
            "ellipsoid_scaling_loss": ell_scale_loss.mean().detach(),
            "ellipsoid_local_mean_loss": ell_mu_loss.mean().detach(),
            "seq_loss": seq_loss.mean().detach(),
            "res_length": loss_mask.sum(dim=-1).float().mean().detach(),
            "examples_per_step": torch.tensor(B, device=total_loss.device),
        }
        return total_loss, aux

