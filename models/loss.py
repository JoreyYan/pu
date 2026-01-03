import torch
import torch.nn.functional as F
import numpy as np
from models.IGA import fused_gaussian_overlap_score,compute_robust_nll_components,analytical_inverse_3x3
import torch.nn as nn
def huber(x, y, mask=None, delta=1.0, reduction='mean'):
    diff = x - y
    if mask is not None:

        if len(mask.shape) != len(diff.shape):
            diff = diff * mask[...,None]
        else:
            diff = diff * mask


    abs_diff = diff.abs()
    quad = torch.clamp(abs_diff, max=delta)
    lin  = abs_diff - quad
    loss = 0.5 * quad**2 + delta * lin
    if reduction == 'mean':
        denom = (mask.sum() if mask is not None else torch.numel(loss)).clamp(min=1.0)
        return loss.sum() / denom
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
def type_ce_loss(
    logits: torch.Tensor,        # [B, N, 20]
    target: torch.Tensor,        # [B, N] in {0..19} or ignore_index
    node_mask: torch.Tensor | None = None,  # [B, N] bool/0-1，可选
    label_smoothing: float = 0.0,
    class_weights: torch.Tensor | None = None,  # [20] 可选
    ignore_index: int | None = None,
):
    B, N, C = logits.shape
    assert C == 20, "num_classes must be 20"

    # 展平
    logits = logits.reshape(-1, C)   # [B*N, 20]
    target = target.reshape(-1)      # [B*N]

    # 有 ignore_index 的情况：转成 mask
    if ignore_index is not None:
        valid_mask = (target != ignore_index)
    else:
        valid_mask = torch.ones_like(target, dtype=torch.bool)

    # 叠加 node_mask（如果给了）
    if node_mask is not None:
        node_mask = node_mask.reshape(-1).bool()
        valid_mask = valid_mask & node_mask

    # 过滤无效位置
    if not valid_mask.all():
        logits = logits[valid_mask]
        target = target[valid_mask]

    # 类别权重（处理类别不均衡时可用）
    weight = None
    if class_weights is not None:
        weight = class_weights.to(logits.device, logits.dtype)  # [20]

    # 交叉熵（支持 label smoothing）
    loss = F.cross_entropy(
        logits, target.long(),
        weight=weight,
        reduction='mean',
        label_smoothing=label_smoothing
    )
    return loss
@torch.no_grad()
def std_lr_from_batch_masked(target, mask, floor=1e-3):
    """
    target: [B,N,C,L+1,2L+1,R]  （注意：这里应传“未置零的 GT”，或者把 mask=0 的位置忽略）
    mask:   可广播到 target 形状，1=参与，0=忽略
    返回:   std_lr [L+1,R]，只基于 mask=1 的条目
    """
    x = target * mask  # 未选中的位置相当于 0，但我们会用 mask 计数
    # 在 B,N,C,m 上做加权均值/方差
    cnt = mask.sum(dim=(0,1,2,4)).clamp_min(1.0)        # [L+1,R]
    mean = x.sum(dim=(0,1,2,4)) / cnt                   # [L+1,R]
    var  = ((x - mean[None,None,None,:,None,:])**2 * mask).sum(dim=(0,1,2,4)) / cnt
    std  = torch.sqrt(var).clamp_min(floor)             # [L+1,R]
    return std

# @torch.no_grad()
# def std_lr_from_batch_masked(target, predict_mask, valid_mask=None, floor=1e-3):
#     """
#     仅用本步 mask=1 的条目统计 std_lr，返回 [L+1, R]
#     target:       [B,N,C,L+1,2L+1,R]（GT；不要把mask位改成0）
#     predict_mask: 可广播到 target 形状；1=参与统计
#     valid_mask:   [B,N] 或 None
#     """
#     device = target.device
#     M = predict_mask.to(device).to(target.dtype)
#     if M.shape != target.shape:
#         M = torch.ones_like(target) * M
#     if valid_mask is not None:
#         V = valid_mask.to(device).float().view(target.shape[0], target.shape[1], 1, 1, 1, 1)
#         M = M * V
#
#     x   = target * M
#     cnt = M.sum(dim=(0,1,2,4)).clamp_min(1.0)                                # [L+1,R]
#     mean= x.sum(dim=(0,1,2,4)) / cnt                                         # [L+1,R]
#     var = ((x - mean[None,None,None,:,None,:])**2 * M).sum(dim=(0,1,2,4)) / cnt
#     std = torch.sqrt(var).clamp_min(floor)                                    # [L+1,R]
#     return std
def make_w_l(L_max=8):
    w = torch.tensor([1.0/(2*l+1) for l in range(L_max+1)], dtype=torch.float32)
    return w * (len(w) / w.sum())  # 归一到均值≈1


def charbonnier(x, eps=1e-6):
    return torch.sqrt(x * x + eps)

def sh_loss_with_masks(
    pred, target,
    std_lr, w_l,
    predict_mask=None,   # [B,N,C,L+1,2L+1,R] 或可广播
    valid_mask=None,     # [B,N]
    update_mask=None,    # [B,N]，1=残基要更新
    power_weight=0.2, phase_weight=0.2, tv2_weight=0.05,
):
    """
    pred/target: [B,N,C,L+1,2L+1,R]
    返回: total_mean(标量), parts(dict 含各分量的 [B] 逐样本损失)
    """
    B,N,C,Lp1,M,R = pred.shape
    device = pred.device

    wl  = w_l.to(device).view(1,1,1,Lp1,1)                       # [1,1,1,L+1,1]
    std = None if std_lr is None else std_lr.to(device).view(1,1,1,Lp1,1,R)

    # 组合权重 W: [B,N,C,L+1,2L+1,R]
    W = torch.ones_like(pred, dtype=pred.dtype, device=device)
    if predict_mask is not None:
        W = W * predict_mask.to(device).to(pred.dtype)
    if valid_mask is not None:
        V = valid_mask.to(device).float().view(B,N,1,1,1,1)
        W = W * V
    if update_mask is not None:
        U = update_mask.to(device).float().view(B,N,1,1,1,1)
        W = W * U

    # 若本批无任何有效位置，直接返回 0
    if (W.sum() == 0):
        zero = pred.new_tensor(0.0)
        return zero, {'total_per_sample': torch.zeros(B, device=device),
                      'coef': torch.zeros(B, device=device),
                      'phase': torch.zeros(B, device=device),
                      'power': torch.zeros(B, device=device),
                      'tv2': torch.zeros(B, device=device)}

    # ---------- 1) 系数鲁棒回归（先在 m 维做 masked mean） ----------
    resid = pred - target if std is None else (pred - target) / std
    coef_num_m = (charbonnier(resid) * W).sum(dim=4)              # [B,N,C,L+1,R]
    coef_den_m = W.sum(dim=4).clamp_min(1.0)                      # [B,N,C,L+1,R]
    coef_per   = coef_num_m / coef_den_m                          # [B,N,C,L+1,R]
    valid_vec  = (W.sum(dim=4) > 0).float()                       # [B,N,C,L+1,R]

    # 带 wl 的样本内加权平均 → [B]
    coef_num_b = (coef_per * wl).sum(dim=(1,2,3,4))               # [B]
    coef_den_b = ((valid_vec * wl).sum(dim=(1,2,3,4))).clamp_min(1.0)
    coef_b     = coef_num_b / coef_den_b                          # [B]

    # ---------- 2) m-向量方向一致（masked cosine over m） ----------
    a, b = pred, target
    dot   = (a * b * W).sum(dim=4)                                # [B,N,C,L+1,R]
    a_nrm = torch.sqrt((a*a*W).sum(dim=4).clamp_min(1e-12))
    b_nrm = torch.sqrt((b*b*W).sum(dim=4).clamp_min(1e-12))
    cos   = 1.0 - (dot / (a_nrm * b_nrm + 1e-12))                 # [B,N,C,L+1,R]

    phase_num_b = (cos * valid_vec * wl).sum(dim=(1,2,3,4))       # [B]
    phase_den_b = ((valid_vec * wl).sum(dim=(1,2,3,4))).clamp_min(1.0)
    phase_b     = phase_num_b / phase_den_b                       # [B]

    # ---------- 3) 能量谱（按 m 的 masked mean） ----------
    P_hat = ((pred**2) * W).sum(dim=4) / coef_den_m               # [B,N,C,L+1,R]
    P_true= ((target**2) * W).sum(dim=4) / coef_den_m             # [B,N,C,L+1,R]

    power_num_b = ((P_hat - P_true).abs() * wl).sum(dim=(1,2,3,4))     # [B]
    power_den_b = ((valid_vec * wl).sum(dim=(1,2,3,4))).clamp_min(1.0) # [B]
    power_b     = power_num_b / power_den_b                             # [B]

    # ---------- 4) 径向二阶平滑（pred；在 r 轴用 m-平均的权重衔接） ----------
    if tv2_weight > 0 and R >= 4:
        # 先得到在 m 上均值后的权重 W_r: [B,N,C,L+1,R]
        W_r = W.mean(dim=4)
        tv2_core = (pred[:,:,:,:,:,2:] - 2*pred[:,:,:,:,:,1:-1] + pred[:,:,:,:,:,0:-2]).abs()  # [B,N,C,L+1,M,R-2]
        # 在 r 三联相邻位置的权重乘积（带 keepdim 以便和 tv2_core 对齐）
        Wm = W_r.unsqueeze(4)  # [B,N,C,L+1,1,R]
        Wm2 = (Wm[:,:,:,:,:,2:] * Wm[:,:,:,:,:,1:-1] * Wm[:,:,:,:,:,0:-2])                    # [B,N,C,L+1,1,R-2]

        # 按样本聚合
        tv2_num_b = (tv2_core * Wm2).sum(dim=(1,2,3,4,5))                                      # [B]
        tv2_den_b = (Wm2.expand_as(tv2_core)).sum(dim=(1,2,3,4,5)).clamp_min(1.0)              # [B]
        tv2_b     = tv2_num_b / tv2_den_b                                                      # [B]
    else:
        tv2_b = torch.zeros(B, device=device)

    # ---------- 组合 ----------
    total_b    = coef_b + phase_weight*phase_b + power_weight*power_b + tv2_weight*tv2_b  # [B]
    total_mean = total_b.mean()

    parts = {
        'total_per_sample': total_b.detach(),
        'coef':  coef_b.detach(),
        'phase': phase_b.detach(),
        'power': power_b.detach(),
        'tv2':   tv2_b.detach(),
    }
    return total_mean, total_b,parts

def compute_CE_perplexity(pred_logits, true_labels, mask=None, reduction='mean'):
    """
    Safe CE + PPL with dynamic class size and consistent masking.

    Args:
        pred_logits: [B, N, C]
        true_labels: [B, N]
        mask:        [B, N] bool/0-1, positions to include in CE
    Returns:
        avg_ce (scalar), perplexity (scalar)
    """
    B, N, C = pred_logits.shape
    logits  = pred_logits.reshape(-1, C)
    targets = true_labels.reshape(-1)

    # basic sanity
    if logits.numel() == 0:
        return torch.tensor(0., device=pred_logits.device), torch.tensor(1., device=pred_logits.device)
    assert targets.min() >= 0 and targets.max() < C, "label out of range for logits"

    ce_vec = F.cross_entropy(logits, targets, reduction='none')  # no smoothing

    if mask is None:
        m = torch.ones_like(targets, dtype=torch.bool)
    else:
        m = mask.reshape(-1).bool()

    m_f = m.float()
    denom = m_f.sum().clamp_min(1.0)
    if reduction == 'mean':
        avg_ce = (ce_vec * m_f).sum() / denom
    elif reduction == 'sum':
        avg_ce = (ce_vec * m_f).sum() / denom
    else:
        # fall back to mean
        avg_ce = (ce_vec * m_f).sum() / denom

    perplexity = torch.exp(avg_ce)
    return avg_ce, perplexity


@torch.no_grad()
def type_top1_acc(logits: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor | None = None, ignore_index: int | None = None):
    B, N, C = logits.shape
    pred = logits.argmax(dim=-1)   # [B,N]
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid &= (target != ignore_index)
    if node_mask is not None:
        valid &= node_mask.bool()
    correct = (pred == target) & valid
    return correct.sum().float() / valid.sum().clamp_min(1)




def backbone_mse_loss(
    gt_bb_atoms: torch.Tensor,        # [B, N, 4, 3]
    pred_all_atoms: torch.Tensor,     # [B, N, 4, 3]（或已取 backbone 的张量）
    loss_mask: torch.Tensor,          # [B, N] 或 [B, N, 1]
    bb_atom_scale: float = 1.0,
) -> torch.Tensor:
    """
    计算 backbone 坐标的 MSE（逐 batch），与原代码等价。
    返回: [B] 每个样本的 loss
    """
    # 对 mask 统一形状 [B, N, 1, 1]
    if loss_mask.dim() == 2:
        mask = loss_mask[..., None, None]
        denom_cnt = loss_mask.sum(dim=-1)  # [B]
    else:
        mask = loss_mask[..., None]
        denom_cnt = loss_mask.squeeze(-1).sum(dim=-1)

    # 缩放
    pred_bb = pred_all_atoms.clone() * bb_atom_scale
    gt_bb   = gt_bb_atoms * bb_atom_scale

    # 分母: 有效原子数 * 3 坐标维度
    loss_denom = (denom_cnt * 1.0).clamp_min(1.0)  # [B]

    # MSE 累加后按有效坐标数归一
    sq = (gt_bb - pred_bb) ** 2

    # nps=MSE.squeeze(0).detach().cpu().numpy()
    #
    # max=torch.max(MSE, )
    bb_atom_loss = (sq * mask).sum() / loss_denom.sum()  # [B]
    return bb_atom_loss


def pairwise_distance_loss(
    gt_bb_atoms: torch.Tensor,      # [B, N, A, 3], A 可为 4 或 14
    pred_bb_atoms: torch.Tensor,    # [B, N, A, 3]
    loss_mask: torch.Tensor,        # [B, N] 或 [B, N, 1]
    use_huber: bool = False,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """
    计算任意 A (backbone 或 atom14) 的 pairwise 距离矩阵误差（逐 batch）。
    返回: [B] 每个样本的 loss
    """
    if gt_bb_atoms.shape[-1] != 3 or pred_bb_atoms.shape[-1] != 3:
        raise ValueError("Expected last dim == 3 for coordinates.")

    B = gt_bb_atoms.shape[0]
    N = gt_bb_atoms.shape[1]
    A = int(np.prod(gt_bb_atoms.shape[2:-1]))  # 允许多层 shape

    gt_flat = gt_bb_atoms.reshape(B, N * A, 3)
    pred_flat = pred_bb_atoms.reshape(B, N * A, 3)

    if loss_mask.dim() == 2:
        mask = loss_mask[..., None].expand(-1, -1, A)
    elif loss_mask.dim() == 3 and loss_mask.shape[-1] == 1:
        mask = loss_mask.expand(-1, -1, A)
    else:
        mask = loss_mask
    mask = mask.to(gt_bb_atoms.dtype)
    flat_mask = mask.reshape(B, N * A)

    gt_d = torch.linalg.norm(gt_flat[:, :, None, :] - gt_flat[:, None, :, :], dim=-1)
    pred_d = torch.linalg.norm(pred_flat[:, :, None, :] - pred_flat[:, None, :, :], dim=-1)

    pair_mask = (flat_mask[:, :, None] * flat_mask[:, None, :])

    diff = (pred_d - gt_d) * pair_mask

    if use_huber:
        absd = diff.abs()
        quad = torch.clamp(absd, max=huber_delta)
        lin = absd - quad
        elem = 0.5 * quad ** 2 + huber_delta * lin
    else:
        elem = diff ** 2

    denom = pair_mask.sum(dim=(1, 2)).clamp_min(1.0)
    dist_mat_loss = elem.sum(dim=(1, 2)) / denom
    return dist_mat_loss


def torsion_angle_loss(
    a: torch.Tensor,        # [*, N, 7, 2]  预测 sin,cos
    a_gt: torch.Tensor,     # [*, N, 7, 2]  GT
    a_alt_gt: torch.Tensor, # [*, N, 7, 2]  π 周期等价的 GT
    mask: torch.Tensor = None,  # [*, N, 7]  有效角掩码 (0/1)
    an_weight: float =1,
    eps: float = 1e-6,
):
    """
    Torsion angle loss with mask and numerical stability.

    Returns:
        loss: [*] scalar tensor
    """
    # [*, N, 7]
    norm = torch.norm(a, dim=-1)

    # [*, N, 7, 2]
    a = a / norm.clamp_min(eps).unsqueeze(-1)

    # [*, N, 7]
    diff_norm_gt = torch.norm(a - a_gt, dim=-1)
    diff_norm_alt_gt = torch.norm(a - a_alt_gt, dim=-1)
    min_diff = torch.minimum(diff_norm_gt ** 2, diff_norm_alt_gt ** 2)

    # 掩码处理
    if mask is not None:
        mask = mask.to(min_diff.dtype)
        valid_count = mask.sum().clamp_min(1.0)
        l_torsion = (min_diff * mask).sum(dim=(-1, -2)) / valid_count
        l_angle_norm = (torch.abs(norm - 1) * mask).sum(dim=(-1, -2)) / valid_count
    else:
        l_torsion = min_diff.mean(dim=(-1, -2))
        l_angle_norm = torch.abs(norm - 1).mean(dim=(-1, -2))

    return l_torsion + an_weight * l_angle_norm







# 假设 fused kernel 已导入
# from iga import fused_gaussian_overlap_score

class SideAtomsIGALoss_Final(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ----------------------------------------------------
        # 1. 权重配置
        # ----------------------------------------------------
        # Legacy Coordinate Weights
        self.w_atom_mse = getattr(config, 'atom_loss_weight', 1.0)
        self.w_pair = getattr(config, 'pair_loss_weight', 1.0)  # 如果 config 里有的话
        self.w_huber = getattr(config, 'huber_loss_weight', 1.0)

        # New IGA Weights
        self.w_param = getattr(config, 'w_param', 5.0)  # 高斯参数辅助
        self.w_nll = getattr(config, 'w_nll', 3e-4)  # 概率分布正则
        self.w_seq = getattr(config, 'type_loss_weight', 1.0)

        # Gaussian Parameters (must be set in config)
        self.base_thickness = config.base_thickness  # Angstrom (用于现场计算 GT Gaussian)

    def _calc_gt_gaussian(self, atoms, mask):
        """(备用) 现场计算 GT 高斯参数"""
        mask_exp = mask.unsqueeze(-1)
        count = mask_exp.sum(dim=-2).clamp(min=1.0)

        # Offset
        local_mean = (atoms * mask_exp).sum(dim=-2) / count
        is_valid = (mask.sum(dim=-1) > 0.5)
        local_mean[~is_valid] = 0.0

        # Scaling
        centered = (atoms - local_mean.unsqueeze(-2)) * mask_exp
        variance = (centered ** 2).sum(dim=-2) / count
        std_dev = torch.sqrt(variance + 1e-10)

        scaling_log = torch.log(std_dev + self.base_thickness)
        return local_mean, scaling_log

    def forward(self, outs, batch, noisy_batch):
        """
        outs: {
            'pred_atoms': [B, N, 11, 3],       # 预测的侧链局部坐标
            'final_gaussian': OffsetGaussianRigid,
            'logits': [B, N, 20]
        }
        """
        # ==================================================================
        # Part A: Legacy Coordinate Loss (你熟悉的原子级硬约束)
        # ==================================================================
        pred_side_local = outs['pred_atoms']  # [B, N, 11, 3]
        logits = outs.get('logits', None)
        training_cfg = self.config  # 兼容你的写法

        # Masking
        loss_mask = noisy_batch['update_mask'] * noisy_batch['res_mask']

        # 组装 14 原子 (Backbone GT + Pred Sidechain)
        gt_atoms14_local = batch['atoms14_local']
        backbone_gt = gt_atoms14_local[..., :4, :]
        atoms14_pred_local = torch.cat([backbone_gt, pred_side_local], dim=-2)

        # 备份 Raw 用于统计
        atoms14_pred_local_raw = atoms14_pred_local.clone()
        atoms14_gt_local_raw = gt_atoms14_local.clone()

        # Scale 处理 (OpenFold 习惯)
        # -----------------------------
        # 0. 取基础量 & 构造各种 Mask
        # -----------------------------
        update_mask = noisy_batch['update_mask'].float()  # [B,N]
        res_mask = noisy_batch['res_mask'].float()  # [B,N]
        exists_full = batch['atom14_gt_exists'].float()  # [B,N,14]
        sc_exists = exists_full[..., 4:14]  # [B,N,11]

        # 残基级：只在「有效 + 被 mask」的残基上监督（Gaussian 参数、序列）
        res_loss_mask = (update_mask * res_mask)  # [B,N]

        # 原子级：只在「有效 + 被 mask 的侧链原子」上监督（坐标、NLL）
        # 如果 noisy_batch 提供了 sidechain_atom_mask，就用它；否则 fallback
        if 'sidechain_atom_mask' in noisy_batch:
            sidechain_atom_mask = noisy_batch['sidechain_atom_mask'].float()  # [B,N,11]
            side_loss_mask = sidechain_atom_mask * res_mask.unsqueeze(-1)  # 再乘上 res_mask
            side_loss_mask=side_loss_mask*sc_exists
        else:
            side_loss_mask = sc_exists * res_loss_mask.unsqueeze(-1)  # [B,N,11]

        bb_atom_scale = getattr(training_cfg, 'bb_atom_scale', 1.0)
        atoms14_pred_local = atoms14_pred_local * bb_atom_scale * exists_full[...,None]
        atoms14_gt_local = gt_atoms14_local * bb_atom_scale * exists_full[...,None]

        # --- 1. Local MSE ---
        # 注意: 这里的 backbone_mse_loss 通常不仅算 backbone，而是算传入的所有原子
        # 我们传入的是 14 原子，所以它计算的是全原子 MSE
        local_mse_loss = backbone_mse_loss(
            atoms14_gt_local[..., 4:, :],
            atoms14_pred_local[..., 4:, :],
            side_loss_mask,
            bb_atom_scale=getattr(training_cfg, 'bb_atom_loss_weight', 1.0),  # 这里的权重可能只影响 BB
        ).mean()

        # --- 2. Pairwise Distance ---
        local_pair_loss = pairwise_distance_loss(
            atoms14_gt_local[..., 4:, :],
            atoms14_pred_local[..., 4:, :],
            side_loss_mask,
            use_huber=False,
        ).mean()

        # --- 3. Huber Loss ---
        local_huber_loss = huber(
            atoms14_pred_local[..., 4:, :],
            atoms14_gt_local[..., 4:, :],
            side_loss_mask,
        )

        # Coordinate Total
        coord_loss = local_mse_loss + self.w_pair * local_pair_loss + self.w_huber * local_huber_loss

        # --- 4. Per-Atom Metrics (监控用) ---
        # --- Per-Atom MSE（监控用，只看侧链原子，且只统计被监督的那部分） ---
        per_atom_side_mse = {}
        gt_side_local = gt_atoms14_local[..., 4:14, :]  # [B,N,11,3]
        atom_sq_error = ((pred_side_local - gt_side_local) ** 2).sum(dim=-1)  # [B,N,11]
        for i in range(4, 14):
            mask_atom = side_loss_mask[..., i-4]  # [B,N]
            denom = mask_atom.sum().clamp(min=1.0)
            mse_atom = (atom_sq_error[..., i-4] * mask_atom).sum() / denom
            per_atom_side_mse[f'atom{i:02d}_mse'] = mse_atom.detach()

        # ==================================================================
        # Part B: Gaussian & Sequence Loss (IGA 独有)
        # ==================================================================

        # --- 5. Gaussian Parameter MSE ---
        # 监督 Offset 和 Scaling，加速 Trunk 学习体积感
        pred_gaussian = outs['final_gaussian']
        pred_offset = pred_gaussian._local_mean  # [B,N,3]
        pred_scale_log = pred_gaussian._scaling_log  # [B,N,3]

        if 'local_mean_1' in batch and 'scaing_log_1' in batch:
            gt_offset = batch['local_mean_1']  # [B,N,3]
            gt_scale_log = batch['scaing_log_1']  # [B,N,3]
        else:
            # 如果没有预存 GT，高斯参数，可以按需打开这段
            raise NotImplementedError("需要 batch['local_mean_1'] 和 batch['scaing_log_1'].")

        # 只在「有侧链 + 被 mask + 有效残基」上监督
        is_valid_res = (sc_exists.sum(-1) > 0.5).float()  # [B,N]
        loss_mask_param = res_loss_mask * is_valid_res  # [B,N]
        denom_param = loss_mask_param.sum().clamp(min=1.0)

        l_off = F.mse_loss(pred_offset, gt_offset, reduction='none').sum(-1)  # [B,N]
        l_scl = F.mse_loss(pred_scale_log, gt_scale_log, reduction='none').sum(-1)  # [B,N]
        loss_param = ((l_off + l_scl) * loss_mask_param).sum() / denom_param

        # --- 6. NLL Loss (形状似然度) ---
        # 检查 GT 原子是否落在预测的高斯分布内
        sigma_pred = pred_gaussian.get_covariance()  # Global
        mu_pred = pred_gaussian.get_gaussian_mean()  # Global

        # GT: 直接用全局坐标（避免重复刚体变换）
        gt_atoms_global =batch['atom14_gt_positions'][..., 4:14, :]  # [B, N, 11, 3]

        # # GT 转 Global (全原子)
        # gt_atoms_global = batch['atom14_gt_positions']  # [B, N, 14, 3] 直接用全原子

        # Mahalanobis
        delta = gt_atoms_global - mu_pred.unsqueeze(-2)
        sigma_exp = sigma_pred.unsqueeze(-3).expand(*delta.shape[:-1], 3, 3)

        mahal_sq, log_det = compute_robust_nll_components(delta, sigma_exp)

        nll_per_atom = (mahal_sq + log_det).mul_(0.5)

        # Mask: Masked Residue & Existing Atom
        loss_mask_nll = sc_exists * loss_mask.unsqueeze(-1)
        loss_nll = (nll_per_atom * loss_mask_nll).sum() / (loss_mask_nll.sum() + 1e-6)

        # --- 7. Sequence Loss ---
        loss_seq = 0.0
        acc = 0.0
        perplexity = 0.0
        if logits is not None:
            typeloss, perplexity = compute_CE_perplexity(
                logits, batch['aatype'], mask=loss_mask
            )
            acc = type_top1_acc(logits, batch['aatype'], node_mask=loss_mask)
            loss_seq = typeloss

        # ==================================================================
        # Total Loss Integration
        # ==================================================================
        total_loss = (
                self.w_atom_mse * coord_loss +
                self.w_param * loss_param +
                self.w_nll * loss_nll +
                self.w_seq * loss_seq
        )

        metrics = {
            'loss': total_loss,
            # Legacy metrics
            'coord_loss': coord_loss.detach(),
            'coord_mse': local_mse_loss.detach(),
            'coord_pair': local_pair_loss.detach(),
            'coord_huber': local_huber_loss.detach(),
            # New IGA metrics
            'gauss_param_mse': loss_param.detach(),
            'gauss_nll': loss_nll.detach(),
            'seq_loss': loss_seq.detach() if isinstance(loss_seq, torch.Tensor) else 0.0,
            'aa_acc': acc.detach() if isinstance(acc, torch.Tensor) else 0.0,
            'perplexity': perplexity.detach() if isinstance(perplexity, torch.Tensor) else 0.0
        }
        # Add per-atom MSEs
        metrics.update(per_atom_side_mse)

        return metrics



class BackboneGaussianAutoEncoderLoss(nn.Module):
    """
    用于你说的 Pre-check / Auto-encoding：
    输入真实结构 -> (down/up/finalup + refine) -> 重建同一个 backbone

    监督：
      1) backbone 原子重建（global MSE + intra-residue pair）
      2) 可选：高斯 NLL（让 GT 原子落在预测高斯里）
      3) 加上 reg_total（FinalUp/Up/Down 的所有正则）
    """
    def __init__(
        self,
        w_mse: float = 1.0,
            w_ca_trans: float = 1.0,
        w_pair_intra: float = 0.5,  # [原有] 局部刚体几何 (Residue内部)
        w_pair_global: float = 1.0,  # [新增] 全局拓扑 (DRMSD, 所有 CA-CA 距离)
        w_gauss_nll: float = 0.0,   # 建议先 0，等坐标能降再开
        w_mu_anchor: float = 0.0,   # debug 用：mu_pred 贴近 CA_gt（很小即可，如 1e-3）
        w_reg: float = 10.0,
            eps: float = 1e-8
    ):
        super().__init__()
        self.w_mse = w_mse
        self.w_ca_trans=w_ca_trans
        self.w_pair_intra = w_pair_intra
        self.w_pair_global = w_pair_global  # 新增权重
        self.w_gauss_nll = w_gauss_nll
        self.w_mu_anchor = w_mu_anchor
        self.w_reg = w_reg
        self.eps = eps

    def forward(self, outs, batch, noisy_batch):
        pred = outs["pred_atoms_global"]  # [B,N,4,3] Å
        gt = batch["atom14_gt_positions"][..., :4, :]  # [B,N,4,3] N/CA/C/O

        res_mask = noisy_batch["res_mask"].float()              # [B,N]
        update_mask = noisy_batch.get("update_mask", res_mask).float()
        m_res = res_mask * update_mask                          # [B,N]
        denom = m_res.sum().clamp_min(1.0)

        # ---------- (1) Backbone global MSE ----------
        mse_per_atom = ((pred - gt) ** 2).sum(dim=-1)           # [B,N,4]
        bb_mse = (mse_per_atom * m_res[..., None]).sum() / (denom * 4.0)

        # ---------- (2) Intra-residue pair distance loss ----------
        # ---------------------------------------------------------
        # (2) Intra-residue Pair (局部刚体保形)
        # ---------------------------------------------------------
        # 0:N, 1:CA, 2:C, 3:O
        idx_a = torch.tensor([0, 0, 0, 1, 1, 2], device=pred.device)
        idx_b = torch.tensor([1, 2, 3, 2, 3, 3], device=pred.device)

        vec_pred_intra = pred[..., idx_a, :] - pred[..., idx_b, :]
        vec_gt_intra = gt[..., idx_a, :] - gt[..., idx_b, :]

        d_pred_intra = torch.sqrt((vec_pred_intra ** 2).sum(dim=-1) + self.eps)
        d_gt_intra = torch.sqrt((vec_gt_intra ** 2).sum(dim=-1) + self.eps)

        bb_pair_intra = ((d_pred_intra - d_gt_intra) ** 2 * m_res.unsqueeze(-1)).sum() / (denom * 6.0)

        # ---------------------------------------------------------
        # (3) [NEW] Global Pairwise Distance (全局拓扑 / DRMSD)
        # ---------------------------------------------------------
        # 我们使用 CA 原子来代表全局拓扑
        # pred_ca: [B, N, 3]
        pred_ca = pred[..., 1, :]
        gt_ca = gt[..., 1, :]

        # 计算所有 pairwise 距离矩阵: [B, N, N]
        # (x_i - x_j)^2 = x_i^2 + x_j^2 - 2*x_i*x_j
        # 或者直接利用 broadcasting (N, 1, 3) - (1, N, 3)

        # Pred Distance Matrix
        diff_pred = pred_ca.unsqueeze(2) - pred_ca.unsqueeze(1)  # [B, N, N, 3]
        dist_pred = torch.sqrt((diff_pred ** 2).sum(dim=-1) + self.eps)  # [B, N, N]

        # GT Distance Matrix
        diff_gt = gt_ca.unsqueeze(2) - gt_ca.unsqueeze(1)
        dist_gt = torch.sqrt((diff_gt ** 2).sum(dim=-1) + self.eps)  # [B, N, N]

        # Global Pair Mask: [B, N, N]
        # 只有当 i 和 j 都是 valid residue 时才计算 loss
        mask_2d = m_res.unsqueeze(1) * m_res.unsqueeze(2)
        denom_2d = mask_2d.sum().clamp(min=1.0)

        # DRMSD Loss (或者叫 Distance Map Loss)
        # 也可以做截断 (clamp max distance)，例如只关注 20A 以内的，但全剧通常更好
        error_map = (dist_pred - dist_gt) ** 2
        bb_pair_global = (error_map * mask_2d).sum() / denom_2d

        # ---------- (3) Optional: Gaussian NLL over backbone atoms ----------
        # 让 GT backbone 原子落在 final_gaussian 的椭球内
        gauss_nll = torch.tensor(0.0, device=pred.device)
        mu_anchor = torch.tensor(0.0, device=pred.device)

        if self.w_gauss_nll > 0.0 or self.w_mu_anchor > 0.0:
            assert "final_gaussian" in outs, "outs 需要包含 final_gaussian (r_res)"
            r = outs["final_gaussian"]
            mu_pred = r.get_gaussian_mean()          # [B,N,3]
            Sigma = r.get_covariance()               # [B,N,3,3]

            # debug: mu_pred 贴 CA_gt
            if self.w_mu_anchor > 0.0:
                bb_center = gt.mean(dim=-2)  # [B,N,3] 4个backbone原子均值
                mu_anchor = (((mu_pred - bb_center) ** 2).sum(dim=-1) * m_res).sum() / denom

            # NLL (backbone 4 atoms)
            if self.w_gauss_nll > 0.0:
                # delta: [B,N,4,3]
                delta = gt - mu_pred.unsqueeze(-2)
                # Sigma expand: [B,N,4,3,3]
                Sigma_exp = Sigma.unsqueeze(-3).expand(delta.shape[0], delta.shape[1], delta.shape[2], 3, 3)

                # 你已有的 robust 组件（和你 SideAtomsLoss 里一致）
                # mahal_sq, log_det = compute_robust_nll_components(delta, Sigma_exp)
                # nll = 0.5*(mahal_sq + log_det)

                # 若你暂时不想依赖 compute_robust_nll_components，可以先用简化版：
                # (不推荐长期用，但用于 debug ok)
                eps = 1e-6
                I = torch.eye(3, device=Sigma.device, dtype=Sigma.dtype)
                Sigma_exp = Sigma_exp + eps * I
                L = torch.linalg.cholesky(Sigma_exp)                 # [B,N,4,3,3]
                # solve y = L^{-1} delta
                y = torch.linalg.solve_triangular(L, delta.unsqueeze(-1), upper=False).squeeze(-1)  # [B,N,4,3]
                mahal_sq = (y * y).sum(dim=-1)                        # [B,N,4]
                # logdet(Sigma) = 2*sum(log(diag(L)))
                log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-12)).sum(dim=-1)  # [B,N,4]
                nll = 0.5 * (mahal_sq + log_det)

                gauss_nll = (nll * m_res[..., None]).sum() / (denom * 4.0)

        # ---------- (4) reg_total ----------
        reg = outs.get("reg_total", torch.tensor(0.0, device=pred.device))
        # if not torch.is_tensor(reg):
        #     reg = torch.tensor(float(reg), device=pred.device)

        # ---------------------------------------------------------
        # (NEW) CA Translation Loss (对齐 final_gaussian 的 trans)
        # ---------------------------------------------------------
        bb_ca_trans = torch.tensor(0.0, device=pred.device)

        if self.w_ca_trans > 0.0:
            assert "final_gaussian" in outs, "outs 需要包含 final_gaussian (r_res)"
            r = outs["final_gaussian"]

            # 1) pred trans: [B,N,3]  (按你类的 API，优先 get_trans)
            if hasattr(r, "get_trans"):
                pred_trans = r.get_trans()
            elif hasattr(r, "trans"):
                pred_trans = r.trans
            elif hasattr(r, "_trans"):
                pred_trans = r._trans
            else:
                raise AttributeError("final_gaussian 没有 get_trans()/trans/_trans, 请确认 OffsetGaussianRigid 的接口")

            # 2) GT CA: 你已经有了
            # gt_ca = gt[..., 1, :]   # [B,N,3]

            # 3) masked MSE (单位保持一致：你 pred/gt 都是 Å 才行)
            err = (pred_trans - gt_ca) * m_res.unsqueeze(-1)  # [B,N,3]
            bb_ca_trans = (err.square().sum(dim=-1).sum()) / (denom * 3.0)

        loss = (
            # self.w_mse * bb_mse +
            self.w_pair_intra * bb_pair_intra +
            self.w_pair_global * bb_pair_global +  # 加入全局项
            # self.w_gauss_nll * gauss_nll +
            # self.w_mu_anchor * mu_anchor +
            self.w_ca_trans *bb_ca_trans+
            self.w_reg * reg
        )

        return {
            "loss": loss,
            "bb_mse": bb_mse.detach(),
            "ca_mse": bb_ca_trans.detach(),
            "bb_pair_intra": bb_pair_intra.detach(),
            "bb_pair_global": bb_pair_global.detach(), # 监控这个指标
            "gauss_nll": gauss_nll.detach(),
            "mu_anchor": mu_anchor.detach(),
            "reg_total": reg.detach(),
        }


class HierarchicalGaussianLoss(nn.Module):
    def __init__(self, w_sep=1.0, w_compact=0.1, eps=1e-6):
        super().__init__()
        self.w_sep = w_sep
        self.w_compact = w_compact
        self.eps = eps

    def forward(self, mu_c, Sigma_c, mask_c):
        """
        输入是某一层的 Coarse 高斯参数。
        mu_c: [B, K, 3]
        Sigma_c: [B, K, 3, 3]
        mask_c: [B, K]
        """
        B, K, _ = mu_c.shape
        device = mu_c.device

        # ----------------------------------------------------
        # 1. Compactness Loss (父椭圆体积约束)
        # ----------------------------------------------------
        # 最小化 Trace 等价于最小化整体散布
        trace = torch.diagonal(Sigma_c, dim1=-2, dim2=-1).sum(dim=-1)  # [B, K]

        denom = mask_c.sum().clamp(min=1.0)
        loss_compact = (trace * mask_c).sum() / denom

        # ----------------------------------------------------
        # 2. Separation Loss (升级版：基于全协方差的排斥)
        # ----------------------------------------------------
        if K <= 1:
            loss_sep = torch.tensor(0.0, device=device)
        else:
            # (A) 准备两两差分向量 delta: [B, K, K, 3]
            delta = mu_c.unsqueeze(2) - mu_c.unsqueeze(1)

            # (B) 准备两两协方差之和 Sigma_sum: [B, K, K, 3, 3]
            # 两个高斯卷积后的方差是各自方差之和
            Sigma_sum = Sigma_c.unsqueeze(2) + Sigma_c.unsqueeze(1)

            # 加上一点点 eps 保证正定 (Cholesky 不崩)
            eye = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            Sigma_sum = Sigma_sum + eye * self.eps

            # (C) 计算高斯重叠分数 (Log Space)
            # 输出范围 (-inf, 0]，越接近 0 表示越重叠
            log_overlap = fused_gaussian_overlap_score(delta, Sigma_sum)

            # (D) 转换到 [0, 1] 概率空间
            overlap = torch.exp(log_overlap)

            # (E) Mask 处理
            # 1. Mask 掉对角线 (自己跟自己重叠是必然的，不惩罚)
            identity = torch.eye(K, device=device).unsqueeze(0)  # [1, K, K]
            # 2. Mask 掉无效的 Coarse Token (Padding)
            mask_2d = mask_c.unsqueeze(1) * mask_c.unsqueeze(2)  # [B, K, K]

            # 只有 (非对角线) AND (有效Token) 才计算 Loss
            valid_overlap = overlap * mask_2d * (1.0 - identity)

            # (F) 求和平均
            # 分母是有效配对的数量 (N*(N-1))

            loss_sep = valid_overlap.sum() /(denom* (denom + 1e-6))

        return {
            "loss_compact": loss_compact,
            "loss_sep": loss_sep,
            "total_hier": self.w_compact * loss_compact + self.w_sep * loss_sep
        }





class SymmetricGaussianLoss(nn.Module):
    def __init__(self, w_center_p=1.0, w_center_c=1.0, w_shape=0.0, eps=1e-6):
        super().__init__()
        self.w_center_p = w_center_p
        self.w_center_c = w_center_c
        self.w_shape = w_shape
        self.eps = eps

    def forward(self, mu_child, Sigma_child, mu_parent, Sigma_parent, A, mask=None):
        """
        全流程无显式求逆的双向高斯 Loss
        """
        B, N, _ = mu_child.shape
        _, K, _ = mu_parent.shape
        device = mu_child.device

        if mask is None:
            mask = torch.ones((B, N), device=device)

        # -----------------------------------------------------------
        # 1. 准备 delta (N*K 对)
        # -----------------------------------------------------------
        # [B, N, K, 3]
        delta = mu_child.unsqueeze(2) - mu_parent.unsqueeze(1)

        # -----------------------------------------------------------
        # 2. 计算双向马氏距离 (直接复用你的 Cholesky 代码)
        # -----------------------------------------------------------

        # (A) Parent View: dist_p
        # Sigma_parent: [B, K, 3, 3] -> 广播成 [B, 1, K, 3, 3] 以匹配 delta
        S_p_exp = Sigma_parent.unsqueeze(1) + torch.eye(3, device=device) * self.eps
        # delta: [B, N, K, 3]
        # 你的函数会自动把 S_p_exp 广播到 [B, N, K, 3, 3]
        dist_p =-2* fused_gaussian_overlap_score(delta, S_p_exp)  # [B, N, K]

        # (B) Child View: dist_c
        # Sigma_child: [B, N, 3, 3] -> 广播成 [B, N, 1, 3, 3]
        S_c_exp = Sigma_child.unsqueeze(2) + torch.eye(3, device=device) * self.eps
        dist_c = -2*fused_gaussian_overlap_score(delta, S_c_exp)  # [B, N, K]
        # dist_c_term = torch.log1p(dist_c)

        # # -----------------------------------------------------------
        # # 3. 形状约束 (Trace Term)
        # # Trace(Sigma_p^-1 * Sigma_c)
        # # -----------------------------------------------------------
        # # 这里需要 Sigma_p 的完整逆矩阵，我们用解析法算
        # # Inv_p: [B, K, 3, 3]
        # Inv_p = analytical_inverse_3x3(Sigma_parent + torch.eye(3, device=device) * self.eps)
        #
        # # 广播: [B, 1, K, 3, 3]
        # Inv_p_exp = Inv_p.unsqueeze(1)
        #
        # # Sigma_c: [B, N, 1, 3, 3]
        # S_c_exp = Sigma_child.unsqueeze(2)
        #
        # # 矩阵乘法: Inv_p @ S_c
        # # [B, 1, K, 3, 3] @ [B, N, 1, 3, 3] -> [B, N, K, 3, 3]
        # Prod = torch.matmul(Inv_p_exp, S_c_exp)
        #
        # # Trace
        # term_shape = torch.diagonal(Prod, dim1=-2, dim2=-1).sum(dim=-1)  # [B, N, K]

        # -----------------------------------------------------------
        # 4. 加权求和
        # -----------------------------------------------------------
        loss_matrix = (
                self.w_center_p * dist_p +
                self.w_center_c * dist_c# +
                # self.w_shape * term_shape
        )

        # 用分配矩阵 A 加权
        weighted_loss = loss_matrix * A
        denom = mask.sum().clamp(min=1.0)
        final_loss = (weighted_loss.sum(dim=-1) * mask).sum() / denom

        return {
            "loss": final_loss,
            "dist_p": (dist_p * A * mask.unsqueeze(-1)).sum() / denom,
            "dist_c": (dist_c * A * mask.unsqueeze(-1)).sum() / denom
        }

if __name__ == '__main__':
    # 示例
    # logits: [B,N,20], target: [B,N]
    loss = type_ce_loss(logits, target, node_mask=None, label_smoothing=0.05)
    acc  = type_top1_acc(logits, target)
