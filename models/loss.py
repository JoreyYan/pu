import torch
import torch.nn.functional as F
def huber(x, y, mask=None, delta=1.0, reduction='mean'):
    diff = x - y
    if mask is not None:
        diff = diff * mask[...,None]
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
    if tv2_weight > 0 and R >= 3:
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


import torch
import torch.nn.functional as F

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
    MSE=(sq** 2).sum(dim=-1)
    # nps=MSE.squeeze(0).detach().cpu().numpy()
    #
    # max=torch.max(MSE, )
    bb_atom_loss = (sq * mask).sum() / loss_denom.sum()  # [B]
    return bb_atom_loss


def pairwise_distance_loss(
    gt_bb_atoms: torch.Tensor,      # [B, N, 4, 3]
    pred_bb_atoms: torch.Tensor,    # [B, N, 4, 3]
    loss_mask: torch.Tensor,        # [B, N] 或 [B, N, 1]
    use_huber: bool = False,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """
    计算 backbone 的 pairwise 距离矩阵误差（逐 batch），与原实现等价。
    返回: [B] 每个样本的 loss
    """
    B, N, A, _ = gt_bb_atoms.shape  # A=4
    # 展平到 [B, N*A, 3]
    gt_flat  = gt_bb_atoms.reshape(B, N * A, 3)
    pred_flat= pred_bb_atoms.reshape(B, N * A, 3)

    # 展平 mask 到 [B, N*A]
    if loss_mask.dim() == 3:
        loss_mask = loss_mask.squeeze(-1)
    flat_mask = loss_mask.reshape(B, N * A)#.repeat_interleave(A, dim=1).to(gt_bb_atoms.dtype)  # [B, N*A]

    # pairwise 距离 [B, NA, NA]
    gt_d   = torch.linalg.norm(gt_flat[:, :, None, :]  - gt_flat[:, None, :, :],  dim=-1)
    pred_d = torch.linalg.norm(pred_flat[:, :, None, :] - pred_flat[:, None, :, :], dim=-1)

    # 有效对的 mask
    pair_mask = (flat_mask[:, :, None] * flat_mask[:, None, :])  # [B, NA, NA]

    # 屏蔽无效条目
    # gt_d   = gt_d   * flat_mask
    # pred_d = pred_d * flat_mask

    # 误差
    diff = (pred_d - gt_d) * pair_mask

    if use_huber:
        absd = diff.abs()
        quad = torch.clamp(absd, max=huber_delta)
        lin  = absd - quad
        elem = 0.5 * quad**2 + huber_delta * lin
    else:
        elem = diff**2

    denom = pair_mask.sum(dim=(1, 2)).clamp_min(1.0)  # [B]
    dist_mat_loss = elem.sum(dim=(1, 2)) / denom      # [B]
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


if __name__ == '__main__':
    # 示例
    # logits: [B,N,20], target: [B,N]
    loss = type_ce_loss(logits, target, node_mask=None, label_smoothing=0.05)
    acc  = type_top1_acc(logits, target)
