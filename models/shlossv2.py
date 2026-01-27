import torch
import torch.nn.functional as F

# 1) Huber，比 Charbonnier 在 0 残差时基线为 0，更贴近“GT==pred → 0”
def huber(x, delta=1.0):
    ax = x.abs()
    return torch.where(ax <= delta, 0.5 * ax * ax, delta * (ax - 0.5 * delta))

# 2) 按阶权重（L=8 默认）
def make_w_l(L_max=8):
    w = torch.tensor([1.0/(2*l+1) for l in range(L_max+1)], dtype=torch.float32)
    return w * (len(w) / w.sum())  # 均值≈1，数值更稳

# 3) 仅在“有值位置”（predict_mask=1）统计 std_lr；不累计
@torch.no_grad()
def std_lr_from_pos_mask(target, predict_mask, valid_mask=None, floor=1e-3):
    """
    target:       [B,N,C,L+1,2L+1,R]  干净GT
    predict_mask: 可广播到 target 形状；1=有值位置（要统计/要计算loss）
    valid_mask:   [B,N] 或 None
    返回: std_lr [L+1,R]
    """
    device = target.device
    M = predict_mask.to(device).to(target.dtype)
    if M.shape != target.shape:
        M = torch.ones_like(target) * M
    if valid_mask is not None:
        V = valid_mask.to(device).float().view(target.shape[0], target.shape[1], 1, 1, 1, 1)
        M = M * V

    x   = target * M
    cnt = M.sum(dim=(0,1,2,4)).clamp_min(1.0)                                # [L+1,R]
    mean= x.sum(dim=(0,1,2,4)) / cnt                                         # [L+1,R]
    var = ((x - mean[None,None,None,:,None,:])**2 * M).sum(dim=(0,1,2,4)) / cnt
    std = torch.sqrt(var).clamp_min(floor)                                    # [L+1,R]
    return std

# 4) 主损失：只在 predict_mask 指示的“有值位置”上计算；保留 B 维
def sh_loss_with_masks(
    pred, target,
    std_lr, w_l,
    predict_mask=None,   # 1=该位置“有值/应计入损失”，可广播到 [B,N,C,L+1,2L+1,R]
    valid_mask=None,     # [B,N]（1=有效残基）
    update_mask=None,    # [B,N]（1=允许更新）
    power_weight=0.2, phase_weight=0.2, tv2_weight=0.05,
):
    """
    pred/target: [B,N,C,L+1,2L+1,R]
    返回: total_mean(标量), parts(dict: 各分量为 [B] per-sample)
    """
    B,N,C,Lp1,M,R = pred.shape
    device = pred.device

    # 按阶权重 / std
    wl  = w_l.to(device).view(1,1,1,Lp1,1)                       # [1,1,1,L+1,1]
    std = None if std_lr is None else std_lr.to(device).view(1,1,1,Lp1,1,R)

    # -------- 组合权重 W：只在“有值位置 & 有效残基 & 允许更新”计算 --------
    W = torch.ones_like(pred, dtype=pred.dtype, device=device)
    if predict_mask is not None:
        W = W * predict_mask.to(device).to(pred.dtype)
    if valid_mask is not None:
        V = valid_mask.to(device).float().view(B,N,1,1,1,1)
        W = W * V
    if update_mask is not None:
        U = update_mask.to(device).float().view(B,N,1,1,1,1)
        W = W * U

    if (W.sum() == 0):
        zero = pred.new_tensor(0.0)
        return zero, {'total_per_sample': torch.zeros(B, device=device),
                      'coef': torch.zeros(B, device=device),
                      'phase': torch.zeros(B, device=device),
                      'power': torch.zeros(B, device=device),
                      'tv2': torch.zeros(B, device=device)}

    # -------- 1) 系数项（Huber，m 维 masked mean；只在 W=1 的位置） --------
    resid = pred - target if std is None else (pred - target) / std
    coef_num_m = (torch.abs(resid) * W).sum(dim=4)                   # [B,N,C,L+1,R]
    coef_den_m = W.sum(dim=4).clamp_min(1.0)                     # [B,N,C,L+1,R]
    coef_per   = coef_num_m / coef_den_m                         # [B,N,C,L+1,R]
    valid_m    = (W.sum(dim=4) > 0).float()                      # [B,N,C,L+1,R]

    coef_num_b = (coef_per * wl).sum(dim=(1,2,3,4))              # [B]
    coef_den_b = ((valid_m * wl).sum(dim=(1,2,3,4))).clamp_min(1.0)
    coef_b     = coef_num_b / coef_den_b

    # -------- 2) phase：只对 W=1 的位置做 m-向量余弦 --------
    a, b = pred, target
    dot   = (a*b*W).sum(dim=4)                                   # [B,N,C,L+1,R]
    a_nrm = torch.sqrt((a*a*W).sum(dim=4).clamp_min(1e-12))
    b_nrm = torch.sqrt((b*b*W).sum(dim=4).clamp_min(1e-12))
    # 对于 W=0 的位置，上面范数也为 0，不会参与分子分母
    cos_full = 1.0 - (dot / (a_nrm * b_nrm + 1e-12))
    # 只在有有效 m 的地方计入
    phase_valid = (W.sum(dim=4) > 0).float()
    phase_num_b = (cos_full * phase_valid * wl).sum(dim=(1,2,3,4))
    phase_den_b = ((phase_valid * wl).sum(dim=(1,2,3,4))).clamp_min(1.0)
    phase_b     = phase_num_b / phase_den_b

    # -------- 3) power：只对 W=1 的 m 做均值；pred==gt 时≈0 --------
    P_hat = ((pred**2) * W).sum(dim=4) / coef_den_m              # [B,N,C,L+1,R]
    P_true= ((target**2) * W).sum(dim=4) / coef_den_m
    power_num_b = ((P_hat - P_true).abs() * wl).sum(dim=(1,2,3,4))
    power_den_b = ((valid_m * wl).sum(dim=(1,2,3,4))).clamp_min(1.0)
    power_b     = power_num_b / power_den_b

    # -------- 4) tv2：做在“残差”上；r 轴权重来自 W 在 m 上的均值 --------
    if tv2_weight > 0 and R >= 3:
        res_full = pred - target
        tv2_core = (res_full[:,:,:,:,:,2:] - 2*res_full[:,:,:,:,:,1:-1] + res_full[:,:,:,:,:,0:-2]).abs()
        W_r = W.mean(dim=4).unsqueeze(4)                          # [B,N,C,L+1,1,R]
        Wm2 = (W_r[:,:,:,:,:,2:] * W_r[:,:,:,:,:,1:-1] * W_r[:,:,:,:,:,0:-2])  # [B,N,C,L+1,1,R-2]
        tv2_num_b = (tv2_core * Wm2).sum(dim=(1,2,3,4,5))
        tv2_den_b = (Wm2.expand_as(tv2_core)).sum(dim=(1,2,3,4,5)).clamp_min(1.0)
        tv2_b     = tv2_num_b / tv2_den_b
    else:
        tv2_b = torch.zeros(B, device=device)

    # -------- 聚合 --------
    total_b    = coef_b + phase_weight*phase_b + power_weight*power_b + tv2_weight*tv2_b  # [B]
    total_mean = total_b.mean()

    parts = {
        'total_per_sample': total_b.detach(),
        'coef':  coef_b.detach(),
        'phase': phase_b.detach(),
        'power': power_b.detach(),
        'tv2':   tv2_b.detach(),
    }
    return total_mean,total_b, parts


def sh_mse_loss(pred, target, predict_mask=None, valid_mask=None, update_mask=None):
    """
    pred/target: [B,N,C,L+1,2L+1,R]
    predict_mask: [B,N,C,L+1,2L+1,R] 或广播；1=有值
    valid_mask:   [B,N] 或 None
    update_mask:  [B,N] 或 None
    """
    B, N = pred.shape[:2]
    device = pred.device

    # --- 组合掩码 W ---
    W = torch.ones_like(pred, dtype=pred.dtype, device=device)
    if predict_mask is not None:
        W = W * predict_mask.to(device).to(pred.dtype)
    if valid_mask is not None:
        V = valid_mask.to(device).float().view(B,N,1,1,1,1)
        W = W * V
    if update_mask is not None:
        U = update_mask.to(device).float().view(B,N,1,1,1,1)
        W = W * U

    # --- 残差平方 ---
    mse_num = ((pred - target).abs() * W).sum(dim=(1,2,3,4,5))  # [B]
    mse_den = W.sum(dim=(1,2,3,4,5)).clamp_min(1.0)          # [B]
    mse_b   = mse_num / mse_den                              # [B]

    return mse_b.mean(), mse_b  # 标量均值 + per-B 向量

def sh_loss_high_contrast(
    pred, target,
    w_l,
    predict_mask=None,     # [B,N,C,L+1,2L+1,R] 或可广播；1=有值
    valid_mask=None,       # [B,N]
    update_mask=None,      # [B,N]
    # 重要：把Zero->GT拉高的杠杆
    tau=0.02,              # 正样本阈值（基于 m-均值后的 |GT|）
    lambda_pos=1.0,        # 正样本权重
    lambda_neg=0.2,        # 负样本权重（可设更小）
    coef_weight=1.0,
    phase_weight=0.5,      # 相位权重 ↑（比之前0.2大）
    power_weight=0.3,      # 能量权重 ↑（比之前0.2大）
    tv2_weight=0.0,        # 诊断期先关掉（=0），需要正则再开
    huber_delta=1.0,
):
    """
    返回: total_mean(标量), parts dict 各项为 [B]
    """
    B,N,C,Lp1,M,R = pred.shape
    device = pred.device
    wl = w_l.to(device).view(1,1,1,Lp1,1)

    # --- 组合权重：只在“你标的有值位置”上算 ---
    W = torch.ones_like(pred, dtype=pred.dtype, device=device)
    if predict_mask is not None:
        W = W * predict_mask.to(device).to(pred.dtype)
    if valid_mask is not None:
        W = W * valid_mask.to(device).float().view(B,N,1,1,1,1)
    if update_mask is not None:
        W = W * update_mask.to(device).float().view(B,N,1,1,1,1)

    if W.sum() == 0:
        zero = pred.new_tensor(0.0)
        return zero, {'total_per_sample': torch.zeros(B, device=device),
                      'coef': torch.zeros(B, device=device),
                      'phase': torch.zeros(B, device=device),
                      'power': torch.zeros(B, device=device),
                      'tv2': torch.zeros(B, device=device)}

    # --- 基于 GT 的幅值（先在 m 上 masked-mean，再判定正负） ---
    m_cnt = W.sum(dim=4).clamp_min(1.0)                      # [B,N,C,L+1,R]
    gt_mag_m = (target.abs() * W).sum(dim=4) / m_cnt         # [B,N,C,L+1,R]
    pos = (gt_mag_m > tau).float()                           # 正样本掩码
    neg = 1.0 - pos

    # ===== 1) 系数（Huber） =====
    resid = pred - target
    coef_m = (huber(resid, delta=huber_delta) * W).sum(dim=4) / m_cnt   # [B,N,C,L+1,R]

    coef_pos_num = (coef_m * wl * pos).sum(dim=(1,2,3,4))
    coef_pos_den = (wl * pos).sum(dim=(1,2,3,4)).clamp_min(1.0)
    coef_neg_num = (coef_m * wl * neg).sum(dim=(1,2,3,4))
    coef_neg_den = (wl * neg).sum(dim=(1,2,3,4)).clamp_min(1.0)
    coef_pos_b = coef_pos_num / coef_pos_den
    coef_neg_b = coef_neg_num / coef_neg_den
    coef_b = lambda_pos * coef_pos_b + lambda_neg * coef_neg_b

    # ===== 2) 相位（m-向量余弦）：只要GT是正样本就计入 =====
    a, b = pred, target
    dot   = (a*b*W).sum(dim=4)
    a_nrm = torch.sqrt((a*a*W).sum(dim=4).clamp_min(1e-12))
    b_nrm = torch.sqrt((b*b*W).sum(dim=4).clamp_min(1e-12))
    cos_full = 1.0 - (dot / (a_nrm * b_nrm + 1e-12))                 # [B,N,C,L+1,R]

    phase_pos_num = (cos_full * wl * pos).sum(dim=(1,2,3,4))
    phase_pos_den = (wl * pos).sum(dim=(1,2,3,4)).clamp_min(1.0)
    # 负样本上可弱惩罚或不惩罚；这里给一个很小的比重
    phase_neg_num = (cos_full * wl * neg).sum(dim=(1,2,3,4))
    phase_neg_den = (wl * neg).sum(dim=(1,2,3,4)).clamp_min(1.0)
    phase_pos_b = phase_pos_num / phase_pos_den
    phase_neg_b = phase_neg_num / phase_neg_den
    phase_b = lambda_pos * phase_pos_b + 0.1 * lambda_neg * phase_neg_b

    # ===== 3) 能量谱：正样本更重 =====
    P_hat = ((pred**2) * W).sum(dim=4) / m_cnt
    P_true= ((target**2) * W).sum(dim=4) / m_cnt
    power_gap = (P_hat - P_true).abs()

    power_pos_num = (power_gap * wl * pos).sum(dim=(1,2,3,4))
    power_pos_den = (wl * pos).sum(dim=(1,2,3,4)).clamp_min(1.0)
    power_neg_num = (power_gap * wl * neg).sum(dim=(1,2,3,4))
    power_neg_den = (wl * neg).sum(dim=(1,2,3,4)).clamp_min(1.0)
    power_pos_b = power_pos_num / power_pos_den
    power_neg_b = power_neg_num / power_neg_den
    power_b = lambda_pos * power_pos_b + lambda_neg * power_neg_b

    # ===== 4) 残差 TV2（诊断期建议关=0） =====
    if tv2_weight > 0 and R >= 3:
        res = pred - target
        tv2_core = (res[:,:,:,:,:,2:] - 2*res[:,:,:,:,:,1:-1] + res[:,:,:,:,:,0:-2]).abs()
        W_r = W.mean(dim=4).unsqueeze(4)  # [B,N,C,L+1,1,R]
        Wm2 = (W_r[:,:,:,:,:,2:] * W_r[:,:,:,:,:,1:-1] * W_r[:,:,:,:,:,0:-2])
        tv2_num_b = (tv2_core * Wm2).sum(dim=(1,2,3,4,5))
        tv2_den_b = (Wm2.expand_as(tv2_core)).sum(dim=(1,2,3,4,5)).clamp_min(1.0)
        tv2_b = tv2_num_b / tv2_den_b
    else:
        tv2_b = torch.zeros(B, device=device)

    total_b = (coef_weight * coef_b) + (phase_weight * phase_b) + (power_weight * power_b) + (tv2_weight * tv2_b)
    total_mean = total_b.mean()

    parts = {
        'total_per_sample': total_b.detach(),
        'coef':  coef_b.detach(),
        'phase': phase_b.detach(),
        'power': power_b.detach(),
        'tv2':   tv2_b.detach(),
        'pos_ratio': pos.mean(dim=(1,2,3,4)).detach(),  # 每样本正样本占比，便于监控
    }
    return total_mean, total_b,parts