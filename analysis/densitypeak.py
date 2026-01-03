import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Bio.PDB
from data.GaussianRigid import OffsetGaussianRigid


# ==========================================
# 1. 基础辅助函数
# ==========================================

def _batched_topk_knn(x, y, k, mask_y=None):
    dist2 = torch.cdist(x, y, p=2) ** 2
    if mask_y is not None:
        dist2 = dist2.masked_fill((mask_y < 0.5).unsqueeze(1), 1e9)
    actual_k = min(k, y.shape[1])
    d, idx = torch.topk(dist2, k=actual_k, dim=-1, largest=False)
    return idx, d


def _nms_select(mu_cand, score, K, r2):
    B, L, _ = mu_cand.shape
    device = mu_cand.device
    sel = torch.zeros((B, K), device=device, dtype=torch.long)
    order = torch.argsort(score, dim=-1, descending=True)
    mu_sorted = mu_cand.gather(1, order.unsqueeze(-1).expand(B, L, 3))

    for b in range(B):
        picked_indices, picked_coords = [], []
        for j in range(L):
            if len(picked_indices) >= K: break
            mj = mu_sorted[b, j]
            if all(((mj - p) ** 2).sum().item() >= r2 for p in picked_coords):
                picked_indices.append(order[b, j])
                picked_coords.append(mj)
        res = torch.tensor(picked_indices, device=device)
        if len(res) < K:
            res = torch.cat([res, order[b, :K - len(res)]])
        sel[b] = res[:K]
    return sel
import torch

def weighted_quantile_sorted(x_sorted: torch.Tensor, w_sorted: torch.Tensor, q: float, eps: float = 1e-8):
    """
    x_sorted: [B,K,M] ascending
    w_sorted: [B,K,M] aligned weights
    return:   [B,K] quantile value with safe fallback
    """
    w_sorted = w_sorted.clamp_min(0.0)
    w_cum = torch.cumsum(w_sorted, dim=-1)                       # [B,K,M]
    w_tot = w_cum[..., -1:].clamp_min(eps)
    ratio = w_cum / w_tot

    hit = ratio >= q                                             # [B,K,M] bool
    idx = hit.float().argmax(dim=-1)                             # first True, or 0 if none
    idx = torch.where(hit.any(dim=-1), idx, torch.full_like(idx, x_sorted.shape[-1]-1))
    return x_sorted.gather(-1, idx.unsqueeze(-1)).squeeze(-1)


def calibrate_cov_by_mahalanobis_quantile(
    mu_i: torch.Tensor,          # [B,K,M,3]
    Sig_i: torch.Tensor,         # [B,K,M,3,3] (optional for radius term)
    w: torch.Tensor,             # [B,K,M]
    mu_c: torch.Tensor,          # [B,K,3]
    Sig_raw: torch.Tensor,       # [B,K,3,3]
    q: float = 0.95,
    add_member_radius: bool = True,
    radius_scale: float = 2.0,   # 2.0 ~ “2σ” 直觉覆盖
    jitter: float = 1e-5,
    eps: float = 1e-8,
):
    """
    核心：用 d^2 = (x-mu)^T Sig_raw^{-1} (x-mu) 的加权分位数来标定 Σ
    然后 Σ = Sig_raw * d2_q

    若 add_member_radius=True，会把 fine 自身尺度转成一个“额外半径”，
    让 coarse 更像在覆盖“椭圆”而不是覆盖点。
    """
    B, K, M, _ = mu_i.shape
    device, dtype = mu_i.device, mu_i.dtype
    I = torch.eye(3, device=device, dtype=dtype).view(1,1,3,3)

    # ---- make Sig_raw SPD ----
    Sig_base = 0.5 * (Sig_raw + Sig_raw.transpose(-1, -2))
    Sig_base = Sig_base + jitter * I                             # [B,K,3,3]

    # ---- Mahalanobis distance of fine centers to coarse center ----
    delta = mu_i - mu_c.unsqueeze(2)                              # [B,K,M,3]
    L = torch.linalg.cholesky(Sig_base)                           # [B,K,3,3]
    # solve L y = delta
    y = torch.linalg.solve_triangular(L.unsqueeze(2), delta.unsqueeze(-1), upper=False)  # [B,K,M,3,1]
    d2 = (y.squeeze(-1) ** 2).sum(dim=-1)                         # [B,K,M]

    # ---- optionally add member radius into d2 (covers ellipsoids, not just points) ----
    if add_member_radius:
        # r_i^2 ≈ radius_scale^2 * tr(Sig_i)
        tr_i = torch.diagonal(Sig_i, dim1=-2, dim2=-1).sum(-1).clamp_min(eps)            # [B,K,M]
        r2 = (radius_scale ** 2) * tr_i

        # 把 r2 映射到“d2空间”的尺度：用 Sig_base 的 trace 做一个粗略归一化
        tr_base = torch.diagonal(Sig_base, dim1=-2, dim2=-1).sum(-1).clamp_min(eps)      # [B,K]
        d2 = d2 + (r2 / tr_base.unsqueeze(-1))                                           # [B,K,M]

    # ---- weighted quantile of d2 ----
    d2_sorted, idx = torch.sort(d2, dim=-1, descending=False)      # [B,K,M]
    w_sorted = w.gather(2, idx)
    d2_q = weighted_quantile_sorted(d2_sorted, w_sorted, q=q, eps=eps).clamp_min(1.0)   # [B,K]

    # ---- calibrate covariance ----
    Sig_c = Sig_base * d2_q.view(B, K, 1, 1)

    # final sym + jitter
    Sig_c = 0.5 * (Sig_c + Sig_c.transpose(-1, -2)) + (jitter * I)
    return Sig_c


def calibrate_cov_by_logdet_target(
        Sig_raw: torch.Tensor,  # [B,K,3,3]
        logdet_target: torch.Tensor,  # [B,K]
        jitter: float = 1e-5,
        eps: float = 1e-8,
):
    """
    让 Sig_raw 的体积（logdet）匹配 logdet_target，但保持其方向（各向异性比例）
    即：Sig = Sig_raw * s, 其中 s 是各向同性缩放
    """
    B, K = Sig_raw.shape[:2]
    device, dtype = Sig_raw.device, Sig_raw.dtype
    I = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)

    Sig = 0.5 * (Sig_raw + Sig_raw.transpose(-1, -2)) + jitter * I
    # logdet(Sig)
    logdet_raw = torch.logdet(Sig).clamp_min(-50.0)  # 防极端数值

    # s^3 = exp(logdet_target - logdet_raw)  -> s = exp((...)/3)
    s = torch.exp((logdet_target - logdet_raw) / 3.0)
    Sig = Sig * s.view(B, K, 1, 1)
    Sig = 0.5 * (Sig + Sig.transpose(-1, -2)) + jitter * I
    return Sig


def quantile_coverage_refine(mu_i, Sig_i, pi_i, w, mu_c, Sig_raw, q=0.95):
    """
    mu_i, Sig_i, pi_i: [B, K, M, ...] 细粒度数据
    w: [B, K, M] 归属权重
    mu_c, Sig_raw: [B, K, ...] 原始合并结果
    q: 覆盖分位数目标 (0.9 ~ 0.95)
    """
    B, K, M = w.shape
    device = w.device

    # 1. 计算每个 fine 成员对中心 k 的“覆盖贡献半径”
    # 中心距 + 成员自身的 1-sigma 伸展
    dist_center = torch.norm(mu_i - mu_c.unsqueeze(2), dim=-1)  # [B, K, M]
    radius_i = torch.sqrt(torch.diagonal(Sig_i, dim1=-2, dim2=-1).sum(-1))  # [B, K, M]
    r_ik = dist_center + radius_i

    # 2. 快速加权分位数 (简单实现：取加权平均后的 Top-K 距离作为参考)
    # 排序距离
    r_sorted, indices = torch.sort(r_ik, dim=-1, descending=False)
    w_sorted = w.gather(2, indices)

    # 计算累积权重比例
    w_cumsum = torch.cumsum(w_sorted, dim=-1)
    w_total = w_cumsum[:, :, -1:].clamp_min(1e-8)
    w_ratio = w_cumsum / w_total

    # 找到第一个满足覆盖率 q 的索引
    # [B, K, M] -> 找到第一个比例 >= q 的位置
    mask = (w_ratio >= q).float()
    # 这里的技巧：找到第一个 1 的位置
    idx_q = torch.argmax(mask, dim=-1)  # [B, K]
    R_k = r_sorted.gather(2, idx_q.unsqueeze(-1)).squeeze(-1)  # [B, K]

    # 3. 标定协方差
    # tr(Sig_raw) 代表了原始合并的平均尺度
    tr_raw = torch.diagonal(Sig_raw, dim1=-2, dim2=-1).sum(-1).clamp_min(1e-8)

    # 缩放因子：R_k^2 是目标方差尺度
    # 我们希望新的 Sig_c 的 trace 能匹配 R_k 的平方量级
    scale = (R_k ** 2) / tr_raw

    # 为了防止某些极端情况，可以给 scale 做个小的 clamp
    Sig_c = Sig_raw * scale.view(B, K, 1, 1)

    return Sig_c

# ==========================================
# 2. 改进后的算法: 体积守恒型密度峰下采样
# ==========================================

def density_peaks_local_moments_refined(
        mu, Sigma, pi, mask, K,
        k_rho=64, h=2.5, L_factor=8, r_nms=5.0,
        k_mom=128, lambda2=0.2,
        radius_scale=2.0,  # 【关键】控制填充度，建议 1.5 - 2.5
        bone_thickness=0.5,  # 【关键】防止某些方向太薄，增加 3D 存在感
        jitter=1e-6, eps=1e-8
):
    B, N, _ = mu.shape
    device, dtype = mu.device, mu.dtype
    I = torch.eye(3, device=device, dtype=dtype)

    # --- Step 1 & 2: 选中心点 (保持不变) ---
    idx_nn, d2_nn = _batched_topk_knn(mu, mu, k=k_rho, mask_y=mask)
    pi_nn = pi.unsqueeze(1).expand(-1, N, -1).gather(2, idx_nn)
    rho = (pi_nn * torch.exp(-d2_nn / (2.0 * h ** 2))).sum(dim=-1) * mask

    L = min(N, int(L_factor * K))
    rho_top, idx_top = torch.topk(rho, k=L, dim=-1)
    mu_cand = mu.gather(1, idx_top.unsqueeze(-1).expand(B, L, 3))
    sel_in_cand = _nms_select(mu_cand, rho_top, K=K, r2=r_nms ** 2)
    idx_center = idx_top.gather(1, sel_in_cand)
    mu_center0 = mu.gather(1, idx_center.unsqueeze(-1).expand(B, K, 3))

    # --- Step 3: 局部合并逻辑 (修复维度广播) ---
    idx_m, _ = _batched_topk_knn(mu_center0, mu, k=k_mom, mask_y=mask)
    M_eff = idx_m.shape[-1]

    def gather_feat_fixed(tensor: torch.Tensor, indices: torch.Tensor):
        """
        tensor:  [B, N, ...]
        indices: [B, K, M]
        return:  [B, K, M, ...]
        """
        B, N = tensor.shape[:2]
        B2, K, M = indices.shape
        assert B == B2

        extra = tensor.shape[2:]  # e.g. () or (3) or (3,3)
        # [B, 1, N, ...] -> [B, K, N, ...]
        t_exp = tensor.unsqueeze(1).expand(B, K, N, *extra)

        # indices -> [B, K, M, 1, 1, ...] to match extra dims
        idx = indices
        for _ in extra:
            idx = idx.unsqueeze(-1)
        idx = idx.expand(B, K, M, *extra)

        # gather along dim=2 (the N dimension)
        out = t_exp.gather(2, idx)
        return out

    mu_i = gather_feat_fixed(mu, idx_m)
    Sig_i = gather_feat_fixed(Sigma, idx_m)
    pi_i = pi.unsqueeze(1).expand(-1, K, -1).gather(2, idx_m)

    delta_init = mu_i - mu_center0.unsqueeze(2)
    A = Sig_i + (lambda2 * I).view(1, 1, 1, 3, 3) + (jitter * I).view(1, 1, 1, 3, 3)
    Lchol = torch.linalg.cholesky(A)
    y = torch.linalg.solve_triangular(Lchol, delta_init.unsqueeze(-1), upper=False)
    maha = (y.squeeze(-1) ** 2).sum(dim=-1)
    w = (pi_i * torch.exp(-0.5 * maha)).clamp_min(0.0)
    wsum = w.sum(dim=-1, keepdim=True).clamp_min(eps)

    # 计算合并后的重心
    mu_c = (w.unsqueeze(-1) * mu_i).sum(dim=2) / wsum

    # --- Step 4: Sig_raw + mahalanobis quantile calibration (isotropic scale), coverage-first ---

    d_rel = mu_i - mu_c.unsqueeze(2)
    outer = d_rel.unsqueeze(-1) * d_rel.unsqueeze(-2)
    Sig_raw = (w.view(B, K, M_eff, 1, 1) * (Sig_i + outer)).sum(dim=2) / wsum.unsqueeze(-1)
    Sig_raw = 0.5 * (Sig_raw + Sig_raw.transpose(-1, -2))

    # 1) 计算 d2 under Sig_raw
    Sig_metric = Sig_raw.unsqueeze(2) + (max(jitter, 1e-4) * I).view(1, 1, 1, 3, 3)  # [B,K,1,3,3]
    L = torch.linalg.cholesky(Sig_metric)

    delta = (mu_i - mu_c.unsqueeze(2)).unsqueeze(-1)  # [B,K,M,3,1]
    y = torch.linalg.solve_triangular(L, delta, upper=False)
    d2 = (y.squeeze(-1) ** 2).sum(dim=-1)  # [B,K,M]

    # 2) 如果要覆盖“成员椭圆”而不是点：把成员尺度折进来（保守但有效）
    # 用 trace(Sig_i) 近似成员半径^2，乘一个系数
    member = torch.diagonal(Sig_i, dim1=-2, dim2=-1).sum(-1).clamp_min(eps)  # [B,K,M]

    d2 = d2 + (radius_scale ** 2) * member  # radius_scale 推荐 1.0~2.0

    # 3) 加权分位数 d2_q
    d2_sorted, idx = torch.sort(d2, dim=-1)
    w_sorted = w.gather(2, idx)
    w_cum = torch.cumsum(w_sorted, dim=-1)
    w_tot = w_cum[..., -1:].clamp_min(eps)
    ratio = w_cum / w_tot
    j = torch.argmax((ratio >= 0.95).float(), dim=-1)  # [B,K]
    d2_q = d2_sorted.gather(2, j.unsqueeze(-1)).squeeze(-1).clamp_min(eps)

    # 4) 目标 chi2 分位数（df=3, q=0.95）
    chi2_q = 7.8147279

    # 只允许放大，不允许缩小（保证覆盖不会被破坏）
    scale = (d2_q / chi2_q).clamp_min(1.0)

    Sig_c = Sig_raw * scale.view(B, K, 1, 1)

    # 5) 厚度 & jitter
    Sig_c = Sig_c + (bone_thickness * 0.2) * I.view(1, 1, 3, 3)
    Sig_c = 0.5 * (Sig_c + Sig_c.transpose(-1, -2)) + (max(jitter, 1e-4) * I).view(1, 1, 3, 3)

    return wsum.squeeze(-1), mu_c, Sig_c, idx_center


# ==========================================
# 3. 可视化与测试
# ==========================================

def plot_ellipsoid_3d(ax, mu, sig, color, alpha=0.2):
    try:
        v, w = np.linalg.eigh(sig)
    except:
        return
    u = np.linspace(0, 2 * np.pi, 20);
    v_s = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v_s));
    y = np.outer(np.sin(u), np.sin(v_s));
    z = np.outer(np.ones_like(u), np.cos(v_s))
    scale = np.sqrt(np.maximum(v, 1e-7))
    ell = np.stack([x, y, z], axis=-1) @ (w * scale).T + mu
    ax.plot_surface(ell[..., 0], ell[..., 1], ell[..., 2], color=color, alpha=alpha, linewidth=0)


def run_refined_visualize(pdb_path, K_target=20):
    # 加载数据
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    n_l, ca_l, c_l, o_l, sc_c, sc_m = [], [], [], [], [], []
    for res in structure.get_residues():
        if Bio.PDB.is_aa(res, standard=True) and all(k in res for k in ['N', 'CA', 'C', 'O']):
            n_l.append(res['N'].get_coord());
            ca_l.append(res['CA'].get_coord())
            c_l.append(res['C'].get_coord());
            o_l.append(res['O'].get_coord())
            atoms = [a.get_coord() for a in res if a.name not in ['N', 'CA', 'C', 'O', 'OXT'] and a.element != 'H']
            tmp_c = np.zeros((14, 3));
            tmp_m = np.zeros(14);
            L = min(len(atoms), 14)
            if L > 0: tmp_c[:L] = np.array(atoms)[:L].reshape(L, 3); tmp_m[:L] = 1.0
            sc_c.append(tmp_c);
            sc_m.append(tmp_m)

    to_t = lambda x: torch.tensor(np.array(x), dtype=torch.float32)
    gr = OffsetGaussianRigid.from_all_atoms(to_t(n_l).unsqueeze(0), to_t(ca_l).unsqueeze(0), to_t(c_l).unsqueeze(0),
                                            to_t(o_l).unsqueeze(0), to_t(sc_c).unsqueeze(0), to_t(sc_m).unsqueeze(0))
    mu, sig = gr.get_gaussian_mean(), gr.get_covariance()
    pi = torch.ones_like(mu[..., 0]);
    mask = torch.ones_like(pi)

    # 执行改进后的算法
    # volume_ratio=1.5 稍微给一点余量以覆盖空隙，但不再是无限外扩
    _, mu_c, sig_c, idx_p = density_peaks_local_moments_refined(
        mu, sig, pi, mask, K=K_target,
        h=2.5,
        r_nms=5.0,
        lambda2=0.1,  # 降低 lambda 让椭圆更尖锐地指向骨架方向

        bone_thickness=0.8  # 【关键】增加厚度感
    )

    fig = plt.figure(figsize=(10, 8));
    ax = fig.add_subplot(111, projection='3d')
    # 绘制原始
    for i in range(mu.shape[1]): plot_ellipsoid_3d(ax, mu[0, i].numpy(), sig[0, i].numpy(), 'blue', alpha=0.1)
    # 绘制下采样
    for k in range(K_target): plot_ellipsoid_3d(ax, mu_c[0, k].detach().numpy(), sig_c[0, k].detach().numpy(), 'red',
                                                alpha=0.5)

    ax.set_title(f"Volume-Preserved Downsampling (K={K_target})")
    plt.show()


if __name__ == "__main__":
    run_refined_visualize("../data/1fna.pdb", K_target=20)