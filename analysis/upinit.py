import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import math
import torch
import torch.nn.functional as F


# -----------------------------
# small utils
# -----------------------------
def _safe_cholesky(A: torch.Tensor, jitter: float = 1e-6, max_tries: int = 4):
    """A: [...,3,3] SPD-ish -> L: [...,3,3]"""
    I = torch.eye(3, device=A.device, dtype=A.dtype)
    for t in range(max_tries):
        try:
            return torch.linalg.cholesky(A + (jitter * (10 ** t)) * I)
        except RuntimeError:
            continue
    # last resort: symmetrize + bigger jitter
    A = 0.5 * (A + A.transpose(-1, -2))
    return torch.linalg.cholesky(A + (jitter * (10 ** (max_tries - 1))) * I)


def _mahalanobis2(delta: torch.Tensor, Sigma: torch.Tensor, jitter: float = 1e-6):
    """
    delta: [...,3]
    Sigma: [...,3,3]
    return: [...], delta^T Sigma^-1 delta
    """
    L = _safe_cholesky(Sigma, jitter=jitter)  # [...,3,3]
    y = torch.linalg.solve_triangular(L, delta.unsqueeze(-1), upper=False)  # [...,3,1]
    return (y.squeeze(-1) ** 2).sum(dim=-1)


def _project_inside_ellipsoid(mu_child, mu_parent, Sig_parent, tau2: float = 9.0, jitter: float = 1e-6):
    """
    Ensure (mu_child - mu_parent)^T Sig_parent^-1 (mu_child - mu_parent) <= tau2
    by radial scaling in parent metric (one-shot projection).
    """
    d = mu_child - mu_parent
    d2 = _mahalanobis2(d, Sig_parent, jitter=jitter)  # [...]
    # if d2 <= tau2 -> keep; else scale down by sqrt(tau2/d2)
    scale = torch.sqrt((tau2 / d2.clamp_min(1e-12))).clamp_max(1.0)
    return mu_parent + d * scale.unsqueeze(-1)


def _allocate_counts(pi: torch.Tensor, mask_parent: torch.Tensor, N: int,
                     min_per_parent: int = 1, max_per_parent: int = None):
    """
    pi: [B,K] normalized over valid parents
    mask_parent: [B,K] {0,1}
    return m: [B,K] int counts, sum_k m = N for each batch b
    """
    B, K = pi.shape
    device = pi.device

    valid = (mask_parent > 0.5)
    m = torch.zeros((B, K), device=device, dtype=torch.long)

    for b in range(B):
        idx = torch.nonzero(valid[b], as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue

        # initial: min_per_parent
        m[b, idx] = min_per_parent
        rem = N - int(min_per_parent * idx.numel())
        if rem < 0:
            # too many parents for N: fall back to 1-hot-ish
            m[b].zero_()
            # keep first N valid
            m[b, idx[:N]] = 1
            continue

        # distribute remaining by pi
        pb = pi[b, idx]
        pb = pb / pb.sum().clamp_min(1e-9)
        add = torch.floor(pb * rem).to(torch.long)
        m[b, idx] += add
        rem2 = rem - int(add.sum().item())

        if rem2 > 0:
            # assign leftover by largest fractional parts
            frac = (pb * rem) - add.to(pb.dtype)
            _, order = torch.sort(frac, descending=True)
            m[b, idx[order[:rem2]]] += 1

        if max_per_parent is not None:
            m[b, idx] = torch.clamp(m[b, idx], max=int(max_per_parent))
            # re-balance if clamped changed total
            total = int(m[b, idx].sum().item())
            if total != N:
                # adjust by adding/removing one-by-one (cheap; K small)
                diff = N - total
                if diff > 0:
                    # add to largest pi
                    _, order = torch.sort(pb, descending=True)
                    for t in range(diff):
                        m[b, idx[order[t % order.numel()]]] += 1
                else:
                    # remove from smallest pi but keep >=1
                    _, order = torch.sort(pb, descending=False)
                    t = 0
                    while diff < 0 and t < 10_000:
                        ksel = idx[order[t % order.numel()]]
                        if m[b, ksel] > min_per_parent:
                            m[b, ksel] -= 1
                            diff += 1
                        t += 1

    return m


def _sobol_normal(n: int, device, dtype, scramble: bool = True):
    """
    Generate n samples ~ N(0,I) in R^3 using Sobol + inverse-CDF.
    """
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=scramble)
    u = engine.draw(n).to(device=device, dtype=dtype).clamp(1e-6, 1 - 1e-6)
    # inverse CDF of standard normal
    z = torch.erfinv(2 * u - 1) * math.sqrt(2.0)
    return z  # [n,3]


def _one_step_repulsion(mu: torch.Tensor, parent_idx: torch.Tensor, eta: float = 0.02, eps: float = 1e-4):
    """
    mu: [B,N,3], parent_idx: [B,N] (int)
    One shot repulsion within same parent to reduce collapse.
    """
    B, N, _ = mu.shape
    mu2 = mu.clone()
    for b in range(B):
        pid = parent_idx[b]
        for k in pid.unique():
            sel = torch.nonzero(pid == k, as_tuple=False).squeeze(-1)
            if sel.numel() <= 1:
                continue
            x = mu2[b, sel]  # [m,3]
            diff = x[:, None, :] - x[None, :, :]  # [m,m,3]
            dist2 = (diff ** 2).sum(dim=-1) + eps
            # zero self
            dist2.fill_diagonal_(1e9)
            force = (diff / dist2[..., None]).sum(dim=1)  # [m,3]
            mu2[b, sel] = x + eta * force
    return mu2


# -----------------------------
# core: cover upsample init
# -----------------------------
@torch.no_grad()
def cover_upsample_init(
    mu_p: torch.Tensor,        # [B,K,3]
    Sig_p: torch.Tensor,       # [B,K,3,3]
    pi: torch.Tensor,          # [B,K] (normalized over valid parents)
    mask_parent: torch.Tensor, # [B,K]
    node_mask: torch.Tensor,   # [B,N]
    jitter: float = 1e-6,
    tau2_inside: float = 9.0,  # how far (Mahalanobis^2) child centers can go in parent
    min_per_parent: int = 1,
    sigma_floor: float = 0.03,
    sigma_ceil: float = 2.0,
    mix_cover_alpha: float = 0.35,   # blend split-sigma with spacing-sigma
    k_nn_spacing: int = 4,           # used only for spacing-sigma
    repulse_eta: float = 0.02,       # 0 to disable
):
    """
    Returns:
      mu0: [B,N,3]
      Sig0: [B,N,3,3]
      parent_idx: [B,N]  (int in [0,K))
      m_counts: [B,K] int
    """
    B, K, _ = mu_p.shape
    N = node_mask.shape[1]
    device, dtype = mu_p.device, mu_p.dtype
    I = torch.eye(3, device=device, dtype=dtype)[None, None]

    # (1) allocate m_k per parent (hard guarantee coverage across parents)
    m = _allocate_counts(pi, mask_parent, N, min_per_parent=min_per_parent)  # [B,K] long

    # (2) generate mu0 per parent using Sobol-normal mapped by parent chol
    mu0 = torch.zeros((B, N, 3), device=device, dtype=dtype)
    Sig0_split = torch.zeros((B, N, 3, 3), device=device, dtype=dtype)
    parent_idx = torch.zeros((B, N), device=device, dtype=torch.long)

    for b in range(B):
        cursor = 0
        for k in range(K):
            if mask_parent[b, k] < 0.5:
                continue
            mk = int(m[b, k].item())
            if mk <= 0:
                continue

            # low-discrepancy samples in R^3
            z = _sobol_normal(mk, device=device, dtype=dtype, scramble=True)  # [mk,3]

            # map into ellipsoid by parent covariance
            Lk = _safe_cholesky(Sig_p[b, k], jitter=jitter)  # [3,3]
            x = mu_p[b, k].unsqueeze(0) + (z @ Lk.transpose(0, 1))  # [mk,3]

            # project inside parent ellipsoid (avoid far tails)
            x = _project_inside_ellipsoid(
                x, mu_p[b, k].unsqueeze(0), Sig_p[b, k].unsqueeze(0),
                tau2=tau2_inside, jitter=jitter
            )

            # write
            mu0[b, cursor:cursor + mk] = x
            parent_idx[b, cursor:cursor + mk] = k

            # split-sigma: Sigma_child = Sigma_parent / mk^(2/3)
            # (mk=1 -> same scale)
            scale = float(max(mk, 1)) ** (2.0 / 3.0)
            Sig_child = Sig_p[b, k] / scale
            Sig0_split[b, cursor:cursor + mk] = Sig_child.unsqueeze(0).expand(mk, 3, 3)

            cursor += mk

        # if for any reason cursor != N, pad with first valid parent (rare)
        if cursor < N:
            # find a valid parent
            kk = int(torch.nonzero(mask_parent[b] > 0.5, as_tuple=False)[0].item())
            mk = N - cursor
            z = _sobol_normal(mk, device=device, dtype=dtype, scramble=True)
            Lk = _safe_cholesky(Sig_p[b, kk], jitter=jitter)
            x = mu_p[b, kk].unsqueeze(0) + (z @ Lk.transpose(0, 1))
            x = _project_inside_ellipsoid(x, mu_p[b, kk].unsqueeze(0), Sig_p[b, kk].unsqueeze(0),
                                          tau2=tau2_inside, jitter=jitter)
            mu0[b, cursor:] = x
            parent_idx[b, cursor:] = kk
            Sig0_split[b, cursor:] = (Sig_p[b, kk].unsqueeze(0).expand(mk, 3, 3))

    # apply node mask (in case padding N)
    mu0 = mu0 * node_mask[..., None]

    # (3) optional one-shot repulsion within same parent (helps avoid clumps)
    if repulse_eta is not None and repulse_eta > 0:
        mu0 = _one_step_repulsion(mu0, parent_idx, eta=float(repulse_eta))
        mu0 = mu0 * node_mask[..., None]

    # (4) spacing-based sigma (your existing cover sigma idea)
    #     You already have init_sigma_from_child_spacing(mu0, node_mask, ...).
    #     We'll re-use it by expecting you to import it. If not available, fallback to isotropic by local nn dist.
    Sig0_cover = init_sigma_from_child_spacing(
        mu0, node_mask, k_nn=int(k_nn_spacing),
        alpha=0.6,  # same meaning as your old function uses
        sigma_floor=float(sigma_floor),
        sigma_ceil=float(sigma_ceil),
    )
    Sig0_cover = Sig0_cover + jitter * I * node_mask[:, :, None, None]

    # (5) blend: split-sigma (mass/partition semantics) + cover-sigma (geometry spacing)
    a = float(mix_cover_alpha)
    Sig0 = (1.0 - a) * Sig0_split + a * Sig0_cover
    Sig0 = 0.5 * (Sig0 + Sig0.transpose(-1, -2)) + jitter * I

    # clamp diag magnitude as sanity
    diag = torch.diagonal(Sig0, dim1=-2, dim2=-1)
    diag = diag.clamp_min(float(sigma_floor) ** 2).clamp_max(float(sigma_ceil) ** 2)
    Sig0 = Sig0.clone()
    Sig0[..., 0, 0] = diag[..., 0]
    Sig0[..., 1, 1] = diag[..., 1]
    Sig0[..., 2, 2] = diag[..., 2]

    return mu0, Sig0, parent_idx, m

# ---------------------------------------------------------
# 1. 模拟你的工具函数 (按照你代码里的逻辑实现)
# ---------------------------------------------------------

def sample_from_mixture(mu_p, Sig_p, pi, M, mask_parent, eps=1e-6):
    B, K, _ = mu_p.shape
    # 模拟从 GMM 采样: 先选分量，再加高斯噪声
    choices = torch.multinomial(pi + eps, M, replacement=True)  # [B, M]
    L = torch.linalg.cholesky(Sig_p + eps * torch.eye(3, device=Sig_p.device))

    z = torch.randn(B, M, 3, device=mu_p.device)
    cand_x = torch.zeros(B, M, 3, device=mu_p.device)
    for b in range(B):
        L_selected = L[b, choices[b]]  # [M, 3, 3]
        mu_selected = mu_p[b, choices[b]]  # [M, 3]
        cand_x[b] = mu_selected + torch.bmm(L_selected, z[b].unsqueeze(-1)).squeeze(-1)
    return cand_x, None


def fps_points_batch(x, n_points):
    # 极简版 FPS 实现用于演示
    B, M, _ = x.shape
    indices = torch.zeros(B, n_points, dtype=torch.long, device=x.device)
    for b in range(B):
        dist = torch.ones(M, device=x.device) * 1e10
        farthest = 0
        for i in range(n_points):
            indices[b, i] = farthest
            centroid = x[b, farthest, :].view(1, 3)
            d2 = torch.sum((x[b] - centroid) ** 2, dim=1)
            dist = torch.min(dist, d2)
            farthest = torch.max(dist, dim=0)[1]
    return indices


def init_sigma_from_child_spacing(mu0, node_mask, k_nn, alpha, sigma_floor, sigma_ceil):
    # 你原来的逻辑：基于 k-NN 距离计算各向同性 Sigma
    B, N, _ = mu0.shape
    dist2 = torch.sum((mu0[:, :, None, :] - mu0[:, None, :, :]) ** 2, dim=-1)
    val, _ = torch.topk(dist2, k=k_nn + 1, largest=False)
    avg_dist = torch.sqrt(val[:, :, 1:].mean(dim=-1))
    sig_val = (avg_dist * alpha).clamp(sigma_floor, sigma_ceil)
    return sig_val[:, :, None, None] * torch.eye(3, device=mu0.device)


# ---------------------------------------------------------
# 2. 引入你提供的 "Cover Up-init" 核心逻辑 (见上文函数，此处略去重复定义)
# ---------------------------------------------------------
# [此处假设已经定义了你提供的 _allocate_counts, _safe_cholesky, cover_upsample_init 等]

# ---------------------------------------------------------
# 3. 运行对比实验
# ---------------------------------------------------------

@torch.no_grad()
def run_comparison():
    B, K, N = 1, 2, 64
    device = "cpu"

    # 构建 Parent: 一个横向长条，一个纵向圆球
    mu_p = torch.tensor([[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    Sig_p = torch.zeros(1, 2, 3, 3)
    Sig_p[0, 0] = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])  # 极细长
    Sig_p[0, 1] = torch.eye(3) * 0.5  # 均匀圆球

    pi = torch.tensor([[0.5, 0.5]])
    mask_parent = torch.ones(1, 2)
    node_mask = torch.ones(1, N)

    # --- 方案 A: 你原来的代码逻辑 ---
    M = 6 * N
    cand_x, _ = sample_from_mixture(mu_p, Sig_p, pi, M, mask_parent)
    idx = fps_points_batch(cand_x, N)
    mu_old = cand_x.gather(1, idx[..., None].expand(B, N, 3))
    Sig_old = init_sigma_from_child_spacing(mu_old, node_mask, k_nn=4, alpha=0.6, sigma_floor=0.03, sigma_ceil=2.0)

    # --- 方案 B: 新的 Cover 代码逻辑 ---
    # 调用你刚才给我的 cover_upsample_init
    mu_new, Sig_new, _, _ = cover_upsample_init(
        mu_p, Sig_p, pi, mask_parent, node_mask,
        mix_cover_alpha=0.2,  # 强调 split 语义
        sigma_floor=0.03, sigma_ceil=2.0
    )

    visualize(mu_p, Sig_p, mu_old, Sig_old, mu_new, Sig_new)


def get_ellipse(mu, sig, color, alpha, label=""):
    vals, vecs = np.linalg.eigh(sig[:2, :2].numpy())
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * np.sqrt(np.maximum(vals, 1e-9))
    return Ellipse(mu[:2].numpy(), w, h, angle=theta, color=color, alpha=alpha, label=label)


def visualize(mu_p, Sig_p, mu_old, Sig_old, mu_new, Sig_new):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, mu, sig, title in zip([ax1, ax2], [mu_old, mu_new], [Sig_old, Sig_new],
                                  ["Original: Density + FPS + Spacing", "Proposed: Cover (Split + Fill)"]):
        # 画 Parent 轮廓 (3-sigma)
        for k in range(mu_p.shape[1]):
            ax.add_patch(get_ellipse(mu_p[0, k], Sig_p[0, k] * 9, 'blue', 0.1, "Parent" if k == 0 else ""))

        # 画 Child (2-sigma)
        for n in range(mu.shape[1]):
            ax.add_patch(get_ellipse(mu[0, n], sig[0, n], 'red', 0.4))
            ax.plot(mu[0, n, 0], mu[0, n, 1], 'k.', markersize=1)

        ax.set_title(title);
        ax.set_aspect('equal');
        ax.set_xlim(-6, 6);
        ax.set_ylim(-3, 3)
        ax.legend()

    plt.show()




def get_ellipsoid_surface(mu, sig, n=10, scale=2.0):
    """生成椭球体的网格数据"""
    # 特征分解获取轴长和旋转
    vals, vecs = torch.linalg.eigh(sig)
    # 避免负值导致的 nan
    radii = scale * torch.sqrt(torch.clamp(vals, min=1e-9)).numpy()

    # 构建单位球网格
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # 旋转和缩放
    ellipsoid = np.stack([x * radii[0], y * radii[1], z * radii[2]], axis=-1)
    ellipsoid = ellipsoid @ vecs.T.numpy() + mu.numpy()

    return ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2]


def visualize_3d_comparison():
    B, K, N = 1, 2, 80
    # 显式指定类型，避免 numpy 引入 double 类型
    device = "cpu"
    dtype = torch.float32

    # --- 场景构建 ---
    mu_p = torch.tensor([[[-2.5, 0.0, 0.0], [2.5, 0.0, 0.0]]], dtype=dtype)
    Sig_p = torch.zeros(1, 2, 3, 3, dtype=dtype)

    # Parent 0: 扁平飞盘
    Sig_p[0, 0] = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.1]], dtype=dtype)

    # Parent 1: 倾斜细长雪茄
    angle = np.pi / 4
    # 确保 rot 是 float32
    rot = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ], dtype=dtype)

    # 现在这里就不会报错了
    Sig_p[0, 1] = rot @ torch.diag(torch.tensor([5.0, 0.2, 0.2], dtype=dtype)) @ rot.T

    # ... 后面的 pi, mask_parent 等也建议加上 dtype=dtype
    pi = torch.tensor([[0.5, 0.5]], dtype=dtype)
    mask_parent = torch.ones(1, 2, dtype=dtype)
    node_mask = torch.ones(1, N, dtype=dtype)

    # --- 1. 旧版逻辑 (Density + FPS) ---
    M = 6 * N
    cand_x, _ = sample_from_mixture(mu_p, Sig_p, pi, M, mask_parent)
    idx = fps_points_batch(cand_x, N)
    mu_old = cand_x.gather(1, idx[..., None].expand(B, N, 3))
    Sig_old = init_sigma_from_child_spacing(mu_old, node_mask, 4, 0.6, 0.03, 2.0)

    # --- 2. 新版逻辑 (Cover) ---
    mu_new, Sig_new, _, _ = cover_upsample_init(
        mu_p, Sig_p, pi, mask_parent, node_mask,
        mix_cover_alpha=0.2,  # 强化 Split 语义
        sigma_floor=0.03, sigma_ceil=2.0
    )

    # --- 绘图 ---
    fig = plt.figure(figsize=(18, 8))

    for i, (mu, sig, title) in enumerate([(mu_old, Sig_old, "Old: Isotropic Gaps"),
                                          (mu_new, Sig_new, "New: Anisotropic Cover")]):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')

        # A. 画 Parent (用浅色网格线表示)
        for k in range(K):
            X, Y, Z = get_ellipsoid_surface(mu_p[0, k], Sig_p[0, k], n=15, scale=2.5)
            ax.plot_wireframe(X, Y, Z, color='blue', alpha=0.08, linewidth=0.5)

        # B. 画 Child (点 + 方向轴)
        mu_plot = mu[0].numpy()
        ax.scatter(mu_plot[:, 0], mu_plot[:, 1], mu_plot[:, 2], c='red', s=10, alpha=0.6)

        # 为了不让 3D 太乱，每个 Child 只画出它的主方向轴 (Eigenvector)
        for n in range(N):
            vals, vecs = np.linalg.eigh(sig[0, n].numpy())
            main_dir = vecs[:, -1] * np.sqrt(vals[-1]) * 2.0  # 主轴方向
            p0 = mu_plot[n]
            p1 = p0 + main_dir
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color='red', alpha=0.4, linewidth=1)

        ax.set_title(title)
        ax.set_xlabel('X');
        ax.set_ylabel('Y');
        ax.set_zlabel('Z')
        ax.set_xlim(-6, 6);
        ax.set_ylim(-6, 6);
        ax.set_zlim(-3, 3)
        ax.view_init(elev=35, azim=-45)

    plt.tight_layout()
    plt.show()




def get_ellipsoid_surface(mu, sig, n=10, scale=1.0):
    """
    生成椭球体的网格数据.
    n: 网格密度 (经纬线数量)
    scale: 缩放因子 (例如 1.0 为 1-sigma, 2.0 为 2-sigma)
    """
    # 特征分解获取轴长和旋转
    # 使用 torch.linalg.eigh 保证对称矩阵特征分解的稳定性
    vals, vecs = torch.linalg.eigh(sig)
    # 加上一个极小值防止 sqrt(0)
    radii = scale * torch.sqrt(torch.clamp(vals, min=1e-9)).numpy()

    # 构建单位球网格
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # 将网格点变换到椭球体坐标系
    # 先缩放，再旋转，最后平移
    ellipsoid = np.stack([x * radii[0], y * radii[1], z * radii[2]], axis=-1)
    ellipsoid = ellipsoid @ vecs.T.numpy() + mu.numpy()

    return ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2]


@torch.no_grad()
def visualize_3d_comparison_full_ellipsoids():
    B, K, N = 1, 2, 80
    device = "cpu"
    dtype = torch.float32  # 明确指定 float32

    # --- 场景构建 ---
    # Parent 0: 极其扁平的“飞盘” (在 Z 轴方向被压缩)
    mu_p = torch.tensor([[[-2.5, 0.0, 0.0], [2.5, 0.0, 0.0]]], dtype=dtype)
    Sig_p = torch.zeros(1, 2, 3, 3, dtype=dtype)
    Sig_p[0, 0] = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.1]], dtype=dtype)

    # Parent 1: 倾斜的细长“雪茄”
    angle = np.pi / 4
    rot = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ], dtype=dtype)
    # 构建一个沿 X 轴拉伸的协方差，然后旋转它
    Sig_p[0, 1] = rot @ torch.diag(torch.tensor([5.0, 0.2, 0.2], dtype=dtype)) @ rot.T

    pi = torch.tensor([[0.5, 0.5]], dtype=dtype)
    mask_parent = torch.ones(1, 2, dtype=dtype)
    node_mask = torch.ones(1, N, dtype=dtype)

    # --- 1. 旧版逻辑 (Density + FPS + Isotropic Spacing) ---
    M = 6 * N
    cand_x, _ = sample_from_mixture(mu_p, Sig_p, pi, M, mask_parent)
    idx = fps_points_batch(cand_x, N)
    mu_old = cand_x.gather(1, idx[..., None].expand(B, N, 3))
    # 生成各向同性的球体 Sigma
    Sig_old = init_sigma_from_child_spacing(mu_old, node_mask, k_nn=4, alpha=0.6, sigma_floor=0.03, sigma_ceil=2.0)

    # --- 2. 新版逻辑 (Cover Split + Fill) ---
    mu_new, Sig_new, _, _ = cover_upsample_init(
        mu_p, Sig_p, pi, mask_parent, node_mask,
        mix_cover_alpha=0.2,  # alpha 较小，强调从 Parent 继承形状
        sigma_floor=0.03, sigma_ceil=2.0,
        tau2_inside=9.0  # 3-sigma 投影范围
    )

    # --- 绘图 ---
    fig = plt.figure(figsize=(18, 8))

    # 循环绘制左右两个子图
    for i, (mu, sig, title) in enumerate([(mu_old, Sig_old, "Old: Isotropic Gaps"),
                                          (mu_new, Sig_new, "New: Anisotropic Cover")]):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')

        # A. 画 Parent (蓝色，大而透明)
        for k in range(K):
            # scale=2.5 大约涵盖 98% 的概率质量
            X, Y, Z = get_ellipsoid_surface(mu_p[0, k], Sig_p[0, k], n=15, scale=2.5)
            ax.plot_wireframe(X, Y, Z, color='blue', alpha=0.08, linewidth=0.5)

        # B. 画 Child (红色，完整的椭圆线框)
        for n in range(N):
            # scale=1.5 展示 Child 的核心区域 (约 87% 质量)
            # n=8 降低网格密度，避免过于密集
            X, Y, Z = get_ellipsoid_surface(mu[0, n], sig[0, n], n=8, scale=1.5)
            # 使用红色线框，透明度适中
            ax.plot_wireframe(X, Y, Z, color='red', alpha=0.3, linewidth=0.8)

        ax.set_title(title)
        ax.set_xlabel('X');
        ax.set_ylabel('Y');
        ax.set_zlabel('Z')
        # 设置坐标轴范围，保证两个图比例一致
        ax.set_xlim(-6, 6);
        ax.set_ylim(-6, 6);
        ax.set_zlim(-3, 3)
        # 设置一个便于观察的视角
        ax.view_init(elev=35, azim=-45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_3d_comparison_full_ellipsoids()