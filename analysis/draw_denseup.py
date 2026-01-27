import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_ellipsoid_3d(ax, mu, sig, color, alpha=0.2, lw=0):
    try:
        v, w = np.linalg.eigh(sig)
    except:
        return
    u = np.linspace(0, 2 * np.pi, 20)
    v_sphere = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v_sphere))
    y = np.outer(np.sin(u), np.sin(v_sphere))
    z = np.outer(np.ones_like(u), np.cos(v_sphere))

    scale = np.sqrt(np.maximum(v, 1e-9))
    ellipsoid = np.stack([x, y, z], axis=-1) @ (w * scale).T + mu
    ax.plot_surface(ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2],
                    color=color, alpha=alpha, linewidth=lw)


def draw_gaussian_upsample_strict_inside(mu_p, Sig_p, N_target):
    K = mu_p.shape[0]
    device = mu_p.device

    # --- Step 1: 严格在椭球内采样中心点 ---
    M = N_target * 5
    p_id = torch.randint(0, K, (M,), device=device)  # 假设父辈权重均匀

    # 生成单位球内的均匀分布点
    # 方法：随机方向 * (U(0,1)^(1/3)) 保证体积均匀
    z = torch.randn(M, 3, device=device)
    z = z / torch.norm(z, dim=1, keepdim=True)  # 投影到球面
    r = torch.rand(M, 1, device=device) ** (1 / 3)  # 径向分布
    z_inside = z * r

    mu_k = mu_p[p_id]
    Sig_k = Sig_p[p_id]
    L = torch.linalg.cholesky(Sig_k + torch.eye(3, device=device) * 1e-4)

    # 变换到父椭球空间：这些点 100% 在父椭球 1-sigma 内部
    cand_x = mu_k + torch.einsum("mij,mj->mi", L, z_inside)

    # --- Step 2: FPS 筛选 ---
    def simple_fps(x, num_points):
        indices = [0]
        dist = torch.cdist(x, x)
        min_dist = dist[0]
        for _ in range(1, num_points):
            far_idx = torch.argmax(min_dist)
            indices.append(far_idx.item())
            min_dist = torch.min(min_dist, dist[far_idx])
        return x[indices], indices

    mu0, selected_indices = simple_fps(cand_x, N_target)
    child_parent_id = p_id[selected_indices]

    # --- Step 3: 调整子椭圆大小以匹配体积 ---
    # 计算每个父辈分到的平均点数
    pts_per_parent = N_target / K
    # 理论缩放因子：1/N^(1/3)。
    # 为了视觉上不溢出，再加一个 0.7 的收缩系数
    scale_factor = (1.0 / (pts_per_parent ** (1 / 3))) * 0.7
    Sig0 = Sig_p[child_parent_id] * (scale_factor ** 2)

    # --- 可视化 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制父椭球（蓝色，半透明线框）
    for k in range(K):
        plot_ellipsoid_3d(ax, mu_p[k].cpu().numpy(), Sig_p[k].cpu().numpy(), 'blue', alpha=0.08)

    # 绘制子椭球（红色）
    for n in range(N_target):
        m_n = mu0[n].cpu().numpy()
        s_n = Sig0[n].cpu().numpy()
        plot_ellipsoid_3d(ax, m_n, s_n, 'red', alpha=0.4)

    ax.set_title(f"Strict Inside Upsampling\n{N_target} children contained within {K} parents")

    # 保持坐标轴比例
    all_pts = cand_x.cpu().numpy()
    max_range = (all_pts.max() - all_pts.min()) / 2.0
    mid = (all_pts.max() + all_pts.min()) / 2.0
    ax.set_xlim(mid - max_range, mid + max_range)
    ax.set_ylim(mid - max_range, mid + max_range)
    ax.set_zlim(mid - max_range, mid + max_range)
    plt.show()


# --- 测试数据 ---
mu_p = torch.tensor([[0., 0., 0.], [6., 6., 0.]])
Sig_p = torch.stack([
    torch.diag(torch.tensor([12., 2., 2.])),  # 长条形
    torch.diag(torch.tensor([3., 3., 10.]))  # 立柱形
])
draw_gaussian_upsample_strict_inside(mu_p, Sig_p, 14)