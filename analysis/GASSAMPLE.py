import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_parent_children_dist(K=5, N=1000):
    device = "cpu"  # 绘图用CPU即可

    # --- 1. 模拟父高斯 (K 个电源) ---
    # 随机中心
    mu_p = torch.randn(K, 3) * 6.0
    # 随机形状 (Sigma)
    L = torch.randn(K, 3, 3) * 0.8
    L[:, 0, 0] *= 5.0  # 让椭圆长一点，方便观察方向
    Sigma_p = torch.matmul(L, L.transpose(-1, -2))

    # 每个父节点的权重 (Occupancy)
    pi = torch.ones(K) / K

    # --- 2. 采样 N 个子点 ---
    # Step A: 选父亲
    parent_idx = torch.multinomial(pi, num_samples=N, replacement=True)

    # Step B: 采样
    curr_mu = mu_p[parent_idx]
    curr_Sigma = Sigma_p[parent_idx]
    curr_L = torch.linalg.cholesky(curr_Sigma + torch.eye(3) * 1e-4)
    eps = torch.randn(N, 3, 1)
    child_points = curr_mu + torch.matmul(curr_L, eps).squeeze(-1)

    # --- 3. 绘图 ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制子点 (蓝色) - 所有的子用一个颜色
    cp = child_points.numpy()
    ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2],
               color='royalblue', s=10, alpha=0.4, label=f'N={N} Sampled Children')

    # 绘制父中心 (红色) - 所有的父用一个颜色
    mp = mu_p.numpy()
    ax.scatter(mp[:, 0], mp[:, 1], mp[:, 2],
               color='crimson', s=200, marker='X', edgecolors='white',
               linewidths=2, label=f'K={K} Parent Sources', zorder=10)

    # 可选：绘制父椭圆的轮廓线 (更直观)
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    unit_sphere = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])

    for i in range(K):
        # 提取 Cholesky 分量用于拉伸球体
        L_np = torch.linalg.cholesky(Sigma_p[i]).numpy()
        # 变换球体到椭球形 (2 sigma 范围)
        ellipsoid = (L_np @ unit_sphere.reshape(3, -1) * 2).reshape(3, 20, 10)
        ax.plot_wireframe(ellipsoid[0] + mp[i, 0],
                          ellipsoid[1] + mp[i, 1],
                          ellipsoid[2] + mp[i, 2],
                          color='crimson', alpha=0.1)

    ax.set_title("Visual Verification: Geometry-Driven Upsampling")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")
    ax.legend()

    # 设置相等的比例尺，防止椭圆变形
    max_range = np.array(
        [cp[:, 0].max() - cp[:, 0].min(), cp[:, 1].max() - cp[:, 1].min(), cp[:, 2].max() - cp[:, 2].min()]).max() / 2.0
    mid_x = (cp[:, 0].max() + cp[:, 0].min()) * 0.5
    mid_y = (cp[:, 1].max() + cp[:, 1].min()) * 0.5
    mid_z = (cp[:, 2].max() + cp[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


if __name__ == "__main__":
    visualize_parent_children_dist(K=24, N=120)