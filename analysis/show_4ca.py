import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def analyze_and_plot_atoms():
    # 1. 你提供的数据 (4个原子)
    points = np.array([
        [9.8911, -5.8318, -17.3436],
        [9.2471, -3.7068, -20.3056],
        [6.9961, -0.7508, -20.0096],
        [5.1011, 0.6062, -23.0426]
    ])

    # 2. 计算中心 (Mean)
    mu = points.mean(axis=0)

    # 3. 计算协方差矩阵 (Sigma)
    # rowvar=False 表示每一行是一个样本(点)，每一列是一个维度(x,y,z)
    cov = np.cov(points, rowvar=False)

    # 4. 特征分解 (Eigen Decomposition) => 得到 R 和 S
    # eigh 返回的特征值是升序排列的 (lambda_0 <= lambda_1 <= lambda_2)
    # evecs 的列向量对应特征向量
    eigvals, eigvecs = np.linalg.eigh(cov)

    # --- 整理 R 和 S ---

    # 我们通常希望长轴在最后(或者最前)，这里 eigh 默认从小到大
    # eigvals[0] 是最小特征值 (短轴)
    # eigvals[2] 是最大特征值 (长轴)

    # 旋转矩阵 R 就是特征向量矩阵
    R = eigvecs

    # 确保 R 是右手系 (行列式为 1)，如果为 -1 则翻转一列
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]

    # 5. 计算包裹系数 (Scale to fit)
    # 纯统计的特征值只代表方差(1-sigma)，包不住离群点。
    # 我们需要计算每个点到中心的“马氏距离”，找到最大的那个距离作为缩放倍数。
    # 公式: dist^2 = (p - mu)^T * Sigma^-1 * (p - mu)
    # 但更简单的方法是投影到主轴空间计算

    # 将点变换到主轴坐标系 (Principal Component Space)
    # projected = (Points - Center) @ R
    points_centered = points - mu
    points_pca = points_centered @ R

    # 在主轴系下，每个轴的标准差是 sqrt(eigval)
    # 归一化距离 = coordinate / sqrt(eigval)
    normalized_dist = points_pca ** 2 / eigvals
    mah_dist_sq = np.sum(normalized_dist, axis=1)  # 每个点的马氏距离平方
    max_dist = np.sqrt(np.max(mah_dist_sq))  # 找到最远点的倍数

    # 最终的轴半径 (半轴长)
    # 乘上 max_dist 就能刚好包住最远的点
    radii = np.sqrt(eigvals) * max_dist

    # --- 打印结果供你核对 ---
    print(f"中心 mu:\n{mu}")
    print(f"特征值 (从小到大):\n{eigvals}")
    print(f"旋转矩阵 R (列向量为轴方向):\n{R}")
    print(f"计算出的轴长 (Radii):\n{radii}")

    # ================= 绘图 =================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # A. 画原始点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=100, label='Atoms', depthshade=False)

    # B. 画中心
    ax.scatter(mu[0], mu[1], mu[2], c='black', s=100, marker='x', label='Mean')

    # C. 画旋转轴 (R 的三列)
    # 红色=长轴(对应最大特征值), 绿色=中轴, 蓝色=短轴
    # 注意：eigvals是升序，所以 col[2] 是最长轴
    colors = ['b', 'g', 'r']  # 短->长
    labels = ['Short', 'Mid', 'Long']
    for i in range(3):
        # 轴的起点是 mu，方向是 R[:, i]，长度画 1.5 倍半径方便看
        axis_vec = R[:, i] * radii[i] * 1.2
        ax.quiver(mu[0], mu[1], mu[2],
                  axis_vec[0], axis_vec[1], axis_vec[2],
                  color=colors[i], label=f'Axis {labels[i]}')

    # D. 画包裹椭球
    # 生成单位球网格
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_unit = np.outer(np.cos(u), np.sin(v))
    y_unit = np.outer(np.sin(u), np.sin(v))
    z_unit = np.outer(np.ones_like(u), np.cos(v))

    # 缩放 (应用半径) - 注意顺序对应特征值顺序
    x_scaled = x_unit * radii[0]
    y_scaled = y_unit * radii[1]
    z_scaled = z_unit * radii[2]

    # 旋转 & 平移
    # 坐标矩阵 [3, N]
    coords = np.stack([x_scaled.flatten(), y_scaled.flatten(), z_scaled.flatten()])
    # R @ coords + mu
    coords_transformed = (R @ coords).T + mu

    # 恢复形状
    x_final = coords_transformed[:, 0].reshape(x_unit.shape)
    y_final = coords_transformed[:, 1].reshape(y_unit.shape)
    z_final = coords_transformed[:, 2].reshape(z_unit.shape)

    ax.plot_wireframe(x_final, y_final, z_final, color='red', alpha=0.3, rstride=2, cstride=2)

    # 设置等比例显示，否则看着会歪
    # 简单的 box trick
    all_coords = np.concatenate([points, coords_transformed])
    max_range = np.array([all_coords[:, 0].max() - all_coords[:, 0].min(),
                          all_coords[:, 1].max() - all_coords[:, 1].min(),
                          all_coords[:, 2].max() - all_coords[:, 2].min()]).max() / 2.0
    mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) * 0.5
    mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) * 0.5
    mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("Exact Fit Ellipsoid from 4 Atoms")
    plt.show()


if __name__ == "__main__":
    analyze_and_plot_atoms()