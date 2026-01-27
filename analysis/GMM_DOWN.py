import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data.GaussianRigid import OffsetGaussianRigid
import Bio.PDB

import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D
from data.GaussianRigid import OffsetGaussianRigid
import Bio.PDB


def aggregate_gaussians_mathematically(mu, sig, cluster_labels):
    """基于二阶矩匹配的解析合并 (一步到位)"""
    device = mu.device
    num_clusters = cluster_labels.max().item() + 1

    # 1. 计算新中心
    new_mu = torch.zeros((num_clusters, 3), device=device)
    counts = torch.zeros(num_clusters, device=device).clamp(min=1.0)
    new_mu.index_add_(0, cluster_labels, mu)
    counts.index_add_(0, cluster_labels, torch.ones(mu.shape[0], device=device))
    new_mu = new_mu / counts.unsqueeze(-1)

    # 2. 计算新协方差 (Sigma_new = E[XX^T] - mu_new @ mu_new^T)
    mu_outer = torch.einsum('ni,nj->nij', mu, mu)
    second_moments = sig + mu_outer
    aggregated_second_moment = torch.zeros((num_clusters, 3, 3), device=device)
    aggregated_second_moment.index_add_(0, cluster_labels, second_moments)

    avg_second_moment = aggregated_second_moment / counts.view(-1, 1, 1)
    new_mu_outer = torch.einsum('ki,kj->kij', new_mu, new_mu)
    new_sig = avg_second_moment - new_mu_outer
    return new_mu, new_sig



def compute_density_grid(mu, sig, grid_size=40):
    """计算 3D 空间的高斯密度场"""
    device = mu.device
    mins = mu.min(dim=0)[0] - 5.0
    maxs = mu.max(dim=0)[0] + 5.0

    x = torch.linspace(mins[0], maxs[0], grid_size, device=device)
    y = torch.linspace(mins[1], maxs[1], grid_size, device=device)
    z = torch.linspace(mins[2], maxs[2], grid_size, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    flat_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(-1, 3)

    inv_sig = torch.inverse(sig)
    density = torch.zeros(flat_points.shape[0], device=device)

    # 叠加密度
    for i in range(mu.shape[0]):
        diff = flat_points - mu[i]
        dist = torch.einsum('pi,ij,pj->p', diff, inv_sig[i], diff)
        # 仅累加有效范围内的密度，优化计算
        density += torch.exp(-0.5 * dist)

    return flat_points.cpu().numpy(), density.cpu().numpy()


def plot_ellipsoid_3d(ax, mu, sig, color, alpha=0.3):
    v, w = np.linalg.eigh(sig)
    u = np.linspace(0, 2 * np.pi, 15)
    v_s = np.linspace(0, np.pi, 15)
    x = np.outer(np.cos(u), np.sin(v_s))
    y = np.outer(np.sin(u), np.sin(v_s))
    z = np.outer(np.ones_like(u), np.cos(v_s))
    scale = np.sqrt(np.maximum(v, 1e-7))
    ell = np.stack([x, y, z], axis=-1) @ (w * scale).T + mu
    ax.plot_surface(ell[..., 0], ell[..., 1], ell[..., 2], color=color, alpha=alpha, linewidth=0, zorder=10)


def compare_density_and_downsample(pdb_path, ratio=6):
    # 1. 解析 PDB (修复 GLY 等无侧链问题)
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    n_l, ca_l, c_l, o_l, sc_c, sc_m = [], [], [], [], [], []
    for res in structure.get_residues():
        if Bio.PDB.is_aa(res, standard=True):
            if not all(k in res for k in ['N', 'CA', 'C', 'O']): continue
            n_l.append(res['N'].get_coord());
            ca_l.append(res['CA'].get_coord())
            c_l.append(res['C'].get_coord());
            o_l.append(res['O'].get_coord())

            atoms = [a.get_coord() for a in res if a.name not in ['N', 'CA', 'C', 'O', 'OXT'] and a.element != 'H']
            tmp_c = np.zeros((14, 3));
            tmp_m = np.zeros(14)
            L = min(len(atoms), 14)
            if L > 0:
                tmp_c[:L] = np.array(atoms)[:L].reshape(L, 3)
                tmp_m[:L] = 1.0
            sc_c.append(tmp_c);
            sc_m.append(tmp_m)

    # 2. 生成原始 Fine 高斯
    device = torch.device("cpu")

    def to_t(x):
        return torch.tensor(np.array(x), dtype=torch.float32).to(device)

    gr_all = OffsetGaussianRigid.from_all_atoms(
        to_t(n_l).unsqueeze(0), to_t(ca_l).unsqueeze(0), to_t(c_l).unsqueeze(0),
        to_t(o_l).unsqueeze(0), to_t(sc_c).unsqueeze(0), to_t(sc_m).unsqueeze(0),
        base_thickness=0.8
    )
    mu_f = gr_all.get_gaussian_mean()[0]
    sig_f = gr_all.get_covariance()[0]

    # 3. 解析下采样 (Moment Matching)

    N = mu_f.shape[0]
    labels = (torch.arange(N) // ratio).to(device)
    mu_c, sig_c = aggregate_gaussians_mathematically(mu_f, sig_f, labels)

    # 4. 计算密度场网格
    print("Computing Global Density Distribution...")
    points, densities = compute_density_grid(mu_f, sig_f, grid_size=35)

    # --- 绘图逻辑 ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 5. 可视化全局密度场 (点云云图)
    # 过滤掉密度极低的点，减少噪声
    mask = densities > 0.05
    display_pts = points[mask]
    display_dens = densities[mask]

    # 归一化密度用于透明度控制 (0.01 到 0.2 之间，避免太遮挡椭圆)
    norm_dens = (display_dens - display_dens.min()) / (display_dens.max() - display_dens.min() + 1e-6)
    alphas = norm_dens * 0.15 + 0.01

    # 使用颜色映射表示密度高低
    scatter = ax.scatter(display_pts[:, 0], display_pts[:, 1], display_pts[:, 2],
                         c=display_dens, cmap='magma', s=10, alpha=0.1, edgecolors='none')

    # 6. 绘制下采样红色大椭圆
    for j in range(mu_c.shape[0]):
        plot_ellipsoid_3d(ax, mu_c[j].numpy(), sig_c[j].numpy(), color='green', alpha=0.3)
        # 绘制中心点
        ax.scatter(mu_c[j, 0], mu_c[j, 1], mu_c[j, 2], color='white', edgecolors='black', s=30, zorder=15)

    # 添加颜色条表示密度
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Gaussian Density Sum')

    ax.set_title("Global Density Field vs Downsampled Structure\n(Bright colors = High density / Residue clusters)")

    # 保持轴比例
    all_mu = mu_f.numpy()
    max_range = (all_mu.max() - all_mu.min()) / 2
    mid = (all_mu.max() + all_mu.min()) / 2
    ax.set_xlim(mid - max_range, mid + max_range)
    ax.set_ylim(mid - max_range, mid + max_range)
    ax.set_zlim(mid - max_range, mid + max_range)

    plt.show()



if __name__ == "__main__":
    # 确保 1fna.pdb 在当前目录下，或者修改路径
    pdb_path = "../data/1fna.pdb"
    import os

    if os.path.exists(pdb_path):
        compare_density_and_downsample(pdb_path)
    else:
        print(f"Error: {pdb_path} not found.")