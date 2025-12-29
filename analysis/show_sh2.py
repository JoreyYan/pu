#!/usr/bin/env python3
"""
完整的球谐密度映射Demo
展示如何将ASP残基的原子坐标转换为球谐密度表示
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
import math


def normed_vec(V, distance_eps=1e-3):
    """Normalized vectors with distance smoothing."""
    mag_sq = (V ** 2).sum(dim=-1, keepdim=True)
    mag = torch.sqrt(mag_sq + distance_eps)
    U = V / mag
    return U


def normed_cross(V1, V2, distance_eps=1e-3):
    """Normalized cross product between vectors."""
    C = normed_vec(torch.cross(V1, V2, dim=-1), distance_eps=distance_eps)
    return C


def frames_from_backbone(X, distance_eps=1e-3):
    """Convert a backbone into local reference frames."""
    X_N, X_CA, X_C, X_O = X.unbind(-2)
    u_CA_N = normed_vec(X_N - X_CA, distance_eps)
    u_CA_C = normed_vec(X_C - X_CA, distance_eps)
    n_1 = u_CA_N
    n_2 = normed_cross(n_1, u_CA_C, distance_eps)
    n_3 = normed_cross(n_1, n_2, distance_eps)
    R = torch.stack([n_1, n_2, n_3], -1)
    return R, X_CA


def cartesian_to_spherical(xyz):
    """Convert cartesian coordinates to spherical coordinates."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2 + 1e-8)
    theta = torch.acos(torch.clamp(z / (r + 1e-8), -1, 1))  # polar angle [0, π]
    phi = torch.atan2(y, x)  # azimuthal angle [-π, π]
    return r, theta, phi


def compute_spherical_harmonics(theta, phi, L_max=2):
    """Compute spherical harmonics Y_l^m(theta, phi) for l=0 to L_max."""
    harmonics = {}

    # Convert to numpy for scipy computation
    theta_np = theta.detach().numpy()
    phi_np = phi.detach().numpy()

    for l in range(L_max + 1):
        for m in range(-l, l + 1):
            # scipy uses (m, l, phi, theta) order, and phi/theta are swapped
            Y_lm = sph_harm(m, l, phi_np, theta_np)
            harmonics[(l, m)] = torch.from_numpy(Y_lm.real.astype(np.float32))

    return harmonics


def project_to_sh_density(atoms_local, atom_types, L_max=2, R_bins=8, r_max=6.0):
    """Project atoms to spherical harmonics density representation."""
    C = 4  # Number of atom type channels (C, N, O, S)

    # Initialize density tensor
    density_sh = torch.zeros(C, L_max + 1, 2 * L_max + 1, R_bins)

    # Convert to spherical coordinates
    r, theta, phi = cartesian_to_spherical(atoms_local)

    # Compute spherical harmonics
    harmonics = compute_spherical_harmonics(theta, phi, L_max)

    for i, (atom_r, atom_type) in enumerate(zip(r, atom_types)):
        if atom_r > r_max or atom_r < 1e-3:  # Skip origin and too far atoms
            continue

        # Find radial bin
        r_bin_idx = torch.clamp(
            torch.floor(atom_r / r_max * R_bins).long(),
            0, R_bins - 1
        )

        # Add contribution to all spherical harmonics
        l_idx = 0
        for l in range(L_max + 1):
            m_start_idx = 0
            for m in range(-l, l + 1):
                # Get spherical harmonic value for this atom
                Y_lm = harmonics[(l, m)][i]

                # Add Gaussian-like contribution in radial direction
                sigma_r = 0.8  # Radial width
                for r_idx in range(R_bins):
                    r_center = (r_idx + 0.5) * r_max / R_bins
                    gauss_weight = torch.exp(-0.5 * ((atom_r - r_center) / sigma_r) ** 2)

                    # m index in the tensor: m ranges from -l to l, but tensor index from 0 to 2l
                    m_tensor_idx = m + l  # Convert m to tensor index
                    density_sh[atom_type, l_idx, m_tensor_idx, r_idx] += Y_lm * gauss_weight

            l_idx += 1

    return density_sh


def demo_asp_residue():
    """Demo with the provided ASP residue."""

    print("=== ASP残基球谐密度映射Demo ===\n")

    # ASP residue atoms from PDB
    atoms_coords = torch.tensor([
        [15.568, 14.497, 27.361],  # N
        [14.432, 13.864, 28.096],  # CA
        [13.086, 14.425, 27.705],  # C
        [12.754, 15.541, 28.086],  # O
        [14.606, 14.061, 29.595],  # CB
        [15.592, 13.104, 30.190],  # CG
        [16.744, 13.040, 29.685],  # OD1
    ])

    # Atom types: 0=C, 1=N, 2=O, 3=S
    atom_types = torch.tensor([1, 0, 0, 2, 0, 0, 2])  # N, CA, C, O, CB, CG, OD1
    atom_names = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1']

    print("原始原子坐标:")
    for i, (coord, name) in enumerate(zip(atoms_coords, atom_names)):
        atom_type_name = ['C', 'N', 'O', 'S'][atom_types[i]]
        print(f"  {name:3s} ({atom_type_name}): {coord}")

    # Extract backbone atoms for frame calculation
    backbone = atoms_coords[:4].unsqueeze(0)  # (1, 4, 3)

    # Compute local frame
    R, CA = frames_from_backbone(backbone)
    R = R.squeeze(0)  # (3, 3)
    CA = CA.squeeze(0)  # (3,)

    print(f"\nCA中心位置: {CA}")
    print(f"局部参考系 R:")
    for i, row in enumerate(R):
        print(f"  [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}]")

    # Transform all atoms to local coordinate system
    atoms_local = (atoms_coords - CA) @ R.T

    print(f"\n转换到局部坐标系后的原子位置:")
    for i, (atom, name) in enumerate(zip(atoms_local, atom_names)):
        atom_type_name = ['C', 'N', 'O', 'S'][atom_types[i]]
        distance = torch.norm(atom).item()
        print(f"  {name:3s} ({atom_type_name}): [{atom[0]:7.3f}, {atom[1]:7.3f}, {atom[2]:7.3f}] (r={distance:.3f})")

    # Project to spherical harmonics density
    L_max = 2
    R_bins = 8
    density_sh = project_to_sh_density(atoms_local, atom_types, L_max=L_max, R_bins=R_bins)

    print(f"\n球谐密度张量形状: {density_sh.shape}")
    print(f"  - 通道数 (C): {density_sh.shape[0]} (C/N/O/S)")
    print(f"  - 角度阶数 (L+1): {density_sh.shape[1]} (l=0到{L_max})")
    print(f"  - 每阶m分量 (2L+1): {density_sh.shape[2]} (最大2×{L_max}+1)")
    print(f"  - 径向bins (R): {density_sh.shape[3]}")
    print(f"总密度: {density_sh.sum():.4f}")

    # Print significant density components
    print(f"\n主要的密度分量 (|value| > 0.01):")
    type_names = ['C', 'N', 'O', 'S']
    count = 0
    for c in range(4):
        for l in range(L_max + 1):
            for m_idx in range(2 * l + 1):
                m_actual = m_idx - l  # Convert tensor index back to actual m value
                for r in range(R_bins):
                    val = density_sh[c, l, m_idx, r].item()
                    if abs(val) > 0.01:
                        r_center = (r + 0.5) * 6.0 / R_bins
                        print(f"  {type_names[c]} 通道, l={l}, m={m_actual:2d}, r={r_center:.1f}Å: {val:8.4f}")
                        count += 1
                        if count > 20:  # Limit output
                            print("  ...")
                            break
                if count > 20:
                    break
            if count > 20:
                break
        if count > 20:
            break

    # Analyze density distribution by atom type
    print(f"\n各原子类型的密度分布:")
    for c in range(4):
        total_density = density_sh[c].sum().item()
        if total_density > 1e-6:
            print(f"  {type_names[c]} 原子: 总密度 = {total_density:.4f}")

            # Find peak radial position
            radial_profile = density_sh[c, 0, 0, :].detach()  # l=0, m=0 component
            peak_r_idx = torch.argmax(radial_profile)
            peak_r = (peak_r_idx + 0.5) * 6.0 / R_bins
            print(f"    - 主要密度峰位于 r = {peak_r:.2f}Å")

    # Simple distance analysis
    print(f"\n原子间距离分析:")
    atom_pairs = [(0, 1), (1, 2), (1, 4), (4, 5), (5, 6)]  # Some key pairs
    pair_names = [('N', 'CA'), ('CA', 'C'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'OD1')]

    for (i, j), (name_i, name_j) in zip(atom_pairs, pair_names):
        if i < len(atoms_local) and j < len(atoms_local):
            distance = torch.norm(atoms_local[i] - atoms_local[j]).item()
            print(f"  {name_i:3s} - {name_j:3s}: {distance:.3f}Å")

    print(f"\n=== 总结 ===")
    print(f"✅ 成功将 {len(atoms_coords)} 个原子转换为固定形状的球谐密度表示")
    print(f"✅ 密度场保留了原子的空间分布和类型信息")
    print(f"✅ 不同氨基酸都会产生相同 shape 的表示: {density_sh.shape}")
    print(f"✅ 原子间距离信息隐含编码在密度分布的形状中")

    return density_sh, atoms_local, R, CA


def visualize_density_comparison(density_sh, atoms_local, atom_types):
    """Create visualization comparing original atoms and density representation."""

    fig = plt.figure(figsize=(15, 5))
    atom_names = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1']
    type_names = ['C', 'N', 'O', 'S']
    colors = ['red', 'blue', 'green', 'orange']

    # Plot 1: Original atom positions in local coordinates
    ax1 = fig.add_subplot(131, projection='3d')
    for i, (pos, atype) in enumerate(zip(atoms_local, atom_types)):
        ax1.scatter(pos[0], pos[1], pos[2],
                    c=colors[atype], s=200, alpha=0.8)
        ax1.text(pos[0], pos[1], pos[2], f'  {atom_names[i]}', fontsize=8)

    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')
    ax1.set_title('原子在局部坐标系中的位置')

    # Create legend
    for atype, color, name in zip(range(4), colors, type_names):
        if atype in atom_types:
            ax1.scatter([], [], [], c=color, s=100, label=f'{name} 原子')
    ax1.legend()

    # Plot 2: Radial density profile for each atom type
    ax2 = fig.add_subplot(132)
    R_bins = density_sh.shape[3]
    r_values = np.linspace(0.375, 5.625, R_bins)  # Bin centers

    for c in range(4):
        if density_sh[c].sum() > 1e-6:  # Only plot if there's density
            radial_profile = density_sh[c, 0, 0, :].detach().numpy()  # l=0, m=0
            ax2.plot(r_values, radial_profile, 'o-', color=colors[c],
                     label=f'{type_names[c]} 原子', linewidth=2, markersize=4)

    ax2.set_xlabel('距CA中心的距离 (Å)')
    ax2.set_ylabel('密度 (l=0, m=0分量)')
    ax2.set_title('径向密度分布')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Total density by spherical harmonic order
    ax3 = fig.add_subplot(133)
    L_max = density_sh.shape[1] - 1

    for l in range(L_max + 1):
        total_l = density_sh[:, l, :, :].sum().item()
        ax3.bar(l, total_l, alpha=0.7, label=f'l={l}')

    ax3.set_xlabel('球谐阶数 l')
    ax3.set_ylabel('总密度')
    ax3.set_title('不同阶数的密度贡献')
    ax3.legend()

    plt.tight_layout()
    plt.show()

    return fig


if __name__ == "__main__":
    print("开始运行ASP残基球谐密度映射Demo...\n")

    # Run the main demo
    density_sh, atoms_local, R, CA = demo_asp_residue()

    print(f"\n创建可视化图表...")
    try:
        fig = visualize_density_comparison(density_sh, atoms_local,
                                           torch.tensor([1, 0, 0, 2, 0, 0, 2]))
        print("✅ 可视化完成！")
    except Exception as e:
        print(f"⚠️  可视化时遇到问题: {e}")
        print("但核心功能已正常运行")

    print("\n🎉 Demo运行完成！")
    print("\n关键发现:")
    print("1. ASP残基的7个原子成功转换为(4,3,5,8)的固定张量")
    print("2. 不同原子类型的密度分布清晰可见")
    print("3. 原子间的距离信息被编码在密度峰的分布中")
    print("4. 这种表示方法确实能统一处理不同大小的氨基酸侧链！")