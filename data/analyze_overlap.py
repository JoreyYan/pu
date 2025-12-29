"""
分析氨基酸之间的实际重叠度和可分离性
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analyze_gaussian_distribution import *

STANDARD_AA = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL'
]

def compute_overlap_coefficient(mean1, std1, mean2, std2):
    """
    计算两个高斯分布的重叠系数 (Overlap Coefficient)
    返回值: 0-1, 0=完全分离, 1=完全重叠

    使用简化公式: OVL ≈ 2 * Φ(-|μ1-μ2|/(2*sqrt(σ1²+σ2²)))
    其中 Φ 是标准正态分布的CDF
    """
    from scipy.stats import norm

    delta_mean = abs(mean1 - mean2)
    pooled_std = np.sqrt(std1**2 + std2**2)

    if pooled_std < 1e-6:
        return 1.0 if delta_mean < 1e-6 else 0.0

    # Bhattacharyya coefficient (更精确的重叠度量)
    bc = np.exp(-0.25 * (delta_mean / pooled_std)**2)

    return bc


def compute_separation_matrix(summary, feature='offset'):
    """
    计算20x20的分离矩阵
    值越大表示越难区分（重叠度高）
    """
    n = len(STANDARD_AA)
    overlap_matrix = np.zeros((n, n))

    for i, aa1 in enumerate(STANDARD_AA):
        for j, aa2 in enumerate(STANDARD_AA):
            if aa1 not in summary or aa2 not in summary:
                overlap_matrix[i, j] = np.nan
                continue

            if i == j:
                overlap_matrix[i, j] = 1.0  # 自己和自己完全重叠
                continue

            mean1 = summary[aa1][f'{feature}_mean']
            std1 = summary[aa1][f'{feature}_std']
            mean2 = summary[aa2][f'{feature}_mean']
            std2 = summary[aa2][f'{feature}_std']

            overlap = compute_overlap_coefficient(mean1, std1, mean2, std2)
            overlap_matrix[i, j] = overlap

    return overlap_matrix


def analyze_confusion_pairs(summary, feature='offset', threshold=0.5):
    """
    找出容易混淆的氨基酸对（重叠度 > threshold）
    """
    confusion_pairs = []

    for i, aa1 in enumerate(STANDARD_AA):
        for j, aa2 in enumerate(STANDARD_AA):
            if i >= j:  # 只看上三角
                continue

            if aa1 not in summary or aa2 not in summary:
                continue

            mean1 = summary[aa1][f'{feature}_mean']
            std1 = summary[aa1][f'{feature}_std']
            mean2 = summary[aa2][f'{feature}_mean']
            std2 = summary[aa2][f'{feature}_std']

            overlap = compute_overlap_coefficient(mean1, std1, mean2, std2)

            if overlap > threshold:
                confusion_pairs.append((aa1, aa2, overlap, mean1, std1, mean2, std2))

    # 按重叠度排序
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    return confusion_pairs


def plot_overlap_heatmap(overlap_matrix, title, save_path):
    """绘制重叠度热图"""
    plt.figure(figsize=(12, 10))

    # 使用mask处理NaN
    mask = np.isnan(overlap_matrix)

    sns.heatmap(overlap_matrix,
                xticklabels=STANDARD_AA,
                yticklabels=STANDARD_AA,
                cmap='RdYlGn_r',  # 红=高重叠（差），绿=低重叠（好）
                vmin=0, vmax=1,
                annot=False,
                fmt='.2f',
                cbar_kws={'label': 'Overlap Coefficient'},
                mask=mask)

    plt.title(title)
    plt.xlabel('Amino Acid')
    plt.ylabel('Amino Acid')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved heatmap to {save_path}")


def compute_average_separability(overlap_matrix):
    """
    计算平均可分离性
    返回: 1 - 平均重叠度
    """
    # 只看上三角（排除对角线）
    n = overlap_matrix.shape[0]
    upper_tri = []

    for i in range(n):
        for j in range(i+1, n):
            if not np.isnan(overlap_matrix[i, j]):
                upper_tri.append(overlap_matrix[i, j])

    avg_overlap = np.mean(upper_tri)
    avg_separability = 1 - avg_overlap

    return avg_separability, avg_overlap


def plot_distribution_comparison(summary, aa_list, feature='offset', save_path='distribution_comparison.png'):
    """
    绘制几个典型氨基酸的分布对比
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x_range = np.linspace(0, 6, 1000)

    for aa in aa_list:
        if aa not in summary:
            continue

        mean = summary[aa][f'{feature}_mean']
        std = summary[aa][f'{feature}_std']

        # 高斯分布
        from scipy.stats import norm
        y = norm.pdf(x_range, mean, std)

        ax.plot(x_range, y, label=f'{aa} ({mean:.2f}±{std:.2f})', linewidth=2)
        ax.fill_between(x_range, 0, y, alpha=0.2)

    ax.set_xlabel(f'{feature.capitalize()} (Å)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Distribution Comparison: {feature.capitalize()}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved distribution comparison to {save_path}")


if __name__ == "__main__":
    import sys
    import os

    data_dir = "/media/junyu/DATA/mmcif/gzipmmcif"

    print("Loading database statistics...")
    stats_sc, stats_all = batch_analyze_database(data_dir, max_files=100)

    summary_sc = compute_statistics(stats_sc)
    summary_all = compute_statistics(stats_all)

    print("\n" + "="*80)
    print("重叠度分析 (Overlap Analysis)")
    print("="*80)

    # ===== Sidechain Only: Offset =====
    print("\n--- Sidechain Only: Offset ---")
    overlap_sc_offset = compute_separation_matrix(summary_sc, 'offset')
    avg_sep, avg_ovl = compute_average_separability(overlap_sc_offset)
    print(f"平均重叠度: {avg_ovl:.3f}")
    print(f"平均可分离性: {avg_sep:.3f}")

    confusion_sc_offset = analyze_confusion_pairs(summary_sc, 'offset', threshold=0.3)
    print(f"\n高重叠对 (overlap > 0.3): {len(confusion_sc_offset)} 对")
    print("\n最容易混淆的前10对:")
    for aa1, aa2, ovl, m1, s1, m2, s2 in confusion_sc_offset[:10]:
        print(f"  {aa1} vs {aa2}: overlap={ovl:.3f}, "
              f"{aa1}({m1:.2f}±{s1:.2f}) vs {aa2}({m2:.2f}±{s2:.2f})")

    plot_overlap_heatmap(overlap_sc_offset,
                         'Sidechain Only: Offset Overlap Matrix',
                         'overlap_sc_offset.png')

    # ===== Sidechain Only: Scale =====
    print("\n--- Sidechain Only: Scale ---")
    overlap_sc_scale = compute_separation_matrix(summary_sc, 'scale')
    avg_sep, avg_ovl = compute_average_separability(overlap_sc_scale)
    print(f"平均重叠度: {avg_ovl:.3f}")
    print(f"平均可分离性: {avg_sep:.3f}")

    confusion_sc_scale = analyze_confusion_pairs(summary_sc, 'scale', threshold=0.3)
    print(f"\n高重叠对 (overlap > 0.3): {len(confusion_sc_scale)} 对")
    print("\n最容易混淆的前10对:")
    for aa1, aa2, ovl, m1, s1, m2, s2 in confusion_sc_scale[:10]:
        print(f"  {aa1} vs {aa2}: overlap={ovl:.3f}, "
              f"{aa1}({m1:.2f}±{s1:.2f}) vs {aa2}({m2:.2f}±{s2:.2f})")

    plot_overlap_heatmap(overlap_sc_scale,
                         'Sidechain Only: Scale Overlap Matrix',
                         'overlap_sc_scale.png')

    # ===== All Atoms: Offset =====
    print("\n--- All Atoms: Offset ---")
    overlap_all_offset = compute_separation_matrix(summary_all, 'offset')
    avg_sep, avg_ovl = compute_average_separability(overlap_all_offset)
    print(f"平均重叠度: {avg_ovl:.3f}")
    print(f"平均可分离性: {avg_sep:.3f}")

    # ===== All Atoms: Scale =====
    print("\n--- All Atoms: Scale ---")
    overlap_all_scale = compute_separation_matrix(summary_all, 'scale')
    avg_sep, avg_ovl = compute_average_separability(overlap_all_scale)
    print(f"平均重叠度: {avg_ovl:.3f}")
    print(f"平均可分离性: {avg_sep:.3f}")

    # 绘制典型氨基酸分布对比
    print("\n绘制分布对比图...")
    plot_distribution_comparison(summary_sc,
                                 ['GLY', 'ALA', 'LEU', 'ARG', 'TRP'],
                                 'offset',
                                 'sc_offset_distributions.png')

    plot_distribution_comparison(summary_sc,
                                 ['GLY', 'SER', 'LEU', 'LYS', 'TRP'],
                                 'scale',
                                 'sc_scale_distributions.png')

    print("\n分析完成!")
