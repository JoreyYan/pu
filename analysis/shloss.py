import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


def visualize_noise_loss_relationship(t_tensor, loss_tensor, save_path=None):
    """
    可视化噪音水平t与loss之间的关系

    Args:
        t_tensor: torch.Tensor, shape [B], 噪音水平 (0~1)
        loss_tensor: torch.Tensor, shape [B], 对应的loss值
        save_path: str, 保存图片的路径（可选）
    """

    # 转换为numpy数组便于处理
    t_np = t_tensor.detach().cpu().numpy()
    loss_np = loss_tensor.detach().cpu().numpy()

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Noise Level t vs Loss Relationship Analysis', fontsize=16)

    # 1. 散点图 - 基础关系
    ax1 = axes[0, 0]
    scatter = ax1.scatter(t_np, loss_np, alpha=0.6, s=20, c=t_np, cmap='viridis')
    ax1.set_xlabel('Noise Level t')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Noise Level (Scatter Plot)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Noise Level')

    # 添加趋势线
    z = np.polyfit(t_np, loss_np, 1)
    p = np.poly1d(z)
    ax1.plot(t_np, p(t_np), "r--", alpha=0.8, linewidth=2, label=f'Trend Line (slope:{z[0]:.3f})')
    ax1.legend()

    # 2. 分桶统计 - 看不同t区间的loss分布
    ax2 = axes[0, 1]
    n_bins = 10
    t_bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (t_bins[:-1] + t_bins[1:]) / 2

    # 计算每个bin的统计量
    bin_means = []
    bin_stds = []
    bin_counts = []

    for i in range(n_bins):
        mask = (t_np >= t_bins[i]) & (t_np < t_bins[i + 1])
        if i == n_bins - 1:  # 最后一个bin包含右边界
            mask = (t_np >= t_bins[i]) & (t_np <= t_bins[i + 1])

        if mask.sum() > 0:
            bin_means.append(loss_np[mask].mean())
            bin_stds.append(loss_np[mask].std())
            bin_counts.append(mask.sum())
        else:
            bin_means.append(0)
            bin_stds.append(0)
            bin_counts.append(0)

    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    bin_counts = np.array(bin_counts)

    # 绘制误差条形图
    ax2.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=5, capthick=2)
    ax2.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds, alpha=0.3)
    ax2.set_xlabel('Noise Level t (Binned)')
    ax2.set_ylabel('Loss (Mean ± Std)')
    ax2.set_title('Loss Statistics at Different Noise Levels')
    ax2.grid(True, alpha=0.3)

    # 在图上标注样本数量
    for i, (x, y, count) in enumerate(zip(bin_centers, bin_means, bin_counts)):
        if count > 0:
            ax2.annotate(f'n={count}', (x, y), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=8)

    # 3. 密度热力图
    ax3 = axes[1, 0]
    # 创建2D直方图
    h, xedges, yedges = np.histogram2d(t_np, loss_np, bins=20)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax3.imshow(h.T, extent=extent, origin='lower', aspect='auto', cmap='YlOrRd')
    ax3.set_xlabel('Noise Level t')
    ax3.set_ylabel('Loss')
    ax3.set_title('Loss-Noise Level Density Distribution')
    plt.colorbar(im, ax=ax3, label='Sample Density')

    # 4. 分布对比 - 高噪音vs低噪音
    ax4 = axes[1, 1]

    # 分为低噪音和高噪音两组
    threshold = 0.5
    low_noise_mask = t_np < threshold
    high_noise_mask = t_np >= threshold

    if low_noise_mask.sum() > 0 and high_noise_mask.sum() > 0:
        low_noise_loss = loss_np[low_noise_mask]
        high_noise_loss = loss_np[high_noise_mask]

        # 绘制分布
        ax4.hist(low_noise_loss, bins=30, alpha=0.7, label=f'Low Noise (t<{threshold})', density=True)
        ax4.hist(high_noise_loss, bins=30, alpha=0.7, label=f'High Noise (t≥{threshold})', density=True)

        # 统计检验
        statistic, p_value = stats.mannwhitneyu(low_noise_loss, high_noise_loss, alternative='two-sided')

        ax4.set_xlabel('Loss')
        ax4.set_ylabel('Density')
        ax4.set_title(f'High vs Low Noise Loss Distribution\n(Mann-Whitney U test p={p_value:.4f})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 添加统计信息
        info_text = f"Low Noise: μ={low_noise_loss.mean():.4f}, σ={low_noise_loss.std():.4f}\n"
        info_text += f"High Noise: μ={high_noise_loss.mean():.4f}, σ={high_noise_loss.std():.4f}"
        ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

    # 打印一些统计信息
    correlation = np.corrcoef(t_np, loss_np)[0, 1]
    print(f"\nStatistical Information:")
    print(f"Sample size: {len(t_np)}")
    print(f"Noise level range: [{t_np.min():.4f}, {t_np.max():.4f}]")
    print(f"Loss range: [{loss_np.min():.4f}, {loss_np.max():.4f}]")
    print(f"Correlation coefficient: {correlation:.4f}")

    return correlation


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    B = 1000
    t_example = torch.rand(B)  # 噪音水平 0~1
    # 模拟一个与噪音相关的loss (这里只是示例)
    loss_example = 0.5 + 0.3 * t_example + 0.1 * torch.randn(B)

    print("Visualizing with example data...")
    correlation = visualize_noise_loss_relationship(t_example, loss_example)

    print(f"\nUsing your actual data:")
    print("correlation = visualize_noise_loss_relationship(your_t_tensor, your_loss_tensor)")