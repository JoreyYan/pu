import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def analyze_atom_distances(tensor):
    """
    Analyze the distribution of all atom distances from origin in [B,N,14,3] tensor

    Args:
        tensor: shape [B,N,14,3] tensor representing local coordinates

    Returns:
        dict: dictionary containing distance statistics
    """
    # Ensure input is torch tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    print(f"输入张量形状: {tensor.shape}")
    B, N, num_atoms, _ = tensor.shape

    # Calculate distance from each atom to origin
    # tensor.shape: [B,N,14,3] -> distances.shape: [B,N,14]
    distances = torch.norm(tensor, dim=-1)  # L2 norm

    # Flatten all distances to 1D array for statistics
    all_distances = distances.reshape(-1)  # shape: [B*N*14]

    # Filter out possible invalid values
    valid_mask = ~torch.isnan(all_distances) & ~torch.isinf(all_distances)
    valid_distances = all_distances[valid_mask]

    print(f"总原子数: {len(all_distances)}")
    print(f"有效原子数: {len(valid_distances)}")

    # Convert to numpy for statistical analysis
    dist_np = valid_distances.cpu().numpy()

    # Calculate statistics
    stats_info = {
        'mean': np.mean(dist_np),
        'std': np.std(dist_np),
        'median': np.median(dist_np),
        'min': np.min(dist_np),
        'max': np.max(dist_np),
        'q25': np.percentile(dist_np, 25),
        'q75': np.percentile(dist_np, 75),
        'total_atoms': len(all_distances),
        'valid_atoms': len(valid_distances)
    }

    return distances, valid_distances, stats_info, dist_np


def plot_distance_distribution(dist_np, stats_info, save_path=None):
    """
    Plot distance distribution
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Atom Distance Distribution Analysis', fontsize=16, fontweight='bold')

    # 1. Histogram
    axes[0, 0].hist(dist_np, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(stats_info['mean'], color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {stats_info["mean"]:.3f}')
    axes[0, 0].axvline(stats_info['median'], color='green', linestyle='--', linewidth=2,
                       label=f'Median: {stats_info["median"]:.3f}')
    axes[0, 0].set_xlabel('Distance (Å)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Distance Distribution Histogram')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Box plot
    box_plot = axes[0, 1].boxplot(dist_np, vert=True, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    axes[0, 1].set_ylabel('Distance (Å)')
    axes[0, 1].set_title('Distance Distribution Box Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Cumulative Distribution Function (CDF)
    sorted_dist = np.sort(dist_np)
    cdf = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
    axes[1, 0].plot(sorted_dist, cdf, linewidth=2, color='purple')
    axes[1, 0].set_xlabel('Distance (Å)')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title('Cumulative Distribution Function')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Q-Q plot (normality test)
    stats.probplot(dist_np, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Test)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")

    plt.show()


def analyze_by_atom_type(distances):
    """
    Analyze distance distribution by atom type (14 atom positions)
    """
    # Assume 14 atoms order (adjust according to actual situation)
    atom_names = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'CD', 'CD1', 'CD2', 'CE', 'NZ', 'OXT']

    B, N, num_atoms = distances.shape

    # Reshape to [B*N, 14] for analysis
    reshaped_distances = distances.reshape(-1, num_atoms)  # [B*N, 14]

    atom_stats = {}

    print("\n各原子类型距离统计:")
    print("=" * 60)
    print(f"{'Atom Type':<10} {'Mean':<8} {'Std':<8} {'Median':<8} {'Min':<8} {'Max':<8}")
    print("-" * 60)

    for i in range(num_atoms):
        atom_distances = reshaped_distances[:, i]
        # Filter valid values
        valid_mask = ~torch.isnan(atom_distances) & ~torch.isinf(atom_distances)
        valid_atom_distances = atom_distances[valid_mask]

        if len(valid_atom_distances) > 0:
            atom_dist_np = valid_atom_distances.cpu().numpy()
            stats_dict = {
                'mean': np.mean(atom_dist_np),
                'std': np.std(atom_dist_np),
                'median': np.median(atom_dist_np),
                'min': np.min(atom_dist_np),
                'max': np.max(atom_dist_np)
            }
            atom_stats[atom_names[i]] = stats_dict

            print(f"{atom_names[i]:<10} {stats_dict['mean']:<8.3f} {stats_dict['std']:<8.3f} "
                  f"{stats_dict['median']:<8.3f} {stats_dict['min']:<8.3f} {stats_dict['max']:<8.3f}")

    return atom_stats


def plot_atom_type_comparison(atom_stats):
    """
    Plot distance distribution comparison for different atom types
    """
    atom_names = list(atom_stats.keys())
    means = [atom_stats[atom]['mean'] for atom in atom_names]
    stds = [atom_stats[atom]['std'] for atom in atom_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Mean comparison
    bars1 = ax1.bar(atom_names, means, alpha=0.7, color='lightcoral')
    ax1.set_xlabel('Atom Type')
    ax1.set_ylabel('Average Distance (Å)')
    ax1.set_title('Average Distance Comparison by Atom Type')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Display values on bars
    for bar, mean in zip(bars1, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{mean:.2f}', ha='center', va='bottom')

    # Standard deviation comparison
    bars2 = ax2.bar(atom_names, stds, alpha=0.7, color='lightblue')
    ax2.set_xlabel('Atom Type')
    ax2.set_ylabel('Distance Standard Deviation (Å)')
    ax2.set_title('Distance Standard Deviation Comparison by Atom Type')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Display values on bars
    for bar, std in zip(bars2, stds):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{std:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function - complete analysis workflow
    """
    # Example: create a simulated tensor for testing
    # In actual use, replace with your real data
    print("创建示例数据...")
    B, N = 2, 100  # 2 batches, 100 residues each

    # Create simulated data: CA-centered local coordinates
    torch.manual_seed(42)  # For reproducible results
    tensor = torch.randn(B, N, 14, 3) * 2.0  # Normal distribution with mean 0, std 2

    print("开始分析距离分布...")

    # 1. Overall distance analysis
    distances, valid_distances, stats_info, dist_np = analyze_atom_distances(tensor)

    # Print statistics
    print("\n整体距离统计信息:")
    print("=" * 40)
    for key, value in stats_info.items():
        if isinstance(value, (int, float)):
            if key in ['total_atoms', 'valid_atoms']:
                print(f"{key}: {value:,}")
            else:
                print(f"{key}: {value:.4f}")

    # 2. Plot distribution
    print("\n绘制分布图...")
    plot_distance_distribution(dist_np, stats_info, save_path='atom_distance_distribution.png')

    # 3. Analyze by atom type
    print("\n按原子类型分析...")
    atom_stats = analyze_by_atom_type(distances)

    # 4. Plot atom type comparison
    print("\n绘制原子类型比较图...")
    plot_atom_type_comparison(atom_stats)

    print("\n分析完成！")

    return distances, stats_info, atom_stats


# If you have actual data, use it like this:
def analyze_your_data(your_tensor):
    """
    Analyze your actual data

    Args:
        your_tensor: shape [B,N,14,3] tensor
    """
    distances, valid_distances, stats_info, dist_np = analyze_atom_distances(your_tensor)
    plot_distance_distribution(dist_np, stats_info, save_path='your_data_distribution.png')
    atom_stats = analyze_by_atom_type(distances)
    plot_atom_type_comparison(atom_stats)

    return distances, stats_info, atom_stats


if __name__ == "__main__":
    # Run example analysis
    main()

    # If you have actual data, uncomment below and replace your_tensor
    # your_tensor = torch.load('your_tensor.pt')  # or other loading method
    # analyze_your_data(your_tensor)