import torch
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
from scipy import stats
from matplotlib.patches import Rectangle


def plot_tensor_distribution(tensor, name="tensor", figsize=(15, 10), bins='auto'):
    """
    Plot distribution of all values in tensor

    Args:
        tensor: PyTorch tensor
        name: Name of tensor for title
        figsize: Figure size
        bins: Number of bins for histogram, can be number or 'auto'
    """
    # Convert tensor to numpy array and flatten
    if isinstance(tensor, torch.Tensor):
        values = tensor.detach().cpu().numpy().flatten()
    else:
        values = np.array(tensor).flatten()

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'{name} Value Distribution Analysis', fontsize=16, fontweight='bold')

    # 1. Histogram + KDE density curve
    ax1 = axes[0, 0]
    ax1.hist(values, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    # Add KDE density curve
    try:
        kde = stats.gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 200)
        ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE Density')
        ax1.legend()
    except:
        pass

    ax1.set_title('Histogram + Density Curve')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.grid(True, alpha=0.3)

    # 2. Box Plot
    ax2 = axes[0, 1]
    box_plot = ax2.boxplot(values, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightblue')
    ax2.set_title('Box Plot')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)

    # 3. Q-Q Plot (compare with normal distribution)
    ax3 = axes[0, 2]
    stats.probplot(values, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normality Test)')
    ax3.grid(True, alpha=0.3)

    # 4. Cumulative Distribution Function (CDF)
    ax4 = axes[1, 0]
    sorted_values = np.sort(values)
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    ax4.plot(sorted_values, cdf, 'b-', linewidth=2)
    ax4.set_title('Cumulative Distribution Function (CDF)')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Cumulative Probability')
    ax4.grid(True, alpha=0.3)

    # 5. Scatter plot of values (by index)
    ax5 = axes[1, 1]
    indices = np.arange(len(values))
    ax5.scatter(indices, values, alpha=0.6, s=1)
    ax5.set_title('Value Sequence Scatter Plot')
    ax5.set_xlabel('Index')
    ax5.set_ylabel('Value')
    ax5.grid(True, alpha=0.3)

    # 6. Statistics text
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Calculate statistics
    stats_text = f"""
Statistics Summary:
━━━━━━━━━━━━━━━━━━━━
📊 Basic Statistics:
  • Sample Count: {len(values):,}
  • Mean: {np.mean(values):.6f}
  • Median: {np.median(values):.6f}
  • Std Dev: {np.std(values):.6f}
  • Variance: {np.var(values):.6f}

📈 Distribution Properties:
  • Minimum: {np.min(values):.6f}
  • Maximum: {np.max(values):.6f}
  • Range: {np.ptp(values):.6f}
  • Skewness: {stats.skew(values):.6f}
  • Kurtosis: {stats.kurtosis(values):.6f}

🎯 Quantiles:
  • 25th Percentile: {np.percentile(values, 25):.6f}
  • 75th Percentile: {np.percentile(values, 75):.6f}
  • IQR: {np.percentile(values, 75) - np.percentile(values, 25):.6f}

🔍 Special Values:
  • Zero Count: {np.sum(values == 0):,}
  • Negative Count: {np.sum(values < 0):,}
  • Positive Count: {np.sum(values > 0):,}
    """

    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    plt.show()
    # return fig


def compare_tensor_distributions(tensors_dict, figsize=(15, 8)):
    """
    Compare distributions of multiple tensors

    Args:
        tensors_dict: Dictionary with tensor names as keys and tensors as values
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Multiple Tensor Distribution Comparison', fontsize=16, fontweight='bold')

    colors = plt.cm.Set3(np.linspace(0, 1, len(tensors_dict)))

    # Prepare data
    all_values = {}
    for name, tensor in tensors_dict.items():
        if isinstance(tensor, torch.Tensor):
            values = tensor.detach().cpu().numpy().flatten()
        else:
            values = np.array(tensor).flatten()
        all_values[name] = values

    # 1. Overlapping histograms
    ax1 = axes[0, 0]
    for i, (name, values) in enumerate(all_values.items()):
        ax1.hist(values, bins=50, alpha=0.6, label=name, color=colors[i], density=True)
    ax1.set_title('Overlapping Histogram Comparison')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Box plot comparison
    ax2 = axes[0, 1]
    data_for_boxplot = [values for values in all_values.values()]
    labels = list(all_values.keys())
    bp = ax2.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_title('Box Plot Comparison')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    # 3. CDF comparison
    ax3 = axes[1, 0]
    for i, (name, values) in enumerate(all_values.items()):
        sorted_values = np.sort(values)
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        ax3.plot(sorted_values, cdf, linewidth=2, label=name, color=colors[i])
    ax3.set_title('Cumulative Distribution Function Comparison')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Cumulative Probability')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Statistics comparison table
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_data = []
    for name, values in all_values.items():
        stats_data.append([
            name,
            f"{np.mean(values):.4f}",
            f"{np.std(values):.4f}",
            f"{np.min(values):.4f}",
            f"{np.max(values):.4f}",
            f"{len(values):,}"
        ])

    table_data = [['Tensor', 'Mean', 'Std Dev', 'Min', 'Max', 'Count']] + stats_data

    # Create table
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                      cellLoc='center', loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Set table style
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header
            cell.set_facecolor('#40466e')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#f1f1f2' if i % 2 == 0 else 'white')

    plt.tight_layout()
    return fig


# Usage examples
if __name__ == "__main__":
    # Create example tensors
    torch.manual_seed(42)

    # Example 1: Single tensor analysis
    density = torch.randn(1000, 512)  # Simulate a density tensor

    print("Analyzing single tensor distribution...")
    fig1 = plot_tensor_distribution(density, name="density", bins=50)
    plt.show()

    # Example 2: Multiple tensor comparison
    print("\nComparing multiple tensor distributions...")
    tensors_to_compare = {
        'Normal': torch.randn(10000),
        'Uniform': torch.rand(10000) * 2 - 1,
        'Exponential': torch.exponential(torch.ones(10000)),
        'Mixed': torch.cat([torch.randn(5000), torch.randn(5000) + 3])
    }

    fig2 = compare_tensor_distributions(tensors_to_compare)
    plt.show()

    print("\nAnalysis complete!")
    print("Usage:")
    print("1. plot_tensor_distribution(your_tensor, 'tensor_name') - Analyze single tensor")
    print("2. compare_tensor_distributions({'name1': tensor1, 'name2': tensor2}) - Compare multiple tensors")