import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_mse_analysis(mse, xi, node_mask):
    """
    统计并绘制 xi 中包含 1 的位置 vs 不包含 1 的位置的 MSE 区别。
    """
    # 1. 数据准备：转为 numpy 并展平
    # 确保张量在 CPU 上
    mse_np = mse.detach().cpu().numpy()
    xi_np = xi.detach().cpu().numpy()
    mask_np = node_mask.detach().cpu().numpy().astype(bool)

    # 2. 处理 xi 的维度
    # 如果 xi 的维度比 mse 多（例如 (B, N, K) vs (B, N)），我们需要判断在该节点是否"包含 1"
    # 这里假设只要该节点对应的 xi 向量中存在 1，即视为"包含 1"
    if xi_np.ndim > mse_np.ndim:
        # 对多出的维度求 any，判断是否存在 1
        # 假设前两个维度是 (Batch, Node)，后续是特征维度
        # 先生成布尔矩阵
        is_one = (xi_np == 1)
        # 沿着后续维度折叠，只要有一个 1 就算 True
        dims_to_reduce = tuple(range(mse_np.ndim, xi_np.ndim))
        xi_has_1 = is_one.any(axis=dims_to_reduce)
    else:
        # 维度一致，直接比较
        xi_has_1 = (xi_np == 1)

    # 3. 筛选有效数据 (应用 node_mask)
    # 只取 node_mask 为 True 的位置
    valid_mse = mse_np[mask_np]
    valid_xi_flag = xi_has_1[mask_np]

    # 4. 分组
    mse_with_1 = valid_mse[valid_xi_flag]  # xi 包含 1 的位置的 MSE
    mse_without_1 = valid_mse[~valid_xi_flag]  # xi 不含 1 的位置的 MSE

    # 5. 打印统计信息
    print(f"Stats for MSE where xi contains 1 (Count: {len(mse_with_1)}):")
    if len(mse_with_1) > 0:
        print(f"  Mean: {np.mean(mse_with_1):.4f}, Std: {np.std(mse_with_1):.4f}, Max: {np.max(mse_with_1):.4f}")
    else:
        print("  No positions found with xi == 1.")

    print(f"\nStats for MSE where xi does NOT contain 1 (Count: {len(mse_without_1)}):")
    if len(mse_without_1) > 0:
        print(
            f"  Mean: {np.mean(mse_without_1):.4f}, Std: {np.std(mse_without_1):.4f}, Max: {np.max(mse_without_1):.4f}")
    else:
        print("  No positions found without xi == 1.")

    # 6. 绘图
    plt.figure(figsize=(12, 5))

    # 子图 1: 箱线图对比
    plt.subplot(1, 2, 1)
    data_to_plot = [d for d in [mse_with_1, mse_without_1] if len(d) > 0]
    labels = [l for d, l in zip([mse_with_1, mse_without_1], ['xi has 1', 'xi no 1']) if len(d) > 0]

    if data_to_plot:
        plt.boxplot(data_to_plot, labels=labels, showfliers=False)  # showfliers=False 忽略异常值以便看清主体分布
        plt.title('MSE Distribution Comparison (Boxplot)')
        plt.ylabel('MSE')
        plt.grid(True, linestyle='--', alpha=0.6)

    # 子图 2: 直方图分布
    plt.subplot(1, 2, 2)
    if len(mse_with_1) > 0:
        plt.hist(mse_with_1, bins=50, alpha=0.6, label='xi has 1', density=True, log=True)
    if len(mse_without_1) > 0:
        plt.hist(mse_without_1, bins=50, alpha=0.6, label='xi no 1', density=True, log=True)

    plt.title('MSE Histogram (Log Scale)')
    plt.xlabel('MSE')
    plt.ylabel('Density (Log)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('mse_xi_comparison.png')
    print("\nPlot saved as 'mse_xi_comparison.png'")
    # plt.show() # 如果在 notebook 中运行可以取消注释


# --- 调用函数 ---
# 假设 mse, xi, node_mask 已经在你的上下文中定义
