"""
IGA Attention 可视化工具

用于可视化 InvariantGaussianAttention 的 attention 矩阵，
特别是 gamma * attn_bias_geo 这一项的影响。
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import numpy as np
from pathlib import Path


def visualize_iga_attention(
    scalar_qk,          # [B, H, N, N] 纯标量QK logits (已乘gamma_scalar)
    pair_bias,          # [B, H, N, N] pair bias (已乘gamma_pair, 可能为None)
    attn_bias_geo,      # [B, H, N, N] 几何bias (负值，未乘gamma)
    gamma_scalar,       # [1, H, 1, 1] scalar QK的gamma
    gamma_pair,         # [1, H, 1, 1] pair的gamma (可能为None)
    gamma_geo,          # [1, H, 1, 1] geo的gamma
    logits_before,      # [B, H, N, N] scalar + pair
    logits_after,       # [B, H, N, N] scalar + pair + gamma_geo*geo
    weights,            # [B, H, N, N] softmax后的attention权重
    rigid_trans,        # [B, N, 3] CA坐标 (用于计算距离)
    save_dir="./attention_vis",
    batch_idx=0,
    head_idx=0,
    num_vis_res=50,     # 可视化前N个residue
    layer_idx=0         # IGA layer index
):
    """
    可视化IGA Attention的各个组件：
    ① Scalar QK (γ_s × QK)
    ② Pair Bias (γ_p × pair)
    ③ Geometric Bias (γ_g × geo)
    ④ Scalar + Pair (① + ②)
    ⑤ Final Logits (① + ② + ③)

    Args:
        scalar_qk: 纯标量QK attention logits (已加权)
        pair_bias: pair bias (已加权, 可能为None)
        attn_bias_geo: 几何bias (overlap score)，负值
        gamma_scalar: scalar QK的gamma权重
        gamma_pair: pair的gamma权重 (可能为None)
        gamma_geo: geo的gamma权重
        logits_before: scalar + pair
        logits_after: scalar + pair + gamma*geo
        weights: softmax后的attention权重
        rigid_trans: CA坐标，用于计算真实距离
        save_dir: 保存目录
        batch_idx: 可视化哪个batch
        head_idx: 可视化哪个head
        num_vis_res: 可视化多少个residue
        layer_idx: IGA layer index
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    B, H, N, _ = attn_bias_geo.shape
    b, h = batch_idx, head_idx
    num_vis_res = min(num_vis_res, N)

    # 边界检查
    if num_vis_res < 2:
        print(f"[IGA Vis] Warning: num_vis_res={num_vis_res} < 2, skipping visualization")
        return {
            'gamma': 0.0, 'geo_bias_mean': 0.0, 'geo_bias_std': 0.0,
            'local_attn_mean': 0.0, 'global_attn_mean': 0.0,
            'local_geo_bias_mean': 0.0, 'global_geo_bias_mean': 0.0,
        }

    # Extract gamma values
    gamma_s_val = gamma_scalar[0, h, 0, 0].item()
    gamma_p_val = gamma_pair[0, h, 0, 0].item() if gamma_pair is not None else 0.0
    gamma_g_val = gamma_geo[0, h, 0, 0].item()

    # Extract all components
    qk_scalar = scalar_qk[b, h, :num_vis_res, :num_vis_res].detach().cpu().numpy() if scalar_qk is not None else np.zeros((num_vis_res, num_vis_res))
    pair = pair_bias[b, h, :num_vis_res, :num_vis_res].detach().cpu().numpy() if pair_bias is not None else np.zeros((num_vis_res, num_vis_res))
    geo_raw = attn_bias_geo[b, h, :num_vis_res, :num_vis_res].detach().cpu().numpy()
    geo_weighted = gamma_g_val * geo_raw  # gamma_geo * geo
    logits_sp = logits_before[b, h, :num_vis_res, :num_vis_res].detach().cpu().numpy()  # scalar + pair
    logits_final = logits_after[b, h, :num_vis_res, :num_vis_res].detach().cpu().numpy()  # scalar + pair + gamma*geo
    attn_weights = weights[b, h, :num_vis_res, :num_vis_res].detach().cpu().numpy()

    # 计算CA距离矩阵
    ca_coords = rigid_trans[b, :num_vis_res, :].detach().cpu().numpy()  # [N, 3]
    dist_matrix = np.linalg.norm(
        ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1
    )  # [N, N]

    # 处理NaN和Inf
    qk_scalar = np.nan_to_num(qk_scalar, nan=0.0)
    pair = np.nan_to_num(pair, nan=0.0)
    geo_raw = np.nan_to_num(geo_raw, nan=0.0, posinf=0.0, neginf=-100.0)
    geo_weighted = np.nan_to_num(geo_weighted, nan=0.0, posinf=0.0, neginf=-100.0)
    logits_sp = np.nan_to_num(logits_sp, nan=0.0)
    logits_final = np.nan_to_num(logits_final, nan=0.0)
    attn_weights = np.nan_to_num(attn_weights, nan=0.0)
    dist_matrix = np.nan_to_num(dist_matrix, nan=0.0)

    # =========================================================================
    # 主图: 展示 Attention 组件分解 (2x3)
    # Row 0: Scalar QK → Pair → Geometric (组件层面)
    # Row 1: Scalar+Pair → Final → Attention Weights (组合层面)
    # =========================================================================
    def plot_heatmap(ax, data, title, xlabel='Key Residue', ylabel='Query Residue'):
        """通用热图绘制函数，自动处理colormap"""
        vmin, vmax = float(data.min()), float(data.max())

        if abs(vmin) < 1e-6 and abs(vmax) < 1e-6:
            im = ax.imshow(data, cmap='viridis', vmin=-0.01, vmax=0.01, aspect='auto')
        elif vmax <= 0:
            # 全负值：Blues_r
            im = ax.imshow(data, cmap='Blues_r', vmin=vmin, vmax=vmax, aspect='auto')
        elif vmin >= 0:
            # 全正值：Reds
            im = ax.imshow(data, cmap='Reds', vmin=vmin, vmax=vmax, aspect='auto')
        else:
            # 有正有负：RdBu_r diverging
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            im = ax.imshow(data, cmap='RdBu_r', norm=norm, aspect='auto')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return im

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'IGA Attention Decomposition (Layer {layer_idx}, Batch {b}, Head {h})\n'
                 f'γ_scalar={gamma_s_val:.4f}, γ_pair={gamma_p_val:.4f}, γ_geo={gamma_g_val:.4f}',
                 fontsize=16, fontweight='bold')

    # Row 0: 三个基础组件
    # --- ① Scalar QK ---
    im1 = plot_heatmap(axes[0, 0], qk_scalar, f'① Scalar Q·K (γ_s={gamma_s_val:.3f})')
    plt.colorbar(im1, ax=axes[0, 0], label='γ_s × QK')

    # --- ② Pair Bias ---
    im2 = plot_heatmap(axes[0, 1], pair, f'② Pair Bias (γ_p={gamma_p_val:.3f})')
    plt.colorbar(im2, ax=axes[0, 1], label='γ_p × Pair')

    # --- ③ Geometric Bias (gamma * geo) ---
    im3 = plot_heatmap(axes[0, 2], geo_weighted, f'③ Geometric (γ_g={gamma_g_val:.3f})')
    plt.colorbar(im3, ax=axes[0, 2], label='γ_g × Geo')

    # Row 1: 三个组合阶段
    # --- ④ Scalar + Pair ---
    im4 = plot_heatmap(axes[1, 0], logits_sp, '④ Scalar + Pair\n(① + ②)')
    plt.colorbar(im4, ax=axes[1, 0], label='Logits')

    # --- ⑤ Final Logits ---
    im5 = plot_heatmap(axes[1, 1], logits_final, '⑤ Final Logits\n(① + ② + ③)')
    plt.colorbar(im5, ax=axes[1, 1], label='Logits')

    # --- ⑥ Attention Weights (After Softmax) ---
    im6 = axes[1, 2].imshow(attn_weights, cmap='viridis', aspect='auto', vmin=0, vmax=attn_weights.max())
    axes[1, 2].set_title('⑥ Attention Weights\n(After Softmax)')
    axes[1, 2].set_xlabel('Key Residue')
    axes[1, 2].set_ylabel('Query Residue')
    plt.colorbar(im6, ax=axes[1, 2], label='Attention Weight')

    plt.tight_layout()
    plt.savefig(save_dir / f'layer{layer_idx}_decomposition_b{b}_h{h}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 图2: 距离 vs Attention 散点图
    # =========================================================================
    fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig2.suptitle(f'Distance vs Attention Analysis (Layer {layer_idx}, Batch {b}, Head {h})', fontsize=14)

    # 取上三角（不包括对角线）
    triu_indices = np.triu_indices(num_vis_res, k=1)
    dist_flat = dist_matrix[triu_indices]
    attn_flat = attn_weights[triu_indices]
    geo_flat = geo_weighted[triu_indices]  # 使用 gamma-weighted geo

    # --- Scatter: Distance vs Attention (colored by geo) ---
    geo_flat_min, geo_flat_max = float(geo_flat.min()), float(geo_flat.max())
    if geo_flat_max <= 0:
        scatter = ax.scatter(dist_flat, attn_flat, c=geo_flat, cmap='Blues_r',
                            alpha=0.6, s=10, vmin=geo_flat_min, vmax=geo_flat_max)
    elif geo_flat_min >= 0:
        scatter = ax.scatter(dist_flat, attn_flat, c=geo_flat, cmap='Reds',
                            alpha=0.6, s=10, vmin=geo_flat_min, vmax=geo_flat_max)
    else:
        scatter = ax.scatter(dist_flat, attn_flat, c=geo_flat, cmap='RdBu_r',
                            alpha=0.6, s=10,
                            norm=TwoSlopeNorm(vmin=geo_flat_min, vcenter=0, vmax=geo_flat_max))
    ax.set_xlabel('CA Distance (nm)')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Attn vs Distance (Color = γ×Geo Bias)')
    ax.set_xlim(0, max(dist_flat.max(), 1.0))
    ax.set_ylim(0, max(attn_flat.max() * 1.1, 0.01))
    plt.colorbar(scatter, ax=ax, label='Geo Bias')

    # 添加局部邻域阈值线
    ax.axvline(x=8.0, color='red', linestyle='--', linewidth=1.5, label='Local (8nm)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_dir / f'layer{layer_idx}_attention_overview_b{b}_h{h}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 图2.5: Contact Map (0.4nm cutoff)
    # =========================================================================
    fig_contact, ax_contact = plt.subplots(1, 1, figsize=(8, 8))
    fig_contact.suptitle(f'Contact Map (Layer {layer_idx}, Batch {b}, Head {h})\nDistance < 0.4nm', fontsize=14)

    # 创建 contact map: 距离 < 0.4nm 为 1，否则为 0
    contact_map = (dist_matrix < 0.4).astype(float)

    # 显示 contact map
    im_contact = ax_contact.imshow(contact_map, cmap='Greys', aspect='auto', vmin=0, vmax=1)
    ax_contact.set_title('Contact Map (Distance < 0.4nm)')
    ax_contact.set_xlabel('Residue Index')
    ax_contact.set_ylabel('Residue Index')
    plt.colorbar(im_contact, ax=ax_contact, label='Contact (1=yes, 0=no)')

    # 添加统计信息
    num_contacts = (contact_map.sum() - num_vis_res) / 2  # 减去对角线，除以2（对称）
    total_pairs = (num_vis_res * (num_vis_res - 1)) / 2
    contact_density = num_contacts / total_pairs if total_pairs > 0 else 0
    ax_contact.text(0.02, 0.98, f'Contacts: {int(num_contacts)}/{int(total_pairs)} ({contact_density*100:.1f}%)',
                   transform=ax_contact.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_dir / f'layer{layer_idx}_contact_map_b{b}_h{h}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 图3: 单个Query的Attention Profile (看邻点相互作用)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Single Query Attention Profiles (Batch {b}, Head {h})', fontsize=14)

    # 选择4个有代表性的query residue
    query_indices = [
        num_vis_res // 4,      # 25%
        num_vis_res // 2,      # 50%
        3 * num_vis_res // 4,  # 75%
        num_vis_res - 1        # 末端
    ]

    for idx, q_idx in enumerate(query_indices):
        ax = axes[idx // 2, idx % 2]

        # 该query对所有key的attention
        attn_profile = attn_weights[q_idx, :]       # [N]
        geo_profile = geo_weighted[q_idx, :]        # [N] gamma-weighted geo
        dist_profile = dist_matrix[q_idx, :]        # [N]

        # 双Y轴
        ax2 = ax.twinx()

        # 主轴: Attention Weights (正值，绿色)
        ax.plot(range(num_vis_res), attn_profile, 'g-', linewidth=2, label='Attention Weight')
        ax.fill_between(range(num_vis_res), 0, attn_profile, alpha=0.3, color='green')
        ax.set_ylabel('Attention Weight', color='g')
        ax.tick_params(axis='y', labelcolor='g')
        ax.set_ylim(0, attn_profile.max() * 1.2)

        # 副轴: Geometric Bias (负值，蓝色)
        ax2.plot(range(num_vis_res), geo_profile, 'b--', linewidth=2, label='Geo Bias')
        ax2.set_ylabel('Geometric Bias (Overlap)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1)

        # 标记query位置
        ax.axvline(x=q_idx, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Query {q_idx}')

        # 标记8nm局部邻域
        local_neighbors = np.where(dist_profile < 8.0)[0]
        for ln in local_neighbors:
            if ln != q_idx:
                ax.axvspan(ln - 0.5, ln + 0.5, alpha=0.1, color='orange')

        ax.set_xlabel('Key Residue Index')
        ax.set_title(f'Query Residue {q_idx}')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f'layer{layer_idx}_attention_profiles_b{b}_h{h}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 图3: 距离 vs Attention 统计 (邻点相互作用的定量分析)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Distance vs Attention Statistics (Batch {b}, Head {h})', fontsize=14)

    # --- 3.1 Binned Average Attention by Distance ---
    ax = axes[0]
    dist_bins = np.arange(0, dist_flat.max() + 1, 1)  # 1nm bins
    bin_centers = (dist_bins[:-1] + dist_bins[1:]) / 2

    binned_attn = []
    binned_std = []
    for i in range(len(dist_bins) - 1):
        mask = (dist_flat >= dist_bins[i]) & (dist_flat < dist_bins[i+1])
        if mask.sum() > 0:
            binned_attn.append(attn_flat[mask].mean())
            binned_std.append(attn_flat[mask].std())
        else:
            binned_attn.append(0)
            binned_std.append(0)

    binned_attn = np.array(binned_attn)
    binned_std = np.array(binned_std)

    ax.plot(bin_centers, binned_attn, 'o-', linewidth=2, markersize=6, color='darkblue')
    ax.fill_between(bin_centers, binned_attn - binned_std, binned_attn + binned_std,
                     alpha=0.3, color='blue')
    ax.axvline(x=8.0, color='red', linestyle='--', linewidth=2, label='Local Threshold (8nm)')
    ax.set_xlabel('CA Distance (nm)')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('Average Attention vs Distance')
    ax.legend()
    ax.grid(alpha=0.3)

    # --- 3.2 Geo Bias vs Distance ---
    ax = axes[1]
    binned_geo = []
    for i in range(len(dist_bins) - 1):
        mask = (dist_flat >= dist_bins[i]) & (dist_flat < dist_bins[i+1])
        if mask.sum() > 0:
            binned_geo.append(geo_flat[mask].mean())
        else:
            binned_geo.append(0)

    binned_geo = np.array(binned_geo)
    ax.plot(bin_centers, binned_geo, 's-', linewidth=2, markersize=6, color='darkred')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax.axvline(x=8.0, color='red', linestyle='--', linewidth=2, label='Local Threshold (8nm)')
    ax.set_xlabel('CA Distance (nm)')
    ax.set_ylabel('Mean Geometric Bias')
    ax.set_title('Geometric Bias vs Distance\n(More Negative = Stronger Interaction)')
    ax.legend()
    ax.grid(alpha=0.3)

    # --- 3.3 Local vs Global Attention Distribution ---
    ax = axes[2]
    local_mask = dist_flat < 8.0
    local_attn = attn_flat[local_mask]
    global_attn = attn_flat[~local_mask]

    ax.hist(local_attn, bins=50, alpha=0.6, label=f'Local (<8nm, n={local_mask.sum()})',
            color='orange', density=True)
    ax.hist(global_attn, bins=50, alpha=0.6, label=f'Global (≥8nm, n={(~local_mask).sum()})',
            color='blue', density=True)
    ax.axvline(x=local_attn.mean(), color='orange', linestyle='--', linewidth=2,
               label=f'Local Mean: {local_attn.mean():.4f}')
    ax.axvline(x=global_attn.mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Global Mean: {global_attn.mean():.4f}')
    ax.set_xlabel('Attention Weight')
    ax.set_ylabel('Density')
    ax.set_title('Attention Distribution:\nLocal vs Global')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f'layer{layer_idx}_distance_statistics_b{b}_h{h}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ [Layer {layer_idx}] Saved attention visualizations to {save_dir}")
    print(f"   - layer{layer_idx}_decomposition_b{b}_h{h}.png (2x3 component breakdown)")
    print(f"   - layer{layer_idx}_attention_overview_b{b}_h{h}.png (distance vs attention)")
    print(f"   - layer{layer_idx}_contact_map_b{b}_h{h}.png (0.4nm contact map)")
    print(f"   - layer{layer_idx}_attention_profiles_b{b}_h{h}.png (single query profiles)")
    print(f"   - layer{layer_idx}_distance_statistics_b{b}_h{h}.png (binned statistics)")

    # 返回统计信息 (处理可能的空数组)
    stats = {
        'gamma_scalar': gamma_s_val,
        'gamma_pair': gamma_p_val,
        'gamma_geo': gamma_g_val,
        'geo_bias_mean': float(geo_weighted.mean()),  # gamma-weighted geo
        'geo_bias_std': float(geo_weighted.std()),
        'local_attn_mean': float(local_attn.mean()) if len(local_attn) > 0 else 0.0,
        'global_attn_mean': float(global_attn.mean()) if len(global_attn) > 0 else 0.0,
        'local_geo_bias_mean': float(geo_flat[local_mask].mean()) if local_mask.sum() > 0 else 0.0,
        'global_geo_bias_mean': float(geo_flat[~local_mask].mean()) if (~local_mask).sum() > 0 else 0.0,
    }

    return stats


def add_visualization_hook(iga_module, save_dir="./attention_vis", vis_interval=100):
    """
    为IGA模块添加可视化hook

    Args:
        iga_module: InvariantGaussianAttention实例
        save_dir: 保存目录
        vis_interval: 每多少次forward可视化一次
    """
    iga_module._vis_counter = 0
    iga_module._vis_interval = vis_interval
    iga_module._save_dir = save_dir

    original_forward = iga_module.forward

    def forward_with_vis(s, r, mask=None):
        """包装forward，添加可视化"""
        # 先运行原始forward，但要capture中间变量
        # 这需要修改IGA.py的forward函数，添加return中间变量的选项
        # 暂时我们在forward内部添加可视化代码
        out = original_forward(s, r, mask)

        # 每vis_interval次可视化一次
        iga_module._vis_counter += 1
        if iga_module._vis_counter % iga_module._vis_interval == 0:
            # 这里需要从模块内部获取中间变量
            # 建议在IGA.py的forward函数中添加self._last_attn_stats来存储
            pass

        return out

    iga_module.forward = forward_with_vis
    print(f"✅ Added visualization hook to IGA module (interval={vis_interval})")

    return iga_module
