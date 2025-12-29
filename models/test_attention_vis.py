"""
测试 IGA Attention 可视化功能

使用方法:
1. 在训练时启用可视化: 修改 flow_model.py 中 IGA 初始化
2. 或者使用这个脚本手动测试可视化
"""

import torch
import numpy as np
from visualize_attention import visualize_iga_attention
from data.rigids import Rigid

def create_test_data(batch_size=2, num_heads=8, num_res=100):
    """创建测试数据"""
    # 模拟 geometric bias (应该是负值)
    attn_bias_geo = -torch.rand(batch_size, num_heads, num_res, num_res) * 5.0
    # 对角线和近邻更负(overlap更大)
    for i in range(num_res):
        for j in range(max(0, i-5), min(num_res, i+6)):
            attn_bias_geo[:, :, i, j] -= 2.0

    # Gamma (learnable weights)
    gamma = torch.rand(1, num_heads, 1, 1) * 2.0

    # Logits before and after
    logits_before = torch.randn(batch_size, num_heads, num_res, num_res)
    logits_after = logits_before + gamma * attn_bias_geo

    # Attention weights (softmax后)
    weights = torch.softmax(logits_after, dim=-1)

    # CA 坐标 (模拟一条chain)
    rigid_trans = torch.zeros(batch_size, num_res, 3)
    for i in range(num_res):
        # 沿着螺旋线
        angle = i * 0.1
        rigid_trans[:, i, 0] = i * 3.8  # X轴方向
        rigid_trans[:, i, 1] = 5 * np.cos(angle)
        rigid_trans[:, i, 2] = 5 * np.sin(angle)

    return attn_bias_geo, gamma, logits_before, logits_after, weights, rigid_trans


def test_visualization():
    """测试可视化功能"""
    print("Creating test data...")
    attn_bias_geo, gamma, logits_before, logits_after, weights, rigid_trans = create_test_data()

    print("Running visualization...")
    try:
        stats = visualize_iga_attention(
            attn_bias_geo=attn_bias_geo,
            gamma=gamma,
            logits_before=logits_before,
            logits_after=logits_after,
            weights=weights,
            rigid_trans=rigid_trans,
            save_dir="./test_attention_vis",
            batch_idx=0,
            head_idx=0,
            num_vis_res=50,
            layer_idx=0
        )

        print("\n✅ Visualization successful!")
        print("\nStatistics:")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}")

    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_visualization()
