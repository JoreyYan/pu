import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from models.utils import fbb_backbone_loss
from scipy.spatial.transform import Rotation as R


# 生成测试数据的辅助函数
def euler_to_rotation_matrix(roll, pitch, yaw):
    """欧拉角转旋转矩阵"""
    r = R.from_euler('xyz', [roll, pitch, yaw])
    return torch.tensor(r.as_matrix(), dtype=torch.float32)


def generate_test_data(batch_size=4, num_frames=10):
    """生成测试数据"""
    # GT数据 - 作为基准
    gt_trans = torch.randn(batch_size, num_frames, 3) * 5  # 随机平移
    gt_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_frames, 1, 1)

    # 添加小的随机旋转到GT
    for b in range(batch_size):
        for f in range(num_frames):
            angles = torch.randn(3) * 0.1  # 小角度旋转
            gt_rot[b, f] = euler_to_rotation_matrix(angles[0], angles[1], angles[2])

    # 生成不同程度偏差的预测数据
    test_cases = []

    # Case 1: 完全相同 (loss应该接近0)
    pred_trans_1 = gt_trans.clone()
    pred_rot_1 = gt_rot.clone()
    test_cases.append(("Perfect Match", pred_trans_1, pred_rot_1))

    # Case 2: 小的平移误差
    pred_trans_2 = gt_trans + torch.randn_like(gt_trans) * 0.5
    pred_rot_2 = gt_rot.clone()
    test_cases.append(("Small Translation Error", pred_trans_2, pred_rot_2))

    # Case 3: 中等平移误差
    pred_trans_3 = gt_trans + torch.randn_like(gt_trans) * 2.0
    pred_rot_3 = gt_rot.clone()
    test_cases.append(("Medium Translation Error", pred_trans_3, pred_rot_3))

    # Case 4: 大的平移误差
    pred_trans_4 = gt_trans + torch.randn_like(gt_trans) * 5.0
    pred_rot_4 = gt_rot.clone()
    test_cases.append(("Large Translation Error", pred_trans_4, pred_rot_4))

    # Case 5: 小的旋转误差
    pred_trans_5 = gt_trans.clone()
    pred_rot_5 = gt_rot.clone()
    for b in range(batch_size):
        for f in range(num_frames):
            angles = torch.randn(3) * 0.2  # 小角度偏差
            rotation_error = euler_to_rotation_matrix(angles[0], angles[1], angles[2])
            pred_rot_5[b, f] = torch.matmul(pred_rot_5[b, f], rotation_error)
    test_cases.append(("Small Rotation Error", pred_trans_5, pred_rot_5))

    # Case 6: 大的旋转误差
    pred_trans_6 = gt_trans.clone()
    pred_rot_6 = gt_rot.clone()
    for b in range(batch_size):
        for f in range(num_frames):
            angles = torch.randn(3) * 0.8  # 大角度偏差
            rotation_error = euler_to_rotation_matrix(angles[0], angles[1], angles[2])
            pred_rot_6[b, f] = torch.matmul(pred_rot_6[b, f], rotation_error)
    test_cases.append(("Large Rotation Error", pred_trans_6, pred_rot_6))

    # Case 7: 平移和旋转都有误差
    pred_trans_7 = gt_trans + torch.randn_like(gt_trans) * 3.0
    pred_rot_7 = gt_rot.clone()
    for b in range(batch_size):
        for f in range(num_frames):
            angles = torch.randn(3) * 0.5
            rotation_error = euler_to_rotation_matrix(angles[0], angles[1], angles[2])
            pred_rot_7[b, f] = torch.matmul(pred_rot_7[b, f], rotation_error)
    test_cases.append(("Both Translation & Rotation Error", pred_trans_7, pred_rot_7))

    return gt_trans, gt_rot, test_cases


# 主要评估函数
print("FAPE Loss 评估测试")
print("=" * 50)

# 生成测试数据
batch_size = 4
num_frames = 10
gt_trans, gt_rot, test_cases = generate_test_data(batch_size, num_frames)

# 创建mask (全部有效)
mask = torch.ones(batch_size, num_frames)

results = []

for case_name, pred_trans, pred_rot in test_cases:
    # 计算FAPE loss
    fape_loss = fbb_backbone_loss(
        pred_trans=pred_trans,
        pred_rot=pred_rot,
        gt_trans=gt_trans,
        gt_rot=gt_rot,
        mask=mask,
        clamp_distance=10.0,
        loss_unit_distance=10.0,
        eps=1e-4
    )

    # 计算平均loss
    avg_loss = torch.mean(fape_loss).item()
    results.append((case_name, avg_loss))

    print(f"{case_name:30s}: {avg_loss:.6f}")

    # 计算基础统计信息
    trans_diff = torch.mean(torch.norm(pred_trans - gt_trans, dim=-1)).item()

    # 计算旋转差异 (Frobenius norm)
    rot_diff = torch.mean(torch.norm(pred_rot - gt_rot, dim=(-2, -1))).item()

    print(f"{'   Translation L2 Diff':30s}: {trans_diff:.6f}")
    print(f"{'   Rotation Frobenius Diff':30s}: {rot_diff:.6f}")
    print("-" * 50)

# 绘制结果
plt.figure(figsize=(12, 8))

case_names = [r[0] for r in results]
losses = [r[1] for r in results]

plt.subplot(2, 1, 1)
plt.bar(range(len(case_names)), losses)
plt.xticks(range(len(case_names)), case_names, rotation=45, ha='right')
plt.ylabel('FAPE Loss')
plt.title('FAPE Loss vs Different Error Types')
plt.yscale('log')  # 使用对数尺度更好地显示差异

# 显示数值
for i, loss in enumerate(losses):
    plt.text(i, loss, f'{loss:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 分析结果
print("\n结果分析:")
print("=" * 50)

# 检查是否满足单调性
if losses[0] < min(losses[1:]):  # Perfect match应该有最小loss
    print("✓ Perfect match有最小的loss")
else:
    print("✗ Perfect match没有最小的loss - 可能有问题!")

# 检查loss是否随误差增大而增大
trans_cases = [(case_names[1], losses[1]), (case_names[2], losses[2]), (case_names[3], losses[3])]
if losses[1] < losses[2] < losses[3]:  # 平移误差递增
    print("✓ 平移误差越大，loss越大")
else:
    print("✗ 平移误差与loss不成正比")

print(f"\nLoss范围: {min(losses):.6f} - {max(losses):.6f}")
print(f"Loss动态范围: {max(losses) / min(losses[1:]):.2f}x")  # 排除perfect match

print("\n✅ 评估完成! 如果看到上面显示'平移误差越大，loss越大'，")
print("   说明你的FAPE loss函数能够正确反映数据之间的差异程度。")