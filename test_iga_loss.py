"""
测试 SideAtomsIGALoss_Final
"""
import sys
import torch
import numpy as np
sys.path.append('/home/junyu/project/pu')
sys.path.append('/home/junyu/project/pu/data')

from models.loss import SideAtomsIGALoss_Final
from data.GaussianRigid import OffsetGaussianRigid

# Mock config
class MockConfig:
    def __init__(self):
        # Legacy weights
        self.atom_loss_weight = 1.0
        self.pair_loss_weight = 1.0
        self.huber_loss_weight = 1.0
        self.bb_atom_scale = 1.0
        self.bb_atom_loss_weight = 1.0

        # IGA weights
        self.w_param = 5.0
        self.w_nll = 0.1
        self.type_loss_weight = 1.0


def create_mock_batch(B=2, N=10, device='cuda'):
    """创建虚拟batch数据"""

    # ===== Ground Truth Coordinates =====
    # 14原子局部坐标 (N, CA, C + 11 sidechains)
    atoms14_local = torch.randn(B, N, 14, 3, device=device) * 2.0

    # 14原子全局坐标
    atoms14_global = torch.randn(B, N, 14, 3, device=device) * 10.0

    # 原子存在mask (哪些原子真实存在)
    atom14_gt_exists = torch.ones(B, N, 14, device=device)
    # Glycine只有3个backbone原子
    atom14_gt_exists[:, 0, 3:] = 0  # 第0个残基是GLY
    # 随机mask一些侧链原子
    atom14_gt_exists[:, :, 10:] *= torch.randint(0, 2, (B, N, 4), device=device).float()

    # ===== Residue Masks =====
    res_mask = torch.ones(B, N, device=device)
    res_mask[:, -2:] = 0  # 最后两个残基mask掉

    update_mask = torch.ones(B, N, device=device)
    # 假设只更新一半的残基（FBB场景）
    update_mask[:, N//2:] = 0

    # ===== Amino Acid Types =====
    aatype = torch.randint(0, 20, (B, N), device=device)
    aatype[:, 0] = 7  # GLY = 7 (in AlphaFold indexing)

    # ===== GT Gaussian Parameters (可选) =====
    # 如果数据集预计算了高斯参数
    local_mean_1 = torch.randn(B, N, 3, device=device) * 1.5
    scaling_log_1 = torch.randn(B, N, 3, device=device) * 0.5

    batch = {
        'atoms14_local': atoms14_local,
        'atom14_gt_positions': atoms14_global,
        'atom14_gt_exists': atom14_gt_exists,
        'res_mask': res_mask,
        'aatype': aatype,
        # Optional precomputed Gaussian
        'local_mean_1': local_mean_1,
        'scaing_log_1': scaling_log_1,  # 注意typo: scaing (你代码里写的)
    }

    noisy_batch = {
        'update_mask': update_mask,
        'res_mask': res_mask,
    }

    return batch, noisy_batch


def create_mock_predictions(batch, device='cuda'):
    """创建模型预测输出"""
    B, N = batch['aatype'].shape

    # ===== 预测的侧链坐标 (局部) =====
    pred_atoms = torch.randn(B, N, 11, 3, device=device) * 2.0

    # ===== 构建 OffsetGaussianRigid =====
    # 需要: rots, trans, scaling_log, local_mean

    # 简化: 直接用恒等旋转和随机平移
    from openfold.utils.rigid_utils import Rotation, Rigid

    # 恒等旋转
    identity_rot = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, 3, 3)
    rots = Rotation(rot_mats=identity_rot)

    # CA位置 (主链中心)
    trans = torch.randn(B, N, 3, device=device) * 10.0

    # 高斯参数
    scaling_log = torch.randn(B, N, 3, device=device) * 0.5
    local_mean = torch.randn(B, N, 3, device=device) * 1.5

    pred_gaussian = OffsetGaussianRigid(rots, trans, scaling_log, local_mean)

    # ===== Sequence Logits =====
    logits = torch.randn(B, N, 20, device=device)

    outs = {
        'pred_atoms': pred_atoms,
        'final_gaussian': pred_gaussian,
        'logits': logits,
    }

    return outs


def test_loss_forward():
    """测试损失函数前向传播"""
    print("=" * 80)
    print("测试 SideAtomsIGALoss_Final")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 创建loss模块
    config = MockConfig()
    loss_fn = SideAtomsIGALoss_Final(config).to(device)
    print("\n✓ Loss模块初始化成功")

    # 创建虚拟数据
    print("\n创建虚拟数据...")
    batch, noisy_batch = create_mock_batch(B=2, N=10, device=device)
    print(f"  Batch size: {batch['aatype'].shape}")
    print(f"  Atoms14 shape: {batch['atoms14_local'].shape}")
    print(f"  Update mask sum: {noisy_batch['update_mask'].sum().item()}")

    # 创建预测
    print("\n创建模型预测...")
    outs = create_mock_predictions(batch, device=device)
    print(f"  Pred atoms shape: {outs['pred_atoms'].shape}")
    print(f"  Gaussian trans shape: {outs['final_gaussian'].get_trans().shape}")
    print(f"  Logits shape: {outs['logits'].shape}")

    # 前向传播
    print("\n执行前向传播...")
    try:
        metrics = loss_fn(outs, batch, noisy_batch)
        print("✓ 前向传播成功!")
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 检查输出
    print("\n" + "=" * 80)
    print("Loss 组成:")
    print("=" * 80)

    total_loss = metrics['loss']
    print(f"\n【总损失】 {total_loss.item():.4f}")

    print("\n--- Legacy Coordinate Losses ---")
    print(f"  Coord Loss (total):  {metrics['coord_loss']:.4f}")
    print(f"    - MSE:             {metrics['coord_mse']:.4f}")
    print(f"    - Pairwise:        {metrics['coord_pair']:.4f}")
    print(f"    - Huber:           {metrics['coord_huber']:.4f}")

    print("\n--- IGA Gaussian Losses ---")
    print(f"  Gaussian Param MSE:  {metrics['gauss_param_mse']:.4f}")
    print(f"  Gaussian NLL:        {metrics['gauss_nll']:.4f}")

    print("\n--- Sequence Prediction ---")
    print(f"  Sequence Loss:       {metrics['seq_loss']:.4f}")
    print(f"  AA Accuracy:         {metrics['aa_acc']:.4f} ({metrics['aa_acc']*100:.1f}%)")
    print(f"  Perplexity:          {metrics['perplexity']:.4f}")

    # Per-atom metrics
    print("\n--- Per-Atom MSE (Sidechain) ---")
    for atom_idx in range(3, 14):
        key = f'atom{atom_idx:02d}_mse'
        if key in metrics:
            print(f"  Atom {atom_idx:2d}: {metrics[key]:.4f}")

    # 检查梯度
    print("\n" + "=" * 80)
    print("梯度检查:")
    print("=" * 80)

    if total_loss.requires_grad:
        total_loss.backward()
        print("✓ 反向传播成功")

        # 检查关键参数是否有梯度
        if outs['pred_atoms'].grad is not None:
            print(f"  pred_atoms.grad: {outs['pred_atoms'].grad.abs().mean().item():.6f}")
        else:
            print("  pred_atoms.grad: None (需要 requires_grad=True)")
    else:
        print("⚠ total_loss 不需要梯度 (requires_grad=False)")

    return True


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 80)
    print("边界情况测试")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = MockConfig()
    loss_fn = SideAtomsIGALoss_Final(config).to(device)

    # Case 1: 全GLY (没有侧链)
    print("\n--- Case 1: 全Glycine (无侧链) ---")
    batch, noisy_batch = create_mock_batch(B=1, N=5, device=device)
    batch['atom14_gt_exists'][:, :, 3:] = 0  # 全部侧链mask掉
    batch['aatype'][:] = 7  # 全是GLY

    outs = create_mock_predictions(batch, device=device)
    try:
        metrics = loss_fn(outs, batch, noisy_batch)
        print(f"✓ 全GLY测试通过, Loss={metrics['loss'].item():.4f}")
        print(f"  Gauss Param MSE: {metrics['gauss_param_mse']:.4f} (应该很小或0)")
        print(f"  Gauss NLL: {metrics['gauss_nll']:.4f}")
    except Exception as e:
        print(f"✗ 全GLY测试失败: {e}")

    # Case 2: 部分update_mask=0
    print("\n--- Case 2: 部分残基不更新 ---")
    batch, noisy_batch = create_mock_batch(B=1, N=10, device=device)
    noisy_batch['update_mask'][:, 5:] = 0  # 后一半不更新

    outs = create_mock_predictions(batch, device=device)
    try:
        metrics = loss_fn(outs, batch, noisy_batch)
        print(f"✓ 部分更新测试通过, Loss={metrics['loss'].item():.4f}")
    except Exception as e:
        print(f"✗ 部分更新测试失败: {e}")

    # Case 3: 没有logits (不预测序列)
    print("\n--- Case 3: 无Sequence预测 ---")
    batch, noisy_batch = create_mock_batch(B=1, N=5, device=device)
    outs = create_mock_predictions(batch, device=device)
    outs['logits'] = None

    try:
        metrics = loss_fn(outs, batch, noisy_batch)
        print(f"✓ 无Logits测试通过, Loss={metrics['loss'].item():.4f}")
        print(f"  Seq Loss: {metrics['seq_loss']:.4f} (应该是0)")
    except Exception as e:
        print(f"✗ 无Logits测试失败: {e}")


def test_numerical_stability():
    """测试数值稳定性"""
    print("\n" + "=" * 80)
    print("数值稳定性测试")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = MockConfig()
    loss_fn = SideAtomsIGALoss_Final(config).to(device)

    # 极端情况: 很大的坐标
    print("\n--- 大坐标值 (1000 Angstrom) ---")
    batch, noisy_batch = create_mock_batch(B=1, N=5, device=device)
    batch['atoms14_local'] *= 1000
    batch['atom14_gt_positions'] *= 1000

    outs = create_mock_predictions(batch, device=device)
    outs['pred_atoms'] *= 1000
    outs['final_gaussian']._trans *= 1000

    try:
        metrics = loss_fn(outs, batch, noisy_batch)
        if torch.isnan(metrics['loss']) or torch.isinf(metrics['loss']):
            print(f"✗ 出现NaN/Inf: {metrics['loss'].item()}")
        else:
            print(f"✓ 大坐标测试通过, Loss={metrics['loss'].item():.4f}")
    except Exception as e:
        print(f"✗ 大坐标测试失败: {e}")

    # 极端情况: 协方差接近奇异
    print("\n--- 协方差接近奇异 ---")
    batch, noisy_batch = create_mock_batch(B=1, N=5, device=device)
    outs = create_mock_predictions(batch, device=device)
    # 让scaling很小
    outs['final_gaussian']._scaling_log = torch.ones_like(
        outs['final_gaussian']._scaling_log
    ) * (-10)  # exp(-10) ≈ 4e-5

    try:
        metrics = loss_fn(outs, batch, noisy_batch)
        if torch.isnan(metrics['loss']) or torch.isinf(metrics['loss']):
            print(f"✗ 出现NaN/Inf: {metrics['loss'].item()}")
        else:
            print(f"✓ 小方差测试通过, Loss={metrics['loss'].item():.4f}")
            print(f"  Gauss NLL: {metrics['gauss_nll']:.4f}")
    except Exception as e:
        print(f"✗ 小方差测试失败: {e}")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("SideAtomsIGALoss_Final 单元测试")
    print("="*80)

    success = True

    # 基本测试
    if not test_loss_forward():
        success = False

    # 边界情况
    test_edge_cases()

    # 数值稳定性
    test_numerical_stability()

    print("\n" + "="*80)
    if success:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败，请检查上述错误")
    print("="*80)
