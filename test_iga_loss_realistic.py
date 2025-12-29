"""
更真实的IGA Loss测试 - 使用一致的高斯分布
"""
import sys
import torch
sys.path.append('/home/junyu/project/pu')
sys.path.append('/home/junyu/project/pu/data')

from models.loss import SideAtomsIGALoss_Final
from data.GaussianRigid import OffsetGaussianRigid
from openfold.utils.rigid_utils import Rotation, Rigid


class MockConfig:
    def __init__(self):
        self.atom_loss_weight = 1.0
        self.pair_loss_weight = 1.0
        self.huber_loss_weight = 1.0
        self.bb_atom_scale = 1.0
        self.bb_atom_loss_weight = 1.0
        self.w_param = 5.0
        self.w_nll = 0.1
        self.type_loss_weight = 1.0


def create_realistic_data(B=2, N=10, device='cuda'):
    """
    创建更realistic的数据：
    - 从真实的高斯分布采样原子
    - 预测和GT在同一分布附近
    """
    # ===== 1. 构建GT Gaussian =====
    # 主链Frame
    identity_rot = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, 3, 3)
    rots = Rotation(rot_mats=identity_rot)
    ca_pos = torch.randn(B, N, 3, device=device) * 5.0  # CA在[-5, 5]范围

    # 高斯参数
    gt_local_mean = torch.randn(B, N, 3, device=device) * 1.0  # Offset ~1Å
    gt_scaling_log = torch.randn(B, N, 3, device=device) * 0.3  # Scaling ~0.7-1.4Å

    gt_gaussian = OffsetGaussianRigid(rots, ca_pos, gt_scaling_log, gt_local_mean)

    # ===== 2. 从GT高斯采样侧链原子 =====
    num_sc_atoms = 11
    mu_global = gt_gaussian.get_gaussian_mean()  # [B, N, 3]
    sigma_global = gt_gaussian.get_covariance()  # [B, N, 3, 3]

    # Cholesky分解
    L = torch.linalg.cholesky(sigma_global)  # [B, N, 3, 3]

    # 采样原子 (每个残基11个侧链原子)
    z = torch.randn(B, N, num_sc_atoms, 3, device=device)  # 标准正态
    atoms_sc_global = mu_global.unsqueeze(2) + torch.einsum('bnij,bnaj->bnai', L, z)  # [B, N, 11, 3]

    # ===== 3. 转换到局部坐标 =====
    rigid_frame = Rigid(rots, ca_pos)
    atoms_sc_local = rigid_frame.unsqueeze(-1).invert_apply(atoms_sc_global)  # [B, N, 11, 3]

    # ===== 4. 组装14原子 (Backbone GT + Sidechain sampled) =====
    # Backbone: N, CA, C (局部坐标，简化为固定几何)
    backbone_local = torch.zeros(B, N, 3, 3, device=device)
    backbone_local[..., 0, :] = torch.tensor([-0.5, 1.4, 0.0], device=device)  # N
    backbone_local[..., 1, :] = torch.tensor([0.0, 0.0, 0.0], device=device)   # CA
    backbone_local[..., 2, :] = torch.tensor([1.5, 0.0, 0.0], device=device)   # C

    atoms14_local = torch.cat([backbone_local, atoms_sc_local], dim=2)  # [B, N, 14, 3]

    # 全局坐标
    atoms14_global = rigid_frame.unsqueeze(-1).apply(atoms14_local)

    # ===== 5. Masks =====
    atom14_gt_exists = torch.ones(B, N, 14, device=device)
    atom14_gt_exists[:, 0, 3:] = 0  # 第0个残基是GLY
    atom14_gt_exists[:, :, 10:] *= torch.randint(0, 2, (B, N, 4), device=device).float()

    res_mask = torch.ones(B, N, device=device)
    res_mask[:, -1] = 0  # 最后一个mask

    update_mask = torch.ones(B, N, device=device)

    aatype = torch.randint(0, 20, (B, N), device=device)
    aatype[:, 0] = 7  # GLY

    batch = {
        'atoms14_local': atoms14_local,
        'atom14_gt_positions': atoms14_global,
        'atom14_gt_exists': atom14_gt_exists,
        'res_mask': res_mask,
        'aatype': aatype,
        'local_mean_1': gt_local_mean,
        'scaing_log_1': gt_scaling_log,
    }

    noisy_batch = {
        'update_mask': update_mask,
        'res_mask': res_mask,
    }

    return batch, noisy_batch, gt_gaussian


def create_realistic_predictions(batch, gt_gaussian, noise_level=0.1, device='cuda'):
    """
    创建接近GT的预测 (添加小噪声)
    """
    B, N = batch['aatype'].shape

    # ===== 预测侧链 =====
    gt_sc_local = batch['atoms14_local'][..., 3:14, :]
    pred_atoms = gt_sc_local + torch.randn_like(gt_sc_local) * noise_level

    # ===== 预测高斯 =====
    # 添加小噪声
    pred_local_mean = gt_gaussian._local_mean + torch.randn_like(gt_gaussian._local_mean) * noise_level
    pred_scaling_log = gt_gaussian._scaling_log + torch.randn_like(gt_gaussian._scaling_log) * (noise_level * 0.5)

    pred_gaussian = OffsetGaussianRigid(
        gt_gaussian.get_rots(),
        gt_gaussian.get_trans(),
        pred_scaling_log,
        pred_local_mean
    )

    # ===== Sequence Logits =====
    # 让大部分预测正确
    logits = torch.randn(B, N, 20, device=device) * 0.5
    for b in range(B):
        for n in range(N):
            logits[b, n, batch['aatype'][b, n]] += 5.0  # 给GT类别加大logit

    outs = {
        'pred_atoms': pred_atoms,
        'final_gaussian': pred_gaussian,
        'logits': logits,
    }

    return outs


def test_realistic_scenario():
    """测试realistic场景下的loss值"""
    print("=" * 80)
    print("Realistic Scenario 测试")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = MockConfig()
    loss_fn = SideAtomsIGALoss_Final(config).to(device)

    # 创建数据
    batch, noisy_batch, gt_gaussian = create_realistic_data(B=2, N=10, device=device)

    # 测试不同噪声水平
    for noise_level in [0.0, 0.1, 0.5, 1.0]:
        print(f"\n--- Noise Level = {noise_level:.1f} Å ---")

        outs = create_realistic_predictions(batch, gt_gaussian, noise_level, device)

        metrics = loss_fn(outs, batch, noisy_batch)

        print(f"  Total Loss:      {metrics['loss']:.4f}")
        print(f"  Coord MSE:       {metrics['coord_mse']:.4f}")
        print(f"  Gauss Param MSE: {metrics['gauss_param_mse']:.4f}")
        print(f"  Gauss NLL:       {metrics['gauss_nll']:.4f}")
        print(f"  Seq Loss:        {metrics['seq_loss']:.4f}")
        print(f"  AA Acc:          {metrics['aa_acc']:.2%}")
        print(f"  Perplexity:      {metrics['perplexity']:.2f}")


def test_gradient_flow():
    """测试梯度是否正常传播"""
    print("\n" + "=" * 80)
    print("梯度传播测试")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = MockConfig()
    loss_fn = SideAtomsIGALoss_Final(config).to(device)

    batch, noisy_batch, gt_gaussian = create_realistic_data(B=1, N=5, device=device)

    # 创建需要梯度的预测
    pred_atoms = torch.randn(1, 5, 11, 3, device=device, requires_grad=True)

    from openfold.utils.rigid_utils import Rotation, Rigid
    identity_rot = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(1, 5, 3, 3)
    rots = Rotation(rot_mats=identity_rot)
    trans = torch.randn(1, 5, 3, device=device, requires_grad=True)
    scaling_log = torch.randn(1, 5, 3, device=device, requires_grad=True)
    local_mean = torch.randn(1, 5, 3, device=device, requires_grad=True)

    pred_gaussian = OffsetGaussianRigid(rots, trans, scaling_log, local_mean)
    logits = torch.randn(1, 5, 20, device=device, requires_grad=True)

    outs = {
        'pred_atoms': pred_atoms,
        'final_gaussian': pred_gaussian,
        'logits': logits,
    }

    # 前向+反向
    metrics = loss_fn(outs, batch, noisy_batch)
    total_loss = metrics['loss']

    print(f"\nTotal Loss: {total_loss.item():.4f}")
    print(f"Requires Grad: {total_loss.requires_grad}")

    if total_loss.requires_grad:
        total_loss.backward()

        print("\n梯度统计:")
        print(f"  pred_atoms.grad:  mean={pred_atoms.grad.abs().mean():.6f}, max={pred_atoms.grad.abs().max():.6f}")
        print(f"  trans.grad:       mean={trans.grad.abs().mean():.6f}, max={trans.grad.abs().max():.6f}")
        print(f"  scaling_log.grad: mean={scaling_log.grad.abs().mean():.6f}, max={scaling_log.grad.abs().max():.6f}")
        print(f"  local_mean.grad:  mean={local_mean.grad.abs().mean():.6f}, max={local_mean.grad.abs().max():.6f}")
        print(f"  logits.grad:      mean={logits.grad.abs().mean():.6f}, max={logits.grad.abs().max():.6f}")

        print("\n✓ 梯度传播正常")
    else:
        print("\n✗ Loss没有梯度")


if __name__ == '__main__':
    test_realistic_scenario()
    test_gradient_flow()
