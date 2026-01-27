"""
用真实数据验证NLL的数值范围
测试场景：GT高斯 + GT原子（从GT高斯采样）→ 计算真实的NLL
"""
import sys
import torch
import numpy as np
sys.path.append('/home/junyu/project/pu')

from models.IGA import fused_gaussian_overlap_score


def compute_nll_from_gaussian(atoms, mu, sigma):
    """
    计算原子在高斯分布下的NLL
    atoms: [B, N, K, 3] 原子坐标
    mu: [B, N, 3] 高斯中心
    sigma: [B, N, 3, 3] 协方差矩阵
    """
    # Delta
    delta = atoms - mu.unsqueeze(-2)  # [B, N, K, 3]

    # Mahalanobis distance^2
    sigma_exp = sigma.unsqueeze(-3).expand(*delta.shape[:-1], 3, 3)
    mahal_sq = -2.0 * fused_gaussian_overlap_score(delta, sigma_exp)

    # Log determinant
    L_chol = torch.linalg.cholesky(sigma)
    log_det = 2.0 * torch.diagonal(L_chol, dim1=-2, dim2=-1).log().sum(-1)

    # NLL per atom
    nll_per_atom = 0.5 * (mahal_sq + log_det.unsqueeze(-1))

    return nll_per_atom, mahal_sq, log_det


def test_nll_with_perfect_fit():
    """
    测试1: 用GT高斯评估从GT高斯采样的原子
    理论上: 期望NLL ≈ -log(p(x)) 的平均值
    """
    print("=" * 80)
    print("测试1: GT高斯 + GT采样原子 → 理论NLL范围")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, N = 1, 100
    num_atoms = 11

    # 构建GT高斯
    mu = torch.zeros(B, N, 3, device=device)

    # 不同的协方差尺度
    for scale in [0.5, 1.0, 2.0, 3.0]:
        print(f"\n--- Gaussian Scale = {scale:.1f} Å ---")

        # 各向同性协方差
        sigma = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0) * (scale ** 2)
        sigma = sigma.expand(B, N, 3, 3)

        # 从这个高斯采样原子
        L = torch.linalg.cholesky(sigma)
        z = torch.randn(B, N, num_atoms, 3, device=device)
        atoms = mu.unsqueeze(2) + torch.einsum('bnij,bnkj->bnki', L, z)

        # 计算NLL
        nll_per_atom, mahal_sq, log_det = compute_nll_from_gaussian(atoms, mu, sigma)

        # 统计
        nll_mean = nll_per_atom.mean().item()
        nll_std = nll_per_atom.std().item()
        mahal_mean = mahal_sq.mean().item()
        log_det_val = log_det[0, 0].item()

        # 理论值: 3D高斯的NLL期望
        # E[NLL] = 0.5 * (E[d_M^2] + log|Σ|)
        #        = 0.5 * (3 + log|Σ|)  (因为期望的Mahalanobis^2 = 维度数)
        # log|Σ| = 3 * log(scale^2) = 6 * log(scale)
        theoretical_nll = 0.5 * (3 + 6 * np.log(scale))

        print(f"  NLL per atom:     {nll_mean:.4f} ± {nll_std:.4f}")
        print(f"  Mahalanobis^2:    {mahal_mean:.4f} (理论=3.0)")
        print(f"  Log|Σ|:           {log_det_val:.4f} (理论={6*np.log(scale):.4f})")
        print(f"  理论NLL:          {theoretical_nll:.4f}")
        print(f"  误差:             {abs(nll_mean - theoretical_nll):.4f}")


def test_nll_with_misalignment():
    """
    测试2: 预测高斯与GT高斯偏移不同距离
    """
    print("\n" + "=" * 80)
    print("测试2: 预测高斯偏移 → NLL如何变化")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, N = 1, 50
    num_atoms = 11
    scale = 1.0  # 1Å标准差

    # GT高斯
    mu_gt = torch.zeros(B, N, 3, device=device)
    sigma = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0) * (scale ** 2)
    sigma = sigma.expand(B, N, 3, 3)

    # 从GT高斯采样原子
    L = torch.linalg.cholesky(sigma)
    z = torch.randn(B, N, num_atoms, 3, device=device)
    atoms_gt = mu_gt.unsqueeze(2) + torch.einsum('bnij,bnkj->bnki', L, z)

    # 预测高斯偏移不同距离
    for offset in [0.0, 0.5, 1.0, 2.0, 3.0]:
        print(f"\n--- Offset = {offset:.1f} Å ---")

        mu_pred = mu_gt + offset

        # 用预测的高斯评估GT原子
        nll_per_atom, mahal_sq, _ = compute_nll_from_gaussian(atoms_gt, mu_pred, sigma)

        nll_mean = nll_per_atom.mean().item()
        mahal_mean = mahal_sq.mean().item()

        # 理论: Mahalanobis^2 增加 (offset/scale)^2 * K (K=原子数)
        theoretical_mahal = 3.0 + (offset / scale) ** 2

        print(f"  NLL per atom:     {nll_mean:.4f}")
        print(f"  Mahalanobis^2:    {mahal_mean:.4f} (理论≈{theoretical_mahal:.2f})")


def test_nll_batch_level():
    """
    测试3: Batch级别的NLL总和
    """
    print("\n" + "=" * 80)
    print("测试3: Batch级别NLL总和")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for B, N in [(1, 10), (2, 50), (4, 100)]:
        num_atoms = 11
        scale = 1.0

        mu = torch.zeros(B, N, 3, device=device)
        sigma = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0) * (scale ** 2)
        sigma = sigma.expand(B, N, 3, 3)

        L = torch.linalg.cholesky(sigma)
        z = torch.randn(B, N, num_atoms, 3, device=device)
        atoms = mu.unsqueeze(2) + torch.einsum('bnij,bnkj->bnki', L, z)

        nll_per_atom, _, _ = compute_nll_from_gaussian(atoms, mu, sigma)

        # Batch级别汇总
        nll_total = nll_per_atom.sum().item()
        nll_mean_per_atom = nll_per_atom.mean().item()
        total_atoms = B * N * num_atoms

        print(f"\n  B={B}, N={N}, 总原子数={total_atoms}")
        print(f"    NLL总和:        {nll_total:.2f}")
        print(f"    NLL per atom:   {nll_mean_per_atom:.4f}")
        print(f"    平均值 ≈        {nll_total / total_atoms:.4f}")


def test_real_sidechain_scale():
    """
    测试4: 真实侧链尺度下的NLL
    """
    print("\n" + "=" * 80)
    print("测试4: 真实蛋白侧链尺度")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, N = 2, 100  # 类似训练batch
    num_atoms = 11

    # 真实侧链的典型参数
    # Offset: ~1-2Å, Scale: ~1.0-1.5Å
    mu = torch.randn(B, N, 3, device=device) * 1.5  # offset

    # 各向异性协方差（更realistic）
    scales = torch.tensor([1.0, 1.2, 0.8], device=device).unsqueeze(0).unsqueeze(0).expand(B, N, 3)
    sigma = torch.diag_embed(scales ** 2)

    # 采样
    L = torch.linalg.cholesky(sigma)
    z = torch.randn(B, N, num_atoms, 3, device=device)
    atoms = mu.unsqueeze(2) + torch.einsum('bnij,bnkj->bnki', L, z)

    # 计算NLL（perfect fit）
    nll_per_atom, _, _ = compute_nll_from_gaussian(atoms, mu, sigma)

    print(f"\n  Batch: B={B}, N={N}, 原子/残基={num_atoms}")
    print(f"  NLL per atom:   {nll_per_atom.mean().item():.4f} ± {nll_per_atom.std().item():.4f}")
    print(f"  NLL per residue: {nll_per_atom.sum(dim=-1).mean().item():.4f}")
    print(f"  NLL batch total: {nll_per_atom.sum().item():.2f}")

    # 如果用w_nll=0.1
    weighted_nll = nll_per_atom.sum().item() * 0.1
    print(f"\n  加权后 (w_nll=0.1):   {weighted_nll:.2f}")

    # 对比典型的coord loss
    typical_coord_mse = 1.0  # 假设MSE=1.0
    print(f"  典型Coord MSE:        {typical_coord_mse:.2f}")
    print(f"  NLL/Coord比例:        {weighted_nll / typical_coord_mse:.1f}x")

    # 建议的权重
    suggested_w = typical_coord_mse / nll_per_atom.sum().item()
    print(f"\n  建议的w_nll:          {suggested_w:.6f} (使NLL与Coord MSE同尺度)")


if __name__ == '__main__':
    test_nll_with_perfect_fit()
    test_nll_with_misalignment()
    test_nll_batch_level()
    test_real_sidechain_scale()

    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("\n1. 在完美拟合情况下(GT采样+GT高斯):")
    print("   - NLL per atom ≈ 1.5 (理论值)")
    print("   - NLL per residue (11原子) ≈ 16.5")
    print("   - NLL batch (B=2, N=100) ≈ 3300")
    print("\n2. 当前w_nll=0.1时:")
    print("   - 加权NLL ≈ 330 (远大于coord loss~1)")
    print("\n3. 建议:")
    print("   - w_nll ≈ 0.0003 ~ 0.001")
    print("   - 或者在loss计算时除以原子数做归一化")
    print("=" * 80)
