import torch
import torch.nn as nn
import math
import numpy as np
# 需要安装: pip install e3nn
try:
    from e3nn.o3 import wigner_D

    E3NN_AVAILABLE = True
except ImportError:
    E3NN_AVAILABLE = False
    print("警告: e3nn库未安装，请运行: pip install e3nn")

import torch

def get_l2_basis(device, dtype):
    basis = torch.stack([
        torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.]]),  # Y_2^-1 (yz)
        torch.tensor([[0., 0., 1.], [0., 0., 0.], [1., 0., 0.]]),  # Y_2^1 (xz)
        torch.tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., 0.]]), # Y_2^-2 (xy)
        torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., -2.]]), # Y_2^0 (2z^2 - x^2 - y^2)
        torch.tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., 0.]]), # Y_2^2 (x^2 - y^2)
    ], dim=0).to(dtype).to(device)  # shape: [5, 3, 3]
    return basis
def wigner_D_l2_from_rotation_matrix(R: torch.Tensor) -> torch.Tensor:
    """
    R: [..., 3, 3]
    return: [..., 5, 5]
    """
    B = R.shape[:-2]
    device = R.device
    dtype = R.dtype

    basis = get_l2_basis(device, dtype)  # [5, 3, 3]

    # Expand for einsum
    Bi = basis.view(1, 5, 3, 3).expand(*B, 5, 3, 3)  # [..., 5, 3, 3]
    Bj = basis.view(1, 5, 3, 3).expand(*B, 5, 3, 3)  # [..., 5, 3, 3]

    # 双旋转变换：D_ij = B_i^{kl} * R^{km} * R^{ln} * B_j^{mn}
    D = torch.einsum("...ikl,...km,...ln,...jmn->...ij", Bi, R, R, Bj)

    return D
def rotation_matrix_to_euler(R):
    """
    旋转矩阵转欧拉角 (ZYZ约定)
    R: [..., 3, 3] -> euler: [..., 3] (alpha, beta, gamma)
    """
    beta = torch.acos(torch.clamp(R[..., 2, 2], -1 + 1e-6, 1 - 1e-6))
    sin_beta = torch.sin(beta)

    # 避免除零
    safe_sin_beta = torch.where(torch.abs(sin_beta) < 1e-6,
                                torch.ones_like(sin_beta), sin_beta)

    alpha = torch.atan2(R[..., 1, 2] / safe_sin_beta, R[..., 0, 2] / safe_sin_beta)
    gamma = torch.atan2(R[..., 2, 1] / safe_sin_beta, -R[..., 2, 0] / safe_sin_beta)

    # 特殊情况：beta接近0或π
    small_beta = torch.abs(sin_beta) < 1e-6
    alpha = torch.where(small_beta, torch.atan2(-R[..., 0, 1], R[..., 0, 0]), alpha)
    gamma = torch.where(small_beta, torch.zeros_like(gamma), gamma)

    return torch.stack([alpha, beta, gamma], dim=-1)


def R_to_wigner_D(R, max_l=2):
    """
    从旋转矩阵计算Wigner D矩阵

    R: [..., 3, 3] 旋转矩阵
    返回: [..., total_dim, total_dim] 块对角Wigner D矩阵
    """
    if not E3NN_AVAILABLE:
        raise ImportError("需要安装e3nn库: pip install e3nn")

    # 1. R -> 欧拉角
    euler = rotation_matrix_to_euler(R)  # [..., 3]

    # 2. 计算各阶Wigner D矩阵
    D_blocks = []
    for l in range(max_l + 1):
        D_l = wigner_D(l, euler[..., 0], euler[..., 1], euler[..., 2])  # [..., 2l+1, 2l+1]
        D_blocks.append(D_l)
        print(D_l)

    # # 3. 构造块对角矩阵
    # D_full = torch.block_diag(*[D_blocks[l][0] for l in range(max_l + 1)])  # 单个样本的块对角
    #
    # # 4. 批量处理
    # batch_shape = R.shape[:-2]
    # total_dim = sum(2 * l + 1 for l in range(max_l + 1))
    # D_batched = torch.zeros(*batch_shape, total_dim, total_dim, device=R.device, dtype=R.dtype)
    #
    # # 为每个批次元素单独构造
    # flat_euler = euler.view(-1, 3)
    # flat_D = D_batched.view(-1, total_dim, total_dim)
    #
    # for i in range(flat_euler.shape[0]):
    #     D_blocks_i = []
    #     for l in range(max_l + 1):
    #         D_l = wigner_D(l, flat_euler[i, 0], flat_euler[i, 1], flat_euler[i, 2])
    #         D_blocks_i.append(D_l)
    #     flat_D[i] = torch.block_diag(*D_blocks_i)
    #
    return D_l

def real_to_complex_D2(D_real: torch.Tensor) -> torch.Tensor:
    """
    将实张量基底下的 D^2 转换为复数 spherical harmonics 基底下的 D^2
    输入:
        D_real: [..., 5, 5]，real spherical harmonics 表示
    输出:
        D_complex: [..., 5, 5]，complex spherical harmonics 表示（复数）
    """
    device = D_real.device
    dtype = D_real.dtype

    # real -> complex 的变换矩阵 U，shape [5,5], complex-valued
    U = torch.tensor([
        [0,           0,           1/np.sqrt(2),   0,           1/np.sqrt(2)],
        [0,           0,          -1j/np.sqrt(2),  0,           1j/np.sqrt(2)],
        [0,           1/np.sqrt(2),0,              1/np.sqrt(2),0],
        [0,          -1j/np.sqrt(2),0,              1j/np.sqrt(2),0],
        [1,           0,           0,              0,           0],
    ], dtype=torch.complex64 if dtype==torch.float32 else torch.complex128, device=device)

    U = U.unsqueeze(0)  # [1, 5, 5]

    # D_complex = U^\dagger D_real U
    D_complex = U.conj().transpose(-1, -2) @ D_real.to(U.dtype) @ U  # [..., 5, 5]

    return D_complex
def wigner_D_l2_complex_from_rotation_matrix(R: torch.Tensor) -> torch.Tensor:
    """
    R: [..., 3, 3]
    return: [..., 5, 5] 复数版本 D^{(2)} 矩阵（complex spherical harmonics 基）
    """
    D_real = wigner_D_l2_from_rotation_matrix(R)  # 先算实基下的
    D_complex = real_to_complex_D2(D_real)        # 再转复数基底
    return D_complex

# 使用示例
def wigner_transform():
    """测试函数"""
    if not E3NN_AVAILABLE:
        print("跳过测试：e3nn库未安装")
        return

    # 创建测试数据
    B, N = 1, 1
    R = torch.randn(B, N, 3, 3)

    # 确保是有效旋转矩阵
    U, _, Vt = torch.svd(R)
    R = torch.matmul(U, Vt)

    print(f"输入R: {R.shape}")
    print(R)

    D_ours = wigner_D_l2_complex_from_rotation_matrix(R)
    print(f"输出D_ours: {D_ours}")

    # 计算Wigner D矩阵
    D_e3nn = R_to_wigner_D(R, max_l=2)  # 1+3+5=9维
    print(f"输出D_e3nn: {D_e3nn.shape}")

    # 比较 Frobenius norm 差异
    diff = torch.norm(D_e3nn - D_ours) / torch.norm(D_e3nn)
    print(f"相对差异: {diff.item():.2e}")




if __name__ == "__main__":
    wigner_transform()