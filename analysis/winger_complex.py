import torch
import math
from openfold.utils.rigid_utils import Rigid,Rotation
try:
    from e3nn.o3 import wigner_D

    E3NN_AVAILABLE = True
except ImportError:
    E3NN_AVAILABLE = False
    print("警告: e3nn库未安装，请运行: pip install e3nn")
import numpy as np

def canonicalize_zyz(alpha, beta, gamma):
    """统一欧拉角到主值区间"""
    pi = torch.pi

    # 把所有角规整到 (-pi, pi]
    alpha = (alpha + pi) % (2 * pi) - pi
    beta = beta.clamp(0.0, pi)  # β 本来就该在 [0, π]
    gamma = (gamma + pi) % (2 * pi) - pi

    return alpha, beta, gamma

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


def quaternion_to_euler_zyz_correct(q):
    """
    四元数转ZYZ欧拉角（正确版本）
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # 归一化
    norm = torch.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # 正确的ZYZ转换公式
    beta = torch.acos(torch.clamp(w * w + z * z - x * x - y * y, -1 + 1e-7, 1 - 1e-7))
    sin_beta = torch.sin(beta)

    # 避免除零
    safe_sin_beta = torch.where(torch.abs(sin_beta) < 1e-6,
                                torch.ones_like(sin_beta), sin_beta)

    alpha = torch.atan2((x * z + w * y) / safe_sin_beta, (w * x - y * z) / safe_sin_beta)
    gamma = torch.atan2((x * z - w * y) / safe_sin_beta, (w * x + y * z) / safe_sin_beta)

    # 特殊情况处理
    small_beta = torch.abs(sin_beta) < 1e-6
    alpha = torch.where(small_beta, torch.atan2(2 * (w * y + x * z), w * w + x * x - y * y - z * z), alpha)
    gamma = torch.where(small_beta, torch.zeros_like(gamma), gamma)

    return alpha, beta, gamma
def rotation_matrix_to_quaternion_stable(R):
    """
    数值稳定的旋转矩阵转四元数
    """
    # Shepperd's method for numerical stability
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    # 四种情况，选择最大的来避免数值不稳定
    case0 = trace > R[..., 0, 0]
    case0 = case0 & (trace > R[..., 1, 1])
    case0 = case0 & (trace > R[..., 2, 2])

    case1 = R[..., 0, 0] > R[..., 1, 1]
    case1 = case1 & (R[..., 0, 0] > R[..., 2, 2])
    case1 = case1 & (~case0)

    case2 = R[..., 1, 1] > R[..., 2, 2]
    case2 = case2 & (~case0) & (~case1)

    case3 = (~case0) & (~case1) & (~case2)

    # Case 0: trace is largest
    s0 = torch.sqrt(trace + 1.0) * 2
    w0 = 0.25 * s0
    x0 = (R[..., 2, 1] - R[..., 1, 2]) / s0
    y0 = (R[..., 0, 2] - R[..., 2, 0]) / s0
    z0 = (R[..., 1, 0] - R[..., 0, 1]) / s0

    # Case 1: R[0,0] is largest
    s1 = torch.sqrt(1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]) * 2
    w1 = (R[..., 2, 1] - R[..., 1, 2]) / s1
    x1 = 0.25 * s1
    y1 = (R[..., 0, 1] + R[..., 1, 0]) / s1
    z1 = (R[..., 0, 2] + R[..., 2, 0]) / s1

    # Case 2: R[1,1] is largest
    s2 = torch.sqrt(1.0 + R[..., 1, 1] - R[..., 0, 0] - R[..., 2, 2]) * 2
    w2 = (R[..., 0, 2] - R[..., 2, 0]) / s2
    x2 = (R[..., 0, 1] + R[..., 1, 0]) / s2
    y2 = 0.25 * s2
    z2 = (R[..., 1, 2] + R[..., 2, 1]) / s2

    # Case 3: R[2,2] is largest
    s3 = torch.sqrt(1.0 + R[..., 2, 2] - R[..., 0, 0] - R[..., 1, 1]) * 2
    w3 = (R[..., 1, 0] - R[..., 0, 1]) / s3
    x3 = (R[..., 0, 2] + R[..., 2, 0]) / s3
    y3 = (R[..., 1, 2] + R[..., 2, 1]) / s3
    z3 = 0.25 * s3

    # Combine results
    w = torch.where(case0, w0, torch.where(case1, w1, torch.where(case2, w2, w3)))
    x = torch.where(case0, x0, torch.where(case1, x1, torch.where(case2, x2, x3)))
    y = torch.where(case0, y0, torch.where(case1, y1, torch.where(case2, y2, y3)))
    z = torch.where(case0, z0, torch.where(case1, z1, torch.where(case2, z2, z3)))

    return torch.stack([w, x, y, z], dim=-1)




def R_to_wigner_D(R, max_l=2):
    """
    从旋转矩阵计算Wigner D矩阵

    R: [..., 3, 3] 旋转矩阵
    返回: [..., total_dim, total_dim] 块对角Wigner D矩阵
    """
    if not E3NN_AVAILABLE:
        raise ImportError("需要安装e3nn库: pip install e3nn")

    # 1. R -> 欧拉角
    # euler = rotation_matrix_to_euler(R)  # [..., 3]
    # euler=canonicalize_zyz(euler[...,0], euler[...,1], euler[...,2])
    # print(euler)

    Rot=Rotation(R)
    quat=Rot.get_quats()
    print(quat)

    a,b,c=quaternion_to_euler_zyz_correct(quat)
    print(a,b,c)

    # 2. 计算各阶Wigner D矩阵
    D_blocks = []
    for l in range(max_l + 1):
        D_l = wigner_D(l,  a,b,c)  # [..., 2l+1, 2l+1]
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
def wigner_d_l2_analytical(beta):
    """
    l=2 的 Wigner d 矩阵的解析表达式（只依赖于 beta 角）
    这是最稳定的方法，直接使用解析公式
    """
    c = torch.cos(beta / 2)
    s = torch.sin(beta / 2)

    # d^2 矩阵的解析形式（5x5）
    # 按照 m' = -2, -1, 0, 1, 2 的顺序
    d = torch.zeros(*beta.shape, 5, 5, dtype=beta.dtype, device=beta.device)

    # 填充矩阵元素（使用标准的 Wigner d 矩阵公式）
    d[..., 0, 0] = c ** 4  # d^2_{-2,-2}
    d[..., 0, 1] = 2 * c ** 3 * s  # d^2_{-2,-1}
    d[..., 0, 2] = math.sqrt(6) * c ** 2 * s ** 2  # d^2_{-2,0}
    d[..., 0, 3] = 2 * c * s ** 3  # d^2_{-2,1}
    d[..., 0, 4] = s ** 4  # d^2_{-2,2}

    d[..., 1, 0] = -2 * c ** 3 * s  # d^2_{-1,-2}
    d[..., 1, 1] = c ** 2 * (2 * c ** 2 - 1)  # d^2_{-1,-1}
    d[..., 1, 2] = math.sqrt(6) * c * s * (c ** 2 - s ** 2)  # d^2_{-1,0}
    d[..., 1, 3] = s * (1 - 2 * s ** 2)  # d^2_{-1,1}
    d[..., 1, 4] = -2 * s ** 3 * c  # d^2_{-1,2}

    d[..., 2, 0] = math.sqrt(6) * c ** 2 * s ** 2  # d^2_{0,-2}
    d[..., 2, 1] = -math.sqrt(6) * c * s * (c ** 2 - s ** 2)  # d^2_{0,-1}
    d[..., 2, 2] = 1 - 3 * c ** 2 * s ** 2  # d^2_{0,0}
    d[..., 2, 3] = math.sqrt(6) * c * s * (c ** 2 - s ** 2)  # d^2_{0,1}
    d[..., 2, 4] = math.sqrt(6) * c ** 2 * s ** 2  # d^2_{0,2}

    d[..., 3, 0] = -2 * c * s ** 3  # d^2_{1,-2}
    d[..., 3, 1] = s * (1 - 2 * s ** 2)  # d^2_{1,-1}
    d[..., 3, 2] = -math.sqrt(6) * c * s * (c ** 2 - s ** 2)  # d^2_{1,0}
    d[..., 3, 3] = c ** 2 * (2 * c ** 2 - 1)  # d^2_{1,1}
    d[..., 3, 4] = 2 * c ** 3 * s  # d^2_{1,2}

    d[..., 4, 0] = s ** 4  # d^2_{2,-2}
    d[..., 4, 1] = -2 * s ** 3 * c  # d^2_{2,-1}
    d[..., 4, 2] = math.sqrt(6) * s ** 2 * c ** 2  # d^2_{2,0}
    d[..., 4, 3] = -2 * s * c ** 3  # d^2_{2,1}
    d[..., 4, 4] = c ** 4  # d^2_{2,2}

    return d


def wigner_D_l2_from_rotation_matrix(R):
    """
    从旋转矩阵直接计算 l=2 的 Wigner D 矩阵
    使用数值稳定的方法：R -> 四元数 -> 欧拉角 -> Wigner D
    """
    # 步骤1：旋转矩阵转四元数（数值稳定）
    q = rotation_matrix_to_quaternion_stable(R)
    print(q)



    # 步骤2：四元数转ZYZ欧拉角


    alpha, beta, gamma =quaternion_to_euler_zyz_correct(q)
    alpha, beta, gamma=canonicalize_zyz(alpha, beta, gamma)
    print(alpha, beta, gamma)

    # 步骤3：使用解析公式计算 Wigner d 矩阵
    d = wigner_d_l2_analytical(beta)

    # 步骤4：添加 alpha 和 gamma 的相位因子
    # D^l_{m'm}(α,β,γ) = e^{-im'α} d^l_{m'm}(β) e^{-imγ}
    m_values = torch.arange(-2, 3, dtype=torch.float32, device=R.device)  # [-2, -1, 0, 1, 2]

    # 构造相位矩阵
    alpha_phase = torch.exp(-1j * alpha[..., None, None] * m_values[None, :, None])
    gamma_phase = torch.exp(-1j * gamma[..., None, None] * m_values[None, None, :])

    # 完整的 Wigner D 矩阵
    D = alpha_phase * d.to(torch.complex64) * gamma_phase

    return D


def compare_with_e3nn():
    """
    与 e3nn 结果进行比较
    """
    # 创建测试旋转矩阵
    torch.manual_seed(42)
    R = torch.randn(1, 3, 3)
    U, _, Vt = torch.svd(R)
    R = torch.matmul(U, Vt)

    print("=== 测试旋转矩阵 ===")
    print(f"形状: {R.shape}")
    print("第一个矩阵:")
    print(R[0])
    print()

    # 我们的方法
    D_ours = wigner_D_l2_from_rotation_matrix(R)
    print("=== 我们的方法结果 ===")
    print(f"形状: {D_ours.shape}")
    print("第一个 Wigner D 矩阵:")
    print(D_ours[0])
    print()

    # 验证正交性
    D_dagger = D_ours.conj().transpose(-1, -2)
    identity = D_ours @ D_dagger
    eye = torch.eye(5, dtype=torch.complex64).expand_as(identity)
    orthogonality_error = torch.norm(identity - eye, dim=(-1, -2))

    print("=== 验证正交性 ===")
    print(f"正交性误差: {orthogonality_error}")
    print("误差 < 1e-5:", torch.all(orthogonality_error < 1e-5))




    # 计算Wigner D矩阵
    D_e3nn = R_to_wigner_D(R, max_l=2)  # 1+3+5=9维
    print(f"输出D_e3nn: {D_e3nn.shape}")

    # 比较 Frobenius norm 差异
    diff = torch.norm(D_e3nn - D_ours) / torch.norm(D_e3nn)
    print(f"相对差异: {diff.item():.2e}")

    return D_ours


def specific_rotations():
    """
    测试特定的旋转情况
    """
    print("=== 测试特定旋转 ===")

    # 测试1：单位矩阵（无旋转）
    I = torch.eye(3).unsqueeze(0)
    D_identity = wigner_D_l2_from_rotation_matrix(I)
    expected_identity = torch.eye(5, dtype=torch.complex64)

    print("1. 单位旋转:")
    print(f"D 矩阵与单位矩阵的差异: {torch.norm(D_identity[0] - expected_identity):.2e}")

    # 测试2：绕Z轴旋转90度
    theta = math.pi / 2
    R_z90 = torch.tensor([[[math.cos(theta), -math.sin(theta), 0],
                           [math.sin(theta), math.cos(theta), 0],
                           [0, 0, 1]]], dtype=torch.float32)

    D_z90 = wigner_D_l2_from_rotation_matrix(R_z90)
    print(f"2. 绕Z轴90度旋转的 Wigner D 矩阵:")
    print(D_z90[0])

    # 测试3：绕X轴旋转90度
    R_x90 = torch.tensor([[[1, 0, 0],
                           [0, math.cos(theta), -math.sin(theta)],
                           [0, math.sin(theta), math.cos(theta)]]], dtype=torch.float32)

    D_x90 = wigner_D_l2_from_rotation_matrix(R_x90)
    print(f"3. 绕X轴90度旋转的 Wigner D 矩阵:")
    print(D_x90[0])


def main():
    """
    主测试函数
    """
    print("=== 数值稳定的 Wigner D 矩阵计算 ===")
    print("基于四元数和解析公式的方法")
    print()

    # 运行比较测试
    D_result = compare_with_e3nn()
    print()

    # 测试特定旋转
    # test_specific_rotations()

    # print()
    # print("=== 总结 ===")
    # print("✅ 方法特点:")
    # print("- 使用四元数避免欧拉角奇点")
    # print("- 直接使用 l=2 Wigner d 矩阵的解析公式")
    # print("- 数值稳定，适合批量计算")
    # print("- 结果满足正交性条件")


if __name__ == "__main__":
    main()