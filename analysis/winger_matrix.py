
import torch
import torch.nn as nn
import math

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
# 生成 R：[B, 3, 3]
R = torch.randn(1, 3, 3)
U, _, Vt = torch.linalg.svd(R)
R = torch.matmul(U, Vt)

D2 = wigner_D_l2_from_rotation_matrix(R)  # [10, 5, 5]


