import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Sequence


class SphericalHarmonicsIPA(nn.Module):
    """
    IPA改造版：使用球谐基和Wigner-D矩阵实现旋转等变性
    """

    def __init__(
            self,
            c_s: int,
            c_z: int,
            c_hidden: int,
            no_heads: int,
            max_l: int = 3,  # 最大球谐阶数
            radial_bins: int = 8,  # 径向分区数
            inf: float = 1e5,
            eps: float = 1e-8,
    ):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.max_l = max_l
        self.radial_bins = radial_bins
        self.inf = inf
        self.eps = eps

        # 计算球谐基的总维数
        self.sh_dim = (max_l+1) * (2*max_l+1) * radial_bins

        # 线性变换层
        hc = self.c_hidden * self.no_heads
        self.linear_q = nn.Linear(self.c_s, hc)
        self.linear_kv = nn.Linear(self.c_s, 2 * hc)

        # 球谐基特征生成
        self.linear_q_sh = nn.Linear(self.c_s, self.no_heads * self.sh_dim)
        self.linear_kv_sh = nn.Linear(self.c_s, 2 * self.no_heads * self.sh_dim)

        # 配对表示处理
        self.linear_b = nn.Linear(self.c_z, self.no_heads)
        self.down_z = nn.Linear(self.c_z, self.c_z // 4)

        # 可学习权重
        self.sh_weights = nn.Parameter(torch.zeros(max_l + 1))  # 每个l阶的权重
        self.head_weights = nn.Parameter(torch.zeros(no_heads))

        # 输出层
        concat_out_dim = self.c_z // 4 + self.c_hidden + self.sh_dim
        self.linear_out = nn.Linear(self.no_heads * concat_out_dim, self.c_s)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        # 预计算Wigner-D矩阵的索引
        self._setup_wigner_indices()

    def _setup_wigner_indices(self):
        """预计算球谐基的l和m索引"""
        l_indices = []
        m_indices = []
        start_idx = 0

        self.l_slices = {}  # 存储每个l对应的切片范围

        for l in range(self.max_l + 1):
            m_size = 2 * l + 1
            l_indices.extend([l] * m_size)
            m_indices.extend(list(range(-l, l + 1)))

            self.l_slices[l] = slice(start_idx, start_idx + m_size)
            start_idx += m_size

        self.register_buffer('l_indices', torch.tensor(l_indices))
        self.register_buffer('m_indices', torch.tensor(m_indices))

    def wigner_d_matrix(self, rotation_matrix: torch.Tensor, l: int) -> torch.Tensor:
        """
        计算给定旋转矩阵和角动量l的Wigner-D矩阵

        Args:
            rotation_matrix: [*, 3, 3] 旋转矩阵
            l: 球谐基阶数

        Returns:
            wigner_d: [*, 2l+1, 2l+1] Wigner-D矩阵
        """
        # 简化实现：这里应该使用完整的Wigner-D矩阵计算
        # 实际应用中可以使用e3nn等库的实现

        batch_shape = rotation_matrix.shape[:-2]
        device = rotation_matrix.device

        if l == 0:
            # l=0时，Wigner-D矩阵是1x1的单位矩阵
            return torch.ones(*batch_shape, 1, 1, device=device)
        elif l == 1:
            # l=1时，Wigner-D矩阵就是旋转矩阵本身
            return rotation_matrix
        else:
            # 高阶需要递推计算或查表
            # 这里用简化版本（实际需要完整实现）
            size = 2 * l + 1
            identity = torch.eye(size, device=device)
            return identity.expand(*batch_shape, size, size)

    def apply_wigner_d_rotation(self, sh_features: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """
        使用Wigner-D矩阵对球谐基特征进行旋转

        Args:
            sh_features: [*, N_res, H, L+1, 2L+1, R] 球谐基特征
            rotation_matrix: [*, N_res, 3, 3] 旋转矩阵

        Returns:
            rotated_features: [*, N_res, H, L+1, 2L+1, R] 旋转后的特征
        """
        *batch_dims, N_res, H, _, _, R = sh_features.shape
        device = sh_features.device

        # 为每个l阶分别应用Wigner-D旋转
        rotated_parts = []

        for l in range(self.max_l + 1):
            # 获取当前l阶的特征 [*, N_res, H, 2l+1, R]
            l_features = sh_features[..., l, :2 * l + 1, :]

            # 计算Wigner-D矩阵 [*, N_res, 2l+1, 2l+1]
            wigner_d = self.wigner_d_matrix(rotation_matrix, l)

            # 正确的维度处理：
            # l_features: [*, N_res, H, 2l+1, R]
            # wigner_d:   [*, N_res, 2l+1, 2l+1]

            # 方法1: 使用einsum (推荐)
            rotated_l = torch.einsum('...nm,...hmr->...hnr', wigner_d, l_features)

            # 方法2: 手动处理维度
            # wigner_d_expanded = wigner_d.unsqueeze(-3).unsqueeze(-1)  # [*, N_res, 1, 2l+1, 2l+1, 1]
            # l_features_expanded = l_features.unsqueeze(-2)            # [*, N_res, H, 2l+1, 1, R]
            # rotated_l = torch.matmul(wigner_d_expanded, l_features_expanded).squeeze(-2)

            # 方法3: 批量矩阵乘法 (最清晰)
            # 重塑为批量格式进行矩阵乘法
            *batch_dims_nr, N_res, H, m_size, R = l_features.shape

            # 使用 reshape 而不是 view，并确保张量连续性
            l_features_contiguous = l_features.contiguous()  # 确保内存连续

            # 将H和R维度合并进行批量处理
            l_features_reshaped = l_features_contiguous.reshape(*batch_dims_nr, N_res, H * R, m_size).transpose(-2,
                                                                                                                -1)  # [*, N_res, 2l+1, H*R]

            # 批量矩阵乘法: [*, N_res, 2l+1, 2l+1] @ [*, N_res, 2l+1, H*R] -> [*, N_res, 2l+1, H*R]
            rotated_reshaped = torch.matmul(wigner_d, l_features_reshaped)

            # 恢复原始形状
            rotated_l = rotated_reshaped.transpose(-2, -1).reshape(*batch_dims_nr, N_res, H, m_size, R)

            rotated_parts.append(rotated_l)

        # 重新组装
        rotated_features = torch.zeros_like(sh_features)
        for l, rotated_l in enumerate(rotated_parts):
            rotated_features[..., l, :2 * l + 1, :] = rotated_l

        return rotated_features

    def compute_sh_attention(self, q_sh: torch.Tensor, k_sh: torch.Tensor) -> torch.Tensor:
        """
        计算基于球谐基的注意力分数

        Args:
            q_sh: [*, N_res, H, L+1, 2L+1, R] query球谐特征
            k_sh: [*, N_res, H, L+1, 2L+1, R] key球谐特征

        Returns:
            attention_scores: [*, H, N_res, N_res]
        """
        *batch_dims, N_res, H, L_plus_1, _, R = q_sh.shape

        # 重塑为便于计算的形状
        q_flat = q_sh.view(*batch_dims, N_res, H, -1)  # [*, N_res, H, SH_dim]
        k_flat = k_sh.view(*batch_dims, N_res, H, -1)  # [*, N_res, H, SH_dim]

        # 计算注意力分数 [*, H, N_res, N_res]
        attention_scores = torch.matmul(
            q_flat.transpose(-3, -2),  # [*, H, N_res, SH_dim]
            k_flat.transpose(-3, -2).transpose(-2, -1)  # [*, H, SH_dim, N_res]
        )

        # 按l阶加权
        l_weights = self.softplus(self.sh_weights)  # [L+1]

        # 这里可以更精细地处理不同l阶的贡献
        # 简化版本：直接缩放
        scale_factor = math.sqrt(1.0 / (self.sh_dim))
        attention_scores = attention_scores * scale_factor

        return attention_scores

    def forward(
            self,
            s: torch.Tensor,
            z: torch.Tensor,
            r: torch.Tensor,  # 旋转矩阵 [*, N_res, 3, 3]
            mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            s: [*, N_res, C_s] 单一表示
            z: [*, N_res, N_res, C_z] 配对表示
            r: [*, N_res, 3, 3] 旋转矩阵
            mask: [*, N_res] 掩码

        Returns:
            [*, N_res, C_s] 更新后的单一表示
        """
        *batch_dims, N_res, _ = s.shape

        #######################################
        # 生成标量和球谐基特征
        #######################################

        # 标量特征
        q = self.linear_q(s).view(*batch_dims, N_res, self.no_heads, -1)
        kv = self.linear_kv(s).view(*batch_dims, N_res, self.no_heads, -1)
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # 球谐基特征
        q_sh = self.linear_q_sh(s)
        kv_sh = self.linear_kv_sh(s)

        # 重塑为球谐基格式 [*, N_res, H, L+1, 2L+1, R]
        q_sh = q_sh.view(*batch_dims, N_res, self.no_heads, self.max_l + 1, 2 * self.max_l + 1, self.radial_bins)
        kv_sh = kv_sh.view(*batch_dims, N_res, 2 * self.no_heads, self.max_l + 1, 2 * self.max_l + 1, self.radial_bins)

        # 分离k和v的球谐特征
        k_sh, v_sh = torch.split(kv_sh, self.no_heads, dim=-4)

        # 应用Wigner-D旋转
        q_sh_rot = self.apply_wigner_d_rotation(q_sh, r)
        k_sh_rot = self.apply_wigner_d_rotation(k_sh, r)
        v_sh_rot = self.apply_wigner_d_rotation(v_sh, r)

        ##########################
        # 计算注意力分数
        ##########################

        # 标量注意力（保留原有的配对偏置）
        b = self.linear_b(z)
        scalar_att = torch.matmul(
            q.transpose(-3, -2),  # [*, H, N_res, C_hidden]
            k.transpose(-3, -2).transpose(-2, -1)  # [*, H, C_hidden, N_res]
        ) / math.sqrt(self.c_hidden)

        # 球谐基注意力
        sh_att = self.compute_sh_attention(q_sh_rot, k_sh_rot)

        # 组合注意力分数
        total_att = scalar_att + sh_att + b.permute(*range(len(batch_dims)), -1, -3, -2)

        # 应用掩码
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)
        total_att = total_att + square_mask.unsqueeze(-3)

        # Softmax
        a = self.softmax(total_att)

        ################
        # 计算输出
        ################

        # 标量输出
        o_scalar = torch.matmul(a, v.transpose(-3, -2)).transpose(-3, -2)
        o_scalar = o_scalar.reshape(*batch_dims, N_res, -1)

        # 球谐基输出
        v_sh_flat = v_sh_rot.view(*batch_dims, N_res, self.no_heads, -1)
        o_sh = torch.matmul(a, v_sh_flat.transpose(-3, -2)).transpose(-3, -2)
        o_sh = o_sh.reshape(*batch_dims, N_res, -1)

        # 配对表示输出
        pair_z = self.down_z(z)
        o_pair = torch.einsum('...hij,...jkd->...hid', a, pair_z)  # [*, H, N_res, C_z//4]
        o_pair = o_pair.reshape(*batch_dims, N_res, -1)  # [*, N_res, H * C_z//4]

        # 组合所有输出
        o_combined = torch.cat([o_scalar, o_sh, o_pair], dim=-1)

        # 最终线性变换
        output = self.linear_out(o_combined)

        return output

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

# 模拟输入数据的参数
batch_size = 2          # 批量大小
N_res = 5               # 每个批次的残基数量
c_s = 64                # 输入的特征维度 (s)
c_z = 32                # 配对表示的维度 (z)
c_hidden = 128          # 隐藏层维度
no_heads = 4            # 注意力头数
max_l = 3               # 最大球谐阶数
radial_bins = 8         # 径向分区数
eps = 1e-8              # 小的正数
inf = 1e5               # 无穷大的值用于掩码

# 模拟输入数据
s = torch.randn(batch_size, N_res, c_s)        # 单一表示
z = torch.randn(batch_size, N_res, N_res, c_z) # 配对表示
mask = torch.ones(batch_size, N_res)           # 掩码（所有位置有效）

# 使用Scipy生成旋转矩阵
# 这里我们生成随机的旋转矩阵，使用 scipy.spatial.transform.Rotation 来生成
rotation_matrices = []
for _ in range(batch_size):
    # 随机生成旋转
    r = R.random(N_res)  # 生成N_res个随机旋转
    rotation_matrices.append(r.as_matrix())  # 转换为旋转矩阵

# 将旋转矩阵从 numpy 转换为 torch tensor，形状为 [batch_size, N_res, 3, 3]
r = torch.tensor(np.stack(rotation_matrices), dtype=torch.float32)

# 实例化模型
model = SphericalHarmonicsIPA(
    c_s=c_s,
    c_z=c_z,
    c_hidden=c_hidden,
    no_heads=no_heads,
    max_l=max_l,
    radial_bins=radial_bins,
    eps=eps,
    inf=inf
)

# 前向传播
output = model(s, z, r, mask)

# 打印输出的形状
print("输出形状:", output.shape)
