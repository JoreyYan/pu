

from typing import Optional, Callable, List, Sequence
from openfold.utils.rigid_utils import Rigid
from models.ipa_pytorch import Linear, _calculate_fan, _prod,ipa_point_weights_init_,permute_final_dims,flatten_final_dims
from e3nn.o3 import spherical_harmonics
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
class InvariantPointAttention_3DROPE(nn.Module):
    """
    Modified Invariant Point Attention with Group-wise 3D-RoPE integration.
    """

    def __init__(
            self,
            ipa_conf,
            inf: float = 1e5,
            eps: float = 1e-8,
            use_groupwise_rope: bool = True,
            rope_aggregation: str = 'sum',  # 'mean', 'sum', 'max', 'learned'
    ):
        """
        Args:
            ipa_conf: IPA configuration object
            group_dim: Group dimension, c_hidden will be group_dim * 3
            inf: Large value for masking
            eps: Small epsilon for numerical stability
            use_groupwise_rope: Whether to use group-wise 3D-RoPE
            rope_aggregation: How to aggregate group-wise scores
        """
        super(InvariantPointAttention_3DROPE, self).__init__()
        self._ipa_conf = ipa_conf
        group_dim=ipa_conf.c_hidden
        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.group_dim = group_dim
        self.c_hidden = group_dim * 3  # Modified: c_hidden = group_dim * 3
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        # Group-wise 3D-RoPE parameters
        self.use_groupwise_rope = use_groupwise_rope
        self.rope_aggregation = rope_aggregation

        if self.use_groupwise_rope:
            # Number of 3D groups per head
            self.num_3d_groups = group_dim  # Since c_hidden = group_dim * 3

            # Learned aggregation weights if needed
            if rope_aggregation == 'learned':
                self.group_weights = nn.Parameter(torch.ones(self.num_3d_groups) / self.num_3d_groups)

        # Modified linear layers with new c_hidden
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.group_weights = nn.Parameter(torch.zeros(( self.num_3d_groups)))
        ipa_point_weights_init_(self.group_weights)

        concat_out_dim = (
                self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def apply_groupwise_rotations(self, features, r):
        """
        Apply group-wise 3D rotations to feature vectors.

        Args:
            features: [*, N_res, H, C_hidden] feature vectors
            r: Rotation object [*, N_res]

        Returns:
            features_rotated: [*, N_res, H, G, 3] rotated and grouped features
        """
        if not self.use_groupwise_rope:
            return features

        # Get dimensions
        *batch_dims, N_res, H, C_hidden = features.shape

        # Reshape to group structure: [*, N_res, H, C_hidden] -> [*, N_res, H, G, 3]
        # where C_hidden = G * 3
        features_grouped = features.view(*batch_dims, N_res, H, self.num_3d_groups, 3)

        # Apply rotation using the Rotation class - this can be done in parallel!
        # r is [*, N_res], features_grouped is [*, N_res, H, G, 3]
        # We need to expand r for the H and G dimensions
        features_rotated = r[..., None, None].apply_rot(features_grouped)

        return features_rotated

    def compute_groupwise_attention(self, q_rot, k_rot):
        """
        Compute group-wise attention scores.

        Args:
            q_rot: [*, N_res, H, G, 3] grouped and rotated query features
            k_rot: [*, N_res, H, G, 3] grouped and rotated key features

        Returns:
            group_scores: [*, H, G, N_res, N_res] attention scores for each group
        """
        if not self.use_groupwise_rope:
            return None

        *batch_dims, N_res, H, G, _ = q_rot.shape

        # Reshape for batch matrix multiplication
        # [*, N_res, H, G, 3] -> [*, H, G, N_res, 3]
        q_for_attn = q_rot.permute(*range(len(batch_dims)), -3, -2, -4, -1)
        # [*, N_res, H, G, 3] -> [*, H, G, 3, N_res]
        k_for_attn = k_rot.permute(*range(len(batch_dims)), -3, -2, -1, -4)

        # Compute attention scores for each group
        # [*, H, G, N_res, 3] @ [*, H, G, 3, N_res] -> [*, H, G, N_res, N_res]
        group_scores = torch.matmul(q_for_attn, k_for_attn)

        # Apply scaling (sqrt(3) since each group has 3 dimensions)
        group_scores = group_scores / math.sqrt(3)

        return group_scores

    def aggregate_group_scores(self, group_scores):
        """
        Aggregate attention scores from different 3D groups using head_weights.

        Args:
            group_scores: [*, H, G, N_res, N_res]

        Returns:
            final_scores: [*, H, N_res, N_res]
        """
        if not self.use_groupwise_rope or group_scores is None:
            return torch.zeros(group_scores.shape[:-3] + group_scores.shape[-2:],
                               device=group_scores.device, dtype=group_scores.dtype)

        # Apply head_weights to group dimension for fusion
        head_weights = self.softplus(self.group_weights).view(
            *((1,) * len(group_scores.shape[:-3]) + (-1, 1,  1))
        )  # [*, H, 1,  1]

        # Scale head weights
        head_weights = head_weights * math.sqrt(
            1.0 / (1 * (self.num_3d_groups ))
        )

        # Apply head weights to group scores
        weighted_group_scores = group_scores * head_weights

        # Aggregate based on the method specified
        if self.rope_aggregation == 'mean':
            final_scores = weighted_group_scores.mean(dim=-3)  # Average over G
        elif self.rope_aggregation == 'sum':
            final_scores = weighted_group_scores.sum(dim=-3)  # Weighted sum
        elif self.rope_aggregation == 'max':
            # Softmax-based max pooling (soft attention over group dim)
            softmax_weights = F.softmax(weighted_group_scores, dim=-3)  # [*, H, G, N, N]
            final_scores = (weighted_group_scores * softmax_weights).sum(dim=-3)
        elif self.rope_aggregation == 'hardmax':
            # True max pooling over group dimension
            final_scores, _ = weighted_group_scores.max(dim=-3)
        elif self.rope_aggregation == 'learned':
            weights = F.softmax(self.group_weights, dim=0)  # [G]
            weights = weights.view(*([1] * (len(group_scores.shape) - 3)), -1, 1, 1)
            weighted_scores = weighted_group_scores * weights
            final_scores = weighted_scores.sum(dim=-3)
        else:
            raise ValueError(f"Unknown aggregation method: {self.rope_aggregation}")

        return final_scores

    def forward(
            self,
            s: torch.Tensor,
            z: Optional[torch.Tensor],
            r: Rigid,  # This should be the Rotation class
            mask: torch.Tensor,
            _offload_inference: bool = False,
            _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with Group-wise 3D-RoPE integration.
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, self.c_hidden))
        kv = kv.view(kv.shape[:-1] + (self.no_heads, 2 * self.c_hidden))
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # Generate point coordinates (keep original point-based attention for v_pts)
        q_pts = self.linear_q_points(s)
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)  # Parallel rotation application
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))

        kv_pts = self.linear_kv_points(s)
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        # kv_pts = r[..., None].apply(kv_pts)  # Parallel rotation application
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))
        k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

        ##########################
        # Compute attention scores
        ##########################
        # Scalar attention (unchanged)
        b = self.linear_b(z[0])
        if _offload_inference:
            z[0] = z[0].cpu()

        # Group-wise 3D-RoPE attention
        if self.use_groupwise_rope:
            # Apply group-wise rotations to q and k features
            q_rot = self.apply_groupwise_rotations(q, r)  # [*, N_res, H, G, 3]
            k_rot = self.apply_groupwise_rotations(k, r)  # [*, N_res, H, G, 3]

            # Compute group-wise attention scores
            group_scores = self.compute_groupwise_attention(q_rot, k_rot)  # [*, H, G, N_res, N_res]

            # Aggregate group scores with head_weights fusion
            rope_att = self.aggregate_group_scores(group_scores)  # [*, H, N_res, N_res]



            # Add bias term
            rope_att += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

            a = rope_att
        else:
            # Original scalar attention
            a = torch.matmul(
                permute_final_dims(q, (1, 0, 2)),
                permute_final_dims(k, (1, 2, 0)),
            )
            a *= math.sqrt(1.0 / (3 * self.c_hidden))
            a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # Apply mask and softmax
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output (unchanged)
        ################
        o = torch.matmul(a, v.transpose(-2, -3)).transpose(-2, -3)
        o = flatten_final_dims(o, 2)

        o_pt = torch.sum(
            (
                    a[..., None, :, :, None]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        # o_pt = r[..., None, None].invert_apply(o_pt)

        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(o_pt_dists, 2)
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if _offload_inference:
            z[0] = z[0].to(o_pt.device)

        pair_z = self.down_z(z[0])
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        s = self.linear_out(torch.cat(o_feats, dim=-1))

        return s





def rope_inspired_3d_frequencies(num_freqs=4, base=8.0, max_freq=1.0):
    """RoPE 启发的 3D 旋转频率设计"""
    freqs = []
    for i in range(num_freqs):
        freq = base ** (-2 * i / num_freqs)
        freq = min(freq * max_freq, max_freq)
        freqs.append(freq)
    return torch.tensor(freqs, dtype=torch.float32)


def hat(v):
    """向量的反对称矩阵 (hat operator)"""
    batch_shape = v.shape[:-1]
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zeros = torch.zeros_like(x)
    hat_matrix = torch.stack([
        torch.stack([zeros, -z, y], dim=-1),
        torch.stack([z, zeros, -x], dim=-1),
        torch.stack([-y, x, zeros], dim=-1)
    ], dim=-2)
    return hat_matrix


class InvariantPointAttention_freROPE(nn.Module):
    """
    Modified Invariant Point Attention with RoPE-inspired multi-frequency 3D rotation.
    """

    def __init__(
            self,
            ipa_conf,
            inf: float = 1e5,
            eps: float = 1e-8,
            use_groupwise_rope: bool = True,
            rope_aggregation: str = 'sum',
            # RoPE 启发参数
            num_frequencies: int = 4,
            freq_base: float = 8.0,
            learnable_freqs: bool = True,
    ):
        super(InvariantPointAttention_freROPE, self).__init__()
        self._ipa_conf = ipa_conf
        group_dim = ipa_conf.c_hidden
        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.group_dim = group_dim
        self.c_hidden = group_dim * 3
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        # Group-wise 3D-RoPE parameters
        self.use_groupwise_rope = use_groupwise_rope
        self.rope_aggregation = rope_aggregation
        self.num_frequencies = num_frequencies

        if self.use_groupwise_rope:
            self.num_3d_groups = group_dim

            # RoPE 启发的频率设计
            base_freqs = rope_inspired_3d_frequencies(num_frequencies, freq_base)

            if learnable_freqs:
                # 可学习的频率调整
                self.log_freq_adjustments = nn.Parameter(torch.zeros(num_frequencies))
                self.register_buffer('base_freqs', base_freqs)
                self.learnable_freqs = True
            else:
                # 固定频率
                self.register_buffer('rope_freqs', base_freqs)
                self.learnable_freqs = False

            # 组权重和频率权重
            self.group_weights = nn.Parameter(torch.zeros(self.num_3d_groups))
            self.freq_weights = nn.Parameter(torch.ones(num_frequencies) / num_frequencies)

        # 原有的线性层
        hc = self.c_hidden * self.no_heads
        self.linear_q = nn.Linear(self.c_s, hc)
        self.linear_kv = nn.Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = nn.Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = nn.Linear(self.c_s, hpkv)

        self.linear_b = nn.Linear(self.c_z, self.no_heads)
        self.down_z = nn.Linear(self.c_z, self.c_z // 4)

        # 初始化组权重
        nn.init.normal_(self.group_weights, std=0.02)

        concat_out_dim = self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        self.linear_out = nn.Linear(self.no_heads * concat_out_dim, self.c_s)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def get_frequencies(self):
        """获取当前频率"""
        if self.learnable_freqs:
            adjustments = torch.exp(self.log_freq_adjustments)
            return self.base_freqs * adjustments
        else:
            return self.rope_freqs

    def apply_groupwise_rotations(self, features, r):
        """
        Apply RoPE-inspired multi-frequency rotations to feature vectors.

        Args:
            features: [*, N_res, H, C_hidden] feature vectors
            r: Rotation object [*, N_res]

        Returns:
            features_rotated: [*, N_res, H, G, F, 3] multi-frequency rotated features
        """
        if not self.use_groupwise_rope:
            return features

        *batch_dims, N_res, H, C_hidden = features.shape

        # 重塑为 3D 组
        features_grouped = features.view(*batch_dims, N_res, H, self.num_3d_groups, 3)

        # 获取旋转向量和频率
        rotvec = r._rots.get_rotvec()  # [..., N_res, 3]
        angle = torch.norm(rotvec, dim=-1, keepdim=True) + 1e-8  # [..., N_res, 1]
        axis = F.normalize(rotvec, dim=-1)  # [..., N_res, 3]

        freqs = self.get_frequencies().to(features.device)  # [F]


        # 扩展维度进行广播
        axis = axis.unsqueeze(-2).unsqueeze(-2)  # [..., N_res, 1, 1, 3]
        angle = angle.unsqueeze(-1).unsqueeze(-1)  # [..., N_res, 1, 1, 1]

        # 多频率调制
        theta_multi = angle * freqs.view(1, 1, 1, 1, -1)  # [..., N_res, 1, 1, F]

        # 计算旋转矩阵 (Rodriguez 公式)
        axis_hat = hat(axis)  # [..., N_res, 1, 1, 3, 3]
        axis_hat = axis_hat.unsqueeze(-4)  # [..., N_res, 1, 1, 1, 3, 3]

        I = torch.eye(3, device=features.device, dtype=features.dtype)
        I = I.view(*([1] * len(batch_dims)), 1, 1, 1, 1, 3, 3)

        sin_theta = torch.sin(theta_multi).unsqueeze(-1).unsqueeze(-1)
        cos_theta = torch.cos(theta_multi).unsqueeze(-1).unsqueeze(-1)

        R_multi = (I +
                   sin_theta * axis_hat +
                   (1 - cos_theta) * torch.matmul(axis_hat, axis_hat))

        # 应用多频率旋转
        features_expanded = features_grouped.unsqueeze(-2).unsqueeze(-1)  # [..., N_res, H, G, 1, 3, 1]
        rotated_multi = torch.matmul(R_multi, features_expanded).squeeze(-1)  # [..., N_res, H, G, F, 3]

        return rotated_multi

    def compute_groupwise_attention(self, q_rot, k_rot):
        """
        Compute group-wise + frequency-wise attention scores.

        Args:
            q_rot: [*, N_res, H, G, F, 3] multi-frequency rotated query features
            k_rot: [*, N_res, H, G, F, 3] multi-frequency rotated key features

        Returns:
            group_scores: [*, H, G*F, N_res, N_res] attention scores
        """
        if not self.use_groupwise_rope:
            return None

        *batch_dims, N_res, H, G, F, _ = q_rot.shape
        GF = G * F

        # 合并 G 和 F 维度
        q_rot = q_rot.view(*batch_dims, N_res, H, GF, 3)
        k_rot = k_rot.view(*batch_dims, N_res, H, GF, 3)

        # 重排维度进行 attention 计算
        q_for_attn = q_rot.permute(*range(len(batch_dims)), -3, -2, -4, -1)  # [..., H, GF, N_res, 3]
        k_for_attn = k_rot.permute(*range(len(batch_dims)), -3, -2, -1, -4)  # [..., H, GF, 3, N_res]

        # 计算 attention 分数
        group_scores = torch.matmul(q_for_attn, k_for_attn)  # [..., H, GF, N_res, N_res]
        group_scores = group_scores / math.sqrt(3)

        return group_scores

    def aggregate_group_scores(self, group_scores):
        """
        Aggregate attention scores from different groups and frequencies.

        Args:
            group_scores: [*, H, G*F, N_res, N_res]

        Returns:
            final_scores: [*, H, N_res, N_res]
        """
        if not self.use_groupwise_rope or group_scores is None:
            return torch.zeros(group_scores.shape[:-3] + group_scores.shape[-2:],
                               device=group_scores.device, dtype=group_scores.dtype)

        *batch_dims, H, GF, N, _ = group_scores.shape
        G, Fre = self.num_3d_groups, self.num_frequencies

        # 重塑回 [*, H, G, F, N, N]
        scores_reshaped = group_scores.view(*batch_dims, H, G, Fre, N, N)

        # 1. 频率维度聚合
        freq_weights = F.softmax(self.freq_weights, dim=0)  # [Fre]
        freq_weights = freq_weights.view(*([1] * len(batch_dims)), 1, 1, -1, 1, 1)
        freq_aggregated = torch.sum(scores_reshaped * freq_weights, dim=-3)  # [*, H, G, N, N]

        # 2. 组维度聚合
        group_weights = self.softplus(self.group_weights)  # [G]
        group_weights = group_weights * math.sqrt(1.0 / self.num_3d_groups)
        group_weights = group_weights.view(*([1] * len(batch_dims)), 1, -1, 1, 1)

        weighted_scores = freq_aggregated * group_weights

        if self.rope_aggregation == 'mean':
            final_scores = weighted_scores.mean(dim=-3)
        elif self.rope_aggregation == 'sum':
            final_scores = weighted_scores.sum(dim=-3)
        elif self.rope_aggregation == 'max':
            softmax_weights = F.softmax(weighted_scores, dim=-3)
            final_scores = (weighted_scores * softmax_weights).sum(dim=-3)
        elif self.rope_aggregation == 'hardmax':
            final_scores, _ = weighted_scores.max(dim=-3)
        else:
            raise ValueError(f"Unknown aggregation method: {self.rope_aggregation}")

        return final_scores

    def forward(
            self,
            s: torch.Tensor,
            z: torch.Tensor,
            r,  # Rotation object
            mask: torch.Tensor,
            _offload_inference: bool = False,
            _z_reference_list=None,
    ) -> torch.Tensor:
        """Forward pass with RoPE-inspired multi-frequency 3D rotation."""

        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        # 生成 scalar 和 point 激活
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        q = q.view(q.shape[:-1] + (self.no_heads, self.c_hidden))
        kv = kv.view(kv.shape[:-1] + (self.no_heads, 2 * self.c_hidden))
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # 生成点坐标 (保持原有的点注意力)

        kv_pts = self.linear_kv_points(s)
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))
        k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

        # 计算 attention 分数
        b = self.linear_b(z[0])
        if _offload_inference:
            z[0] = z[0].cpu()

        if self.use_groupwise_rope:
            # RoPE 启发的多频率旋转 attention
            q_rot = self.apply_groupwise_rotations(q, r)  # [*, N_res, H, G, F, 3]
            k_rot = self.apply_groupwise_rotations(k, r)  # [*, N_res, H, G, F, 3]

            group_scores = self.compute_groupwise_attention(q_rot, k_rot)  # [*, H, G*F, N_res, N_res]
            rope_att = self.aggregate_group_scores(group_scores)  # [*, H, N_res, N_res]

            # 添加 bias

            rope_att += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))
            a = rope_att
        else:
            # 原始标量注意力
            a = torch.matmul(
                permute_final_dims(q, (1, 0, 2)),
                permute_final_dims(k, (1, 2, 0)),
            )
            a *= math.sqrt(1.0 / (3 * self.c_hidden))
            a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # 应用 mask 和 softmax
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        # 计算输出 (保持原有逻辑)


        o = torch.matmul(a, v.transpose(-2, -3)).transpose(-2, -3)

        # o_ptx=o.view(*o.shape[:-1], -1, 3).permute(0, 2, 4, 1, 3)

        o = flatten_final_dims(o, 2)

        o_pt = torch.sum(
            (a[..., None, :, :, None] * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]),
            dim=-2,
        )

        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(o_pt_dists, 2)
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if _offload_inference:
            z[0] = z[0].to(o_pt.device)

        pair_z = self.down_z(z[0])
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]
        s = self.linear_out(torch.cat(o_feats, dim=-1))

        return s


class InvariantPointAttention_onevpts_freROPE(nn.Module):
    """
    Modified Invariant Point Attention with RoPE-inspired multi-frequency 3D rotation.
    """

    def __init__(
            self,
            ipa_conf,
            inf: float = 1e5,
            eps: float = 1e-8,
            use_groupwise_rope: bool = True,
            rope_aggregation: str = 'sum',
            # RoPE 启发参数
            num_frequencies: int = 4,
            freq_base: float = 8.0,
            learnable_freqs: bool = False,
    ):
        super(InvariantPointAttention_onevpts_freROPE, self).__init__()
        self._ipa_conf = ipa_conf
        group_dim = ipa_conf.c_hidden
        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.group_dim = group_dim
        self.c_hidden = group_dim * 3
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        # Group-wise 3D-RoPE parameters
        self.use_groupwise_rope = use_groupwise_rope
        self.rope_aggregation = rope_aggregation
        self.num_frequencies = num_frequencies

        if self.use_groupwise_rope:
            self.num_3d_groups = group_dim

            # RoPE 启发的频率设计
            base_freqs = rope_inspired_3d_frequencies(num_frequencies, freq_base)

            if learnable_freqs:
                # 可学习的频率调整
                self.log_freq_adjustments = nn.Parameter(torch.zeros(num_frequencies))
                self.register_buffer('base_freqs', base_freqs)
                self.learnable_freqs = True
            else:
                # 固定频率
                self.register_buffer('rope_freqs', base_freqs)
                self.learnable_freqs = False

            # 组权重和频率权重
            self.group_weights = nn.Parameter(torch.zeros(self.num_3d_groups))
            self.freq_weights = nn.Parameter(torch.ones(num_frequencies) / num_frequencies)

        # 原有的线性层
        hc = self.c_hidden * self.no_heads
        self.linear_q = nn.Linear(self.c_s, hc)
        self.linear_kv = nn.Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = nn.Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = nn.Linear(self.c_s, hpkv)

        self.linear_b = nn.Linear(self.c_z, self.no_heads)
        self.down_z = nn.Linear(self.c_z, self.c_z // 4)

        # 初始化组权重
        nn.init.normal_(self.group_weights, std=0.02)

        concat_out_dim = self.c_z // 4 + self.c_hidden + self.group_dim * 4
        self.linear_out = nn.Linear(self.no_heads * concat_out_dim, self.c_s)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def get_frequencies(self):
        """获取当前频率"""
        if self.learnable_freqs:
            adjustments = torch.exp(self.log_freq_adjustments)
            return self.base_freqs * adjustments
        else:
            return self.rope_freqs

    def apply_groupwise_rotations(self, features, r):
        """
        Apply RoPE-inspired multi-frequency rotations to feature vectors.

        Args:
            features: [*, N_res, H, C_hidden] feature vectors
            r: Rotation object [*, N_res]

        Returns:
            features_rotated: [*, N_res, H, G, F, 3] multi-frequency rotated features
        """
        if not self.use_groupwise_rope:
            return features

        *batch_dims, N_res, H, C_hidden = features.shape

        # 重塑为 3D 组
        features_grouped = features.view(*batch_dims, N_res, H, self.num_3d_groups, 3)

        # 获取旋转向量和频率
        rotvec = r._rots.get_rotvec()  # [..., N_res, 3]
        angle = torch.norm(rotvec, dim=-1, keepdim=True) + 1e-8  # [..., N_res, 1]
        axis = F.normalize(rotvec, dim=-1)  # [..., N_res, 3]

        freqs = self.get_frequencies().to(features.device)  # [F]


        # 扩展维度进行广播
        axis = axis.unsqueeze(-2).unsqueeze(-2)  # [..., N_res, 1, 1, 3]
        angle = angle.unsqueeze(-1).unsqueeze(-1)  # [..., N_res, 1, 1, 1]

        # 多频率调制
        theta_multi = angle * freqs.view(1, 1, 1, 1, -1)  # [..., N_res, 1, 1, F]

        # 计算旋转矩阵 (Rodriguez 公式)
        axis_hat = hat(axis)  # [..., N_res, 1, 1, 3, 3]
        axis_hat = axis_hat.unsqueeze(-4)  # [..., N_res, 1, 1, 1, 3, 3]

        I = torch.eye(3, device=features.device, dtype=features.dtype)
        I = I.view(*([1] * len(batch_dims)), 1, 1, 1, 1, 3, 3)

        sin_theta = torch.sin(theta_multi).unsqueeze(-1).unsqueeze(-1)
        cos_theta = torch.cos(theta_multi).unsqueeze(-1).unsqueeze(-1)

        R_multi = (I +
                   sin_theta * axis_hat +
                   (1 - cos_theta) * torch.matmul(axis_hat, axis_hat))

        # 应用多频率旋转
        features_expanded = features_grouped.unsqueeze(-2).unsqueeze(-1)  # [..., N_res, H, G, 1, 3, 1]
        rotated_multi = torch.matmul(R_multi, features_expanded).squeeze(-1)  # [..., N_res, H, G, F, 3]

        return rotated_multi

    def compute_groupwise_attention(self, q_rot, k_rot):
        """
        Compute group-wise + frequency-wise attention scores.

        Args:
            q_rot: [*, N_res, H, G, F, 3] multi-frequency rotated query features
            k_rot: [*, N_res, H, G, F, 3] multi-frequency rotated key features

        Returns:
            group_scores: [*, H, G*F, N_res, N_res] attention scores
        """
        if not self.use_groupwise_rope:
            return None

        *batch_dims, N_res, H, G, F, _ = q_rot.shape
        GF = G * F

        # 合并 G 和 F 维度
        q_rot = q_rot.view(*batch_dims, N_res, H, GF, 3)
        k_rot = k_rot.view(*batch_dims, N_res, H, GF, 3)

        # 重排维度进行 attention 计算
        q_for_attn = q_rot.permute(*range(len(batch_dims)), -3, -2, -4, -1)  # [..., H, GF, N_res, 3]
        k_for_attn = k_rot.permute(*range(len(batch_dims)), -3, -2, -1, -4)  # [..., H, GF, 3, N_res]

        # 计算 attention 分数
        group_scores = torch.matmul(q_for_attn, k_for_attn)  # [..., H, GF, N_res, N_res]
        group_scores = group_scores / math.sqrt(3)

        return group_scores

    def aggregate_group_scores(self, group_scores):
        """
        Aggregate attention scores from different groups and frequencies.

        Args:
            group_scores: [*, H, G*F, N_res, N_res]

        Returns:
            final_scores: [*, H, N_res, N_res]
        """
        if not self.use_groupwise_rope or group_scores is None:
            return torch.zeros(group_scores.shape[:-3] + group_scores.shape[-2:],
                               device=group_scores.device, dtype=group_scores.dtype)

        *batch_dims, H, GF, N, _ = group_scores.shape
        G, Fre = self.num_3d_groups, self.num_frequencies

        # 重塑回 [*, H, G, F, N, N]
        scores_reshaped = group_scores.view(*batch_dims, H, G, Fre, N, N)

        # 1. 频率维度聚合
        freq_weights = F.softmax(self.freq_weights, dim=0)  # [Fre]
        freq_weights = freq_weights.view(*([1] * len(batch_dims)), 1, 1, -1, 1, 1)
        freq_aggregated = torch.sum(scores_reshaped * freq_weights, dim=-3)  # [*, H, G, N, N]

        # 2. 组维度聚合
        group_weights = self.softplus(self.group_weights)  # [G]
        group_weights = group_weights * math.sqrt(1.0 / self.num_3d_groups)
        group_weights = group_weights.view(*([1] * len(batch_dims)), 1, -1, 1, 1)

        weighted_scores = freq_aggregated * group_weights

        if self.rope_aggregation == 'mean':
            final_scores = weighted_scores.mean(dim=-3)
        elif self.rope_aggregation == 'sum':
            final_scores = weighted_scores.sum(dim=-3)
        elif self.rope_aggregation == 'max':
            softmax_weights = F.softmax(weighted_scores, dim=-3)
            final_scores = (weighted_scores * softmax_weights).sum(dim=-3)
        elif self.rope_aggregation == 'hardmax':
            final_scores, _ = weighted_scores.max(dim=-3)
        else:
            raise ValueError(f"Unknown aggregation method: {self.rope_aggregation}")

        return final_scores

    def forward(
            self,
            s: torch.Tensor,
            z: torch.Tensor,
            r,  # Rotation object
            mask: torch.Tensor,
            _offload_inference: bool = False,
            _z_reference_list=None,
    ) -> torch.Tensor:
        """Forward pass with RoPE-inspired multi-frequency 3D rotation."""

        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        # 生成 scalar 和 point 激活
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        q = q.view(q.shape[:-1] + (self.no_heads, self.c_hidden))
        kv = kv.view(kv.shape[:-1] + (self.no_heads, 2 * self.c_hidden))
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # 生成点坐标 (保持原有的点注意力)

        kv_pts = self.linear_kv_points(s)
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))
        k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

        # 计算 attention 分数
        b = self.linear_b(z[0])
        if _offload_inference:
            z[0] = z[0].cpu()

        if self.use_groupwise_rope:
            # RoPE 启发的多频率旋转 attention
            q_rot = self.apply_groupwise_rotations(q, r)  # [*, N_res, H, G, F, 3]
            k_rot = self.apply_groupwise_rotations(k, r)  # [*, N_res, H, G, F, 3]

            group_scores = self.compute_groupwise_attention(q_rot, k_rot)  # [*, H, G*F, N_res, N_res]
            rope_att = self.aggregate_group_scores(group_scores)  # [*, H, N_res, N_res]

            # 添加 bias

            rope_att += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))
            a = rope_att
        else:
            # 原始标量注意力
            a = torch.matmul(
                permute_final_dims(q, (1, 0, 2)),
                permute_final_dims(k, (1, 2, 0)),
            )
            a *= math.sqrt(1.0 / (3 * self.c_hidden))
            a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # 应用 mask 和 softmax
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        # 计算输出 (保持原有逻辑)


        o = torch.matmul(a, v.transpose(-2, -3)).transpose(-2, -3)

        o_ptx=o.view(*o.shape[:-1], -1, 3).permute(0, 2, 4, 1, 3)

        o = flatten_final_dims(o, 2)

        # o_pt = torch.sum(
        #     (a[..., None, :, :, None] * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]),
        #     dim=-2,
        # )

        o_pt = permute_final_dims(o_ptx, (2, 0, 3, 1))
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(o_pt_dists, 2)
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if _offload_inference:
            z[0] = z[0].to(o_pt.device)

        pair_z = self.down_z(z[0])
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]
        s = self.linear_out(torch.cat(o_feats, dim=-1))

        return s



class InvariantPointAttention_SPH(nn.Module):
    """
    Modified Invariant Point Attention with Group-wise 3D-RoPE integration.
    """

    def __init__(
            self,
            ipa_conf,
            inf: float = 1e5,
            eps: float = 1e-8,
            use_groupwise_rope: bool = True,
            rope_aggregation: str = 'sum',  # 'mean', 'sum', 'max', 'learned'
    ):
        """
        Args:
            ipa_conf: IPA configuration object
            group_dim: Group dimension, c_hidden will be group_dim * 3
            inf: Large value for masking
            eps: Small epsilon for numerical stability
            use_groupwise_rope: Whether to use group-wise 3D-RoPE
            rope_aggregation: How to aggregate group-wise scores
        """
        super(InvariantPointAttention_SPH, self).__init__()
        self._ipa_conf = ipa_conf
        group_dim=ipa_conf.c_hidden
        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.group_dim = group_dim
        self.c_hidden = group_dim * 3  # Modified: c_hidden = group_dim * 3
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.inf = inf
        self.eps = eps

        self.l_max=[0,1,2,3]

        # Group-wise 3D-RoPE parameters
        self.use_groupwise_rope = use_groupwise_rope
        self.rope_aggregation = rope_aggregation

        if self.use_groupwise_rope:
            # Number of 3D groups per head
            self.num_3d_groups = group_dim  # Since c_hidden = group_dim * 3

            # Learned aggregation weights if needed
            if rope_aggregation == 'learned':
                self.group_weights = nn.Parameter(torch.ones(self.num_3d_groups) / self.num_3d_groups)

        # Modified linear layers with new c_hidden
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.group_weights = nn.Parameter(torch.zeros(( self.num_3d_groups)))
        ipa_point_weights_init_(self.group_weights)

        concat_out_dim = (
                self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def apply_groupwise_rotations(self, features, r):
        """
        Apply group-wise 3D rotations to feature vectors.

        Args:
            features: [*, N_res, H, C_hidden] feature vectors
            r: Rotation object [*, N_res]

        Returns:
            features_rotated: [*, N_res, H, G, 3] rotated and grouped features
        """
        if not self.use_groupwise_rope:
            return features

        # Get dimensions
        *batch_dims, N_res, H, C_hidden = features.shape

        # Reshape to group structure: [*, N_res, H, C_hidden] -> [*, N_res, H, G, 3]
        # where C_hidden = G * 3
        features_grouped = features.view(*batch_dims, N_res, H, self.num_3d_groups, 3)

        # Apply rotation using the Rotation class - this can be done in parallel!
        # r is [*, N_res], features_grouped is [*, N_res, H, G, 3]
        # We need to expand r for the H and G dimensions
        features_rotated = r[..., None, None].apply_rot(features_grouped)

        return features_rotated
    def spherical_harmonic_expansion(self, features):
        """
        Perform spherical harmonic expansion on 3D vectors.
        Args:
            features: [B, N, H, G, 3]
        Returns:
            spherical_harmonics: [B, N, H, G, l_max]
        """
        # Perform spherical harmonics expansion on each 3D vector in the batch
        # Using e3nn's spherical_harmonics function to compute the spherical harmonic coefficients
        Y = spherical_harmonics(self.l_max, features, normalize="component")
        return Y
    def compute_groupwise_attention(self, q_rot, k_rot):
        """
        Compute group-wise attention scores.

        Args:
            q_rot: [*, N_res, H, G, 3] grouped and rotated query features
            k_rot: [*, N_res, H, G, 3] grouped and rotated key features

        Returns:
            group_scores: [*, H, G, N_res, N_res] attention scores for each group
        """
        if not self.use_groupwise_rope:
            return None

        *batch_dims, N_res, H, G, _ = q_rot.shape

        # Reshape for batch matrix multiplication
        # [*, N_res, H, G, 3] -> [*, H, G, N_res, 3]
        q_for_attn = q_rot.permute(*range(len(batch_dims)), -3, -2, -4, -1)
        # [*, N_res, H, G, 3] -> [*, H, G, 3, N_res]
        k_for_attn = k_rot.permute(*range(len(batch_dims)), -3, -2, -1, -4)

        # Compute attention scores for each group
        # [*, H, G, N_res, 3] @ [*, H, G, 3, N_res] -> [*, H, G, N_res, N_res]
        group_scores = torch.matmul(q_for_attn, k_for_attn)

        # Apply scaling (sqrt(3) since each group has 3 dimensions)
        group_scores = group_scores / math.sqrt(3)

        return group_scores

    def aggregate_group_scores(self, group_scores):
        """
        Aggregate attention scores from different 3D groups using head_weights.

        Args:
            group_scores: [*, H, G, N_res, N_res]

        Returns:
            final_scores: [*, H, N_res, N_res]
        """
        if not self.use_groupwise_rope or group_scores is None:
            return torch.zeros(group_scores.shape[:-3] + group_scores.shape[-2:],
                               device=group_scores.device, dtype=group_scores.dtype)

        # Apply head_weights to group dimension for fusion
        head_weights = self.softplus(self.group_weights).view(
            *((1,) * len(group_scores.shape[:-3]) + (-1, 1,  1))
        )  # [*, H, 1,  1]

        # Scale head weights
        head_weights = head_weights * math.sqrt(
            1.0 / (1 * (self.num_3d_groups ))
        )

        # Apply head weights to group scores
        weighted_group_scores = group_scores * head_weights

        # Aggregate based on the method specified
        if self.rope_aggregation == 'mean':
            final_scores = weighted_group_scores.mean(dim=-3)  # Average over G
        elif self.rope_aggregation == 'sum':
            final_scores = weighted_group_scores.sum(dim=-3)  # Weighted sum
        elif self.rope_aggregation == 'max':
            # Softmax-based max pooling (soft attention over group dim)
            softmax_weights = F.softmax(weighted_group_scores, dim=-3)  # [*, H, G, N, N]
            final_scores = (weighted_group_scores * softmax_weights).sum(dim=-3)
        elif self.rope_aggregation == 'hardmax':
            # True max pooling over group dimension
            final_scores, _ = weighted_group_scores.max(dim=-3)
        elif self.rope_aggregation == 'learned':
            weights = F.softmax(self.group_weights, dim=0)  # [G]
            weights = weights.view(*([1] * (len(group_scores.shape) - 3)), -1, 1, 1)
            weighted_scores = weighted_group_scores * weights
            final_scores = weighted_scores.sum(dim=-3)
        else:
            raise ValueError(f"Unknown aggregation method: {self.rope_aggregation}")

        return final_scores

    def forward(
            self,
            s: torch.Tensor,
            z: Optional[torch.Tensor],
            r: Rigid,  # This should be the Rotation class
            mask: torch.Tensor,
            _offload_inference: bool = False,
            _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with Group-wise 3D-RoPE integration.
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, self.c_hidden))
        kv = kv.view(kv.shape[:-1] + (self.no_heads, 2 * self.c_hidden))
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # Generate point coordinates (keep original point-based attention for v_pts)
        q_pts = self.linear_q_points(s)
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)  # Parallel rotation application
        q_pts = q_pts.view(q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3))

        kv_pts = self.linear_kv_points(s)
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        # kv_pts = r[..., None].apply(kv_pts)  # Parallel rotation application
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))
        k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

        ##########################
        # Compute attention scores
        ##########################
        # Scalar attention (unchanged)
        b = self.linear_b(z[0])
        if _offload_inference:
            z[0] = z[0].cpu()

        # Group-wise 3D-RoPE attention
        if self.use_groupwise_rope:
            # Apply group-wise rotations to q and k features
            q_rot = self.apply_groupwise_rotations(q, r)  # [*, N_res, H, G, 3]
            k_rot = self.apply_groupwise_rotations(k, r)  # [*, N_res, H, G, 3]

            # Step 2: Spherical harmonics expansion on the direction vectors
            Y_q = self.spherical_harmonic_expansion(q_rot)  # [B, N, H, G, l_max]
            Y_k = self.spherical_harmonic_expansion(k_rot)  # [B, N, H, G, l_max]

            # 2. 正确的维度重排用于attention计算
            # [B, N, H, G, l_max] -> [B, H, G, N, l_max]
            Y_q_attn = Y_q.permute(0, 2, 3, 1, 4)  # [B, H, G, N_res, l_max]
            Y_k_attn = Y_k.permute(0, 2, 3, 1, 4)  # [B, H, G, N_res, l_max]

            # 3. 计算attention分数
            # [B, H, G, N_res, l_max] @ [B, H, G, l_max, N_res] -> [B, H, G, N_res, N_res]
            attention_scores = torch.matmul(Y_q_attn, Y_k_attn.transpose(-2, -1))

            # 4. 缩放
            attention_scores = attention_scores / math.sqrt(Y_q_attn.size(-1))  # sqrt(l_max)

            attention_scores = self.aggregate_group_scores(attention_scores)  # [*, H, N_res, N_res]

            # Add bias term
            attention_scores += (math.sqrt(1.0 / Y_q_attn.size(-1)) * permute_final_dims(b, (2, 0, 1)))

            a = attention_scores
        else:
            # Original scalar attention
            a = torch.matmul(
                permute_final_dims(q, (1, 0, 2)),
                permute_final_dims(k, (1, 2, 0)),
            )
            a *= math.sqrt(1.0 / (3 * self.c_hidden))
            a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # Apply mask and softmax
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output (unchanged)
        ################
        o = torch.matmul(a, v.transpose(-2, -3)).transpose(-2, -3)
        o = flatten_final_dims(o, 2)

        o_pt = torch.sum(
            (
                    a[..., None, :, :, None]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )



        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        # o_pt = r[..., None, None].invert_apply(o_pt)

        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(o_pt_dists, 2)
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if _offload_inference:
            z[0] = z[0].to(o_pt.device)

        pair_z = self.down_z(z[0])
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        s = self.linear_out(torch.cat(o_feats, dim=-1))

        return s