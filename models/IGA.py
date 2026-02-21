import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Sequence, Tuple


from data.GaussianRigid import OffsetGaussianRigid

from typing import Optional

from openfold.model.primitives import Linear
from models import ipa_pytorch  # ,so3_theta,rope3D
# ==========================================
# Part 1: 高性能数学核 (The Math Engine)
# ==========================================

# @torch.compile(fullgraph=True, dynamic=False)
def fused_gaussian_overlap_score(delta: torch.Tensor, sigma: torch.Tensor,    eps: float = 1e-6,
    with_logdet: bool = False,
) -> torch.Tensor:
    """
    计算 Gaussian Overlap Score (Mahalanobis Distance)。
    Score = -0.5 * delta^T * (Sigma)^-1 * delta

    假设输入的 sigma 已经在外部加了 eps 保证正定，这里不再重复加。
    """
    # 1. 提取元素 (Sigma 是对称矩阵)
    # sigma: [..., 3, 3]
    s00 = sigma[..., 0, 0]
    s11 = sigma[..., 1, 1]
    s22 = sigma[..., 2, 2]
    s01 = sigma[..., 0, 1]
    s02 = sigma[..., 0, 2]
    s12 = sigma[..., 1, 2]

    d0 = delta[..., 0]
    d1 = delta[..., 1]
    d2 = delta[..., 2]

    # 2. 隐式 Cholesky 分解 (L * L^T = Sigma)
    # L = [[l00, 0, 0], [l10, l11, 0], [l20, l21, l22]]
    # clamp 依然保留作为最后一道防线
    eps = 1e-8

    inv_l00 = torch.rsqrt(torch.clamp(s00, min=eps))
    l10 = s01 * inv_l00
    l20 = s02 * inv_l00

    diag_11 = s11 - l10 * l10
    inv_l11 = torch.rsqrt(torch.clamp(diag_11, min=eps))
    l21 = (s12 - l20 * l10) * inv_l11

    diag_22 = s22 - l20 * l20 - l21 * l21
    inv_l22 = torch.rsqrt(torch.clamp(diag_22, min=eps))

    # 3. 前代法求解 y = L^-1 * delta
    y0 = d0 * inv_l00
    y1 = (d1 - l10 * y0) * inv_l11
    y2 = (d2 - l20 * y0 - l21 * y1) * inv_l22

    # 4. 距离平方
    dist_sq = y0 * y0 + y1 * y1 + y2 * y2

    return -0.5 * dist_sq


def analytical_inverse_3x3(sigma: torch.Tensor):
    """
    3x3 矩阵的解析求逆 (Cramer's Rule / Adjugate Matrix)。
    完全避免 torch.linalg.inv，速度极快。
    """
    # 提取元素
    s00 = sigma[..., 0, 0]
    s11 = sigma[..., 1, 1]
    s22 = sigma[..., 2, 2]
    s01 = sigma[..., 0, 1]
    s02 = sigma[..., 0, 2]
    s12 = sigma[..., 1, 2]  # 对称阵 s12=s21

    # 1. 计算余子式 (Cofactors)
    # 00位置的余子式: s11*s22 - s12*s12
    c00 = s11 * s22 - s12 * s12
    c11 = s00 * s22 - s02 * s02
    c22 = s00 * s11 - s01 * s01

    c01 = s02 * s12 - s01 * s22  # 注意符号，这里计算的是伴随矩阵元素
    c02 = s01 * s12 - s02 * s11
    c12 = s01 * s02 - s00 * s12

    # 2. 计算行列式 det = s00*c00 + s01*c01_transpose + ...
    # 简单写法: s00*(s11s22 - s12^2) - s01*(s01s22 - s12s02) + s02*(s01s12 - s11s02)
    det = s00 * c00 + s01 * (s12 * s02 - s01 * s22) + s02 * (s01 * s12 - s11 * s02)

    # 3. 逆矩阵 = adj(A) / det
    inv_det = 1.0 / det.clamp(min=1e-8)

    # 构建逆矩阵 (利用对称性)
    # [ c00, c01, c02 ]
    # [ c01, c11, c12 ]
    # [ c02, c12, c22 ]

    out_00 = c00 * inv_det
    out_11 = c11 * inv_det
    out_22 = c22 * inv_det
    out_01 = c01 * inv_det
    out_02 = c02 * inv_det
    out_12 = c12 * inv_det

    # 拼回 tensor
    row0 = torch.stack([out_00, out_01, out_02], dim=-1)
    row1 = torch.stack([out_01, out_11, out_12], dim=-1)
    row2 = torch.stack([out_02, out_12, out_22], dim=-1)

    inv_sigma = torch.stack([row0, row1, row2], dim=-2)
    return inv_sigma
# ... (保留原有的 fused_gaussian_overlap_score 用于 Attention，保持极速) ...

def compute_robust_nll_components(delta: torch.Tensor, sigma: torch.Tensor):
    """
    专为 Loss 设计的鲁棒计算函数。
    使用与 Attention 相同的隐式 Cholesky 逻辑，但同时返回:
    1. dist_sq (Mahalanobis 距离平方 d^T S^-1 d)
    2. log_det (对数行列式 ln|S|)

    避免了 torch.linalg.cholesky 的崩溃问题。
    """
    # 1. 提取元素
    s00 = sigma[..., 0, 0];
    s01 = sigma[..., 0, 1];
    s02 = sigma[..., 0, 2]
    s11 = sigma[..., 1, 1];
    s12 = sigma[..., 1, 2];
    s22 = sigma[..., 2, 2]
    d0 = delta[..., 0];
    d1 = delta[..., 1];
    d2 = delta[..., 2]

    eps = 1e-8  # 数值保护

    # 2. 鲁棒 Cholesky 分解 (L00, L11, L22 的平方)
    # l00_sq = L[0,0]^2
    l00_sq = torch.clamp(s00, min=eps)
    inv_l00 = torch.rsqrt(l00_sq)  # 1/L00

    l10 = s01 * inv_l00
    l20 = s02 * inv_l00

    # l11_sq = L[1,1]^2
    l11_sq = torch.clamp(s11 - l10 * l10, min=eps)
    inv_l11 = torch.rsqrt(l11_sq)  # 1/L11
    l21 = (s12 - l20 * l10) * inv_l11

    # l22_sq = L[2,2]^2
    l22_sq = torch.clamp(s22 - l20 * l20 - l21 * l21, min=eps)
    inv_l22 = torch.rsqrt(l22_sq)  # 1/L22

    # 3. 计算 Log Determinant
    # log(|Sigma|) = log(|L|^2) = 2 * sum(log(diag(L))) = sum(log(diag(L)^2))
    # 直接用平方项计算，避免开根号后再 log，精度更高
    log_det = torch.log(l00_sq) + torch.log(l11_sq) + torch.log(l22_sq)

    # 4. 前代法求解 y = L^-1 * delta
    y0 = d0 * inv_l00
    y1 = (d1 - l10 * y0) * inv_l11
    y2 = (d2 - l20 * y0 - l21 * y1) * inv_l22

    # 5. Mahalanobis 距离平方
    dist_sq = y0 * y0 + y1 * y1 + y2 * y2

    return dist_sq, log_det
# fused_gaussian_overlap_score = torch.compile(fused_gaussian_overlap_score, fullgraph=True, dynamic=True)
# ==========================================
# Part 2: 几何变换工具
# ==========================================

def transform_to_global_gaussian(
        r_backbone: OffsetGaussianRigid,  # 必须是 OffsetGaussianRigid 类型
        local_offset_u: torch.Tensor,  # [B, N, H, P, 3] unitless coords in ellipsoid-aligned frame
        local_scale_log_delta: torch.Tensor,  # [B, N, H, P, 3] log-scale deltas
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将局部预测转换为全局高斯参数 (Global Gaussian)。

    逻辑: 基于当前时刻的高斯状态 (Anchor)，叠加网络预测的微扰 (Delta)，
          计算出用于 Attention 的临时全局高斯。
    """
    # 1. 扩展 Backbone 维度以匹配 Head 和 Points
    # r_backbone: [B, N] -> [B, N, 1, 1] (用于广播)
    r_expanded = r_backbone.unsqueeze(-1).unsqueeze(-1)

    # 2. 计算全局中心 (Mean)
    # We parameterize the mean offset in *units of sigma* to avoid
    # unbounded Mahalanobis distances (e.g. -1e4 vs -10 saturation).
    # Global = Frame * (Anchor_Local_Mean + u * sigma_local)
    # Anchor is the current state local mean (centroid offset in backbone frame).
    anchor_mean = r_backbone._local_mean.unsqueeze(-2).unsqueeze(-2)

    # sigma_local: exp(base + delta), broadcast to [B,N,H,P,3]
    base_scale_log = r_backbone._scaling_log.unsqueeze(-2).unsqueeze(-2)
    sigma_local = torch.exp(base_scale_log + local_scale_log_delta).clamp_min(1e-6)

    # Compose local mean using bounded u
    total_local_pos = anchor_mean + local_offset_u * sigma_local

    # 变换到全局
    mu_global = r_expanded.apply(total_local_pos)

    # 3. 计算全局协方差 (Covariance)
    # 逻辑: 调用类方法，基于 Current_Scale + Delta 计算 Sigma
    # r_expanded 内部的 _rots 会自动广播
    Sigma = r_expanded.get_covariance_with_delta(local_scale_log_delta)

    return mu_global, Sigma


# ==========================================
# Part 3: IGA 模块 (The Module)
# ==========================================




# 假设 fused_gaussian_overlap_score 和 transform_to_global_gaussian 已经定义好
# 如果在同一个文件中，可以直接使用；如果在其他模块，请 import

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Sequence, Tuple

from openfold.model.primitives import Linear
from openfold.utils.rigid_utils import Rigid


# 假设 fused_gaussian_overlap_score 和 transform_to_global_gaussian 已导入
# from iga import fused_gaussian_overlap_score, transform_to_global_gaussian

class InvariantGaussianAttention(nn.Module):
    """
    Standard IGA: With Pair Feature Support.
    既有高斯几何交互 (Physics)，又有 Pair Bias (Evolution/Topology)。
    """

    def __init__(
            self,
            c_s: int,
            c_z: int,  # [Restored] Pair channel dim
            c_hidden: int,
            no_heads: int,
            no_qk_gaussians: int,
            no_v_points: int,
            inf: float = 1e5,
            enable_vis: bool = False,
            vis_interval: int = 100,
            vis_dir: str = "./attention_vis",
            layer_idx: int = 0,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_gaussians = no_qk_gaussians
        self.no_v_points = no_v_points
        self.inf = inf

        # Vis config
        self.enable_vis = enable_vis
        self.vis_interval = vis_interval
        self.vis_dir = vis_dir
        self.layer_idx = layer_idx
        self._vis_counter = 0

        hc = c_hidden * no_heads

        # 1. Scalar Projection
        self.linear_q = Linear(c_s, hc, bias=False)
        self.linear_k = Linear(c_s, hc, bias=False)
        self.linear_v = Linear(c_s, hc, bias=False)

        # [Restored] Pair Bias Projection
        # 将 Pair 特征 (z) 投影为 Attention Bias (b)
        if self.c_z > 0:
            self.linear_b = Linear(c_z, no_heads)
            self.down_z = Linear(c_z, c_z // 4)  # 用于聚合 Pair 信息到 Single
        else:
            self.linear_b = None
            self.down_z = None

        # 2. Geometric Projection (Gaussian Heads)
        # Offset(3) + Scale(3) = 6 params
        self.linear_q_gaussian = Linear(c_s, no_heads * no_qk_gaussians * 6)
        self.linear_k_gaussian = Linear(c_s, no_heads * no_qk_gaussians * 6)
        self.linear_v_points = Linear(c_s, no_heads * no_v_points * 3)

        # # 3. Output Projection
        # self.head_weights_raw = nn.Parameter(torch.zeros(no_heads))  # Gamma raw

        # C. Gamma 参数 (黄金配方: 初始化为负数)
        # softplus(-4.0) ≈ 0.018，保证初始时几何项权重很小，不干扰语义学习
        # self.head_weights_raw = nn.Parameter(torch.full((no_heads,), 1.0))
        # self.geohead_weights = nn.Parameter(torch.full((no_heads,), 1.0))
        # self.scalar_qk_weights = nn.Parameter(torch.full((no_heads,), 1.0))
        # self.pair_weights = nn.Parameter(torch.full((no_heads,), 1.0))
        # >>> 新增：几何分支的 a、b（每个 head 一对），初始全 0 <<<
        self.geo_scale = nn.Parameter(torch.full((no_heads,), 1.0))  # a_h
        self.geo_bias = nn.Parameter(torch.zeros(no_heads))  # b_h

        # Bound Gaussian parameterization to keep geometry logits in a usable range.
        # - u_max controls how many sigmas the learned Gaussian centers can move.
        # - log_scale_max controls exp(delta) range: [e^-m, e^m].
        self.u_max = 3.0
        self.log_scale_max = 2.0



        # [Modified] Output Dim includes Pair aggregation
        # Scalar(C) + Points(4P) + Pair(Cz//4)
        pair_out_dim = (c_z // 4) if self.c_z > 0 else 0
        concat_out_dim = no_heads * (c_hidden + no_v_points * 4 + pair_out_dim)

        self.linear_out = Linear(concat_out_dim, c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self._last_debug = None

    def get_last_debug(self):
        """Returns a dict of scalar tensors from the last forward() call (or None)."""
        return self._last_debug

    def _masked_mean_std_min_max(self, x: torch.Tensor, mask_2d: torch.Tensor):
        """
        x: [B, H, N, N]
        mask_2d: [B, 1, N, N] (0/1)
        Returns scalar tensors: mean, std, min, max over valid entries.
        """
        m = mask_2d
        denom = m.sum(dim=(-1, -2), keepdim=True).clamp_min(1.0)
        mean = (x * m).sum(dim=(-1, -2), keepdim=True) / denom
        var = ((x - mean) ** 2 * m).sum(dim=(-1, -2), keepdim=True) / denom
        std = torch.sqrt(var + 1e-6)

        # Use +/-inf via large constants to avoid NaNs when all-masked.
        x_min = (x + (1.0 - m) * self.inf).amin(dim=(-1, -2))
        x_max = (x + (1.0 - m) * (-self.inf)).amax(dim=(-1, -2))

        # Reduce to scalars (avg over batch + heads)
        return mean.mean(), std.mean(), x_min.mean(), x_max.mean()

    def norm_component(self,x, mask):
        """
        对 [B, H, N, N] 的矩阵进行 Map-wise Z-score 归一化。
        范围: 每个 Batch, 每个 Head 独立计算 (N, N) 的均值和方差。
        """
        # x: [B, H, N, N]
        # mask: [B, N] -> 扩展为 [B, 1, N, N] 的 2D mask
        mask_2d = mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)

        # 1. 计算有效元素的均值 (Mean)
        # 维度: [B, H, 1, 1]
        # 只在最后两个维度 (-2, -1) 上求和
        num_valid = mask_2d.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
        sum_val = (x * mask_2d).sum(dim=(-2, -1), keepdim=True)
        mean = sum_val / num_valid

        # 2. 计算有效元素的标准差 (Std)
        # 维度: [B, H, 1, 1]
        sum_sq_diff = ((x - mean) ** 2 * mask_2d).sum(dim=(-2, -1), keepdim=True)
        std = torch.sqrt(sum_sq_diff / num_valid + 1e-6)

        # 3. 归一化
        # 广播机制会自动处理 B 和 H 维度
        x_norm = (x - mean) / std

        # 再次 Mask 掉 Padding 区域 (保持 0 或其他无效值)
        return x_norm * mask_2d
    def forward(
            self,
            s: torch.Tensor,  # [B, N, C_s]
            z: Optional[torch.Tensor],  # [B, N, N, C_z] [Restored]
            r: OffsetGaussianRigid,  # [B, N]
            mask: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:

        B, N, _ = s.shape

        # -------------------------------------------------------
        # A. Scalar Attention + Pair Bias
        # -------------------------------------------------------
        q = self.linear_q(s).view(B, N, self.no_heads, -1).transpose(1, 2)  # [B, H, N, C]
        k = self.linear_k(s).view(B, N, self.no_heads, -1).transpose(1, 2)
        v = self.linear_v(s).view(B, N, self.no_heads, -1).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.c_hidden)
        # logits=self.norm_component(logits, mask)

        # Gamma for scalar QK
       # gamma_scalar = F.softplus(self.scalar_qk_weights).view(1, -1, 1, 1)
        logits = logits  # gamma_scalar *

        # Keep a handle to scalar(+pair) logits before adding geometry. We will use this
        # for debug stats without the padding mask (-inf) contaminating reductions.
        logits_scalar_pair = logits

        # Vis Cache: save scalar QK (already weighted by gamma)
        scalar_qk = logits.clone() if self.enable_vis else None

        # [Restored] Add Pair Bias
        pair_bias = None
        gamma_pair = None
        if self.linear_b is not None and z is not None:
            # z: [B, N, N, Cz] -> b: [B, N, N, H] -> [B, H, N, N]
            b = self.linear_b(z).permute(0, 3, 1, 2)
            #b = self.norm_component(b, mask)

            # Gamma for pair
            #gamma_pair = F.softplus(self.pair_weights).view(1, -1, 1, 1)
            b =  b #gamma_pair *

            pair_bias = b.clone() if self.enable_vis else None
            logits = logits + b

        # -------------------------------------------------------
        # B. Geometric Attention (Gaussian Overlap)
        # -------------------------------------------------------
        def parse_gaussian_lite(feat):
            raw = feat.view(B, N, self.no_heads, self.no_qk_gaussians, 6)
            u_raw = raw[..., :3]
            s_raw = raw[..., 3:]
            # Keep deltas bounded; the ellipsoid scale carries the true metric.
            u = torch.tanh(u_raw) * self.u_max
            s_log_delta = torch.tanh(s_raw) * self.log_scale_max
            return u, s_log_delta

        q_off_delta, q_s_delta = parse_gaussian_lite(self.linear_q_gaussian(s))
        k_off_delta, k_s_delta = parse_gaussian_lite(self.linear_k_gaussian(s))

        # Global Transform
        q_mu, q_sigma = transform_to_global_gaussian(r, q_off_delta, q_s_delta)
        k_mu, k_sigma = transform_to_global_gaussian(r, k_off_delta, k_s_delta)

        # Broadcasting
        q_mu = q_mu.permute(0, 2, 1, 3, 4);
        k_mu = k_mu.permute(0, 2, 1, 3, 4)
        q_sigma = q_sigma.permute(0, 2, 1, 3, 4, 5);
        k_sigma = k_sigma.permute(0, 2, 1, 3, 4, 5)

        delta_mu = q_mu.unsqueeze(3) - k_mu.unsqueeze(2)
        sigma_sum = q_sigma.unsqueeze(3) + k_sigma.unsqueeze(2)

        # Kernel (Gaussian log-likelihood up to an additive constant):
        #   log p(delta | Sigma) = -0.5 * d^T Sigma^{-1} d - 0.5 * log|Sigma| + const
        # Using the same robust implicit-Cholesky logic as our loss code.
        dist_sq, log_det = compute_robust_nll_components(delta_mu, sigma_sum)
        overlap_scores = -0.5 * dist_sq - 0.5 * log_det

        # Aggregate across Gaussian points per head
        attn_bias_geo = overlap_scores.sum(dim=-1)

        # Z-Score Norm (Preserved)
        # geo = attn_bias_geo
        # geo = geo - geo.mean(dim=-1, keepdim=True)
        # attn_bias_geo = geo / (geo.std(dim=-1, keepdim=True) + 1e-6)

        # 2. 压缩: Sign-preserving Log1p
        # 将 [-inf, 0] 压缩到 [-log(1+abs), 0]
        # 例如: -3000 -> -8.0; -10 -> -2.4; -0.5 -> -0.4
        # attn_bias_geo = torch.sign(attn_bias_geo) * torch.log1p(attn_bias_geo.abs())
        # attn_bias_geo = self.norm_component(attn_bias_geo, mask)
        #
        # Gamma for geo
       # gamma_geo = F.softplus(self.geohead_weights).view(1, -1, 1, 1)

        # 3) 用 2D mask 清掉 padding 区域
        if mask is not None:
            # mask: [B, N]
            mask_2d = mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)  # [B, 1, N, N]
            G_raw = attn_bias_geo * mask_2d
        else:
            G_raw = attn_bias_geo

        # 4) 几何分支：G_geo = softplus(a * G_raw + b)
        #    a, b 是每个 head 各自的可学习标量，不依赖当前 map 的 mean/std
        a = F.softplus(self.geo_scale).view(1, -1, 1, 1)  # a_h >= 0
        b = self.geo_bias.view(1, -1, 1, 1)  # [1, H, 1, 1]
        # G_geo = F.softplus(a * G_raw + b)  # [B, H, N, N]
        G_geo = a * G_raw + b  # [B, H, N, N]


        # Vis Cache: save logits before adding geo (scalar + pair)
        logits_before = logits.clone() if self.enable_vis else None

        # Combine: Logits = Scalar + Pair + Gamma * Geometric
        logits = logits + G_geo

        logits_pre_mask = logits

        # Vis Cache: save logits after adding geo (before row norm)
        logits_after = logits.clone() if self.enable_vis else None

        # Row Norm
        # mean = logits.mean(dim=-1, keepdim=True)
        # std = logits.std(dim=-1, keepdim=True) + 1e-6
        # logits = (logits - mean) / std

        # -------------------------------------------------------
        # C. Softmax & Output Aggregation
        # -------------------------------------------------------
        if mask is not None:
            mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
            logits = logits + (1.0 - mask_2d.unsqueeze(1)) * -self.inf

        weights = self.softmax(logits)

        # -------------------------------------------------------
        # Debug stats (numeric stability + saturation diagnostics)
        # -------------------------------------------------------
        if mask is not None:
            mask_2d = mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)  # [B,1,N,N]
        else:
            mask_2d = torch.ones((B, 1, N, N), device=logits.device, dtype=logits.dtype)

        try:
            # Scalar(+pair) logits stats (before geo is added), without padding-mask -inf.
            scalar_logits = logits_scalar_pair.detach()
            s_mean, s_std, s_min, s_max = self._masked_mean_std_min_max(scalar_logits.detach(), mask_2d)
            g_mean, g_std, g_min, g_max = self._masked_mean_std_min_max(G_geo.detach(), mask_2d)

            # Raw geo term stats (pre affine a*G + b): attn_bias_geo is [B,H,N,N]
            gr_mean, gr_std, gr_min, gr_max = self._masked_mean_std_min_max(attn_bias_geo.detach(), mask_2d)

            # Dist / logdet summaries (use means over valid entries; these tensors are large so keep it light)
            # dist_sq/log_det: [B,H,N,N,P,3]?? -> after compute_robust_nll_components: [B,H,N,N,P,3]? actually [B,H,N,N,P,3] collapsed? compute returns [...,3]? No: it returns [B,H,N,N,P,3]? then summed? Here dist_sq/log_det are [B,H,N,N,P,3]? We'll reduce safely.
            ds = dist_sq.detach()
            ld = log_det.detach()
            # Reduce over last dims to get [B,H,N,N]
            while ds.dim() > 4:
                ds = ds.mean(dim=-1)
            while ld.dim() > 4:
                ld = ld.mean(dim=-1)
            ds_mean, ds_std, ds_min, ds_max = self._masked_mean_std_min_max(ds, mask_2d)
            ld_mean, ld_std, ld_min, ld_max = self._masked_mean_std_min_max(ld, mask_2d)

            # Softmax saturation: max prob per query row
            wmax = weights.detach().max(dim=-1).values  # [B,H,N]
            qmask = mask[:, None, :].to(wmax.dtype) if mask is not None else torch.ones_like(wmax)
            denom_q = qmask.sum(dim=-1).clamp_min(1.0)  # [B,1]
            wmax_mean = (wmax * qmask).sum(dim=-1) / denom_q
            sat_frac = ((wmax > 0.9).to(wmax.dtype) * qmask).sum(dim=-1) / denom_q

            self._last_debug = {
                "scalar_mean": s_mean,
                "scalar_std": s_std,
                "scalar_min": s_min,
                "scalar_max": s_max,
                "geo_scaled_mean": g_mean,
                "geo_scaled_std": g_std,
                "geo_scaled_min": g_min,
                "geo_scaled_max": g_max,
                "geo_raw_mean": gr_mean,
                "geo_raw_std": gr_std,
                "geo_raw_min": gr_min,
                "geo_raw_max": gr_max,
                "dist_sq_mean": ds_mean,
                "dist_sq_std": ds_std,
                "dist_sq_min": ds_min,
                "dist_sq_max": ds_max,
                "log_det_mean": ld_mean,
                "log_det_std": ld_std,
                "log_det_min": ld_min,
                "log_det_max": ld_max,
                "wmax_mean": wmax_mean.mean(),
                "wmax_sat_frac": sat_frac.mean(),
                "geo_a_mean": a.detach().mean(),
                "geo_a_max": a.detach().max(),
                "geo_b_mean": b.detach().mean(),
            }
        except Exception:
            # Don't crash training due to debug logging.
            self._last_debug = None

        # [Optional] Visualization
        if self.enable_vis and self.training:
            self._vis_counter += 1
            if self._vis_counter % self.vis_interval == 0:
                self._visualize_attention(
                    scalar_qk=scalar_qk.detach() if scalar_qk is not None else None,
                    pair_bias=pair_bias.detach() if pair_bias is not None else None,
                    attn_bias_geo=attn_bias_geo.detach(),
                    gamma_scalar=1,
                    gamma_pair=gamma_pair.detach() if gamma_pair is not None else None,
                    gamma_geo=a.detach(),
                    logits_before=logits_before.detach(),
                    logits_after=logits_after.detach(),
                    weights=weights.detach(),
                    rigid_trans=r.get_trans().detach(),
                )

        # 1. Scalar Agg
        o_scalar = torch.matmul(weights, v)

        # 2. Point Agg
        v_pts_local = self.linear_v_points(s).view(B, N, self.no_heads, self.no_v_points, 3)
        v_pts_global = r.unsqueeze(-1).unsqueeze(-1).apply(v_pts_local).permute(0, 2, 1, 3, 4)
        o_pts_global = torch.einsum('bhij,bhjpv->bhipv', weights, v_pts_global)
        o_pts_local = r.unsqueeze(1).unsqueeze(-1).invert_apply(o_pts_global)
        o_pts_norm = torch.sqrt(torch.sum(o_pts_local ** 2, dim=-1) + 1e-8)

        # 3. [Restored] Pair Aggregation
        to_concat = [
            o_scalar.transpose(1, 2).reshape(B, N, -1),
            o_pts_local.transpose(1, 2).reshape(B, N, -1),
            o_pts_norm.transpose(1, 2).reshape(B, N, -1)
        ]

        if self.down_z is not None and z is not None:
            # z: [B, N, N, C_z] -> projected: [B, N, N, C_z//4]
            z_proj = self.down_z(z)
            # weights: [B, H, N, N]
            # Aggregate: o_pair = weights * z_proj -> [B, H, N, C_z//4]
            o_pair = torch.einsum('bhij,bijc->bhic', weights, z_proj)
            to_concat.append(o_pair.transpose(1, 2).reshape(B, N, -1))

        # 4. Final Projection
        out = self.linear_out(torch.cat(to_concat, dim=-1))

        return out
    def _visualize_attention(self, scalar_qk, pair_bias, attn_bias_geo, gamma_scalar, gamma_pair, gamma_geo, logits_before, logits_after, weights, rigid_trans):
        """调用可视化函数"""
        try:
            from models.visualize_attention import visualize_iga_attention
            stats = visualize_iga_attention(
                scalar_qk=scalar_qk,
                pair_bias=pair_bias,
                attn_bias_geo=attn_bias_geo,
                gamma_scalar=gamma_scalar,
                gamma_pair=gamma_pair,
                gamma_geo=gamma_geo,
                logits_before=logits_before,
                logits_after=logits_after,
                weights=weights,
                rigid_trans=rigid_trans,
                save_dir=self.vis_dir,
                batch_idx=0,
                head_idx=0,
                num_vis_res=50,
                layer_idx=self.layer_idx
            )
            print(stats)
            print(f"[IGA Vis Layer {self.layer_idx}] Step {self._vis_counter}: "
                  f"γ_s={stats['gamma_scalar']:.3f}, γ_p={stats['gamma_pair']:.3f}, γ_g={stats['gamma_geo']:.3f}, "
                  f"geo_bias={stats['geo_bias_mean']:.3f}, "
                  f"local_attn={stats['local_attn_mean']:.4f}, "
                  f"global_attn={stats['global_attn_mean']:.4f}")
        except Exception as e:
            print(f"[IGA Vis Layer {self.layer_idx}] Warning: Failed to visualize attention: {e}")



# class InvariantGaussianAttention(nn.Module):
#     """
#     Lite IGA: No Pair Feature Version.
#     专为 FBB / Inverse Folding 设计，移除冗余的 Pair 分支，专注于物理几何交互。
#     """
#
#     def __init__(
#             self,
#             c_s: int,
#             c_hidden: int,
#             no_heads: int,
#             no_qk_gaussians: int,
#             no_v_points: int,
#             inf: float = 1e5,
#             enable_vis: bool = True,
#             vis_interval: int = 1000,
#             vis_dir: str = "./attention_vis",
#             layer_idx: int = 0,
#     ):
#         super().__init__()
#         self.c_s = c_s
#         self.c_hidden = c_hidden
#         self.no_heads = no_heads
#         self.no_qk_gaussians = no_qk_gaussians
#         self.no_v_points = no_v_points
#         self.inf = inf
#
#         # Visualization settings
#         self.enable_vis = enable_vis
#         self.vis_interval = vis_interval
#         self.vis_dir = vis_dir
#         self.layer_idx = layer_idx
#         self._vis_counter = 0
#
#         hc = c_hidden * no_heads
#
#         # 1. 标量投影 (Scalar Q, K, V)
#         self.linear_q = Linear(c_s, hc, bias=False)
#         self.linear_k = Linear(c_s, hc, bias=False)
#         self.linear_v = Linear(c_s, hc, bias=False)
#         # [Deleted] self.linear_b (Pair bias)
#
#         # # 2. 几何特征投影 (Gaussian Heads)
#         # # 预测: Offset(3) + Scale(3) + Rotation(4) = 10 params
#         # self.linear_q_gaussian = Linear(c_s, no_heads * no_qk_gaussians * 10)
#         # self.linear_k_gaussian = Linear(c_s, no_heads * no_qk_gaussians * 10)
#
#         # B. 几何高斯投影 (Gaussian Heads)
#         # 之前: Offset(3) + Scale(3) + Rotation(4) = 10
#         # 现在: Offset(3) + Scale(3) = 6
#         # 我们只预测 Delta，并在 transform 里面叠加到 Anchor 上
#         self.linear_q_gaussian = Linear(c_s, no_heads * no_qk_gaussians * 6)
#         self.linear_k_gaussian = Linear(c_s, no_heads * no_qk_gaussians * 6)
#
#
#         # V Points (Local coords) - Value 依然搬运点信息用于更新主链
#         self.linear_v_points = Linear(c_s, no_heads * no_v_points * 3)
#
#         # 3. 输出层
#         # gamma 的上限，可以从 config 里给一个超参
#         # self.gamma_max = getattr(model_conf, "iga_gamma_max", 3.0)
#         #rev1
#         # self.gamma_max = 3
#         # # 原来的 head_weights 改名为原始参数，不再直接拿来当 gamma
#         # self.head_weights_raw = nn.Parameter(torch.zeros(no_heads))
#
#         #rev2
#         self.head_weights_raw = nn.Parameter(torch.zeros(no_heads))
#
#         # rev0
#         # self.head_weights = nn.Parameter(torch.zeros(no_heads))  # Gamma
#
#         # [Modified] 输出维度不再包含 Pair 部分
#         # Output = Scalar(C) + Points_Coords(3P) + Points_Norms(1P) = C + 4P
#         concat_out_dim = no_heads * (c_hidden + no_v_points * 4)
#         self.linear_out = Linear(concat_out_dim, c_s, init="final")
#
#         # [Deleted] self.down_z
#         # [Deleted] self.softplus - 不再使用，改用 sigmoid
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(
#             self,
#             s: torch.Tensor,  # [B, N, C_s]
#             r: OffsetGaussianRigid,  # [B, N] 主链刚体
#             mask: torch.Tensor,  # [B, N]
#     ) -> torch.Tensor:
#         """
#         不再需要 z (Pair features) 作为输入
#         """
#         B, N, _ = s.shape
#
#         # -------------------------------------------------------
#         # A. 标量 Attention (Scalar Term)
#         # -------------------------------------------------------
#         q = self.linear_q(s).view(B, N, self.no_heads, -1).transpose(1, 2)  # [B, H, N, C]
#         k = self.linear_k(s).view(B, N, self.no_heads, -1).transpose(1, 2)  # [B, H, N, C]
#         v = self.linear_v(s).view(B, N, self.no_heads, -1).transpose(1, 2)  # [B, H, N, C]
#
#         # Dot product
#         # [B, H, N, N]
#         logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.c_hidden)
#
#         # [Deleted] logits = logits + b (不再加 Pair Bias)
#
#         # -------------------------------------------------------
#         # B. 几何 Attention (Gaussian Overlap Term)
#         # -------------------------------------------------------
#         # def parse_gaussian(feat):
#         #     # [B, N, H * P * 10] -> [B, N, H, P, 10]
#         #     raw = feat.view(B, N, self.no_heads, self.no_qk_gaussians, 10)
#         #     return raw[..., :3], raw[..., 3:6], raw[..., 6:]
#         #
#         # q_off, q_s, q_r = parse_gaussian(self.linear_q_gaussian(s))
#         # k_off, k_s, k_r = parse_gaussian(self.linear_k_gaussian(s))
#         #
#         # # 转换到全局坐标系
#         # # q_mu: [B, N, H, P, 3], q_sigma: [..., 3, 3]
#         # q_mu, q_sigma = transform_to_global_gaussian(r, q_off, q_s, q_r)
#         # k_mu, k_sigma = transform_to_global_gaussian(r, k_off, k_s, k_r)
#
#         # -------------------------------------------------------
#         # B. 几何 Attention (Gaussian Overlap Term)
#         # -------------------------------------------------------
#         def parse_gaussian_lite(feat):
#             # [B, N, H * P * 6] -> [B, N, H, P, 6]
#             raw = feat.view(B, N, self.no_heads, self.no_qk_gaussians, 6)
#             # split into offset_delta (3) and scale_delta (3)
#             return raw[..., :3], raw[..., 3:]
#
#         # 1. 预测局部高斯参数 (Deltas)
#         q_off_delta, q_s_delta = parse_gaussian_lite(self.linear_q_gaussian(s))
#         k_off_delta, k_s_delta = parse_gaussian_lite(self.linear_k_gaussian(s))
#
#         # 2. 转换到全局坐标系 (使用 Anchor + Delta 机制)
#         # transform_to_global_gaussian 内部会调用 r.get_covariance_with_delta
#         # 不需要再传 rotation 了
#         q_mu, q_sigma = transform_to_global_gaussian(r, q_off_delta, q_s_delta)
#         k_mu, k_sigma = transform_to_global_gaussian(r, k_off_delta, k_s_delta)
#
#
#
#         # 准备广播 -> [B, H, N, P, ...]
#         q_mu = q_mu.permute(0, 2, 1, 3, 4)
#         k_mu = k_mu.permute(0, 2, 1, 3, 4)
#         q_sigma = q_sigma.permute(0, 2, 1, 3, 4, 5)
#         k_sigma = k_sigma.permute(0, 2, 1, 3, 4, 5)
#
#         # Delta Mu: [B, H, N_q, N_k, P, 3]
#         delta_mu = q_mu.unsqueeze(3) - k_mu.unsqueeze(2)
#
#         # Sigma Sum: [B, H, N_q, N_k, P, 3, 3]
#         # 注意: 协方差加法 (Convolution Property)
#         sigma_sum = q_sigma.unsqueeze(3) + k_sigma.unsqueeze(2)
#
#         # Kernel: Fused Cholesky Mahalanobis Distance
#         # [B, H, N, N, P]
#         overlap_scores = fused_gaussian_overlap_score(delta_mu, sigma_sum)
#
#
#         # Aggregation: Sum over P gaussians, scaled
#         scale_factor = 1.0 / math.sqrt(self.no_qk_gaussians)
#         attn_bias_geo = overlap_scores.sum(dim=-1) * scale_factor
#
#         # 每个 query i 上做 z-score (行标准化)
#         geo = attn_bias_geo
#         geo = geo - geo.mean(dim=-1, keepdim=True)  # 去掉行均值
#         attn_bias_geo = geo / (geo.std(dim=-1, keepdim=True) + 1e-6)  # 除以行标准差
#
#         # Gamma Weighting (Learnable)
#         gamma = (math.sqrt(1.0 / 3.0) * F.softplus(self.head_weights_raw)).view(1, -1, 1, 1)
#
#         # gamma_head = torch.sigmoid(self.head_weights_raw)
#         # gamma = (math.sqrt(1.0 / 3.0) * gamma_head).view(1, -1, 1, 1)
#
#         # Save logits before adding geometric bias (for visualization)
#         logits_before = logits.clone() if self.enable_vis else None
#
#         # Combine: Logits = Scalar + Gamma * Gaussian_Overlap
#         logits = logits + gamma * attn_bias_geo
#
#         mean = logits.mean(dim=-1, keepdim=True)
#         std = logits.std(dim=-1, keepdim=True) + 1e-6
#         logits = (logits - mean) / std
#
#         # scale to softmax-friendly range
#         logits = logits   # or 2.5
#
#         # -------------------------------------------------------
#         # C. Softmax & Output Aggregation
#         # -------------------------------------------------------
#         if mask is not None:
#             mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
#             logits = logits + (1.0 - mask_2d.unsqueeze(1)) * -self.inf
#
#         weights = self.softmax(logits)  # [B, H, N, N]
#
#         # -------------------------------------------------------
#         # Visualization (Optional)
#         # -------------------------------------------------------
#         if self.enable_vis and self.training:
#             self._vis_counter += 1
#             if self._vis_counter % self.vis_interval == 0:
#                 self._visualize_attention(
#                     attn_bias_geo=attn_bias_geo.detach(),
#                     gamma=gamma.detach(),
#                     logits_before=logits_before.detach(),
#                     logits_after=logits.detach(),
#                     weights=weights.detach(),
#                     rigid_trans=r.get_trans().detach(),
#                 )
#
#         # 1. 聚合 Scalar Value
#         # [B, H, N, C]
#         o_scalar = torch.matmul(weights, v)
#
#         # 2. 聚合 Geometric Value (Points)
#         v_pts_local = self.linear_v_points(s).view(B, N, self.no_heads, self.no_v_points, 3)
#         v_pts_global = r.unsqueeze(-1).unsqueeze(-1).apply(v_pts_local)
#
#         # Permute for einsum: [B, H, N_k, P, 3]
#         v_pts_global = v_pts_global.permute(0, 2, 1, 3, 4)
#
#         # Weighted Sum
#         # [B, H, N_i, P, 3]
#         o_pts_global = torch.einsum('bhij,bhjpv->bhipv', weights, v_pts_global)
#
#         # 转回局部 (Invert Apply) -> 保证 Invariant
#         o_pts_local = r.unsqueeze(1).unsqueeze(-1).invert_apply(o_pts_global)
#
#         # Point Norms
#         o_pts_norm = torch.sqrt(torch.sum(o_pts_local ** 2, dim=-1) + 1e-8)
#
#         # [Deleted] Pair Aggregation Logic
#
#         # -------------------------------------------------------
#         # D. Final Projection
#         # -------------------------------------------------------
#         # Flatten heads
#         o_scalar = o_scalar.transpose(1, 2).reshape(B, N, -1)  # [B, N, H*C]
#         o_pts_local = o_pts_local.transpose(1, 2).reshape(B, N, -1)  # [B, N, H*P*3]
#         o_pts_norm = o_pts_norm.transpose(1, 2).reshape(B, N, -1)  # [B, N, H*P*1]
#
#         # Concat & Linear
#         out = torch.cat([o_scalar, o_pts_local, o_pts_norm], dim=-1)
#         out = self.linear_out(out)
#
#         return out
#
#     def _visualize_attention(self, attn_bias_geo, gamma, logits_before, logits_after, weights, rigid_trans):
#         """调用可视化函数"""
#         try:
#             from models.visualize_attention import visualize_iga_attention
#             stats = visualize_iga_attention(
#                 attn_bias_geo=attn_bias_geo,
#                 gamma=gamma,
#                 logits_before=logits_before,
#                 logits_after=logits_after,
#                 weights=weights,
#                 rigid_trans=rigid_trans,
#                 save_dir=self.vis_dir,
#                 batch_idx=0,
#                 head_idx=0,
#                 num_vis_res=50,
#                 layer_idx=self.layer_idx
#             )
#             print(f"[IGA Vis Layer {self.layer_idx}] Step {self._vis_counter}: "
#                   f"γ={stats['gamma']:.3f}, "
#                   f"geo_bias={stats['geo_bias_mean']:.3f}, "
#                   f"local_attn={stats['local_attn_mean']:.4f}, "
#                   f"global_attn={stats['global_attn_mean']:.4f}")
#         except Exception as e:
#             print(f"[IGA Vis Layer {self.layer_idx}] Warning: Failed to visualize attention: {e}")

class GaussianUpdateBlock(nn.Module):
    def __init__(self, c_s, update_gaussian: bool = True):
        super().__init__()
        # 6D: backbone (qvec3 + t3); 12D adds (alpha3 + log_scale3)
        self.update_gaussian = bool(update_gaussian)
        out_dim = 12 if self.update_gaussian else 6
        self.linear = Linear(c_s, out_dim, init="final")

    def forward(self, s, gaussian_rigid, mask=None):
        """
        Args:
            s: [B, N, C]
            gaussian_rigid: Current object
            mask: [B, N] Update Mask (1=Update, 0=Freeze)
        """
        # 1. 预测增量
        updates = self.linear(s)

        # 2. 【核心修改】应用 Mask
        # Context 区域 (mask=0) 的增量被强制为 0 -> 保持静止
        if mask is not None:
            updates = updates * mask.unsqueeze(-1)

        # 3. 更新几何体
        # 此时 Context 的 Offset/Scale 保持不变
        # Masked 的 Offset/Scale 发生演化
        new_gaussian_rigid = gaussian_rigid.compose_update_12D(updates)

        return new_gaussian_rigid


class CoarseIGABlock(nn.Module):
    """
    一个 coarse IGA block:
        s <- LN(s + IGA(s, r))
        s <- StructureModuleTransition(s)
        r <- GaussianUpdateBlock(s, r)

    不用 z，不用 edge。
    """

    def __init__(
        self,
        iga: nn.Module,                      # InvariantGaussianAttention
        transition: nn.Module,               # StructureModuleTransition
        edgetransition: nn.Module,
            c_s: int,
        gau_update: Optional[nn.Module] = None,               # GaussianUpdateBlock

    ):
        super().__init__()
        self.iga = iga
        self.ln = nn.LayerNorm(c_s)
        self.transition = transition
        self.edgetransition=edgetransition
        self.gau_update = gau_update

    def forward(self, s,z, r, mask):
        """
        s: [B, K, C]
        r: OffsetGaussianRigid [B, K]
        mask: [B, K]
        """

        # 1. IGA
        iga_out = self.iga(
            s=s,
            z=z,        # <- 你说的路线 A：无 z
            r=r,
            mask=mask,
        )
        iga_out = iga_out * mask[..., None]

        # 2. Residual + LN
        s = self.ln(s + iga_out)

        # 3. Transition (你给的)
        s = self.transition(s)
        s = s * mask[..., None]

        # 4. Gaussian Update
        if self.gau_update is not None:
            r = self.gau_update(
                s,
                r,
                mask=mask,     # coarse 一般全 1，也可以来自 pooling
            )

        z=self.edgetransition(s, z)

        return s,z, r


class CoarseIGATower(nn.Module):
    """
    Down 之后用的 coarse tower：
        (IGA → Trans → GauUpdate) × num_layers
    """

    def __init__(
        self,
        iga: nn.Module,
        c_s: int,
        hgfc_z: int,
        num_layers: int,
        gau_update: Optional[nn.Module] = None,

    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            CoarseIGABlock(
                iga=iga,
                transition=ipa_pytorch.StructureModuleTransition(c_s),
                edgetransition=ipa_pytorch.EdgeTransition(
                    node_embed_size=iga.c_s,
                    edge_embed_in=hgfc_z,
                    edge_embed_out=hgfc_z),
                gau_update=gau_update,
                c_s=c_s,
            )
            for _ in range(num_layers)
        ])

    def forward(self, s,z, r, mask):
        for blk in self.blocks:
            s,z, r = blk(s,z, r, mask)
        return s,z, r


class BottleneckIGAModule(nn.Module):
    """
    在最粗层 token 上跑 IGA tower，用于全局一致性推理（hourglass bottleneck）。
    输入/输出 key 与你的 downsampler levels 对齐：
      in:  s: [B, K, C], r: OffsetGaussianRigid[B,K], mask: [B,K]
      out: s', r'  (mask 不变)

    你可以把它当成一个“coarse tower”，只不过专门用于 bottleneck。
    """

    def __init__(
        self,
        c_s: int,
        iga_conf,
        bottleneck_layers: int = 6,
        layer_idx_base: int = 3000,
        enable_vis: bool = False,
    ):
        super().__init__()

        # 1) IGA (no z)
        iga = InvariantGaussianAttention(
            c_s=c_s,
            c_z=getattr(iga_conf, "hgfc_z", 0),   # 仍传一下，内部你路线A会忽略
            c_hidden=iga_conf.c_hidden,
            no_heads=iga_conf.no_heads,
            no_qk_gaussians=iga_conf.no_qk_points,
            no_v_points=iga_conf.no_v_points,
            layer_idx=layer_idx_base,
            enable_vis=enable_vis,
        )

        # 2) transition + update

        gau_update = GaussianUpdateBlock(c_s)

        # 3) tower（你已有的）
        # 注意：你给我的 CoarseIGATower 构造函数里目前是 (iga, gau_update, c_s, num_layers)
        # 但你后来又写了一个版本带 transition 参数。
        # 下面我按“你当前这份定义”(CoarseIGATower 里自己 new transition)来写。
        self.tower = CoarseIGATower(
            iga=iga,
            gau_update=gau_update,
            c_s=c_s,
            hgfc_z=iga_conf.hgfc_z,
            num_layers=bottleneck_layers,
        )

        # 如果你想用外部传入 transition 的版本（你最后那段 pseudo 里），改成：
        # self.tower = CoarseIGATower(
        #     iga=iga, gau_update=gau_update, transition=transition,
        #     num_layers=bottleneck_layers
        # )

    def forward(self, s,z, r, mask):
        """
        s: [B,K,C]
        r: OffsetGaussianRigid [B,K]
        mask: [B,K]
        """
        s_out, r_out,z_out = self.tower(s, r, mask,z)
        return s_out, z_out,r_out

class FastTransformerBlock(nn.Module):
    """更快的 pre-norm Transformer block（不需要几何 r）"""
    def __init__(self, c_s: int, n_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(c_s)
        self.attn = nn.MultiheadAttention(embed_dim=c_s, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(c_s)

        hidden = int(c_s * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(c_s, hidden),
            nn.GELU(),
            nn.Linear(hidden, c_s),
        )

    def forward(self, s: torch.Tensor, mask: torch.Tensor):
        # s: [B,K,C], mask: [B,K] float(0/1)
        key_padding_mask = (mask < 0.5)  # True means "ignore"
        x = self.ln1(s)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        s = s + attn_out
        s = s * mask[..., None]

        x = self.ln2(s)
        s = s + self.mlp(x)
        s = s * mask[..., None]
        return s


class FastTransformerTower(nn.Module):
    def __init__(self, c_s: int, num_layers: int, n_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            FastTransformerBlock(c_s=c_s, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, s: torch.Tensor, mask: torch.Tensor):
        for blk in self.blocks:
            s = blk(s, mask)
        return s


class BottleneckSemanticModule(nn.Module):
    """
    Pure semantic bottleneck (FAST):
      in : s [B,K,C], r (ignored passthrough), mask [B,K]
      out: s' [B,K,C], r passthrough

    Uses your FastTransformerTower (MultiheadAttention + MLP).
    """

    def __init__(
        self,
        c_s: int,
        bottleneck_layers: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.tower = FastTransformerTower(
            c_s=c_s,
            num_layers=bottleneck_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.ln_out = nn.LayerNorm(c_s)

    def forward(self, s: torch.Tensor, r, mask: torch.Tensor):
        """
        s:    [B,K,C]
        r:    anything (passthrough)
        mask: [B,K] float/bool
        """
        s = self.tower(s, mask)
        s = self.ln_out(s) * mask[..., None]
        return s, r
