# -*- coding: utf-8 -*-
"""
upsample_block.py

A fully-differentiable "Up" module for hierarchical Gaussian fields (IGA-style),
designed to match your code style and requirements:

- Input is a coarse-up level: (s_l, r_l, mask_l) where r_l is OffsetGaussianRigid.
- Upsampling generates MORE Gaussians (candidates) via geometric "split / splat".
- The number of children is NOT hard-coded: we use a continuous gate + a soft
  global budget constraint (expected count).
- We compute a responsibility matrix B (child -> parent) WITHOUT explicit matrix
  inverse, using an implicit Cholesky solve (stable and fast in 3x3).
- Geometry is refined by moment-matching with the "between-component" term.
- Semantics are uplifted with the same B and gated to keep semantic–geometry coupling.
- Output is (s_{l-1}, r_{l-1}, mask_{l-1}, losses) where mask is SOFT (float in [0,1]).
  You can later threshold/topk at inference.

This file is self-contained except for your OffsetGaussianRigid class.
If your OffsetGaussianRigid constructor differs, adjust build_rigid_from_mu_sigma().

Author: ChatGPT (for Junyu Yan)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pool import coarse_rigids_from_mu_sigma
from models.IGA import InvariantGaussianAttention,CoarseIGATower,GaussianUpdateBlock,fused_gaussian_overlap_score
from models.loss import HierarchicalGaussianLoss,SymmetricGaussianLoss
# --------------------------
# utilities: overlap score
# --------------------------




def moment_refine_children_from_B(
    mu_parent: torch.Tensor,            # [B, Kp, 3]
    Sigma_parent: torch.Tensor,         # [B, Kp, 3, 3]
    B_resp: torch.Tensor,               # [B, Kc, Kp]  (row-softmax over parents)
    phi: float = 1.6,
    lam: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Moment-matching refinement:
      mu_child = sum_j B_ij mu_j
      Sigma_child = sum_j B_ij (phi^{-2} Sigma_j + (mu_j - mu_child)(mu_j - mu_child)^T) + lam I

    Important: the inter-component term is REQUIRED to avoid "collapse".

    Returns:
      mu_child:    [B, Kc, 3]
      Sigma_child: [B, Kc, 3, 3]
    """
    device, dtype = mu_parent.device, mu_parent.dtype

    mu_child = torch.einsum("bik,bkd->bid", B_resp, mu_parent)  # [B,Kc,3]

    # intra term: sum B * (phi^-2 Sigma)
    Sigma_scaled = (Sigma_parent * (phi ** -2))  # [B,Kp,3,3]
    intra = torch.einsum("bik,bkmn->bimn", B_resp, Sigma_scaled)  # [B,Kc,3,3]

    # inter term
    diff = mu_parent.unsqueeze(1) - mu_child.unsqueeze(2)           # [B,Kc,Kp,3]
    outer = diff.unsqueeze(-1) * diff.unsqueeze(-2)                 # [B,Kc,Kp,3,3]
    inter = torch.einsum("bik,bikmn->bimn", B_resp, outer)        # [B,Kc,3,3]

    I = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
    Sigma_child = intra + inter + lam * I
    return mu_child, Sigma_child


# --------------------------
# utilities: build rigid
# --------------------------

def build_rigid_from_mu_sigma(
    mu: torch.Tensor,            # [B,K,3]
    Sigma: torch.Tensor,         # [B,K,3,3]
    OffsetGaussianRigid_cls,
    eps: float = 1e-6,
):
    """
    Convert (mu, Sigma) to your OffsetGaussianRigid parameterization.

    We do a stable eigen-decomposition of Sigma:
      Sigma = R diag(evals) R^T
      scales = sqrt(evals)

    Then set:
      trans = mu
      rot   = R
      scaling_log = log(scales + eps)
      local_mean = 0 (since coarse-up tokens don't have residue frames)

    NOTE:
      Adjust the constructor / field names to match your class.
      Here we assume:
        OffsetGaussianRigid_cls(rotmats, trans, scaling_log, local_mean)
      OR provides a classmethod. Modify accordingly.
    """
    B, K, _ = mu.shape
    device, dtype = mu.device, mu.dtype

    Sigma_sym = 0.5 * (Sigma + Sigma.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(Sigma_sym)  # [B,K,3], [B,K,3,3]
    evals = torch.clamp(evals, min=eps)
    scales = torch.sqrt(evals)
    scaling_log = torch.log(scales + eps)

    rotmats = evecs
    trans = mu
    local_mean = torch.zeros((B, K, 3), device=device, dtype=dtype)

    try:
        r = OffsetGaussianRigid_cls(rotmats, trans, scaling_log, local_mean)
    except TypeError:
        if hasattr(OffsetGaussianRigid_cls, "from_rot_trans_scaling_local"):
            r = OffsetGaussianRigid_cls.from_rot_trans_scaling_local(rotmats, trans, scaling_log, local_mean)
        else:
            raise

    return r

import torch

def geom_feat_from_mu_sigma_fast(mu: torch.Tensor,
                                Sigma: torch.Tensor,
                                eps: float = 1e-8,
                                symmetrize: bool = True) -> torch.Tensor:
    """
    Build cheap geometry features from (mu, Sigma) without eig/cholesky.

    Args:
        mu:    [B, K, 3]
        Sigma: [B, K, 3, 3]
        eps:   numerical floor
        symmetrize: whether to enforce Sigma = (Sigma + Sigma^T)/2

    Returns:
        geom_feat: [B, K, 8] = [mu(3), logdet(1), logtr(1), logdiag(3)]
    """
    assert mu.shape[-1] == 3
    assert Sigma.shape[-2:] == (3, 3)
    assert mu.shape[:-1] == Sigma.shape[:-2]

    S = 0.5 * (Sigma + Sigma.transpose(-1, -2)) if symmetrize else Sigma

    # Unpack symmetric entries
    s00 = S[..., 0, 0]
    s01 = S[..., 0, 1]
    s02 = S[..., 0, 2]
    s11 = S[..., 1, 1]
    s12 = S[..., 1, 2]
    s22 = S[..., 2, 2]

    # det(S) for symmetric 3x3 via explicit formula (no decomposition)
    det = (
        s00 * (s11 * s22 - s12 * s12)
        - s01 * (s01 * s22 - s02 * s12)
        + s02 * (s01 * s12 - s02 * s11)
    )
    det = det.clamp_min(eps)
    logdet = torch.log(det).unsqueeze(-1)  # [B,K,1]

    # diag + trace
    diag = torch.stack([s00, s11, s22], dim=-1).clamp_min(eps)  # [B,K,3]
    tr = diag.sum(dim=-1, keepdim=True)                         # [B,K,1]
    logtr = torch.log(tr + eps)                                 # [B,K,1]
    logdiag = torch.log(diag)                                   # [B,K,3]

    geom_feat = torch.cat([mu, logdet, logtr, logdiag], dim=-1)  # [B,K,8]
    return geom_feat

# --------------------------
# losses container
# --------------------------

@dataclass
class UpLoss:
    count: torch.Tensor     # budget constraint
    sparse: torch.Tensor    # optional sparsity
    total: torch.Tensor


# --------------------------
# main Up module
# --------------------------

class GaussianUpsampleIGAModule(nn.Module):
    """
    Upsample from level l (K_l tokens) to candidate children (K_cand = K_l * M_max).

    You can then run a "coarse IGA tower" on the candidate children using the SOFT mask.

    Key design decisions:
    1) We ALWAYS generate a fixed candidate pool per parent (M_max). (batched tensors)
    2) Learn continuous gate g_{j,t} in [0,1] to control expected count.
    3) Responsibility B_{i->j} uses overlap score (no explicit inverse).
       - parent_topk=1 => parent-only (pure split)
       - parent_topk=None => all parents (helps when geometry is wrong; global correction)
    4) Geometry refined by moment matching under B.
    5) Semantics uplifted using same B and g (strong coupling).
    """

    def __init__(
        self,
        c_s: int,
        M_max: int = 8,
        phi: float = 1.6,
        lam: float = 1e-4,
        eta_init: float = 1.0,
        beta_init: float = 0.0,
        alpha_init: float = 1.0,
        parent_topk: Optional[int] = None,
        c_out: Optional[int] = None,
    ):
        super().__init__()
        self.c_s = c_s
        self.M_max = int(M_max)
        self.phi = float(phi)
        self.lam = float(lam)

        # anneal knobs
        self.eta = float(eta_init)
        self.beta = float(beta_init)
        self.alpha = float(alpha_init)

        self.parent_topk = parent_topk
        self.c_out = int(c_out) if c_out is not None else c_s

        # prior a_j = softplus(u_j)
        self.prior_mlp = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_s),
            nn.SiLU(),
            nn.Linear(c_s, 1),
        )

        # candidate id embedding (t=1..M_max)
        self.child_id_embed = nn.Embedding(self.M_max, c_s)

        # gate logits
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(2 * c_s),
            nn.Linear(2 * c_s, c_s),
            nn.SiLU(),
            nn.Linear(c_s, 1),
        )

        # semantic projection
        self.sem_proj = nn.Linear(c_s, self.c_out)

        # geometry embedding -> semantics
        self.geom_proj = nn.Sequential(
            nn.Linear(8, self.c_out),
            nn.SiLU(),
            nn.Linear(self.c_out, self.c_out),
        )

    @torch.no_grad()
    def update_anneal(self, t: float):
        """Optional: call with t in [0,1] each step."""
        t = float(max(0.0, min(1.0, t)))
        self.eta = 1.5 - 1.0 * t     # softer -> harder
        self.beta = 0.0 + 1.0 * t    # prior grows
        self.alpha = 1.0 + 3.0 * t   # B sharper

    def forward(
        self,
        s_l: torch.Tensor,         # [B, Kp, C]
        r_l,                       # OffsetGaussianRigid-like [B, Kp]
        mask_l: torch.Tensor,      # [B, Kp]
        OffsetGaussianRigid_cls,
        K_target: Optional[torch.Tensor] = None,  # scalar or [B], expected total children
        w_count: float = 1.0,
        w_sparse: float = 0.0,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, object, torch.Tensor, UpLoss, dict]:
        Bsz, Kp, C = s_l.shape
        device, dtype = s_l.device, s_l.dtype

        # 1) parent geometry
        mu_p = r_l.get_gaussian_mean()     # [B,Kp,3]
        Sigma_p = r_l.get_covariance()     # [B,Kp,3,3]
        Sigma_p = 0.5 * (Sigma_p + Sigma_p.transpose(-1, -2))

        I = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
        Sigma_p_j = Sigma_p + (self.lam + eps) * I

        # 2) prior a_j > 0
        u = self.prior_mlp(s_l).squeeze(-1)            # [B,Kp]
        a = F.softplus(u) * mask_l                     # [B,Kp]

        # 3) candidate pool geometry split
        M = self.M_max
        Kcand = Kp * M

        xi = torch.randn((Bsz, Kp, M, 3), device=device, dtype=dtype)
        L = torch.linalg.cholesky(Sigma_p_j)  # [B,Kp,3,3]
        d = torch.einsum("bkij,bkmj->bkmi", L, xi)  # [B,Kp,M,3]
        mu0 = mu_p.unsqueeze(2) + d  # [B,Kp,M,3]
        Sigma0 = (Sigma_p * (self.phi ** -2)).unsqueeze(2) + self.lam * I  # [B,Kp,M,3,3]

        mu0_flat = mu0.reshape(Bsz, Kcand, 3)
        Sigma0_flat = Sigma0.reshape(Bsz, Kcand, 3, 3)

        mask0 = mask_l.unsqueeze(2).expand(Bsz, Kp, M).reshape(Bsz, Kcand)

        # 4) gates g_{j,t}
        e_t = self.child_id_embed(torch.arange(M, device=device)).view(1, 1, M, C)
        e_t = e_t.expand(Bsz, Kp, M, C)
        s_rep = s_l.unsqueeze(2).expand(Bsz, Kp, M, C)
        gate_in = torch.cat([s_rep, e_t], dim=-1)
        b = self.gate_mlp(gate_in).squeeze(-1)  # [B,Kp,M]
        b = b + self.beta * torch.log(a + eps).unsqueeze(-1)

        eta = max(self.eta, 1e-4)
        g = torch.sigmoid(b / eta) * mask_l.unsqueeze(-1)   # [B,Kp,M]
        g_flat = g.reshape(Bsz, Kcand) * mask0              # [B,Kcand]

        # 5) budget loss
        if K_target is None:
            K_target = torch.full((Bsz,), float(Kcand) * 0.5, device=device, dtype=dtype)
        elif K_target.ndim == 0:
            K_target = K_target.expand(Bsz)
        else:
            assert K_target.shape == (Bsz,), "K_target must be scalar or [B]"

        expK = g_flat.sum(dim=-1)  # [B]
        loss_count = ((expK - K_target) ** 2).mean()
        loss_sparse = g_flat.mean()

        # 6) responsibility B_{i->j}
        delta = mu0_flat.unsqueeze(2) - mu_p.unsqueeze(1)  # [B,Kcand,Kp,3]
        sigma = Sigma_p_j.unsqueeze(1).expand(Bsz, Kcand, Kp, 3, 3)
        score = fused_gaussian_overlap_score(delta, sigma)  # [B,Kcand,Kp]
        score = score.masked_fill((mask_l < 0.5).unsqueeze(1), -1e9)

        if self.parent_topk is not None and self.parent_topk < Kp:
            topk = int(self.parent_topk)
            vals, idx = torch.topk(score, k=topk, dim=-1)  # [B,Kcand,topk]
            probs = F.softmax(self.alpha * vals, dim=-1)
            B_resp = torch.zeros((Bsz, Kcand, Kp), device=device, dtype=dtype)
            B_resp.scatter_(dim=-1, index=idx, src=probs)
        else:
            B_resp = F.softmax(self.alpha * score, dim=-1)

        # 7) geometry moment refinement
        mu_c, Sigma_c = moment_refine_children_from_B(
            mu_parent=mu_p,
            Sigma_parent=Sigma_p_j,
            B_resp=B_resp,
            phi=self.phi,
            lam=self.lam,
        )

        # 8) build child rigids
        r_child = build_rigid_from_mu_sigma(mu_c, Sigma_c, OffsetGaussianRigid_cls, eps=eps)

        # 9) semantics uplift (same B and gate)
        s_tilde = torch.einsum("bik,bkc->bic", B_resp, s_l)  # [B,Kcand,C]
        s_child = self.sem_proj(s_tilde)                       # [B,Kcand,C_out]

        geom_feat = geom_feat_from_mu_sigma_fast(mu_c, Sigma_c, eps=eps)
        s_child = s_child + self.geom_proj(geom_feat)

        mask_child = g_flat.clamp(0.0, 1.0)
        s_child = s_child * mask_child.unsqueeze(-1)

        total = w_count * loss_count + w_sparse * loss_sparse
        losses = UpLoss(count=loss_count, sparse=loss_sparse, total=total)

        aux = {
            "a_prior": a,          # [B,Kp]
            "g_gate": g_flat,      # [B,Kcand]
            "B_resp": B_resp,      # [B,Kcand,Kp]
            "mu0": mu0_flat,       # [B,Kcand,3]
            "Sigma0": Sigma0_flat, # [B,Kcand,3,3]
            "mu": mu_c,            # [B,Kcand,3]
            "Sigma": Sigma_c,      # [B,Kcand,3,3]
            "expK": expK,          # [B]
            "K_target": K_target,  # [B]
        }

        return s_child, r_child, mask_child, losses, aux




# ========== 你已有的: fused_gaussian_overlap_score ==========
# from your_code import fused_gaussian_overlap_score

# ========== 你已有的: moment refine + rigids build ==========
# from upsample_block import moment_refine_children_from_B
# from your_rigid_utils import coarse_rigids_from_mu_sigma





class GaussianSplatGateUpInit(nn.Module):
    """
    Up-init: 仅生成初始化，不做 IGA
    1) 每个父 j 生成 M_max 个候选子点 (mu0,Sigma0)
    2) gate g_{j,t} 控制“期望数量” + count loss
    3) 计算 B_{i,j}（局部邻域可选：只看出生父+TopR-1）用于 moment refine
    4) moment_refine 得到 (mu_child, Sigma_child)
    5) 语义初始化：用同一套 B/g 做强绑定（防语义-几何脱钩）
    """
    def __init__(
        self,
        c_s: int,
        M_max: int = 8,
        neighbor_R: int = 1,          # N(i)={j0} 就是 R=1；R>1 则加入 Top(R-1) 近邻父
        phi: float = 1.6,             # split shrink
        jitter: float = 1e-4,         # SPD jitter
        eta_init: float = 1.0,        # gate temperature
        beta_init: float = 0.0,       # a_j prior weight
    ):
        super().__init__()
        self.c_s = c_s
        self.M_max = M_max
        self.neighbor_R = neighbor_R
        self.phi = phi
        self.jitter = jitter

        # gate 分支（候选子序号 embedding + 父语义）
        self.child_id_embed = nn.Embedding(M_max, c_s)
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_s),
            nn.SiLU(),
            nn.Linear(c_s, 1),
        )

        # 父“拆分强度”先验 a_j（可选）
        self.prior_mlp = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_s),
            nn.SiLU(),
            nn.Linear(c_s, 1),
        )

        # 退火参数（由外部 schedule 改）
        self.eta = eta_init
        self.beta = beta_init

        # [新增] 将候选 ID (embedding) 映射为几何偏移
        # 它可以学习到：Child_0 -> (-1, 0, 0), Child_1 -> (1, 0, 0) 这种模式
        self.geom_bias_proj = nn.Linear(c_s, 3)

        # 初始化为接近 0，让随机噪声先主导，慢慢学出结构
        nn.init.normal_(self.geom_bias_proj.weight, std=0.01)
        nn.init.zeros_(self.geom_bias_proj.bias)

    @torch.no_grad()
    def _top_neighbors_by_overlap(self, mu_p, Sigma_p, j0_idx, R: int):
        """
        选 Top(R-1) 近邻父的 index（按 overlap score）。
        mu_p: [B,Kp,3], Sigma_p:[B,Kp,3,3]
        j0_idx: [B,Kcand] 出生父索引
        return neigh_idx: [B,Kcand,R]，第0列就是j0
        """
        B, Kp, _ = mu_p.shape
        B2, Kcand = j0_idx.shape
        assert B == B2
        device = mu_p.device

        # delta cand-parent: 先用出生父中心作为 proxy，找与其它父的相近度
        mu0 = mu_p.gather(1, j0_idx[..., None].expand(-1, -1, 3))  # [B,Kcand,3]
        delta = mu0[:, :, None, :] - mu_p[:, None, :, :]          # [B,Kcand,Kp,3]
        # 使用 parent Sigma 做近似（也可以用 Sigma_sum）
        sigma = Sigma_p[:, None, :, :, :].expand(B, Kcand, Kp, 3, 3)
        score = fused_gaussian_overlap_score(delta, sigma)        # [B,Kcand,Kp]

        # 把出生父的位置强制放到最高
        # score[b,i,j0]=+inf
        big = torch.full_like(score[..., 0], 1e9)
        score.scatter_(2, j0_idx.unsqueeze(-1), big.unsqueeze(-1))

        top = torch.topk(score, k=min(R, Kp), dim=2).indices       # [B,Kcand,R]
        return top

    def forward(self, s_parent, r_parent, mask_parent, K_target: float | None = None):
        """
        Inputs:
          s_parent: [B,Kp,C]
          r_parent: OffsetGaussianRigid [B,Kp]
          mask_parent: [B,Kp] (0/1)

        Outputs:
          s_child0: [B,Kcand,C]  (初始化)
          mu_child: [B,Kcand,3]
          Sigma_child:[B,Kcand,3,3]
          mask_child_soft: [B,Kcand] (用 g 做软mask)
          aux: dict( g, B, losses... )
        """
        B, Kp, C = s_parent.shape
        device = s_parent.device
        dtype = s_parent.dtype

        mu_p = r_parent.get_gaussian_mean()      # [B,Kp,3]
        Sigma_p = r_parent.get_covariance()      # [B,Kp,3,3]

        # ---- 1) 每个父生成 M_max 个候选（几何 splat / split）
        # 候选索引展平：i = j*M + t
        M = self.M_max
        Kcand = Kp * M

        # 采样噪声：xi ~ N(0, I)
        xi_noise = torch.randn(B, Kp, M, 3, device=device, dtype=dtype)
        # 2. [新增] 准备可学习的结构化偏移 (确定性)
        # t_ids: [B, Kcand] -> [B, Kp, M] (假设你这里reshape一下或者直接用 range)
        # 更简单的写法，直接生成 embedding:
        t_range = torch.arange(M, device=device)
        child_emb = self.child_id_embed(t_range)  # [M, c_s]

        # 映射到 3D 偏移
        xi_bias = self.geom_bias_proj(child_emb)  # [M, 3]

        # 广播到 [B, Kp, M, 3]
        xi_bias = xi_bias.unsqueeze(0).unsqueeze(0).expand(B, Kp, M, 3)

        # 3. 混合：位置 = 父中心 + 形状 * (随机噪声 + 结构化偏移)
        # 这样梯度就能传导给 xi_bias，让模型学会把不同的 M 放到不同位置！
        xi_total = xi_noise + xi_bias

        # 用 Cholesky 得到 L（你也可以用你已有的更稳定实现）
        eps = 1e-6
        I = torch.eye(3, device=device, dtype=dtype)
        L = torch.linalg.cholesky(Sigma_p + eps * I)               # [B,Kp,3,3]

        # mu0_{j,t} = mu_j + L_j xi
        mu0 = mu_p[:, :, None, :] + torch.einsum("bkae,bkme->bkma", L, xi_total)  # [B,Kp,M,3]

        # Sigma0 收缩 + jitter
        Sigma0 = (Sigma_p[:, :, None, :, :] / (self.phi ** 2)) + self.jitter * I  # [B,Kp,M,3,3]

        # 展平候选
        mu0 = mu0.reshape(B, Kcand, 3)
        # Sigma0 = Sigma0.reshape(B, Kcand, 3, 3)

        # 出生父索引 j0
        j0 = torch.arange(Kp, device=device)[None, :, None].expand(B, Kp, M).reshape(B, Kcand)

        # 候选的父mask
        m0 = mask_parent.gather(1, j0)  # [B,Kcand]

        # ---- 2) gate：连续决定期望数量 + count loss
        # gate logits b_{j,t} = MLP([s_j + e_t])
        t_ids = torch.arange(M, device=device)[None, None, :].expand(B, Kp, M).reshape(B, Kcand)
        e_t = self.child_id_embed(t_ids)                                # [B,Kcand,C]
        s_j = s_parent.gather(1, j0[..., None].expand(-1, -1, C))       # [B,Kcand,C]
        gate_in = s_j + e_t

        b = self.gate_mlp(gate_in).squeeze(-1)                          # [B,Kcand]

        # a_j prior（可选）：只提供偏置，不单独决定数量
        a = F.softplus(self.prior_mlp(s_parent).squeeze(-1))            # [B,Kp]
        a_i = a.gather(1, j0)                                           # [B,Kcand]
        b = b + self.beta * torch.log(a_i + 1e-8)

        g = torch.sigmoid(b / max(self.eta, 1e-6)) * m0                 # [B,Kcand] in [0,1]

        # 期望数量预算
        if K_target is None:
            # 不给预算：用轻量稀疏正则替代（你也可以直接返回0）
            loss_count = (g.mean() * 0.0)
        else:
            loss_count = (g.sum(dim=1) - float(K_target)).pow(2).mean()

        # ---- 3) 责任 B：用 overlap 构建（建议局部邻域）
        # 你问：子椭圆需不需要和“非父”的 j 交互？
        # 答：Up-init 的 B 用于“纠错/重定位”时，建议至少看 Top(R-1) 近邻父；
        #     但最小可用版本 R=1 只看出生父。
        R = self.neighbor_R
        if R <= 1:
            # B 退化为 one-hot 出生父（这就是你说的“档 B”，纠错弱但最稳）
            Bmat = torch.zeros(B, Kcand, Kp, device=device, dtype=dtype)
            Bmat.scatter_(2, j0.unsqueeze(-1), 1.0)
        else:
            neigh = self._top_neighbors_by_overlap(mu_p, Sigma_p, j0, R)  # [B,Kcand,R]
            # 对 neigh 内做 softmax
            mu_nei = mu_p[:, None, :, :].expand(B, Kcand, Kp, 3).gather(
                dim=2,
                index=neigh[..., None].expand(B, Kcand, R, 3),
            )  # [B, Kcand, R, 3]
            Sig_nei = Sigma_p[:, None, :, :, :].expand(B, Kcand, Kp, 3, 3).gather(
                dim=2,
                index=neigh[..., None, None].expand(B, Kcand, R, 3, 3),
            )  # [B, Kcand, R, 3, 3]

            delta = mu0[:, :, None, :] - mu_nei                                           # [B,Kcand,R,3]
            score = fused_gaussian_overlap_score(delta, Sig_nei)                          # [B,Kcand,R]

            alpha = 1.0
            w = F.softmax(alpha * score, dim=-1)                                          # [B,Kcand,R]

            Bmat = torch.zeros(B, Kcand, Kp, device=device, dtype=dtype)
            Bmat.scatter_add_(2, neigh, w)  # 把 w 加到对应 parent index 上

        # ---- 4) moment refine：得到子几何（mu,Sigma）
        mu_child, Sigma_child = moment_refine_children_from_B(
            mu_parent=mu_p,
            Sigma_parent=Sigma_p,
            B_resp=Bmat,
            phi=self.phi,
            lam=self.jitter,
        )

        # ---- 5) 语义初始化：强绑定（同一套 B + gate）
        s_mix = torch.einsum("bik,bkc->bic", Bmat, s_parent)            # [B,Kcand,C]
        s_child0 = g.unsqueeze(-1) * s_mix

        # soft mask：训练期可用 g；推理期阈值化
        mask_child_soft = g

        aux = {
            "g": g,
            "B": Bmat,
            "loss_count": loss_count,
        }
        return s_child0, mu_child, Sigma_child, mask_child_soft, aux






class HierarchicalUpsampleIGAModule(nn.Module):
    """
    做 num_upsample 次：
      1) Up-init（splat+gate+B+moment）得到 (s0, mu, Sigma, mask_soft)
      2) build rigids r0
      3) coarse IGA × k： (s,r) = IGA+Transition+Update
      4) 作为下一次输入

    返回每层 levels（skip / debug / 可视化都能用）
    """

    def __init__(
        self,
        c_s: int,
        iga_conf,
        OffsetGaussianRigid_cls,
        num_upsample: int = 2,
        M_max: int = 8,
        K_target: int | list[int] | None = None,   # None 表示用 up_ratio 自动给“期望预算”
        up_ratio: float = 6.0,
        neighbor_R: int = 2,
        coarse_iga_layers: int | list[int] = 4,
        phi: float = 1.6,
        jitter: float = 1e-4,
        eta_init: float = 1.0,
        beta_init: float = 0.0,
    ):
        super().__init__()
        self.num_upsample = num_upsample
        self.OffsetGaussianRigid_cls = OffsetGaussianRigid_cls
        self.up_ratio = up_ratio

        # per-level K_target
        if isinstance(K_target, list):
            assert len(K_target) == num_upsample, "K_target list length must equal num_upsample"
            self.K_target_list = list(K_target)
        else:
            self.K_target_list = [K_target] * num_upsample  # None 或 int

        # per-level iga layers
        if isinstance(coarse_iga_layers, int):
            per_level_layers = [coarse_iga_layers] * num_upsample
        else:
            assert len(coarse_iga_layers) == num_upsample, "coarse_iga_layers list length must equal num_upsample"
            per_level_layers = list(coarse_iga_layers)

        # Up-init blocks（纯上采样初始化：几何/语义候选 + gate + B + moment）
        self.up_inits = nn.ModuleList([
            GaussianSplatGateUpInit(
                c_s=c_s,
                M_max=M_max,
                neighbor_R=neighbor_R,
                phi=phi,
                jitter=jitter,
                eta_init=eta_init,
                beta_init=beta_init,
            )
            for _ in range(num_upsample)
        ])

        # Up-IGA towers per level（复用你已有的 CoarseIGATower 语义：IGA+Trans+Update × k）
        self.up_towers = nn.ModuleList()
        for lv in range(num_upsample):
            iga = InvariantGaussianAttention(
                c_s=c_s,
                c_z=getattr(iga_conf, "hgfc_z", 0),
                c_hidden=iga_conf.c_hidden,
                no_heads=iga_conf.no_heads,
                no_qk_gaussians=iga_conf.no_qk_points,
                no_v_points=iga_conf.no_v_points,
                layer_idx=2000 + lv,
                enable_vis=False,
            )
            gau_update = GaussianUpdateBlock(c_s)

            self.up_towers.append(
                CoarseIGATower(
                    iga=iga,
                    gau_update=gau_update,
                    c_s=c_s,
                    num_layers=per_level_layers[lv],
                )
            )

        self.sibling_loss_fn = HierarchicalGaussianLoss(w_sep=50.0, w_compact=1.0)

    @torch.no_grad()
    def _auto_K_target(self, mask_parent: torch.Tensor, up_ratio: float) -> int:
        """
        给一个“期望预算”（不是最终 K）：用当前有效父 token 数 × up_ratio
        mask_parent: [B, K_l] float/bool
        """
        # batch mean of active tokens
        k_l = mask_parent.sum(dim=-1).float().mean()
        K_target = int(torch.round(k_l * up_ratio).clamp_min(1.0).item())
        return K_target

    def forward(self, s_l, r_l, mask_l, step: int, total_steps: int):
        """
        Inputs:
          s_l: [B, K_l, C]
          r_l: OffsetGaussianRigid [B, K_l]
          mask_l: [B, K_l] (float/bool)
        Outputs:
          levels: list[dict]
          reg_total: scalar (count/prior 等正则总和)
        """
        levels = []
        reg_total = 0.0

        s, r, mask = s_l, r_l, mask_l

        # schedule（跟 down 一样的哲学：早期软，后期硬）
        t = float(step) / max(int(total_steps), 1)

        for lv in range(self.num_upsample):
            # 退火：gate 硬度 eta、prior 权重 beta
            # 你可以按实验调整这两条线
            self.up_inits[lv].eta = 1.5 - 1.0 * t     # 早期更软，后期更硬
            self.up_inits[lv].beta = 0.0 + 1.0 * t    # 早期不用 prior，后期加强 prior

            # 目标期望预算（不是最终 K）
            K_target = self.K_target_list[lv]
            if K_target is None:
                K_target = self._auto_K_target(mask, self.up_ratio)

            # --- Up-init：候选子椭圆（几何 splat）+ gate（连续数量）+ B（责任）+ moment refine ---
            # s0:       [B, Kcand, C]
            # mu,Sigma: [B, Kcand, ...]
            # mask_soft:[B, Kcand] 0~1
            s0, mu, Sigma, mask_soft, aux = self.up_inits[lv](
                s_parent=s,
                r_parent=r,
                mask_parent=mask,
                K_target=K_target,
            )

            # -----------------------------------------------------------
            # 【关键插入点】计算兄弟节点排斥 Loss
            # -----------------------------------------------------------
            # mu: [B, Kcand, 3], Sigma: [B, Kcand, 3, 3]
            # Kcand = Kp * M
            # 我们需要 reshape 回 [B, Kp, M] 来计算 sibling repulsion

            B, Kcand, _ = mu.shape
            M = self.up_inits[lv].M_max
            Kp = Kcand // M

            # Reshape: 把同一个父节点的 M 个孩子聚在一起
            # [B * Kp, M, 3]
            mu_siblings = mu.view(B * Kp, M, 3)
            Sigma_siblings = Sigma.view(B * Kp, M, 3, 3)

            # Mask 也要 reshape
            # mask_soft: [B, Kcand] -> [B * Kp, M]
            mask_siblings = mask_soft.view(B * Kp, M)

            # 计算 Loss
            # 这里实际上是在并行的计算 (B*Kp) 组高斯分布内部的排斥力
            loss_dict = self.sibling_loss_fn(
                mu_c=mu_siblings,
                Sigma_c=Sigma_siblings,
                mask_c=mask_siblings
            )
            # 加入总正则
            reg_total += loss_dict["total_hier"]  # 主要是排斥


            # 正则（至少包含 count loss；aux 里你也可以放 prior/min 等）
            if isinstance(aux, dict) and ("loss_count" in aux):
                reg_total = reg_total + aux["loss_count"]

            # --- build rigids（你要的 OffsetGaussianRigid） ---
            r0 = coarse_rigids_from_mu_sigma(mu, Sigma, self.OffsetGaussianRigid_cls)

            # --- Up-IGA refine × k（必须）：IGA + Transition + GauUpdate ---
            s1, r1 = self.up_towers[lv](s0, r0, mask_soft)

            levels.append({
                "s0": s0,
                "r0": r0,
                "mask0": mask_soft,
                "aux": aux,
                "s": s1,
                "r": r1,
                "mask": mask_soft,
                "K_target": K_target,
            })

            # next level
            s, r, mask = s1, r1, mask_soft

        return levels, reg_total

