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
from models.IGA import InvariantGaussianAttention,CoarseIGATower,GaussianUpdateBlock,fused_gaussian_overlap_score,FastTransformerTower
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



@torch.no_grad()
def sample_from_mixture(mu_p, Sig_p, pi, M, mask_parent=None, eps=1e-6):
    """
    mu_p: [B,K,3], Sig_p: [B,K,3,3], pi: [B,K] normalized
    return cand_x: [B,M,3], cand_pid: [B,M]
    """
    B, K, _ = mu_p.shape
    device, dtype = mu_p.device, mu_p.dtype
    I = torch.eye(3, device=device, dtype=dtype)[None, None]

    if mask_parent is not None:
        pi = pi * (mask_parent > 0.5).to(pi.dtype)
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    pid = torch.multinomial(pi, num_samples=M, replacement=True)  # [B,M]
    mu_k = mu_p.gather(1, pid[..., None].expand(B, M, 3))
    Sig_k = Sig_p.gather(1, pid[..., None, None].expand(B, M, 3, 3))
    Sig_k = 0.5 * (Sig_k + Sig_k.transpose(-1, -2)) + eps * I

    L = torch.linalg.cholesky(Sig_k)
    z = torch.randn((B, M, 3), device=device, dtype=dtype)
    x = mu_k + torch.einsum("bmij,bmj->bmi", L, z)
    return x, pid



@torch.no_grad()
def init_sigma_from_child_spacing(
    mu0: torch.Tensor,            # [B,N,3]
    node_mask: torch.Tensor,      # [B,N] 1/0
    k_nn: int = 8,
    alpha: float = 1.5,
    sigma_floor: float = 1e-3,
    sigma_ceil: float = 1e2,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    Estimate an isotropic covariance from local spacing.
    Return: [B,N,3,3]
    """
    B, N, _ = mu0.shape
    device, dtype = mu0.device, mu0.dtype
    I = torch.eye(3, device=device, dtype=dtype)

    # pairwise dist2
    dist2 = torch.cdist(mu0, mu0, p=2) ** 2  # [B,N,N]
    # mask invalid children
    big = 1e9
    dist2 = dist2.masked_fill((node_mask < 0.5).unsqueeze(1), big)
    dist2 = dist2.masked_fill((node_mask < 0.5).unsqueeze(2), big)
    # exclude self
    dist2 = dist2 + torch.eye(N, device=device, dtype=dtype).unsqueeze(0) * big

    k_eff = min(k_nn, max(N - 1, 1))
    d2_knn, _ = torch.topk(dist2, k=k_eff, dim=-1, largest=False)  # [B,N,k]
    # median or mean of nearest distances
    d = torch.sqrt(d2_knn.clamp_min(0.0) + 1e-12)
    # use mean as radius proxy
    r = d.mean(dim=-1) * alpha  # [B,N]

    r = r.clamp(min=sigma_floor, max=sigma_ceil)
    var = (r ** 2).unsqueeze(-1)  # [B,N,1]
    Sig = I.view(1, 1, 3, 3) * var.view(B, N, 1, 1)
    Sig = Sig + jitter * I.view(1, 1, 3, 3)
    Sig = Sig * node_mask[:, :, None, None]
    return Sig

@torch.no_grad()
def init_semantic_from_mu_to_parents(
    mu0: torch.Tensor,            # [B,N,3]
    s_parent: torch.Tensor,       # [B,K,C]
    mu_p: torch.Tensor,           # [B,K,3]
    pi: torch.Tensor,             # [B,K]
    mask_parent: torch.Tensor,    # [B,K]
    sigma_s: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute soft parent responsibility w (B,N,K) from distance in mu-space,
    then s0 = w @ s_parent.
    """
    B, N, _ = mu0.shape
    K = mu_p.shape[1]
    device, dtype = mu0.device, mu0.dtype

    dist2 = ((mu0.unsqueeze(2) - mu_p.unsqueeze(1)) ** 2).sum(dim=-1)  # [B,N,K]
    sigma2 = max(float(sigma_s), 1e-6) ** 2
    logits = -dist2 / (2.0 * sigma2)

    # include pi prior
    logits = logits + (pi.clamp_min(1e-9).log().unsqueeze(1))

    # mask invalid parents
    logits = logits.masked_fill((mask_parent < 0.5).unsqueeze(1), -1e9)

    w = torch.softmax(logits, dim=-1)  # [B,N,K]
    s0 = torch.einsum("bnk,bkc->bnc", w, s_parent)
    return s0, w


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
        # L = torch.linalg.cholesky(Sigma_p + eps * I)               # [B,Kp,3,3]
        #
        # # mu0_{j,t} = mu_j + L_j xi
        # mu0 = mu_p[:, :, None, :] + torch.einsum("bkae,bkme->bkma", L, xi_total)  # [B,Kp,M,3]

        # ---- robust sqrt-cov via eigh (sampling only) ----
        S = 0.5 * (Sigma_p + Sigma_p.transpose(-1, -2))  # [B,Kp,3,3]
        evals, evecs = torch.linalg.eigh(S)  # evals: [B,Kp,3], evecs: [B,Kp,3,3]

        # clamp eigenvalues to make SPD
        jitter = eps  # 你原本的 eps 或者更大一点如 1e-6/1e-5
        evals = evals.clamp_min(jitter)

        # build sqrt factor A = Q * sqrt(Λ)
        sqrt_evals = torch.sqrt(evals)  # [B,Kp,3]
        A = evecs * sqrt_evals.unsqueeze(-2)  # [B,Kp,3,3]  (each column scaled)

        # mu0_{j,t} = mu_j + A_j xi
        mu0 = mu_p[:, :, None, :] + torch.einsum("bkae,bkme->bkma", A, xi_total)  # [B,Kp,M,3]

        # Sigma0 收缩 + jitter
        Sigma_p = 0.5 * (Sigma_p + Sigma_p.transpose(-1, -2))
        Sigma0 = (Sigma_p[:, :, None] / (self.phi ** 2)) + self.jitter * I

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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpLoss(nn.Module):
    # 你也可以用 dataclass，这里保持简单
    pass


class SemanticGateUpInit(nn.Module):
    """
    Pure semantic up-init:
      parent: (s_l, mask_l) with Kp tokens
      child candidates: Kcand = Kp * M_max
      gate g in [0,1] controls expected active count

    Returns:
      s_child: [B, Kcand, C]
      mask_child: [B, Kcand] (soft in [0,1])
      losses: dict (count/sparse/total)
      aux: dict
    """
    def __init__(
        self,
        c_s: int,
        M_max: int = 8,
        tau_init: float = 1.0,    # gate temperature: smaller -> harder
        beta_init: float = 0.0,   # parent prior strength
        c_out: int | None = None,
        attn_heads: int = 4,      # optional cross-attn
        use_cross_attn: bool = True,
    ):
        super().__init__()
        self.c_s = c_s
        self.M_max = int(M_max)
        self.tau = float(tau_init)
        self.beta = float(beta_init)
        self.c_out = int(c_out) if c_out is not None else c_s
        self.use_cross_attn = bool(use_cross_attn)

        # parent prior a_j = softplus(u_j)
        self.prior_mlp = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, c_s),
            nn.SiLU(),
            nn.Linear(c_s, 1),
        )

        # candidate id embedding (t=0..M-1)
        self.child_id_embed = nn.Embedding(self.M_max, c_s)

        # gate logits: b_{j,t} from [s_j, e_t]
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(2 * c_s),
            nn.Linear(2 * c_s, c_s),
            nn.SiLU(),
            nn.Linear(c_s, 1),
        )

        # semantic projection
        self.sem_proj = nn.Linear(c_s, self.c_out)

        # optional cross-attn: child queries attend to parent memory
        self.attn_heads = int(attn_heads)
        assert self.c_out % self.attn_heads == 0
        d_head = self.c_out // self.attn_heads

        self.q_proj = nn.Linear(self.c_out, self.c_out, bias=False)
        self.k_proj = nn.Linear(self.c_out, self.c_out, bias=False)
        self.v_proj = nn.Linear(self.c_out, self.c_out, bias=False)
        self.out_proj = nn.Linear(self.c_out, self.c_out, bias=False)
        self.d_head = d_head

        self.ln = nn.LayerNorm(self.c_out)

    @torch.no_grad()
    def update_anneal(self, t: float):
        """Optional schedule t in [0,1]."""
        t = float(max(0.0, min(1.0, t)))
        self.tau = 1.2 - 0.9 * t  # soft -> harder
        self.beta = 0.0 + 1.0 * t # prior grows

    def _cross_attn(self, q, k, v, mask_k):
        """
        q: [B,Q,C], k/v: [B,K,C], mask_k: [B,K]
        """
        B, Q, C = q.shape
        K = k.shape[1]
        H = self.attn_heads
        d = self.d_head

        qh = self.q_proj(q).view(B, Q, H, d).transpose(1, 2)  # [B,H,Q,d]
        kh = self.k_proj(k).view(B, K, H, d).transpose(1, 2)  # [B,H,K,d]
        vh = self.v_proj(v).view(B, K, H, d).transpose(1, 2)  # [B,H,K,d]

        logits = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(d)  # [B,H,Q,K]
        logits = logits + (mask_k[:, None, None, :] - 1.0) * 1e9

        w = torch.softmax(logits, dim=-1)
        out = torch.matmul(w, vh)  # [B,H,Q,d]
        out = out.transpose(1, 2).contiguous().view(B, Q, C)
        return self.out_proj(out)

    def forward(
        self,
        s_l: torch.Tensor,      # [B,Kp,C]
        mask_l: torch.Tensor,   # [B,Kp]
        K_target: torch.Tensor | float | int | None = None,  # scalar or [B]
        w_count: float = 1.0,
        w_sparse: float = 0.0,
        eps: float = 1e-6,
    ):
        B, Kp, C = s_l.shape
        device, dtype = s_l.device, s_l.dtype
        M = self.M_max
        Kcand = Kp * M

        # parent prior a_j > 0
        a = F.softplus(self.prior_mlp(s_l).squeeze(-1)) * mask_l  # [B,Kp]

        # candidate mapping: i = j*M + t
        j0 = torch.arange(Kp, device=device)[None, :, None].expand(B, Kp, M).reshape(B, Kcand)
        t0 = torch.arange(M, device=device)[None, None, :].expand(B, Kp, M).reshape(B, Kcand)

        s_j = s_l.gather(1, j0[..., None].expand(-1, -1, C))  # [B,Kcand,C]
        e_t = self.child_id_embed(t0)                          # [B,Kcand,C]

        # gate logits
        gate_in = torch.cat([s_j, e_t], dim=-1)                # [B,Kcand,2C]
        b = self.gate_mlp(gate_in).squeeze(-1)                 # [B,Kcand]

        # add parent prior term
        a_i = a.gather(1, j0)                                  # [B,Kcand]
        b = b + self.beta * torch.log(a_i + eps)

        g = torch.sigmoid(b / max(self.tau, 1e-4))
        g = g * mask_l.gather(1, j0)  # invalid parents -> 0

        # count loss (budget)
        if K_target is None:
            loss_count = g.mean() * 0.0
            K_target_b = None
        else:
            if not torch.is_tensor(K_target):
                K_target = torch.tensor(float(K_target), device=device, dtype=dtype)
            if K_target.ndim == 0:
                K_target_b = K_target.expand(B)
            else:
                K_target_b = K_target
            expK = g.sum(dim=1)
            loss_count = (expK - K_target_b).pow(2).mean()

        loss_sparse = g.mean()
        losses = {
            "count": loss_count,
            "sparse": loss_sparse,
            "total": w_count * loss_count + w_sparse * loss_sparse
        }

        # semantic init
        s_child = self.sem_proj(s_j + e_t)  # [B,Kcand,C_out]

        # optional cross-attn to inject global parent context
        if self.use_cross_attn:
            s_parent_proj = self.sem_proj(s_l)  # [B,Kp,C_out]
            attn_out = self._cross_attn(s_child, s_parent_proj, s_parent_proj, mask_l)
            s_child = self.ln(s_child + attn_out)

        # apply soft mask
        s_child = s_child * g.unsqueeze(-1)

        aux = {
            "a_prior": a,      # [B,Kp]
            "g_gate": g,       # [B,Kcand]
            "j0": j0,          # [B,Kcand]
            "t0": t0,          # [B,Kcand]
            "K_target": K_target_b,
        }
        return s_child, g, losses, aux

class HierarchicalUpsampleSemanticModule(nn.Module):
    """
    Pure semantic up (variable length) with gates + transformer refine.
    Drop-in compatible with previous up forward signature:
      forward(s_l, r_l, mask_l, step, total_steps)
    Here r_l is ignored (kept only for compatibility).
    """
    def __init__(
        self,
        c_s: int,
        num_upsample: int = 2,
        M_max: int = 8,
        up_ratio: float = 6.0,
        K_target: int | list[int] | None = None,
        tower_layers: int | list[int] = 4,
        tower_heads: int = 4,
        use_cross_attn: bool = True,
        tau_init: float = 1.0,
        beta_init: float = 0.0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_upsample = int(num_upsample)
        self.up_ratio = float(up_ratio)

        if isinstance(K_target, list):
            assert len(K_target) == self.num_upsample
            self.K_target_list = list(K_target)
        else:
            self.K_target_list = [K_target] * self.num_upsample

        if isinstance(tower_layers, list):
            assert len(tower_layers) == self.num_upsample
            per_layers = list(tower_layers)
        else:
            per_layers = [int(tower_layers)] * self.num_upsample

        self.up_inits = nn.ModuleList([
            SemanticGateUpInit(
                c_s=c_s,
                M_max=M_max,
                tau_init=tau_init,
                beta_init=beta_init,
                use_cross_attn=use_cross_attn,
                attn_heads=tower_heads,
                c_out=c_s,
            )
            for _ in range(self.num_upsample)
        ])

        # ✅ 用你已经写好的更快 tower
        self.towers = nn.ModuleList([
            FastTransformerTower(
                c_s=c_s,
                num_layers=per_layers[lv],
                n_heads=tower_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for lv in range(self.num_upsample)
        ])

    @torch.no_grad()
    def _auto_K_target(self, mask_parent: torch.Tensor, up_ratio: float) -> int:
        # batch mean active tokens
        k_l = mask_parent.sum(dim=-1).float().mean()
        return int(torch.round(k_l * up_ratio).clamp_min(1.0).item())

    def forward(self, s_l, r_l, mask_l, step: int, total_steps: int):
        # r_l ignored, kept for compatibility
        levels = []
        reg_total = torch.zeros((), device=s_l.device, dtype=s_l.dtype)

        s, mask = s_l, mask_l
        t = float(step) / max(int(total_steps), 1)

        for lv in range(self.num_upsample):
            self.up_inits[lv].update_anneal(t)

            Kt = self.K_target_list[lv]
            if Kt is None:
                Kt = self._auto_K_target(mask, self.up_ratio)

            s0, mask0, losses, aux = self.up_inits[lv](
                s_l=s,
                mask_l=mask,
                K_target=Kt,
                w_count=1.0,
                w_sparse=0.0,
            )
            reg_total = reg_total + losses["total"]

            # refine
            s1 = self.towers[lv](s0, mask0)

            levels.append({
                "s0": s0,
                "mask0": mask0,
                "aux": aux,
                "losses": losses,
                "s": s1,
                "mask": mask0,
                "K_target": Kt,
            })

            s, mask = s1, mask0

        return levels, reg_total



import math
import torch
import torch.nn.functional as F


# -----------------------------
# small utils
# -----------------------------
def _safe_cholesky(A: torch.Tensor, jitter: float = 1e-6, max_tries: int = 4):
    """A: [...,3,3] SPD-ish -> L: [...,3,3]"""
    I = torch.eye(3, device=A.device, dtype=A.dtype)
    for t in range(max_tries):
        try:
            return torch.linalg.cholesky(A + (jitter * (10 ** t)) * I)
        except RuntimeError:
            continue
    # last resort: symmetrize + bigger jitter
    A = 0.5 * (A + A.transpose(-1, -2))
    return torch.linalg.cholesky(A + (jitter * (10 ** (max_tries - 1))) * I)


def _mahalanobis2(delta: torch.Tensor, Sigma: torch.Tensor, jitter: float = 1e-6):
    """
    delta: [...,3]
    Sigma: [...,3,3]
    return: [...], delta^T Sigma^-1 delta
    """
    L = _safe_cholesky(Sigma, jitter=jitter)  # [...,3,3]
    y = torch.linalg.solve_triangular(L, delta.unsqueeze(-1), upper=False)  # [...,3,1]
    return (y.squeeze(-1) ** 2).sum(dim=-1)


def _project_inside_ellipsoid(mu_child, mu_parent, Sig_parent, tau2: float = 9.0, jitter: float = 1e-6):
    """
    Ensure (mu_child - mu_parent)^T Sig_parent^-1 (mu_child - mu_parent) <= tau2
    by radial scaling in parent metric (one-shot projection).
    """
    d = mu_child - mu_parent
    d2 = _mahalanobis2(d, Sig_parent, jitter=jitter)  # [...]
    # if d2 <= tau2 -> keep; else scale down by sqrt(tau2/d2)
    scale = torch.sqrt((tau2 / d2.clamp_min(1e-12))).clamp_max(1.0)
    return mu_parent + d * scale.unsqueeze(-1)


def _allocate_counts(pi: torch.Tensor, mask_parent: torch.Tensor, N: int,
                     min_per_parent: int = 1, max_per_parent: int = None):
    """
    pi: [B,K] normalized over valid parents
    mask_parent: [B,K] {0,1}
    return m: [B,K] int counts, sum_k m = N for each batch b
    """
    B, K = pi.shape
    device = pi.device

    valid = (mask_parent > 0.5)
    m = torch.zeros((B, K), device=device, dtype=torch.long)

    for b in range(B):
        idx = torch.nonzero(valid[b], as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue

        # initial: min_per_parent
        m[b, idx] = min_per_parent
        rem = N - int(min_per_parent * idx.numel())
        if rem < 0:
            # too many parents for N: fall back to 1-hot-ish
            m[b].zero_()
            # keep first N valid
            m[b, idx[:N]] = 1
            continue

        # distribute remaining by pi
        pb = pi[b, idx]
        pb = pb / pb.sum().clamp_min(1e-9)
        add = torch.floor(pb * rem).to(torch.long)
        m[b, idx] += add
        rem2 = rem - int(add.sum().item())

        if rem2 > 0:
            # assign leftover by largest fractional parts
            frac = (pb * rem) - add.to(pb.dtype)
            _, order = torch.sort(frac, descending=True)
            m[b, idx[order[:rem2]]] += 1

        if max_per_parent is not None:
            m[b, idx] = torch.clamp(m[b, idx], max=int(max_per_parent))
            # re-balance if clamped changed total
            total = int(m[b, idx].sum().item())
            if total != N:
                # adjust by adding/removing one-by-one (cheap; K small)
                diff = N - total
                if diff > 0:
                    # add to largest pi
                    _, order = torch.sort(pb, descending=True)
                    for t in range(diff):
                        m[b, idx[order[t % order.numel()]]] += 1
                else:
                    # remove from smallest pi but keep >=1
                    _, order = torch.sort(pb, descending=False)
                    t = 0
                    while diff < 0 and t < 10_000:
                        ksel = idx[order[t % order.numel()]]
                        if m[b, ksel] > min_per_parent:
                            m[b, ksel] -= 1
                            diff += 1
                        t += 1

    return m


def _sobol_normal(n: int, device, dtype, scramble: bool = True):
    """
    Generate n samples ~ N(0,I) in R^3 using Sobol + inverse-CDF.
    """
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=scramble)
    u = engine.draw(n).to(device=device, dtype=dtype).clamp(1e-6, 1 - 1e-6)
    # inverse CDF of standard normal
    z = torch.erfinv(2 * u - 1) * math.sqrt(2.0)
    return z  # [n,3]


def _one_step_repulsion(mu: torch.Tensor, parent_idx: torch.Tensor, eta: float = 0.02, eps: float = 1e-4):
    """
    mu: [B,N,3], parent_idx: [B,N] (int)
    One shot repulsion within same parent to reduce collapse.
    """
    B, N, _ = mu.shape
    mu2 = mu.clone()
    for b in range(B):
        pid = parent_idx[b]
        for k in pid.unique():
            sel = torch.nonzero(pid == k, as_tuple=False).squeeze(-1)
            if sel.numel() <= 1:
                continue
            x = mu2[b, sel]  # [m,3]
            diff = x[:, None, :] - x[None, :, :]  # [m,m,3]
            dist2 = (diff ** 2).sum(dim=-1) + eps
            # zero self
            dist2.fill_diagonal_(1e9)
            force = (diff / dist2[..., None]).sum(dim=1)  # [m,3]
            mu2[b, sel] = x + eta * force
    return mu2


# -----------------------------
# core: cover upsample init
# -----------------------------
@torch.no_grad()
def cover_upsample_init(
    mu_p: torch.Tensor,        # [B,K,3]
    Sig_p: torch.Tensor,       # [B,K,3,3]
    pi: torch.Tensor,          # [B,K] (normalized over valid parents)
    mask_parent: torch.Tensor, # [B,K]
    node_mask: torch.Tensor,   # [B,N]
    jitter: float = 1e-6,
    tau2_inside: float = 9.0,  # how far (Mahalanobis^2) child centers can go in parent
    min_per_parent: int = 1,
    sigma_floor: float = 0.03,
    sigma_ceil: float = 2.0,
    mix_cover_alpha: float = 0.35,   # blend split-sigma with spacing-sigma
    k_nn_spacing: int = 4,           # used only for spacing-sigma
    repulse_eta: float = 0.02,       # 0 to disable
):
    """
    Returns:
      mu0: [B,N,3]
      Sig0: [B,N,3,3]
      parent_idx: [B,N]  (int in [0,K))
      m_counts: [B,K] int
    """
    B, K, _ = mu_p.shape
    N = node_mask.shape[1]
    device, dtype = mu_p.device, mu_p.dtype
    I = torch.eye(3, device=device, dtype=dtype)[None, None]

    # (1) allocate m_k per parent (hard guarantee coverage across parents)
    m = _allocate_counts(pi, mask_parent, N, min_per_parent=min_per_parent)  # [B,K] long

    # (2) generate mu0 per parent using Sobol-normal mapped by parent chol
    mu0 = torch.zeros((B, N, 3), device=device, dtype=dtype)
    Sig0_split = torch.zeros((B, N, 3, 3), device=device, dtype=dtype)
    parent_idx = torch.zeros((B, N), device=device, dtype=torch.long)

    for b in range(B):
        cursor = 0
        for k in range(K):
            if mask_parent[b, k] < 0.5:
                continue
            mk = int(m[b, k].item())
            if mk <= 0:
                continue

            # low-discrepancy samples in R^3
            z = _sobol_normal(mk, device=device, dtype=dtype, scramble=True)  # [mk,3]

            # map into ellipsoid by parent covariance
            Lk = _safe_cholesky(Sig_p[b, k], jitter=jitter)  # [3,3]
            x = mu_p[b, k].unsqueeze(0) + (z @ Lk.transpose(0, 1))  # [mk,3]

            # project inside parent ellipsoid (avoid far tails)
            x = _project_inside_ellipsoid(
                x, mu_p[b, k].unsqueeze(0), Sig_p[b, k].unsqueeze(0),
                tau2=tau2_inside, jitter=jitter
            )

            # write
            mu0[b, cursor:cursor + mk] = x
            parent_idx[b, cursor:cursor + mk] = k

            # split-sigma: Sigma_child = Sigma_parent / mk^(2/3)
            # (mk=1 -> same scale)
            scale = float(max(mk, 1)) ** (2.0 / 3.0)
            Sig_child = Sig_p[b, k] / scale
            Sig0_split[b, cursor:cursor + mk] = Sig_child.unsqueeze(0).expand(mk, 3, 3)

            cursor += mk

        # if for any reason cursor != N, pad with first valid parent (rare)
        if cursor < N:
            # find a valid parent
            kk = int(torch.nonzero(mask_parent[b] > 0.5, as_tuple=False)[0].item())
            mk = N - cursor
            z = _sobol_normal(mk, device=device, dtype=dtype, scramble=True)
            Lk = _safe_cholesky(Sig_p[b, kk], jitter=jitter)
            x = mu_p[b, kk].unsqueeze(0) + (z @ Lk.transpose(0, 1))
            x = _project_inside_ellipsoid(x, mu_p[b, kk].unsqueeze(0), Sig_p[b, kk].unsqueeze(0),
                                          tau2=tau2_inside, jitter=jitter)
            mu0[b, cursor:] = x
            parent_idx[b, cursor:] = kk
            Sig0_split[b, cursor:] = (Sig_p[b, kk].unsqueeze(0).expand(mk, 3, 3))

    # apply node mask (in case padding N)
    mu0 = mu0 * node_mask[..., None]

    # (3) optional one-shot repulsion within same parent (helps avoid clumps)
    if repulse_eta is not None and repulse_eta > 0:
        mu0 = _one_step_repulsion(mu0, parent_idx, eta=float(repulse_eta))
        mu0 = mu0 * node_mask[..., None]

    # (4) spacing-based sigma (your existing cover sigma idea)
    #     You already have init_sigma_from_child_spacing(mu0, node_mask, ...).
    #     We'll re-use it by expecting you to import it. If not available, fallback to isotropic by local nn dist.
    Sig0_cover = init_sigma_from_child_spacing(
        mu0, node_mask, k_nn=int(k_nn_spacing),
        alpha=0.6,  # same meaning as your old function uses
        sigma_floor=float(sigma_floor),
        sigma_ceil=float(sigma_ceil),
    )
    Sig0_cover = Sig0_cover + jitter * I * node_mask[:, :, None, None]

    # (5) blend: split-sigma (mass/partition semantics) + cover-sigma (geometry spacing)
    a = float(mix_cover_alpha)
    Sig0 = (1.0 - a) * Sig0_split + a * Sig0_cover
    Sig0 = 0.5 * (Sig0 + Sig0.transpose(-1, -2)) + jitter * I

    # clamp diag magnitude as sanity
    diag = torch.diagonal(Sig0, dim1=-2, dim2=-1)
    diag = diag.clamp_min(float(sigma_floor) ** 2).clamp_max(float(sigma_ceil) ** 2)
    Sig0 = Sig0.clone()
    Sig0[..., 0, 0] = diag[..., 0]
    Sig0[..., 1, 1] = diag[..., 1]
    Sig0[..., 2, 2] = diag[..., 2]

    return mu0, Sig0, parent_idx, m
