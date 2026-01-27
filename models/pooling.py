import math
from dataclasses import dataclass
from typing import Optional, Tuple
from data import utils as du
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.GaussianRigid import OffsetGaussianRigid
from openfold.utils.rigid_utils import Rotation
def choose_K(N: int, ratio: float = 6.0, k_min: int = 1, k_max: Optional[int] = None) -> int:
    k = int(math.ceil(N / ratio))
    k = max(k, k_min)
    if k_max is not None:
        k = min(k, k_max)
    return k


def merge_gaussians_soft(mu: torch.Tensor, Sigma: torch.Tensor, A: torch.Tensor, eps: float = 1e-8):
    """
    Moment matching merge with soft assignment.
    mu:    [B, N, 3]
    Sigma: [B, N, 3, 3]
    A:     [B, N, K]  (rows sum to 1 on valid residues)
    return:
      mu_c:    [B, K, 3]
      Sigma_c: [B, K, 3, 3]
    """
    B, N, K = A.shape

    # normalize per-cluster
    w = A / (A.sum(dim=1, keepdim=True) + eps)         # [B, N, K]

    mu_c = torch.einsum("bnk,bnd->bkd", w, mu)         # [B, K, 3]
    intra = torch.einsum("bnk,bnij->bkij", w, Sigma)   # [B, K, 3, 3]

    diff = mu.unsqueeze(2) - mu_c.unsqueeze(1)         # [B, N, K, 3]
    outer = diff.unsqueeze(-1) * diff.unsqueeze(-2)    # [B, N, K, 3, 3]
    inter = torch.einsum("bnk,bnkij->bkij", w, outer)  # [B, K, 3, 3]

    Sigma_c = intra + inter
    return mu_c, Sigma_c


def gaussian_overlap_score(delta: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    score = -0.5 * (d^T (sigma)^-1 d)
    delta: [...,3]
    sigma: [...,3,3] (SPD-ish)
    return: [...]  <=0, closer => nearer 0 (more overlap)
    """
    I = torch.eye(3, device=sigma.device, dtype=sigma.dtype)
    inv = torch.linalg.inv(sigma + eps * I)
    dist2 = torch.einsum("...i,...ij,...j->...", delta, inv, delta)
    return -0.5 * dist2


@dataclass
class PoolLoss:
    occ: torch.Tensor
    rep: torch.Tensor
    ent: torch.Tensor
    total: torch.Tensor


class LearnOnlyGaussianPooling(nn.Module):
    """
    Learn-only pooling:
      A = softmax( (Ws) dot slot_k / tau )
      merge by moment matching
      losses: occupancy balance + repulsion (overlap) + entropy anneal

    No FPS. Works even if geometry is noisy; relies on continuous optimization + constraints.
    """

    def __init__(
        self,
        c_s: int,
        ratio: float = 6.0,            # K = ceil(N/ratio)
        k_max_cap: Optional[int] = None,
        tau_init: float = 1.0,
        slots_init_scale: float = 0.02,
    ):
        super().__init__()
        self.ratio = ratio
        self.k_max_cap = k_max_cap

        self.tau = tau_init
        self.proj = nn.Linear(c_s, c_s)
        # slot embeddings are created lazily when K is known
        self.slot_embed: Optional[nn.Parameter] = None
        self._slot_K: Optional[int] = None
        self._slots_init_scale = slots_init_scale

    def _ensure_slots(self, K: int, C: int, device, dtype):
        if (self.slot_embed is None) or (self._slot_K != K) or (self.slot_embed.shape[-1] != C):
            self._slot_K = K
            slots = torch.randn(K, C, device=device, dtype=dtype) * self._slots_init_scale
            self.slot_embed = nn.Parameter(slots)

    def forward(
        self,
        s: torch.Tensor,                 # [B,N,C]
        r,                        # OffsetGaussianRigid (curr_rigids)

        mask: Optional[torch.Tensor] = None,  # [B,N] {0,1}
        # loss weights
        w_occ: float = 1.0,
        w_rep: float = 0.1,
        w_ent: float = 0.01,
        # repulsion settings
        rep_topk: int = 4,
        rep_margin: float = -1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PoolLoss]:
        """
        Returns:
          A: [B,N,K]
          s_c: [B,K,C]
          mu_c: [B,K,3]
          Sigma_c: [B,K,3,3]
          losses: PoolLoss
        """
        B, N, C = s.shape
        device, dtype = s.device, s.dtype
        if mask is None:
            mask = torch.ones(B, N, device=device, dtype=dtype)

        # ✅ 新增：从 rigids 提取 mu/Sigma
        mu = r.get_gaussian_mean()  # [B,N,3]
        Sigma = r.get_covariance()  # [B,N,3,3]
        K = choose_K(N, ratio=self.ratio, k_min=1, k_max=self.k_max_cap)
        self._ensure_slots(K, C, device, dtype)

        # --- assignment A (only from s) ---
        s_proj = self.proj(s)  # [B,N,C]
        slots = self.slot_embed.unsqueeze(0).expand(B, -1, -1)  # [B,K,C]

        logits = torch.einsum("bnc,bkc->bnk", s_proj, slots) / max(self.tau, 1e-6)

        # mask invalid residues
        logits = logits.masked_fill((mask < 0.5).unsqueeze(-1), -1e9)
        A = F.softmax(logits, dim=-1)  # [B,N,K]
        A = A * mask.unsqueeze(-1)

        # --- pooled semantic ---
        denom = A.sum(dim=1).clamp_min(1e-8)          # [B,K]
        s_c = torch.einsum("bnk,bnc->bkc", A, s) / denom.unsqueeze(-1)

        # --- merge gaussians ---
        mu_c, Sigma_c = merge_gaussians_soft(mu, Sigma, A)

        # =========================
        # losses (three must-haves)
        # =========================

        # (1) occupancy / load balance
        # target: roughly uniform across K among valid residues
        occ = A.sum(dim=1)  # [B,K]
        occ = occ / (occ.sum(dim=-1, keepdim=True).clamp_min(1e-8))
        target = torch.full_like(occ, 1.0 / K)
        loss_occ = F.mse_loss(occ, target)

        # (2) repulsion: penalize high overlap between coarse gaussians
        if K <= 1:
            loss_rep = torch.zeros((), device=device, dtype=dtype)
        else:
            delta = mu_c.unsqueeze(2) - mu_c.unsqueeze(1)            # [B,K,K,3]
            sigma_sum = Sigma_c.unsqueeze(2) + Sigma_c.unsqueeze(1)  # [B,K,K,3,3]
            score = gaussian_overlap_score(delta, sigma_sum)         # [B,K,K] <=0
            # remove diagonal
            eye = torch.eye(K, device=device, dtype=torch.bool).unsqueeze(0)
            score = score.masked_fill(eye, -1e9)

            # take top overlaps per k (closest-to-0 scores)
            top = torch.topk(score, k=min(rep_topk, K - 1), dim=-1).values  # [B,K,topk]
            # want them more negative than margin => penalize if top > margin
            loss_rep = F.relu(top - rep_margin).mean()

        # (3) entropy (encourage softness early; then you anneal w_ent/tau)
        # H(A_i)= -sum_k A_ik log A_ik ; we minimize negative entropy? you choose.
        # Here: encourage lower entropy gradually by MINIMIZING entropy (later set w_ent>0).
        ent = -(A.clamp_min(1e-8) * A.clamp_min(1e-8).log()).sum(dim=-1)  # [B,N]
        ent = (ent * mask).sum() / mask.sum().clamp_min(1.0)
        loss_ent = ent

        total = w_occ * loss_occ + w_rep * loss_rep + w_ent * loss_ent
        losses = PoolLoss(occ=loss_occ, rep=loss_rep, ent=loss_ent, total=total)

        return A, s_c, mu_c, Sigma_c, losses

import torch

def _make_so3(R: torch.Tensor) -> torch.Tensor:
    """
    Force det(R)=+1 by flipping the 3rd column if needed.
    R: [...,3,3]
    """
    det = torch.linalg.det(R)  # [...]
    flip = (det < 0).to(R.dtype)  # [...]
    # flip last column
    R = R.clone()
    R[..., :, 2] = R[..., :, 2] * (1.0 - 2.0 * flip).unsqueeze(-1)
    return R

def _sign_fix_with_prev(R: torch.Tensor, prev_R: torch.Tensor) -> torch.Tensor:
    """
    Reduce eigenvector sign ambiguity by aligning columns with prev_R.
    R, prev_R: [...,3,3]
    """
    # column-wise dot
    dots = (R * prev_R).sum(dim=-2)  # [...,3]  (sum over rows)
    sgn = torch.where(dots >= 0, torch.ones_like(dots), -torch.ones_like(dots))  # [...,3]
    return R * sgn.unsqueeze(-2)

def coarse_rigids_from_mu_sigma(mu_c, Sigma_c, OffsetGaussianRigid_cls, eps=1e-6):
    """
    mu_c:    [B, K, 3]
    Sigma_c: [B, K, 3, 3]
    return:  OffsetGaussianRigid [B, K]
    """
    B, K, _ = mu_c.shape
    device, dtype = mu_c.device, mu_c.dtype

    # 1) R = Identity
    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(B, K, 3, 3)
    rots = du.Rotation(rot_mats=eye) if hasattr(du, "Rotation") else Rotation(rot_mats=eye)  # 兼容你的 Rotation 定义

    # 2) scaling = sqrt(diag(Sigma))
    diag = torch.diagonal(Sigma_c, dim1=-2, dim2=-1)  # [B,K,3]
    scaling = torch.sqrt(torch.clamp(diag, min=eps))
    scaling_log = torch.log(scaling + eps)

    # 3) local_mean = 0 (因为 trans 就是 mu)
    local_mean = torch.zeros_like(mu_c)

    return OffsetGaussianRigid_cls(
        rots=rots,
        trans=mu_c,
        scaling_log=scaling_log,
        local_mean=local_mean,
    )
