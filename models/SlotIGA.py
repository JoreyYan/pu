import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.IGA import fused_gaussian_overlap_score

# ----------------------------
# Loss container (align with your PoolLoss style)
# ----------------------------

@dataclass
class PoolLoss:
    # gating/codebook usage
    usage_kl: torch.Tensor
    usage_ent: torch.Tensor
    # in-sample occupancy balance
    occ_mse: torch.Tensor
    # anti-collapse
    collapse: torch.Tensor
    total: torch.Tensor



# ----------------------------
# Small utils
# ----------------------------
def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-8):
    m = mask
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    num = (x * m).sum(dim=dim)
    den = m.sum(dim=dim).clamp_min(eps)
    return num / den
def masked_Sigavg(Sigma: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    w = mask[:, :, None, None]
    Sig = (Sigma * w).sum(dim=1) / w.sum(dim=1).clamp_min(eps)
    return 0.5 * (Sig + Sig.transpose(-1, -2))


def masked_std_mu(mu: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    # 1. 计算均值，保持维度以便广播
    # mean: [B, 3], count: [B, 1]
    count = mask.sum(dim=1, keepdim=True).clamp_min(eps)
    mean = (mu * mask.unsqueeze(-1)).sum(dim=1) / count  # [B, 3]

    # 2. 去中心化
    dm = mu - mean.unsqueeze(1)  # [B, N, 3]
    w = mask.unsqueeze(-1)  # [B, N, 1]

    # 3. 计算协方差 [B, 3, 3]
    # 关键点：count 需要变成 [B, 1, 1] 才能正确除以 [B, 3, 3]
    cov = torch.einsum("bni,bnj->bij", dm * w, dm) / count.unsqueeze(-1)

    # 4. 对称化处理
    cov = 0.5 * (cov + cov.transpose(-1, -2))

    # 5. 计算标准差 [B, 3]
    std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(eps))

    return mean, std
def masked_cov3(mu: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """
    mu: [B,N,3], mask: [B,N]
    returns mean [B,3], cov [B,3,3], std [B,3]
    """
    mean = masked_mean(mu, mask, dim=1, eps=eps)  # [B,3]
    dm = mu - mean[:, None, :]
    w = mask[:, :, None]  # [B,N,1]
    cov = torch.einsum("bni,bnj->bij", dm * w, dm) / w.sum(dim=1).clamp_min(eps)
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    diag = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(eps)
    std = torch.sqrt(diag)
    return mean, cov, std


def merge_gaussians_soft(
    mu: torch.Tensor,          # [B,N,3]
    Sigma: torch.Tensor,       # [B,N,3,3]
    A: torch.Tensor,           # [B,N,K]
    mask: Optional[torch.Tensor] = None,  # [B,N]
    eps: float = 1e-8,
    jitter: float = 1e-6,
):
    """
    Moment matching with membership weights A (soft assignment).
    Returns:
      mu_k: [B,K,3]
      Sig_k:[B,K,3,3]
      occ:  [B,K]
    """
    B, N, _ = mu.shape
    K = A.shape[-1]
    device, dtype = mu.device, mu.dtype
    I = torch.eye(3, device=device, dtype=dtype)[None, None]

    if mask is not None:
        A = A * mask.unsqueeze(-1)

    occ = A.sum(dim=1).clamp_min(eps)  # [B,K]

    mu_k = torch.einsum("bnk,bni->bki", A, mu) / occ.unsqueeze(-1)

    d = mu[:, :, None, :] - mu_k[:, None, :, :]  # [B,N,K,3]
    outer = d.unsqueeze(-1) * d.unsqueeze(-2)    # [B,N,K,3,3]

    Sig_k = (
        torch.einsum("bnk,bnij->bkij", A, Sigma) +
        torch.einsum("bnk,bnkij->bkij", A, outer)
    ) / occ.view(B, K, 1, 1)

    Sig_k = 0.5 * (Sig_k + Sig_k.transpose(-1, -2)) + jitter * I
    return mu_k, Sig_k, occ



def collapse_loss(A: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """
    A: [B,N,K]
    penalize similarity among columns
    """
    B, N, K = A.shape
    A = A * mask.unsqueeze(-1)
    X = A.transpose(1, 2)  # [B,K,N]
    X = X / X.norm(dim=-1, keepdim=True).clamp_min(eps)
    G = torch.einsum("bkn,bln->bkl", X, X)  # [B,K,K]
    eye = torch.eye(K, device=A.device, dtype=A.dtype).unsqueeze(0)
    off = (G - eye) * (1.0 - eye)
    return (off ** 2).mean()


def codebook_usage_loss(idx: torch.Tensor, k_max: int, eps: float = 1e-8):
    """
    idx: [B,K_use] 选槽结果
    returns:
      KL(p || u) and (-entropy(p)) for monitoring
    """
    B, K_use = idx.shape
    device = idx.device

    # histogram over codebook entries
    flat = idx.reshape(-1)
    hist = torch.zeros(k_max, device=device, dtype=torch.float32)
    ones = torch.ones_like(flat, dtype=torch.float32)
    hist.scatter_add_(0, flat, ones)
    p = hist / hist.sum().clamp_min(1.0)  # [k_max]

    u = torch.full_like(p, 1.0 / k_max)

    # KL(p||u) = sum p log(p/u)
    kl = (p.clamp_min(eps) * (p.clamp_min(eps).log() - u.log())).sum()

    # encourage high entropy => minimize (-H)
    ent = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum()
    neg_ent = -ent
    return kl, neg_ent, p


# ----------------------------
# NK fused overlap (stable): returns [B,N,K]
# ----------------------------
def fused_overlap_nk_mahalanobis(
    delta: torch.Tensor,     # [B,N,K,3]
    Sigma: torch.Tensor,     # [B,N,K,3,3]  (SPD-ish)
    eps: float = 1e-6,
    with_logdet: bool = False,
):
    """
    A stable "overlap-like" score that matches your use:
      score = -0.5 * (delta^T Sigma^{-1} delta)  (+ optional -0.5*logdet Sigma)
    Output is typically <= 0, near 0 when close, very negative when far.

    This is the NK version you asked for.
    """
    B, N, K, _ = delta.shape
    device, dtype = delta.device, delta.dtype
    I = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3)

    S = 0.5 * (Sigma + Sigma.transpose(-1, -2)) + eps * I  # ensure SPD-ish
    # Cholesky: S = L L^T
    L = torch.linalg.cholesky(S)  # [B,N,K,3,3]

    # Solve L y = delta  -> y
    y = torch.linalg.solve_triangular(L, delta.unsqueeze(-1), upper=False)  # [B,N,K,3,1]
    maha2 = (y.squeeze(-1) ** 2).sum(dim=-1)  # [B,N,K]

    score = -0.5 * maha2

    if with_logdet:
        # logdet(S) = 2*sum(log(diag(L)))
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-12)).sum(dim=-1)  # [B,N,K]
        score = score - 0.5 * logdet

    return score


# ----------------------------
# Main: IGA-style Slot Pooling (N -> K)
# ----------------------------
class IGASlotPoolingV1(nn.Module):
    def __init__(
        self,
        c_s: int,
        k_max: int = 1024,
        ratio: float = 12.0,
        iters: int = 3,
        geo_heads: int = 4,

        tau_init: float = 1.0,
        w_geo_init: float = 1.0,

        sigma_floor: float = 0.03,
        init_sigma_mix: float = 0.1,
        init_offset_scale: float = 0.5,

        rand_slots: int = 0,
        gate_noise: float = 0.02,
        jitter: float = 1e-6,

        # ----- losses weights -----
        w_usage_kl: float = 0.05,      # codebook usage 均匀化（batch级）
        w_usage_ent: float = 0.00,     # 可选（通常 KL 就够）
        w_occ: float = 0.5,            # in-sample occupancy 均匀化
        w_collapse: float = 0.1,       # anti-collapse
    ):
        super().__init__()
        self.c_s = c_s
        self.k_max = int(k_max)
        self.ratio = float(ratio)
        self.iters = int(iters)
        self.tau = float(tau_init)

        self.w_geo = nn.Parameter(torch.tensor(float(w_geo_init)))

        self.sigma_floor = float(sigma_floor)
        self.init_sigma_mix = float(init_sigma_mix)
        self.init_offset_scale = float(init_offset_scale)
        self.jitter = float(jitter)

        self.rand_slots = int(rand_slots)
        self.gate_noise = float(gate_noise)

        # loss weights
        self.w_usage_kl = float(w_usage_kl)
        self.w_usage_ent = float(w_usage_ent)
        self.w_occ = float(w_occ)
        self.w_collapse = float(w_collapse)

        # bank
        self.slot_embed_pool = nn.Parameter(torch.randn(self.k_max, c_s) * 0.02)
        self.geo_offsets_pool = nn.Parameter(torch.randn(self.k_max, 3) * 0.5)

        # cosine gating query projection
        self.gating_proj = nn.Linear(c_s, c_s, bias=False)

        # refine projections
        self.proj_q = nn.Linear(c_s, c_s, bias=False)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)

        self.gru = nn.GRUCell(c_s, c_s)
        self.mlp = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, 4 * c_s),
            nn.GELU(),
            nn.Linear(4 * c_s, c_s),
        )

        H = int(geo_heads)
        self.geo_heads = H
        self.geo_scale_raw = nn.Parameter(torch.full((H,), -4.0))
        self.geo_bias = nn.Parameter(torch.full((H,), -6.0))

    @staticmethod
    def _choose_K(N: int, ratio: float):
        return max(1, int(math.ceil(N / max(ratio, 1e-6))))

    def _select_slots(self, s: torch.Tensor, mask: torch.Tensor, K_use: int):
        """
        cosine/attention-style gating:
          scores = normalize(W s_global) @ normalize(codebook)^T
        """
        B, N, C = s.shape
        K_use = min(K_use, self.k_max)

        s_global = masked_mean(s, mask, dim=1)                  # [B,C]
        q = F.normalize(self.gating_proj(s_global), dim=-1)      # [B,C]
        k = F.normalize(self.slot_embed_pool, dim=-1)            # [Kmax,C]
        scores = q @ k.t()                                       # [B,Kmax]

        if self.training and self.gate_noise > 0:
            scores = scores + torch.randn_like(scores) * self.gate_noise

        K_rand = min(self.rand_slots, max(0, K_use - 1))
        K_top = K_use - K_rand

        top_idx = torch.topk(scores, K_top, dim=-1).indices      # [B,K_top]
        if K_rand > 0:
            rand_idx = torch.randint(0, self.k_max, (B, K_rand), device=s.device)
            idx = torch.cat([top_idx, rand_idx], dim=-1)
        else:
            idx = top_idx
        return idx  # [B,K_use]

    def forward(
        self,
        s: torch.Tensor,              # [B,N,C]
        mu: torch.Tensor,             # [B,N,3]
        Sigma: torch.Tensor,          # [B,N,3,3]
        mask: torch.Tensor,           # [B,N] (0/1)

    ):
        B, N, C = s.shape
        device, dtype = s.device, s.dtype
        mask = mask.to(device=device, dtype=dtype)

        K_use = self._choose_K(N, self.ratio)
        K_use = min(K_use, self.k_max)

        # ---- Stage A: select slots ----
        idx = self._select_slots(s, mask, K_use)                # [B,K]
        slots = self.slot_embed_pool[idx]                       # [B,K,C]
        geo_offsets = self.geo_offsets_pool[idx]                # [B,K,3]

        # ---- init geo ----



        mean,  std = masked_std_mu(mu, mask)
        std = std.clamp_min(self.sigma_floor)
        mu_k = mean[:, None, :] + geo_offsets * (self.init_offset_scale * std[:, None, :])

        Sig_avg = masked_Sigavg(Sigma, mask)
        I = torch.eye(3, device=device, dtype=dtype)[None, None]
        Sig_k = Sig_avg[:, None, :, :].expand(B, K_use, 3, 3).contiguous() * self.init_sigma_mix
        Sig_k = 0.5 * (Sig_k + Sig_k.transpose(-1, -2))
        Sig_k = Sig_k + (self.sigma_floor ** 2) * I + self.jitter * I

        # ---- refine loop ----
        k_sem = self.proj_k(s)
        A = None
        for _ in range(self.iters):
            q_sem = self.proj_q(slots)
            logits_sem = torch.einsum("bnc,bkc->bnk", k_sem, q_sem) / math.sqrt(C)

            delta = mu[:, :, None, :] - mu_k[:, None, :, :]
            sigma_sum = Sigma[:, :, None, :, :] + Sig_k[:, None, :, :, :]

            G = fused_gaussian_overlap_score(delta, sigma_sum)      # [B,N,K] (<=0)

            a = F.softplus(self.geo_scale_raw).view(1, self.geo_heads, 1, 1)
            b = self.geo_bias.view(1, self.geo_heads, 1, 1)
            geo_bias = (a * G.unsqueeze(1) + b).sum(dim=1)          # [B,N,K]

            logits = (logits_sem + self.w_geo * geo_bias) / max(float(self.tau), 1e-6)
            logits = logits.masked_fill((mask < 0.5).unsqueeze(-1), -1e9)

            A = torch.softmax(logits, dim=-1) * mask.unsqueeze(-1)  # [B,N,K]

            mu_k, Sig_k, _occ = merge_gaussians_soft(mu, Sigma, A, mask=mask, jitter=max(self.jitter, 1e-6))

            diag = torch.diagonal(Sig_k, dim1=-2, dim2=-1).clamp_min(self.sigma_floor ** 2)
            Sig_k = Sig_k - torch.diag_embed(torch.diagonal(Sig_k, dim1=-2, dim2=-1)) + torch.diag_embed(diag)
            Sig_k = 0.5 * (Sig_k + Sig_k.transpose(-1, -2)) + self.jitter * I

            denom = A.sum(dim=1).clamp_min(1e-8)
            slot_in = torch.einsum("bnk,bnc->bkc", A, s) / denom.unsqueeze(-1)
            slots = self.gru(slot_in.reshape(-1, C), slots.reshape(-1, C)).view(B, K_use, C)
            slots = slots + self.mlp(slots)

        # ---- outputs ----
        s_c, mu_c, Sigma_c = slots, mu_k, Sig_k

        # =====================================================
        # losses: usage diversity + in-sample balance + anti-collapse
        # =====================================================
        # (1) codebook usage uniform (batch-level, based on idx)
        usage_kl, usage_neg_ent, p_usage = codebook_usage_loss(idx, self.k_max)

        # (2) in-sample occupancy balance (based on A column sums)
        occ = (A * mask.unsqueeze(-1)).sum(dim=1)                 # [B,K]
        occ = occ / occ.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        target = torch.full_like(occ, 1.0 / max(K_use, 1))
        occ_mse = F.mse_loss(occ, target)

        # (3) anti-collapse
        col = collapse_loss(A, mask)

        total = (
            self.w_usage_kl * usage_kl +
            self.w_usage_ent * usage_neg_ent +
            self.w_occ * occ_mse +
            self.w_collapse * col
        )

        losses = PoolLoss(
            usage_kl=usage_kl,
            usage_ent=usage_neg_ent,
            occ_mse=occ_mse,
            collapse=col,
            total=total,
        )

        return A, s_c, mu_c, Sigma_c, idx, losses
