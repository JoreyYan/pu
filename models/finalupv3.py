import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 你工程里已有的：
# CoarseEdgeCoarsenAndFuse, InvariantGaussianAttention, GaussianUpdateBlock, CoarseIGATower
# init_sigma_from_child_spacing, coarse_rigids_from_mu_sigma, _sym, UpAux

class FinalCoarseToFineDensenSampleIGAModulev3(nn.Module):
    """
    v3 Up-sampling (ordered / identity-safe):
      - Router: w = Attn( Q(s_trunk_detach), K(s_parent, mu_p) )
      - Ordered init:
          s0 = w @ V(s_parent)
          mu0 = w @ mu_p (+ optional delta from parent features)
          Sig0 = beta*(w @ Sig_p) + (1-beta)*Sig_spacing(mu0)
      - Edge fuse with A=w
      - IGA refinement tower (unchanged)
    """

    def __init__(
        self,
        c_s: int,
        iga_conf,
        OffsetGaussianRigid_cls,
        num_refine_layers: int = 4,

        # ---- spacing sigma knobs (keep from v2) ----
        fps_knn: int = 8,
        sigma_cover_alpha: float = 1.5,
        sigma_floor: float = 1e-3,
        sigma_ceil: float = 1e2,
        jitter: float = 1e-6,

        # ---- parent-child sigma blend ----
        beta_parent_sigma: float = 0.85,
        parent_sigma_shrink: float = 1.0,

        # ---- router knobs ----
        use_ln: bool = True,
        tau_init: float = 1.0,          # softmax temperature
        w_clip_min: float = 0.0,        # optional floor (usually 0)

        # ---- optional regularizers ----
        lambda_route_smooth: float = 0.0,    # KL(w_i||w_{i+1}) weight
        lambda_route_collapse: float = 0.0,  # column-corr / coverage weight

        # ---- optional mu offset (helps avoid all kids at parent center) ----
        enable_mu_offset: bool = True,
        mu_offset_scale: float = 0.25,       # small init scale
    ):
        super().__init__()
        self.c_s = c_s
        self.OffsetGaussianRigid_cls = OffsetGaussianRigid_cls

        # sigma / spacing
        self.fps_knn = int(fps_knn)
        self.sigma_cover_alpha = float(sigma_cover_alpha)
        self.sigma_floor = float(sigma_floor)
        self.sigma_ceil = float(sigma_ceil)
        self.jitter = float(jitter)

        self.beta_parent_sigma = float(beta_parent_sigma)
        self.parent_sigma_shrink = float(parent_sigma_shrink)

        # router
        self.tau_init = float(tau_init)
        self.w_clip_min = float(w_clip_min)
        self.lambda_route_smooth = float(lambda_route_smooth)
        self.lambda_route_collapse = float(lambda_route_collapse)

        self.use_ln = bool(use_ln)
        if self.use_ln:
            self.ln_q = nn.LayerNorm(c_s)
            self.ln_k = nn.LayerNorm(c_s)
            self.ln_v = nn.LayerNorm(c_s)

        # Q from s_trunk, K/V from s_parent, plus position bias from mu_p
        self.q_proj = nn.Linear(c_s, c_s, bias=False)
        self.k_proj = nn.Linear(c_s, c_s, bias=False)
        self.v_proj = nn.Linear(c_s, c_s, bias=False)
        self.pos_proj = nn.Linear(3, c_s, bias=False)

        # optional mu offset head (from aggregated parent feature)
        self.enable_mu_offset = bool(enable_mu_offset)
        if self.enable_mu_offset:
            self.mu_offset = nn.Sequential(
                nn.LayerNorm(c_s),
                nn.Linear(c_s, c_s),
                nn.SiLU(),
                nn.Linear(c_s, 3),
            )
            # small init helps stability
            with torch.no_grad():
                self.mu_offset[-1].weight.mul_(self.mu_offset_scale if hasattr(self, "mu_offset_scale") else 0.0)
            self.mu_offset_scale = float(mu_offset_scale)

        # Edge fuser (UP mode)
        self.edge_fusers = CoarseEdgeCoarsenAndFuse(
            c_z_in=getattr(iga_conf, "hgfc_z", 0),
            c_z_out=getattr(iga_conf, "hgfc_z", 0),
            mode="up",
        )

        # IGA refinement tower (same as v2)
        iga = InvariantGaussianAttention(
            c_s=c_s,
            c_z=getattr(iga_conf, "hgfc_z", 0),
            c_hidden=iga_conf.c_hidden,
            no_heads=iga_conf.no_heads,
            no_qk_gaussians=iga_conf.no_qk_points,
            no_v_points=iga_conf.no_v_points,
            layer_idx=9001,
            enable_vis=False,
        )
        gau_update = GaussianUpdateBlock(c_s)
        self.refine_tower = CoarseIGATower(
            iga=iga,
            gau_update=gau_update,
            c_s=c_s,
            hgfc_z=iga_conf.hgfc_z,
            num_layers=num_refine_layers,
        )

    @staticmethod
    def _route_smooth_kl(w: torch.Tensor, node_mask: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """
        KL(w_i || w_{i+1}) along sequence. w: [B,N,K]
        """
        # valid pairs
        m0 = node_mask[:, :-1].float()
        m1 = node_mask[:, 1:].float()
        mp = (m0 * m1).unsqueeze(-1)  # [B,N-1,1]

        p = w[:, :-1].clamp_min(eps)
        q = w[:, 1:].clamp_min(eps)
        kl = (p * (p.log() - q.log())).sum(dim=-1)  # [B,N-1]
        kl = kl * (m0 * m1)
        denom = (m0 * m1).sum().clamp_min(1.0)
        return kl.sum() / denom

    @staticmethod
    def _route_collapse_corr(w: torch.Tensor, node_mask: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """
        Simple anti-collapse: column correlation of assignment matrix.
        Encourage columns to be less correlated.
        w: [B,N,K]
        """
        B, N, K = w.shape
        m = node_mask.float().unsqueeze(-1)  # [B,N,1]
        x = w * m
        # center over N
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = x.sum(dim=1, keepdim=True) / denom
        xc = x - mean
        # cov: [B,K,K]
        cov = torch.einsum("bnk,bnj->bkj", xc, xc) / denom.squeeze(1).clamp_min(1.0).unsqueeze(-1)
        # normalize to corr
        var = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(eps)  # [B,K]
        inv_std = var.rsqrt()
        corr = cov * inv_std[:, :, None] * inv_std[:, None, :]
        # penalize off-diagonal corr^2
        eye = torch.eye(K, device=w.device, dtype=w.dtype)[None]
        off = (corr * (1 - eye)) ** 2
        return off.mean()

    def forward(
        self,
        s_parent: torch.Tensor,                 # [B,K,C]
        z_parent,                               # None or tensor (edge)
        r_parent,                               # OffsetGaussianRigid [B,K]
        mask_parent: torch.Tensor,              # [B,K]
        node_mask: torch.Tensor,                # [B,N]
        s_trunk: torch.Tensor,                  # [B,N,C]  <<<< V3 新增
        occ_parent: Optional[torch.Tensor] = None,  # [B,K] optional
        res_idx: Optional[torch.Tensor] = None,     # optional, V3 不强依赖
        tau: Optional[float] = None,                # optional override
    ):
        B, K, C = s_parent.shape
        N = node_mask.shape[1]
        device, dtype = s_parent.device, s_parent.dtype

        mu_p = r_parent.get_gaussian_mean()      # [B,K,3]
        Sig_p = r_parent.get_covariance()        # [B,K,3,3]
        Sig_p = _sym(Sig_p)  # safety

        # ---- pi from occ (optional, used as mild bias/normalizer if you want later) ----
        if occ_parent is None:
            pi = torch.ones((B, K), device=device, dtype=dtype)
        else:
            pi = occ_parent.to(dtype=dtype)
        pi = pi * (mask_parent > 0.5).to(dtype)
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # ==========================================================
        # 1) Router: compute w [B,N,K] (ORDERED by residue index)
        # ==========================================================
        s_trunk_detached = s_trunk.detach()

        if self.use_ln:
            q_in = self.ln_q(s_trunk_detached)
            k_in = self.ln_k(s_parent)
            v_in = self.ln_v(s_parent)
        else:
            q_in, k_in, v_in = s_trunk_detached, s_parent, s_parent

        q = self.q_proj(q_in)                                   # [B,N,C]
        k = self.k_proj(k_in) + self.pos_proj(mu_p)             # [B,K,C]
        v = self.v_proj(v_in)                                   # [B,K,C]

        logits = torch.einsum("bnc,bkc->bnk", q, k)              # [B,N,K]
        temp = float(self.tau_init if tau is None else tau)
        logits = logits / max(temp, 1e-6)

        # mask parents + residues
        logits = logits.masked_fill(mask_parent[:, None, :] <= 0.5, -1e9)
        logits = logits.masked_fill(node_mask[:, :, None] <= 0.5, -1e9)

        w = F.softmax(logits, dim=-1)                            # [B,N,K]
        if self.w_clip_min > 0:
            w = w.clamp_min(self.w_clip_min)
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # ensure padded rows are exactly 0
        w = w * node_mask[:, :, None].to(dtype)

        # ==========================================================
        # 2) Ordered init: s0, mu0, Sig_parent_child
        # ==========================================================
        s0 = torch.einsum("bnk,bkc->bnc", w, v)                  # [B,N,C]
        mu0 = torch.einsum("bnk,bkd->bnd", w, mu_p)              # [B,N,3]

        # optional per-residue offset from aggregated parent feature (NOT from q)
        if self.enable_mu_offset:
            parent_feat = torch.einsum("bnk,bkc->bnc", w, s_parent)  # [B,N,C]
            delta = self.mu_offset(parent_feat) * self.mu_offset_scale
            mu0 = mu0 + delta

        mu0 = mu0 * node_mask[..., None].to(dtype)

        # weighted parent Sigma (soft inherit)
        Sig_parent_child = torch.einsum("bnk,bkij->bnij", w, Sig_p * self.parent_sigma_shrink)
        I = torch.eye(3, device=device, dtype=dtype)[None, None]
        Sig_parent_child = _sym(Sig_parent_child) + max(self.jitter, 1e-6) * I

        # ==========================================================
        # 3) Spacing Sigma + blend (keep v2 spirit)
        # ==========================================================
        Sig_spacing = init_sigma_from_child_spacing(
            mu0, node_mask,
            k_nn=self.fps_knn,
            alpha=self.sigma_cover_alpha,
            sigma_floor=self.sigma_floor,
            sigma_ceil=self.sigma_ceil,
            jitter=max(self.jitter, 1e-6),
        )  # [B,N,3,3]

        beta = self.beta_parent_sigma
        Sig0 = beta * Sig_parent_child + (1.0 - beta) * Sig_spacing
        Sig0 = _sym(Sig0) + max(self.jitter, 1e-6) * I
        Sig0 = Sig0 * node_mask[:, :, None, None].to(dtype)

        # ==========================================================
        # 4) Build residue rigids
        # ==========================================================
        r0 = coarse_rigids_from_mu_sigma(mu0, Sig0, self.OffsetGaussianRigid_cls)

        # ==========================================================
        # 5) Edge fuse using A=w (huge win vs v2 back-infer)
        # ==========================================================
        if self.edge_fusers is not None and (z_parent is not None):
            z_c, Z_sem_c, Z_geo_c = self.edge_fusers(
                A=w, Z_in=z_parent, r_target=r0,
                mask_f=node_mask, mask_c=mask_parent
            )
        else:
            z_c = None

        # ==========================================================
        # 6) Refine (unchanged)
        # ==========================================================
        s1, r1, z1 = self.refine_tower(s0, r0, node_mask, z_c)

        # ==========================================================
        # 7) Regularizers (optional)
        # ==========================================================
        reg_total = torch.tensor(0.0, device=device, dtype=dtype)
        if self.lambda_route_smooth > 0:
            reg_total = reg_total + self.lambda_route_smooth * self._route_smooth_kl(w, node_mask)
        if self.lambda_route_collapse > 0:
            reg_total = reg_total + self.lambda_route_collapse * self._route_collapse_corr(w, node_mask)

        # parent_idx: soft router doesn't have hard parent_idx; store argmax for logging/debug
        parent_idx = torch.argmax(w, dim=-1)  # [B,N]

        levels = [{
            "s0": s0,
            "mu0": mu0,
            "Sigma0": Sig0,
            "r0": r0,
            "s": s1,
            "z": z1,
            "r": r1,
            "mask": node_mask,
            "aux": UpAux(w_parent=w, pi=pi, parent_idx=parent_idx)
        }]

        return levels, reg_total
