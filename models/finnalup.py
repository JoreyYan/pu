
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math

import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
# 你已有：
# - InvariantGaussianAttention (from models.IGA import InvariantGaussianAttention)
# - StructureModuleTransition
# - GaussianUpdateBlock
# - coarse_rigids_from_mu_sigma(mu, Sigma, OffsetGaussianRigid_cls)
# - fused_gaussian_overlap_score(delta, sigma)
from models.downblock import save_gaussian_as_pdb
from models.upsample_block import init_sigma_from_child_spacing,init_semantic_from_mu_to_parents,sample_from_mixture
from models.pool import coarse_rigids_from_mu_sigma,fps_points_batch
from models.IGA import InvariantGaussianAttention,CoarseIGATower,GaussianUpdateBlock,fused_gaussian_overlap_score
from chroma.layers.basic import FourierFeaturization
from models.EdgeCoarsen import CoarseEdgeCoarsenAndFuse
from models.pool import segment_pos01_from_assignment,Pos01FourierEncoder
from data.GaussianRigid import save_gaussian_as_pdb,OffsetGaussianRigid
from openfold.utils.rigid_utils import Rigid,Rotation
class ResIdxFourierEmbedding(nn.Module):
    """
    res_idx -> continuous Fourier positional embedding

    Inputs:
      res_idx:   [B, N]  (int/long)
      node_mask: [B, N]  (0/1 float or bool) 用来估计每条序列长度 L
      chain_id:  [B, N]  optional (int/long) 多链区分（可选）

    Output:
      q0: [B, N, C]
    """
    def __init__(
        self,
        c_s: int,
        scale: float = 1.0,
        trainable: bool = False,
        use_chain_emb: bool = False,
        max_chain_id: int = 8,
    ):
        super().__init__()
        self.c_s = int(c_s)
        self.use_chain_emb = bool(use_chain_emb)

        # FourierFeaturization 要求 d_model 是偶数
        ff_dim = self.c_s if (self.c_s % 2 == 0) else (self.c_s + 1)
        self.ff_dim = ff_dim

        self.pos_ff = FourierFeaturization(
            d_input=1,
            d_model=ff_dim,
            trainable=trainable,
            scale=scale,
        )

        # 若 c_s 是奇数，做一个线性投影回 c_s
        self.proj = nn.Identity() if ff_dim == self.c_s else nn.Linear(ff_dim, self.c_s, bias=False)

        if self.use_chain_emb:
            self.chain_emb = nn.Embedding(max_chain_id, self.c_s)

        self.out_ln = nn.LayerNorm(self.c_s)

    def forward(self, res_idx, node_mask, chain_id=None):
        # res_idx: [B,N]
        # node_mask: [B,N]
        B, N = res_idx.shape
        device = res_idx.device

        # 1) 估计每条序列的有效长度 L（用 node_mask）
        #    node_mask 允许 float/bool
        if node_mask.dtype != torch.float32 and node_mask.dtype != torch.float16 and node_mask.dtype != torch.bfloat16:
            m = node_mask.float()
        else:
            m = node_mask

        L = m.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]

        # 2) 把 res_idx 归一化到 [0,1]（关键！避免“序号很大/长度变化”导致频域错位）
        denom = (L - 1.0).clamp_min(1.0)
        pos = res_idx.float() / denom                      # [B,N]
        pos = pos.clamp(0.0, 1.0).unsqueeze(-1)            # [B,N,1]

        # 3) Fourier features
        q0 = self.pos_ff(pos)                              # [B,N,ff_dim]
        q0 = self.proj(q0)                                 # [B,N,C]

        # 4) 可选：多链区分
        if self.use_chain_emb:
            assert chain_id is not None, "use_chain_emb=True requires chain_id"
            q0 = q0 + self.chain_emb(chain_id.clamp_min(0).clamp_max(self.chain_emb.num_embeddings - 1))

        # 5) mask + LN
        q0 = q0 * m.unsqueeze(-1)                           # [B,N,C]
        q0 = self.out_ln(q0)
        return q0

class FinalCoarseToFineIGAModule(nn.Module):
    """
    在所有 upsample 结束后，用最后 coarse(K) 直接生成 residue-level(N)：
      1) Query-based B (local parents)
      2) Moment lift: (mu0, Sigma0) + s0
      3) geo regularization: attach + ent_B + occ
      4) build rigids
      5) residue IGA refine x k

    返回：
      levels: [ { "B", "parent_idx", "s0","mu0","Sigma0","r0","mask0",
                 "s","r","mask", "aux"} ]
      reg_total: loss_geo (或你想加权后的总正则)
    """

    def __init__(
        self,
        c_s: int,
        iga_conf,
        OffsetGaussianRigid_cls,
        num_refine_layers: int = 4,
        neighbor_R: int = 1,          # R=1: 只看出生父；R>1: 加近邻父
        jitter: float = 1e-4,
        w_attach: float = 1.0,
        w_entB: float = 0.0,          # early=0, late可调大
        w_occ: float = 0.0,           # 可选
        enable_occ_loss: bool = False,
    use_chain_emb: bool = False,
        max_chain_id: int = 8,
    ):
        super().__init__()
        self.c_s = c_s
        self.OffsetGaussianRigid_cls = OffsetGaussianRigid_cls
        self.neighbor_R = neighbor_R
        self.jitter = jitter

        self.w_attach = w_attach
        self.w_entB = w_entB
        self.w_occ = w_occ
        self.enable_occ_loss = enable_occ_loss

        # residue index query embedding (实名 query)
        # self.res_idx_emb = nn.Embedding(4096, c_s)  # 足够大即可

        self.res_idx_emb = ResIdxFourierEmbedding(
            c_s=c_s,
            scale=1.0,  # 常用 0.5~2.0，可调
            trainable=False,  # 建议先 False，稳定
            use_chain_emb=use_chain_emb,
            max_chain_id=max_chain_id,
        )


        self.q_proj = nn.Linear(c_s, c_s, bias=False)
        self.k_proj = nn.Linear(c_s, c_s, bias=False)
        self.v_proj = nn.Linear(c_s, c_s, bias=False)

        # residue refine tower
        iga = InvariantGaussianAttention(
            c_s=c_s,
            c_z=getattr(iga_conf, "hgfc_z", 0),
            c_hidden=iga_conf.c_hidden,
            no_heads=iga_conf.no_heads,
            no_qk_gaussians=iga_conf.no_qk_points,
            no_v_points=iga_conf.no_v_points,
            layer_idx=9000,
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

    @torch.no_grad()
    def _topk_parents_by_overlap(self, mu_parent, Sig_parent, mask_parent, R: int):
        """
        【已修改】使用高斯重叠分数 (Gaussian Overlap) 代替欧氏距离寻找近邻父节点。

        这本质上是基于 (Sigma_i + Sigma_j) 的双向马氏距离。
        只有当两个椭圆在几何形状上真正重叠时，Score 才会高。

        mu_parent: [B, K, 3]
        Sig_parent: [B, K, 3, 3]
        mask_parent: [B, K]
        """
        B, K, _ = mu_parent.shape
        device = mu_parent.device

        # 1. 准备两两差分 delta: [B, K, K, 3]
        delta = mu_parent.unsqueeze(2) - mu_parent.unsqueeze(1)

        # 2. 准备两两协方差之和 Sigma_sum: [B, K, K, 3, 3]
        #    Sigma_sum = Sigma_i + Sigma_j
        Sig_sum = Sig_parent.unsqueeze(2) + Sig_parent.unsqueeze(1)

        #    加上 eps 防止奇异
        eps = 1e-6
        eye = torch.eye(3, device=device).reshape(1, 1, 1, 3, 3)
        Sig_sum = Sig_sum + eye * eps

        # 3. 计算重叠分数 (Score 越大越重叠, 范围 (-inf, 0])
        #    注意：这里不需要求 exp，直接比大小即可
        score = fused_gaussian_overlap_score(delta, Sig_sum)  # [B, K, K]

        # 4. Mask 处理
        #    如果任一节点无效，Score 设为 -inf
        mask_2d = mask_parent.unsqueeze(1) * mask_parent.unsqueeze(2)  # [B, K, K]
        score = score.masked_fill(mask_2d < 0.5, -1e9)

        # 5. TopK (选取分数最大的 R 个)
        #    largest=True 因为 score 是负数，越接近 0 越好
        top_idx = torch.topk(score, k=min(R, K), dim=-1, largest=True).indices  # [B, K, R]

        return top_idx

    def forward(self, s_parent, r_parent, mask_parent, node_mask, res_idx):
        """
        s_parent: [B,K,C]
        r_parent: OffsetGaussianRigid [B,K]
        mask_parent: [B,K]
        node_mask: [B,N]
        res_idx: [B,N] (真实 residue index)
        """
        levels = []
        reg_total = 0.0

        Bsz, K, C = s_parent.shape
        N = node_mask.shape[1]
        mask0 = node_mask

        # ---- geometry from parent ----
        mu_p = r_parent.get_gaussian_mean()     # [B,K,3]
        Sig_p = r_parent.get_covariance()       # [B,K,3,3]

        # ---- build local parent candidate set N(i) ----
        # 1) 先选“出生父” j0：用 query-key attention 的 argmax 近似（不一定hard，用soft也行）
        q = self.q_proj(self.res_idx_emb(res_idx.clamp_min(0).clamp_max(4095)))   # [B,N,C]
        k = self.k_proj(s_parent)                                                 # [B,K,C]

        logits_full = torch.einsum("bnc,bkc->bnk", q, k) / (C ** 0.5)             # [B,N,K]
        logits_full = logits_full + (mask_parent[:, None, :] - 1.0) * 1e9         # mask
        j0 = torch.argmax(logits_full, dim=-1)                                    # [B,N]

        if self.neighbor_R <= 1:
            parent_idx = j0[..., None]                                            # [B,N,1]
        else:
            # -----------------------------------------------------------------
            # 2) 【修改点】加近邻父：使用高斯重叠分数 (Overlap Score)
            # -----------------------------------------------------------------
            # knn: [B, K, R] - 每个父节点，找到了 R 个几何上最重叠的“邻居父”
            knn = self._topk_parents_by_overlap(mu_p, Sig_p, mask_parent, self.neighbor_R)

            # Indexing: 根据每个 residue 的出生父 j0，查表得到它的邻居集合
            # gather indices: [B, N, R]
            parent_idx = torch.gather(knn, 1, j0[..., None].expand(-1, -1, knn.shape[-1]))

        R = parent_idx.shape[-1]

        # ---- gather parent subset ----
        # mu_sub: [B,N,R,3], Sig_sub: [B,N,R,3,3], s_sub: [B,N,R,C]
        mu_sub = mu_p[:, None, :, :].expand(Bsz, N, K, 3).gather(
            2, parent_idx[..., None].expand(Bsz, N, R, 3)
        )
        Sig_sub = Sig_p[:, None, :, :, :].expand(Bsz, N, K, 3, 3).gather(
            2, parent_idx[..., None, None].expand(Bsz, N, R, 3, 3)
        )
        s_sub = s_parent[:, None, :, :].expand(Bsz, N, K, C).gather(
            2, parent_idx[..., None].expand(Bsz, N, R, C)
        )

        # ---- query-based B over local parents ----
        k_sub = self.k_proj(s_sub)                                                 # [B,N,R,C]
        v_sub = self.v_proj(s_sub)
        logits = (q[:, :, None, :] * k_sub).sum(dim=-1) / (C ** 0.5)              # [B,N,R]
        # node mask
        logits = logits + (mask0[:, :, None] - 1.0) * 1e9
        B_local = F.softmax(logits, dim=-1)                                       # [B,N,R]

        # ---- lift semantic ----
        s0 = torch.einsum("bnr,bnrc->bnc", B_local, v_sub)                         # [B,N,C]
        s0 = s0 * mask0[..., None]

        # ---- lift geometry: moment ----
        mu0 = torch.einsum("bnr,bnrp->bnp", B_local, mu_sub)                       # [B,N,3]
        d = (mu_sub - mu0[:, :, None, :])                                          # [B,N,R,3]
        outer = d[..., :, None] * d[..., None, :]                                  # [B,N,R,3,3]
        Sig0 = (B_local[..., None, None] * (Sig_sub + outer)).sum(dim=2)  # sum over R # Sig0: [B, N, 3, 3]
        # [B,N,3,3]
        Sig0 = Sig0 + self.jitter * torch.eye(3, device=Sig0.device)[None, None]

        # ---- geo loss using your fused overlap score ----
        # score: [B,N,R]
        delta = mu0[:, :, None, :] - mu_sub                                        # [B,N,R,3]
        score = fused_gaussian_overlap_score(delta, Sig_sub)                       # [B,N,R]  (<=0)
        # attach
        denom = mask0.sum().clamp_min(1.0)
        loss_attach = - (mask0[:, :, None] * (B_local * score)).sum() / denom

        # entropy of B
        ent = -(B_local * torch.log(B_local.clamp_min(1e-9))).sum(dim=-1)          # [B,N]
        loss_ent_B = (ent * mask0).sum() / denom

        # occupancy (scatter to [B,K])
        occ = torch.zeros((Bsz, K), device=s_parent.device, dtype=s_parent.dtype)
        # add contributions of B_local into occ via parent_idx
        occ.scatter_add_(1, parent_idx.reshape(Bsz, -1),
                         (B_local * mask0[:, :, None]).reshape(Bsz, -1))
        occ_norm = occ / (occ.sum(dim=-1, keepdim=True).clamp_min(1e-9))

        if self.enable_occ_loss:
            uni = torch.full_like(occ_norm, 1.0 / max(K, 1))
            loss_occ = F.mse_loss(occ_norm, uni)
        else:
            loss_occ = torch.tensor(0.0, device=s_parent.device)

        loss_geo = self.w_attach * loss_attach + self.w_entB * loss_ent_B + self.w_occ * loss_occ
        reg_total = reg_total + loss_geo

        # ---- build residue rigids (no residue anchor used) ----
        r0 = coarse_rigids_from_mu_sigma(mu0, Sig0, self.OffsetGaussianRigid_cls)

        # ---- refine (IGA + transition + update) ----
        s1, r1 = self.refine_tower(s0, r0, mask0)

        aux = {
            "B": B_local,                # local form [B,N,R]
            "parent_idx": parent_idx,    # [B,N,R]
            "occ": occ,                  # [B,K]
            "occ_norm": occ_norm,        # [B,K]
            "score_ij": score,           # [B,N,R]
            "loss_attach": loss_attach,
            "loss_ent_B": loss_ent_B,
            "loss_occ": loss_occ,
            "loss_geo": loss_geo,
        }

        levels.append({
            "B": B_local,
            "parent_idx": parent_idx,
            "s0": s0,
            "mu0": mu0,
            "Sigma0": Sig0,
            "r0": r0,
            "mask0": mask0,
            "s": s1,
            "r": r1,
            "mask": mask0,
            "aux": aux,
        })

        return levels, reg_total


class FinalCoarseToFineDensenSampleIGAModule(nn.Module):
    """
    Up-init by density sampling + FPS (geometry coverage),
    semantic init by distance-to-parent-mu (+ occ prior),
    then residue-level IGA refine.
    """

    def __init__(
        self,
        c_s: int,
        iga_conf,
        OffsetGaussianRigid_cls,
        num_refine_layers: int = 4,
        oversample_mul: int = 6,     # M = oversample_mul * N
        fps_knn: int = 4,
        sigma_cover_alpha: float = 0.6,
        sigma_floor: float = 0.03,   # nm
        sigma_ceil: float = 2.0,     # nm
        sigma_s: float = 1.0,        # nm: semantic-from-geometry bandwidth
        jitter: float = 1e-6,
    ):
        super().__init__()
        self.c_s = c_s
        self.OffsetGaussianRigid_cls = OffsetGaussianRigid_cls

        self.oversample_mul = int(oversample_mul)
        self.fps_knn = int(fps_knn)
        self.sigma_cover_alpha = float(sigma_cover_alpha)
        self.sigma_floor = float(sigma_floor)
        self.sigma_ceil = float(sigma_ceil)
        self.sigma_s = float(sigma_s)
        self.jitter = float(jitter)


        self.edge_fusers = CoarseEdgeCoarsenAndFuse(
                c_z_in=getattr(iga_conf, "hgfc_z", 0),  # 你 IGA 用的 Cz
                c_z_out=getattr(iga_conf, "hgfc_z", 0),  # 你 IGA 用的 Cz
            mode= "up",
                # 这里填你类需要的其它配置，比如:
                # geo_dim=..., sem_dim=..., fuse_dim=..., use_geo=True ...
            )


        iga = InvariantGaussianAttention(
            c_s=c_s,
            c_z=getattr(iga_conf, "hgfc_z", 0),
            c_hidden=iga_conf.c_hidden,
            no_heads=iga_conf.no_heads,
            no_qk_gaussians=iga_conf.no_qk_points,
            no_v_points=iga_conf.no_v_points,
            layer_idx=9000,
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

    def forward(self, s_parent,z_parent, r_parent, mask_parent, node_mask, occ_parent=None, res_idx=None):
        """
        Inputs:
          s_parent: [B,K,C]
          r_parent: OffsetGaussianRigid [B,K]
          mask_parent: [B,K]
          node_mask: [B,N]
          occ_parent: [B,K]  (optional; pass your curr_occ / occ_norm from down)
        """
        B, K, C = s_parent.shape
        N = node_mask.shape[1]
        device, dtype = s_parent.device, s_parent.dtype

        mu_p = r_parent.get_gaussian_mean()   # [B,K,3]
        Sig_p = r_parent.get_covariance()     # [B,K,3,3]

        # ---- pi from occ ----
        if occ_parent is None:
            pi = torch.ones((B, K), device=device, dtype=dtype)
        else:
            pi = occ_parent.to(dtype=dtype)
        pi = pi * (mask_parent > 0.5).to(dtype)
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # ---- mixture oversample ----
        M = int(self.oversample_mul * N)
        cand_x, _ = sample_from_mixture(mu_p, Sig_p, pi, M, mask_parent=mask_parent, eps=1e-6)  # [B,M,3]

        # ---- FPS select N ----
        idx = fps_points_batch(cand_x, N)  # [B,N]
        mu0 = cand_x.gather(1, idx[..., None].expand(B, N, 3))  # [B,N,3]
        mu0 = mu0 * node_mask[..., None]




        # ---- Sigma0 from child spacing coverage ----
        Sig0 = init_sigma_from_child_spacing(
            mu0, node_mask, k_nn=self.fps_knn,
            alpha=self.sigma_cover_alpha,
            sigma_floor=self.sigma_floor,
            sigma_ceil=self.sigma_ceil,
        )
        I = torch.eye(3, device=device, dtype=dtype)[None, None]
        Sig0 = Sig0 + self.jitter * I * node_mask[:, :, None, None]

        # ---- s0 from distance-to-parent-mu (+ pi) ----
        s0, w = init_semantic_from_mu_to_parents(
            mu0, s_parent, mu_p, pi, mask_parent=mask_parent, sigma_s=self.sigma_s
        )
        s0 = s0 * node_mask[..., None]

        # ---- build residue rigids (Identity R) ----
        r0 = coarse_rigids_from_mu_sigma(mu0, Sig0, self.OffsetGaussianRigid_cls)

        z_c, Z_sem_c, Z_geo_c = self.edge_fusers(
            A=w, Z_in=z_parent, r_target=r0,
            mask_f=node_mask, mask_c=mask_parent
        )

        # ang_rigids = r0.scale_translation(10.0)  # 0.1
        #
        # # pooling 得到 r_c / mask_c 后
        # save_gaussian_as_pdb(
        #     gaussian_rigid=ang_rigids,
        #     filename=f"debug__up_r0.pdb",
        #     mask=node_mask,
        #     center_mode="gaussian_mean",
        # )
        # ---- refine ----
        s1, r1,z1 = self.refine_tower(s0, r0, node_mask,z_c)

        # ang_rigids = r1.scale_translation(10.0)  # 0.1
        #
        # # pooling 得到 r_c / mask_c 后
        # save_gaussian_as_pdb(
        #     gaussian_rigid=ang_rigids,
        #     filename=f"debug__up_r1.pdb",
        #     mask=node_mask,
        #     center_mode="gaussian_mean",
        # )


        levels = [{
            "s0": s0, "mu0": mu0, "Sigma0": Sig0, "r0": r0,
            "s": s1, "z": z1,"r": r1, "mask": node_mask,
            "aux": {"w_parent": w, "pi": pi}
        }]
        reg_total = torch.tensor(0.0, device=device, dtype=dtype)
        return levels, reg_total





# ----------------------------
# small numerics utils
# ----------------------------
def _sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))

def _safe_cholesky_3x3_manual(sigma: torch.Tensor, eps: float = 1e-6):
    """
    手动实现 3x3 Cholesky 分解，永不崩溃。
    返回 L，使得 L @ L^T = sigma
    """
    # sigma: [..., 3, 3]
    s00 = sigma[..., 0, 0]
    s11 = sigma[..., 1, 1]
    s22 = sigma[..., 2, 2]
    s01 = sigma[..., 0, 1]
    s02 = sigma[..., 0, 2]
    s12 = sigma[..., 1, 2]

    # L = [[l00, 0, 0], [l10, l11, 0], [l20, l21, l22]]
    l00 = torch.sqrt(torch.clamp(s00, min=eps))
    l10 = s01 / torch.clamp(l00, min=eps)
    l20 = s02 / torch.clamp(l00, min=eps)

    l11_sq = s11 - l10**2
    l11 = torch.sqrt(torch.clamp(l11_sq, min=eps))
    l21 = (s12 - l20 * l10) / torch.clamp(l11, min=eps)

    l22_sq = s22 - l20**2 - l21**2
    l22 = torch.sqrt(torch.clamp(l22_sq, min=eps))

    # 构造 L 矩阵
    res_shape = list(sigma.shape)
    L = torch.zeros(res_shape, device=sigma.device, dtype=sigma.dtype)
    L[..., 0, 0] = l00
    L[..., 1, 0] = l10
    L[..., 1, 1] = l11
    L[..., 2, 0] = l20
    L[..., 2, 1] = l21
    L[..., 2, 2] = l22
    return L
def _safe_cholesky(A: torch.Tensor, jitter: float = 1e-6, max_tries: int = 5) -> torch.Tensor:
    """
    A: [...,3,3] SPD-ish -> L: [...,3,3]
    """
    I = torch.eye(3, device=A.device, dtype=A.dtype)
    A = _sym(A)
    for t in range(max_tries):
        eps = jitter * (10 ** t)
        try:
            return torch.linalg.cholesky(A + eps * I)
        except RuntimeError:
            continue
    # last resort
    return _safe_cholesky_3x3_manual(A + jitter * (10 ** (max_tries - 1)) * I)


def _solve_maha2(L: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    L: [...,3,3] lower cholesky of Sigma
    x: [...,3]
    return maha2 = x^T Sigma^{-1} x
    """
    y = torch.linalg.solve_triangular(L, x.unsqueeze(-1), upper=False)  # [...,3,1]
    return (y.squeeze(-1) ** 2).sum(dim=-1)


def _chi2_3d_quantile(q: float) -> float:
    """
    Common quantiles for Chi-square with df=3.
    Avoid scipy dependency.
    """
    # df=3 chi2 quantiles (approx):
    # 0.90: 6.251, 0.95: 7.815, 0.975: 9.348, 0.99: 11.345
    if q >= 0.99:
        return 11.345
    if q >= 0.975:
        return 9.348
    if q >= 0.95:
        return 7.815
    if q >= 0.90:
        return 6.251
    # fallback linear-ish
    return 6.251


# ----------------------------
# FPS (your existing one is ok; keep compatible)
# ----------------------------



# ----------------------------
# per-parent quota allocation
# ----------------------------
def _allocate_quota(pi: torch.Tensor, N: int, mask_parent: torch.Tensor) -> torch.Tensor:
    """
    pi: [B,K] normalized (or not)
    mask_parent: [B,K] {0,1}
    return n_k: [B,K] int, sum_k n_k == N (per batch)
    """
    B, K = pi.shape
    pi = pi * (mask_parent > 0.5).to(pi.dtype)
    pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    raw = pi * float(N)  # [B,K]
    base = torch.floor(raw).to(torch.long)
    rem = N - base.sum(dim=-1)  # [B]
    # distribute remaining by largest fractional parts
    frac = (raw - base.to(raw.dtype))
    frac = frac.masked_fill(mask_parent < 0.5, -1e9)

    n_k = base.clone()
    for b in range(B):
        r = int(rem[b].item())
        if r <= 0:
            continue
        top = torch.topk(frac[b], k=min(r, K), dim=-1).indices
        n_k[b, top] += 1

    # ensure masked parents get 0
    n_k = n_k * (mask_parent > 0.5).to(torch.long)

    # fix any numerical mismatch
    diff = N - n_k.sum(dim=-1)  # [B]
    for b in range(B):
        d = int(diff[b].item())
        if d == 0:
            continue
        valid = torch.where(mask_parent[b] > 0.5)[0]
        if valid.numel() == 0:
            continue
        if d > 0:
            # add to largest pi
            order = torch.argsort(pi[b, valid], descending=True)
            for t in range(d):
                n_k[b, valid[order[t % order.numel()]]] += 1
        else:
            # remove from largest n_k
            order = torch.argsort(n_k[b, valid], descending=True)
            for t in range(-d):
                j = valid[order[t % order.numel()]]
                if n_k[b, j] > 0:
                    n_k[b, j] -= 1

    return n_k


# ----------------------------
# per-parent inside sampling + local FPS
# ----------------------------
@torch.no_grad()
def _sample_inside_parent(
    mu_p: torch.Tensor,          # [B,K,3]
    Sig_p: torch.Tensor,         # [B,K,3,3]
    n_k: torch.Tensor,           # [B,K] int
    chi2_q: float = 0.95,
    oversample: int = 4,
    jitter: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each parent k, sample n_k points inside ellipsoid (Mahalanobis^2 <= chi2(q)),
    using rejection-lite (oversample and keep best) + local FPS.

    Returns:
      mu0: [B,N,3]
      parent_idx: [B,N] long, indicates which parent each child belongs to
    """
    B, K, _ = mu_p.shape
    device, dtype = mu_p.device, mu_p.dtype
    chi2_thr = _chi2_3d_quantile(chi2_q)

    # total N per batch
    N = int(n_k.sum(dim=-1).max().item())
    # but we need exact per-batch N; assume all batches same N (your node_mask gives N)
    # We'll pack to max then trim per batch later.
    # To keep simple: build lists then cat.

    mu0_list = []
    pid_list = []

    I = torch.eye(3, device=device, dtype=dtype)

    for b in range(B):
        pts_b = []
        pid_b = []
        for k in range(K):
            nk = int(n_k[b, k].item())
            if nk <= 0:
                continue

            # oversample candidates
            M = max(nk * oversample, nk)
            Sig = _sym(Sig_p[b, k]) + jitter * I
            L = _safe_cholesky(Sig, jitter=jitter)

            z = torch.randn((M, 3), device=device, dtype=dtype)
            x = mu_p[b, k] + (z @ L.transpose(0, 1))  # [M,3]

            # inside score by mahalanobis
            delta = x - mu_p[b, k]
            maha2 = _solve_maha2(L, delta)  # [M]

            # prefer inside; if not enough inside, take smallest maha2
            inside = (maha2 <= chi2_thr)
            if inside.sum().item() >= nk:
                cand = x[inside][: nk]
            else:
                order = torch.argsort(maha2, descending=False)
                cand = x[order][: nk]

            # local FPS within this parent's selected candidates (stabilize coverage)
            if nk >= 2:
                cand_b = cand.unsqueeze(0)  # [1,nk,3]
                idx = fps_points_batch(cand_b, nk)  # trivial but makes consistent
                cand = cand_b.gather(1, idx[..., None].expand(1, nk, 3)).squeeze(0)

            pts_b.append(cand)
            pid_b.append(torch.full((nk,), k, device=device, dtype=torch.long))

        if len(pts_b) == 0:
            # fallback: one dummy
            pts_b = [mu_p[b, 0:1]]
            pid_b = [torch.zeros((1,), device=device, dtype=torch.long)]

        pts_b = torch.cat(pts_b, dim=0)  # [Nb,3]
        pid_b = torch.cat(pid_b, dim=0)  # [Nb]

        mu0_list.append(pts_b)
        pid_list.append(pid_b)

    # pad to max_N then stack
    maxN = max(x.shape[0] for x in mu0_list)
    mu0 = torch.zeros((B, maxN, 3), device=device, dtype=dtype)
    pid = torch.zeros((B, maxN), device=device, dtype=torch.long)
    for b in range(B):
        nb = mu0_list[b].shape[0]
        mu0[b, :nb] = mu0_list[b]
        pid[b, :nb] = pid_list[b]
        if nb < maxN:
            pid[b, nb:] = 0

    return mu0, pid


# ----------------------------
# spacing-based Sigma init (local kNN in child space)
# ----------------------------


# ----------------------------
# semantic init + parent assignment weights (for edge_fuser A)
# ----------------------------






@dataclass
class UpAux:
    w_parent: torch.Tensor  # [B,N,K]
    # pi: torch.Tensor        # [B,K]
    parent_idx: torch.Tensor  # [B,N]
    debug:dict


# =========================================================
# FinalCoarseToFineDensenSampleIGAModule (per-parent cover)
# =========================================================
class FinalCoarseToFineDensenSampleIGAModulev2(nn.Module):
    """
    v2 Up-sampling:
    Cover-based (quota + inside-ellipsoid) init
    + anisotropic parent-aware Sigma
    + spacing fill as regularizer
    + IGA refinement
    """

    def __init__(
        self,
        c_s: int,
        iga_conf,
        OffsetGaussianRigid_cls,
        num_refine_layers: int = 4,

        # ---------- retained v1 knobs (now re-interpreted) ----------
        oversample_mul: int = 4,        # kept for fallback / ablation
        chi2_q: float = 0.95,            # χ² quantile for inside-parent constraint
        fps_knn: int = 8,                # spacing-based sigma
        sigma_cover_alpha: float = 1.5,
        sigma_floor: float = 1e-3,
        sigma_ceil: float = 1e2,
        sigma_s: float = 2.0,
        jitter: float = 1e-6,

        # ---------- parent-child Sigma control ----------
        beta_parent_sigma: float = 0.85,   # blend parent Sigma vs spacing Sigma
        parent_sigma_shrink: float = 1.0,  # optional shrink of parent Sigma
    ):
        super().__init__()

        # ============================================================
        # basic
        # ============================================================
        self.c_s = c_s
        self.OffsetGaussianRigid_cls = OffsetGaussianRigid_cls

        # ============================================================
        # store geometry / cover parameters
        # ============================================================
        self.oversample_mul = int(oversample_mul)
        self.chi2_q = float(chi2_q)

        self.fps_knn = int(fps_knn)
        self.sigma_cover_alpha = float(sigma_cover_alpha)
        self.sigma_floor = float(sigma_floor)
        self.sigma_ceil = float(sigma_ceil)
        self.sigma_s = float(sigma_s)
        self.jitter = float(jitter)

        self.beta_parent_sigma = float(beta_parent_sigma)
        self.parent_sigma_shrink = float(parent_sigma_shrink)

        # ============================================================
        # Edge fuser (UP mode only, as you specified)
        # ============================================================
        self.edge_fusers = CoarseEdgeCoarsenAndFuse(
            c_z_in=getattr(iga_conf, "hgfc_z", 0),
            c_z_out=getattr(iga_conf, "hgfc_z", 0),
            mode="up",
        )

        # ============================================================
        # IGA refinement tower (same as v1, untouched)
        # ============================================================
        iga = InvariantGaussianAttention(
            c_s=c_s,
            c_z=getattr(iga_conf, "hgfc_z", 0),
            c_hidden=iga_conf.c_hidden,
            no_heads=iga_conf.no_heads,
            no_qk_gaussians=iga_conf.no_qk_points,
            no_v_points=iga_conf.no_v_points,
            layer_idx=9000,
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


    def forward(
        self,
        s_parent: torch.Tensor,                 # [B,K,C]
        z_parent,                               # can be None or tensor (edge)
        r_parent,                               # OffsetGaussianRigid [B,K]
        mask_parent: torch.Tensor,              # [B,K]
        node_mask: torch.Tensor,                # [B,N]
        occ_parent: Optional[torch.Tensor] = None,  # [B,K] (optional)
        res_idx: Optional[torch.Tensor] = None,
    ):
        B, K, C = s_parent.shape
        N = node_mask.shape[1]
        device, dtype = s_parent.device, s_parent.dtype

        mu_p = r_parent.get_gaussian_mean()      # [B,K,3]
        Sig_p = r_parent.get_covariance()        # [B,K,3,3]

        # ---- pi from occ ----
        if occ_parent is None:
            pi = torch.ones((B, K), device=device, dtype=dtype)
        else:
            pi = occ_parent.to(dtype=dtype)

        pi = pi * (mask_parent > 0.5).to(dtype)
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # ---- quota per parent ----
        n_k = _allocate_quota(pi, N, mask_parent)  # [B,K] int

        # ---- per-parent inside sampling ----
        # we sample a packed list; then truncate to N by node_mask
        mu0_packed, parent_idx_packed = _sample_inside_parent(
            mu_p, Sig_p * self.parent_sigma_shrink, n_k,
            chi2_q=self.chi2_q,
            oversample=self.oversample_mul,
            jitter=max(self.jitter, 1e-6),
        )

        # trim/pad to N
        if mu0_packed.shape[1] < N:
            # pad by repeating first points
            pad = N - mu0_packed.shape[1]
            mu_pad = mu0_packed[:, :1].expand(B, pad, 3)
            pid_pad = parent_idx_packed[:, :1].expand(B, pad)
            mu0 = torch.cat([mu0_packed, mu_pad], dim=1)
            parent_idx = torch.cat([parent_idx_packed, pid_pad], dim=1)
        else:
            mu0 = mu0_packed[:, :N]
            parent_idx = parent_idx_packed[:, :N]

        mu0 = mu0 * node_mask[..., None]

        # ---- spacing Sigma (anti-collapse) ----
        Sig_spacing = init_sigma_from_child_spacing(
            mu0, node_mask,
            k_nn=self.fps_knn,
            alpha=self.sigma_cover_alpha,
            sigma_floor=self.sigma_floor,
            sigma_ceil=self.sigma_ceil,
            jitter=max(self.jitter, 1e-6),
        )  # [B,N,3,3]

        # ---- inherit parent Sigma per child ----
        # gather Sigma_parent for each child by parent_idx
        Sig_parent_child = Sig_p.gather(
            1, parent_idx[..., None, None].expand(B, N, 3, 3)
        )  # [B,N,3,3]

        I = torch.eye(3, device=device, dtype=dtype)[None, None]
        Sig_parent_child = _sym(Sig_parent_child) + max(self.jitter, 1e-6) * I

        # ---- Sigma0 blend ----
        beta = self.beta_parent_sigma
        Sig0 = beta * Sig_parent_child + (1.0 - beta) * Sig_spacing
        Sig0 = _sym(Sig0) + max(self.jitter, 1e-6) * I
        Sig0 = Sig0 * node_mask[:, :, None, None]

        # ---- semantic init + w for edge_fuser ----
        s0, w = init_semantic_from_mu_to_parents(
            mu0, s_parent, mu_p, pi, mask_parent=mask_parent, sigma_s=self.sigma_s
        )
        s0 = s0 * node_mask[..., None]
        w = w * node_mask[..., None]

        # ---- build residue rigids ----
        r0 = coarse_rigids_from_mu_sigma(mu0, Sig0, self.OffsetGaussianRigid_cls)

        # ---- build z for fine from parent edges ----
        if self.edge_fusers is not None and (z_parent is not None):
            z_c, Z_sem_c, Z_geo_c = self.edge_fusers(
                A=w, Z_in=z_parent, r_target=r0,
                mask_f=node_mask, mask_c=mask_parent
            )
        else:
            z_c = None

        # ---- refine ----
        s1, r1, z1 = self.refine_tower(s0, r0, node_mask, z_c)

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

        reg_total = torch.tensor(0.0, device=device, dtype=dtype)
        return levels, reg_total





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
        num_refine_layers: int = 6,

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
        tau_init: float = 0.1,          # softmax temperature
        w_clip_min: float = 0.0,        # optional floor (usually 0)

        # ---- optional regularizers ----
        lambda_route_smooth: float = 0.1,    # KL(w_i||w_{i+1}) weight
        lambda_route_collapse: float = 0.1,  # column-corr / coverage weight

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
    def _w_flip_rate(w: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        相邻残基 parent argmax 切换率。
        w: [B,N,K], node_mask: [B,N] (0/1)
        return: scalar tensor
        """
        # hard assignment
        a = torch.argmax(w, dim=-1)  # [B,N]
        # valid adjacent pairs
        m = (node_mask[:, :-1] > 0.5) & (node_mask[:, 1:] > 0.5)  # [B,N-1]
        diff = (a[:, :-1] != a[:, 1:]) & m
        denom = m.sum().clamp_min(1)
        return diff.sum().float() / denom.float()

    @staticmethod
    def _w_entropy(w: torch.Tensor, node_mask: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """
        平均熵（只统计有效 residue）。
        H(w_i) = -Σ_k w_i,k log w_i,k
        """
        m = (node_mask > 0.5).float()  # [B,N]
        p = w.clamp_min(eps)
        h = -(p * p.log()).sum(dim=-1)  # [B,N]
        denom = m.sum().clamp_min(1.0)
        return (h * m).sum() / denom

    @staticmethod
    def _w_top1_mean(w: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        每个 residue 的 top1 概率平均值（衡量 w 是否很尖）。
        """
        m = (node_mask > 0.5).float()
        top1 = w.max(dim=-1).values  # [B,N]
        denom = m.sum().clamp_min(1.0)
        return (top1 * m).sum() / denom

    @staticmethod
    def _w_column_stats(w: torch.Tensor, node_mask: torch.Tensor) -> dict:
        """
        统计列占用情况：w_sum[k] = Σ_i w[i,k]
        返回几个标量：有效 parent 数、top1列占比、Gini-like 等
        """
        # w_sum: [B,K]
        m = (node_mask > 0.5).float().unsqueeze(-1)  # [B,N,1]
        w_sum = (w * m).sum(dim=1)  # [B,K]
        # normalize per batch for interpretability
        w_norm = w_sum / w_sum.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # [B,K]

        # 有效 parent 数：权重大于阈值的列数（阈值按 1/K 的比例）
        B, K = w_norm.shape
        thr = (1.0 / max(K, 1)) * 0.5
        active = (w_norm > thr).sum(dim=-1).float().mean()  # scalar

        # 最大列占比（越大越塌缩）
        top_col = w_norm.max(dim=-1).values.mean()

        # 简单“集中度”指标：sum(p^2)（越大越集中）
        conc = (w_norm ** 2).sum(dim=-1).mean()

        return {
            "w_active_cols": active,   # 越大越分散
            "w_top_col": top_col,      # 越大越塌缩
            "w_col_conc": conc,        # 越大越塌缩
        }
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
        # if self.w_clip_min > 0:
        #     w = w.clamp_min(self.w_clip_min)
        #     w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # ensure padded rows are exactly 0
        w = w * node_mask[:, :, None].to(dtype)

        # ---- DEBUG stats (cheap) ----
        with torch.no_grad():
            w_flip = self._w_flip_rate(w, node_mask)  # 相邻切换率
            w_ent = self._w_entropy(w, node_mask)  # 平均熵
            w_top1 = self._w_top1_mean(w, node_mask)  # top1均值
            col_stats = self._w_column_stats(w, node_mask)  # 列占用/塌缩

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

        # ----------------------------------------------------------
        # 【新增】防塌缩核心 Loss (Anti-Collapse Kit)
        # ----------------------------------------------------------

        # A. 最小化熵 (MinEntropy) —— 拒绝“平均主义”
        # 作用：强迫 w 变得尖锐 (接近 One-Hot)。
        # 原理：如果 w 很软 (0.1, 0.1...), mu0 就会被拉向原点。如果 w 尖锐 (1, 0...), mu0 就跳到父节点位置。
        w_eps = w.clamp(min=1e-9)
        entropy = -(w * w_eps.log()).sum(dim=-1)  # [B, N]
        # 只统计有效节点
        mask_sum = node_mask.sum().clamp_min(1.0)
        loss_entropy = (entropy * node_mask).sum() / mask_sum

        # B. 负载均衡 (LoadBalance) —— 拒绝“赢家通吃”
        # 作用：强迫所有父节点都被用到。
        # 原理：一旦 Entropy 变低，模型很容易所有点都选同一个父节点。这个 Loss 强迫它分散。
        usage = w.mean(dim=1)  # [B, K] 平均每个父节点分到了多少比例的子节点
        target_usage = torch.ones_like(usage) / max(K, 1)  # 目标：大家均分
        loss_balance = F.mse_loss(usage, target_usage)

        # C. 全局尺度惩罚 (Scale Hinge Loss) —— 暴力撑开
        # 作用：直接告诉模型，如果你生成的直径太小，就要受罚。
        # 这是最直接救 Pred Scale 的手段。
        # 计算当前预测的 Scale (以 Batch 为单位的平均回转半径)
        center = mu0.mean(dim=1, keepdim=True)
        dist = (mu0 - center).norm(dim=-1)  # [B, N]
        current_scale = (dist * node_mask).sum(dim=1) / mask_sum  # [B]
        # 目标 Scale: 设一个合理的下限，比如 10.0 (Angstrom)
        # 如果小于 10，就惩罚；大于 10，不惩罚。
        loss_scale_hinge = F.relu(0.5 - current_scale).mean()

        # ----------------------------------------------------------
        # 【权重建议】
        # Balance: 0.1 ~ 0.5 (必须够大，防止 Mode Collapse)
        # Entropy: 0.01 ~ 0.05 (辅助变尖)
        # Scale:   1.0 (前期强力撑开，Scale 正常后这个 Loss 会自动变 0)
        # ----------------------------------------------------------

        # 将这些加到 reg_total 里
        reg_total = reg_total + 0.5 * loss_balance + 0.1 * loss_entropy + 2.0 * loss_scale_hinge



        # parent_idx: soft router doesn't have hard parent_idx; store argmax for logging/debug
        parent_idx = torch.argmax(w, dim=-1)  # [B,N]

        aux_debug = {
            "w_flip": w_flip,
            "w_entropy": w_ent,
            "w_top1": w_top1,
            **col_stats,
        }

        levels = [{
            "s0": s0,
            "mu0": mu0,
            "Sigma0": Sig0,
            "r0": r0,
            "s": s1,
            "z": z1,
            "r": r1,
            "mask": node_mask,
            "aux": UpAux(w_parent=w, pi=pi, parent_idx=parent_idx, debug=aux_debug)
        }]

        return levels, reg_total


# =========================================================================
# V3.2 核心模块
# =========================================================================
class FinalCoarseToFineDensenSampleIGAModulev3_2(nn.Module):
    """
    V3.2 Robust Up-sampling Module
    ------------------------------
    Key Features:
    1. Hard Gumbel Routing: Prevents averaging collapse (points stuck in center).
    2. Scaled Offset Gen: Predicts relative position within parent volume.
    3. Numerical Safety: Uses diagonal scale instead of Cholesky.
    4. Built-in Reg: Includes LoadBalance and ScaleHinge losses.
    """

    def __init__(
            self,
            c_s: int,
            iga_conf,
            OffsetGaussianRigid_cls,
            num_refine_layers: int = 6,

            # ---- Spacing / Sigma Knobs ----
            fps_knn: int = 8,
            sigma_cover_alpha: float = 1.5,
            sigma_floor: float = 1e-3,
            sigma_ceil: float = 1e2,
            jitter: float = 1e-6,

            # ---- Parent-Child Blend ----
            beta_parent_sigma: float = 0.85,
            parent_sigma_shrink: float = 1.0,

            # ---- Router Knobs ----
            use_ln: bool = True,
            tau_init: float = 1.0,  # Gumbel temperature (start high, e.g., 1.0)

            # ---- Generator Knobs ----
            delta_scale: float = 1.0,  # Scale multiplier for predicted offset
    ):
        super().__init__()
        self.c_s = c_s
        self.OffsetGaussianRigid_cls = OffsetGaussianRigid_cls

        # Sigma parameters
        self.fps_knn = int(fps_knn)
        self.sigma_cover_alpha = float(sigma_cover_alpha)
        self.sigma_floor = float(sigma_floor)
        self.sigma_ceil = float(sigma_ceil)
        self.jitter = float(jitter)
        self.beta_parent_sigma = float(beta_parent_sigma)
        self.parent_sigma_shrink = float(parent_sigma_shrink)
        self.delta_scale = float(delta_scale)

        # Router parameters
        self.tau_init = float(tau_init)
        self.use_ln = bool(use_ln)

        # -------------------------------------------------------
        # 1. Router Components (Simplified Cross Attention)
        # -------------------------------------------------------
        if self.use_ln:
            self.ln_q = nn.LayerNorm(c_s)
            self.ln_k = nn.LayerNorm(c_s)
            self.ln_v = nn.LayerNorm(c_s)

        # Q comes from s_trunk (Child), K/V come from s_parent
        self.q_proj = nn.Linear(c_s, c_s, bias=False)
        self.k_proj = nn.Linear(c_s, c_s, bias=False)
        self.v_proj = nn.Linear(c_s, c_s, bias=False)
        self.pos_proj = nn.Linear(3, c_s, bias=False)  # Positional bias for parents

        # -------------------------------------------------------
        # 2. Generator Components (Offset & Scale Prediction)
        # -------------------------------------------------------
        # Input: Concat[s_trunk, s_parent_selected] -> 2 * c_s
        # Predicts relative offset in local frame (approx -1 to 1)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(2 * c_s),
            nn.Linear(2 * c_s, c_s),
            nn.GELU(),
            nn.Linear(c_s, 3),
            nn.Tanh()  # [Vital] Prevents points from flying to infinity
        )

        # Optional: Predict log-scale update for sigma
        self.logscale_head = nn.Sequential(
            nn.LayerNorm(2 * c_s),
            nn.Linear(2 * c_s, c_s),
            nn.GELU(),
            nn.Linear(c_s, 3),
            nn.Tanh()  # Limit scaling factor range
        )

        # -------------------------------------------------------
        # 3. Refine Tower & Utils (Existing infrastructure)
        # -------------------------------------------------------
        self.edge_fusers = CoarseEdgeCoarsenAndFuse(
            c_z_in=getattr(iga_conf, "hgfc_z", 0),
            c_z_out=getattr(iga_conf, "hgfc_z", 0),
            mode="up",
        )

        # IGA Tower setup
        iga = InvariantGaussianAttention(
            c_s=c_s,
            c_z=getattr(iga_conf, "hgfc_z", 0),
            c_hidden=iga_conf.c_hidden,
            no_heads=iga_conf.no_heads,
            no_qk_gaussians=iga_conf.no_qk_points,
            no_v_points=iga_conf.no_v_points,
            layer_idx=9000,
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

    def forward(
            self,
            s_parent: torch.Tensor,  # [B,K,C]
            z_parent: Optional[torch.Tensor],  # [B,K,K,Cz]
            r_parent,  # OffsetGaussianRigid [B,K]
            mask_parent: torch.Tensor,  # [B,K]
            node_mask: torch.Tensor,  # [B,N]
            s_trunk: torch.Tensor,  # [B,N,C] (Child Query)
            tau: Optional[float] = None,  # Temperature override
            **kwargs
    ):
        B, K, C = s_parent.shape
        N = node_mask.shape[1]
        device, dtype = s_parent.device, s_parent.dtype
        eps = 1e-6  # Numerical safety

        # Extract parent geometry
        mu_p = r_parent.get_gaussian_mean()  # [B,K,3]
        Sig_p = r_parent.get_covariance()  # [B,K,3,3]
        Sig_p = _sym(Sig_p)  # Enforce symmetry

        # ==========================================================
        # 1. Robust Routing (Gumbel-Softmax)
        # ==========================================================
        # Stop gradient on trunk for router stability (optional but rec.)
        s_trunk_detached = s_trunk.detach()

        # Projections
        if self.use_ln:
            q = self.q_proj(self.ln_q(s_trunk_detached))
            k = self.k_proj(self.ln_k(s_parent)) + self.pos_proj(mu_p)
            v = self.v_proj(self.ln_v(s_parent))
        else:
            q = self.q_proj(s_trunk_detached)
            k = self.k_proj(s_parent) + self.pos_proj(mu_p)
            v = self.v_proj(s_parent)

        # Logits [B,N,K]
        # Scale by sqrt(C) to prevent large logits -> nan
        logits = torch.einsum("bnc,bkc->bnk", q, k) / (math.sqrt(C) + eps)

        # Masking (Use large negative number, not -inf)
        logits = logits.masked_fill(mask_parent[:, None, :] < 0.5, -1e4)
        logits = logits.masked_fill(node_mask[:, :, None] < 0.5, -1e4)

        # Gumbel Selection
        # Training: One-hot forward, Soft backward
        # Inference: Hard Argmax
        current_tau = float(self.tau_init if tau is None else tau)

        if self.training:
            w = F.gumbel_softmax(logits, tau=current_tau, hard=True, dim=-1)
        else:
            idx = torch.argmax(logits, dim=-1)
            w = F.one_hot(idx, num_classes=K).to(dtype)

        # Mask padding
        w = w * node_mask[:, :, None]

        # ---- DEBUG stats (cheap) ----
        with torch.no_grad():
            w_flip = self._w_flip_rate(w, node_mask)  # 相邻切换率
            w_ent = self._w_entropy(w, node_mask)  # 平均熵
            w_top1 = self._w_top1_mean(w, node_mask)  # top1均值
            col_stats = self._w_column_stats(w, node_mask)  # 列占用/塌缩


        # ==========================================================
        # 2. Base Selection & Scaled Generation
        # ==========================================================
        # Select Base Parent Info (Anchor)
        mu_base = torch.einsum("bnk,bkd->bnd", w, mu_p)  # [B,N,3]
        s_base = torch.einsum("bnk,bkc->bnc", w, v)  # [B,N,C]
        Sig_base = torch.einsum("bnk,bkij->bnij", w, Sig_p)  # [B,N,3,3]

        # [Key Stability Fix] Use Diagonal Scale instead of Cholesky
        # Extract diagonal elements as approximate scale
        diag_val = torch.diagonal(Sig_base, dim1=-2, dim2=-1)
        scale_base = (diag_val.clamp(min=1e-8)).sqrt()  # [B,N,3]

        # [Key Logic] Scaled Offset Prediction
        # "Who am I" (s_trunk) + "Where is my base" (s_base)
        feat_in = torch.cat([s_trunk_detached, s_base], dim=-1)  # [B,N,2C]

        # Predict relative offset (-1 to 1 via Tanh)
        # raw_offset: [B,N,3]
        raw_offset = self.delta_head(feat_in)

        # Scale to physical world: Offset * ParentSize
        # This gives "Volume" to the generation immediately
        delta_real = raw_offset * self.delta_scale * scale_base

        # Final Position
        mu0 = mu_base + delta_real
        mu0 = mu0 * node_mask[..., None]

        # ==========================================================
        # 3. Sigma Update (Log-Scale)
        # ==========================================================
        # Optional: modify parent shape
        log_s_delta = self.logscale_head(feat_in)  # ~[-1, 1]
        s_delta = torch.exp(log_s_delta * 0.5)  # Gentle scaling

        # Update Sig_base: S * Sigma * S^T
        # Diagonal update for efficiency
        scale_mat = s_delta.unsqueeze(-1) * s_delta.unsqueeze(-2)
        Sig_inherit = Sig_base * scale_mat

        # Blend with Spacing Sigma (Local geometry heuristic)
        Sig_spacing = init_sigma_from_child_spacing(
            mu0, node_mask,
            k_nn=self.fps_knn,
            alpha=self.sigma_cover_alpha,
            sigma_floor=self.sigma_floor,
            sigma_ceil=self.sigma_ceil,
            jitter=max(self.jitter, 1e-6)
        )

        beta = self.beta_parent_sigma
        Sig0 = beta * Sig_inherit + (1.0 - beta) * Sig_spacing

        I = torch.eye(3, device=device, dtype=dtype)[None, None]
        Sig0 = _sym(Sig0) + max(self.jitter, 1e-6) * I
        Sig0 = Sig0 * node_mask[:, :, None, None]

        # ==========================================================
        # 4. Build Rigids & Refine
        # ==========================================================
        # Initialize semantic features from parent selection
        s0 = s_base

        r0 = coarse_rigids_from_mu_sigma(mu0, Sig0, self.OffsetGaussianRigid_cls)

        #
        # ang_rigids = r0.scale_translation(10.0)  # 0.1
        #
        # # pooling 得到 r_c / mask_c 后
        # save_gaussian_as_pdb(
        #     gaussian_rigid=ang_rigids,
        #     filename=f"debug__up_r0.pdb",
        #     mask=node_mask,
        #     center_mode="gaussian_mean",
        # )

        z_c = None
        if self.edge_fusers is not None and (z_parent is not None):
            z_c, _, _ = self.edge_fusers(
                A=w, Z_in=z_parent, r_target=r0,
                mask_f=node_mask, mask_c=mask_parent
            )

        # Run IGA Refine Tower
        s1, r1, z1 = self.refine_tower(s0, r0, node_mask, z_c)


        # ang_rigids = r1.scale_translation(10.0)  # 0.1
        #
        # # pooling 得到 r_c / mask_c 后
        # save_gaussian_as_pdb(
        #     gaussian_rigid=ang_rigids,
        #     filename=f"debug__up_r0.pdb",
        #     mask=node_mask,
        #     center_mode="gaussian_mean",
        # )

        # ==========================================================
        # 5. Anti-Collapse Losses (Vital!)
        # ==========================================================
        reg_total = torch.tensor(0.0, device=device, dtype=dtype)

        # Use Soft w for differentiable loss calculation
        w_soft = F.softmax(logits, dim=-1)

        # A. Load Balance (Force usage of multiple parents)
        usage = w_soft.mean(dim=1)  # [B, K]
        target_usage = torch.ones_like(usage) / max(K, 1)
        loss_bal = F.mse_loss(usage, target_usage)

        # B. Scale Hinge (Force expansion if collapsed)
        # Calculate Gyration Radius of generated points
        center = mu0.mean(dim=1, keepdim=True)
        dist = (mu0 - center).norm(dim=-1)
        mask_sum = node_mask.sum().clamp_min(1.0)
        curr_scale = (dist * node_mask).sum(dim=1) / mask_sum

        # Hinge loss: Penalize if scale < 8.0 Angstrom
        # This forcibly pushes points apart if they are stuck at origin
        loss_scale = F.relu(0.5 - curr_scale).mean()

        # Weighted Sum (Adjust weights as needed)
        # High balance weight to break monopoly
        reg_total = 2.0 * loss_bal + 1.0 * loss_scale

        # Debug info
        parent_idx = torch.argmax(w, dim=-1)


        aux_debug = {
            "w_flip": w_flip,
            "w_entropy": w_ent,
            "w_top1": w_top1,
            **col_stats,
        }

        levels = [{
            "s0": s0,
            "mu0": mu0,
            "Sigma0": Sig0,
            "r0": r0,
            "s": s1,
            "z": z1,
            "r": r1,
            "mask": node_mask,
            "aux": UpAux(w_parent=w, parent_idx=parent_idx,debug=aux_debug)
        }]

        return levels, reg_total

    @staticmethod
    def _w_flip_rate(w: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        相邻残基 parent argmax 切换率。
        w: [B,N,K], node_mask: [B,N] (0/1)
        return: scalar tensor
        """
        # hard assignment
        a = torch.argmax(w, dim=-1)  # [B,N]
        # valid adjacent pairs
        m = (node_mask[:, :-1] > 0.5) & (node_mask[:, 1:] > 0.5)  # [B,N-1]
        diff = (a[:, :-1] != a[:, 1:]) & m
        denom = m.sum().clamp_min(1)
        return diff.sum().float() / denom.float()

    @staticmethod
    def _w_entropy(w: torch.Tensor, node_mask: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """
        平均熵（只统计有效 residue）。
        H(w_i) = -Σ_k w_i,k log w_i,k
        """
        m = (node_mask > 0.5).float()  # [B,N]
        p = w.clamp_min(eps)
        h = -(p * p.log()).sum(dim=-1)  # [B,N]
        denom = m.sum().clamp_min(1.0)
        return (h * m).sum() / denom

    @staticmethod
    def _w_top1_mean(w: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """
        每个 residue 的 top1 概率平均值（衡量 w 是否很尖）。
        """
        m = (node_mask > 0.5).float()
        top1 = w.max(dim=-1).values  # [B,N]
        denom = m.sum().clamp_min(1.0)
        return (top1 * m).sum() / denom

    @staticmethod
    def _w_column_stats(w: torch.Tensor, node_mask: torch.Tensor) -> dict:
        """
        统计列占用情况：w_sum[k] = Σ_i w[i,k]
        返回几个标量：有效 parent 数、top1列占比、Gini-like 等
        """
        # w_sum: [B,K]
        m = (node_mask > 0.5).float().unsqueeze(-1)  # [B,N,1]
        w_sum = (w * m).sum(dim=1)  # [B,K]
        # normalize per batch for interpretability
        w_norm = w_sum / w_sum.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # [B,K]

        # 有效 parent 数：权重大于阈值的列数（阈值按 1/K 的比例）
        B, K = w_norm.shape
        thr = (1.0 / max(K, 1)) * 0.5
        active = (w_norm > thr).sum(dim=-1).float().mean()  # scalar

        # 最大列占比（越大越塌缩）
        top_col = w_norm.max(dim=-1).values.mean()

        # 简单“集中度”指标：sum(p^2)（越大越集中）
        conc = (w_norm ** 2).sum(dim=-1).mean()

        return {
            "w_active_cols": active,   # 越大越分散
            "w_top_col": top_col,      # 越大越塌缩
            "w_col_conc": conc,        # 越大越塌缩
        }





# =========================
# 3) 从 r_parent 抽取 (mu, R, s)
# =========================
def extract_parent_params(r_parent, eps: float = 1e-8):
    """
    r_parent: OffsetGaussianRigid [B,K]
    期望它有:
      - get_gaussian_mean() -> [B,K,3]   (更通用)
      - get_rots().get_rot_mats() -> [B,K,3,3]
      - scaling 或 _scaling_log -> [B,K,3]
    """
    # mu
    if hasattr(r_parent, "get_gaussian_mean"):
        mu = r_parent.get_gaussian_mean()
    elif hasattr(r_parent, "get_trans"):
        mu = r_parent.get_trans()
    else:
        raise ValueError("r_parent needs get_gaussian_mean() or get_trans().")

    # R
    if hasattr(r_parent, "get_rots"):
        rots = r_parent.get_rots()
        if hasattr(rots, "get_rot_mats"):
            R = rots.get_rot_mats()
        else:
            raise ValueError("r_parent.get_rots() must provide get_rot_mats().")
    else:
        raise ValueError("r_parent needs get_rots().get_rot_mats().")

    # s (diag scale)
    if hasattr(r_parent, "scaling"):
        s = r_parent.scaling
    elif hasattr(r_parent, "_scaling_log"):
        s = torch.exp(r_parent._scaling_log)
    else:
        # 退化：用单位尺度
        s = torch.ones((*mu.shape[:-1], 3), device=mu.device, dtype=mu.dtype)

    s = s.clamp_min(eps)
    return mu, R, s


# =========================
# 4) K->N broadcast + 预测 xi_hat + decode x_hat
# =========================
class UpXiPredictor(nn.Module):
    """
    核心：
      - gather s_parent 到每个 residue：s_pi = s_parent[a_idx]
      - query 可以是 (s_pi + pos_emb) 或 (s_trunk + s_pi + pos_emb)
      - 输出 xi_hat [B,N,3]
      - 用 parent 的 (mu,R,s) decode 回 x_hat
    """
    def __init__(
        self,
        c_s: int,
        n_freq: int = 16,
        pos_weight: float = 1.0,
        use_trunk_query: bool = False,  # 你说“query 可用 s_trunk”，就开这个
        mlp_hidden: int = 256,
    ):
        super().__init__()
        self.c_s = int(c_s)
        self.pos_weight = float(pos_weight)
        self.use_trunk_query = bool(use_trunk_query)

        self.pos_enc = Pos01FourierEncoder(c_s=self.c_s, n_freq=n_freq)

        # 把输入 query 映射到统一维度
        in_dim = self.c_s * (2 if self.use_trunk_query else 1)  # [s_pi] (+ [s_trunk])
        self.q_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.c_s),
            nn.SiLU(),
            nn.Linear(self.c_s, self.c_s),
        )

        # 输出 xi_hat
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.c_s),
            nn.Linear(self.c_s, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 3),
        )

    def forward(
        self,
        s_parent: torch.Tensor,     # [B,K,C]
        r_parent,                  # OffsetGaussianRigid [B,K]
        a_idx: torch.Tensor,        # [B,N] long
        node_mask: torch.Tensor,    # [B,N] 0/1
        pos01: torch.Tensor = None, # [B,N] float in [0,1] (可选；没给就现场算)
        s_trunk: torch.Tensor = None,  # [B,N,C] (可选；use_trunk_query=True时用)
    ):
        B, K, C = s_parent.shape
        N = a_idx.shape[1]
        device = s_parent.device
        dtype = s_parent.dtype

        # pos01
        if pos01 is None:
            # 需要 Kmax 来算 seg_len，但 pos01 只用到 a_idx/node_mask
            # 这里用 K 来当 Kmax 即可（你一般 K=Kmax 或 mask_parent 后的 Kmax）
            pos01, _ = segment_pos01_from_assignment(a_idx, node_mask, Kmax=K)
        pos01 = pos01.to(device=device, dtype=torch.float32)

        # pos embedding
        pos_emb = self.pos_enc(pos01, node_mask).to(dtype=dtype)  # [B,N,C]

        # gather s_parent -> residue
        idx = a_idx.clamp_min(0).clamp_max(K - 1)
        s_pi = s_parent.gather(1, idx[..., None].expand(B, N, C))  # [B,N,C]
        s_pi = s_pi + self.pos_weight * pos_emb

        # build query
        if self.use_trunk_query:
            assert s_trunk is not None, "use_trunk_query=True requires s_trunk."
            q_in = torch.cat([s_trunk, s_pi], dim=-1)  # [B,N,2C]
        else:
            q_in = s_pi  # [B,N,C]

        q = self.q_proj(q_in) * node_mask[..., None].to(dtype=dtype)  # [B,N,C]

        # xi_hat
        scale = 1.5
        xi_hat = self.mlp(q) * node_mask[..., None].to(dtype=dtype)   # [B,N,3]
        xi_hat=torch.tanh(xi_hat)*scale

        # decode x_hat
        mu_k, R_k, s_k = extract_parent_params(r_parent)  # [B,K,3], [B,K,3,3], [B,K,3]
        mu_i = mu_k.gather(1, idx[..., None].expand(B, N, 3))                       # [B,N,3]
        s_i  = s_k.gather(1, idx[..., None].expand(B, N, 3))                        # [B,N,3]
        R_i  = R_k.gather(1, idx[..., None, None].expand(B, N, 3, 3))               # [B,N,3,3]

        G_n=OffsetGaussianRigid( rots=Rotation(R_i), trans=mu_i,  scaling_log=torch.log(s_i), local_mean=torch.zeros_like(mu_i) )
        x_hat=G_n.apply(xi_hat* s_i)

        #
        # local = xi_hat * s_i
        # x_hat = mu_i + torch.einsum("bnij,bnj->bni", R_i, local)
        x_hat = x_hat * node_mask[..., None].to(dtype=dtype)

        return {
            "xi_hat": xi_hat,   # [B,N,3]
            "x_hat": x_hat*10,     # [B,N,3]
            "pos01": pos01,     # [B,N]
            "r_parent":r_parent.scale_translation(10),
            "r_child": G_n.scale_translation(10),
            "a_idx":a_idx
        }



# 你已有：Pos01FourierEncoder, OffsetGaussianRigid, Rotation, extract_parent_params
# - Pos01FourierEncoder(c_s, n_freq)
# - extract_parent_params(r_parent) -> (mu_k [B,K,3], R_k [B,K,3,3], s_k [B,K,3])



class UpXiPredictorAsoftPos(nn.Module):
    """
    Upsample without a_idx.
    Use A_soft -> router w, compute soft pos01 from w, inject pos emb, predict xi_hat, decode with parent frame.

    Inputs:
      s_parent:   [B,K,C]
      r_parent:   OffsetGaussianRigid [B,K]
      A_soft:     [B,N,K]
      mask_parent:[B,K] 0/1
      node_mask:  [B,N] 0/1
      s_trunk:    [B,N,C] optional

    Outputs:
      xi_hat, x_hat, pos01, w, parent_idx, r_child(optional)
    """

    def __init__(
        self,
        c_s: int,
        n_freq: int = 9,
        pos_weight: float = 1.0,
        use_trunk_query: bool = False,
        mlp_hidden: int = 256,
        xi_tanh_scale: float = 1.5,

        # routing/geometry options
        hard_forward: bool = True,      # ✅ 推荐：hard forward + soft backward
        geom_soft_mu_s: bool = True,    # ✅ 推荐：mu,s 用 w(可导)；R 用 argmax(稳定)
        detach_pos01: bool = False,     # pos01 是否 stop-grad（一般不需要，但可开）
    ):
        super().__init__()
        self.c_s = int(c_s)
        self.pos_weight = float(pos_weight)
        self.use_trunk_query = bool(use_trunk_query)
        self.xi_tanh_scale = float(xi_tanh_scale)

        self.hard_forward = bool(hard_forward)
        self.geom_soft_mu_s = bool(geom_soft_mu_s)
        self.detach_pos01 = bool(detach_pos01)

        # 你已有的 Pos01FourierEncoder
        self.pos_enc = Pos01FourierEncoder(c_s=self.c_s, n_freq=n_freq)

        in_dim = self.c_s * (2 if self.use_trunk_query else 1)  # [s_pi] (+ [s_trunk])
        self.q_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.c_s),
            nn.SiLU(),
            nn.Linear(self.c_s, self.c_s),
        )

        self.mlp = nn.Sequential(
            nn.LayerNorm(self.c_s),
            nn.Linear(self.c_s, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 3),
        )

    @staticmethod
    def _row_normalize(w: torch.Tensor, eps: float = 1e-8):
        return w / w.sum(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def build_pos01_from_router(w: torch.Tensor, node_mask: torch.Tensor, eps: float = 1e-8):
        """
        可导的“段内位置”构造（关键）：
        对每个 parent k：
          cum_k[i] = Σ_{t<=i} w[t,k]
          total_k  = Σ_{t}    w[t,k]
          pos_{i,k} = cum_k[i] / total_k   ∈ [0,1]
        然后对 residue i 做混合：
          pos01[i] = Σ_k w[i,k] * pos_{i,k}
        这样得到一个 [B,N] 的连续 pos01，且天然随 sequence 单调推进（对角块倾向更强）。
        """
        # w: [B,N,K] rows already sum to 1 on valid residues
        m = node_mask.float().unsqueeze(-1)  # [B,N,1]
        w = w * m

        # cum along N
        cum = torch.cumsum(w, dim=1)  # [B,N,K]
        total = w.sum(dim=1, keepdim=True).clamp_min(eps)  # [B,1,K]
        pos_k = cum / total  # [B,N,K] in [0,1]

        pos01 = (w * pos_k).sum(dim=-1)  # [B,N]
        pos01 = pos01 * node_mask.float()
        return pos01

    def forward(
        self,
        s_parent: torch.Tensor,      # [B,K,C]
        r_parent,                    # OffsetGaussianRigid [B,K]
        A_soft: torch.Tensor,        # [B,N,K]
        mask_parent: torch.Tensor,   # [B,K] 0/1
        node_mask: torch.Tensor,     # [B,N] 0/1
        s_trunk: torch.Tensor = None,# [B,N,C]
        eps: float = 1e-8,
        ang_scale: float = 10.0,
    ):
        B, K, C = s_parent.shape
        _, N, K2 = A_soft.shape
        assert K2 == K, f"A_soft last dim {K2} must match K {K}"

        dtype = s_parent.dtype
        device = s_parent.device

        mN = node_mask.to(dtype=dtype)[:, :, None]        # [B,N,1]
        mK = mask_parent.to(dtype=dtype)[:, None, :]      # [B,1,K]

        # ------------------------------------
        # 1) router w from A_soft
        # ------------------------------------
        w = A_soft.to(dtype=dtype) * mN * mK              # [B,N,K]
        w = self._row_normalize(w, eps=eps) * mN          # padded rows -> 0

        # ST-hard forward (avoid averaging collapse), keep gradient through w
        parent_idx = torch.argmax(w, dim=-1)              # [B,N] (no grad)
        if self.hard_forward:
            w_hard = F.one_hot(parent_idx, num_classes=K).to(dtype=dtype) * mN
            w_use = w_hard - w.detach() + w              # straight-through
        else:
            w_use = w
        # viz_w(w, node_mask, w_use=w_use, title="router_w", b=0, k_max_show=128,save_path='hard.png')
        # ------------------------------------
        # 2) pos01 from router (可导)
        # ------------------------------------
        pos01 = self.build_pos01_from_router(w, node_mask, eps=eps)  # [B,N] float
        if self.detach_pos01:
            pos01_in = pos01.detach()
        else:
            pos01_in = pos01
        pos01_in = pos01_in.to(device=device, dtype=torch.float32)

        pos_emb = self.pos_enc(pos01_in, node_mask).to(dtype=dtype)  # [B,N,C]

        # ------------------------------------
        # 3) semantic mix + pos emb
        # ------------------------------------
        s_pi = torch.einsum("bnk,bkc->bnc", w_use, s_parent)          # [B,N,C]
        s_pi = (s_pi + self.pos_weight * pos_emb) * node_mask[..., None].to(dtype=dtype)

        if self.use_trunk_query:
            assert s_trunk is not None, "use_trunk_query=True requires s_trunk."
            q_in = torch.cat([s_trunk, s_pi], dim=-1)                # [B,N,2C]
        else:
            q_in = s_pi                                              # [B,N,C]

        q = self.q_proj(q_in) * node_mask[..., None].to(dtype=dtype)  # [B,N,C]

        # ------------------------------------
        # 4) xi_hat
        # ------------------------------------
        xi_hat = self.mlp(q) * node_mask[..., None].to(dtype=dtype)   # [B,N,3]
        xi_hat = torch.tanh(xi_hat) * self.xi_tanh_scale

        # ------------------------------------
        # 5) decode geometry
        # ------------------------------------
        mu_k, R_k, s_k = extract_parent_params(r_parent)              # mu:[B,K,3], R:[B,K,3,3], s:[B,K,3]

        # R 用 argmax（稳定）
        idx = parent_idx.clamp(0, K - 1)
        R_i = R_k.gather(1, idx[..., None, None].expand(B, N, 3, 3))  # [B,N,3,3]

        if self.geom_soft_mu_s:
            # ✅ mu,s 用 w_use（可导或ST可导）
            mu_i = torch.einsum("bnk,bkd->bnd", w_use, mu_k)          # [B,N,3]
            s_i  = torch.einsum("bnk,bkd->bnd", w_use, s_k)           # [B,N,3]
        else:
            mu_i = mu_k.gather(1, idx[..., None].expand(B, N, 3))
            s_i  = s_k.gather(1, idx[..., None].expand(B, N, 3))

        s_i = s_i.clamp_min(1e-6)
        local = xi_hat * s_i
        x_hat = mu_i + torch.einsum("bnij,bnj->bni", R_i, local)
        x_hat = x_hat * node_mask[..., None].to(dtype=dtype)

        # 如果你仍想输出 child 的 rigid（用于 debug 存 pdb）
        # 注意：这里的 child 旋转就是 R_i，尺度 s_i，trans mu_i
        G_child = OffsetGaussianRigid(
            rots=Rotation(rot_mats=R_i),
            trans=mu_i,
            scaling_log=torch.log(s_i),
            local_mean=torch.zeros_like(mu_i),
        )

        return {
            'a_idx': torch.argmax(w_use, dim=-1) ,
            "w": w,                        # [B,N,K] soft router
            "w_use": w_use,                # [B,N,K] used router (ST-hard or soft)
            "parent_idx": parent_idx,      # [B,N]
            "pos01": pos01,                # [B,N] (soft segment position)
            "xi_hat": xi_hat,              # [B,N,3]
            "x_hat": x_hat * ang_scale,    # [B,N,3]
            "r_parent": r_parent.scale_translation(ang_scale),
            "r_child": G_child.scale_translation(ang_scale),}

class UpXiPredictorAttnPos_Query(nn.Module):
    """
    Upsample without a_idx.
    Router = learned cross-attention from trunk->parents, optionally with A_soft prior.
    Then build pos01 from router w, inject pos emb, predict xi_hat, decode with parent frame.

    Inputs:
      s_parent:   [B,K,C]
      r_parent:   OffsetGaussianRigid [B,K]
      node_mask:  [B,N] 0/1
      mask_parent:[B,K] 0/1
      s_trunk:    [B,N,C]  (recommended)
      A_soft:     [B,N,K] optional (recommended as prior / band constraint)

    Outputs:
      w_soft, w_use, parent_idx, pos01, xi_hat, x_hat, r_child(optional)
    """

    def __init__(
            self,
            c_s: int,
            n_freq: int = 9,
            pos_weight: float = 1.0,
            use_trunk_query: bool = False,
            mlp_hidden: int = 256,
            xi_tanh_scale: float = 1.5,

            # router knobs
            tau: float = 1.0,  # softmax temperature
            use_ln: bool = True,  # layernorm for router stability
            use_mu_bias: bool = True,  # add parent mu_k positional bias into keys
            prior_strength: float = 2.0,  # beta in logits += beta*log(A_soft)
            prior_floor: float = 1e-6,  # avoid log(0)

            # routing/geometry options
            hard_forward: bool = True,  # ST-hard forward (recommended)
            geom_soft_mu_s: bool = True,  # mu,s via w_use; R via argmax
            detach_pos01: bool = False,  # stop-grad on pos01 if needed
    ):
        super().__init__()
        self.c_s = int(c_s)
        self.pos_weight = float(pos_weight)
        self.use_trunk_query = bool(use_trunk_query)
        self.xi_tanh_scale = float(xi_tanh_scale)

        self.tau = float(tau)
        self.use_ln = bool(use_ln)
        self.use_mu_bias = bool(use_mu_bias)
        self.prior_strength = float(prior_strength)
        self.prior_floor = float(prior_floor)

        self.hard_forward = bool(hard_forward)
        self.geom_soft_mu_s = bool(geom_soft_mu_s)
        self.detach_pos01 = bool(detach_pos01)

        # ----- pos encoding (same as your AsoftPos) -----
        self.pos_enc = Pos01FourierEncoder(c_s=self.c_s, n_freq=n_freq)

        # ----- router projections -----
        if self.use_ln:
            self.ln_q = nn.LayerNorm(self.c_s)
            self.ln_k = nn.LayerNorm(self.c_s)
            self.ln_v = nn.LayerNorm(self.c_s)

        self.q_proj = nn.Linear(self.c_s, self.c_s, bias=False)
        self.k_proj = nn.Linear(self.c_s, self.c_s, bias=False)
        self.v_proj = nn.Linear(self.c_s, self.c_s, bias=False)
        self.mu_proj = nn.Linear(3, self.c_s, bias=False)  # parent mu bias

        # ----- xi predictor -----
        in_dim = self.c_s * 2 if self.use_trunk_query else self.c_s
        self.fuse = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.c_s),
            nn.SiLU(),
            nn.Linear(self.c_s, self.c_s),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.c_s),
            nn.Linear(self.c_s, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 3),
        )

    @staticmethod
    def _row_normalize(w: torch.Tensor, eps: float = 1e-8):
        return w / w.sum(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def build_pos01_from_router(w: torch.Tensor, node_mask: torch.Tensor, eps: float = 1e-8):
        """
        Same as your AsoftPos: pos01 from cumulative mass inside each parent column, then mix by w.
        """
        m = node_mask.float().unsqueeze(-1)  # [B,N,1]
        w = w * m

        cum = torch.cumsum(w, dim=1)  # [B,N,K]
        total = w.sum(dim=1, keepdim=True).clamp_min(eps)  # [B,1,K]
        pos_k = cum / total  # [B,N,K] in [0,1]

        pos01 = (w * pos_k).sum(dim=-1)  # [B,N]
        return pos01 * node_mask.float()

    def forward(
            self,
            s_parent: torch.Tensor,  # [B,K,C]
            r_parent,  # OffsetGaussianRigid [B,K]
            node_mask: torch.Tensor,  # [B,N]
            mask_parent: torch.Tensor,  # [B,K]
            s_trunk: torch.Tensor,  # [B,N,C]
            A_soft: torch.Tensor = None,  # [B,N,K] optional prior
            eps: float = 1e-8,
            ang_scale: float = 10.0,
    ):
        B, K, C = s_parent.shape
        B2, N, C2 = s_trunk.shape
        assert B2 == B and C2 == C, "s_trunk must be [B,N,C] matching s_parent C"
        dtype, device = s_parent.dtype, s_parent.device

        mN = node_mask.to(dtype=dtype)[:, :, None]  # [B,N,1]
        mK = mask_parent.to(dtype=dtype)[:, None, :]  # [B,1,K]

        # ------------------------------------------------
        # 1) Router: cross-attn logits [B,N,K]
        # ------------------------------------------------
        # parent geometry bias
        mu_k, R_k, s_k = extract_parent_params(r_parent)  # mu:[B,K,3], R:[B,K,3,3], s:[B,K,3]

        if self.use_ln:
            q_in = self.ln_q(s_trunk)
            k_in = self.ln_k(s_parent)
            v_in = self.ln_v(s_parent)
        else:
            q_in, k_in, v_in = s_trunk, s_parent, s_parent

        q = self.q_proj(q_in)  # [B,N,C]
        k = self.k_proj(k_in)  # [B,K,C]
        v = self.v_proj(v_in)  # [B,K,C]

        if self.use_mu_bias:
            k = k + self.mu_proj(mu_k)  # [B,K,C]

        logits = torch.einsum("bnc,bkc->bnk", q, k) / math.sqrt(max(C, 1))
        logits = logits / max(self.tau, 1e-6)

        # mask invalid parents/residues
        logits = logits.masked_fill(mask_parent[:, None, :] < 0.5, -1e4)
        logits = logits.masked_fill(node_mask[:, :, None] < 0.5, -1e4)

        # --- optional prior from A_soft (VERY recommended) ---
        # This is the key to avoid "拐弯/跨段": constrain router to stay near the segmentation band.
        if A_soft is not None:
            assert A_soft.shape == (B, N, K), f"A_soft must be [B,N,K], got {A_soft.shape}"
            prior = (A_soft.to(dtype=dtype) * mN * mK).clamp_min(self.prior_floor)
            logits = logits + self.prior_strength * prior.log()

        # soft router
        w_soft = F.softmax(logits, dim=-1) * mN  # [B,N,K], padded rows->0
        w_soft = self._row_normalize(w_soft, eps=eps) * mN

        # ST-hard forward
        parent_idx = torch.argmax(w_soft, dim=-1)  # [B,N] (no grad)
        if self.hard_forward:
            w_hard = F.one_hot(parent_idx, num_classes=K).to(dtype=dtype) * mN
            w_use = w_hard - w_soft.detach() + w_soft
        else:
            w_use = w_soft

        # ------------------------------------------------
        # 2) pos01 from router (differentiable)
        # ------------------------------------------------
        pos01 = self.build_pos01_from_router(w_soft, node_mask, eps=eps)  # [B,N]
        pos01_in = pos01.detach() if self.detach_pos01 else pos01
        pos_emb = self.pos_enc(pos01_in.to(device=device, dtype=torch.float32), node_mask).to(dtype=dtype)

        # ------------------------------------------------
        # 3) semantic mix + pos emb
        # ------------------------------------------------
        s_pi = torch.einsum("bnk,bkc->bnc", w_use, s_parent)  # [B,N,C]
        s_pi = (s_pi + self.pos_weight * pos_emb) * node_mask[..., None].to(dtype=dtype)

        if self.use_trunk_query:
            q_in2 = torch.cat([s_trunk, s_pi], dim=-1)  # [B,N,2C]
        else:
            q_in2 = s_pi  # [B,N,C]

        feat = self.fuse(q_in2) * node_mask[..., None].to(dtype=dtype)

        # ------------------------------------------------
        # 4) xi_hat
        # ------------------------------------------------
        xi_hat = self.mlp(feat) * node_mask[..., None].to(dtype=dtype)
        xi_hat = torch.tanh(xi_hat) * self.xi_tanh_scale

        # ------------------------------------------------
        # 5) decode geometry (self-consistent)
        # ------------------------------------------------
        # R: use argmax for orthogonality stability
        idx = parent_idx.clamp(0, K - 1)
        R_i = R_k.gather(1, idx[..., None, None].expand(B, N, 3, 3))  # [B,N,3,3]

        if self.geom_soft_mu_s:
            mu_i = torch.einsum("bnk,bkd->bnd", w_use, mu_k)  # [B,N,3]
            s_i = torch.einsum("bnk,bkd->bnd", w_use, s_k)  # [B,N,3]
        else:
            mu_i = mu_k.gather(1, idx[..., None].expand(B, N, 3))
            s_i = s_k.gather(1, idx[..., None].expand(B, N, 3))

        s_i = s_i.clamp_min(1e-6)
        local = xi_hat * s_i
        x_hat = mu_i + torch.einsum("bnij,bnj->bni", R_i, local)
        x_hat = x_hat * node_mask[..., None].to(dtype=dtype)

        G_child = OffsetGaussianRigid(
            rots=Rotation(rot_mats=R_i),
            trans=mu_i,
            scaling_log=torch.log(s_i),
            local_mean=torch.zeros_like(mu_i),
        )

        return {
            # hard index only for logging/debug
            "a_idx": idx,
            "w": w_soft,  # [B,N,K] (soft, differentiable)
            "w_use": w_use,  # [B,N,K] (ST-hard forward)
            "parent_idx": parent_idx,  # [B,N]
            "pos01": pos01,  # [B,N]
            "xi_hat": xi_hat,  # [B,N,3]
            "x_hat": x_hat * ang_scale,  # [B,N,3]
            "r_parent": r_parent.scale_translation(ang_scale),
            "r_child": G_child.scale_translation(ang_scale),
        }












@torch.no_grad()
def viz_w(
    w: torch.Tensor,              # [B,N,K]
    node_mask: torch.Tensor,      # [B,N]
    w_use: torch.Tensor = None,   # [B,N,K] optional
    title: str = "w",
    b: int = 0,
    k_max_show: int | None = None,  # e.g. 128 to avoid ultra-wide
    save_path: str | None = None,   # if None -> plt.show()
):
    assert w.dim() == 3
    B, N, K = w.shape
    assert 0 <= b < B

    # valid N
    n_valid = int((node_mask[b] > 0.5).sum().item())
    n_valid = max(n_valid, 1)

    # slice
    w_b = w[b, :n_valid]  # [n_valid, K]
    if k_max_show is not None:
        K_show = min(int(k_max_show), K)
        w_b = w_b[:, :K_show]
    else:
        K_show = K

    # to numpy
    W = w_b.detach().float().cpu().numpy()  # [n_valid, K_show]
    a = np.argmax(W, axis=1)                # [n_valid]

    fig, ax = plt.subplots(figsize=(max(6, K_show * 0.05), max(4, n_valid * 0.03)))

    im = ax.imshow(
        W,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
    )
    ax.plot(a, np.arange(n_valid), linewidth=1.0)  # argmax path

    ax.set_xlabel("parent k")
    ax.set_ylabel("residue i")
    ax.set_title(f"{title} | b={b} | N_valid={n_valid} | K_show={K_show}")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    # optional: compare w_use
    if w_use is not None:
        wuse_b = w_use[b, :n_valid, :K_show].detach().float().cpu().numpy()
        a2 = np.argmax(wuse_b, axis=1)

        fig2, ax2 = plt.subplots(figsize=(max(6, K_show * 0.05), max(4, n_valid * 0.03)))
        im2 = ax2.imshow(
            wuse_b,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
        )
        ax2.plot(a2, np.arange(n_valid), linewidth=1.0)
        ax2.set_xlabel("parent k")
        ax2.set_ylabel("residue i")
        ax2.set_title(f"{title} (w_use) | b={b} | N_valid={n_valid} | K_show={K_show}")
        fig2.colorbar(im2, ax=ax2, fraction=0.02, pad=0.02)

    if save_path is not None:
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        if w_use is not None:
            base, ext = os.path.splitext(save_path)
            fig2.savefig(base + "_wuse" + ext, dpi=180, bbox_inches="tight")
        plt.close(fig)
        if w_use is not None:
            plt.close(fig2)
    else:
        plt.show()

# =========================
# 5) 极简用法示例
# =========================
if __name__ == "__main__":
    B, N, K, C = 2, 240, 64, 128
    s_parent = torch.randn(B, K, C)
    node_mask = torch.ones(B, N)
    a_idx = torch.randint(0, K, (B, N))

    # 这里用 dummy r_parent：你实际用 OffsetGaussianRigid [B,K]
    # 为了演示，这里造一个最小替身对象
    class DummyR:
        def __init__(self, mu, R, s):
            self._mu = mu
            self._R = R
            self._s = s
        def get_gaussian_mean(self): return self._mu
        class _Rots:
            def __init__(self, R): self._R = R
            def get_rot_mats(self): return self._R
        def get_rots(self): return DummyR._Rots(self._R)
        @property
        def scaling(self): return self._s

    mu = torch.randn(B, K, 3)
    # 先用单位旋转
    R = torch.eye(3).view(1,1,3,3).expand(B, K, 3, 3).contiguous()
    s = torch.ones(B, K, 3) * 0.2
    r_parent = DummyR(mu, R, s)

    # (A) 严格 K->N
    up = UpXiPredictor(c_s=C, use_trunk_query=False)
    out = up(s_parent=s_parent, r_parent=r_parent, a_idx=a_idx, node_mask=node_mask)
    print(out["xi_hat"].shape, out["x_hat"].shape)

    # (B) trunk query（你允许的版本）
    s_trunk = torch.randn(B, N, C)
    up2 = UpXiPredictor(c_s=C, use_trunk_query=True)
    out2 = up2(s_parent=s_parent, r_parent=r_parent, a_idx=a_idx, node_mask=node_mask, s_trunk=s_trunk)
    print(out2["xi_hat"].shape, out2["x_hat"].shape)


