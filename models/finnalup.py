
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
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

        ang_rigids = r0.scale_translation(10.0)  # 0.1

        # pooling 得到 r_c / mask_c 后
        save_gaussian_as_pdb(
            gaussian_rigid=ang_rigids,
            filename=f"debug__up_r0.pdb",
            mask=node_mask,
            center_mode="gaussian_mean",
        )
        # ---- refine ----
        s1, r1,z1 = self.refine_tower(s0, r0, node_mask,z_c)

        ang_rigids = r1.scale_translation(10.0)  # 0.1

        # pooling 得到 r_c / mask_c 后
        save_gaussian_as_pdb(
            gaussian_rigid=ang_rigids,
            filename=f"debug__up_r1.pdb",
            mask=node_mask,
            center_mode="gaussian_mean",
        )


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
    pi: torch.Tensor        # [B,K]
    parent_idx: torch.Tensor  # [B,N]


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
