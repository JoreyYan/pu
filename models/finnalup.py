import torch
import torch.nn as nn
import torch.nn.functional as F

# 你已有：
# - InvariantGaussianAttention (from models.IGA import InvariantGaussianAttention)
# - StructureModuleTransition
# - GaussianUpdateBlock
# - coarse_rigids_from_mu_sigma(mu, Sigma, OffsetGaussianRigid_cls)
# - fused_gaussian_overlap_score(delta, sigma)


from models.pool import coarse_rigids_from_mu_sigma
from models.IGA import InvariantGaussianAttention,CoarseIGATower,GaussianUpdateBlock,fused_gaussian_overlap_score
from chroma.layers.basic import FourierFeaturization


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
