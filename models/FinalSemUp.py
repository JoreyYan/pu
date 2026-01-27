import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
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

# 你工程里已有的 Rotation/Rigid/OffsetGaussianRigid 的来源我不知道你具体放哪
# 下面两个函数你按你现有的 Rotation/Rigid API 改一下即可（通常都有 identity/eye 构造）
def _identity_rots(B, N, device, dtype, RotationCls):
    # RotationCls 应该能用 rot_mats 或 quats 构造 identity
    # 1) 如果你有 Rotation.identity([B,N]) 就直接用它
    if hasattr(RotationCls, "identity"):
        return RotationCls.identity((B, N), device=device, dtype=dtype)
    # 2) 否则用 rot_mats=I
    I = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(B, N, 3, 3)
    return RotationCls(rot_mats=I)  # 你按你的 Rotation 构造函数签名改

def _build_unit_offset_gaussian_rigid(B, N, device, dtype, OffsetGaussianRigid_cls, RotationCls):
    rots = _identity_rots(B, N, device, dtype, RotationCls)
    trans = torch.zeros((B, N, 3), device=device, dtype=dtype)
    local_mean = torch.zeros((B, N, 3), device=device, dtype=dtype)
    scaling_log = torch.zeros((B, N, 3), device=device, dtype=dtype)  # exp(0)=1
    return OffsetGaussianRigid_cls(rots, trans, scaling_log, local_mean)


class FinalCoarseToFineSemanticUpModule(nn.Module):
    """
    语义版 coarse(K) -> fine(N) up：
      - 仅用 res_idx / mask / (可选 chain id) 产生 query
      - 对 coarse token 做 cross-attn 得到 B_local，再 lift 出 residue-level s0
      - 不做几何 moment、不做几何 loss、不跑 IGA
      - 只初始化一个“归一偏心高斯” r0，留给后续 finalup(IGA) 再更新

    Inputs:
      s_parent: [B, K, C]
      mask_parent: [B, K]
      node_mask: [B, N]
      res_idx: [B, N]   (真实序号 or 位置 index)
      (optional) chain_id: [B, N]  (如果你有多链想加区分，可以拼进去)

    Outputs:
      s_fine: [B, N, C]
      r0:     OffsetGaussianRigid [B, N]  (unit isotropic init)
      mask0:  [B, N]
      aux:    {B_local, parent_idx}
    """

    def __init__(
        self,
            conf,
        c_s: int,
        OffsetGaussianRigid_cls,
        RotationCls,                 # 你工程里 Rotation 的类
        num_tf_layers: int = 2,
        neighbor_R: int = 2,         # 每个 residue topR 个父节点（语义 topk）
        dropout: float = 0.0,
        max_res_idx: int = 4096,
        use_chain_emb: bool = False,
        max_chain_id: int = 8,
    ):
        super().__init__()
        self.c_s = c_s
        self.OffsetGaussianRigid_cls = OffsetGaussianRigid_cls
        self.RotationCls = RotationCls
        self.neighbor_R = int(neighbor_R)

        # ---- query from (res_idx, optional chain) ----
        self.res_idx_emb = ResIdxFourierEmbedding(
            c_s=c_s,
            scale=1.0,  # 常用 0.5~2.0，可调
            trainable=False,  # 建议先 False，稳定
            use_chain_emb=use_chain_emb,
            max_chain_id=max_chain_id,
        )

        self.use_chain_emb = bool(use_chain_emb)
        if self.use_chain_emb:
            self.chain_emb = nn.Embedding(max_chain_id, c_s)

        self.q_proj = nn.Linear(c_s, c_s, bias=False)
        self.k_proj = nn.Linear(c_s, c_s, bias=False)
        self.v_proj = nn.Linear(c_s, c_s, bias=False)

        # ---- optional: light transformer refine on s_fine (fast) ----
        # enc_layer = nn.TransformerEncoderLayer(
        #     d_model=c_s, nhead=8, dim_feedforward=4 * c_s,
        #     dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        # )
        # self.tf = nn.TransformerEncoder(enc_layer, num_layers=num_tf_layers) if num_tf_layers > 0 else None


        # residue refine tower
        iga = InvariantGaussianAttention(
            c_s=c_s,
            c_z=getattr(conf, "hgfc_z", 0),
            c_hidden=conf.c_hidden,
            no_heads=conf.no_heads,
            no_qk_gaussians=conf.no_qk_points,
            no_v_points=conf.no_v_points,
            layer_idx=9000,
            enable_vis=False,
        )

        gau_update = GaussianUpdateBlock(c_s)
        self.refine_tower = CoarseIGATower(
            iga=iga,

            gau_update=gau_update,
            c_s=c_s,
            num_layers=num_tf_layers,
        )


        self.out_ln = nn.LayerNorm(c_s)




    def forward(self, s_parent, mask_parent, node_mask, res_idx, chain_id=None):
        B, K, C = s_parent.shape
        N = node_mask.shape[1]
        device, dtype = s_parent.device, s_parent.dtype

        mask_p = mask_parent.float()
        mask0 = node_mask.float()

        # ---- build query ----
        q0 = self.res_idx_emb(res_idx=res_idx, node_mask=node_mask, chain_id=chain_id)  # [B,N,C]

        if self.use_chain_emb:
            assert chain_id is not None, "use_chain_emb=True requires chain_id"
            cid = chain_id.clamp_min(0).clamp_max(self.chain_emb.num_embeddings - 1)
            q0 = q0 + self.chain_emb(cid)

        q = self.q_proj(q0)                 # [B,N,C]
        k = self.k_proj(s_parent)           # [B,K,C]
        v = self.v_proj(s_parent)           # [B,K,C]

        # ---- full logits over parents ----
        logits_full = torch.einsum("bnc,bkc->bnk", q, k) / math.sqrt(C)  # [B,N,K]
        logits_full = logits_full + (mask_p[:, None, :] - 1.0) * 1e9     # mask invalid parents
        logits_full = logits_full + (mask0[:, :, None] - 1.0) * 1e9      # mask invalid residues

        # ---- pick topR parents per residue ----
        R = min(self.neighbor_R, K)
        top_val, parent_idx = torch.topk(logits_full, k=R, dim=-1)       # [B,N,R]
        B_local = F.softmax(top_val, dim=-1)                             # [B,N,R]
        B_local = B_local * mask0[:, :, None]

        # ---- gather v_sub and lift ----
        # v_sub: [B,N,R,C]
        v_sub = v[:, None, :, :].expand(B, N, K, C).gather(
            2, parent_idx[..., None].expand(B, N, R, C)
        )
        # 1) 先得到 s0（你原来 lift 的结果）
        s0 = torch.einsum("bnr,bnrc->bnc", B_local, v_sub)               # [B,N,C]
        s0 = s0 * mask0[..., None]

        # # ---- optional transformer refine on residue tokens ----
        # if self.tf is not None:
        #     # TransformerEncoder expects src_key_padding_mask: True for PAD
        #     pad = (mask0 < 0.5)
        #     s1 = self.tf(s0, src_key_padding_mask=pad)
        #     s1 = s1 * mask0[..., None]
        # else:
        #     s1 = s0
        # ---- unit isotropic offset gaussian rigid init (for later finalup IGA) ----
        # r0 = _build_unit_offset_gaussian_rigid(
        #     B, N, device, dtype,
        #     OffsetGaussianRigid_cls=self.OffsetGaussianRigid_cls,
        #     RotationCls=self.RotationCls,
        # )

        # # ---- optional IGAtransformer refine on residue tokens ----



        # 2) 先构建 r0（纯初始化）
        r0 = _build_unit_offset_gaussian_rigid(
            B, N, device, dtype,
            OffsetGaussianRigid_cls=self.OffsetGaussianRigid_cls,
            RotationCls=self.RotationCls,
        )

        # 3) 直接进 N 层 IGA refine（用几何 attention 来“修正” token + rigid）
        #    这里的 refine_tower 就是你 finn alup.py 的 CoarseIGATower 那套 :contentReference[oaicite:4]{index=4}
        s1, r1 = self.refine_tower(s0, r0, mask0)

        # 4) 输出
        s_fine = self.out_ln(s1)  # 也可以不 LN，看你全局习惯
        r_fine = r1




        aux = {
            "B": B_local,          # [B,N,R]
            "parent_idx": parent_idx,  # [B,N,R]
        }
        levels = [{
            "s": s_fine,
            "r": r_fine,
            "mask": mask0,
            "aux": aux,
        }]
        reg_final = torch.tensor(0.0, device=s_fine.device)
        return levels, reg_final

