import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from openfold.utils import rigid_utils as ru
import numpy as np
import random
try:
    from torch_scatter import scatter_add
except Exception:
    scatter_add = None
import e3nn.nn as enn
from e3nn import o3

############################################
# Utils
############################################

@torch.no_grad()
def project_to_SO3(R: torch.Tensor) -> torch.Tensor:
    # R: [..., 3, 3]
    U, _, Vt = torch.linalg.svd(R)
    R_proj = U @ Vt
    # 保证 det=+1（修正可能出现的反射）
    det = torch.det(R_proj)
    neg = det < 0
    if neg.any():
        # 翻转 U 的最后一列，修正手性
        U[..., :, -1] *= -1
        R_proj = U @ Vt
    return R_proj

def knn_idx_to_global_edge_index(idx: torch.Tensor,  # [B, N, k]  每个点的邻居(批内局部索引)
                                 N: int | None = None,
                                 make_undirected: bool = False,
                                 remove_self_loops: bool = True,
                                 deduplicate: bool = False) -> torch.Tensor:
    """
    把批内的 KNN 邻接 idx -> 全局 edge_index（适合 PyG 风格）。
    返回: edge_index [2, E]  其中节点 id ∈ [0, B*N-1]
    """
    B, N_infer, k = idx.shape
    if N is None: N = N_infer
    device = idx.device

    # 目标节点（每个 batch 内 0..N-1 的每个 i 向其邻居连边）
    dst_local = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k)  # [B,N,k]
    src_local = idx  # [B,N,k]

    # 全局偏移：b*N
    batch_offset = (torch.arange(B, device=device).view(B, 1, 1) * N)          # [B,1,1]
    src_glb = src_local + batch_offset                                         # [B,N,k]
    dst_glb = dst_local + batch_offset                                         # [B,N,k]

    # 展平为 [E]
    src = src_glb.reshape(-1)
    dst = dst_glb.reshape(-1)

    # 可选：去自环
    if remove_self_loops:
        mask = (src != dst)
        src, dst = src[mask], dst[mask]

    # 可选：无向化（加反向边）
    if make_undirected:
        src = torch.cat([src, dst], dim=0)
        dst = torch.cat([dst, src[:len(dst)]], dim=0)  # 注意用之前的 src 备份或先保存

    edge_index = torch.stack([src, dst], dim=0)  # [2,E]

    # 可选：去重（按列唯一）
    if deduplicate:
        # 将每条边编码成线性键 (u*(B*N)+v) 去重
        V = B * N
        key = edge_index[0].to(torch.int64) * V + edge_index[1].to(torch.int64)
        uniq, inv = torch.unique(key, sorted=False, return_inverse=False, return_counts=False, dim=0)
        # 取唯一的位置
        _, unique_pos = torch.unique(key, return_inverse=False, return_counts=False, return_index=True)
        edge_index = edge_index[:, unique_pos]

    return edge_index


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Fallback scatter_sum along dim=0.
    Args:
        src: [E, D]
        index: [E] int64 in [0, dim_size)
        dim_size: N
    Returns:
        out: [N, D]
    """
    if scatter_add is not None:
        return scatter_add(src, index, dim=0, dim_size=dim_size)
    out = src.new_zeros(dim_size, *src.shape[1:])
    out.index_add_(0, index, src)
    return out


def soft_one_hot_linspace(
    x: torch.Tensor,
    *,
    start: float,
    end: float,
    number: int,
    basis: str = "smooth_finite",
    cutoff: bool = True,
) -> torch.Tensor:
    """Smooth RBF-like embedding of distances (API similar to e3nn)."""
    device = x.device
    centers = torch.linspace(start, end, number, device=device)
    if number > 1:
        delta = (end - start) / (number - 1)
    else:
        delta = max(1.0, end - start)
    gamma = 1.0 / (2.0 * (delta ** 2) + 1e-12)
    emb = torch.exp(-gamma * (x[..., None] - centers[None, ...]) ** 2)
    if cutoff:
        c = (x - start) / max(1e-12, (end - start))
        c = torch.clamp(c, 0.0, 1.0)
        cutoff_val = 0.5 * (1.0 + torch.cos(math.pi * (1.0 - c)))
        emb = emb * cutoff_val[..., None]
    emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
    return emb

############################################
# SH pack / unpack
############################################
class SHPacker:
    def __init__(self, C: int, L_max: int, R_bins: int):
        self.C = int(C)
        self.L_max = int(L_max)
        self.R = int(R_bins)
        self.irreps = o3.Irreps("+".join([f"{self.C*self.R}x{l}e" for l in range(self.L_max + 1)]))

        # 预计算 (l,m) 有效掩码：shape [L+1, 2L+1]，位置 (l,m) 有效当 m < 2l+1
        Lp1 = self.L_max + 1
        Mmax = 2 * self.L_max + 1
        l_idx = torch.arange(Lp1)
        m_idx = torch.arange(Mmax)
        valid = (m_idx[None, :] < (2 * l_idx[:, None] + 1))  # [Lp1, Mmax]
        # 展平后的布尔掩码（最后两个维度一起看）
        self.register_buffer = getattr(nn.Module, "register_buffer", lambda *a, **k: None)  # 兼容裸 class
        self.register_buffer_name = "_lm_mask_flat"
        setattr(self, self.register_buffer_name, valid.reshape(-1))  # [Lp1*Mmax]，True 的个数=(L+1)^2

    @torch.no_grad()
    def dim_ir(self) -> int:
        return (self.C * self.R) * (self.L_max + 1) * (self.L_max + 1)

    def pack(self, SH: torch.Tensor) -> torch.Tensor:
        """
        [B,N,C,L+1,2L+1,R] -> [B,N, (C*R)*(L+1)^2]
        矢量化实现，无 for 循环
        """
        B, N, C, Lp1, Mmax, R = SH.shape
        assert C == self.C and R == self.R and Lp1 == self.L_max + 1 and Mmax == 2 * self.L_max + 1
        lm_mask_flat: torch.Tensor = getattr(self, self.register_buffer_name).to(SH.device)  # [Lp1*Mmax]

        # 先把 (C,R) 合并，并把 (L+1,2L+1) 合并成一维，再用掩码选择有效 (l,m)
        # [B,N,C,L+1,2L+1,R] -> [B,N,C,R,L+1,2L+1]
        x = SH.permute(0, 1, 2, 5, 3, 4).contiguous()
        # -> [B,N,C*R, Lp1*Mmax]
        x = x.view(B, N, C * R, Lp1 * Mmax)
        # 选择有效 (l,m)：得到 [B,N,C*R, (L+1)^2]
        x = x[..., lm_mask_flat]
        # 合并 (C*R) 到最后： [B,N, (C*R)*(L+1)^2]
        x = x.reshape(B, N, C * R * (Lp1 * Lp1))
        return x

    def unpack(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B,N, (C*R)*(L+1)^2] -> [B,N,C,L+1,2L+1,R]
        矢量化逆操作
        """
        B, N, D = x.shape
        assert D == self.dim_ir()
        C, R = self.C, self.R
        Lp1 = self.L_max + 1
        Mmax = 2 * self.L_max + 1
        lm_mask_flat: torch.Tensor = getattr(self, self.register_buffer_name).to(x.device)  # [Lp1*Mmax]

        # 还原到 [B,N,C*R,(L+1)^2]
        xr = x.view(B, N, C * R, Lp1 * Lp1)
        # 创建零张量 [B,N,C*R, Lp1*Mmax]，把有效 (l,m) 填回去
        out_flat = x.new_zeros(B, N, C * R, Lp1 * Mmax)
        out_flat[..., lm_mask_flat] = xr  # 只填 True 的位置，其它 m（>2l+1）保持 0

        # 还原形状： [B,N,C,R,L+1,2L+1]
        out = out_flat.view(B, N, C, R, Lp1, Mmax)
        # -> [B,N,C,L+1,2L+1,R]
        out = out.permute(0, 1, 2, 4, 5, 3).contiguous()
        return out


############################################
# Frame-aware SH → SH graph attention with masks
############################################
class SHFrameAwareAttention(nn.Module):
    """
    Frame-aware cross-residue attention in SH space using e3nn, **with masks**.

    Inputs
    ------
    SH_in:      [B, N, C, L+1, 2L+1, R]
    Rmats:      [B, N, 3, 3]
    tpos:       [B, N, 3]
    edge_src, edge_dst: [B, E] or [E]
    node_mask:  Optional[bool] [B, N] / [N]  (valid nodes; False = padding/invalid)

    Returns
    -------
    SH_out:     [B, N, C, L+1, 2L+1, R]
    """

    def __init__(
        self,
        C: int,
        L_max: int,
        R_bins: int,
        L_edge: int = 2,
        n_radial: int = 16,
        hidden_scalar: int = 128,
    ):
        super().__init__()
        self.C, self.L_max, self.R_bins = C, L_max, R_bins
        self.packer = SHPacker(C, L_max, R_bins)

        # Node irreps (input/output)
        self.ir_in = self.packer.irreps
        self.ir_q = self.ir_in
        self.ir_k = self.ir_in
        self.ir_v = self.ir_in

        # Query projection (node-wise)
        self.to_q = o3.Linear(self.ir_in, self.ir_q)

        # Edge spherical harmonics irreps up to L_edge
        self.ir_sh = o3.Irreps.spherical_harmonics(L_edge)

        # K,V tensor products with edge SH; weights from radial MLP
        self.tp_k = o3.FullyConnectedTensorProduct(self.ir_in, self.ir_sh, self.ir_k, shared_weights=False)
        self.tp_v = o3.FullyConnectedTensorProduct(self.ir_in, self.ir_sh, self.ir_v, shared_weights=False)

        self.radial_k = nn.Sequential(
            nn.Linear(n_radial, hidden_scalar), nn.SiLU(), nn.Linear(hidden_scalar, self.tp_k.weight_numel)
        )
        self.radial_v = nn.Sequential(
            nn.Linear(n_radial, hidden_scalar), nn.SiLU(), nn.Linear(hidden_scalar, self.tp_v.weight_numel)
        )

        # Scalar dot product for attention scores
        self.dot = o3.FullyConnectedTensorProduct(self.ir_q, self.ir_k, "0e")

        # Radial params
        self.n_radial = n_radial

    def Gforward(
            self,
            SH_in: torch.Tensor,
            Rmats: torch.Tensor,
            tpos: torch.Tensor,
            edge_src: Optional[torch.Tensor] = None,
            edge_dst: Optional[torch.Tensor] = None,
            max_radius: Optional[float] = None,
            node_mask: Optional[torch.Tensor] = None,  # [B,N] or [N]
            include_self: bool = False,  # NEW: 是否包含自环，默认为 False 以避免 Y_lm(0) 问题
    ) -> torch.Tensor:
        B, N, C, Lp1, Mmax, R = SH_in.shape
        device = SH_in.device

        # ----- 展平节点到 [B*N] 空间（原逻辑）-----
        SHf = SH_in.reshape(B * N, C, Lp1, Mmax, R)
        Rf = Rmats.reshape(B * N, 3, 3)
        tf = tpos.reshape(B * N, 3)

        # 节点 mask 展平（原逻辑）
        if node_mask is not None:
            node_mask = node_mask.to(device)
            node_mask_flat = node_mask.reshape(B * N) if node_mask.dim() == 2 else node_mask
        else:
            node_mask_flat = torch.ones(B * N, dtype=torch.bool, device=device)

        # ----- 如果没给边：自动构造 batch 内全连接边 -----
        if (edge_src is None) or (edge_dst is None):
            ii = torch.arange(N, device=device)
            edge_src = ii[None, :, None].expand(B, N, N).reshape(B, -1)  # [B, N*N]
            edge_dst = ii[None, None, :].expand(B, N, N).reshape(B, -1)  # [B, N*N]

        # ----- 加 batch 偏移后展平为 [E]（原逻辑）-----
        if edge_src.dim() == 2:
            offs = torch.arange(B, device=device)[:, None] * N
            src = (edge_src + offs).reshape(-1)
            dst = (edge_dst + offs).reshape(-1)
        else:
            src = edge_src
            dst = edge_dst
        # 注意：这里正是防止不同 batch 的 0..N-1 索引被混在一起的关键:contentReference[oaicite:1]{index=1}

        # ----- 过滤无效边（原逻辑）+ 去自环（新增）-----
        edge_valid = node_mask_flat[src] & node_mask_flat[dst]
        if not include_self:
            edge_valid = edge_valid & (src != dst)
        if edge_valid.sum() == 0:
            return SH_in
        src = src[edge_valid]
        dst = dst[edge_valid]
        E = src.numel()

        # ----- 下面保持你原有的 SH 并行转运 + 球谐 + 软最大聚合 -----
        x = self.packer.pack(SHf.unsqueeze(0)).squeeze(0)  # [Ntot, dim_ir]
        x = x * node_mask_flat[:, None].to(x.dtype)  # 零出 padding

        # 平行转运 Q_{i<-j}
        Q_rel = (Rf[dst].transpose(-1, -2) @ Rf[src])  # [E,3,3]
        D_edge = self.ir_in.D_from_matrix(Q_rel)  # [E, dim_ir, dim_ir]
        x_j_rot = (D_edge @ x[src].unsqueeze(-1)).squeeze(-1)  # [E, dim_ir]

        # 接收端坐标系下的相对向量 & 半径嵌入
        e_ij_local = (Rf[dst].transpose(-1, -2) @ (tf[src] - tf[dst]).unsqueeze(-1)).squeeze(-1)  # [E,3]
        r_ij = e_ij_local.norm(dim=-1)  # [E]
        if max_radius is None:
            max_radius = torch.quantile(r_ij.detach(), 0.95).clamp(min=1.0).item()
        r_emb = soft_one_hot_linspace(r_ij, start=0.0, end=float(max_radius), number=self.n_radial, cutoff=True)

        # 球谐 & 生成 K,V
        Y_ij = o3.spherical_harmonics(self.ir_sh, e_ij_local, True, normalization="component")  # [E, dim_sh]
        q = self.to_q(x)  # [Ntot, dim(ir_q)]
        k = self.tp_k(x_j_rot, Y_ij, self.radial_k(r_emb))  # [E, dim(ir_k)]
        v = self.tp_v(x_j_rot, Y_ij, self.radial_v(r_emb))  # [E, dim(ir_v)]

        # 按 dst 归一化的 softmax（原逻辑）
        scores = self.dot(q[dst], k)  # [E,1]
        attn_num = torch.exp(scores)
        denom = _scatter_sum(attn_num, dst, dim_size=x.shape[0])  # [Ntot,1]
        denom = (denom * node_mask_flat[:, None].to(denom.dtype)).clamp_min_(1e-9)
        alpha = attn_num / denom[dst]

        out_vec = _scatter_sum(alpha * v, dst, dim_size=x.shape[0])  # [Ntot, dim(ir_v)]
        out_vec = out_vec * node_mask_flat[:, None].to(out_vec.dtype)
        SH_out = self.packer.unpack(out_vec.unsqueeze(0)).squeeze(0)  # [Ntot,C,L+1,2L+1,R]
        return SH_out.reshape(B, N, C, Lp1, Mmax, R)
    def forward(
        self,
        SH_in: torch.Tensor,
        Rmats: torch.Tensor,
        tpos: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        max_radius: Optional[float] = None,
        node_mask: Optional[torch.Tensor] = None,  # [B,N] or [N]
    ) -> torch.Tensor:
        B, N, C, Lp1, Mmax, R = SH_in.shape
        assert C == self.C and R == self.R_bins and Lp1 == self.L_max + 1 and Mmax == 2 * self.L_max + 1

        device = SH_in.device
        SHf = SH_in.reshape(B * N, C, Lp1, Mmax, R)
        Rf = Rmats.reshape(B * N, 3, 3)
        tf = tpos.reshape(B * N, 3)

        # masks
        if node_mask is not None:
            node_mask = node_mask.to(device)
            if node_mask.dim() == 2:
                node_mask_flat = node_mask.reshape(B * N)
            else:
                node_mask_flat = node_mask
        else:
            node_mask_flat = torch.ones(B * N, dtype=torch.bool, device=device)

        # build flattened edges
        if edge_src.dim() == 2:
            offs = torch.arange(B, device=device)[:, None] * N
            src = (edge_src + offs).reshape(-1)
            dst = (edge_dst + offs).reshape(-1)
        else:
            src = edge_src
            dst = edge_dst
        E = src.numel()

        # remove edges touching invalid nodes
        edge_valid = node_mask_flat[src] & node_mask_flat[dst]
        if edge_valid.sum() == 0:
            # nothing valid; return input as-is
            return SH_in
        src = src[edge_valid]
        dst = dst[edge_valid]

        # Pack SH → irreps vector per node
        x = self.packer.pack(SHf.unsqueeze(0)).squeeze(0)  # [Ntot, dim_ir]
        Ntot = x.shape[0]

        # zero-out invalid nodes in q-space as well
        x = x * node_mask_flat[:, None].to(x.dtype)

        # Parallel transport: Q_{i<-j} = R_i^T R_j  (per edge)
        #Q_rel = (Rf[dst].transpose(-1, -2) @ Rf[src])  # [E,3,3]

        # 1) 按节点构造一次 Rotation（建议投影以确保稳定）
        R_all=ru.Rotation(Rf)

        # 2) 取出 src/dst 的旋转（不会复制数据，仍然是批量）
        R_src = R_all[src]  # [E]
        R_dst = R_all[dst]  # [E]

        # 3) 相对旋转：Q_rel = R_dst^{-1} ∘ R_src  （等价于 R_dst^T @ R_src）
        Q_rel = R_dst.invert().compose_r(R_src).get_rot_mats()  # [E,3,3]

        # R=Q_rel
        # det_R = torch.det(R)
        # print(f"行列式: {det_R}")
        # print(f"应该接近1，实际偏差: {abs(det_R - 1)}")
        #
        # # 检查正交性
        # orthogonality_error = torch.norm(R @ R.transpose(-1, -2) - torch.eye(3, device=R.device))
        # print(f"正交性误差: {orthogonality_error}")



        D_edge = self.ir_in.D_from_matrix(Q_rel)       # [E, dim_ir, dim_ir]
        x_j = x[src]                                    # [E, dim_ir]
        x_j_rot = torch.matmul(D_edge, x_j.unsqueeze(-1)).squeeze(-1)  # [E, dim_ir]

        # Edge vector in receiver frame
        # e_ij_local = (Rf[dst].transpose(-1, -2) @ (tf[src] - tf[dst]).unsqueeze(-1)).squeeze(-1)  # [E,3]

        # 4) 相对位移转到接收端坐标系： R_dst^{-1}·(t_src - t_dst)
        delta = (tpos.reshape(-1, 3)[src] - tpos.reshape(-1, 3)[dst])  # [E,3]
        e_ij_local = R_dst.invert().apply(delta)  # [E,3]

        r_ij = e_ij_local.norm(dim=-1)  # [E]

        if max_radius is None:
            max_radius = torch.quantile(r_ij.detach(), 0.95).clamp(min=1.0).item()
        r_emb = soft_one_hot_linspace(r_ij, start=0.0, end=float(max_radius), number=self.n_radial, cutoff=True)  # [E,K]

        ir_sh = self.ir_sh
        Y_ij = o3.spherical_harmonics(ir_sh, e_ij_local, True, normalization="component")  # [E, dim_sh]

        # Q/K/V
        q = self.to_q(x)  # [Ntot, dim(ir_q)]
        k = self.tp_k(x_j_rot, Y_ij, self.radial_k(r_emb))  # [E, dim(ir_k)]
        v = self.tp_v(x_j_rot, Y_ij, self.radial_v(r_emb))  # [E, dim(ir_v)]

        # scores & softmax per dst
        scores = self.dot(q[dst], k)  # [E,1]
        attn_num = torch.exp(scores)
        denom = _scatter_sum(attn_num, dst, dim_size=Ntot)  # [Ntot,1]
        # avoid using invalid dst in denom
        denom = denom * node_mask_flat[:, None].to(denom.dtype)
        denom = denom.clamp_min_(1e-9)
        alpha = attn_num / denom[dst]

        out_vec = _scatter_sum(alpha * v, dst, dim_size=Ntot)  # [Ntot, dim(ir_v)]
        # zero invalid rows
        out_vec = out_vec * node_mask_flat[:, None].to(out_vec.dtype)

        SH_out = self.packer.unpack(out_vec.unsqueeze(0)).squeeze(0)  # [Ntot,C,L+1,2L+1,R]
        SH_out = SH_out.reshape(B, N, C, Lp1, Mmax, R)
        return SH_out

############################################
# Equivariant FFN + Block + Transformer (with masks and PreNorm)
############################################
class SHEquiFFN(nn.Module):
    def __init__(self, packer: SHPacker):
        super().__init__()
        ir = packer.irreps
        self.lin1 = o3.Linear(ir, ir)
        try:
            self.gate = o3.Gate(ir)
            self.use_gate = True
        except Exception:
            self.use_gate = False
        self.tp = o3.FullyConnectedTensorProduct(ir, ir, ir,internal_weights=True,   shared_weights=True)
        self.lin2 = o3.Linear(ir, ir)

    def forward(self, x_ir: torch.Tensor) -> torch.Tensor:
        y = self.lin1(x_ir)
        if self.use_gate:
            y = self.gate(y)
        y = self.tp(y, y)
        y = self.lin2(y)
        return y


class SHTransformerBlock(nn.Module):
    def __init__(
        self,

        C: int,
        L_max: int,
        R_bins: int,
        L_edge: int = 2,
        n_radial: int = 16,
        dropout: float = 0.0,
            hidden_scalar: int = 256,

    ):
        super().__init__()
        self.packer = SHPacker(C, L_max, R_bins)
        self.attn = SHFrameAwareAttention(C, L_max, R_bins, L_edge, n_radial,hidden_scalar)
        self.ffn = SHEquiFFN(self.packer)
        self.do = nn.Dropout(dropout)
        # equivariant PreNorms
        self.norm1 = enn.BatchNorm(self.packer.irreps)
        self.norm2 = enn.BatchNorm(self.packer.irreps)
    def Gforward(
        self,
        SH_in: torch.Tensor,
        Rmats: torch.Tensor,
        tpos: torch.Tensor,
        edge_src: Optional[torch.Tensor] = None,   # 改：可为 None
        edge_dst: Optional[torch.Tensor] = None,   # 改：可为 None
        max_radius: Optional[float] = None,
        node_mask: Optional[torch.Tensor] = None,  # [B,N] or [N]
        update_mask: Optional[torch.Tensor] = None,# [B,N] or [N]
        include_self: bool = False,                # NEW
    ) -> torch.Tensor:
        ...
        # ---- PreNorm for Attention ----
        x_ir = self.packer.pack(SH_in)
        x_ir = x_ir * node_mask[..., None].to(x_ir.dtype)
        x_norm = self.norm1(x_ir)
        x_norm = self.packer.unpack(x_norm)

        # ---- 全局 SH-Attention（当 edge_* 缺省时会自动全连接）----
        attn_out = self.attn(
            x_norm, Rmats, tpos,
            edge_src=edge_src, edge_dst=edge_dst,
            max_radius=max_radius, node_mask=node_mask,
            include_self=include_self
        )
        x = SH_in + self.do(attn_out)
        # selective residual write: only update where update_mask=True
        x = torch.where(update_mask[..., None, None, None, None], x, SH_in)

        # ---- PreNorm for FFN ----
        x_ir = self.packer.pack(x)
        x_ir = x_ir * node_mask[..., None].to(x_ir.dtype)
        x_ir = self.norm2(x_ir)
        y_ir = self.ffn(x_ir)
        y = self.packer.unpack(y_ir)
        x2 = x + self.do(y)
        x2 = torch.where(update_mask[..., None, None, None, None], x2, x)
        return x2
    def forward(
        self,
        SH_in: torch.Tensor,
        Rmats: torch.Tensor,
        tpos: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        max_radius: Optional[float] = None,
        node_mask: Optional[torch.Tensor] = None,   # [B,N] or [N]
        update_mask: Optional[torch.Tensor] = None, # [B,N] or [N]; False = do NOT update
    ) -> torch.Tensor:
        B, N = SH_in.shape[:2]
        device = SH_in.device
        # default masks
        if node_mask is None:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        if update_mask is None:
            update_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # ---- PreNorm for Attention ----
        x_ir = self.packer.pack(SH_in)
        assert x_ir.shape[-1] == (SH_in.shape[2] * SH_in.shape[-1]) * (SH_in.shape[3] ** 2)
        # [B,N,dim_ir]
        x_ir = x_ir * node_mask[..., None].to(x_ir.dtype)       # zero invalid
        x_ir = self.norm1(x_ir)
        x_norm = self.packer.unpack(x_ir)

        # ---- Attention ----
        attn_out = self.attn(x_norm, Rmats, tpos, edge_src, edge_dst, max_radius, node_mask=node_mask)
        x = SH_in + self.do(attn_out)
        # selective residual write: only update where update_mask=True
        x = torch.where(update_mask[..., None, None, None, None], x, SH_in)

        # ---- PreNorm for FFN ----
        x_ir = self.packer.pack(x)
        x_ir = x_ir * node_mask[..., None].to(x_ir.dtype)
        x_ir = self.norm2(x_ir)
        y_ir = self.ffn(x_ir)
        y = self.packer.unpack(y_ir)
        x2 = x + self.do(y)
        x2 = torch.where(update_mask[..., None, None, None, None], x2, x)
        return x2


class SHTransformer(nn.Module):
    def __init__(
        self,
        C: int = 4,
        L_max: int = 2,
        R_bins: int = 16,
        L_edge: int = 2,
        n_layers: int = 4,
        n_radial: int = 16,
        dropout: float = 0.0,
            hidden_scalar : int = 256,**kwargs
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            SHTransformerBlock(C, L_max, R_bins, L_edge, n_radial, dropout,hidden_scalar)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        SH_in: torch.Tensor,
        Rmats: torch.Tensor,
        tpos: torch.Tensor,
        edge_src:  Optional[torch.Tensor] = None,
        edge_dst:  Optional[torch.Tensor] = None,
        max_radius: Optional[float] = None,
        node_mask: Optional[torch.Tensor] = None,
        update_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = SH_in
        for blk in self.blocks:
            x = blk(x, Rmats, tpos, edge_src, edge_dst, max_radius, node_mask, update_mask)
        return x


def set_seed(seed=42):
    """设置所有随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 确保CUDA操作的确定性（可能会影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




if __name__ == "__main__":
    # 使用
    set_seed(42)
    B, N = 2, 23
    C, L_max, R_bins = 4, 3, 16
    SH = torch.randn(B, N, C, L_max+1, 2*L_max+1, R_bins)
    Rm = o3.rand_matrix(B, N)


    # # knn
    t = torch.randn(B, N, 3)
    # simple knn (toy)
    d = torch.cdist(t, t)  # [B, N, N]
    k = 6
    idx = torch.topk(-d, k + 1, dim=-1).indices[:, :, 1:]  # [B, N, k]
    src = idx.reshape(B, -1)  # 每个 batch 各自的 knn 源点索引
    dst = torch.arange(N).repeat_interleave(k)[None, :].repeat(B, 1)  # 目标索引






    # masks
    node_mask = torch.ones(B, N, dtype=torch.bool)
    node_mask[:, -2:] = False  # last two are padding
    update_mask = torch.ones(B, N, dtype=torch.bool)
    update_mask[:, :2] = False  # first two are frozen (no update)

    model = SHTransformer(C=C, L_max=L_max, R_bins=R_bins, n_layers=2)

    out = model(SH, Rm, t, src, dst, node_mask=node_mask, update_mask=update_mask)

    # 不传 edge_src/edge_dst -> 自动全连接（N 维度 attention）
    # out = model(SH, Rm, t, node_mask=node_mask, update_mask=update_mask)

    print(out.shape)
    print(out)
