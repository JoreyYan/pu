import torch
import torch.nn as nn
import math
from esm.rotary_embedding import RotaryEmbedding
def soft_one_hot_linspace(x, start, end, steps):
    # 简易 RBF / 三角核；你也可以换成高斯 RBF
    x = (x - start) / max(end - start, 1e-6)
    centers = torch.linspace(0, 1, steps, device=x.device, dtype=x.dtype)
    width = 1.0 / (steps - 1 + 1e-9)
    d = (x[..., None] - centers[None, ...]).abs()
    w = (1.0 - (d / (width + 1e-9))).clamp_min(0.0)
    return w  # [..., steps]

class FrameAwareAttentionRot(nn.Module):
    """
    使用 Rotation 类（矩阵或四元数）来做 frame-aware 的几何注意力。
    - x: 已经从 SH 特征映射好的标量 embedding [B,N,d_model]
    - R: Rotation 实例；支持矩阵或四元数存储
    - t: 坐标 [B,N,3]
    """
    def __init__(
        self,
        d_model, n_heads, d_head=None,
        rbf_bins=16,
        dst_chunk_size=512,   # 按 dst 分块，控制显存

    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head or (d_model // n_heads)
        assert self.d_head * n_heads == d_model

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

        self.rbf_bins = rbf_bins
        geo_in = 1 + 3 + rbf_bins  # [r, ehat(x/y/z), rbf(r)]
        self.geo_mlp = nn.Sequential(
            nn.Linear(geo_in, 64), nn.ReLU(),
            nn.Linear(64, n_heads)  # 每个头一个标量 bias
        )

        self.dst_chunk_size = dst_chunk_size
        # ★ RoPE：只在 rope_dims>0 时启用；要求 rope_dims 为偶数
        self.rope_dims = int(d_model/n_heads)
        if self.rope_dims > 0:
            assert self.rope_dims % 2 == 0, "rope_dims 必须是偶数（rotate_half 会按两两配对）"
            self.rope = RotaryEmbedding(self.rope_dims)

    def forward(self, x, R, t, node_mask=None):
        """
        x: [B,N,d_model]
        R: Rotation (batch shape [B,N]) —— 你传进来就是 Rotation 对象
        t: [B,N,3]
        node_mask: [B,N] bool
        """
        B, N, _ = x.shape
        device, dtype = x.device, x.dtype
        if node_mask is None:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # q/k/v
        q = self.q_proj(x).view(B, N, self.n_heads, self.d_head)    # [B,N,H,D]
        k = self.k_proj(x).view(B, N, self.n_heads, self.d_head)

        q, k = self.rope(q, k)


        v = self.v_proj(x).view(B, N, self.n_heads, self.d_head)
        scale = 1.0 / math.sqrt(self.d_head)

        out = x.new_zeros(B, N, self.n_heads, self.d_head)

        # 展平成单批索引空间，方便构造 (i,j) 对；Rotation 支持花式索引
        # 注意：不要逐边构造 Rotation；按节点一次构造/传入，按索引取子 Rotation
        # 这里我们分块遍历 dst=i，src=j 全部节点
        for start in range(0, N,N):
            end = min(start + N, N)

            # 取本块 i 的旋转/坐标/查询/有效掩码
            # Rotation 的 __getitem__ 支持 (b, i) 风格索引；我们先把 batch 维度一起处理
            # 用张量广播计算 (t_j - t_i) 并转到 i 的局部坐标：R_i^{-1}·(t_j - t_i)
            Ri = R[:, start:end]                     # Rotation，形状虚拟为 [B, Nd]
            ti = t[:, start:end]                     # [B,Nd,3]
            qi = q[:, start:end]                     # [B,Nd,H,D]
            vi_mask = node_mask[:, start:end]        # [B,Nd]

            # 计算所有 (i,j) 的相对位移（先全局差，再用 Rotation.apply 到局部）
            # 全局位移 Δt_ij = t_j - t_i
            delta_ij = t[:, None, :, :] - ti[:, :, None, :]  # [B,Nd,N,3]

            # 取矩阵并求逆（旋转矩阵逆=转置）

            Ri_inv = Ri.transpose(-1, -2)  # [B,Nd,3,3]

            # e_{i<-j} = R_i^{-1} Δt_ij  （一次性批量乘）
            # einsum: (B,Nd,3,3) x (B,Nd,N,3) -> (B,Nd,N,3)
            e_local = torch.einsum('b i m n, b i j n -> b i j m', Ri_inv, delta_ij)

            r = e_local.norm(dim=-1).clamp_min(1e-8)  # [B,Nd,N]
            ehat = e_local / r[..., None]  # [B,Nd,N,3]

            # 几何编码 → per-head bias（和原逻辑一样）
            r_end = float(torch.quantile(r.detach(), 0.95).item() + 1e-6)
            rbf = soft_one_hot_linspace(r, 0.0, r_end, self.rbf_bins)  # [B,Nd,N,RB]
            geo = torch.cat([r[..., None], ehat, rbf], dim=-1)  # [B,Nd,N,1+3+RB]
            bias = self.geo_mlp(geo).permute(0, 1, 3, 2).contiguous()  # [B,Nd,H,N]

            # 分块打分/聚合（i-block × all-j）
            # mask：无效 j 置 -inf；无效 i 最后清零
            scores = torch.einsum('bihd,bjhd->bihj', qi, k) * scale     # [B,Nd,H,N]
            scores = scores + bias
            scores = scores.masked_fill(~node_mask.bool()[:, None, None, :], float('-inf'))
            attn = torch.softmax(scores, dim=-1)                        # [B,Nd,H,N]
            out[:, start:end] = torch.einsum('bihj,bjhd->bihd', attn, v)
            out[:, start:end] = out[:, start:end] * vi_mask[:, :, None, None].to(out.dtype)

        y = out.reshape(B, N, self.n_heads * self.d_head)
        return self.o_proj(y)

class FrameAwareTransformerLayerRot(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, **attn_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FrameAwareAttentionRot(d_model, n_heads, **attn_kwargs)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def forward(self, x, R, t, node_mask=None):
        h = self.attn(self.norm1(x), R, t, node_mask=node_mask)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        if node_mask is not None:
            x = x * node_mask[..., None].to(x.dtype)
        return x


class FrameAwareTransformerRot(nn.Module):
    """
    堆叠 n 层的 FrameAwareTransformerLayerRot。
    x: [B, N, d_model]
    R: Rotation（批形状 [B, N]）
    t: [B, N, 3]
    node_mask: [B, N] (bool)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        norm_output: bool = True,
        # 这些参数会传给每一层的注意力
        rbf_bins: int = 24,
        dst_chunk_size: int = 512,

    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            FrameAwareTransformerLayerRot(
                d_model=d_model,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                rbf_bins=rbf_bins,
                dst_chunk_size=dst_chunk_size,
                # 注意：你的 FrameAwareTransformerLayerRot.__init__ 应当接受这些 kwargs

            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) if norm_output else nn.Identity()

    @torch.no_grad()
    def set_dst_chunk_size(self, chunk_size: int):
        """一键修改所有层的 dst_chunk_size（显存/速度调参方便）"""
        for blk in self.layers:
            blk.attn.dst_chunk_size = chunk_size

    def forward(
        self,
        x: torch.Tensor,           # [B,N,d_model]
        R,                          # Rotation([B,N])
        t: torch.Tensor,            # [B,N,3]
        node_mask: torch.Tensor | None = None,  # [B,N] bool
        return_hidden_states: bool = False,
    ):
        hiddens = [] if return_hidden_states else None
        for blk in self.layers:
            x = blk(x, R, t, node_mask=node_mask)
            if return_hidden_states:
                hiddens.append(x)
        x = self.norm(x)
        return (x, hiddens) if return_hidden_states else x

if __name__ == '__main__':
    # 假设你已经有 x = sh_feature 映射后的标量嵌入
    B, N, d = 2, 1024, 512
    x = torch.randn(B, N, d, device='cuda')

    # 你的 Rotation 类：
    # - 如果你手里是 rot_mats: [B,N,3,3]
    #   R = Rotation(rot_mats=rot_mats)  或  Rotation.from_rotmats_safe(rot_mats, project="qr")
    # - 如果是四元数 quats: [B,N,4]
    #   R = Rotation(quats=quats)  （可选 normalize_quats=True）





    B, N, d = 2, 1024, 512
    x = torch.randn(B, N, d, device='cuda')
    rot_mats = torch.eye(3, device='cuda').repeat(B, N, 1, 1)
    t = torch.randn(B, N, 3, device='cuda')
    node_mask = torch.ones(B, N, dtype=torch.bool, device='cuda')

    model = FrameAwareTransformerRot(
        d_model=d, n_heads=8, num_layers=6,
        rbf_bins=24, dst_chunk_size=256,   # rope_dims=0 表示关闭 RoPE
    ).cuda()

    # 不分块（显存允许时）：
    model.set_dst_chunk_size(N)  # 或者设一个 >= N 的大数

    y = model(x, rot_mats, t, node_mask=node_mask)  # [B,N,d]
    print(y.shape)
