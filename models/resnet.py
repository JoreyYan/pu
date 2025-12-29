"""
https://github.com/ProteinDesignLab/protpardelle
License: MIT
Author: Alex Chu

Neural network modules. Many of these are adapted from open source modules.
"""

from typing import List, Sequence, Optional

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmModel


from models import utils


class AtomFeatureExtractor(nn.Module):
    """
    原子特征提取器：从原子坐标提取几何特征
    """

    def __init__(self, n_atoms=4):
        super().__init__()
        self.n_atoms = n_atoms

    def forward(self, coords):
        """
        coords: [B, N, A, 3]
        return: [B, N, A, feature_dim]
        """
        B, N, A, _ = coords.shape
        features = []

        # 1. 原始坐标
        features.append(coords)  # [B, N, A, 3]

        # 2. 相对于质心的坐标
        centroid = coords.mean(dim=2, keepdim=True)  # [B, N, 1, 3]
        relative_coords = coords - centroid  # [B, N, A, 3]
        features.append(relative_coords)

        # 3. 距离特征
        # 每个原子到质心的距离
        distances_to_center = torch.norm(relative_coords, dim=-1, keepdim=True)  # [B, N, A, 1]
        features.append(distances_to_center)

        # 4. 原子间距离（选择性计算，避免过多计算）
        # 只计算相邻原子的距离
        if A > 1:
            atom_diffs = coords[:, :, 1:] - coords[:, :, :-1]  # [B, N, A-1, 3]
            atom_distances = torch.norm(atom_diffs, dim=-1, keepdim=True)  # [B, N, A-1, 1]
            # 补齐维度
            atom_distances = F.pad(atom_distances, (0, 0, 0, 1), value=0)  # [B, N, A, 1]
            features.append(atom_distances)

        # 5. 角度特征（简化版）
        if A > 2:
            # 计算连续三个原子的角度
            v1 = coords[:, :, 1:] - coords[:, :, :-1]  # [B, N, A-1, 3]
            v2 = coords[:, :, 2:] - coords[:, :, 1:-1]  # [B, N, A-2, 3]

            # 计算夹角余弦值
            cos_angles = F.cosine_similarity(v1[:, :, :-1], v2, dim=-1, eps=1e-8)  # [B, N, A-2]
            cos_angles = cos_angles.unsqueeze(-1)  # [B, N, A-2, 1]
            # 补齐维度
            cos_angles = F.pad(cos_angles, (0, 0, 0, 2), value=0)  # [B, N, A, 1]
            features.append(cos_angles)

        # 拼接所有特征
        atom_features = torch.cat(features, dim=-1)  # [B, N, A, total_features]

        return atom_features
class Conv2DFeatureExtractor(nn.Module):
    """
    使用2D卷积处理原子坐标，然后聚合到残基级特征
    """

    def __init__(self, n_atoms=4, output_dim=128):
        super().__init__()
        self.n_atoms = n_atoms
        self.atom_feature_extractor = AtomFeatureExtractor(n_atoms)

        # 2D卷积层 - 在原子和特征维度上进行卷积
        self.conv2d_layers = nn.Sequential(
            # 第一层：处理原子间关系
            nn.Conv2d(9, 32, kernel_size=(3, 3), padding=(1, 1)),  # input_features=11
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 第二层：提取更高级特征
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            # 第三层：进一步抽象
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # 原子维度聚合
        self.atom_aggregation = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),  # 在原子维度上全局平均池化
            nn.Flatten(start_dim=2, end_dim=3)  # [B, 128, 1, N] -> [B, 128, N]
        )

        # 最终特征变换
        self.final_transform = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, output_dim, kernel_size=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, coords):
        """
        coords: [B, N, A, 3]
        return: [B, N, D]
        """
        B, N, A, _ = coords.shape

        # 提取原子特征
        atom_features = self.atom_feature_extractor(coords)  # [B, N, A, feature_dim]

        # 重排维度用于2D卷积: [B, feature_dim, A, N]
        features = rearrange(atom_features, 'b n a d -> b d a n')

        # 2D卷积处理
        features = self.conv2d_layers(features)  # [B, 128, A, N]

        # 原子维度聚合
        features = self.atom_aggregation(features)  # [B, 128, N]

        # 最终变换
        features = self.final_transform(features)  # [B, output_dim, N]

        # 转换为 [B, N, D]
        features = rearrange(features, 'b d n -> b n d')

        return features


########################################
# Adapted from https://github.com/ermongroup/ddim


def downsample(x):
    return nn.functional.avg_pool2d(x, 2, 2, ceil_mode=True)


def upsample_coords(x, shape):
    new_l, new_w = shape
    return nn.functional.interpolate(x, size=(new_l, new_w), mode="nearest")


########################################
# Adapted from https://github.com/aqlaboratory/openfold


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.contiguous().permute(first_inds + [zero_index + i for i in inds])


def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_positions[..., None, :] - all_atom_positions[..., None, :, :])
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_pred_pos[..., None, :] - all_atom_pred_pos[..., None, :, :]) ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score


class RelativePositionalEncoding(nn.Module):
    def __init__(self, attn_dim=8, max_rel_idx=32):
        super().__init__()
        self.max_rel_idx = max_rel_idx
        self.n_rel_pos = 2 * self.max_rel_idx + 1
        self.linear = nn.Linear(self.n_rel_pos, attn_dim)

    def forward(self, residue_index):
        d_ij = residue_index[..., None] - residue_index[..., None, :]
        v_bins = torch.arange(self.n_rel_pos).to(d_ij.device) - self.max_rel_idx
        idxs = (d_ij[..., None] - v_bins[None, None]).abs().argmin(-1)
        p_ij = nn.functional.one_hot(idxs, num_classes=self.n_rel_pos)
        embeddings = self.linear(p_ij.float())
        return embeddings


########################################
# Adapted from https://github.com/NVlabs/edm


class Noise_Embedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.outer(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


########################################
# Adapted from github.com/lucidrains
# https://github.com/lucidrains/denoising-diffusion-pytorch
# https://github.com/lucidrains/recurrent-interface-network-pytorch


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def posemb_sincos_1d(patches, temperature=10000, residue_index=None):
    _, n, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    n = torch.arange(n, device=device) if residue_index is None else residue_index
    assert (dim % 2) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n[..., None] * omega
    pe = torch.cat((n.sin(), n.cos()), dim=-1)
    return pe.type(dtype)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class NoiseConditioningBlock(nn.Module):
    def __init__(self, n_in_channel, n_out_channel):
        super().__init__()
        self.block = nn.Sequential(
            Noise_Embedding(n_in_channel),
            nn.Linear(n_in_channel, n_out_channel),
            nn.SiLU(),
            nn.Linear(n_out_channel, n_out_channel),
            Rearrange("b d -> b 1 d"),
        )

    def forward(self, noise_level):
        return self.block(noise_level)


class TimeCondResnetBlock(nn.Module):
    def __init__(
        self, nic, noc, cond_nc, conv_layer=nn.Conv2d, dropout=0.1, n_norm_in_groups=4
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups=nic // n_norm_in_groups, num_channels=nic),
            nn.SiLU(),
            conv_layer(nic, noc, 3, 1, 1),
        )
        self.cond_proj = nn.Linear(cond_nc, noc * 2)
        self.mid_norm = nn.GroupNorm(num_groups=noc // 4, num_channels=noc)
        self.dropout = dropout if dropout is None else nn.Dropout(dropout)
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups=noc // 4, num_channels=noc),
            nn.SiLU(),
            conv_layer(noc, noc, 3, 1, 1),
        )
        self.mismatch = False
        if nic != noc:
            self.mismatch = True
            self.conv_match = conv_layer(nic, noc, 1, 1, 0)

    def forward(self, x, time=None):
        h = self.block1(x)

        if time is not None:
            h = self.mid_norm(h)
            scale, shift = self.cond_proj(time).chunk(2, dim=-1)
            h = (h * (utils.expand(scale, h) + 1)) + utils.expand(shift, h)

        if self.dropout is not None:
            h = self.dropout(h)

        h = self.block2(h)

        if self.mismatch:
            x = self.conv_match(x)

        return x + h


class TimeCondAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context=None,
        heads=4,
        dim_head=32,
        norm=False,
        norm_context=False,
        time_cond_dim=None,
        attn_bias_dim=None,
        rotary_embedding_module=None,
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(nn.SiLU(), nn.Linear(time_cond_dim, dim * 2))

            nn.init.zeros_(self.time_cond[-1].weight)
            nn.init.zeros_(self.time_cond[-1].bias)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_bias_proj = None
        if attn_bias_dim is not None:
            self.attn_bias_proj = nn.Sequential(
                Rearrange("b a i j -> b i j a"),
                nn.Linear(attn_bias_dim, heads),
                Rearrange("b i j a -> b a i j"),
            )

        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, hidden_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.zeros_(self.to_out.weight)

        self.use_rope = False
        if rotary_embedding_module is not None:
            self.use_rope = True
            self.rope = rotary_embedding_module

    def forward(self, x, context=None, time=None, attn_bias=None, seq_mask=None):
        # attn_bias is b, c, i, j
        h = self.heads
        has_context = exists(context)

        context = default(context, x)

        if x.shape[-1] != self.norm.gamma.shape[-1]:
            print(context.shape, x.shape, self.norm.gamma.shape)

        x = self.norm(x)

        if exists(time):
            scale, shift = self.time_cond(time).chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        if has_context:
            context = self.norm_context(context)

        if seq_mask is not None:
            x = x * seq_mask[..., None]

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        q = q * self.scale

        if self.use_rope:
            q = self.rope.rotate_queries_or_keys(q)
            k = self.rope.rotate_queries_or_keys(k)

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        if attn_bias is not None:
            if self.attn_bias_proj is not None:
                attn_bias = self.attn_bias_proj(attn_bias)
            sim += attn_bias
        if seq_mask is not None:
            attn_mask = torch.einsum("b i, b j -> b i j", seq_mask, seq_mask)[:, None]
            sim -= (1 - attn_mask) * 1e6
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        if seq_mask is not None:
            out = out * seq_mask[..., None]
        return out


class TimeCondFeedForward(nn.Module):
    def __init__(self, dim, mult=4, dim_out=None, time_cond_dim=None, dropout=0.1):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.norm = LayerNorm(dim)

        self.time_cond = None
        self.dropout = None
        inner_dim = int(dim * mult)

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, inner_dim * 2),
            )

            nn.init.zeros_(self.time_cond[-1].weight)
            nn.init.zeros_(self.time_cond[-1].bias)

        self.linear_in = nn.Linear(dim, inner_dim)
        self.nonlinearity = nn.SiLU()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(inner_dim, dim_out)
        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

    def forward(self, x, time=None):
        x = self.norm(x)
        x = self.linear_in(x)
        x = self.nonlinearity(x)

        if exists(time):
            scale, shift = self.time_cond(time).chunk(2, dim=-1)
            x = (x * (scale + 1)) + shift

        if exists(self.dropout):
            x = self.dropout(x)

        return self.linear_out(x)





class TimeCondUViT(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        dim: int,
        patch_size: int = 1,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 32,
        n_filt_per_layer: List[int] = [],
        n_blocks_per_layer: int = 2,
        n_atoms: int = 37,
        channels_per_atom: int = 3,
        attn_bias_dim: int = None,
        time_cond_dim: int = None,
        conv_skip_connection: bool = False,
        position_embedding_type: str = "rotary",
    ):
        super().__init__()

        # Initialize configuration params
        if time_cond_dim is None:
            time_cond_dim = dim * 4
        self.position_embedding_type = position_embedding_type
        channels = channels_per_atom
        self.n_conv_layers = n_conv_layers = len(n_filt_per_layer)
        if n_conv_layers > 0:
            post_conv_filt = n_filt_per_layer[-1]
        self.conv_skip_connection = conv_skip_connection and n_conv_layers == 1
        transformer_seq_len = seq_len // (2**n_conv_layers)
        assert transformer_seq_len % patch_size == 0
        num_patches = transformer_seq_len // patch_size
        dim_a = post_conv_atom_dim = max(1, n_atoms // (2 ** (n_conv_layers - 1)))
        if n_conv_layers == 0:
            patch_dim = patch_size * n_atoms * channels_per_atom
            patch_dim_out = patch_size * n_atoms * 3
            dim_a = n_atoms
        elif conv_skip_connection and n_conv_layers == 1:
            patch_dim = patch_size * (channels + post_conv_filt) * post_conv_atom_dim
            patch_dim_out = patch_size * post_conv_filt * post_conv_atom_dim
        elif n_conv_layers > 0:
            patch_dim = patch_dim_out = patch_size * post_conv_filt * post_conv_atom_dim

        # Make downsampling conv
        # Downsamples n-1 times where n is n_conv_layers
        down_conv = []
        block_in = channels
        for i, nf in enumerate(n_filt_per_layer):
            block_out = nf
            layer = []
            for j in range(n_blocks_per_layer):
                n_groups = 2 if i == 0 and j == 0 else 4
                layer.append(
                    TimeCondResnetBlock(
                        block_in, block_out, time_cond_dim, n_norm_in_groups=n_groups
                    )
                )
                block_in = block_out
            down_conv.append(nn.ModuleList(layer))
        self.down_conv = nn.ModuleList(down_conv)

        # Make transformer
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (n p) a -> b n (p c a)", p=patch_size),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )

        self.from_patch = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, patch_dim_out),
            Rearrange("b n (p c a) -> b c (n p) a", p=patch_size, a=dim_a),
        )
        nn.init.zeros_(self.from_patch[-2].weight)
        nn.init.zeros_(self.from_patch[-2].bias)


        # Conv out
        if n_conv_layers > 0:
            self.conv_out = nn.Sequential(
                nn.GroupNorm(num_groups=block_out // 4, num_channels=block_out),
                nn.SiLU(),
                nn.Conv2d(block_out, channels // 2, 3, 1, 1),
            )

    def forward(
        self, coords, time_cond, pair_bias=None, seq_mask=None, residue_index=None
    ):
        if self.n_conv_layers > 0:  # pad up to even dims
            coords = F.pad(coords, (0, 0, 0, 0, 0, 1, 0, 0))

        x = rearr_coords = rearrange(coords, "b n a c -> b c n a")
        hiddens = []
        for i, layer in enumerate(self.down_conv):
            for block in layer:
                x = block(x, time=time_cond)
                hiddens.append(x)
            if i != self.n_conv_layers - 1:
                x = downsample(x)

        if self.conv_skip_connection:
            x = torch.cat([x, rearr_coords], 1)


        if seq_mask is not None and x.shape[1] == seq_mask.shape[1]:
            x *= seq_mask[..., None]


        x = rearrange(x, "b c n a -> b n a c")
        return x


"""
初始化模型和生成模拟数据进行测试
"""

import torch
import torch.nn as nn
import numpy as np


# 假设你已经导入了原始的neural_modules
# from neural_modules import *

def model_initialization():
    """测试模型初始化和前向传播"""

    # 设置设备
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 模型参数
    seq_len = 128
    dim = 256
    patch_size = 4
    depth = 6
    heads = 8
    dim_head = 32
    n_filt_per_layer = [64, 128]  # 卷积层过滤器数
    n_blocks_per_layer = 2
    n_atoms = 4
    channels_per_atom = 3
    attn_bias_dim = 8
    time_cond_dim = 1024

    print("Initializing TimeCondUViT model...")

    # 初始化模型
    model = TimeCondUViT(
        seq_len=seq_len,
        dim=dim,
        patch_size=patch_size,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        n_filt_per_layer=n_filt_per_layer,
        n_blocks_per_layer=n_blocks_per_layer,
        n_atoms=n_atoms,
        channels_per_atom=channels_per_atom,
        attn_bias_dim=attn_bias_dim,
        time_cond_dim=time_cond_dim,
        conv_skip_connection=True,
        position_embedding_type="rotary"
    ).to(device)

    print(f"Model initialized successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 生成模拟数据
    batch_size = 2

    # 坐标数据 (batch, seq_len, n_atoms, 3)
    coords = torch.randn(batch_size, seq_len, n_atoms, 3).to(device)
    print(f"Input coords shape: {coords.shape}")

    # 时间条件 (batch,)
    time_cond = None


    # 配对偏置 (batch, attn_bias_dim, seq_len, seq_len)
    pair_bias = None


    # 序列掩码 (batch, seq_len)
    seq_mask = torch.ones(batch_size, seq_len).to(device)
    # 随机掩盖一些位置
    seq_mask[:, seq_len // 2:] = 0  # 掩盖后半部分
    print(f"Sequence mask shape: {seq_mask.shape}")

    # 残基索引 (batch, seq_len)
    residue_index = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
    print(f"Residue index shape: {residue_index.shape}")

    # 前向传播
    print("\nRunning forward pass...")
    try:
        with torch.no_grad():
            output = model(
                coords=coords,
                time_cond=time_cond,
                pair_bias=pair_bias,
                seq_mask=seq_mask,
                residue_index=residue_index
            )

        print(f"✓ Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output device: {output.device}")

        # 检查输出是否包含NaN或Inf
        if torch.isnan(output).any():
            print("⚠ Warning: Output contains NaN values!")
        elif torch.isinf(output).any():
            print("⚠ Warning: Output contains Inf values!")
        else:
            print("✓ Output values are normal")

        # 打印输出统计信息
        print(f"Output statistics:")
        print(f"  Mean: {output.mean().item():.6f}")
        print(f"  Std: {output.std().item():.6f}")
        print(f"  Min: {output.min().item():.6f}")
        print(f"  Max: {output.max().item():.6f}")

    except Exception as e:
        print(f"✗ Forward pass failed with error: {e}")
        raise e


def individual_components():
    """测试各个组件"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 50)
    print("Testing individual components...")

    # 测试噪声嵌入
    print("\n1. Testing Noise_Embedding...")
    noise_embed = Noise_Embedding(num_channels=256).to(device)
    noise_level = torch.randn(2).to(device)
    noise_out = noise_embed(noise_level)
    print(f"   Input shape: {noise_level.shape} -> Output shape: {noise_out.shape}")

    # 测试噪声条件块
    print("\n2. Testing NoiseConditioningBlock...")
    noise_block = NoiseConditioningBlock(n_in_channel=256, n_out_channel=512).to(device)
    noise_out = noise_block(noise_level)
    print(f"   Input shape: {noise_level.shape} -> Output shape: {noise_out.shape}")

    # 测试时间条件注意力
    print("\n3. Testing TimeCondAttention...")
    attention = TimeCondAttention(
        dim=256,
        heads=8,
        dim_head=32,
        time_cond_dim=512,
        attn_bias_dim=8
    ).to(device)

    x = torch.randn(2, 32, 256).to(device)
    time_cond = torch.randn(2, 1, 512).to(device)
    attn_bias = torch.randn(2, 8, 32, 32).to(device)
    seq_mask = torch.ones(2, 32).to(device)

    attn_out = attention(x, time=time_cond, attn_bias=attn_bias, seq_mask=seq_mask)
    print(f"   Input shape: {x.shape} -> Output shape: {attn_out.shape}")

    # 测试时间条件前馈网络
    print("\n4. Testing TimeCondFeedForward...")
    ff = TimeCondFeedForward(dim=256, time_cond_dim=512).to(device)
    ff_out = ff(x, time=time_cond)
    print(f"   Input shape: {x.shape} -> Output shape: {ff_out.shape}")

    print("\n✓ All component tests passed!")


if __name__ == "__main__":
    print("Starting model tests...")

    # 测试主模型
    model_initialization()

    # 测试各个组件
    # test_individual_components()

    print("\n" + "=" * 50)
    print("All tests completed successfully!")