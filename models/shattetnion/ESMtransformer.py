# esm/esm/minimal_rope_transformer.py
# Minimal ESM-style Transformer encoder with Rotary Positional Embeddings (RoPE).
# Uses only modules you provided in esm/esm/modules.py.

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn

# 只引用你提供过的模块
from esm.modules import (
    ESM1LayerNorm,
    TransformerLayer,
)


class MinimalRoPETransformer(nn.Module):
    """
    一个最小的 ESM 风格 Transformer 编码器，仅使用旋转位置编码（RoPE）。
    - 不包含 Learned/Sinusoidal 位置编码
    - 不包含 LM Head / Contact Head
    - 支持 causal mask 与 padding mask
    - 层内归一化与注意力/FFN 实现完全复用你给的 TransformerLayer

    Args:
        num_layers: Transformer 层数
        embed_dim: 通道维度
        ffn_dim: FFN 隐层维度
        heads: 注意力头数
        dropout: dropout 概率（作用于各层内部）
        final_layer_norm: 是否在堆叠结束后再做一次 LayerNorm
    """

    def __init__(
        self,
        num_layers: int = 6,
        embed_dim: int = 512,
        ffn_dim: int = 2048,
        heads: int = 8,
        dropout: float = 0.1,
        final_layer_norm: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.heads = heads

        # 堆叠 N 个 ESM 的 TransformerLayer；
        # 关键：use_rotary_embeddings=True，仅用 RoPE，不使用外部位置嵌入
        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim=embed_dim,
                ffn_embed_dim=ffn_dim,
                attention_heads=heads,
                add_bias_kv=True,
                use_esm1b_layer_norm=False,   # 不用 ESM1b fused LN，保持最小依赖
                use_rotary_embeddings=True,   # 🔑 只用 RoPE
            )
            for _ in range(num_layers)
        ])

        self.final_ln = ESM1LayerNorm(embed_dim) if final_layer_norm else nn.Identity()

    @staticmethod
    def build_causal_mask(L: int, device: torch.device) -> torch.Tensor:
        """
        生成上三角 True 的布尔 mask，形状 [L, L]。
        True 表示不可见（被 mask）。
        """
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(
            self,
            x: torch.Tensor,
            *,
            causal: bool = False,
            padding_mask: Optional[torch.Tensor] = None,
            need_head_weights: bool = False,
    ) -> Dict[str, Any]:
        """
        x: [B, L, C]  -> 内部会转换为 [L, B, C] 以符合 ESM 的注意力实现
        padding_mask: [B, L] (bool; True=mask)
        """
        B, L, C = x.shape
        assert C == self.embed_dim, f"Expected last dim {self.embed_dim}, got {C}"

        # 准备 mask
        if padding_mask is not None:
            # 确保形状与 dtype
            assert padding_mask.shape == (B, L), f"padding_mask shape {padding_mask.shape} != {(B, L)}"
            padding_mask = padding_mask.to(dtype=torch.bool, device=x.device, non_blocking=True)
        attn_mask = self.build_causal_mask(L, x.device) if causal else None  # [L, L], bool

        # [B, L, C] -> [L, B, C]
        h = x.transpose(0, 1).contiguous()

        all_attn = []
        for layer in self.layers:
            # ESM 的 MultiheadAttention 期望 [T, B, C] 输入，以及
            # attn_mask: [T, T], key_padding_mask: [B, T]
            h, attn = layer(
                h,
                self_attn_mask=attn_mask,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if need_head_weights:
                # 规范成 [B, H, L, L]
                if attn.dim() != 4:
                    raise ValueError(f"Unexpected attention rank: {attn.shape}")
                if attn.size(0) == B:
                    # [B, H, L, L]
                    attn_bhll = attn
                elif attn.size(1) == B:
                    # [H, B, L, L] -> [B, H, L, L]
                    attn_bhll = attn.permute(1, 0, 2, 3).contiguous()
                else:
                    # 有些实现会给 [B, L, L, H]
                    attn_bhll = attn.permute(0, 3, 1, 2).contiguous()
                all_attn.append(attn_bhll)

        # [L, B, C] -> [B, L, C]
        h = h.transpose(0, 1).contiguous()
        h = self.final_ln(h)

        out: Dict[str, Any] = {"hidden_states": h}
        if need_head_weights and len(all_attn) > 0:
            # 堆叠到 [B, num_layers, H, L, L]
            out["attentions"] = torch.stack(all_attn, dim=1)
        return out


# -----------------------------
# 简单用例（示范）
# -----------------------------
if __name__ == "__main__":
    B, L, C = 2, 128, 512
    model = MinimalRoPETransformer(
        num_layers=6,
        embed_dim=C,
        ffn_dim=2048,
        heads=8,
        dropout=0.1,
        final_layer_norm=True,
    )

    # 你自己的外部嵌入：例如 token embedding 或线性投影到 C 维
    x = torch.randn(B, L, C)

    # padding mask：True=mask（例如后 10 个为 pad）
    pad = torch.zeros(B, L, dtype=torch.bool)
    pad[:, -10:] = True

    out = model(x, causal=False, padding_mask=pad, need_head_weights=True)
    print(out["hidden_states"].shape)   # [B, L, C]
    print(out["attentions"].shape)      # [B, num_layers, heads, L, L]
