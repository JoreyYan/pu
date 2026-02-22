"""EllipsoidDecoderV2: Two-head decoder (node_embed + ellipsoid → aatype + atom14).

Compared to V1 (ellipsoid_decoder.py) which takes aatype as **input**,
V2 **predicts** aatype from the combined node_embed + ellipsoid features,
then uses that prediction (or GT aatype via teacher forcing) to predict atom14.

Input per residue:
    node_embed  [C=256]: IGA trunk output (rich spatial context from 6 layers)
    scaling_log [3]:     log-scale of ellipsoid axes
    local_mean  [3]:     ellipsoid center offset in backbone frame

Output per residue:
    aa_logits    [20]:    amino acid type logits
    atom14_local [14, 3]: atom positions in backbone-local frame
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from openfold.np import residue_constants


class MLPResBlock(nn.Module):
    """Residual MLP block with pre-norm."""
    def __init__(self, d_in, d_hidden, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_in)
        self.ln = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = F.gelu(self.fc1(x))
        y = self.dropout(y)
        y = self.fc2(y)
        return self.ln(x + y)


class EllipsoidDecoderV2(nn.Module):
    """Two-head decoder: node_embed + ellipsoid → aatype + atom14.

    Architecture:
        Input: node_embed[C] + scaling_log[3] + local_mean[3] = (C+6)D
          ↓ Shared stem: LayerNorm → Linear → GELU → d_model
          ↓ num_shared_blocks × MLPResBlock
          ├─→ aa_head: LayerNorm → Linear → Linear → 20 logits
          └─→ atom14_head: concat(shared_feat, aa_embed) → num_atom14_blocks × MLPResBlock → 14×3
              Training: teacher forcing (GT aatype)
              Inference: predicted aatype from aa_head
    """

    def __init__(
        self,
        c_in: int = 256,
        d_model: int = 256,
        num_shared_blocks: int = 2,
        num_atom14_blocks: int = 2,
        aatype_embed_dim: int = 64,
        num_aa_types: int = 21,
        dropout: float = 0.0,
        out_range: float = 16.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.out_range = out_range
        self.num_aa_types = num_aa_types

        d_input = c_in + 3 + 3  # node_embed + scaling_log + local_mean

        # Shared stem
        self.stem = nn.Sequential(
            nn.LayerNorm(d_input),
            nn.Linear(d_input, d_model),
            nn.GELU(),
        )

        # Shared blocks
        self.shared_blocks = nn.ModuleList([
            MLPResBlock(d_model, d_model * 4, dropout=dropout)
            for _ in range(num_shared_blocks)
        ])

        # AA head
        self.aa_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 20),
        )

        # Atom14 branch
        self.aa_embed = nn.Embedding(num_aa_types, aatype_embed_dim)
        self.atom14_proj = nn.Sequential(
            nn.LayerNorm(d_model + aatype_embed_dim),
            nn.Linear(d_model + aatype_embed_dim, d_model),
            nn.GELU(),
        )
        self.atom14_blocks = nn.ModuleList([
            MLPResBlock(d_model, d_model * 4, dropout=dropout)
            for _ in range(num_atom14_blocks)
        ])
        self.atom14_head = nn.Linear(d_model, 14 * 3)

        # Initialize output near zero for stable training start
        nn.init.zeros_(self.atom14_head.weight)
        nn.init.zeros_(self.atom14_head.bias)

        # Register atom14 mask per residue type
        atom14_mask = torch.tensor(
            residue_constants.restype_atom14_mask, dtype=torch.float32
        )
        if atom14_mask.shape[0] == 20:
            atom14_mask = torch.cat([
                atom14_mask,
                torch.zeros(1, 14, dtype=torch.float32)
            ], dim=0)
        self.register_buffer('atom14_mask_per_type', atom14_mask)

    def forward(
        self,
        node_embed: torch.Tensor,   # [B, N, C]
        scaling_log: torch.Tensor,   # [B, N, 3]
        local_mean: torch.Tensor,    # [B, N, 3]
        aatype: torch.Tensor | None = None,  # [B, N] int, None = use predicted
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with:
            aa_logits:    [B, N, 20]
            atom14_local: [B, N, 14, 3]
        """
        B, N = scaling_log.shape[:2]

        # Concatenate inputs
        feat = torch.cat([node_embed, scaling_log, local_mean], dim=-1)

        # Shared stem + blocks
        x = self.stem(feat)
        for blk in self.shared_blocks:
            x = blk(x)  # [B, N, d_model]

        # AA prediction
        aa_logits = self.aa_head(x)  # [B, N, 20]

        # Determine aatype for atom14 branch
        if aatype is not None:
            aa_idx = aatype.clamp(0, self.num_aa_types - 1).long()
        else:
            aa_idx = aa_logits.argmax(dim=-1).clamp(0, self.num_aa_types - 1)
        aa_emb = self.aa_embed(aa_idx)

        # Atom14 branch
        atom_feat = torch.cat([x, aa_emb], dim=-1)
        atom_feat = self.atom14_proj(atom_feat)
        for blk in self.atom14_blocks:
            atom_feat = blk(atom_feat)
        out = self.atom14_head(atom_feat)
        out = torch.tanh(out) * self.out_range
        atom14_local = out.view(B, N, 14, 3)

        # Mask invalid atoms
        type_mask = self.atom14_mask_per_type[aa_idx]
        atom14_local = atom14_local * type_mask.unsqueeze(-1)

        return {
            'aa_logits': aa_logits,
            'atom14_local': atom14_local,
        }

    def get_atom14_mask(self, aatype: torch.Tensor) -> torch.Tensor:
        """Get per-residue atom14 existence mask."""
        aatype_clamped = aatype.clamp(0, self.num_aa_types - 1).long()
        return self.atom14_mask_per_type[aatype_clamped]
