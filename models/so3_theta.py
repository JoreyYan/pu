import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Sequence


def permute_final_dims(tensor, inds):
    """Permute the final dimensions of a tensor."""
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t, no_dims):
    """Flatten the final dimensions of a tensor."""
    return t.reshape(t.shape[:-no_dims] + (-1,))


class IPA3DRoPE(nn.Module):
    def __init__(
            self,
            c_s: int,
            c_z: int,
            no_heads: int,
            c_hidden: int,
            inf: float = 1e5,
            eps: float = 1e-8,
            base_freq: float = 10000.0,**kwargs
    ):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.no_heads = no_heads
        self.c_hidden = c_hidden
        self.inf = inf
        self.eps = eps
        self.base_freq = base_freq

        # Standard QKV projections
        self.linear_q = nn.Linear(c_s, no_heads * c_hidden)
        self.linear_kv = nn.Linear(c_s, no_heads * c_hidden * 2)

        # Pair representation bias
        self.linear_b = nn.Linear(c_z, no_heads)

        # Down projection for pair features
        self.down_z = nn.Linear(c_z, c_z // 4)

        # Output projection
        self.linear_out = nn.Linear(
            no_heads * c_hidden , c_s
        )

        # 3D-RoPE frequencies
        # Generate frequencies for each dimension pair
        freqs = 1.0 / (base_freq ** (torch.arange(0, c_hidden, 2).float() / c_hidden))
        self.register_buffer('freqs', freqs)

        self.softmax = nn.Softmax(dim=-1)

    def compute_rotation_distance(self, R_i, R_j):
        """
        Compute rotation distance between rotation matrices.
        Args:
            R_i: [*, N_res, 3, 3] rotation matrices for queries
            R_j: [*, N_res, 3, 3] rotation matrices for keys
        Returns:
            [*, N_res, N_res] rotation angles between all pairs
        """
        # Expand dimensions for pairwise computation
        R_i_expanded = R_i.unsqueeze(-4)  # [*, N_res, 1, 3, 3]
        R_j_expanded = R_j.unsqueeze(-5)  # [*, 1, N_res, 3, 3]

        # Compute relative rotation: R_i^T @ R_j
        R_rel = torch.matmul(
            R_i_expanded.transpose(-1, -2),
            R_j_expanded
        )  # [*, N_res, N_res, 3, 3]

        # Extract rotation angle from relative rotation matrix
        # trace(R) = 1 + 2*cos(θ), so θ = arccos((trace-1)/2)
        trace_val = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)  # [*, N_res, N_res]
        cos_angle = (trace_val - 1) / 2
        cos_angle = torch.clamp(cos_angle, -1 + self.eps, 1 - self.eps)  # numerical stability
        angles = torch.acos(cos_angle)  # [*, N_res, N_res]

        return angles

    def apply_3d_rope(self, q, k, rigid_transforms):
        """
        Apply true 3D-RoPE by rotating Q and K vectors based on relative rotation angles.
        Simplified version using absolute rotation encoding for each position.
        Args:
            q: [*, N_res, H, C_hidden] queries
            k: [*, N_res, H, C_hidden] keys
            rigid_transforms: Rigid object with .get_rot_mats() method
        Returns:
            q_rot, k_rot: rotated queries and keys
        """
        B, N_res, H, C_hidden = q.shape

        # Ensure C_hidden is even for pairing
        assert C_hidden % 2 == 0, "C_hidden must be even for RoPE pairing"

        # Extract rotation matrices from Rigid object
        rotation_matrices = rigid_transforms.get_rot_mats()  # [*, N_res, 3, 3]

        # Extract absolute rotation angle for each position
        # trace(R) = 1 + 2*cos(θ), so θ = arccos((trace-1)/2)
        trace_val = torch.diagonal(rotation_matrices, dim1=-2, dim2=-1).sum(-1)  # [*, N_res]
        cos_angle = (trace_val - 1) / 2
        cos_angle = torch.clamp(cos_angle, -1 + self.eps, 1 - self.eps)  # numerical stability
        angles = torch.acos(cos_angle)  # [*, N_res] - absolute rotation angle for each position

        # Split q and k into even/odd pairs
        q_pairs = q.view(*q.shape[:-1], -1, 2)  # [*, N_res, H, C_hidden//2, 2]
        k_pairs = k.view(*k.shape[:-1], -1, 2)  # [*, N_res, H, C_hidden//2, 2]

        # Generate frequencies for each pair
        num_pairs = C_hidden // 2
        freqs = self.freqs[:num_pairs]  # [C_hidden//2]

        # Compute phases for each position and frequency
        # angles: [*, N_res] -> [*, N_res, 1]
        # freqs: [C_hidden//2] -> [1, 1, C_hidden//2]
        angles_expanded = angles.unsqueeze(-1)  # [*, N_res, 1]
        freqs_expanded = freqs.view(1, 1, -1)  # [1, 1, C_hidden//2]

        # Compute phases: [*, N_res, C_hidden//2]
        phases = angles_expanded * freqs_expanded  # [*, N_res, C_hidden//2]

        # Compute cos and sin
        cos_phases = torch.cos(phases)  # [*, N_res, C_hidden//2]
        sin_phases = torch.sin(phases)  # [*, N_res, C_hidden//2]

        # Expand for heads dimension: [*, N_res, 1, C_hidden//2]
        cos_phases = cos_phases.unsqueeze(-2)  # [*, N_res, 1, C_hidden//2]
        sin_phases = sin_phases.unsqueeze(-2)  # [*, N_res, 1, C_hidden//2]

        # Extract even and odd components
        q_even = q_pairs[..., 0]  # [*, N_res, H, C_hidden//2]
        q_odd = q_pairs[..., 1]  # [*, N_res, H, C_hidden//2]
        k_even = k_pairs[..., 0]  # [*, N_res, H, C_hidden//2]
        k_odd = k_pairs[..., 1]  # [*, N_res, H, C_hidden//2]

        # Apply 2D rotations (RoPE-style rotation for each position)
        # Broadcasting: [*, N_res, H, C_hidden//2] * [*, N_res, 1, C_hidden//2]
        q_rot_even = q_even * cos_phases - q_odd * sin_phases  # [*, N_res, H, C_hidden//2]
        q_rot_odd = q_even * sin_phases + q_odd * cos_phases  # [*, N_res, H, C_hidden//2]

        k_rot_even = k_even * cos_phases - k_odd * sin_phases  # [*, N_res, H, C_hidden//2]
        k_rot_odd = k_even * sin_phases + k_odd * cos_phases  # [*, N_res, H, C_hidden//2]

        # Recombine even and odd components
        q_rot_pairs = torch.stack([q_rot_even, q_rot_odd], dim=-1)  # [*, N_res, H, C_hidden//2, 2]
        k_rot_pairs = torch.stack([k_rot_even, k_rot_odd], dim=-1)  # [*, N_res, H, C_hidden//2, 2]

        # Reshape back to original dimensions
        q_rot = q_rot_pairs.view(*q.shape)  # [*, N_res, H, C_hidden]
        k_rot = k_rot_pairs.view(*k.shape)  # [*, N_res, H, C_hidden]

        return q_rot, k_rot

    def forward(
            self,
            s: torch.Tensor,
            z: Optional[torch.Tensor],
            r,  # Rigid transformation object
            mask: torch.Tensor,
            _offload_inference: bool = False,
            _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s: [*, N_res, C_s] single representation
            z: [*, N_res, N_res, C_z] pair representation
            r: [*, N_res] transformation object (contains rotation matrices)
            mask: [*, N_res] mask
        Returns:
            [*, N_res, C_s] single representation update
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        ##########################
        # Compute attention scores
        ##########################

        # Extract rotation matrices from rigid transformation
        rotation_matrices = r._rots  # [*, N_res, 3, 3]

        # Apply 3D-RoPE to rotate Q and K vectors
        q_rot, k_rot = self.apply_3d_rope(q, k, rotation_matrices)

        # Compute attention scores with rotated Q and K
        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q_rot, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k_rot, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))

        # Add pair representation bias
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        if _offload_inference:
            z[0] = z[0].cpu()

        # Apply mask
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        if _offload_inference:
            z[0] = z[0].to(o.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0])
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        # Combine outputs
        # o_feats = [o, o_pair]

        # [*, N_res, C_s]
        s = self.linear_out(o)

        return s