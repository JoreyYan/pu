import torch
import torch.nn as nn
from typing import Optional, Tuple

def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.sqrt(torch.sum(v * v, dim=-1, keepdim=True) + eps)

def frames_from_backbone_openfold(
    X: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    OpenFold/AlphaFold-compatible frame from backbone:
      origin = CA
      p_neg_x_axis = N   (N lies on negative x-axis)
      p_xy_plane   = C   (C lies in xy-plane with +y)

    Args:
      X: (..., 3, 3) backbone coords in order (N, CA, C)
    Returns:
      R: (..., 3, 3) rotation matrix with columns [e0,e1,e2]
      t: (..., 3)    translation (CA)
    """
    N, CA, C = X.unbind(dim=-2)  # (...,3)

    e0 = _normalize(CA - N, eps)          # x-axis (CA -> N is -x, so CA-N is +x)
    # Wait, OpenFold definition:
    # p_neg_x_axis = N. So vector from Origin(CA) to N is along -x.
    # So vector (N - CA) is along -x? No.
    # Let's check Rigid.from_3_points logic:
    # e0 = normalize(origin - p_neg_x_axis) = normalize(CA - N)
    # So e0 points from N to CA. This is +x direction.
    # So N is at -x. Correct.
    
    e1 = C - CA
    e1 = e1 - e0 * torch.sum(e0 * e1, dim=-1, keepdim=True)  # remove x component
    e1 = _normalize(e1, eps)              # y-axis
    e2 = torch.cross(e0, e1, dim=-1)      # z-axis (right-handed)

    R = torch.stack([e0, e1, e2], dim=-1) # columns
    t = CA
    return R, t

def backbone_from_frame_openfold(
    R: torch.Tensor,
    t: torch.Tensor,
    template: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build (N,CA,C) from OpenFold/AlphaFold-compatible (R,t):
      X = t + R @ x_local

    Args:
      R: (...,3,3)
      t: (...,3)
      template: (3,3) rows are local coords for (N,CA,C)
    Returns:
      X: (...,3,3) in order (N,CA,C)
    """
    if template is None:
        # OpenFold-compatible local backbone template: N on -x, C in xy plane
        template = torch.tensor(
            [
                [-1.459, 0.0,   0.0],   # N
                [ 0.0,   0.0,   0.0],   # CA
                [ 0.547, 1.424, 0.0],   # C  (approx from 1.525A @ 111deg)
            ],
            dtype=R.dtype,
            device=R.device,
        )

    # einsum: (...,3,3) @ (A=3,3) -> (...,A=3,3)
    # R is rotation matrix (columns are axes). 
    # x_global = R @ x_local + t
    # x_local is (3,) column vector.
    # Here template is (A, 3) row vectors.
    # So we want (R @ template.T).T = template @ R.T
    # Or using einsum: ...ij, aj -> ...ai
    X = t.unsqueeze(-2) + torch.einsum("...ij,aj->...ai", R, template)
    return X

class OpenFoldFrameBuilder(nn.Module):
    """
    A FrameBuilder-like module but matched to OpenFold/AlphaFold Rigid.from_3_points
    for backbone (N,CA,C).
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        template = torch.tensor(
            [
                [1.459, 0.0,   0.0],   # N  (negative x)
                [ 0.0,   0.0,   0.0],   # CA
                [ -0.547, 1.424, 0.0],   # C  (xy plane)
            ],
            dtype=torch.float32,
        )
        self.register_buffer("t_atom", template)  # (3,3)

    def forward(self, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return backbone_from_frame_openfold(R, t, self.t_atom.to(R.dtype))

    def inverse(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return frames_from_backbone_openfold(X, eps=self.eps)
