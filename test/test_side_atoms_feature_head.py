import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.shattetnion.ShDecoderSidechain import SideAtomsFeatureHead


def run_once(B=2, N=5, A=10, hidden=256, num_classes=20, use_masks=True, device="cpu"):
    model = SideAtomsFeatureHead(A=A, hidden=hidden, num_classes=num_classes)
    model.to(device)
    model.eval()

    # Create dummy sidechain coordinates [B,N,A,3]
    X_sc = torch.randn(B, N, A, 3, device=device)

    # Optional masks
    atom_mask = None
    node_mask = None
    if use_masks:
        # Random atom mask with some missing atoms
        atom_mask = (torch.rand(B, N, A, device=device) > 0.2)
        # Make some residues padded (masked out entirely)
        node_mask = torch.ones(B, N, device=device, dtype=torch.bool)
        if N >= 2:
            node_mask[:, -1] = False  # last residue masked

    with torch.no_grad():
        logits, feat = model(X_sc, atom_mask=atom_mask, node_mask=node_mask)

    # Shape checks
    assert feat.shape == (B, N, hidden), f"feat shape mismatch: {feat.shape}"
    if num_classes > 0:
        assert logits is not None, "logits should not be None when num_classes > 0"
        assert logits.shape == (B, N, num_classes), f"logits shape mismatch: {logits.shape}"
    else:
        assert logits is None, "logits should be None when num_classes == 0"

    # Basic numeric sanity
    assert torch.isfinite(feat).all(), "Non-finite values in feat"
    if num_classes > 0:
        assert torch.isfinite(logits).all() or (node_mask is not None), "Non-finite values in logits"

    # If node_mask provided, verify masked residue logits are strongly negative (due to mask add)
    if num_classes > 0 and node_mask is not None:
        masked_logits = logits[:, -1]  # last residue was masked
        # Expect most values to be very negative (around -1e9)
        assert (masked_logits < -1e6).all(), "Masked node logits are not strongly suppressed"

    print("OK: shapes and masking behavior look correct.")


if __name__ == "__main__":
    run_once()
    # Also exercise without masks and with num_classes=0
    run_once(use_masks=False)
    run_once(num_classes=0)
    print("All SideAtomsFeatureHead tests passed.")


