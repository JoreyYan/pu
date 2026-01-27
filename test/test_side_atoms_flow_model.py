import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from types import SimpleNamespace

from models.flow_model import SideAtomsFlowModel


def create_dummy_config():
    """Create a dummy configuration for testing"""
    config = SimpleNamespace()
    
    # IPA configuration
    config.ipa = SimpleNamespace()
    config.ipa.num_blocks = 2
    config.ipa.c_s = 256
    config.ipa.no_heads = 8
    config.ipa.seq_tfmr_num_heads = 8
    config.ipa.seq_tfmr_num_layers = 2
    
    # Node features
    config.node_features = SimpleNamespace()
    config.node_features.c_s = 256
    config.node_features.c_t = 64
    config.node_features.c_p = 64
    config.node_features.c = 256
    
    # Edge features
    config.edge_features = SimpleNamespace()
    config.edge_features.c_s = 256
    config.edge_features.c_t = 64
    config.edge_features.c_p = 64
    config.edge_features.c = 256
    
    # Model configuration
    config.edge_embed_size = 128
    
    # Sidechain atoms configuration
    config.sidechain_atoms = SimpleNamespace()
    config.sidechain_atoms.A = 10
    config.sidechain_atoms.hidden = 128
    config.sidechain_atoms.dropout = 0.1
    config.sidechain_atoms.conv_blocks = 2
    config.sidechain_atoms.mlp_blocks = 2
    config.sidechain_atoms.fuse_blocks = 2
    config.sidechain_atoms.conv_groups = 1
    
    return config


def create_dummy_input(B=2, N=10, device="cpu"):
    """Create dummy input features for testing"""
    input_feats = {
        'res_mask': torch.ones(B, N, device=device, dtype=torch.bool),
        'diffuse_mask': torch.ones(B, N, device=device, dtype=torch.bool),
        'res_idx': torch.arange(N, device=device, dtype=torch.float32)[None].repeat(B, 1),
        'chain_idx': torch.zeros(B, N, device=device, dtype=torch.long),
        'so3_t': torch.randn(B, N, device=device),
        'r3_t': torch.randn(B, N, device=device),
        'trans_t': torch.randn(B, N, 3, device=device),
        'rotmats_t': torch.randn(B, N, 3, 3, device=device),
    }
    
    # Add sidechain atom data
    input_feats['atoms14_local_t'] = torch.randn(B, N, 14, 3, device=device)
    input_feats['atom14_gt_exists'] = torch.ones(B, N, 14, device=device, dtype=torch.bool)
    
    # Make some atoms missing
    input_feats['atom14_gt_exists'][:, :, 5:] = False  # Make some sidechain atoms missing
    
    return input_feats


def test_side_atoms_flow_model():
    """Test the SideAtomsFlowModel with and without sidechain features"""
    print("Testing SideAtomsFlowModel...")
    
    config = create_dummy_config()
    model = SideAtomsFlowModel(config)
    model.eval()
    
    B, N = 2, 8
    device = "cpu"
    
    # Test with sidechain features
    print("Testing with sidechain features...")
    input_feats = create_dummy_input(B, N, device)
    
    with torch.no_grad():
        output = model(input_feats)
    
    # Check output shapes
    assert 'pred_trans' in output, "Missing pred_trans in output"
    assert 'pred_rotmats' in output, "Missing pred_rotmats in output"
    assert output['pred_trans'].shape == (B, N, 3), f"Wrong pred_trans shape: {output['pred_trans'].shape}"
    assert output['pred_rotmats'].shape == (B, N, 3, 3), f"Wrong pred_rotmats shape: {output['pred_rotmats'].shape}"
    
    # Check for finite values
    assert torch.isfinite(output['pred_trans']).all(), "Non-finite values in pred_trans"
    assert torch.isfinite(output['pred_rotmats']).all(), "Non-finite values in pred_rotmats"
    
    print("✓ With sidechain features: OK")
    
    # Test without sidechain features
    print("Testing without sidechain features...")
    input_feats_no_sidechain = create_dummy_input(B, N, device)
    del input_feats_no_sidechain['atoms14_local_t']
    del input_feats_no_sidechain['atom14_gt_exists']
    
    with torch.no_grad():
        output_no_sidechain = model(input_feats_no_sidechain)
    
    # Check output shapes
    assert output_no_sidechain['pred_trans'].shape == (B, N, 3), f"Wrong pred_trans shape: {output_no_sidechain['pred_trans'].shape}"
    assert output_no_sidechain['pred_rotmats'].shape == (B, N, 3, 3), f"Wrong pred_rotmats shape: {output_no_sidechain['pred_rotmats'].shape}"
    
    print("✓ Without sidechain features: OK")
    
    # Test with partial sidechain data
    print("Testing with partial sidechain data...")
    input_feats_partial = create_dummy_input(B, N, device)
    # Remove atom14_gt_exists to test missing mask
    del input_feats_partial['atom14_gt_exists']
    
    with torch.no_grad():
        output_partial = model(input_feats_partial)
    
    assert output_partial['pred_trans'].shape == (B, N, 3), f"Wrong pred_trans shape: {output_partial['pred_trans'].shape}"
    
    print("✓ With partial sidechain data: OK")
    
    print("All SideAtomsFlowModel tests passed!")


if __name__ == "__main__":
    test_side_atoms_flow_model()

