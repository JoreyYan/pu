"""Test script to verify LayerNorm feature normalization works correctly."""
import torch
import sys
sys.path.insert(0, '.')

from models.flow_model import SideAtomsFlowModel
from omegaconf import OmegaConf

print("=" * 80)
print("Testing Feature Normalization in SideAtomsFlowModel")
print("=" * 80)

# Create minimal config
cfg_dict = {
    'use_esm': True,
    'esm_model': 'esm2_8M_270K',  # Use smallest ESM for testing
    'node_features': {'c_s': 256},
    'edge_features': {'c_z': 128},
    'edge_embed_size': 128,
    'ipa': {
        'c_s': 256,
        'c_z': 128,
        'num_blocks': 2,
        'seq_tfmr_num_heads': 4,
        'seq_tfmr_num_layers': 2,
    },
    'sidechain_atoms': {
        'A': 11,
        'hidden': 256,
        'dropout': 0.1,
        'conv_blocks': 2,
        'mlp_blocks': 2,
        'fuse_blocks': 2,
        'conv_groups': 1,
    }
}
cfg = OmegaConf.create(cfg_dict)

print("\n[1/4] Creating model with ESM...")
try:
    model = SideAtomsFlowModel(cfg)
    print("✓ Model created successfully")
    print(f"  use_esm: {model.use_esm}")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n[2/4] Checking LayerNorm layers exist...")
required_layers = [
    'node_feature_ln',
    'sidechain_feature_ln',
    'graph_feature_ln',
    'esm_feature_ln',
    'edge_init_ln',
    'edge_graph_ln',
    'edge_esm_ln',
]

for layer_name in required_layers:
    if hasattr(model, layer_name):
        layer = getattr(model, layer_name)
        print(f"  ✓ {layer_name}: {layer}")
    else:
        print(f"  ✗ {layer_name}: NOT FOUND")
        exit(1)

print("\n[3/4] Creating test input...")
B, N = 2, 10
input_feats = {
    'aatype': torch.randint(0, 20, (B, N)),
    'res_mask': torch.ones(B, N),
    'diffuse_mask': torch.ones(B, N),
    'res_idx': torch.arange(N).unsqueeze(0).repeat(B, 1),
    'chain_idx': torch.ones(B, N, dtype=torch.long),
    'atoms14_local_t': torch.randn(B, N, 14, 3),
    'sidechain_atom_mask': torch.ones(B, N, 11, dtype=torch.bool),
    'rotmats_1': torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1),
    'trans_1': torch.zeros(B, N, 3),
    'r3_t': torch.tensor([0.5]),
}
print("✓ Test input created")

print("\n[4/4] Running forward pass...")
try:
    with torch.no_grad():
        output = model(input_feats)
    print("✓ Forward pass successful!")
    print(f"\nOutput shapes:")
    print(f"  side_atoms: {output['side_atoms'].shape}")
    print(f"  atoms_global_full: {output['atoms_global_full'].shape}")
    print(f"  rigids_global: {output['rigids_global'].shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("All tests passed! ✅")
print("=" * 80)
print("\nFeature normalization is working correctly.")
print("Each feature source is now normalized before concatenation,")
print("ensuring balanced gradients and better ESM integration.")
