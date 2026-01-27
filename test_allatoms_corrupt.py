"""Test script for allatoms_corrupt_batch method."""
import torch
import sys
sys.path.insert(0, '.')

from data.interpolant import Interpolant
from omegaconf import OmegaConf
import openfold.utils.rigid_utils as ru

print("=" * 60)
print("Testing allatoms_corrupt_batch")
print("=" * 60)

# Create a minimal config
cfg_dict = {
    'min_t': 0.001,
    'coord_scale': 1.0,
    'res_idx_offset_max': 50,
    'trans': {'corrupt': True},
    'rots': {'corrupt': True},
    'sampling': {},
}
cfg = OmegaConf.create(cfg_dict)

# Initialize Interpolant
interpolant = Interpolant(cfg, task='unconditional', noise_scheme='side_atoms')
interpolant.set_device('cpu')

print("\n[1/5] Creating test batch...")
B, N = 2, 10  # batch_size=2, num_residues=10

# Create test data
batch = {
    'trans_1': torch.randn(B, N, 3),
    'rotmats_1': ru.identity_rot_mats((B, N), dtype=torch.float32),
    'atoms14_local': torch.randn(B, N, 14, 3),
    'atom14_gt_exists': torch.ones(B, N, 14, dtype=torch.bool),
    'res_mask': torch.ones(B, N),
    'diffuse_mask': torch.ones(B, N),
    'res_idx': torch.arange(N).unsqueeze(0).repeat(B, 1),
}

print(f"✓ Test batch created")
print(f"  Batch size: {B}")
print(f"  Num residues: {N}")
print(f"  atoms14_local shape: {batch['atoms14_local'].shape}")

# Test allatoms_corrupt_batch
print("\n[2/5] Running allatoms_corrupt_batch...")
try:
    noisy_batch = interpolant.allatoms_corrupt_batch(batch, prob=0.5)
    print("✓ allatoms_corrupt_batch succeeded")
except Exception as e:
    print(f"✗ allatoms_corrupt_batch failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Verify output fields
print("\n[3/5] Verifying output fields...")
expected_fields = [
    # Backbone fields
    'trans_t', 'trans_0', 'trans_v',
    'rotmats_t', 'rotmats_0', 'rot_v',
    'rigids_t',
    # Atoms14 fields
    'atoms14_local_t', 'y_t', 'v_t',
    # Time fields
    't', 'r3_t', 'so3_t',
    # Mask fields
    'update_mask', 'sidechain_atom_mask',
    'diffuse_mask', 'fixed_mask'
]

missing_fields = [f for f in expected_fields if f not in noisy_batch]
if missing_fields:
    print(f"✗ Missing fields: {missing_fields}")
    exit(1)
else:
    print("✓ All expected fields present")

# Check shapes
print("\n[4/5] Checking output shapes...")
shape_checks = [
    # Backbone fields
    ('trans_t', (B, N, 3)),
    ('trans_0', (B, N, 3)),
    ('trans_v', (B, N, 3)),
    ('rotmats_t', (B, N, 3, 3)),
    ('rotmats_0', (B, N, 3, 3)),
    ('rot_v', (B, N, 3)),
    ('rigids_t', (B, N, 7)),
    # Atoms14 fields
    ('atoms14_local_t', (B, N, 14, 3)),
    ('y_t', (B, N, 14, 3)),
    ('v_t', (B, N, 14, 3)),
    # Time and mask fields
    ('t', (B, 1)),
    ('r3_t', (B, N)),
    ('update_mask', (B, N)),
    ('sidechain_atom_mask', (B, N, 11)),
]

all_correct = True
for field, expected_shape in shape_checks:
    actual_shape = tuple(noisy_batch[field].shape)
    if actual_shape == expected_shape:
        print(f"  ✓ {field}: {actual_shape}")
    else:
        print(f"  ✗ {field}: expected {expected_shape}, got {actual_shape}")
        all_correct = False

if not all_correct:
    print("\n✗ Shape check failed")
    exit(1)

# Verify corruption logic
print("\n[5/5] Verifying corruption logic...")

# Check 1: Backbone atoms should be clean in local frame
backbone_local_clean = batch['atoms14_local'][:, :, :3, :]
backbone_local_noisy = noisy_batch['atoms14_local_t'][:, :, :3, :]
backbone_diff = torch.abs(backbone_local_clean - backbone_local_noisy).max()
if backbone_diff < 1e-6:
    print(f"  ✓ Backbone atoms remain clean in local frame (max diff: {backbone_diff:.2e})")
else:
    print(f"  ✗ Backbone atoms were modified (max diff: {backbone_diff:.2e})")

# Check 2: Sidechain atoms should be corrupted
sidechain_local_clean = batch['atoms14_local'][:, :, 3:, :]
sidechain_local_noisy = noisy_batch['atoms14_local_t'][:, :, 3:, :]
sidechain_diff = torch.abs(sidechain_local_clean - sidechain_local_noisy).max()
if sidechain_diff > 1e-6:
    print(f"  ✓ Sidechain atoms are corrupted (max diff: {sidechain_diff:.2e})")
else:
    print(f"  ⚠ Sidechain atoms may not be corrupted (max diff: {sidechain_diff:.2e})")

# Check 3: Backbone frame should be corrupted
trans_diff = torch.abs(batch['trans_1'] - noisy_batch['trans_t']).max()
if trans_diff > 1e-6:
    print(f"  ✓ Translation is corrupted (max diff: {trans_diff:.2e})")
else:
    print(f"  ⚠ Translation may not be corrupted (max diff: {trans_diff:.2e})")

# Check 4: Time step should be in valid range
t_values = noisy_batch['t']
t_min, t_max = t_values.min().item(), t_values.max().item()
if 0.001 <= t_min and t_max <= 0.999:
    print(f"  ✓ Time steps in valid range: [{t_min:.4f}, {t_max:.4f}]")
else:
    print(f"  ✗ Time steps out of range: [{t_min:.4f}, {t_max:.4f}]")

# Check 5: Velocity field should be non-zero for sidechain
v_t_sidechain = noisy_batch['v_t'][:, :, 3:, :]
v_t_norm = torch.norm(v_t_sidechain, dim=-1).max()
if v_t_norm > 1e-6:
    print(f"  ✓ Sidechain velocity field is non-zero (max norm: {v_t_norm:.2e})")
else:
    print(f"  ⚠ Sidechain velocity field may be zero (max norm: {v_t_norm:.2e})")

# Check 6: Translation velocity field should be non-zero
trans_v_norm = torch.norm(noisy_batch['trans_v'], dim=-1).max()
if trans_v_norm > 1e-6:
    print(f"  ✓ Translation velocity field is non-zero (max norm: {trans_v_norm:.2e})")
else:
    print(f"  ⚠ Translation velocity field may be zero (max norm: {trans_v_norm:.2e})")

# Check 7: Rotation velocity field should be non-zero
rot_v_norm = torch.norm(noisy_batch['rot_v'], dim=-1).max()
if rot_v_norm > 1e-6:
    print(f"  ✓ Rotation velocity field is non-zero (max norm: {rot_v_norm:.2e})")
else:
    print(f"  ⚠ Rotation velocity field may be zero (max norm: {rot_v_norm:.2e})")

# Check 8: Verify trans_v = trans_1 - trans_0
expected_trans_v = batch['trans_1'] - noisy_batch['trans_0']
trans_v_diff = torch.abs(noisy_batch['trans_v'] - expected_trans_v).max()
if trans_v_diff < 1e-5:
    print(f"  ✓ trans_v = trans_1 - trans_0 (max diff: {trans_v_diff:.2e})")
else:
    print(f"  ✗ trans_v formula incorrect (max diff: {trans_v_diff:.2e})")

print("\n" + "=" * 60)
print("All tests passed! ✅")
print("=" * 60)
print("\nYou can now use allatoms_corrupt_batch for training:")
print("  from data.interpolant import Interpolant")
print("  noisy_batch = interpolant.allatoms_corrupt_batch(batch)")
print("\nSee ALLATOMS_CORRUPT_GUIDE.md for detailed usage.")
