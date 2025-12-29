"""
Extract native structure CIF files for epoch163 validation samples
"""
import os
import shutil
import glob
from pathlib import Path

# Paths
val_dir = '/media/junyu/DATA/pu5090weight/pdb__fbb_iga_simplified_attention_xlocal_2025-12-09_13-52-05/val_samples_epoch163'
mmcif_dir = '/media/junyu/DATA/mmcif/gzipmmcif'
output_dir = '/media/junyu/DATA/epoch163_native_structures'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Get all sample directories
sample_dirs = sorted([d for d in os.listdir(val_dir) if d.startswith('sample_')])

print(f"Found {len(sample_dirs)} samples in {val_dir}")
print(f"Searching for CIF files in {mmcif_dir}")
print(f"Output directory: {output_dir}")
print("=" * 80)

# Extract PDB IDs and search for CIF files
found_count = 0
not_found = []

for sample_name in sample_dirs:
    # Extract PDB ID: sample_1kzf_000021 -> 1kzf
    parts = sample_name.split('_')
    if len(parts) >= 2:
        pdb_id = parts[1]
    else:
        print(f"Warning: Cannot extract PDB ID from {sample_name}")
        continue

    # Try different possible locations for the CIF file
    # Files are organized in subdirectories by middle two chars (2nd and 3rd chars)
    possible_paths = [
        # Subdirectory based on middle two chars: kz/1kzf.cif
        os.path.join(mmcif_dir, pdb_id[1:3], f'{pdb_id}.cif'),
        # Also check for PDB format
        os.path.join(mmcif_dir, pdb_id[1:3], f'{pdb_id}.pdb'),
        # Direct path: 1kzf.cif
        os.path.join(mmcif_dir, f'{pdb_id}.cif'),
    ]

    # Search for the file
    found = False
    for path in possible_paths:
        if os.path.exists(path):
            # Copy to output directory, preserving the extension
            ext = os.path.splitext(path)[1]
            dest_path = os.path.join(output_dir, f'{pdb_id}{ext}')
            shutil.copy2(path, dest_path)
            print(f"✓ {pdb_id}: Found and copied {os.path.basename(path)}")
            found_count += 1
            found = True
            break

    if not found:
        not_found.append(pdb_id)
        print(f"✗ {pdb_id}: Not found")

print("=" * 80)
print(f"\nSummary:")
print(f"  Total samples: {len(sample_dirs)}")
print(f"  CIF files found: {found_count}")
print(f"  CIF files not found: {len(not_found)}")

if not_found:
    print(f"\nMissing PDB IDs:")
    for pdb_id in not_found:
        print(f"  - {pdb_id}")

print(f"\nAll found CIF files copied to: {output_dir}")
