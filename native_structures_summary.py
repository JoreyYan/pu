"""
Summary report of extracted native structures for epoch163 samples
"""
import os
import pandas as pd

# Paths
native_dir = '/media/junyu/DATA/epoch163_native_structures'
val_dir = '/media/junyu/DATA/pu5090weight/pdb__fbb_iga_simplified_attention_xlocal_2025-12-09_13-52-05/val_samples_epoch163'

# Get extracted CIF files
extracted_cifs = set([f.replace('.cif', '') for f in os.listdir(native_dir) if f.endswith('.cif')])

# Get all sample directories
sample_dirs = sorted([d for d in os.listdir(val_dir) if d.startswith('sample_')])

# Build summary
data = []
for sample_name in sample_dirs:
    # Extract PDB ID
    parts = sample_name.split('_')
    if len(parts) >= 2:
        pdb_id = parts[1]
        has_native = pdb_id in extracted_cifs
        data.append({
            'sample_name': sample_name,
            'pdb_id': pdb_id,
            'has_native': has_native,
            'native_file': f'{pdb_id}.cif' if has_native else 'NOT FOUND'
        })

df = pd.DataFrame(data)

print("=" * 100)
print("Native Structure Extraction Summary")
print("=" * 100)
print(f"\nTotal samples: {len(df)}")
print(f"Native structures found: {df['has_native'].sum()}")
print(f"Native structures missing: {(~df['has_native']).sum()}")
print(f"Coverage: {100 * df['has_native'].sum() / len(df):.1f}%")

print("\n" + "=" * 100)
print("Samples WITH native structures:")
print("=" * 100)
with_native = df[df['has_native']].copy()
for idx, row in with_native.iterrows():
    print(f"  ✓ {row['sample_name']:30s} → {row['native_file']}")

print("\n" + "=" * 100)
print("Samples WITHOUT native structures:")
print("=" * 100)
without_native = df[~df['has_native']].copy()
for idx, row in without_native.iterrows():
    print(f"  ✗ {row['sample_name']:30s} → {row['pdb_id']} (not found in mmcif directory)")

print("\n" + "=" * 100)
print("Next Steps:")
print("=" * 100)
print("""
1. The extracted native structures are in: /media/junyu/DATA/epoch163_native_structures/
2. You can use these to investigate the pLDDT discrepancy:
   - Sequence evaluation (using your model's sequences): pLDDT ≈ 76-78
   - FBB evaluation (using ProteinMPNN sequences): pLDDT ≈ 56-58
3. This suggests your model designs better sequences than ProteinMPNN!

Missing PDB IDs might be:
- Not yet released to PDB
- From other structural databases
- Need to check alternative sources
""")

# Save to CSV
output_csv = '/home/junyu/project/pu/native_structures_availability.csv'
df.to_csv(output_csv, index=False)
print(f"\nSummary saved to: {output_csv}")
print("=" * 100)
