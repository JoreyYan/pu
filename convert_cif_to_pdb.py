"""
Convert CIF files to PDB format
"""
import os
import glob
from Bio.PDB import MMCIFParser, PDBIO

# Paths
cif_dir = '/media/junyu/DATA/epoch163_native_structures'
pdb_dir = '/media/junyu/DATA/epoch163_native_structures_pdb'

# Create output directory
os.makedirs(pdb_dir, exist_ok=True)

# Get all CIF files
cif_files = glob.glob(os.path.join(cif_dir, '*.cif'))

print(f"Found {len(cif_files)} CIF files to convert")
print(f"Output directory: {pdb_dir}")
print("=" * 80)

# Initialize parser and writer
parser = MMCIFParser(QUIET=True)
io = PDBIO()

success_count = 0
failed = []

for cif_file in sorted(cif_files):
    pdb_id = os.path.basename(cif_file).replace('.cif', '')
    pdb_file = os.path.join(pdb_dir, f'{pdb_id}.pdb')

    try:
        # Parse CIF file
        structure = parser.get_structure(pdb_id, cif_file)

        # Write to PDB format
        io.set_structure(structure)
        io.save(pdb_file)

        print(f"✓ {pdb_id}: CIF → PDB")
        success_count += 1

    except Exception as e:
        print(f"✗ {pdb_id}: Failed - {str(e)}")
        failed.append(pdb_id)

print("=" * 80)
print(f"\nConversion Summary:")
print(f"  Total CIF files: {len(cif_files)}")
print(f"  Successfully converted: {success_count}")
print(f"  Failed: {len(failed)}")

if failed:
    print(f"\nFailed conversions:")
    for pdb_id in failed:
        print(f"  - {pdb_id}")

print(f"\nPDB files saved to: {pdb_dir}")
