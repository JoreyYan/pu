"""
完整的侧链几何质量检查
1. 所有共价键长（CB-CG, CG-CD, CD-NE等）
2. 键角
3. 侧链内clash
"""

import sys
sys.path.append('/home/junyu/project/pu')

import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

# 标准键长（单位：Å）
STANDARD_BOND_LENGTHS = {
    # C-C键
    'C-C': 1.53,
    # C-N键
    'C-N': 1.47,
    # C-O键
    'C-O': 1.43,
    # C-S键
    'C-S': 1.82,
}

# 侧链键连接定义（简化版，只检查主要键）
SIDECHAIN_BONDS = {
    'ARG': [('CB', 'CG', 'C-C'), ('CG', 'CD', 'C-C'), ('CD', 'NE', 'C-N'), ('NE', 'CZ', 'C-N')],
    'ASN': [('CB', 'CG', 'C-C'), ('CG', 'OD1', 'C-O'), ('CG', 'ND2', 'C-N')],
    'ASP': [('CB', 'CG', 'C-C'), ('CG', 'OD1', 'C-O'), ('CG', 'OD2', 'C-O')],
    'CYS': [('CB', 'SG', 'C-S')],
    'GLN': [('CB', 'CG', 'C-C'), ('CG', 'CD', 'C-C'), ('CD', 'OE1', 'C-O'), ('CD', 'NE2', 'C-N')],
    'GLU': [('CB', 'CG', 'C-C'), ('CG', 'CD', 'C-C'), ('CD', 'OE1', 'C-O'), ('CD', 'OE2', 'C-O')],
    'HIS': [('CB', 'CG', 'C-C'), ('CG', 'ND1', 'C-N'), ('CG', 'CD2', 'C-C')],
    'ILE': [('CB', 'CG1', 'C-C'), ('CB', 'CG2', 'C-C'), ('CG1', 'CD1', 'C-C')],
    'LEU': [('CB', 'CG', 'C-C'), ('CG', 'CD1', 'C-C'), ('CG', 'CD2', 'C-C')],
    'LYS': [('CB', 'CG', 'C-C'), ('CG', 'CD', 'C-C'), ('CD', 'CE', 'C-C'), ('CE', 'NZ', 'C-N')],
    'MET': [('CB', 'CG', 'C-C'), ('CG', 'SD', 'C-S'), ('SD', 'CE', 'C-S')],
    'PHE': [('CB', 'CG', 'C-C'), ('CG', 'CD1', 'C-C'), ('CG', 'CD2', 'C-C')],
    'PRO': [('CB', 'CG', 'C-C'), ('CG', 'CD', 'C-C')],
    'SER': [('CB', 'OG', 'C-O')],
    'THR': [('CB', 'OG1', 'C-O'), ('CB', 'CG2', 'C-C')],
    'TRP': [('CB', 'CG', 'C-C'), ('CG', 'CD1', 'C-C'), ('CG', 'CD2', 'C-C')],
    'TYR': [('CB', 'CG', 'C-C'), ('CG', 'CD1', 'C-C'), ('CG', 'CD2', 'C-C')],
    'VAL': [('CB', 'CG1', 'C-C'), ('CB', 'CG2', 'C-C')],
}

def check_sidechain_bonds(pdb_path):
    """检查侧链键长"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    all_bond_errors = []

    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()

                # 跳过GLY和ALA（没有复杂侧链）
                if res_name not in SIDECHAIN_BONDS:
                    continue

                # 获取该残基的所有原子坐标
                atom_coords = {}
                for atom in residue:
                    atom_coords[atom.get_name()] = atom.get_coord()

                # 检查定义的键
                for atom1_name, atom2_name, bond_type in SIDECHAIN_BONDS[res_name]:
                    if atom1_name in atom_coords and atom2_name in atom_coords:
                        coord1 = atom_coords[atom1_name]
                        coord2 = atom_coords[atom2_name]

                        actual_length = np.linalg.norm(coord1 - coord2)
                        expected_length = STANDARD_BOND_LENGTHS[bond_type]
                        error = actual_length - expected_length

                        all_bond_errors.append({
                            'res_name': res_name,
                            'bond': f'{atom1_name}-{atom2_name}',
                            'bond_type': bond_type,
                            'actual': actual_length,
                            'expected': expected_length,
                            'error': error,
                            'abs_error': abs(error),
                            'rel_error': abs(error) / expected_length
                        })

    return all_bond_errors

def check_sidechain_angles(pdb_path):
    """检查关键键角（简化：只检查C-C-C和C-C-N角度）"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    all_angles = []

    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()

                # 获取原子坐标
                atom_coords = {}
                for atom in residue:
                    atom_coords[atom.get_name()] = atom.get_coord()

                # 检查一些常见的三原子角度
                # CA-CB-CG
                if 'CA' in atom_coords and 'CB' in atom_coords and 'CG' in atom_coords:
                    v1 = atom_coords['CA'] - atom_coords['CB']
                    v2 = atom_coords['CG'] - atom_coords['CB']

                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cos_angle))

                    all_angles.append({
                        'res_name': res_name,
                        'angle': 'CA-CB-CG',
                        'value': angle_deg
                    })

    return all_angles

def check_sidechain_internal_clashes(pdb_path):
    """检查侧链内部clash（同一残基内，非相邻原子）"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    clashes = []

    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()

                # 获取所有侧链原子
                sidechain_atoms = []
                for atom in residue:
                    atom_name = atom.get_name()
                    # 侧链原子（排除主链N, CA, C, O）
                    if atom_name not in ['N', 'CA', 'C', 'O']:
                        sidechain_atoms.append({
                            'name': atom_name,
                            'coord': atom.get_coord()
                        })

                # 检查非相邻原子间的距离
                for i, atom1 in enumerate(sidechain_atoms):
                    for j in range(i+2, len(sidechain_atoms)):  # 跳过相邻原子
                        atom2 = sidechain_atoms[j]

                        dist = np.linalg.norm(atom1['coord'] - atom2['coord'])

                        # 如果距离太近（<2.0Å）认为是clash
                        if dist < 2.0:
                            clashes.append({
                                'res_name': res_name,
                                'atom1': atom1['name'],
                                'atom2': atom2['name'],
                                'dist': dist
                            })

    return clashes

def analyze_sample(sample_dir):
    """分析一个样本"""
    pdb_path = Path(sample_dir) / 'predicted.pdb'

    if not pdb_path.exists():
        return None

    # 1. 侧链键长
    bond_errors = check_sidechain_bonds(str(pdb_path))

    # 2. 键角
    angles = check_sidechain_angles(str(pdb_path))

    # 3. 侧链内部clash
    internal_clashes = check_sidechain_internal_clashes(str(pdb_path))

    return {
        'sample_name': Path(sample_dir).name,
        'bond_errors': bond_errors,
        'angles': angles,
        'internal_clashes': internal_clashes
    }

def main():
    experiments = [
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step10/val_seperated_Rm0_t0_step0_20251116_210156', '10步'),
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step100/val_seperated_Rm0_t0_step0_20251116_210400', '100步'),
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step500/val_seperated_Rm0_t0_step0_20251116_211051', '500步'),
    ]

    for exp_dir, name in experiments:
        print(f"\n{'='*80}")
        print(f"分析 {name} - 侧链几何质量")
        print(f"{'='*80}")

        exp_path = Path(exp_dir)
        sample_dirs = sorted([d for d in exp_path.iterdir()
                             if d.is_dir() and d.name.startswith('sample_')])

        all_bond_errors = []
        all_angles = []
        all_internal_clashes = []

        # 分析所有样本
        for sample_dir in sample_dirs:
            result = analyze_sample(sample_dir)
            if result:
                all_bond_errors.extend(result['bond_errors'])
                all_angles.extend(result['angles'])
                all_internal_clashes.extend(result['internal_clashes'])

        # 统计键长质量
        if all_bond_errors:
            print(f"\n1. 侧链键长质量（共 {len(all_bond_errors)} 个键）:")

            # 按键类型分组
            by_type = {}
            for err in all_bond_errors:
                bond_type = err['bond_type']
                if bond_type not in by_type:
                    by_type[bond_type] = []
                by_type[bond_type].append(err)

            for bond_type in sorted(by_type.keys()):
                bonds = by_type[bond_type]
                abs_errors = [b['abs_error'] for b in bonds]
                rel_errors = [b['rel_error'] for b in bonds]

                print(f"\n   {bond_type} 键（{len(bonds)}个）:")
                print(f"     期望长度: {bonds[0]['expected']:.3f} Å")
                print(f"     平均误差: {np.mean(abs_errors):.4f} Å ({np.mean(rel_errors)*100:.2f}%)")
                print(f"     标准差:   {np.std(abs_errors):.4f} Å")
                print(f"     最大误差: {np.max(abs_errors):.4f} Å")

                # 判断质量
                if np.mean(abs_errors) < 0.05:
                    print(f"     ✓ 质量很好")
                elif np.mean(abs_errors) < 0.1:
                    print(f"     ⚠️  有轻微偏差")
                else:
                    print(f"     ❌ 偏差较大")

            # 找最坏的键
            worst_bonds = sorted(all_bond_errors, key=lambda x: x['abs_error'], reverse=True)[:5]
            print(f"\n   最大偏差的5个键:")
            for i, bond in enumerate(worst_bonds):
                print(f"     {i+1}. {bond['res_name']} {bond['bond']}: {bond['actual']:.3f}Å (期望{bond['expected']:.3f}Å, 误差{bond['error']:+.3f}Å)")

        # 统计键角
        if all_angles:
            print(f"\n2. 键角质量（共 {len(all_angles)} 个角）:")
            angle_values = [a['value'] for a in all_angles]
            print(f"   CA-CB-CG角度:")
            print(f"     平均: {np.mean(angle_values):.1f}°")
            print(f"     标准差: {np.std(angle_values):.1f}°")
            print(f"     范围: [{np.min(angle_values):.1f}°, {np.max(angle_values):.1f}°]")

            # 理想值约110-115°（sp3杂化）
            expected_angle = 110.0
            angle_errors = [abs(a - expected_angle) for a in angle_values]
            print(f"     平均误差: {np.mean(angle_errors):.1f}°")

            if np.mean(angle_errors) < 10:
                print(f"     ✓ 质量很好")
            elif np.mean(angle_errors) < 20:
                print(f"     ⚠️  有偏差")
            else:
                print(f"     ❌ 偏差较大")

        # 统计侧链内部clash
        print(f"\n3. 侧链内部clash:")
        print(f"   总计: {len(all_internal_clashes)} 次")
        print(f"   平均每样本: {len(all_internal_clashes) / len(sample_dirs):.2f}")

        if len(all_internal_clashes) == 0:
            print(f"   ✓ 无侧链内部clash")
        elif len(all_internal_clashes) < len(sample_dirs) * 2:
            print(f"   ⚠️  有少量clash")
        else:
            print(f"   ❌ Clash较多")

        if all_internal_clashes:
            print(f"\n   示例（前5个）:")
            for i, clash in enumerate(all_internal_clashes[:5]):
                print(f"     {i+1}. {clash['res_name']}: {clash['atom1']}-{clash['atom2']} = {clash['dist']:.2f}Å")

    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
