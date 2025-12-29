"""
正确的几何质量检查（修复了之前的bug）
"""

import sys
sys.path.append('/home/junyu/project/pu')

import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

def check_ca_cb_bonds(pdb_path):
    """正确地检查CA-CB键长"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    ca_cb_bonds = []

    for model in structure:
        for chain in model:
            for residue in chain:
                ca_coord = None
                cb_coord = None

                for atom in residue:
                    if atom.get_name() == 'CA':
                        ca_coord = atom.get_coord()
                    elif atom.get_name() == 'CB':
                        cb_coord = atom.get_coord()

                # 如果该残基同时有CA和CB
                if ca_coord is not None and cb_coord is not None:
                    dist = np.linalg.norm(ca_coord - cb_coord)
                    ca_cb_bonds.append(dist)

    return ca_cb_bonds

def check_inter_residue_clashes(pdb_path):
    """
    检查残基间clash（不包括同一残基内的原子）
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    # 提取所有原子，记录所属残基
    all_atoms = []
    for model in structure:
        for chain in model:
            for res_idx, residue in enumerate(chain):
                for atom in residue:
                    all_atoms.append({
                        'res_idx': res_idx,
                        'res_name': residue.get_resname(),
                        'atom_name': atom.get_name(),
                        'coord': atom.get_coord()
                    })

    # 检查不同残基之间的距离
    severe_clashes = []
    mild_clashes = []

    for i, atom1 in enumerate(all_atoms):
        for j in range(i+1, len(all_atoms)):
            atom2 = all_atoms[j]

            # 只检查不同残基的原子
            if atom1['res_idx'] != atom2['res_idx']:
                dist = np.linalg.norm(atom1['coord'] - atom2['coord'])

                # 严重clash: < 2.0Å
                if dist < 2.0:
                    severe_clashes.append({
                        'res1': atom1['res_idx'],
                        'atom1': f"{atom1['res_name']}-{atom1['atom_name']}",
                        'res2': atom2['res_idx'],
                        'atom2': f"{atom2['res_name']}-{atom2['atom_name']}",
                        'dist': dist
                    })
                # 轻微clash: 2.0-2.5Å
                elif dist < 2.5:
                    mild_clashes.append({
                        'res1': atom1['res_idx'],
                        'atom1': f"{atom1['res_name']}-{atom1['atom_name']}",
                        'res2': atom2['res_idx'],
                        'atom2': f"{atom2['res_name']}-{atom2['atom_name']}",
                        'dist': dist
                    })

    return severe_clashes, mild_clashes

def analyze_sample(sample_dir):
    """分析一个样本"""
    sample_path = Path(sample_dir)
    pdb_path = sample_path / 'predicted.pdb'

    if not pdb_path.exists():
        return None

    # 1. CA-CB键长
    ca_cb_bonds = check_ca_cb_bonds(str(pdb_path))

    # 2. Clash检查
    severe_clashes, mild_clashes = check_inter_residue_clashes(str(pdb_path))

    return {
        'sample_name': sample_path.name,
        'ca_cb_bonds': ca_cb_bonds,
        'ca_cb_mean': np.mean(ca_cb_bonds) if ca_cb_bonds else np.nan,
        'ca_cb_std': np.std(ca_cb_bonds) if ca_cb_bonds else np.nan,
        'ca_cb_error': abs(np.mean(ca_cb_bonds) - 1.54) if ca_cb_bonds else np.nan,
        'severe_clashes': severe_clashes,
        'mild_clashes': mild_clashes,
        'num_severe_clashes': len(severe_clashes),
        'num_mild_clashes': len(mild_clashes),
    }

def main():
    experiments = [
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step10/val_seperated_Rm0_t0_step0_20251116_210156', '10步'),
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step100/val_seperated_Rm0_t0_step0_20251116_210400', '100步'),
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step500/val_seperated_Rm0_t0_step0_20251116_211051', '500步'),
    ]

    for exp_dir, name in experiments:
        print(f"\n{'='*80}")
        print(f"分析 {name}")
        print(f"{'='*80}")

        exp_path = Path(exp_dir)
        sample_dirs = sorted([d for d in exp_path.iterdir()
                             if d.is_dir() and d.name.startswith('sample_')])

        all_results = []

        # 分析所有样本
        for sample_dir in sample_dirs:
            result = analyze_sample(sample_dir)
            if result:
                all_results.append(result)

        if not all_results:
            print("没有有效结果")
            continue

        # 统计CA-CB键长
        all_ca_cb = []
        for r in all_results:
            all_ca_cb.extend(r['ca_cb_bonds'])

        print(f"\n1. CA-CB键长质量（共 {len(all_ca_cb)} 个键）:")
        print(f"   平均: {np.mean(all_ca_cb):.4f} Å (理想: 1.540 Å)")
        print(f"   标准差: {np.std(all_ca_cb):.4f} Å")
        print(f"   最小: {np.min(all_ca_cb):.4f} Å")
        print(f"   最大: {np.max(all_ca_cb):.4f} Å")
        print(f"   平均误差: {abs(np.mean(all_ca_cb) - 1.54):.4f} Å")

        if abs(np.mean(all_ca_cb) - 1.54) < 0.05:
            print(f"   ✓ CA-CB键长质量很好")
        elif abs(np.mean(all_ca_cb) - 1.54) < 0.1:
            print(f"   ⚠️  CA-CB键长有轻微偏差")
        else:
            print(f"   ❌ CA-CB键长偏差较大")

        # 统计clash
        total_severe = sum(r['num_severe_clashes'] for r in all_results)
        total_mild = sum(r['num_mild_clashes'] for r in all_results)

        print(f"\n2. 残基间Clash统计:")
        print(f"   样本数: {len(all_results)}")
        print(f"   严重clash (<2.0Å): {total_severe} 次 (平均每样本 {total_severe/len(all_results):.1f})")
        print(f"   轻微clash (2.0-2.5Å): {total_mild} 次 (平均每样本 {total_mild/len(all_results):.1f})")

        if total_severe == 0:
            print(f"   ✓ 无严重clash")
        elif total_severe / len(all_results) < 5:
            print(f"   ⚠️  有少量clash")
        else:
            print(f"   ❌ Clash问题较严重")

        # 显示几个clash的例子
        if total_severe > 0:
            print(f"\n   严重clash示例（前5个）:")
            count = 0
            for r in all_results:
                for clash in r['severe_clashes'][:2]:
                    print(f"     {r['sample_name']}: res{clash['res1']}-{clash['atom1']} <-> res{clash['res2']}-{clash['atom2']}, dist={clash['dist']:.2f}Å")
                    count += 1
                    if count >= 5:
                        break
                if count >= 5:
                    break

    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
