"""
检查最关键的问题：
predicted sequence中的氨基酸类型 vs PDB中实际的侧链结构是否一致

比如：如果sequence说是ALA（只有CB），但PDB里有CG、CD等原子 → BUG！
"""

import sys
sys.path.append('/home/junyu/project/pu')

import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser
from openfold.np import residue_constants
import warnings
warnings.filterwarnings('ignore')

# 标准氨基酸的侧链原子（不包括N, CA, C, O）
STANDARD_SIDECHAIN_ATOMS = {
    'ALA': ['CB'],
    'ARG': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'ASN': ['CB', 'CG', 'OD1', 'ND2'],
    'ASP': ['CB', 'CG', 'OD1', 'OD2'],
    'CYS': ['CB', 'SG'],
    'GLN': ['CB', 'CG', 'CD', 'OE1', 'NE2'],
    'GLU': ['CB', 'CG', 'CD', 'OE1', 'OE2'],
    'GLY': [],  # No sidechain
    'HIS': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
    'ILE': ['CB', 'CG1', 'CG2', 'CD1'],
    'LEU': ['CB', 'CG', 'CD1', 'CD2'],
    'LYS': ['CB', 'CG', 'CD', 'CE', 'NZ'],
    'MET': ['CB', 'CG', 'SD', 'CE'],
    'PHE': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'PRO': ['CB', 'CG', 'CD'],
    'SER': ['CB', 'OG'],
    'THR': ['CB', 'OG1', 'CG2'],
    'TRP': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
    'VAL': ['CB', 'CG1', 'CG2'],
}

def parse_fasta_file(fasta_path):
    """解析FASTA文件"""
    seqs = {}
    with open(fasta_path) as f:
        current_name = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name is not None:
                    seqs[current_name] = ''.join(current_seq)
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_name is not None:
            seqs[current_name] = ''.join(current_seq)
    return seqs

def parse_pdb_structure(pdb_path):
    """解析PDB，返回每个残基的实际原子列表"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()
                atoms = [atom.get_name() for atom in residue]
                coords = {atom.get_name(): atom.get_coord() for atom in residue}

                residues.append({
                    'res_name': res_name,
                    'atoms': atoms,
                    'coords': coords
                })

    return residues

def check_consistency(predicted_seq, pdb_residues):
    """
    检查predicted sequence和PDB结构的一致性

    返回：
    - total: 总残基数
    - consistent: 一致的残基数
    - mismatches: 不一致的详细信息
    """
    if len(predicted_seq) != len(pdb_residues):
        return {
            'error': f'Sequence length mismatch: {len(predicted_seq)} vs {len(pdb_residues)}'
        }

    mismatches = []

    for i, (aa, pdb_res) in enumerate(zip(predicted_seq, pdb_residues)):
        # 获取PDB中的残基名称
        pdb_res_name = pdb_res['res_name']

        # 3字母码转1字母码
        if pdb_res_name in residue_constants.restype_3to1:
            pdb_aa_1 = residue_constants.restype_3to1[pdb_res_name]
        else:
            pdb_aa_1 = 'X'

        # 检查sequence和PDB的残基类型是否一致
        if aa != pdb_aa_1:
            mismatches.append({
                'position': i,
                'seq_aa': aa,
                'pdb_aa': pdb_aa_1,
                'type': 'residue_type_mismatch',
                'pdb_atoms': pdb_res['atoms']
            })
            continue

        # 残基类型一致，检查侧链原子是否符合标准
        expected_sidechain = set(STANDARD_SIDECHAIN_ATOMS.get(pdb_res_name, []))
        actual_sidechain = set([a for a in pdb_res['atoms']
                                if a not in ['N', 'CA', 'C', 'O']])

        # 检查是否有多余的原子
        extra_atoms = actual_sidechain - expected_sidechain
        # 检查是否缺少原子
        missing_atoms = expected_sidechain - actual_sidechain

        if extra_atoms or missing_atoms:
            # 计算几何质量
            geom_issues = []

            # 检查CA-CB键长（如果有CB）
            if 'CA' in pdb_res['coords'] and 'CB' in pdb_res['coords']:
                ca_cb_dist = np.linalg.norm(
                    pdb_res['coords']['CB'] - pdb_res['coords']['CA']
                )
                if abs(ca_cb_dist - 1.54) > 0.3:  # 偏差>0.3Å
                    geom_issues.append(f'CA-CB={ca_cb_dist:.2f}Å')

            mismatches.append({
                'position': i,
                'seq_aa': aa,
                'pdb_aa': pdb_aa_1,
                'type': 'sidechain_atoms_mismatch',
                'expected_atoms': sorted(expected_sidechain),
                'actual_atoms': sorted(actual_sidechain),
                'extra_atoms': sorted(extra_atoms),
                'missing_atoms': sorted(missing_atoms),
                'geom_issues': geom_issues
            })

    return {
        'total': len(predicted_seq),
        'consistent': len(predicted_seq) - len(mismatches),
        'consistency_rate': (len(predicted_seq) - len(mismatches)) / len(predicted_seq),
        'mismatches': mismatches
    }

def analyze_sample(sample_dir):
    """分析一个样本"""
    sample_path = Path(sample_dir)

    # 读取predicted sequence
    fasta_path = sample_path / 'sequence.fasta'
    if not fasta_path.exists():
        return None

    seqs = parse_fasta_file(fasta_path)

    # 找到predicted sequence
    predicted_seq = None
    for name, seq in seqs.items():
        if 'predicted' in name.lower():
            predicted_seq = seq
            break

    if predicted_seq is None:
        return None

    # 读取PDB结构
    pdb_path = sample_path / 'predicted.pdb'
    if not pdb_path.exists():
        return None

    pdb_residues = parse_pdb_structure(str(pdb_path))

    # 检查一致性
    consistency = check_consistency(predicted_seq, pdb_residues)

    if 'error' in consistency:
        return None

    return {
        'sample_name': sample_path.name,
        'consistency_rate': consistency['consistency_rate'],
        'total_residues': consistency['total'],
        'consistent_residues': consistency['consistent'],
        'mismatches': consistency['mismatches']
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
        all_mismatches = []

        for sample_dir in sample_dirs[:10]:  # 先测试10个样本
            result = analyze_sample(sample_dir)
            if result:
                all_results.append(result)
                all_mismatches.extend(result['mismatches'])

        if not all_results:
            print("没有有效结果")
            continue

        # 统计
        avg_consistency = np.mean([r['consistency_rate'] for r in all_results])

        print(f"\n样本数: {len(all_results)}")
        print(f"平均一致性率: {avg_consistency*100:.1f}%")
        print(f"总不一致残基数: {len(all_mismatches)}")

        # 按问题类型统计
        type_mismatch = [m for m in all_mismatches if m['type'] == 'residue_type_mismatch']
        sidechain_mismatch = [m for m in all_mismatches if m['type'] == 'sidechain_atoms_mismatch']

        print(f"\n问题分类：")
        print(f"  残基类型不匹配（seq vs PDB）: {len(type_mismatch)}")
        print(f"  侧链原子数量/类型不匹配: {len(sidechain_mismatch)}")

        # 详细分析侧链问题
        if sidechain_mismatch:
            print(f"\n侧链问题详情（前10个）：")
            for i, m in enumerate(sidechain_mismatch[:10]):
                print(f"\n  [{i+1}] 位置 {m['position']}, 氨基酸: {m['seq_aa']}")
                if m['extra_atoms']:
                    print(f"      多余原子: {m['extra_atoms']}")
                if m['missing_atoms']:
                    print(f"      缺失原子: {m['missing_atoms']}")
                if m['geom_issues']:
                    print(f"      几何问题: {', '.join(m['geom_issues'])}")

        # 残基类型不匹配的例子
        if type_mismatch:
            print(f"\n残基类型不匹配详情（前10个）：")
            for i, m in enumerate(type_mismatch[:10]):
                print(f"  [{i+1}] 位置 {m['position']}: Seq说是 {m['seq_aa']}, PDB却是 {m['pdb_aa']}")

    print(f"\n{'='*80}")
    print("结论")
    print(f"{'='*80}")
    print("\n如果一致性率 < 100%，说明存在以下问题之一：")
    print("  1. Predicted sequence和PDB的残基类型不匹配")
    print("  2. 侧链原子数量/类型不符合氨基酸的标准结构")
    print("  3. 几何质量问题（键长异常等）")

if __name__ == '__main__':
    main()
