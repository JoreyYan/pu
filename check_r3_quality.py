"""
检查R3 FBB生成结构的质量
包括：
1. Sequence一致性（GT vs predicted）
2. 侧链几何质量（键长、clash）
3. 结构质量（二级结构、回旋半径等）
"""

import sys
sys.path.append('/home/junyu/project/pu')

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import PDBParser
import warnings
warnings.filterwarnings('ignore')

# 导入已有工具
from analysis.metrics import calc_mdtraj_metrics, calc_ca_ca_metrics
from openfold.np import residue_constants

def parse_fasta_file(fasta_path):
    """解析FASTA文件，返回(name, seq)字典"""
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

def check_sequence_consistency(fasta_path):
    """检查sequence.fasta中GT和predicted的一致性"""
    seqs = parse_fasta_file(fasta_path)

    # 查找original和predicted序列
    original_seq = None
    predicted_seq = None

    for name, seq in seqs.items():
        if 'original' in name.lower():
            original_seq = seq
        elif 'predicted' in name.lower():
            predicted_seq = seq

    if original_seq is None or predicted_seq is None:
        return {
            'sequence_match': False,
            'error': 'Cannot find original or predicted sequence',
            'original_len': 0,
            'predicted_len': 0,
            'identity': 0.0
        }

    # 比较
    match = (original_seq == predicted_seq)
    identity = sum(a == b for a, b in zip(original_seq, predicted_seq)) / len(original_seq)

    return {
        'sequence_match': match,
        'original_len': len(original_seq),
        'predicted_len': len(predicted_seq),
        'identity': identity,
        'original_seq': original_seq,
        'predicted_seq': predicted_seq
    }

def check_sidechain_geometry(pdb_path):
    """检查侧链几何质量"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    # 提取所有原子坐标
    all_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    all_atoms.append({
                        'residue': residue.get_resname(),
                        'atom': atom.get_name(),
                        'coord': atom.get_coord()
                    })

    if len(all_atoms) == 0:
        return {'error': 'No atoms found'}

    # 1. 检查CA-CB键长
    ca_cb_bonds = []
    for i, atom in enumerate(all_atoms):
        if atom['atom'] == 'CA':
            # 找同一残基的CB
            for j, other in enumerate(all_atoms):
                if (other['residue'] == atom['residue'] and
                    other['atom'] == 'CB' and
                    abs(i - j) < 20):  # 应该在附近
                    dist = np.linalg.norm(atom['coord'] - other['coord'])
                    ca_cb_bonds.append(dist)
                    break

    # 2. 检查所有原子的clash
    coords = np.array([a['coord'] for a in all_atoms])

    # 距离矩阵
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    # 排除自己和对角线
    mask = np.triu(np.ones_like(dists), k=1).astype(bool)
    inter_dists = dists[mask]

    # Clash统计（< 2.0 Å认为是严重clash）
    severe_clashes = np.sum(inter_dists < 2.0)
    mild_clashes = np.sum((inter_dists >= 2.0) & (inter_dists < 2.5))

    results = {
        'num_atoms': len(all_atoms),
        'ca_cb_bonds': ca_cb_bonds,
        'ca_cb_mean': np.mean(ca_cb_bonds) if ca_cb_bonds else np.nan,
        'ca_cb_std': np.std(ca_cb_bonds) if ca_cb_bonds else np.nan,
        'ca_cb_error': abs(np.mean(ca_cb_bonds) - 1.54) if ca_cb_bonds else np.nan,
        'severe_clashes': severe_clashes,
        'mild_clashes': mild_clashes,
        'clash_rate': severe_clashes / len(all_atoms) if len(all_atoms) > 0 else 0.0
    }

    return results

def analyze_sample(sample_dir):
    """分析一个样本目录"""
    sample_path = Path(sample_dir)

    results = {
        'sample_name': sample_path.name
    }

    # 1. Sequence一致性
    fasta_path = sample_path / 'sequence.fasta'
    if fasta_path.exists():
        seq_check = check_sequence_consistency(fasta_path)
        results.update({
            'seq_match': seq_check['sequence_match'],
            'seq_identity': seq_check['identity'],
            'seq_len_original': seq_check['original_len'],
            'seq_len_predicted': seq_check['predicted_len']
        })
    else:
        results.update({
            'seq_match': False,
            'seq_identity': 0.0
        })

    # 2. 侧链几何质量
    pdb_path = sample_path / 'predicted.pdb'
    if pdb_path.exists():
        geom = check_sidechain_geometry(str(pdb_path))
        results.update({
            'num_atoms': geom.get('num_atoms', 0),
            'ca_cb_mean': geom.get('ca_cb_mean', np.nan),
            'ca_cb_std': geom.get('ca_cb_std', np.nan),
            'ca_cb_error': geom.get('ca_cb_error', np.nan),
            'severe_clashes': geom.get('severe_clashes', 0),
            'mild_clashes': geom.get('mild_clashes', 0),
            'clash_rate': geom.get('clash_rate', 0.0)
        })

        # 3. 结构质量（mdtraj）
        try:
            mdtraj_metrics = calc_mdtraj_metrics(str(pdb_path))
            results.update({
                'helix_percent': mdtraj_metrics['helix_percent'],
                'strand_percent': mdtraj_metrics['strand_percent'],
                'coil_percent': mdtraj_metrics['coil_percent'],
                'radius_of_gyration': mdtraj_metrics['radius_of_gyration']
            })
        except Exception as e:
            print(f"  mdtraj failed: {e}")
            results.update({
                'helix_percent': np.nan,
                'strand_percent': np.nan,
                'coil_percent': np.nan,
                'radius_of_gyration': np.nan
            })

    return results

def analyze_experiment(exp_dir, name):
    """分析一个实验的所有样本"""
    exp_path = Path(exp_dir)
    sample_dirs = sorted([d for d in exp_path.iterdir()
                         if d.is_dir() and d.name.startswith('sample_')])

    print(f"\n{'='*80}")
    print(f"分析 {name}: {len(sample_dirs)} 个样本")
    print(f"{'='*80}")

    all_results = []

    for sample_dir in tqdm(sample_dirs, desc=f"{name}"):
        result = analyze_sample(sample_dir)
        all_results.append(result)

    return pd.DataFrame(all_results)

def print_summary(df, name):
    """打印摘要统计"""
    print(f"\n{'='*80}")
    print(f"{name} - 质量检查摘要")
    print(f"{'='*80}")

    print(f"\n1. Sequence一致性：")
    print(f"   完全匹配: {df['seq_match'].sum()} / {len(df)} ({df['seq_match'].mean()*100:.1f}%)")
    print(f"   平均identity: {df['seq_identity'].mean():.3f}")

    if df['seq_match'].mean() < 1.0:
        print(f"   ⚠️  有 {(~df['seq_match']).sum()} 个样本sequence不匹配！")
    else:
        print(f"   ✓ 所有样本sequence完全一致")

    print(f"\n2. CA-CB键长质量：")
    print(f"   平均键长: {df['ca_cb_mean'].mean():.3f} Å (理想值: 1.54 Å)")
    print(f"   平均误差: {df['ca_cb_error'].mean():.3f} Å")
    print(f"   标准差:   {df['ca_cb_std'].mean():.3f} Å")

    if df['ca_cb_error'].mean() < 0.1:
        print(f"   ✓ CA-CB键长质量很好")
    elif df['ca_cb_error'].mean() < 0.2:
        print(f"   ⚠️  CA-CB键长有一定偏差")
    else:
        print(f"   ❌ CA-CB键长偏差较大！")

    print(f"\n3. Clash统计：")
    print(f"   严重clash (<2.0Å): {df['severe_clashes'].sum()} 次")
    print(f"   轻微clash (2.0-2.5Å): {df['mild_clashes'].sum()} 次")
    print(f"   平均clash率: {df['clash_rate'].mean()*100:.2f}%")

    if df['severe_clashes'].sum() == 0:
        print(f"   ✓ 无严重clash")
    elif df['severe_clashes'].mean() < 1:
        print(f"   ⚠️  有少量clash")
    else:
        print(f"   ❌ Clash问题较严重！")

    print(f"\n4. 二级结构：")
    print(f"   Helix:  {df['helix_percent'].mean()*100:.1f}%")
    print(f"   Strand: {df['strand_percent'].mean()*100:.1f}%")
    print(f"   Coil:   {df['coil_percent'].mean()*100:.1f}%")
    print(f"   回旋半径: {df['radius_of_gyration'].mean():.2f} nm")

def main():
    experiments = [
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step10/val_seperated_Rm0_t0_step0_20251116_210156', '10步'),
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step100/val_seperated_Rm0_t0_step0_20251116_210400', '100步'),
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step500/val_seperated_Rm0_t0_step0_20251116_211051', '500步'),
    ]

    all_dfs = {}

    for exp_dir, name in experiments:
        df = analyze_experiment(exp_dir, name)
        all_dfs[name] = df
        print_summary(df, name)

    # 对比总结
    print(f"\n{'='*80}")
    print("三种步数的对比总结")
    print(f"{'='*80}")

    print(f"\n{'指标':<25} {'10步':<20} {'100步':<20} {'500步':<20}")
    print("-"*85)

    metrics = [
        ('seq_match', 'Sequence匹配率', lambda x: f"{x.mean()*100:.1f}%"),
        ('ca_cb_mean', 'CA-CB键长 (Å)', lambda x: f"{x.mean():.3f}"),
        ('ca_cb_error', 'CA-CB误差 (Å)', lambda x: f"{x.mean():.3f}"),
        ('severe_clashes', '严重clash (总计)', lambda x: f"{x.sum():.0f}"),
        ('clash_rate', 'Clash率 (%)', lambda x: f"{x.mean()*100:.2f}"),
        ('helix_percent', 'Helix (%)', lambda x: f"{x.mean()*100:.1f}"),
        ('strand_percent', 'Strand (%)', lambda x: f"{x.mean()*100:.1f}"),
    ]

    for metric_key, metric_name, formatter in metrics:
        values = [formatter(all_dfs[name][metric_key]) for name in ['10步', '100步', '500步']]
        print(f"{metric_name:<25} {values[0]:<20} {values[1]:<20} {values[2]:<20}")

if __name__ == '__main__':
    main()
