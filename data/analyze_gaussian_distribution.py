"""
分析不同氨基酸在两种高斯构建方法下的统计特性：
1. from_sidechain_atoms (仅侧链)
2. from_all_atoms (全原子)

统计内容：
- 质心相对于CA的偏移 (mu - CA)
- Scaling (S) 的分布
- 类内方差 vs 类间差异
"""

import torch
import numpy as np
import Bio.PDB
from collections import defaultdict
import matplotlib.pyplot as plt
from GaussianRigid import GaussianRigid

# 20种标准氨基酸
STANDARD_AA = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL'
]

def load_pdb_data(pdb_path):
    """加载PDB数据，按照GaussianRigid.py的逻辑"""
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    n, ca, c, o = [], [], [], []
    sc_coords, sc_masks = [], []
    res_names = []

    MAX_SC = 14

    for res in structure.get_residues():
        if Bio.PDB.is_aa(res, standard=True):
            if not (res.has_id('N') and res.has_id('CA') and res.has_id('C') and res.has_id('O')):
                continue

            n.append(res['N'].get_coord())
            ca.append(res['CA'].get_coord())
            c.append(res['C'].get_coord())
            o.append(res['O'].get_coord())
            res_names.append(res.get_resname())

            atoms = []
            for a in res:
                if a.name not in ['N', 'CA', 'C', 'O', 'OXT'] and a.element != 'H':
                    atoms.append(a.get_coord())

            tmp_c = np.zeros((MAX_SC, 3))
            tmp_m = np.zeros((MAX_SC))
            L = min(len(atoms), MAX_SC)
            if L > 0:
                tmp_c[:L] = np.array(atoms)[:L]
                tmp_m[:L] = 1.0
            sc_coords.append(tmp_c)
            sc_masks.append(tmp_m)

    # To Tensor
    def t(x):
        return torch.tensor(np.array(x), dtype=torch.float32).unsqueeze(0)

    return t(n), t(ca), t(c), t(o), t(sc_coords), t(sc_masks), res_names


def analyze_gaussian_stats(pdb_path):
    """统计分析两种方法的氨基酸分布"""

    print(f"Analyzing {pdb_path}...")
    n_t, ca_t, c_t, o_t, sc_t, mask_t, res_names = load_pdb_data(pdb_path)

    print(f"Loaded {len(res_names)} residues")

    # 生成两种高斯
    gr_sc = GaussianRigid.from_sidechain_atoms(n_t, ca_t, c_t, sc_t, mask_t, base_thickness=0.8)
    gr_all = GaussianRigid.from_all_atoms(n_t, ca_t, c_t, o_t, sc_t, mask_t, base_thickness=0.8)

    # 获取数据
    ca_coords = ca_t.squeeze(0).numpy()  # [N, 3]

    mu_sc = gr_sc.get_trans().squeeze(0).numpy()  # [N, 3]
    s_sc = gr_sc.scaling.squeeze(0).numpy()  # [N, 3]

    mu_all = gr_all.get_trans().squeeze(0).numpy()  # [N, 3]
    s_all = gr_all.scaling.squeeze(0).numpy()  # [N, 3]

    # 按氨基酸类型统计
    stats_sc = defaultdict(lambda: {'offset': [], 'scale': []})
    stats_all = defaultdict(lambda: {'offset': [], 'scale': []})

    for i, res_name in enumerate(res_names):
        # 质心相对CA的偏移
        offset_sc = np.linalg.norm(mu_sc[i] - ca_coords[i])
        offset_all = np.linalg.norm(mu_all[i] - ca_coords[i])

        # Scaling (取3个轴的平均)
        scale_sc = s_sc[i].mean()
        scale_all = s_all[i].mean()

        stats_sc[res_name]['offset'].append(offset_sc)
        stats_sc[res_name]['scale'].append(scale_sc)

        stats_all[res_name]['offset'].append(offset_all)
        stats_all[res_name]['scale'].append(scale_all)

    return stats_sc, stats_all, res_names


def compute_statistics(stats_dict):
    """计算统计量：均值、标准差"""
    summary = {}
    for aa in STANDARD_AA:
        if aa in stats_dict and len(stats_dict[aa]['offset']) > 0:
            offsets = np.array(stats_dict[aa]['offset'])
            scales = np.array(stats_dict[aa]['scale'])

            summary[aa] = {
                'count': len(offsets),
                'offset_mean': offsets.mean(),
                'offset_std': offsets.std(),
                'scale_mean': scales.mean(),
                'scale_std': scales.std(),
            }
    return summary


def print_comparison_table(summary_sc, summary_all):
    """打印对比表格"""
    print("\n" + "="*100)
    print("氨基酸统计对比 (Sidechain Only vs All Atoms)")
    print("="*100)
    print(f"{'AA':5s} | {'Count':5s} | {'SC Offset':12s} | {'All Offset':12s} | {'SC Scale':12s} | {'All Scale':12s}")
    print("-"*100)

    for aa in STANDARD_AA:
        if aa in summary_sc:
            sc = summary_sc[aa]
            all_data = summary_all.get(aa, None)

            if all_data:
                print(f"{aa:5s} | {sc['count']:5d} | "
                      f"{sc['offset_mean']:5.2f}±{sc['offset_std']:4.2f} | "
                      f"{all_data['offset_mean']:5.2f}±{all_data['offset_std']:4.2f} | "
                      f"{sc['scale_mean']:5.2f}±{sc['scale_std']:4.2f} | "
                      f"{all_data['scale_mean']:5.2f}±{all_data['scale_std']:4.2f}")

    print("="*100)


def analyze_inter_intra_class_variance(summary):
    """分析类内方差 vs 类间差异"""
    offset_means = []
    scale_means = []
    offset_stds = []
    scale_stds = []

    for aa in STANDARD_AA:
        if aa in summary:
            offset_means.append(summary[aa]['offset_mean'])
            scale_means.append(summary[aa]['scale_mean'])
            offset_stds.append(summary[aa]['offset_std'])
            scale_stds.append(summary[aa]['scale_std'])

    offset_means = np.array(offset_means)
    scale_means = np.array(scale_means)
    offset_stds = np.array(offset_stds)
    scale_stds = np.array(scale_stds)

    # 类间方差 (Inter-class variance)
    inter_offset_var = offset_means.var()
    inter_scale_var = scale_means.var()

    # 类内方差 (Intra-class variance) - 平均
    intra_offset_var = (offset_stds ** 2).mean()
    intra_scale_var = (scale_stds ** 2).mean()

    print("\n" + "="*60)
    print("类内 vs 类间方差分析")
    print("="*60)
    print(f"Offset (质心偏移):")
    print(f"  类间方差 (Inter-class): {inter_offset_var:.4f}")
    print(f"  类内方差 (Intra-class): {intra_offset_var:.4f}")
    print(f"  比值 (Inter/Intra):     {inter_offset_var/intra_offset_var:.2f}")
    print()
    print(f"Scale (椭球大小):")
    print(f"  类间方差 (Inter-class): {inter_scale_var:.4f}")
    print(f"  类内方差 (Intra-class): {intra_scale_var:.4f}")
    print(f"  比值 (Inter/Intra):     {inter_scale_var/intra_scale_var:.2f}")
    print("="*60)

    return {
        'inter_offset': inter_offset_var,
        'intra_offset': intra_offset_var,
        'inter_scale': inter_scale_var,
        'intra_scale': intra_scale_var,
    }


def plot_distributions(summary_sc, summary_all, save_path="gaussian_distribution.png"):
    """可视化分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    aa_list = [aa for aa in STANDARD_AA if aa in summary_sc]

    # SC: Offset
    offsets_sc = [summary_sc[aa]['offset_mean'] for aa in aa_list]
    offset_errs_sc = [summary_sc[aa]['offset_std'] for aa in aa_list]
    axes[0, 0].bar(range(len(aa_list)), offsets_sc, yerr=offset_errs_sc, capsize=3)
    axes[0, 0].set_xticks(range(len(aa_list)))
    axes[0, 0].set_xticklabels(aa_list, rotation=45)
    axes[0, 0].set_ylabel('Offset (Å)')
    axes[0, 0].set_title('Sidechain Only: Centroid Offset')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # SC: Scale
    scales_sc = [summary_sc[aa]['scale_mean'] for aa in aa_list]
    scale_errs_sc = [summary_sc[aa]['scale_std'] for aa in aa_list]
    axes[0, 1].bar(range(len(aa_list)), scales_sc, yerr=scale_errs_sc, capsize=3, color='orange')
    axes[0, 1].set_xticks(range(len(aa_list)))
    axes[0, 1].set_xticklabels(aa_list, rotation=45)
    axes[0, 1].set_ylabel('Scale (Å)')
    axes[0, 1].set_title('Sidechain Only: Ellipsoid Size')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # All: Offset
    offsets_all = [summary_all[aa]['offset_mean'] for aa in aa_list]
    offset_errs_all = [summary_all[aa]['offset_std'] for aa in aa_list]
    axes[1, 0].bar(range(len(aa_list)), offsets_all, yerr=offset_errs_all, capsize=3, color='green')
    axes[1, 0].set_xticks(range(len(aa_list)))
    axes[1, 0].set_xticklabels(aa_list, rotation=45)
    axes[1, 0].set_ylabel('Offset (Å)')
    axes[1, 0].set_title('All Atoms: Centroid Offset')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # All: Scale
    scales_all = [summary_all[aa]['scale_mean'] for aa in aa_list]
    scale_errs_all = [summary_all[aa]['scale_std'] for aa in aa_list]
    axes[1, 1].bar(range(len(aa_list)), scales_all, yerr=scale_errs_all, capsize=3, color='red')
    axes[1, 1].set_xticks(range(len(aa_list)))
    axes[1, 1].set_xticklabels(aa_list, rotation=45)
    axes[1, 1].set_ylabel('Scale (Å)')
    axes[1, 1].set_title('All Atoms: Ellipsoid Size')
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")


def find_pdb_files(root_dir, max_files=100):
    """在目录中递归搜索PDB和mmCIF文件"""
    import os
    import gzip

    pdb_files = []

    print(f"Searching for PDB/mmCIF files in {root_dir}...")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # 匹配 .pdb, .cif, .pdb.gz, .cif.gz
            if filename.endswith('.pdb') or filename.endswith('.cif') or \
               filename.endswith('.pdb.gz') or filename.endswith('.cif.gz'):
                full_path = os.path.join(dirpath, filename)
                pdb_files.append(full_path)

                if len(pdb_files) >= max_files:
                    print(f"Found {max_files} files, stopping search.")
                    return pdb_files

    print(f"Found {len(pdb_files)} PDB/mmCIF files in total.")
    return pdb_files


def load_mmcif_or_pdb(file_path):
    """加载mmCIF或PDB文件（支持gzip压缩）"""
    import gzip

    try:
        if file_path.endswith('.gz'):
            # 解压缩
            with gzip.open(file_path, 'rt') as f:
                content = f.read()
            # 判断是mmCIF还是PDB
            if '.cif' in file_path:
                parser = Bio.PDB.MMCIFParser(QUIET=True)
            else:
                parser = Bio.PDB.PDBParser(QUIET=True)

            # 写入临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cif' if '.cif' in file_path else '.pdb', delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            structure = parser.get_structure("protein", tmp_path)
            os.remove(tmp_path)
        else:
            # 未压缩
            if file_path.endswith('.cif'):
                parser = Bio.PDB.MMCIFParser(QUIET=True)
            else:
                parser = Bio.PDB.PDBParser(QUIET=True)
            structure = parser.get_structure("protein", file_path)

        return structure
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None


def load_structure_data(structure):
    """从structure对象提取数据"""
    if structure is None:
        return None

    n, ca, c, o = [], [], [], []
    sc_coords, sc_masks = [], []
    res_names = []

    MAX_SC = 14

    for res in structure.get_residues():
        if Bio.PDB.is_aa(res, standard=True):
            if not (res.has_id('N') and res.has_id('CA') and res.has_id('C') and res.has_id('O')):
                continue

            n.append(res['N'].get_coord())
            ca.append(res['CA'].get_coord())
            c.append(res['C'].get_coord())
            o.append(res['O'].get_coord())
            res_names.append(res.get_resname())

            atoms = []
            for a in res:
                if a.name not in ['N', 'CA', 'C', 'O', 'OXT'] and a.element != 'H':
                    atoms.append(a.get_coord())

            tmp_c = np.zeros((MAX_SC, 3))
            tmp_m = np.zeros((MAX_SC))
            L = min(len(atoms), MAX_SC)
            if L > 0:
                tmp_c[:L] = np.array(atoms)[:L]
                tmp_m[:L] = 1.0
            sc_coords.append(tmp_c)
            sc_masks.append(tmp_m)

    if len(n) == 0:
        return None

    # To Tensor
    def t(x):
        return torch.tensor(np.array(x), dtype=torch.float32).unsqueeze(0)

    return t(n), t(ca), t(c), t(o), t(sc_coords), t(sc_masks), res_names


def batch_analyze_database(data_dir, max_files=100):
    """批量分析数据库中的PDB文件"""

    # 搜索PDB文件
    pdb_files = find_pdb_files(data_dir, max_files)

    if len(pdb_files) == 0:
        print("No PDB files found!")
        return None, None

    # 累积统计
    stats_sc_total = defaultdict(lambda: {'offset': [], 'scale': []})
    stats_all_total = defaultdict(lambda: {'offset': [], 'scale': []})

    processed_count = 0
    failed_count = 0

    print(f"\nProcessing {len(pdb_files)} files...")

    for i, pdb_path in enumerate(pdb_files):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(pdb_files)} (processed: {processed_count}, failed: {failed_count})")

        try:
            # 加载结构
            structure = load_mmcif_or_pdb(pdb_path)
            data = load_structure_data(structure)

            if data is None:
                failed_count += 1
                continue

            n_t, ca_t, c_t, o_t, sc_t, mask_t, res_names = data

            # 生成两种高斯
            gr_sc = GaussianRigid.from_sidechain_atoms(n_t, ca_t, c_t, sc_t, mask_t, base_thickness=0.8)
            gr_all = GaussianRigid.from_all_atoms(n_t, ca_t, c_t, o_t, sc_t, mask_t, base_thickness=0.8)

            # 获取数据
            ca_coords = ca_t.squeeze(0).numpy()
            mu_sc = gr_sc.get_trans().squeeze(0).numpy()
            s_sc = gr_sc.scaling.squeeze(0).numpy()
            mu_all = gr_all.get_trans().squeeze(0).numpy()
            s_all = gr_all.scaling.squeeze(0).numpy()

            # 统计
            for j, res_name in enumerate(res_names):
                offset_sc = np.linalg.norm(mu_sc[j] - ca_coords[j])
                offset_all = np.linalg.norm(mu_all[j] - ca_coords[j])
                scale_sc = s_sc[j].mean()
                scale_all = s_all[j].mean()

                stats_sc_total[res_name]['offset'].append(offset_sc)
                stats_sc_total[res_name]['scale'].append(scale_sc)
                stats_all_total[res_name]['offset'].append(offset_all)
                stats_all_total[res_name]['scale'].append(scale_all)

            processed_count += 1

        except Exception as e:
            failed_count += 1
            if failed_count <= 5:  # 只打印前5个错误
                print(f"Error processing {pdb_path}: {e}")

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count}/{len(pdb_files)}")
    print(f"Failed: {failed_count}/{len(pdb_files)}")

    return stats_sc_total, stats_all_total


if __name__ == "__main__":
    import sys
    import os

    # 数据库路径
    data_dir = "/media/junyu/DATA/mmcif/gzipmmcif"

    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found!")
        sys.exit(1)

    # 批量分析
    stats_sc, stats_all = batch_analyze_database(data_dir, max_files=100)

    if stats_sc is None:
        print("Analysis failed!")
        sys.exit(1)

    # 计算统计量
    summary_sc = compute_statistics(stats_sc)
    summary_all = compute_statistics(stats_all)

    # 打印对比表格
    print_comparison_table(summary_sc, summary_all)

    # 分析类内类间方差
    print("\n--- Sidechain Only ---")
    variance_sc = analyze_inter_intra_class_variance(summary_sc)

    print("\n--- All Atoms ---")
    variance_all = analyze_inter_intra_class_variance(summary_all)

    # 可视化
    plot_distributions(summary_sc, summary_all, save_path="gaussian_distribution_database.png")

    print("\nAnalysis complete!")
