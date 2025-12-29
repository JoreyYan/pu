"""
分析CASP15泛化性能问题
"""
import pandas as pd
import numpy as np

# Load CASP15 results
casp15 = pd.read_csv('/home/junyu/project/pu/outputs/IGA_xlocal=μ+u⊙σ/val_seperated_Rm0_t0_step0_20251210_165253_eval/fbb_results/fbb_scores.csv')

# Load Regular PDB validation results
pdb_val = pd.read_csv('/home/junyu/project/pu/outputs/IGA_xlocal=μ+u⊙σ_ffvaldata/val_seperated_Rm0_t0_step0_20251210_183038_fbb_eval/fbb_results/fbb_scores.csv')

print("=" * 100)
print("泛化性能分析：Regular PDB Validation vs CASP15")
print("=" * 100)

print(f"\n样本数:")
print(f"  Regular PDB Validation: {len(pdb_val)} 个样本")
print(f"  CASP15:                 {len(casp15)} 个样本")

print("\n" + "=" * 100)
print("整体性能对比")
print("=" * 100)

metrics = [
    ('TM_score', 'TM-score', 'higher'),
    ('RMSD', 'RMSD (Å)', 'lower'),
    ('pLDDT', 'pLDDT', 'higher'),
    ('pAE', 'pAE', 'lower'),
]

print(f"\n{'Metric':<20} {'PDB Val':<20} {'CASP15':<20} {'Δ (CASP-PDB)':<20} {'% Change':<15}")
print("-" * 100)

for metric, desc, direction in metrics:
    pdb_mean = pdb_val[metric].mean()
    pdb_std = pdb_val[metric].std()
    casp_mean = casp15[metric].mean()
    casp_std = casp15[metric].std()

    delta = casp_mean - pdb_mean
    pct_change = 100 * delta / pdb_mean if pdb_mean != 0 else 0

    print(f"{desc:<20} {pdb_mean:6.3f} ± {pdb_std:5.3f}    {casp_mean:6.3f} ± {casp_std:5.3f}    {delta:+7.3f}            {pct_change:+6.1f}%")

print("\n" + "=" * 100)
print("关键发现")
print("=" * 100)

# pLDDT analysis
pdb_plddt = pdb_val['pLDDT'].mean()
casp_plddt = casp15['pLDDT'].mean()
plddt_drop = pdb_plddt - casp_plddt

print(f"\n1. pLDDT 大幅下降:")
print(f"   Regular PDB: {pdb_plddt:.1f}")
print(f"   CASP15:      {casp_plddt:.1f}")
print(f"   下降:        {plddt_drop:.1f} points ({100*plddt_drop/pdb_plddt:.1f}%)")
print(f"   ✗ 这表明模型对CASP15序列的折叠信心非常低")

# TM-score analysis
pdb_tm = pdb_val['TM_score'].mean()
casp_tm = casp15['TM_score'].mean()
tm_drop = pdb_tm - casp_tm

print(f"\n2. TM-score 也显著下降:")
print(f"   Regular PDB: {pdb_tm:.3f}")
print(f"   CASP15:      {casp_tm:.3f}")
print(f"   下降:        {tm_drop:.3f} ({100*tm_drop/pdb_tm:.1f}%)")
print(f"   ✗ 说明生成的序列折叠后与真实结构差异大")

# RMSD analysis
pdb_rmsd = pdb_val['RMSD'].mean()
casp_rmsd = casp15['RMSD'].mean()
rmsd_increase = casp_rmsd - pdb_rmsd

print(f"\n3. RMSD 大幅增加:")
print(f"   Regular PDB: {pdb_rmsd:.3f} Å")
print(f"   CASP15:      {casp_rmsd:.3f} Å")
print(f"   增加:        {rmsd_increase:.3f} Å ({100*rmsd_increase/pdb_rmsd:.1f}%)")

print("\n" + "=" * 100)
print("CASP15 样本质量分布")
print("=" * 100)

# pLDDT distribution
plddt_bins = [0, 50, 60, 70, 80, 90, 100]
plddt_labels = ['<50', '50-60', '60-70', '70-80', '80-90', '90-100']
casp15['plddt_bin'] = pd.cut(casp15['pLDDT'], bins=plddt_bins, labels=plddt_labels)

print("\npLDDT 分布:")
plddt_dist = casp15['plddt_bin'].value_counts().sort_index()
for bin_label, count in plddt_dist.items():
    pct = 100 * count / len(casp15)
    print(f"  {bin_label}: {count:2d} samples ({pct:5.1f}%)")

# TM-score distribution
tm_bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
tm_labels = ['<0.3 (Bad)', '0.3-0.5 (Poor)', '0.5-0.7 (OK)', '0.7-0.9 (Good)', '0.9-1.0 (Excellent)']
casp15['tm_bin'] = pd.cut(casp15['TM_score'], bins=tm_bins, labels=tm_labels)

print("\nTM-score 分布:")
tm_dist = casp15['tm_bin'].value_counts().sort_index()
for bin_label, count in tm_dist.items():
    pct = 100 * count / len(casp15)
    print(f"  {bin_label}: {count:2d} samples ({pct:5.1f}%)")

print("\n" + "=" * 100)
print("最差样本分析 (pLDDT < 40)")
print("=" * 100)

worst = casp15[casp15['pLDDT'] < 40].sort_values('pLDDT')
print(f"\n找到 {len(worst)} 个pLDDT < 40的样本:")
print(f"\n{'Domain':<20} {'SeqLen':<10} {'TM-score':<12} {'pLDDT':<10} {'pAE':<10}")
print("-" * 70)
for _, row in worst.iterrows():
    print(f"{row['domain_name']:<20} {row['seqlen']:<10} {row['TM_score']:<12.3f} {row['pLDDT']:<10.1f} {row['pAE']:<10.1f}")

print("\n" + "=" * 100)
print("最好样本分析 (pLDDT > 80)")
print("=" * 100)

best = casp15[casp15['pLDDT'] > 80].sort_values('pLDDT', ascending=False)
print(f"\n找到 {len(best)} 个pLDDT > 80的样本:")
print(f"\n{'Domain':<20} {'SeqLen':<10} {'TM-score':<12} {'pLDDT':<10} {'pAE':<10}")
print("-" * 70)
for _, row in best.iterrows():
    print(f"{row['domain_name']:<20} {row['seqlen']:<10} {row['TM_score']:<12.3f} {row['pLDDT']:<10.1f} {row['pAE']:<10.1f}")

print("\n" + "=" * 100)
print("序列长度影响分析")
print("=" * 100)

# Analyze by sequence length
casp15['length_bin'] = pd.cut(casp15['seqlen'], bins=[0, 100, 200, 300, 1000], labels=['<100', '100-200', '200-300', '>300'])

print(f"\n按序列长度分组的pLDDT:")
for length_bin in ['<100', '100-200', '200-300', '>300']:
    subset = casp15[casp15['length_bin'] == length_bin]
    if len(subset) > 0:
        mean_plddt = subset['pLDDT'].mean()
        mean_tm = subset['TM_score'].mean()
        count = len(subset)
        print(f"  {length_bin:<10}: {count:2d} samples, pLDDT={mean_plddt:.1f}, TM-score={mean_tm:.3f}")

print("\n" + "=" * 100)
print("结论与建议")
print("=" * 100)

print("""
泛化性能差的主要原因:

1. 训练数据分布偏差
   - 训练集主要包含PDB中的常见折叠类型
   - CASP15包含新颖、罕见的折叠类型
   - 模型没有学习到足够多样的序列-结构关系

2. 困难样本不足
   - PDB validation样本相对简单 (pLDDT 82.5)
   - CASP15专门选择困难目标 (pLDDT 53.5)
   - 训练时缺乏类似难度的样本

3. 评估方式的差异
   - 注意：你现在比较的FBB pLDDT其实是native序列的pLDDT
   - predicted序列的真实pLDDT应该是 ~78.4 (Regular PDB)
   - CASP15的predicted pLDDT是 53.5
   - 实际下降是: 78.4 → 53.5 (24.9 points, 32%)

改进建议:

A. 数据增强
   - 增加困难样本的训练权重
   - 使用更多样化的蛋白质结构
   - 考虑synthetic难例生成

B. 模型改进
   - 检查模型是否过拟合到训练分布
   - 增加正则化
   - 考虑domain adaptation技术

C. 评估改进
   - 在验证集中包含更多困难样本
   - 创建分层验证集 (easy/medium/hard)
   - 早期检测泛化问题

D. 对比实验
   - 比较baseline (IPA) 在CASP15上的表现
   - 如果IPA也大幅下降，说明是数据问题
   - 如果IPA下降较小，说明是模型架构问题
""")

print("=" * 100)

# Save detailed comparison
output = pd.DataFrame({
    'Metric': ['TM-score', 'RMSD', 'pLDDT (注意：这是native pLDDT!)', 'pAE'],
    'PDB_Validation': [
        f"{pdb_val['TM_score'].mean():.3f}",
        f"{pdb_val['RMSD'].mean():.3f}",
        f"{pdb_val['pLDDT'].mean():.3f}",
        f"{pdb_val['pAE'].mean():.3f}"
    ],
    'CASP15': [
        f"{casp15['TM_score'].mean():.3f}",
        f"{casp15['RMSD'].mean():.3f}",
        f"{casp15['pLDDT'].mean():.3f}",
        f"{casp15['pAE'].mean():.3f}"
    ],
    'Delta': [
        f"{casp15['TM_score'].mean() - pdb_val['TM_score'].mean():.3f}",
        f"{casp15['RMSD'].mean() - pdb_val['RMSD'].mean():.3f}",
        f"{casp15['pLDDT'].mean() - pdb_val['pLDDT'].mean():.3f}",
        f"{casp15['pAE'].mean() - pdb_val['pAE'].mean():.3f}"
    ]
})

output_file = '/home/junyu/project/pu/pdb_vs_casp15_comparison.csv'
output.to_csv(output_file, index=False)
print(f"\n详细对比数据已保存: {output_file}")
