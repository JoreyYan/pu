"""
对比IGA vs IPA在CASP15上的表现
"""
import pandas as pd
import numpy as np

# Load results
iga = pd.read_csv('/home/junyu/project/pu/outputs/IGA_xlocal=μ+u⊙σ/val_seperated_Rm0_t0_step0_20251210_165253_eval/fbb_results/fbb_scores.csv')
ipa = pd.read_csv('/home/junyu/project/pu/outputs/IPA/val_seperated_Rm0_t0_step0_20251208_130502_eval/fbb_results/fbb_scores.csv')

print("=" * 100)
print("IGA vs IPA 对比 - CASP15 Test Set")
print("=" * 100)

print(f"\n样本数:")
print(f"  IGA: {len(iga)} 个样本")
print(f"  IPA: {len(ipa)} 个样本")

# Check if same samples
iga_domains = set(iga['domain_name'])
ipa_domains = set(ipa['domain_name'])
common_domains = iga_domains & ipa_domains
print(f"  共同样本: {len(common_domains)} 个")

# Align by domain
iga['domain'] = iga['domain_name']
ipa['domain'] = ipa['domain_name']

merged = pd.merge(
    iga[['domain', 'TM_score', 'RMSD', 'pLDDT', 'pAE', 'recovery', 'perplexity']],
    ipa[['domain', 'TM_score', 'RMSD', 'pLDDT', 'pAE', 'recovery', 'perplexity']],
    on='domain',
    suffixes=('_iga', '_ipa')
)

print(f"  对齐后样本数: {len(merged)}")

print("\n" + "=" * 100)
print("整体性能对比 (CASP15)")
print("=" * 100)

metrics = [
    ('TM_score', 'TM-score', 'higher'),
    ('RMSD', 'RMSD (Å)', 'lower'),
    ('pLDDT', 'pLDDT', 'higher'),
    ('pAE', 'pAE', 'lower'),
    ('recovery', 'Sequence Recovery', 'higher'),
    ('perplexity', 'Perplexity', 'lower'),
]

print(f"\n{'Metric':<25} {'IGA':<20} {'IPA':<20} {'Δ (IGA-IPA)':<20} {'Winner':<10}")
print("-" * 100)

for metric, desc, direction in metrics:
    iga_col = f'{metric}_iga'
    ipa_col = f'{metric}_ipa'

    iga_mean = merged[iga_col].mean()
    iga_std = merged[iga_col].std()
    ipa_mean = merged[ipa_col].mean()
    ipa_std = merged[ipa_col].std()

    delta = iga_mean - ipa_mean

    # Determine winner
    if direction == 'higher':
        winner = 'IGA' if iga_mean > ipa_mean else 'IPA'
        symbol = '✓' if iga_mean > ipa_mean else '✗'
    elif direction == 'lower':
        winner = 'IGA' if iga_mean < ipa_mean else 'IPA'
        symbol = '✓' if iga_mean < ipa_mean else '✗'
    else:
        winner = '-'
        symbol = ''

    print(f"{desc:<25} {iga_mean:6.3f} ± {iga_std:5.3f}   {ipa_mean:6.3f} ± {ipa_std:5.3f}   {delta:+7.3f}            {symbol} {winner}")

print("\n" + "=" * 100)
print("关键发现")
print("=" * 100)

# TM-score comparison
tm_iga = merged['TM_score_iga'].mean()
tm_ipa = merged['TM_score_ipa'].mean()
tm_diff = tm_iga - tm_ipa

print(f"\n1. TM-score 对比:")
print(f"   IGA: {tm_iga:.3f}")
print(f"   IPA: {tm_ipa:.3f}")
print(f"   差异: {tm_diff:+.3f} ({100*tm_diff/tm_ipa:+.1f}%)")
if tm_diff > 0.05:
    print(f"   ✓ IGA显著优于IPA")
elif tm_diff < -0.05:
    print(f"   ✗ IGA显著差于IPA")
else:
    print(f"   ≈ IGA和IPA表现相近")

# pLDDT comparison
plddt_iga = merged['pLDDT_iga'].mean()
plddt_ipa = merged['pLDDT_ipa'].mean()
plddt_diff = plddt_iga - plddt_ipa

print(f"\n2. pLDDT 对比 (注意：这是native pLDDT!):")
print(f"   IGA: {plddt_iga:.1f}")
print(f"   IPA: {plddt_ipa:.1f}")
print(f"   差异: {plddt_diff:+.1f}")

# RMSD comparison
rmsd_iga = merged['RMSD_iga'].mean()
rmsd_ipa = merged['RMSD_ipa'].mean()
rmsd_diff = rmsd_iga - rmsd_ipa

print(f"\n3. RMSD 对比:")
print(f"   IGA: {rmsd_iga:.3f} Å")
print(f"   IPA: {rmsd_ipa:.3f} Å")
print(f"   差异: {rmsd_diff:+.3f} Å")
if rmsd_diff < -1:
    print(f"   ✓ IGA更准确")
elif rmsd_diff > 1:
    print(f"   ✗ IPA更准确")
else:
    print(f"   ≈ 差异不大")

print("\n" + "=" * 100)
print("逐样本对比 (按IGA TM-score排序)")
print("=" * 100)

merged_sorted = merged.sort_values('TM_score_iga', ascending=False)

print(f"\n{'Domain':<15} {'IGA TM':<10} {'IPA TM':<10} {'Δ':<10} {'IGA pLDDT':<12} {'IPA pLDDT':<12} {'Winner':<10}")
print("-" * 90)

better_count = 0
worse_count = 0
similar_count = 0

for _, row in merged_sorted.iterrows():
    domain = row['domain']
    tm_iga = row['TM_score_iga']
    tm_ipa = row['TM_score_ipa']
    tm_delta = tm_iga - tm_ipa
    plddt_iga = row['pLDDT_iga']
    plddt_ipa = row['pLDDT_ipa']

    if tm_delta > 0.05:
        winner = 'IGA ✓'
        better_count += 1
    elif tm_delta < -0.05:
        winner = 'IPA'
        worse_count += 1
    else:
        winner = '≈'
        similar_count += 1

    print(f"{domain:<15} {tm_iga:<10.3f} {tm_ipa:<10.3f} {tm_delta:+10.3f} {plddt_iga:<12.1f} {plddt_ipa:<12.1f} {winner:<10}")

print("\n" + "=" * 100)
print("胜负统计")
print("=" * 100)

total = len(merged_sorted)
print(f"\nIGA更好 (Δ TM > 0.05):  {better_count:2d} samples ({100*better_count/total:.1f}%)")
print(f"差不多 (|Δ TM| ≤ 0.05):  {similar_count:2d} samples ({100*similar_count/total:.1f}%)")
print(f"IPA更好 (Δ TM < -0.05):  {worse_count:2d} samples ({100*worse_count/total:.1f}%)")

print("\n" + "=" * 100)
print("性能分布对比")
print("=" * 100)

# TM-score distribution
print("\nTM-score 分布:")
tm_bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
tm_labels = ['<0.3 (Bad)', '0.3-0.5 (Poor)', '0.5-0.7 (OK)', '0.7-0.9 (Good)', '0.9-1.0 (Excellent)']

merged['tm_bin_iga'] = pd.cut(merged['TM_score_iga'], bins=tm_bins, labels=tm_labels)
merged['tm_bin_ipa'] = pd.cut(merged['TM_score_ipa'], bins=tm_bins, labels=tm_labels)

print(f"\n  {'Category':<20} {'IGA':<15} {'IPA':<15}")
print("  " + "-" * 50)
for label in tm_labels:
    iga_count = (merged['tm_bin_iga'] == label).sum()
    ipa_count = (merged['tm_bin_ipa'] == label).sum()
    iga_pct = 100 * iga_count / len(merged)
    ipa_pct = 100 * ipa_count / len(merged)
    print(f"  {label:<20} {iga_count:2d} ({iga_pct:4.1f}%)     {ipa_count:2d} ({ipa_pct:4.1f}%)")

# pLDDT distribution
print("\npLDDT 分布:")
plddt_bins = [0, 50, 60, 70, 80, 90, 100]
plddt_labels = ['<50', '50-60', '60-70', '70-80', '80-90', '90-100']

merged['plddt_bin_iga'] = pd.cut(merged['pLDDT_iga'], bins=plddt_bins, labels=plddt_labels)
merged['plddt_bin_ipa'] = pd.cut(merged['pLDDT_ipa'], bins=plddt_bins, labels=plddt_labels)

print(f"\n  {'Category':<20} {'IGA':<15} {'IPA':<15}")
print("  " + "-" * 50)
for label in plddt_labels:
    iga_count = (merged['plddt_bin_iga'] == label).sum()
    ipa_count = (merged['plddt_bin_ipa'] == label).sum()
    iga_pct = 100 * iga_count / len(merged)
    ipa_pct = 100 * ipa_count / len(merged)
    print(f"  {label:<20} {iga_count:2d} ({iga_pct:4.1f}%)     {ipa_count:2d} ({ipa_pct:4.1f}%)")

print("\n" + "=" * 100)
print("结论")
print("=" * 100)

if tm_diff > 0.05:
    conclusion = f"""
✓ IGA在CASP15上显著优于IPA

IGA的TM-score比IPA高{tm_diff:.3f} ({100*tm_diff/tm_ipa:+.1f}%)，说明：
1. IGA架构改进有效，泛化能力更强
2. Invariant Gaussian Attention机制在困难样本上有优势
3. 你的模型改进是成功的！

但需要注意：
- 绝对性能仍然不够高 (TM={tm_iga:.3f})
- 仍有很多困难样本 (pLDDT < 50的比例)
- 继续改进训练数据和模型可能进一步提升
"""
elif tm_diff < -0.05:
    conclusion = f"""
✗ IGA在CASP15上差于IPA

IGA的TM-score比IPA低{-tm_diff:.3f} ({100*tm_diff/tm_ipa:+.1f}%)，说明：
1. IGA架构可能有问题，泛化能力不如IPA
2. 需要重新审视IGA的设计
3. 可能是训练策略或超参数的问题

建议：
- 检查IGA的attention机制是否正确实现
- 对比训练曲线，看是否过拟合
- 尝试调整正则化参数
"""
else:
    conclusion = f"""
≈ IGA和IPA在CASP15上表现相近

IGA的TM-score与IPA差异很小 ({tm_diff:+.3f})，说明：
1. IGA架构改进没有明显提升泛化能力
2. 但也没有损害性能
3. 问题可能是训练数据，而非模型架构

建议：
- 改进训练数据质量和多样性
- 增加困难样本
- 尝试数据增强技术
"""

print(conclusion)

print("=" * 100)

# Save detailed comparison
output_file = '/home/junyu/project/pu/iga_vs_ipa_casp15_comparison.csv'
merged.to_csv(output_file, index=False)
print(f"\n详细对比数据已保存: {output_file}")
