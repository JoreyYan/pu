import pandas as pd
import numpy as np

# 读取两个CSV文件
ipa_df = pd.read_csv('/home/junyu/project/pu/outputs/IPA/val_seperated_Rm0_t0_step0_20251208_130502_eval/fbb_results/fbb_scores.csv')
iga_df = pd.read_csv('/home/junyu/project/pu/outputs/IGA/val_seperated_Rm0_t0_step0_20251205_221136_eval/fbb_results/fbb_scores.csv')

print("=" * 80)
print("IPA vs IGA Comparison Report")
print("=" * 80)
print()

# 计算每个指标的平均值
metrics = ['TM_score', 'RMSD', 'pLDDT', 'pAE', 'recovery', 'perplexity']

print("Overall Metrics (Mean ± Std):")
print("-" * 80)
print(f"{'Metric':<15} {'IPA':<25} {'IGA':<25} {'Delta (IGA-IPA)':<20}")
print("-" * 80)

for metric in metrics:
    ipa_mean = ipa_df[metric].mean()
    ipa_std = ipa_df[metric].std()
    iga_mean = iga_df[metric].mean()
    iga_std = iga_df[metric].std()
    delta = iga_mean - ipa_mean

    # 对于 RMSD, pAE, perplexity，越小越好；对于其他指标，越大越好
    if metric in ['RMSD', 'pAE', 'perplexity']:
        better = "IGA ✓" if delta < 0 else "IPA ✓"
    else:
        better = "IGA ✓" if delta > 0 else "IPA ✓"

    print(f"{metric:<15} {ipa_mean:6.3f} ± {ipa_std:5.3f}      {iga_mean:6.3f} ± {iga_std:5.3f}      {delta:+7.3f} ({better})")

print()
print("=" * 80)
print("Key Findings:")
print("=" * 80)

# 计算哪个模型在每个样本上表现更好
tm_better_count = (iga_df['TM_score'] > ipa_df['TM_score']).sum()
rmsd_better_count = (iga_df['RMSD'] < ipa_df['RMSD']).sum()
plddt_better_count = (iga_df['pLDDT'] > ipa_df['pLDDT']).sum()
recovery_better_count = (iga_df['recovery'] > ipa_df['recovery']).sum()

total_samples = len(ipa_df)
print(f"Total samples: {total_samples}")
print()
print(f"IGA wins on TM_score:   {tm_better_count}/{total_samples} samples ({100*tm_better_count/total_samples:.1f}%)")
print(f"IGA wins on RMSD:       {rmsd_better_count}/{total_samples} samples ({100*rmsd_better_count/total_samples:.1f}%)")
print(f"IGA wins on pLDDT:      {plddt_better_count}/{total_samples} samples ({100*plddt_better_count/total_samples:.1f}%)")
print(f"IGA wins on recovery:   {recovery_better_count}/{total_samples} samples ({100*recovery_better_count/total_samples:.1f}%)")

print()
print("=" * 80)
print("Per-Sample Comparison (Top 10 largest TM_score improvements with IGA):")
print("=" * 80)

# 合并数据进行对比
comparison = pd.DataFrame({
    'sample': ipa_df['sample_name'],
    'domain': ipa_df['domain_name'],
    'IPA_TM': ipa_df['TM_score'],
    'IGA_TM': iga_df['TM_score'],
    'TM_delta': iga_df['TM_score'] - ipa_df['TM_score'],
    'IPA_RMSD': ipa_df['RMSD'],
    'IGA_RMSD': iga_df['RMSD'],
    'RMSD_delta': iga_df['RMSD'] - ipa_df['RMSD'],
    'IPA_pLDDT': ipa_df['pLDDT'],
    'IGA_pLDDT': iga_df['pLDDT'],
    'pLDDT_delta': iga_df['pLDDT'] - ipa_df['pLDDT']
})

# 按TM_score改进排序（IGA更好）
top_improvements = comparison.nlargest(10, 'TM_delta')
print(f"{'Domain':<15} {'IPA_TM':<10} {'IGA_TM':<10} {'Δ TM':<10} {'IPA_RMSD':<10} {'IGA_RMSD':<10} {'Δ RMSD':<10}")
print("-" * 80)
for _, row in top_improvements.iterrows():
    print(f"{row['domain']:<15} {row['IPA_TM']:7.3f}    {row['IGA_TM']:7.3f}    {row['TM_delta']:+7.3f}    "
          f"{row['IPA_RMSD']:7.3f}    {row['IGA_RMSD']:7.3f}    {row['RMSD_delta']:+7.3f}")

print()
print("=" * 80)
print("Per-Sample Comparison (Top 10 largest TM_score regressions with IGA):")
print("=" * 80)

# 按TM_score退步排序（IPA更好）
top_regressions = comparison.nsmallest(10, 'TM_delta')
print(f"{'Domain':<15} {'IPA_TM':<10} {'IGA_TM':<10} {'Δ TM':<10} {'IPA_RMSD':<10} {'IGA_RMSD':<10} {'Δ RMSD':<10}")
print("-" * 80)
for _, row in top_regressions.iterrows():
    print(f"{row['domain']:<15} {row['IPA_TM']:7.3f}    {row['IGA_TM']:7.3f}    {row['TM_delta']:+7.3f}    "
          f"{row['IPA_RMSD']:7.3f}    {row['IGA_RMSD']:7.3f}    {row['RMSD_delta']:+7.3f}")

print()
print("=" * 80)
print("Statistical Significance Test (Paired t-test):")
print("=" * 80)

from scipy import stats

for metric in metrics:
    ipa_vals = ipa_df[metric].values
    iga_vals = iga_df[metric].values

    t_stat, p_value = stats.ttest_rel(iga_vals, ipa_vals)

    if metric in ['RMSD', 'pAE', 'perplexity']:
        # 越小越好，所以负的t_stat说明IGA更好
        direction = "IGA < IPA" if t_stat < 0 else "IGA > IPA"
    else:
        # 越大越好，所以正的t_stat说明IGA更好
        direction = "IGA > IPA" if t_stat > 0 else "IGA < IPA"

    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."

    print(f"{metric:<15} t={t_stat:7.3f}, p={p_value:.4f} {significance:>5}  ({direction})")

print()
print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant")
print("=" * 80)
