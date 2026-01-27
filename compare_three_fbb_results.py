"""
对比三个模型的FBB评估结果：IPA vs IGA (old) vs IGA (corrected)
"""
import pandas as pd
import numpy as np
from scipy import stats

# Load all three results
ipa_df = pd.read_csv('/home/junyu/project/pu/outputs/IPA/val_seperated_Rm0_t0_step0_20251208_130502_eval/fbb_results/fbb_scores.csv')
iga_old_df = pd.read_csv('/home/junyu/project/pu/outputs/IGA/val_seperated_Rm0_t0_step0_20251205_221136_eval/fbb_results/fbb_scores.csv')
iga_new_df = pd.read_csv('/home/junyu/project/pu/outputs/IGA_xlocal=μ+uraw⊙σ/val_seperated_Rm0_t0_step0_20251209_105001_eval/fbb_results/fbb_scores.csv')

print("=" * 100)
print("FBB评估三方对比：IPA vs IGA (old) vs IGA (corrected xlocal=μ+u⊙σ)")
print("=" * 100)

print(f"\nIPA样本数: {len(ipa_df)}")
print(f"IGA (old)样本数: {len(iga_old_df)}")
print(f"IGA (new)样本数: {len(iga_new_df)}")

# Metrics to compare
metrics = ['TM_score', 'RMSD', 'pLDDT', 'pAE', 'recovery', 'perplexity']

print("\n" + "=" * 100)
print("整体统计对比")
print("=" * 100)

results = []
for metric in metrics:
    ipa_mean = ipa_df[metric].mean()
    ipa_std = ipa_df[metric].std()
    iga_old_mean = iga_old_df[metric].mean()
    iga_old_std = iga_old_df[metric].std()
    iga_new_mean = iga_new_df[metric].mean()
    iga_new_std = iga_new_df[metric].std()

    results.append({
        'Metric': metric,
        'IPA': f'{ipa_mean:.3f} ± {ipa_std:.3f}',
        'IGA (old)': f'{iga_old_mean:.3f} ± {iga_old_std:.3f}',
        'IGA (new)': f'{iga_new_mean:.3f} ± {iga_new_std:.3f}',
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n" + "=" * 100)
print("关键指标对比")
print("=" * 100)

# TM-score comparison
print("\n1. TM-score (结构质量):")
print(f"   IPA:        {ipa_df['TM_score'].mean():.3f} ± {ipa_df['TM_score'].std():.3f}")
print(f"   IGA (old):  {iga_old_df['TM_score'].mean():.3f} ± {iga_old_df['TM_score'].std():.3f}")
print(f"   IGA (new):  {iga_new_df['TM_score'].mean():.3f} ± {iga_new_df['TM_score'].std():.3f}")

tm_best = max(ipa_df['TM_score'].mean(), iga_old_df['TM_score'].mean(), iga_new_df['TM_score'].mean())
if ipa_df['TM_score'].mean() == tm_best:
    print("   ✓ 最好: IPA")
elif iga_old_df['TM_score'].mean() == tm_best:
    print("   ✓ 最好: IGA (old)")
else:
    print("   ✓ 最好: IGA (new)")

# pLDDT comparison
print("\n2. pLDDT (折叠置信度):")
print(f"   IPA:        {ipa_df['pLDDT'].mean():.3f} ± {ipa_df['pLDDT'].std():.3f}")
print(f"   IGA (old):  {iga_old_df['pLDDT'].mean():.3f} ± {iga_old_df['pLDDT'].std():.3f}")
print(f"   IGA (new):  {iga_new_df['pLDDT'].mean():.3f} ± {iga_new_df['pLDDT'].std():.3f}")

plddt_best = max(ipa_df['pLDDT'].mean(), iga_old_df['pLDDT'].mean(), iga_new_df['pLDDT'].mean())
if ipa_df['pLDDT'].mean() == plddt_best:
    print("   ✓ 最好: IPA")
elif iga_old_df['pLDDT'].mean() == plddt_best:
    print("   ✓ 最好: IGA (old)")
else:
    print("   ✓ 最好: IGA (new)")

# RMSD comparison
print("\n3. RMSD (结构偏差, 越低越好):")
print(f"   IPA:        {ipa_df['RMSD'].mean():.3f} ± {ipa_df['RMSD'].std():.3f}")
print(f"   IGA (old):  {iga_old_df['RMSD'].mean():.3f} ± {iga_old_df['RMSD'].std():.3f}")
print(f"   IGA (new):  {iga_new_df['RMSD'].mean():.3f} ± {iga_new_df['RMSD'].std():.3f}")

rmsd_best = min(ipa_df['RMSD'].mean(), iga_old_df['RMSD'].mean(), iga_new_df['RMSD'].mean())
if ipa_df['RMSD'].mean() == rmsd_best:
    print("   ✓ 最好: IPA")
elif iga_old_df['RMSD'].mean() == rmsd_best:
    print("   ✓ 最好: IGA (old)")
else:
    print("   ✓ 最好: IGA (new)")

# Recovery comparison
print("\n4. Recovery (序列恢复率):")
print(f"   IPA:        {ipa_df['recovery'].mean():.3f} ± {ipa_df['recovery'].std():.3f}")
print(f"   IGA (old):  {iga_old_df['recovery'].mean():.3f} ± {iga_old_df['recovery'].std():.3f}")
print(f"   IGA (new):  {iga_new_df['recovery'].mean():.3f} ± {iga_new_df['recovery'].std():.3f}")

# Perplexity comparison
print("\n5. Perplexity (序列自然度, 越低越好):")
print(f"   IPA:        {ipa_df['perplexity'].mean():.3f} ± {ipa_df['perplexity'].std():.3f}")
print(f"   IGA (old):  {iga_old_df['perplexity'].mean():.3f} ± {iga_old_df['perplexity'].std():.3f}")
print(f"   IGA (new):  {iga_new_df['perplexity'].mean():.3f} ± {iga_new_df['perplexity'].std():.3f}")

perp_best = min(ipa_df['perplexity'].mean(), iga_old_df['perplexity'].mean(), iga_new_df['perplexity'].mean())
if ipa_df['perplexity'].mean() == perp_best:
    print("   ✓ 最好: IPA")
elif iga_old_df['perplexity'].mean() == perp_best:
    print("   ✓ 最好: IGA (old)")
else:
    print("   ✓ 最好: IGA (new)")

print("\n" + "=" * 100)
print("质量分布对比")
print("=" * 100)

# High pLDDT samples
ipa_high = (ipa_df['pLDDT'] > 80).sum()
iga_old_high = (iga_old_df['pLDDT'] > 80).sum()
iga_new_high = (iga_new_df['pLDDT'] > 80).sum()

print(f"\n高置信度样本 (pLDDT > 80):")
print(f"   IPA:        {ipa_high}/{len(ipa_df)} ({100*ipa_high/len(ipa_df):.1f}%)")
print(f"   IGA (old):  {iga_old_high}/{len(iga_old_df)} ({100*iga_old_high/len(iga_old_df):.1f}%)")
print(f"   IGA (new):  {iga_new_high}/{len(iga_new_df)} ({100*iga_new_high/len(iga_new_df):.1f}%)")

# Low pLDDT samples
ipa_low = (ipa_df['pLDDT'] < 70).sum()
iga_old_low = (iga_old_df['pLDDT'] < 70).sum()
iga_new_low = (iga_new_df['pLDDT'] < 70).sum()

print(f"\n低置信度样本 (pLDDT < 70):")
print(f"   IPA:        {ipa_low}/{len(ipa_df)} ({100*ipa_low/len(ipa_df):.1f}%)")
print(f"   IGA (old):  {iga_old_low}/{len(iga_old_df)} ({100*iga_old_low/len(iga_old_df):.1f}%)")
print(f"   IGA (new):  {iga_new_low}/{len(iga_new_df)} ({100*iga_new_low/len(iga_new_df):.1f}%)")

# Good structure samples (TM > 0.5)
ipa_good = (ipa_df['TM_score'] > 0.5).sum()
iga_old_good = (iga_old_df['TM_score'] > 0.5).sum()
iga_new_good = (iga_new_df['TM_score'] > 0.5).sum()

print(f"\n好的结构 (TM-score > 0.5):")
print(f"   IPA:        {ipa_good}/{len(ipa_df)} ({100*ipa_good/len(ipa_df):.1f}%)")
print(f"   IGA (old):  {iga_old_good}/{len(iga_old_df)} ({100*iga_old_good/len(iga_old_df):.1f}%)")
print(f"   IGA (new):  {iga_new_good}/{len(iga_new_df)} ({100*iga_new_good/len(iga_new_df):.1f}%)")

# Poor structure samples (TM < 0.5)
ipa_poor = (ipa_df['TM_score'] < 0.5).sum()
iga_old_poor = (iga_old_df['TM_score'] < 0.5).sum()
iga_new_poor = (iga_new_df['TM_score'] < 0.5).sum()

print(f"\n差的结构 (TM-score < 0.5):")
print(f"   IPA:        {ipa_poor}/{len(ipa_df)} ({100*ipa_poor/len(ipa_df):.1f}%)")
print(f"   IGA (old):  {iga_old_poor}/{len(iga_old_df)} ({100*iga_old_poor/len(iga_old_df):.1f}%)")
print(f"   IGA (new):  {iga_new_poor}/{len(iga_new_df)} ({100*iga_new_poor/len(iga_new_df):.1f}%)")

print("\n" + "=" * 100)
print("总结")
print("=" * 100)

# Count which model wins on each metric
ipa_wins = 0
iga_old_wins = 0
iga_new_wins = 0

# TM-score (higher is better)
tm_vals = [ipa_df['TM_score'].mean(), iga_old_df['TM_score'].mean(), iga_new_df['TM_score'].mean()]
if tm_vals.index(max(tm_vals)) == 0: ipa_wins += 1
elif tm_vals.index(max(tm_vals)) == 1: iga_old_wins += 1
else: iga_new_wins += 1

# RMSD (lower is better)
rmsd_vals = [ipa_df['RMSD'].mean(), iga_old_df['RMSD'].mean(), iga_new_df['RMSD'].mean()]
if rmsd_vals.index(min(rmsd_vals)) == 0: ipa_wins += 1
elif rmsd_vals.index(min(rmsd_vals)) == 1: iga_old_wins += 1
else: iga_new_wins += 1

# pLDDT (higher is better)
plddt_vals = [ipa_df['pLDDT'].mean(), iga_old_df['pLDDT'].mean(), iga_new_df['pLDDT'].mean()]
if plddt_vals.index(max(plddt_vals)) == 0: ipa_wins += 1
elif plddt_vals.index(max(plddt_vals)) == 1: iga_old_wins += 1
else: iga_new_wins += 1

# pAE (lower is better)
pae_vals = [ipa_df['pAE'].mean(), iga_old_df['pAE'].mean(), iga_new_df['pAE'].mean()]
if pae_vals.index(min(pae_vals)) == 0: ipa_wins += 1
elif pae_vals.index(min(pae_vals)) == 1: iga_old_wins += 1
else: iga_new_wins += 1

# Perplexity (lower is better)
perp_vals = [ipa_df['perplexity'].mean(), iga_old_df['perplexity'].mean(), iga_new_df['perplexity'].mean()]
if perp_vals.index(min(perp_vals)) == 0: ipa_wins += 1
elif perp_vals.index(min(perp_vals)) == 1: iga_old_wins += 1
else: iga_new_wins += 1

print(f"\n指标胜出次数 (共5个核心指标):")
print(f"   IPA:        {ipa_wins}/5")
print(f"   IGA (old):  {iga_old_wins}/5")
print(f"   IGA (new):  {iga_new_wins}/5")

if ipa_wins > max(iga_old_wins, iga_new_wins):
    print("\n✓ 总体最优: IPA")
elif iga_old_wins > max(ipa_wins, iga_new_wins):
    print("\n✓ 总体最优: IGA (old)")
elif iga_new_wins > max(ipa_wins, iga_old_wins):
    print("\n✓ 总体最优: IGA (new, corrected)")
else:
    print("\n≈ 三个模型表现接近")

print("\n关键发现:")
print(f"- TM-score: {['IPA', 'IGA (old)', 'IGA (new)'][tm_vals.index(max(tm_vals))]} 最高 ({max(tm_vals):.3f})")
print(f"- pLDDT: {['IPA', 'IGA (old)', 'IGA (new)'][plddt_vals.index(max(plddt_vals))]} 最高 ({max(plddt_vals):.3f})")
print(f"- RMSD: {['IPA', 'IGA (old)', 'IGA (new)'][rmsd_vals.index(min(rmsd_vals))]} 最低 ({min(rmsd_vals):.3f})")
print(f"- Perplexity: {['IPA', 'IGA (old)', 'IGA (new)'][perp_vals.index(min(perp_vals))]} 最低 ({min(perp_vals):.3f})")

print("\n" + "=" * 100)
