"""
对比IPA和IGA模型在序列设计质量上的差异
"""
import pandas as pd
import numpy as np
from scipy import stats

# Load results
ipa_df = pd.read_csv('/home/junyu/project/pu/ckpt/se3-fm_sh/pdb__fbb_ipa_baseline_vs_t3dmk4mh/2025-12-07_21-27-32/val_samples_epoch23/sequence_evaluation.csv')
iga_df = pd.read_csv('/media/junyu/DATA/pu5090weight/pdb__fbb_iga_simplified_attention_xlocal=μ+uraw⊙σ-2025-12-08_23-17-11/val_samples_epoch75/sequence_evaluation.csv')

print("=" * 100)
print("IPA vs IGA: 序列设计质量对比")
print("=" * 100)

print(f"\nIPA样本数: {len(ipa_df)}")
print(f"IGA样本数: {len(iga_df)}")

# Metrics to compare
metrics = [
    ('tm_score', 'TM-score (ESMFold native vs predicted)', 'higher'),
    ('rmsd', 'RMSD', 'lower'),
    ('native_plddt', 'Native pLDDT', 'higher'),
    ('predicted_plddt', 'Predicted pLDDT', 'higher'),
    ('native_pae', 'Native pAE', 'lower'),
    ('predicted_pae', 'Predicted pAE', 'lower'),
    ('recovery', 'Sequence Recovery', 'context'),
    ('perplexity', 'Perplexity', 'lower'),
]

print("\n" + "=" * 100)
print("整体统计对比")
print("=" * 100)

results = []
for metric, desc, direction in metrics:
    ipa_mean = ipa_df[metric].mean()
    ipa_std = ipa_df[metric].std()
    iga_mean = iga_df[metric].mean()
    iga_std = iga_df[metric].std()

    delta = iga_mean - ipa_mean

    # Determine winner
    if direction == 'higher':
        winner = 'IGA ✓' if delta > 0 else 'IPA ✓'
    elif direction == 'lower':
        winner = 'IGA ✓' if delta < 0 else 'IPA ✓'
    else:
        winner = '-'

    results.append({
        'Metric': desc,
        'IPA': f'{ipa_mean:.3f} ± {ipa_std:.3f}',
        'IGA': f'{iga_mean:.3f} ± {iga_std:.3f}',
        'Δ (IGA-IPA)': f'{delta:+.3f}',
        'Winner': winner
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n" + "=" * 100)
print("关键发现")
print("=" * 100)

# TM-score comparison
tm_delta = iga_df['tm_score'].mean() - ipa_df['tm_score'].mean()
print(f"\n1. 结构相似度 (TM-score):")
print(f"   IPA: {ipa_df['tm_score'].mean():.3f} ± {ipa_df['tm_score'].std():.3f}")
print(f"   IGA: {iga_df['tm_score'].mean():.3f} ± {iga_df['tm_score'].std():.3f}")
print(f"   IGA比IPA高 {tm_delta:.3f} ({100*tm_delta/ipa_df['tm_score'].mean():+.1f}%)")
if tm_delta > 0:
    print(f"   ✓ IGA设计的序列折叠后与native更相似")
else:
    print(f"   ✗ IPA设计的序列折叠后与native更相似")

# pLDDT comparison
ipa_plddt_drop = ipa_df['predicted_plddt'].mean() - ipa_df['native_plddt'].mean()
iga_plddt_drop = iga_df['predicted_plddt'].mean() - iga_df['native_plddt'].mean()
print(f"\n2. pLDDT下降 (Predicted - Native):")
print(f"   IPA: {ipa_plddt_drop:.3f}")
print(f"   IGA: {iga_plddt_drop:.3f}")
print(f"   Δ: {abs(iga_plddt_drop) - abs(ipa_plddt_drop):.3f}")
if abs(iga_plddt_drop) < abs(ipa_plddt_drop):
    print(f"   ✓ IGA的pLDDT下降更小（更接近native质量）")
else:
    print(f"   ✗ IPA的pLDDT下降更小（更接近native质量）")

# Predicted pLDDT comparison
pred_plddt_delta = iga_df['predicted_plddt'].mean() - ipa_df['predicted_plddt'].mean()
print(f"\n3. Predicted Sequence pLDDT:")
print(f"   IPA: {ipa_df['predicted_plddt'].mean():.3f}")
print(f"   IGA: {iga_df['predicted_plddt'].mean():.3f}")
print(f"   IGA比IPA高 {pred_plddt_delta:.3f}")
if pred_plddt_delta > 0:
    print(f"   ✓ IGA设计的序列ESMFold折叠置信度更高")
else:
    print(f"   ✗ IPA设计的序列ESMFold折叠置信度更高")

# Recovery comparison
recovery_delta = iga_df['recovery'].mean() - ipa_df['recovery'].mean()
print(f"\n4. Sequence Recovery:")
print(f"   IPA: {ipa_df['recovery'].mean():.3f}")
print(f"   IGA: {iga_df['recovery'].mean():.3f}")
print(f"   Δ: {recovery_delta:.3f}")
print(f"   解读: Recovery相近，说明两个模型的序列多样性相似")

# High pLDDT samples
ipa_high_plddt = (ipa_df['predicted_plddt'] > 80).sum()
iga_high_plddt = (iga_df['predicted_plddt'] > 80).sum()
print(f"\n5. 高置信度样本 (pLDDT > 80):")
print(f"   IPA: {ipa_high_plddt}/{len(ipa_df)} ({100*ipa_high_plddt/len(ipa_df):.1f}%)")
print(f"   IGA: {iga_high_plddt}/{len(iga_df)} ({100*iga_high_plddt/len(iga_df):.1f}%)")

# Low pLDDT samples
ipa_low_plddt = (ipa_df['predicted_plddt'] < 70).sum()
iga_low_plddt = (iga_df['predicted_plddt'] < 70).sum()
print(f"\n6. 低置信度样本 (pLDDT < 70):")
print(f"   IPA: {ipa_low_plddt}/{len(ipa_df)} ({100*ipa_low_plddt/len(ipa_df):.1f}%)")
print(f"   IGA: {iga_low_plddt}/{len(iga_df)} ({100*iga_low_plddt/len(iga_df):.1f}%)")

# Poor structure samples (TM < 0.7)
ipa_poor_tm = (ipa_df['tm_score'] < 0.7).sum()
iga_poor_tm = (iga_df['tm_score'] < 0.7).sum()
print(f"\n7. 结构偏差大的样本 (TM-score < 0.7):")
print(f"   IPA: {ipa_poor_tm}/{len(ipa_df)} ({100*ipa_poor_tm/len(ipa_df):.1f}%)")
print(f"   IGA: {iga_poor_tm}/{len(iga_df)} ({100*iga_poor_tm/len(iga_df):.1f}%)")

print("\n" + "=" * 100)
print("总结")
print("=" * 100)

print("\n结构相似度 (TM-score):")
if tm_delta > 0.01:
    print(f"  ✓ IGA明显更好 (+{tm_delta:.3f})")
elif tm_delta > 0:
    print(f"  ≈ IGA略好 (+{tm_delta:.3f})")
elif tm_delta > -0.01:
    print(f"  ≈ IPA略好 ({tm_delta:.3f})")
else:
    print(f"  ✗ IPA明显更好 ({tm_delta:.3f})")

print("\n折叠置信度 (Predicted pLDDT):")
if pred_plddt_delta > 1:
    print(f"  ✓ IGA明显更好 (+{pred_plddt_delta:.3f})")
elif pred_plddt_delta > 0:
    print(f"  ≈ IGA略好 (+{pred_plddt_delta:.3f})")
elif pred_plddt_delta > -1:
    print(f"  ≈ IPA略好 ({pred_plddt_delta:.3f})")
else:
    print(f"  ✗ IPA明显更好 ({pred_plddt_delta:.3f})")

print("\n序列困惑度 (Perplexity):")
perp_delta = iga_df['perplexity'].mean() - ipa_df['perplexity'].mean()
if perp_delta < -0.5:
    print(f"  ✓ IGA明显更好 ({perp_delta:.3f})")
elif perp_delta < 0:
    print(f"  ≈ IGA略好 ({perp_delta:.3f})")
elif perp_delta < 0.5:
    print(f"  ≈ IPA略好 (+{perp_delta:.3f})")
else:
    print(f"  ✗ IPA明显更好 (+{perp_delta:.3f})")

print("\n整体评价:")
score = 0
if tm_delta > 0: score += 1
if pred_plddt_delta > 0: score += 1
if perp_delta < 0: score += 1

if score >= 2:
    print("  ✓ IGA在序列设计质量上整体优于IPA")
elif score == 1:
    print("  ≈ IGA和IPA在序列设计质量上相当")
else:
    print("  ✗ IPA在序列设计质量上整体优于IGA")

print("\n" + "=" * 100)
