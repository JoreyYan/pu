"""
对比IGA模型 epoch75 vs epoch163 的序列设计质量
"""
import pandas as pd
import numpy as np
from scipy import stats

# Load results
epoch75_df = pd.read_csv('/media/junyu/DATA/pu5090weight/pdb__fbb_iga_simplified_attention_xlocal=μ+uraw⊙σ-2025-12-08_23-17-11/val_samples_epoch75/sequence_evaluation.csv')
epoch163_df = pd.read_csv('/media/junyu/DATA/pu5090weight/pdb__fbb_iga_simplified_attention_xlocal=μ + u ⊙ σ_2025-12-09_13-52-05/val_samples_epoch163/sequence_evaluation.csv')

print("=" * 100)
print("IGA模型训练进展：Epoch 75 vs Epoch 163")
print("=" * 100)

print(f"\nEpoch 75样本数: {len(epoch75_df)}")
print(f"Epoch 163样本数: {len(epoch163_df)}")

# Metrics to compare
metrics = [
    ('tm_score', 'TM-score (ESMFold native vs predicted)', 'higher'),
    ('rmsd', 'RMSD', 'lower'),
    ('predicted_plddt', 'Predicted pLDDT', 'higher'),
    ('predicted_pae', 'Predicted pAE', 'lower'),
    ('recovery', 'Sequence Recovery', 'context'),
    ('perplexity', 'Perplexity', 'lower'),
]

print("\n" + "=" * 100)
print("整体统计对比")
print("=" * 100)

results = []
for metric, desc, direction in metrics:
    e75_mean = epoch75_df[metric].mean()
    e75_std = epoch75_df[metric].std()
    e163_mean = epoch163_df[metric].mean()
    e163_std = epoch163_df[metric].std()

    delta = e163_mean - e75_mean
    pct_change = 100 * delta / e75_mean if e75_mean != 0 else 0

    # Determine improvement
    if direction == 'higher':
        improved = '✓' if delta > 0 else '✗'
    elif direction == 'lower':
        improved = '✓' if delta < 0 else '✗'
    else:
        improved = '-'

    results.append({
        'Metric': desc,
        'Epoch 75': f'{e75_mean:.3f} ± {e75_std:.3f}',
        'Epoch 163': f'{e163_mean:.3f} ± {e163_std:.3f}',
        'Δ': f'{delta:+.3f}',
        '% Change': f'{pct_change:+.1f}%',
        'Status': improved
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n" + "=" * 100)
print("关键改进")
print("=" * 100)

# TM-score improvement
tm_delta = epoch163_df['tm_score'].mean() - epoch75_df['tm_score'].mean()
tm_pct = 100 * tm_delta / epoch75_df['tm_score'].mean()
print(f"\n1. 结构相似度 (TM-score):")
print(f"   Epoch 75:  {epoch75_df['tm_score'].mean():.3f} ± {epoch75_df['tm_score'].std():.3f}")
print(f"   Epoch 163: {epoch163_df['tm_score'].mean():.3f} ± {epoch163_df['tm_score'].std():.3f}")
print(f"   提升: +{tm_delta:.3f} ({tm_pct:+.1f}%)")
if tm_delta > 0.01:
    print(f"   ✓ 显著提升！设计序列与native结构更相似")
elif tm_delta > 0:
    print(f"   ≈ 略有提升")
else:
    print(f"   ✗ 有所下降")

# pLDDT improvement
plddt_delta = epoch163_df['predicted_plddt'].mean() - epoch75_df['predicted_plddt'].mean()
plddt_pct = 100 * plddt_delta / epoch75_df['predicted_plddt'].mean()
print(f"\n2. 折叠置信度 (Predicted pLDDT):")
print(f"   Epoch 75:  {epoch75_df['predicted_plddt'].mean():.3f} ± {epoch75_df['predicted_plddt'].std():.3f}")
print(f"   Epoch 163: {epoch163_df['predicted_plddt'].mean():.3f} ± {epoch163_df['predicted_plddt'].std():.3f}")
print(f"   提升: +{plddt_delta:.3f} ({plddt_pct:+.1f}%)")
if plddt_delta > 1:
    print(f"   ✓ 显著提升！设计序列更可折叠")
elif plddt_delta > 0:
    print(f"   ≈ 略有提升")
else:
    print(f"   ✗ 有所下降")

# pLDDT gap reduction
e75_gap = epoch75_df['predicted_plddt'].mean() - epoch75_df['native_plddt'].mean()
e163_gap = epoch163_df['predicted_plddt'].mean() - epoch163_df['native_plddt'].mean()
gap_reduction = e75_gap - e163_gap
print(f"\n3. pLDDT差距缩小:")
print(f"   Epoch 75:  predicted - native = {e75_gap:.3f}")
print(f"   Epoch 163: predicted - native = {e163_gap:.3f}")
print(f"   差距缩小: {gap_reduction:.3f}")
if gap_reduction > 0.5:
    print(f"   ✓ 显著改善！设计序列更接近native质量")
elif gap_reduction > 0:
    print(f"   ≈ 有所改善")
else:
    print(f"   ✗ 差距扩大")

# Recovery improvement
recovery_delta = epoch163_df['recovery'].mean() - epoch75_df['recovery'].mean()
recovery_pct = 100 * recovery_delta / epoch75_df['recovery'].mean()
print(f"\n4. 序列恢复率 (Recovery):")
print(f"   Epoch 75:  {epoch75_df['recovery'].mean():.3f} ± {epoch75_df['recovery'].std():.3f}")
print(f"   Epoch 163: {epoch163_df['recovery'].mean():.3f} ± {epoch163_df['recovery'].std():.3f}")
print(f"   提升: +{recovery_delta:.3f} ({recovery_pct:+.1f}%)")
if recovery_delta > 0.05:
    print(f"   ✓ 显著提升！设计序列更接近native")
elif recovery_delta > 0:
    print(f"   ≈ 略有提升")
else:
    print(f"   ✗ 有所下降")

# Perplexity improvement
perp_delta = epoch163_df['perplexity'].mean() - epoch75_df['perplexity'].mean()
perp_pct = 100 * perp_delta / epoch75_df['perplexity'].mean()
print(f"\n5. 序列困惑度 (Perplexity, 越低越好):")
print(f"   Epoch 75:  {epoch75_df['perplexity'].mean():.3f} ± {epoch75_df['perplexity'].std():.3f}")
print(f"   Epoch 163: {epoch163_df['perplexity'].mean():.3f} ± {epoch163_df['perplexity'].std():.3f}")
print(f"   变化: {perp_delta:+.3f} ({perp_pct:+.1f}%)")
if perp_delta < -0.5:
    print(f"   ✓ 显著改善！序列更自然")
elif perp_delta < 0:
    print(f"   ≈ 略有改善")
else:
    print(f"   ✗ 有所上升")

# High confidence samples
e75_high = (epoch75_df['predicted_plddt'] > 80).sum()
e163_high = (epoch163_df['predicted_plddt'] > 80).sum()
print(f"\n6. 高置信度样本 (pLDDT > 80):")
print(f"   Epoch 75:  {e75_high}/{len(epoch75_df)} ({100*e75_high/len(epoch75_df):.1f}%)")
print(f"   Epoch 163: {e163_high}/{len(epoch163_df)} ({100*e163_high/len(epoch163_df):.1f}%)")
print(f"   增加: {e163_high - e75_high} 个样本")

# Low confidence samples
e75_low = (epoch75_df['predicted_plddt'] < 70).sum()
e163_low = (epoch163_df['predicted_plddt'] < 70).sum()
print(f"\n7. 低置信度样本 (pLDDT < 70):")
print(f"   Epoch 75:  {e75_low}/{len(epoch75_df)} ({100*e75_low/len(epoch75_df):.1f}%)")
print(f"   Epoch 163: {e163_low}/{len(epoch163_df)} ({100*e163_low/len(epoch163_df):.1f}%)")
print(f"   减少: {e75_low - e163_low} 个样本")

# Poor structure samples (TM < 0.7)
e75_poor = (epoch75_df['tm_score'] < 0.7).sum()
e163_poor = (epoch163_df['tm_score'] < 0.7).sum()
print(f"\n8. 结构偏差大的样本 (TM-score < 0.7):")
print(f"   Epoch 75:  {e75_poor}/{len(epoch75_df)} ({100*e75_poor/len(epoch75_df):.1f}%)")
print(f"   Epoch 163: {e163_poor}/{len(epoch163_df)} ({100*e163_poor/len(epoch163_df):.1f}%)")
print(f"   减少: {e75_poor - e163_poor} 个样本")

print("\n" + "=" * 100)
print("总结")
print("=" * 100)

# Count improvements
score = 0
if tm_delta > 0: score += 1
if plddt_delta > 0: score += 1
if gap_reduction > 0: score += 1
if recovery_delta > 0: score += 1
if perp_delta < 0: score += 1

print(f"\n改进指标数量: {score}/5")

if score >= 4:
    print("✓✓✓ 训练效果显著！从Epoch 75到163有明显提升")
elif score >= 3:
    print("✓✓ 训练有效果，多数指标有改善")
elif score >= 2:
    print("≈ 训练略有效果，部分指标改善")
else:
    print("✗ 训练效果不明显或有退化")

print("\n关键亮点:")
improvements = []
if tm_delta > 0.01:
    improvements.append(f"- TM-score提升{tm_pct:.1f}%，结构相似度显著提高")
if plddt_delta > 1:
    improvements.append(f"- pLDDT提升{plddt_delta:.1f}分，折叠置信度明显提高")
if gap_reduction > 0.5:
    improvements.append(f"- pLDDT gap缩小{gap_reduction:.1f}分，更接近native质量")
if recovery_delta > 0.05:
    improvements.append(f"- Recovery提升{recovery_pct:.1f}%，序列更接近native")
if e163_high > e75_high:
    improvements.append(f"- 高置信度样本增加{e163_high - e75_high}个")
if e163_low < e75_low:
    improvements.append(f"- 低置信度样本减少{e75_low - e163_low}个")
if e163_poor < e75_poor:
    improvements.append(f"- 结构失败样本减少{e75_poor - e163_poor}个")

if improvements:
    for imp in improvements:
        print(imp)
else:
    print("- 各项指标变化不大")

print("\n" + "=" * 100)
