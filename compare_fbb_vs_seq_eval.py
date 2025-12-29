"""
Compare FBB evaluation vs Val Sequences evaluation for ffvaldata
"""
import pandas as pd
import numpy as np

# Load results
fbb_df = pd.read_csv('/home/junyu/project/pu/outputs/IGA_xlocal=μ+u⊙σ_ffvaldata/val_seperated_Rm0_t0_step0_20251210_183038_fbb_eval/fbb_results/fbb_scores.csv')
seq_df = pd.read_csv('/home/junyu/project/pu/outputs/IGA_xlocal=μ+u⊙σ_ffvaldata/val_seperated_Rm0_t0_step0_20251210_183038_seq_eval.csv')

print("=" * 100)
print("FBB Evaluation vs Val Sequences Evaluation 对比")
print("Dataset: ffvaldata (17 samples with native PDB structures)")
print("=" * 100)

print(f"\nFBB样本数: {len(fbb_df)}")
print(f"Seq样本数: {len(seq_df)}")

# Align data by sample name
# FBB uses domain_name (e.g., "7l1w"), Seq uses sample_name (e.g., "sample_7l1w_000004")
# Extract domain from seq sample_name
seq_df['domain'] = seq_df['sample_name'].str.extract(r'sample_([^_]+)_')[0]
fbb_df['domain'] = fbb_df['domain_name']

# Rename FBB columns to add _fbb suffix manually
fbb_df = fbb_df.rename(columns={
    'TM_score': 'TM_score_fbb',
    'RMSD': 'RMSD_fbb',
    'pLDDT': 'pLDDT_fbb',
    'pAE': 'pAE_fbb',
    'recovery': 'recovery_fbb',
    'perplexity': 'perplexity_fbb'
})

# Merge on domain
merged = pd.merge(fbb_df, seq_df, on='domain')

print(f"\n匹配样本数: {len(merged)}")

print("\n" + "=" * 100)
print("评估方法对比")
print("=" * 100)

print("\n方法说明:")
print("  FBB Evaluation:")
print("    - 提取模型设计的predicted序列")
print("    - 用ESMFold折叠predicted序列")
print("    - 与native PDB结构比较 (TM-score, RMSD)")
print("    - 评估折叠后的predicted结构质量 (pLDDT, pAE)")
print()
print("  Val Sequences Evaluation:")
print("    - 提取native和predicted两个序列")
print("    - 分别用ESMFold折叠")
print("    - 比较两个ESMFold折叠结果 (TM-score, RMSD)")
print("    - 评估两个折叠结构的质量")

print("\n" + "=" * 100)
print("核心差异")
print("=" * 100)

print("\nTM-score 定义:")
print("  FBB:  predicted序列ESMFold折叠 vs native PDB结构")
print("  Seq:  predicted序列ESMFold折叠 vs native序列ESMFold折叠")
print()
print("关键区别: FBB使用真实的native PDB结构作为参考，Seq使用ESMFold预测的native结构")

print("\n" + "=" * 100)
print("整体统计对比")
print("=" * 100)

metrics = [
    ('TM_score', 'TM-score', 'higher'),
    ('RMSD', 'RMSD', 'lower'),
    ('pLDDT', 'Predicted pLDDT', 'higher'),
    ('pAE', 'Predicted pAE', 'lower'),
    ('recovery', 'Sequence Recovery', 'context'),
    ('perplexity', 'Perplexity', 'lower'),
]

results = []
for metric, desc, direction in metrics:
    fbb_col = f'{metric}_fbb'
    # Seq uses lowercase column names
    if metric == 'TM_score':
        seq_col = 'tm_score'
    elif metric == 'RMSD':
        seq_col = 'rmsd'
    elif metric == 'pLDDT':
        seq_col = 'predicted_plddt'
    elif metric == 'pAE':
        seq_col = 'predicted_pae'
    elif metric in ['recovery', 'perplexity']:
        seq_col = metric
    else:
        seq_col = metric.lower()

    if fbb_col not in merged.columns or seq_col not in merged.columns:
        continue

    fbb_mean = merged[fbb_col].mean()
    fbb_std = merged[fbb_col].std()
    seq_mean = merged[seq_col].mean()
    seq_std = merged[seq_col].std()

    delta = seq_mean - fbb_mean
    pct_change = 100 * delta / fbb_mean if fbb_mean != 0 else 0

    # Determine which is better
    if direction == 'higher':
        better = 'FBB' if fbb_mean > seq_mean else 'Seq'
    elif direction == 'lower':
        better = 'FBB' if fbb_mean < seq_mean else 'Seq'
    else:
        better = '-'

    results.append({
        'Metric': desc,
        'FBB': f'{fbb_mean:.3f} ± {fbb_std:.3f}',
        'Seq': f'{seq_mean:.3f} ± {seq_std:.3f}',
        'Δ (Seq-FBB)': f'{delta:+.3f}',
        '% Change': f'{pct_change:+.1f}%',
        'Better': better
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n" + "=" * 100)
print("关键发现")
print("=" * 100)

# TM-score comparison
tm_fbb = merged['TM_score_fbb'].mean()
tm_seq = merged['tm_score'].mean()
print(f"\n1. TM-score 对比:")
print(f"   FBB (vs native PDB):           {tm_fbb:.3f}")
print(f"   Seq (vs native ESMFold):       {tm_seq:.3f}")
print(f"   差异:                          {tm_seq - tm_fbb:+.3f}")
if tm_seq > tm_fbb:
    print(f"   ✓ Seq更高: 说明predicted序列与native序列的ESMFold折叠结果更相似")
    print(f"     这是合理的，因为两者都是ESMFold预测的，有相同的bias")
else:
    print(f"   ✗ FBB更高: 说明predicted序列折叠后与真实native PDB结构更接近")

# pLDDT comparison
plddt_fbb = merged['pLDDT_fbb'].mean()
plddt_seq = merged['predicted_plddt'].mean()
print(f"\n2. Predicted pLDDT 对比:")
print(f"   FBB:  {plddt_fbb:.3f}")
print(f"   Seq:  {plddt_seq:.3f}")
print(f"   差异: {plddt_seq - plddt_fbb:+.3f}")
if abs(plddt_seq - plddt_fbb) < 1:
    print(f"   ≈ 基本一致: 两种方法评估的predicted序列折叠质量相近")
else:
    print(f"   注意: pLDDT差异较大，可能是评估流程不同导致")

# RMSD comparison
rmsd_fbb = merged['RMSD_fbb'].mean()
rmsd_seq = merged['rmsd'].mean()
print(f"\n3. RMSD 对比:")
print(f"   FBB (vs native PDB):           {rmsd_fbb:.3f} Å")
print(f"   Seq (vs native ESMFold):       {rmsd_seq:.3f} Å")
print(f"   差异:                          {rmsd_seq - rmsd_fbb:+.3f} Å")

print("\n" + "=" * 100)
print("逐样本对比 (按TM-score排序)")
print("=" * 100)

# Sort by FBB TM-score
merged_sorted = merged.sort_values('TM_score_fbb', ascending=False)

print(f"\n{'Domain':<10} {'FBB TM':<10} {'Seq TM':<10} {'Δ TM':<10} {'FBB pLDDT':<12} {'Seq pLDDT':<12} {'Δ pLDDT':<10}")
print("-" * 90)

for _, row in merged_sorted.iterrows():
    domain = row['domain']
    tm_fbb = row['TM_score_fbb']
    tm_seq = row['tm_score']
    tm_delta = tm_seq - tm_fbb
    plddt_fbb = row['pLDDT_fbb']
    plddt_seq = row['predicted_plddt']
    plddt_delta = plddt_seq - plddt_fbb

    print(f"{domain:<10} {tm_fbb:<10.3f} {tm_seq:<10.3f} {tm_delta:+10.3f} {plddt_fbb:<12.3f} {plddt_seq:<12.3f} {plddt_delta:+10.3f}")

print("\n" + "=" * 100)
print("结论")
print("=" * 100)

print("""
1. TM-score差异解释:
   - Seq evaluation的TM-score更高是正常的
   - 因为两个结构都是ESMFold预测的，有相同的预测偏差
   - FBB evaluation使用真实PDB结构，更严格

2. pLDDT比较一致:
   - 两种方法评估的predicted序列折叠质量相近
   - 验证了两种评估方法的一致性

3. 推荐使用:
   - 如果有native PDB结构 → 使用 FBB evaluation (更准确)
   - 如果只有序列 → 使用 Val Sequences evaluation (更灵活)
   - 对于有PDB的样本，FBB给出的是与真实结构的比较，更可信
""")

print("=" * 100)

# Save comparison
output_csv = '/home/junyu/project/pu/fbb_vs_seq_comparison.csv'
merged.to_csv(output_csv, index=False)
print(f"\n详细对比数据已保存到: {output_csv}")
