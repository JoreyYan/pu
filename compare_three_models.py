"""
Three-way comparison: IPA vs IGA (original) vs IGA (corrected decoding)
"""
import pandas as pd
import numpy as np
from scipy import stats

# Load all three results
ipa_df = pd.read_csv('/home/junyu/project/pu/outputs/IPA/val_seperated_Rm0_t0_step0_20251208_130502_eval/fbb_results/fbb_scores.csv')
iga_old_df = pd.read_csv('/home/junyu/project/pu/outputs/IGA/val_seperated_Rm0_t0_step0_20251205_221136_eval/fbb_results/fbb_scores.csv')
iga_new_df = pd.read_csv('/home/junyu/project/pu/outputs/IGA_xlocal=μ+uraw⊙σ/val_seperated_Rm0_t0_step0_20251209_105001_eval/fbb_results/fbb_scores.csv')

print("=" * 100)
print("THREE-WAY COMPARISON: IPA vs IGA (old) vs IGA (corrected: xlocal=μ+uraw⊙σ)")
print("=" * 100)

# Merge on sample_name to ensure alignment
df = ipa_df[['sample_name', 'domain_name']].copy()
df = df.merge(ipa_df[['sample_name', 'TM_score', 'RMSD', 'pLDDT', 'pAE', 'recovery', 'perplexity']],
              on='sample_name', suffixes=('', '_ipa'))
df = df.merge(iga_old_df[['sample_name', 'TM_score', 'RMSD', 'pLDDT', 'pAE', 'recovery', 'perplexity']],
              on='sample_name', suffixes=('_ipa', '_iga_old'))
df = df.merge(iga_new_df[['sample_name', 'TM_score', 'RMSD', 'pLDDT', 'pAE', 'recovery', 'perplexity']],
              on='sample_name', suffixes=('', '_iga_new'))

# Rename columns for clarity
df.columns = ['sample_name', 'domain_name',
              'TM_score_ipa', 'RMSD_ipa', 'pLDDT_ipa', 'pAE_ipa', 'recovery_ipa', 'perplexity_ipa',
              'TM_score_iga_old', 'RMSD_iga_old', 'pLDDT_iga_old', 'pAE_iga_old', 'recovery_iga_old', 'perplexity_iga_old',
              'TM_score_iga_new', 'RMSD_iga_new', 'pLDDT_iga_new', 'pAE_iga_new', 'recovery_iga_new', 'perplexity_iga_new']

print(f"\nSamples: {len(df)}")
print(f"IPA samples: {len(ipa_df)}")
print(f"IGA (old) samples: {len(iga_old_df)}")
print(f"IGA (new) samples: {len(iga_new_df)}")

# Metrics to compare (higher is better for TM_score, pLDDT, recovery; lower is better for RMSD, pAE, perplexity)
metrics = ['TM_score', 'RMSD', 'pLDDT', 'pAE', 'recovery', 'perplexity']
better_higher = ['TM_score', 'pLDDT', 'recovery']
better_lower = ['RMSD', 'pAE', 'perplexity']

print("\n" + "=" * 100)
print("OVERALL STATISTICS")
print("=" * 100)

results = []
for metric in metrics:
    ipa_vals = df[f'{metric}_ipa'].values
    iga_old_vals = df[f'{metric}_iga_old'].values
    iga_new_vals = df[f'{metric}_iga_new'].values

    ipa_mean, ipa_std = ipa_vals.mean(), ipa_vals.std()
    iga_old_mean, iga_old_std = iga_old_vals.mean(), iga_old_vals.std()
    iga_new_mean, iga_new_std = iga_new_vals.mean(), iga_new_vals.std()

    results.append({
        'Metric': metric,
        'IPA': f'{ipa_mean:.3f} ± {ipa_std:.3f}',
        'IGA (old)': f'{iga_old_mean:.3f} ± {iga_old_std:.3f}',
        'IGA (new)': f'{iga_new_mean:.3f} ± {iga_new_std:.3f}',
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n" + "=" * 100)
print("PAIRWISE STATISTICAL TESTS (paired t-test)")
print("=" * 100)

comparisons = [
    ('IGA (new)', 'IPA', 'iga_new', 'ipa'),
    ('IGA (new)', 'IGA (old)', 'iga_new', 'iga_old'),
    ('IGA (old)', 'IPA', 'iga_old', 'ipa'),
]

for comp_name1, comp_name2, suffix1, suffix2 in comparisons:
    print(f"\n{comp_name1} vs {comp_name2}:")
    print("-" * 80)

    for metric in metrics:
        vals1 = df[f'{metric}_{suffix1}'].values
        vals2 = df[f'{metric}_{suffix2}'].values

        delta = vals1.mean() - vals2.mean()
        t_stat, p_value = stats.ttest_rel(vals1, vals2)

        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "n.s."

        # Determine if improvement
        if metric in better_higher:
            improved = "✓" if delta > 0 else "✗"
        else:
            improved = "✓" if delta < 0 else "✗"

        print(f"  {metric:12s}: Δ = {delta:+7.3f}  (p={p_value:.4f} {sig})  {improved}")

print("\n" + "=" * 100)
print("WIN RATES (percentage of samples where model performs better)")
print("=" * 100)

win_rates = []
for metric in metrics:
    ipa_vals = df[f'{metric}_ipa'].values
    iga_old_vals = df[f'{metric}_iga_old'].values
    iga_new_vals = df[f'{metric}_iga_new'].values

    if metric in better_higher:
        # Higher is better
        new_vs_ipa = (iga_new_vals > ipa_vals).mean() * 100
        new_vs_old = (iga_new_vals > iga_old_vals).mean() * 100
        old_vs_ipa = (iga_old_vals > ipa_vals).mean() * 100
    else:
        # Lower is better
        new_vs_ipa = (iga_new_vals < ipa_vals).mean() * 100
        new_vs_old = (iga_new_vals < iga_old_vals).mean() * 100
        old_vs_ipa = (iga_old_vals < ipa_vals).mean() * 100

    win_rates.append({
        'Metric': metric,
        'IGA(new) vs IPA': f'{new_vs_ipa:.1f}%',
        'IGA(new) vs IGA(old)': f'{new_vs_old:.1f}%',
        'IGA(old) vs IPA': f'{old_vs_ipa:.1f}%',
    })

win_df = pd.DataFrame(win_rates)
print(win_df.to_string(index=False))

print("\n" + "=" * 100)
print("TOP 10 IMPROVEMENTS: IGA (new) vs IGA (old)")
print("=" * 100)

# Calculate improvement score (normalized)
df['improvement_score'] = 0
for metric in metrics:
    vals_new = df[f'{metric}_iga_new'].values
    vals_old = df[f'{metric}_iga_old'].values

    if metric in better_higher:
        improvement = vals_new - vals_old
    else:
        improvement = vals_old - vals_new

    # Normalize by std
    if vals_old.std() > 0:
        improvement = improvement / vals_old.std()

    df['improvement_score'] += improvement

top_improvements = df.nlargest(10, 'improvement_score')[
    ['sample_name', 'domain_name', 'TM_score_iga_old', 'TM_score_iga_new',
     'RMSD_iga_old', 'RMSD_iga_new', 'pLDDT_iga_old', 'pLDDT_iga_new']
].copy()

top_improvements['ΔTM'] = top_improvements['TM_score_iga_new'] - top_improvements['TM_score_iga_old']
top_improvements['ΔRMSD'] = top_improvements['RMSD_iga_new'] - top_improvements['RMSD_iga_old']
top_improvements['ΔpLDDT'] = top_improvements['pLDDT_iga_new'] - top_improvements['pLDDT_iga_old']

print(top_improvements[['domain_name', 'TM_score_iga_old', 'TM_score_iga_new', 'ΔTM',
                         'RMSD_iga_old', 'RMSD_iga_new', 'ΔRMSD']].to_string(index=False))

print("\n" + "=" * 100)
print("TOP 10 REGRESSIONS: IGA (new) vs IGA (old)")
print("=" * 100)

top_regressions = df.nsmallest(10, 'improvement_score')[
    ['sample_name', 'domain_name', 'TM_score_iga_old', 'TM_score_iga_new',
     'RMSD_iga_old', 'RMSD_iga_new', 'pLDDT_iga_old', 'pLDDT_iga_new']
].copy()

top_regressions['ΔTM'] = top_regressions['TM_score_iga_new'] - top_regressions['TM_score_iga_old']
top_regressions['ΔRMSD'] = top_regressions['RMSD_iga_new'] - top_regressions['RMSD_iga_old']
top_regressions['ΔpLDDT'] = top_regressions['pLDDT_iga_new'] - top_regressions['pLDDT_iga_old']

print(top_regressions[['domain_name', 'TM_score_iga_old', 'TM_score_iga_new', 'ΔTM',
                        'RMSD_iga_old', 'RMSD_iga_new', 'ΔRMSD']].to_string(index=False))

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

print("\n1. IGA (new, corrected decoding) vs IPA:")
for metric in metrics:
    vals_new = df[f'{metric}_iga_new'].values
    vals_ipa = df[f'{metric}_ipa'].values
    delta = vals_new.mean() - vals_ipa.mean()
    t_stat, p_value = stats.ttest_rel(vals_new, vals_ipa)

    if metric in better_higher:
        if delta > 0 and p_value < 0.05:
            print(f"   ✓ {metric}: Better by {abs(delta):.3f} (p={p_value:.4f})")
        elif delta < 0 and p_value < 0.05:
            print(f"   ✗ {metric}: Worse by {abs(delta):.3f} (p={p_value:.4f})")
        else:
            print(f"   ~ {metric}: Similar (Δ={delta:+.3f}, p={p_value:.4f})")
    else:
        if delta < 0 and p_value < 0.05:
            print(f"   ✓ {metric}: Better by {abs(delta):.3f} (p={p_value:.4f})")
        elif delta > 0 and p_value < 0.05:
            print(f"   ✗ {metric}: Worse by {abs(delta):.3f} (p={p_value:.4f})")
        else:
            print(f"   ~ {metric}: Similar (Δ={delta:+.3f}, p={p_value:.4f})")

print("\n2. IGA (new) vs IGA (old):")
for metric in metrics:
    vals_new = df[f'{metric}_iga_new'].values
    vals_old = df[f'{metric}_iga_old'].values
    delta = vals_new.mean() - vals_old.mean()
    t_stat, p_value = stats.ttest_rel(vals_new, vals_old)

    if metric in better_higher:
        if delta > 0 and p_value < 0.05:
            print(f"   ✓ {metric}: Better by {abs(delta):.3f} (p={p_value:.4f})")
        elif delta < 0 and p_value < 0.05:
            print(f"   ✗ {metric}: Worse by {abs(delta):.3f} (p={p_value:.4f})")
        else:
            print(f"   ~ {metric}: Similar (Δ={delta:+.3f}, p={p_value:.4f})")
    else:
        if delta < 0 and p_value < 0.05:
            print(f"   ✓ {metric}: Better by {abs(delta):.3f} (p={p_value:.4f})")
        elif delta > 0 and p_value < 0.05:
            print(f"   ✗ {metric}: Worse by {abs(delta):.3f} (p={p_value:.4f})")
        else:
            print(f"   ~ {metric}: Similar (Δ={delta:+.3f}, p={p_value:.4f})")

print("\n" + "=" * 100)
