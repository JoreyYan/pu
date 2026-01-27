"""
对比 SH+FBB vs R3 FBB 的ESMFold评估结果
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_and_analyze(csv_path, name):
    """加载并分析结果"""
    df = pd.read_csv(csv_path)

    print(f"\n{'='*80}")
    print(f"{name} - ESMFold评估结果")
    print(f"{'='*80}")

    print(f"\n样本数: {len(df)}")

    metrics = [
        ('TM_score', 'TM-score', '.3f'),
        ('RMSD', 'RMSD (Å)', '.3f'),
        ('pLDDT', 'pLDDT', '.2f'),
        ('pAE', 'pAE', '.3f'),
        ('recovery', 'Recovery', '.3f'),
        ('perplexity', 'Perplexity', '.2f'),
    ]

    print(f"\n{'指标':<20} {'平均':<12} {'标准差':<12} {'中位数':<12} {'范围'}")
    print("-"*80)

    stats = {}
    for col, label, fmt in metrics:
        if col in df.columns:
            values = df[col].dropna()
            mean_val = values.mean()
            std_val = values.std()
            median_val = values.median()
            min_val = values.min()
            max_val = values.max()

            stats[col] = {
                'mean': mean_val,
                'std': std_val,
                'median': median_val,
                'min': min_val,
                'max': max_val
            }

            print(f"{label:<20} {mean_val:<12{fmt}} {std_val:<12{fmt}} {median_val:<12{fmt}} [{min_val:{fmt}}, {max_val:{fmt}}]")

    return df, stats

def compare_sh_vs_r3(sh_stats, r3_stats):
    """对比SH+FBB vs R3 FBB"""
    print(f"\n{'='*80}")
    print("SH+FBB vs R3 FBB 对比")
    print(f"{'='*80}")

    metrics = [
        ('TM_score', 'TM-score', '.4f', True),  # True表示越大越好
        ('RMSD', 'RMSD (Å)', '.3f', False),
        ('pLDDT', 'pLDDT', '.2f', True),
        ('pAE', 'pAE', '.3f', False),
        ('recovery', 'Recovery', '.3f', True),
        ('perplexity', 'Perplexity', '.2f', False),
    ]

    print(f"\n{'指标':<20} {'SH+FBB':<15} {'R3 FBB':<15} {'差异':<15} {'胜负'}")
    print("-"*80)

    for col, label, fmt, higher_better in metrics:
        sh_val = sh_stats[col]['mean']
        r3_val = r3_stats[col]['mean']

        diff = sh_val - r3_val
        diff_pct = (diff / abs(r3_val)) * 100

        # 判断胜负
        if higher_better:
            winner = "SH+FBB ✓" if sh_val > r3_val else "R3 FBB ✓"
        else:
            winner = "SH+FBB ✓" if sh_val < r3_val else "R3 FBB ✓"

        diff_str = f"{diff:+{fmt}} ({diff_pct:+.1f}%)"

        print(f"{label:<20} {sh_val:<15{fmt}} {r3_val:<15{fmt}} {diff_str:<15} {winner}")

    # 详细分析
    print(f"\n{'='*80}")
    print("关键发现")
    print(f"{'='*80}")

    # 1. TM-score
    tm_sh = sh_stats['TM_score']['mean']
    tm_r3 = r3_stats['TM_score']['mean']

    print(f"\n1. TM-score（结构相似度）:")
    print(f"   SH+FBB: {tm_sh:.4f}")
    print(f"   R3 FBB: {tm_r3:.4f}")
    print(f"   差异:   {(tm_sh-tm_r3):.4f} ({(tm_sh-tm_r3)/tm_r3*100:+.1f}%)")

    if abs(tm_sh - tm_r3) < 0.02:
        print(f"   → TM-score基本相同")
    elif tm_sh > tm_r3:
        print(f"   ✓ SH+FBB的sequence可折叠性略好")
    else:
        print(f"   ✓ R3 FBB的sequence可折叠性略好")

    # 2. Recovery
    rec_sh = sh_stats['recovery']['mean']
    rec_r3 = r3_stats['recovery']['mean']

    print(f"\n2. Recovery（序列恢复率）:")
    print(f"   SH+FBB: {rec_sh:.3f} ({rec_sh*100:.1f}%)")
    print(f"   R3 FBB: {rec_r3:.3f} ({rec_r3*100:.1f}%)")
    print(f"   差异:   {(rec_sh-rec_r3)*100:.1f}%")

    if abs(rec_sh - rec_r3) < 0.02:
        print(f"   → Recovery基本相同")
    elif rec_sh > rec_r3:
        print(f"   ✓ SH+FBB的sequence恢复率更高")
    else:
        print(f"   ✓ R3 FBB的sequence恢复率更高")

    # 3. pLDDT
    plddt_sh = sh_stats['pLDDT']['mean']
    plddt_r3 = r3_stats['pLDDT']['mean']

    print(f"\n3. pLDDT（ESMFold置信度）:")
    print(f"   SH+FBB: {plddt_sh:.2f}")
    print(f"   R3 FBB: {plddt_r3:.2f}")
    print(f"   差异:   {(plddt_sh-plddt_r3):.2f} ({(plddt_sh-plddt_r3)/plddt_r3*100:+.1f}%)")

    if plddt_sh < plddt_r3 - 5:
        print(f"   ❌ SH+FBB的sequence质量明显更差（置信度低）")
    elif plddt_r3 < plddt_sh - 5:
        print(f"   ❌ R3 FBB的sequence质量明显更差（置信度低）")
    else:
        print(f"   → 置信度基本相当")

    # 4. Perplexity
    ppl_sh = sh_stats['perplexity']['mean']
    ppl_r3 = r3_stats['perplexity']['mean']

    print(f"\n4. Perplexity（序列-结构一致性）:")
    print(f"   SH+FBB: {ppl_sh:.2f}")
    print(f"   R3 FBB: {ppl_r3:.2f}")
    print(f"   差异:   {(ppl_sh-ppl_r3):.2f} ({(ppl_sh-ppl_r3)/ppl_r3*100:+.1f}%)")

    if ppl_sh < ppl_r3:
        print(f"   ✓ SH+FBB的perplexity更低（更好）")
    else:
        print(f"   ✓ R3 FBB的perplexity更低（更好）")

    # 5. RMSD
    rmsd_sh = sh_stats['RMSD']['mean']
    rmsd_r3 = r3_stats['RMSD']['mean']

    print(f"\n5. RMSD（vs ESMFold折叠结构）:")
    print(f"   SH+FBB: {rmsd_sh:.3f} Å")
    print(f"   R3 FBB: {rmsd_r3:.3f} Å")
    print(f"   差异:   {(rmsd_sh-rmsd_r3):.3f} Å ({(rmsd_sh-rmsd_r3)/rmsd_r3*100:+.1f}%)")

    if rmsd_sh < rmsd_r3:
        print(f"   ✓ SH+FBB的RMSD更低")
    else:
        print(f"   ✓ R3 FBB的RMSD更低")

def compare_with_direct_coordinates(sh_stats, r3_stats):
    """对比ESMFold结果和直接坐标的诊断"""
    print(f"\n{'='*80}")
    print("ESMFold vs 直接坐标诊断对比")
    print(f"{'='*80}")

    # 从之前的诊断数据（手动输入）
    direct_coords = {
        'SH+FBB': {
            'rmsd': 2.31,
            'recovery': 0.643,
            'perplexity': 4.73,
        },
        'R3 FBB': {
            'rmsd': 1.059,
            'recovery': 0.682,
            'perplexity': 8.87,
        }
    }

    print(f"\n{'指标':<20} {'方法':<15} {'直接坐标':<15} {'ESMFold':<15} {'一致性'}")
    print("-"*80)

    # Recovery
    print(f"{'Recovery':<20} {'SH+FBB':<15} {direct_coords['SH+FBB']['recovery']:<15.3f} {sh_stats['recovery']['mean']:<15.3f} ", end="")
    if abs(direct_coords['SH+FBB']['recovery'] - sh_stats['recovery']['mean']) < 0.05:
        print("✓ 一致")
    else:
        print("⚠️  有差异")

    print(f"{'Recovery':<20} {'R3 FBB':<15} {direct_coords['R3 FBB']['recovery']:<15.3f} {r3_stats['recovery']['mean']:<15.3f} ", end="")
    if abs(direct_coords['R3 FBB']['recovery'] - r3_stats['recovery']['mean']) < 0.05:
        print("✓ 一致")
    else:
        print("⚠️  有差异")

    # Perplexity
    print(f"{'Perplexity':<20} {'SH+FBB':<15} {direct_coords['SH+FBB']['perplexity']:<15.2f} {sh_stats['perplexity']['mean']:<15.2f} ", end="")
    if abs(direct_coords['SH+FBB']['perplexity'] - sh_stats['perplexity']['mean']) < 2:
        print("✓ 一致")
    else:
        print("⚠️  有差异")

    print(f"{'Perplexity':<20} {'R3 FBB':<15} {direct_coords['R3 FBB']['perplexity']:<15.2f} {r3_stats['perplexity']['mean']:<15.2f} ", end="")
    if abs(direct_coords['R3 FBB']['perplexity'] - r3_stats['perplexity']['mean']) < 2:
        print("✓ 一致")
    else:
        print("⚠️  有差异")

    print(f"\n注意：直接坐标的RMSD是相对GT的，ESMFold的RMSD是相对重折叠结构的，不可比。")

def main():
    # SH+FBB
    sh_path = '/home/junyu/project/pu/outputs/shfbb_atoms_cords2_step10/esmfold_eval/fbb_results/fbb_scores.csv'

    # R3 FBB
    r3_path = '/outputs/r3fbb_atoms_cords1_step10/val_seperated_Rm0_t0_step0_20251116_210156/esmfold_eval/fbb_results/fbb_scores.csv'

    if not Path(sh_path).exists():
        print(f"⚠️  文件不存在: {sh_path}")
        return

    if not Path(r3_path).exists():
        print(f"⚠️  文件不存在: {r3_path}")
        return

    # 加载数据
    sh_df, sh_stats = load_and_analyze(sh_path, "SH+FBB (10步)")
    r3_df, r3_stats = load_and_analyze(r3_path, "R3 FBB (10步)")

    # 对比
    compare_sh_vs_r3(sh_stats, r3_stats)

    # 和直接坐标对比
    compare_with_direct_coordinates(sh_stats, r3_stats)

    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    print("\nESMFold评估反映的是predicted sequence的质量：")
    print("  - TM-score: 折叠后的结构与GT的相似度")
    print("  - pLDDT: ESMFold对sequence的置信度")
    print("  - Recovery: ESMFold折叠后预测的sequence vs GT的匹配度")
    print("\n关键洞察：")
    print("  1. 如果SH+FBB和R3 FBB的ESMFold指标相近")
    print("     → 说明predicted sequence质量差不多")
    print("  2. 但直接坐标的RMSD差异大（2.3 vs 1.06Å）")
    print("     → 说明问题在坐标质量，不在sequence层面")

if __name__ == '__main__':
    main()
