"""
对比三个step的ESMFold评估结果
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_esmfold_results(csv_path):
    """加载ESMFold评估结果"""
    df = pd.read_csv(csv_path)
    return df

def print_statistics(df, name):
    """打印统计信息"""
    print(f"\n{'='*80}")
    print(f"{name} - ESMFold评估结果")
    print(f"{'='*80}")

    print(f"\n样本数: {len(df)}")

    metrics = [
        ('TM_score', 'TM-score', '.3f', True),  # True表示越大越好
        ('RMSD', 'RMSD (Å)', '.3f', False),
        ('pLDDT', 'pLDDT', '.2f', True),
        ('pAE', 'pAE', '.3f', False),
        ('recovery', 'Recovery', '.3f', True),
        ('perplexity', 'Perplexity', '.2f', False),
    ]

    print(f"\n{'指标':<20} {'平均':<12} {'标准差':<12} {'中位数':<12} {'范围'}")
    print("-"*80)

    stats = {}
    for col, label, fmt, _ in metrics:
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

    return stats

def compare_results(results_dict):
    """对比三个step的结果"""
    print(f"\n{'='*80}")
    print("三个步数的ESMFold结果对比")
    print(f"{'='*80}")

    names = ['10步', '100步', '500步']

    metrics = [
        ('TM_score', 'TM-score', '.4f', True),
        ('RMSD', 'RMSD (Å)', '.3f', False),
        ('pLDDT', 'pLDDT', '.2f', True),
        ('pAE', 'pAE', '.3f', False),
        ('recovery', 'Recovery', '.3f', True),
        ('perplexity', 'Perplexity', '.2f', False),
    ]

    print(f"\n{'指标':<20} {'10步':<15} {'100步':<15} {'500步':<15} {'变化趋势'}")
    print("-"*90)

    for col, label, fmt, higher_better in metrics:
        values = []
        for name in names:
            if name in results_dict and col in results_dict[name]:
                values.append(results_dict[name][col]['mean'])
            else:
                values.append(np.nan)

        val_10, val_100, val_500 = values

        # 计算变化
        if not np.isnan(val_10) and not np.isnan(val_100):
            change_100 = (val_100 - val_10) / abs(val_10) * 100
        else:
            change_100 = 0

        if not np.isnan(val_10) and not np.isnan(val_500):
            change_500 = (val_500 - val_10) / abs(val_10) * 100
        else:
            change_500 = 0

        # 判断趋势
        if higher_better:
            trend_100 = "↑" if change_100 > 1 else "↓" if change_100 < -1 else "→"
            trend_500 = "↑" if change_500 > 1 else "↓" if change_500 < -1 else "→"
        else:
            trend_100 = "↓" if change_100 > 1 else "↑" if change_100 < -1 else "→"
            trend_500 = "↓" if change_500 > 1 else "↑" if change_500 < -1 else "→"

        trend_str = f"{trend_100} {change_100:+.1f}% / {trend_500} {change_500:+.1f}%"

        print(f"{label:<20} {val_10:<15{fmt}} {val_100:<15{fmt}} {val_500:<15{fmt}} {trend_str}")

    # 详细分析
    print(f"\n{'='*80}")
    print("关键发现")
    print(f"{'='*80}")

    # TM-score
    tm_10 = results_dict['10步']['TM_score']['mean']
    tm_100 = results_dict['100步']['TM_score']['mean']
    tm_500 = results_dict['500步']['TM_score']['mean']

    print(f"\n1. TM-score（结构相似度）:")
    print(f"   10步:  {tm_10:.4f}")
    print(f"   100步: {tm_100:.4f} ({(tm_100-tm_10)/tm_10*100:+.1f}%)")
    print(f"   500步: {tm_500:.4f} ({(tm_500-tm_10)/tm_10*100:+.1f}%)")

    if tm_100 > tm_10 and tm_500 > tm_10:
        print(f"   ✓ 增加步数提升了TM-score")
    elif abs(tm_100 - tm_10) < 0.01 and abs(tm_500 - tm_10) < 0.01:
        print(f"   → TM-score基本不变")
    else:
        print(f"   ❌ TM-score有变化但无明确趋势")

    # RMSD
    rmsd_10 = results_dict['10步']['RMSD']['mean']
    rmsd_100 = results_dict['100步']['RMSD']['mean']
    rmsd_500 = results_dict['500步']['RMSD']['mean']

    print(f"\n2. RMSD（vs ESMFold折叠结构）:")
    print(f"   10步:  {rmsd_10:.3f} Å")
    print(f"   100步: {rmsd_100:.3f} Å ({(rmsd_100-rmsd_10)/rmsd_10*100:+.1f}%)")
    print(f"   500步: {rmsd_500:.3f} Å ({(rmsd_500-rmsd_10)/rmsd_10*100:+.1f}%)")

    if rmsd_100 < rmsd_10 and rmsd_500 < rmsd_10:
        print(f"   ✓ 增加步数降低了RMSD")
    elif abs(rmsd_100 - rmsd_10) < 0.5 and abs(rmsd_500 - rmsd_10) < 0.5:
        print(f"   → RMSD基本不变")
    else:
        print(f"   ⚠️  RMSD变化无明确趋势")

    # pLDDT
    plddt_10 = results_dict['10步']['pLDDT']['mean']
    plddt_100 = results_dict['100步']['pLDDT']['mean']
    plddt_500 = results_dict['500步']['pLDDT']['mean']

    print(f"\n3. pLDDT（ESMFold预测置信度）:")
    print(f"   10步:  {plddt_10:.2f}")
    print(f"   100步: {plddt_100:.2f} ({(plddt_100-plddt_10)/plddt_10*100:+.1f}%)")
    print(f"   500步: {plddt_500:.2f} ({(plddt_500-plddt_10)/plddt_10*100:+.1f}%)")

    if plddt_100 > plddt_10 and plddt_500 > plddt_10:
        print(f"   ✓ 增加步数提升了pLDDT（序列质量更好）")
    elif abs(plddt_100 - plddt_10) < 2 and abs(plddt_500 - plddt_10) < 2:
        print(f"   → pLDDT基本不变")
    else:
        print(f"   ⚠️  pLDDT变化无明确趋势")

    # Recovery
    rec_10 = results_dict['10步']['recovery']['mean']
    rec_100 = results_dict['100步']['recovery']['mean']
    rec_500 = results_dict['500步']['recovery']['mean']

    print(f"\n4. Recovery（序列恢复率）:")
    print(f"   10步:  {rec_10:.3f} ({rec_10*100:.1f}%)")
    print(f"   100步: {rec_100:.3f} ({rec_100*100:.1f}%, {(rec_100-rec_10)*100:+.1f}%)")
    print(f"   500步: {rec_500:.3f} ({rec_500*100:.1f}%, {(rec_500-rec_10)*100:+.1f}%)")

    print(f"\n   注意：这里的recovery是ESMFold折叠后的结构预测的sequence vs 原始GT sequence")
    print(f"   不同于之前的recovery（基于R3生成坐标的logits预测）")

    # Perplexity
    ppl_10 = results_dict['10步']['perplexity']['mean']
    ppl_100 = results_dict['100步']['perplexity']['mean']
    ppl_500 = results_dict['500步']['perplexity']['mean']

    print(f"\n5. Perplexity（序列-结构一致性）:")
    print(f"   10步:  {ppl_10:.2f}")
    print(f"   100步: {ppl_100:.2f} ({(ppl_100-ppl_10)/ppl_10*100:+.1f}%)")
    print(f"   500步: {ppl_500:.2f} ({(ppl_500-ppl_10)/ppl_10*100:+.1f}%)")

def main():
    experiments = [
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step10/esmfold_eval/fbb_results/fbb_scores.csv', '10步'),
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step100/esmfold_eval/fbb_results/fbb_scores.csv', '100步'),
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step500/esmfold_eval/fbb_results/fbb_scores.csv', '500步'),
    ]

    results_dict = {}

    for csv_path, name in experiments:
        if Path(csv_path).exists():
            df = load_esmfold_results(csv_path)
            stats = print_statistics(df, name)
            results_dict[name] = stats
        else:
            print(f"\n⚠️  文件不存在: {csv_path}")

    if len(results_dict) == 3:
        compare_results(results_dict)
    else:
        print(f"\n⚠️  未能加载所有三个实验的结果")

    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    print("\nESMFold评估的是：")
    print("  1. 用predicted sequence（从R3坐标的logits得到）")
    print("  2. 通过ESMFold折叠得到新的结构")
    print("  3. 对比这个结构与GT结构的相似度")
    print("\n这个指标反映：predicted sequence的质量（可折叠性、合理性）")

if __name__ == '__main__':
    main()
