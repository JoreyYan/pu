"""
对比 coord_scale=1 vs 原始SH vs R3 的结果
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

def parse_diagnostics(diag_file):
    """解析diagnostics.txt"""
    with open(diag_file) as f:
        content = f.read()

    data = {}

    # Sidechain RMSD
    match = re.search(r'Sidechain RMSD.*?:\s*([\d.]+)', content)
    if match:
        data['rmsd'] = float(match.group(1))

    # Perplexity
    match = re.search(r'Perplexity with predicted coords:\s*([\d.]+)', content)
    if match:
        data['ppl_pred'] = float(match.group(1))

    match = re.search(r'Perplexity with GT coords:\s*([\d.]+)', content)
    if match:
        data['ppl_gt'] = float(match.group(1))

    # Recovery
    match = re.search(r'Recovery with predicted coords:\s*([\d.]+)', content)
    if match:
        data['rec_pred'] = float(match.group(1))

    match = re.search(r'Recovery with GT coords:\s*([\d.]+)', content)
    if match:
        data['rec_gt'] = float(match.group(1))

    return data

def analyze_experiment(exp_dir, name):
    """分析一个实验"""
    exp_path = Path(exp_dir)

    if not exp_path.exists():
        print(f"⚠️  目录不存在: {exp_dir}")
        return None

    sample_dirs = sorted([d for d in exp_path.iterdir()
                         if d.is_dir() and d.name.startswith('sample_')])

    if not sample_dirs:
        print(f"⚠️  {name}: 没有找到样本目录")
        return None

    all_data = []
    for sample_dir in sample_dirs:
        diag_file = sample_dir / 'diagnostics.txt'
        if diag_file.exists():
            data = parse_diagnostics(diag_file)
            data['sample'] = sample_dir.name
            all_data.append(data)

    if not all_data:
        print(f"⚠️  {name}: 没有有效数据")
        return None

    # 统计
    metrics = ['rmsd', 'ppl_pred', 'rec_pred']
    stats = {}

    for metric in metrics:
        values = [d[metric] for d in all_data if metric in d]
        if values:
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values)
            }

    return {
        'name': name,
        'n_samples': len(all_data),
        'stats': stats,
        'all_data': all_data
    }

def print_comparison(results_list):
    """打印对比"""
    print("\n" + "="*90)
    print("coord_scale=1 vs 原始SH vs R3 对比")
    print("="*90)

    names = [r['name'] for r in results_list]

    print(f"\n{'指标':<20}", end="")
    for name in names:
        print(f"{name:<25}", end="")
    print()
    print("-"*90)

    # RMSD
    print(f"{'Sidechain RMSD (Å)':<20}", end="")
    rmsd_values = []
    for r in results_list:
        if 'rmsd' in r['stats']:
            val = r['stats']['rmsd']['mean']
            rmsd_values.append(val)
            print(f"{val:<25.4f}", end="")
        else:
            rmsd_values.append(np.nan)
            print(f"{'N/A':<25}", end="")
    print()

    # Perplexity
    print(f"{'Perplexity (pred)':<20}", end="")
    ppl_values = []
    for r in results_list:
        if 'ppl_pred' in r['stats']:
            val = r['stats']['ppl_pred']['mean']
            ppl_values.append(val)
            print(f"{val:<25.3f}", end="")
        else:
            ppl_values.append(np.nan)
            print(f"{'N/A':<25}", end="")
    print()

    # Recovery
    print(f"{'Recovery (pred)':<20}", end="")
    rec_values = []
    for r in results_list:
        if 'rec_pred' in r['stats']:
            val = r['stats']['rec_pred']['mean']
            rec_values.append(val)
            print(f"{val:<25.3f}", end="")
        else:
            rec_values.append(np.nan)
            print(f"{'N/A':<25}", end="")
    print()

    # 判断
    print("\n" + "="*90)
    print("结论")
    print("="*90)

    if len(rmsd_values) >= 3:
        coordscale1_rmsd = rmsd_values[0]
        original_sh_rmsd = rmsd_values[1]
        r3_rmsd = rmsd_values[2]

        print(f"\n1. coord_scale=1 vs 原始SH:")
        if not np.isnan(coordscale1_rmsd) and not np.isnan(original_sh_rmsd):
            improvement = (original_sh_rmsd - coordscale1_rmsd) / original_sh_rmsd * 100
            print(f"   RMSD: {original_sh_rmsd:.3f}Å → {coordscale1_rmsd:.3f}Å")
            print(f"   改善: {improvement:+.1f}%")

            if coordscale1_rmsd < 1.5:
                print(f"   ✅ 显著改善！coord_scale是关键问题")
                print(f"   → 建议：使用coord_scale=1继续SH方向")
            elif coordscale1_rmsd < 2.0:
                print(f"   ⚠️  有一定改善，但仍不如R3")
                print(f"   → 建议：测试其他coord_scale值（如5, 8）")
            elif improvement > 5:
                print(f"   ⚠️  略有改善")
                print(f"   → 建议：可能需要重新训练而非仅推理")
            else:
                print(f"   ❌ 基本没改善")
                print(f"   → 建议：放弃SH，专注R3")

        print(f"\n2. coord_scale=1 vs R3:")
        if not np.isnan(coordscale1_rmsd) and not np.isnan(r3_rmsd):
            diff = coordscale1_rmsd - r3_rmsd
            print(f"   coord_scale=1: {coordscale1_rmsd:.3f}Å")
            print(f"   R3 FBB:        {r3_rmsd:.3f}Å")
            print(f"   差异:          {diff:+.3f}Å ({diff/r3_rmsd*100:+.1f}%)")

            if coordscale1_rmsd < r3_rmsd:
                print(f"   ✅ coord_scale=1的SH已超过R3！")
            elif abs(diff) < 0.2:
                print(f"   ✓ coord_scale=1的SH接近R3水平")
            else:
                print(f"   ❌ coord_scale=1的SH仍不如R3")

def main():
    # 定义实验目录
    experiments = [
        # 修改为实际的coord_scale=1输出目录
        ('/home/junyu/project/pu/outputs/shfbb_coordscale1_step10/val_seperated_*', 'SH coordscale=1'),

        # 原始SH
        ('/home/junyu/project/pu/outputs/shfbb_atoms_cords2_step10/val_seperated_Rm0_t0_step0_20251116_185102', 'SH 原始'),

        # R3 FBB
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step10/val_seperated_Rm0_t0_step0_20251116_210156', 'R3 FBB'),
    ]

    results_list = []

    for exp_dir, name in experiments:
        print(f"处理 {name}...")

        # 如果路径包含通配符，尝试找到匹配的目录
        if '*' in exp_dir:
            from glob import glob
            matches = glob(exp_dir)
            if matches:
                exp_dir = matches[0]
            else:
                print(f"⚠️  未找到匹配的目录: {exp_dir}")
                continue

        result = analyze_experiment(exp_dir, name)
        if result:
            results_list.append(result)

    if len(results_list) >= 2:
        print_comparison(results_list)
    else:
        print("\n⚠️  数据不足，无法对比")

    print("\n" + "="*90)
    print("详细数据")
    print("="*90)
    for r in results_list:
        print(f"\n{r['name']}:")
        print(f"  样本数: {r['n_samples']}")
        for metric, stats in r['stats'].items():
            print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

if __name__ == '__main__':
    main()
