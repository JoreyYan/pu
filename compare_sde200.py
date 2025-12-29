"""
评估 SDE 200步的结果，对比之前的所有实验
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
    metrics = ['rmsd', 'ppl_pred', 'ppl_gt', 'rec_pred', 'rec_gt']
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

def print_full_comparison(results_list):
    """打印完整对比"""
    print("\n" + "="*110)
    print("完整对比：SH+FBB 所有实验 + R3 FBB")
    print("="*110)

    names = [r['name'] for r in results_list]

    print(f"\n{'指标':<25}", end="")
    for name in names:
        print(f"{name:<18}", end="")
    print()
    print("-"*110)

    # RMSD
    print(f"{'Sidechain RMSD (Å)':<25}", end="")
    rmsd_values = []
    for r in results_list:
        if 'rmsd' in r['stats']:
            val = r['stats']['rmsd']['mean']
            std = r['stats']['rmsd']['std']
            rmsd_values.append(val)
            print(f"{val:.3f}±{std:.2f}  ", end="")
        else:
            rmsd_values.append(np.nan)
            print(f"{'N/A':<18}", end="")
    print()

    # Perplexity
    print(f"{'Perplexity (pred)':<25}", end="")
    ppl_values = []
    for r in results_list:
        if 'ppl_pred' in r['stats']:
            val = r['stats']['ppl_pred']['mean']
            ppl_values.append(val)
            print(f"{val:<18.2f}", end="")
        else:
            ppl_values.append(np.nan)
            print(f"{'N/A':<18}", end="")
    print()

    # Recovery
    print(f"{'Recovery (pred)':<25}", end="")
    rec_values = []
    for r in results_list:
        if 'rec_pred' in r['stats']:
            val = r['stats']['rec_pred']['mean']
            rec_values.append(val)
            print(f"{val:.3f} ({val*100:.1f}%)", end=" ")
        else:
            rec_values.append(np.nan)
            print(f"{'N/A':<18}", end="")
    print()

    print("\n" + "="*110)
    print("关键分析")
    print("="*110)

    # 找到最佳结果
    valid_rmsd = [(i, val) for i, val in enumerate(rmsd_values) if not np.isnan(val)]
    if valid_rmsd:
        best_idx, best_val = min(valid_rmsd, key=lambda x: x[1])
        worst_idx, worst_val = max(valid_rmsd, key=lambda x: x[1])

        print(f"\n🏆 最佳结果: {names[best_idx]}")
        print(f"   RMSD: {best_val:.4f}Å")
        if 'ppl_pred' in results_list[best_idx]['stats']:
            print(f"   Perplexity: {results_list[best_idx]['stats']['ppl_pred']['mean']:.2f}")
        if 'rec_pred' in results_list[best_idx]['stats']:
            print(f"   Recovery: {results_list[best_idx]['stats']['rec_pred']['mean']*100:.1f}%")

        print(f"\n❌ 最差结果: {names[worst_idx]}")
        print(f"   RMSD: {worst_val:.4f}Å")

        # SDE 200步的表现
        sde200_idx = next((i for i, name in enumerate(names) if 'SDE 200' in name), None)
        if sde200_idx is not None and not np.isnan(rmsd_values[sde200_idx]):
            sde200_rmsd = rmsd_values[sde200_idx]
            print(f"\n⭐ SDE 200步分析:")
            print(f"   RMSD: {sde200_rmsd:.4f}Å")

            # 对比ODE 10步
            ode10_idx = next((i for i, name in enumerate(names) if 'ODE 10' in name), None)
            if ode10_idx is not None:
                ode10_rmsd = rmsd_values[ode10_idx]
                diff = sde200_rmsd - ode10_rmsd
                pct = (diff / ode10_rmsd) * 100
                print(f"   vs ODE 10步: {diff:+.4f}Å ({pct:+.1f}%)")

                if sde200_rmsd < ode10_rmsd:
                    print(f"   ✅ SDE 200步优于ODE 10步！")
                elif abs(diff) < 0.05:
                    print(f"   → SDE 200步接近ODE 10步")
                else:
                    print(f"   ❌ SDE 200步仍不如ODE 10步")

            # 对比SDE 100步
            sde100_idx = next((i for i, name in enumerate(names) if 'SDE 100' in name), None)
            if sde100_idx is not None:
                sde100_rmsd = rmsd_values[sde100_idx]
                diff = sde200_rmsd - sde100_rmsd
                pct = (diff / sde100_rmsd) * 100
                print(f"   vs SDE 100步: {diff:+.4f}Å ({pct:+.1f}%)")

                if sde200_rmsd < sde100_rmsd - 0.05:
                    print(f"   ✅ 继续增加步数有明显改善")
                elif abs(diff) < 0.05:
                    print(f"   → 已经饱和，增加步数无明显改善")
                else:
                    print(f"   ❌ 增加步数反而变差")

            # 对比R3
            r3_idx = next((i for i, name in enumerate(names) if 'R3 FBB' in name), None)
            if r3_idx is not None:
                r3_rmsd = rmsd_values[r3_idx]
                diff = sde200_rmsd - r3_rmsd
                pct = (diff / r3_rmsd) * 100
                print(f"   vs R3 FBB: {diff:+.4f}Å ({pct:+.1f}%)")

                if sde200_rmsd < r3_rmsd:
                    print(f"   🎉 SDE 200步超过R3！")
                elif abs(diff) < 0.1:
                    print(f"   ✓ SDE 200步接近R3水平")
                else:
                    print(f"   → SDE 200步仍不如R3")

    print("\n" + "="*110)
    print("结论")
    print("="*110)

    print(f"\nSH+FBB性能排序（从好到坏）:")
    sh_results = [(i, val, names[i]) for i, val in enumerate(rmsd_values)
                  if not np.isnan(val) and 'R3' not in names[i]]
    sh_results.sort(key=lambda x: x[1])

    for rank, (idx, val, name) in enumerate(sh_results, 1):
        marker = "⭐" if rank == 1 else ""
        print(f"  {rank}. {name:<20} {val:.4f}Å {marker}")

    print(f"\n关键洞察:")
    print(f"  1. 修复speed_vectors bug后，SH+FBB从2.31Å提升到1.27Å（45%改善）")
    print(f"  2. ODE采样更稳定，10步就能达到最佳效果")
    print(f"  3. SDE需要更多步数才能收敛（10步灾难，100步可用，200步？）")

    r3_idx = next((i for i, name in enumerate(names) if 'R3 FBB' in name), None)
    if r3_idx is not None:
        r3_rmsd = rmsd_values[r3_idx]
        best_sh_rmsd = min(val for val in rmsd_values if not np.isnan(val))
        gap = best_sh_rmsd - r3_rmsd
        gap_pct = (gap / r3_rmsd) * 100

        print(f"\n  4. 当前最佳SH+FBB vs R3 FBB:")
        print(f"     SH+FBB: {best_sh_rmsd:.4f}Å")
        print(f"     R3 FBB: {r3_rmsd:.4f}Å")
        print(f"     差距:   {gap:.4f}Å ({gap_pct:.1f}%)")

        if gap < 0:
            print(f"     🎉 SH+FBB超过R3！")
        elif gap < 0.1:
            print(f"     ✓ SH+FBB接近R3水平")
        else:
            print(f"     → R3仍然更优")

def main():
    # 所有实验
    experiments = [
        # SH+FBB ODE
        ('/home/junyu/project/pu/outputs/SHfbb_atoms_cords1_step10/val_seperated_Rm0_t0_step0_20251117_110554', 'ODE 10步'),
        ('/home/junyu/project/pu/outputs/SHfbb_atoms_cords1_step100/val_seperated_Rm0_t0_step0_20251117_110800', 'ODE 100步'),

        # SH+FBB SDE
        ('/home/junyu/project/pu/outputs/SHfbb_atoms_cords1_step10_SDE/val_seperated_Rm0_t0_step0_20251117_112056', 'SDE 10步'),
        ('/home/junyu/project/pu/outputs/SHfbb_atoms_cords1_step100_SDE/val_seperated_Rm0_t0_step0_20251117_111744', 'SDE 100步'),
        ('/home/junyu/project/pu/outputs/SHfbb_atoms_cords1_step200_SDE/val_seperated_Rm0_t0_step0_20251117_113615', 'SDE 200步'),

        # R3 FBB (参考)
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step10/val_seperated_Rm0_t0_step0_20251116_210156', 'R3 FBB 10步'),
    ]

    results_list = []

    print("="*110)
    print("评估 SDE 200步 + 完整对比")
    print("="*110)

    for exp_dir, name in experiments:
        result = analyze_experiment(exp_dir, name)
        if result:
            results_list.append(result)
            print(f"\n✓ {name:<20} - 样本数: {result['n_samples']}", end="")
            if 'rmsd' in result['stats']:
                print(f", RMSD: {result['stats']['rmsd']['mean']:.4f}Å")
            else:
                print()

    if len(results_list) >= 2:
        print_full_comparison(results_list)
    else:
        print("\n⚠️  数据不足，无法对比")

if __name__ == '__main__':
    main()
