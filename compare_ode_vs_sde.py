"""
对比 SH+FBB 的 ODE vs SDE 采样结果
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

    match = re.search(r'Perplexity degradation:\s*([\d.]+)', content)
    if match:
        data['ppl_deg'] = float(match.group(1))

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
    metrics = ['rmsd', 'ppl_pred', 'ppl_gt', 'ppl_deg', 'rec_pred', 'rec_gt']
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
    print("\n" + "="*100)
    print("SH+FBB: ODE vs SDE 对比")
    print("="*100)

    names = [r['name'] for r in results_list]

    print(f"\n{'指标':<25}", end="")
    for name in names:
        print(f"{name:<20}", end="")
    print()
    print("-"*100)

    # RMSD
    print(f"{'Sidechain RMSD (Å)':<25}", end="")
    rmsd_values = []
    for r in results_list:
        if 'rmsd' in r['stats']:
            val = r['stats']['rmsd']['mean']
            std = r['stats']['rmsd']['std']
            rmsd_values.append(val)
            print(f"{val:.4f}±{std:.3f}  ", end="")
        else:
            rmsd_values.append(np.nan)
            print(f"{'N/A':<20}", end="")
    print()

    # Perplexity (pred)
    print(f"{'Perplexity (pred)':<25}", end="")
    ppl_values = []
    for r in results_list:
        if 'ppl_pred' in r['stats']:
            val = r['stats']['ppl_pred']['mean']
            std = r['stats']['ppl_pred']['std']
            ppl_values.append(val)
            print(f"{val:.3f}±{std:.2f}    ", end="")
        else:
            ppl_values.append(np.nan)
            print(f"{'N/A':<20}", end="")
    print()

    # Perplexity (GT)
    print(f"{'Perplexity (GT)':<25}", end="")
    for r in results_list:
        if 'ppl_gt' in r['stats']:
            val = r['stats']['ppl_gt']['mean']
            std = r['stats']['ppl_gt']['std']
            print(f"{val:.3f}±{std:.2f}    ", end="")
        else:
            print(f"{'N/A':<20}", end="")
    print()

    # Recovery (pred)
    print(f"{'Recovery (pred)':<25}", end="")
    rec_values = []
    for r in results_list:
        if 'rec_pred' in r['stats']:
            val = r['stats']['rec_pred']['mean']
            std = r['stats']['rec_pred']['std']
            rec_values.append(val)
            print(f"{val:.3f}±{std:.3f}  ", end="")
        else:
            rec_values.append(np.nan)
            print(f"{'N/A':<20}", end="")
    print()

    # Recovery (GT)
    print(f"{'Recovery (GT)':<25}", end="")
    for r in results_list:
        if 'rec_gt' in r['stats']:
            val = r['stats']['rec_gt']['mean']
            std = r['stats']['rec_gt']['std']
            print(f"{val:.3f}±{std:.3f}  ", end="")
        else:
            print(f"{'N/A':<20}", end="")
    print()

    print("\n" + "="*100)
    print("关键发现")
    print("="*100)

    if len(rmsd_values) == 4:
        ode_10, ode_100, sde_10, sde_100 = rmsd_values

        print(f"\n1. ODE采样（增加步数的效果）:")
        print(f"   10步:  {ode_10:.4f}Å")
        print(f"   100步: {ode_100:.4f}Å ({(ode_100-ode_10)/ode_10*100:+.1f}%)")

        if abs(ode_100 - ode_10) < 0.1:
            print(f"   → ODE增加步数几乎无改善")
        elif ode_100 < ode_10:
            print(f"   ✓ ODE增加步数有改善")
        else:
            print(f"   ❌ ODE增加步数反而变差")

        print(f"\n2. SDE采样（增加步数的效果）:")
        print(f"   10步:  {sde_10:.4f}Å")
        print(f"   100步: {sde_100:.4f}Å ({(sde_100-sde_10)/sde_10*100:+.1f}%)")

        if abs(sde_100 - sde_10) < 0.1:
            print(f"   → SDE增加步数几乎无改善")
        elif sde_100 < sde_10:
            print(f"   ✓ SDE增加步数有改善")
        else:
            print(f"   ❌ SDE增加步数反而变差")

        print(f"\n3. ODE vs SDE（10步对比）:")
        print(f"   ODE 10步: {ode_10:.4f}Å")
        print(f"   SDE 10步: {sde_10:.4f}Å")
        print(f"   差异:     {(sde_10-ode_10):.4f}Å ({(sde_10-ode_10)/ode_10*100:+.1f}%)")

        if sde_10 < ode_10 - 0.05:
            print(f"   ✓ SDE明显优于ODE")
        elif abs(sde_10 - ode_10) < 0.05:
            print(f"   → SDE和ODE基本相当")
        else:
            print(f"   ❌ SDE不如ODE")

        print(f"\n4. ODE vs SDE（100步对比）:")
        print(f"   ODE 100步: {ode_100:.4f}Å")
        print(f"   SDE 100步: {sde_100:.4f}Å")
        print(f"   差异:      {(sde_100-ode_100):.4f}Å ({(sde_100-ode_100)/ode_100*100:+.1f}%)")

        if sde_100 < ode_100 - 0.05:
            print(f"   ✓ SDE明显优于ODE")
        elif abs(sde_100 - ode_100) < 0.05:
            print(f"   → SDE和ODE基本相当")
        else:
            print(f"   ❌ SDE不如ODE")

        print(f"\n5. 最佳结果:")
        best_val = min(rmsd_values)
        best_idx = rmsd_values.index(best_val)
        best_name = names[best_idx]
        print(f"   {best_name}: {best_val:.4f}Å ⭐")

    # Perplexity分析
    if len(ppl_values) == 4:
        print(f"\n6. Perplexity对比:")
        ode_10_ppl, ode_100_ppl, sde_10_ppl, sde_100_ppl = ppl_values
        print(f"   ODE 10步:  {ode_10_ppl:.3f}")
        print(f"   ODE 100步: {ode_100_ppl:.3f} ({(ode_100_ppl-ode_10_ppl)/ode_10_ppl*100:+.1f}%)")
        print(f"   SDE 10步:  {sde_10_ppl:.3f}")
        print(f"   SDE 100步: {sde_100_ppl:.3f} ({(sde_100_ppl-sde_10_ppl)/sde_10_ppl*100:+.1f}%)")

    # Recovery分析
    if len(rec_values) == 4:
        print(f"\n7. Recovery对比:")
        ode_10_rec, ode_100_rec, sde_10_rec, sde_100_rec = rec_values
        print(f"   ODE 10步:  {ode_10_rec:.3f} ({ode_10_rec*100:.1f}%)")
        print(f"   ODE 100步: {ode_100_rec:.3f} ({ode_100_rec*100:.1f}%)")
        print(f"   SDE 10步:  {sde_10_rec:.3f} ({sde_10_rec*100:.1f}%)")
        print(f"   SDE 100步: {sde_100_rec:.3f} ({sde_100_rec*100:.1f}%)")

    print("\n" + "="*100)
    print("结论")
    print("="*100)

    if len(rmsd_values) == 4:
        best_val = min(rmsd_values)
        worst_val = max(rmsd_values)
        improvement = (worst_val - best_val) / worst_val * 100

        print(f"\n最佳方法: {names[rmsd_values.index(best_val)]}")
        print(f"RMSD范围: {best_val:.4f}Å - {worst_val:.4f}Å")
        print(f"最大改善: {improvement:.1f}%")

        # 和之前的结果对比
        print(f"\n与之前实验对比:")
        print(f"  原始SH+FBB (10步, 旧代码): 2.31Å")
        print(f"  R3 FBB (10步):              1.06Å")
        print(f"  当前最佳 ({names[rmsd_values.index(best_val)]}): {best_val:.4f}Å")

        if best_val < 1.5:
            print(f"\n  ✅ 当前结果显著优于之前的SH+FBB！")
            print(f"     可能原因：修复了speed_vectors的bug")
        elif best_val < 2.0:
            print(f"\n  ⚠️  当前结果有改善，但仍不如R3")
        else:
            print(f"\n  ❌ 当前结果仍然接近之前的SH+FBB")
            print(f"     说明SH密度仍是主要瓶颈")

def main():
    experiments = [
        ('/home/junyu/project/pu/outputs/SHfbb_atoms_cords1_step10/val_seperated_Rm0_t0_step0_20251117_110554', 'ODE 10步'),
        ('/home/junyu/project/pu/outputs/SHfbb_atoms_cords1_step100/val_seperated_Rm0_t0_step0_20251117_110800', 'ODE 100步'),
        ('/home/junyu/project/pu/outputs/SHfbb_atoms_cords1_step10_SDE/val_seperated_Rm0_t0_step0_20251117_112056', 'SDE 10步'),
        ('/home/junyu/project/pu/outputs/SHfbb_atoms_cords1_step100_SDE/val_seperated_Rm0_t0_step0_20251117_111744', 'SDE 100步'),
    ]

    results_list = []

    for exp_dir, name in experiments:
        print(f"\n处理 {name}...")
        result = analyze_experiment(exp_dir, name)
        if result:
            results_list.append(result)
            print(f"  样本数: {result['n_samples']}")
            if 'rmsd' in result['stats']:
                print(f"  RMSD: {result['stats']['rmsd']['mean']:.4f} ± {result['stats']['rmsd']['std']:.4f}")

    if len(results_list) >= 2:
        print_comparison(results_list)
    else:
        print("\n⚠️  数据不足，无法对比")

if __name__ == '__main__':
    main()
