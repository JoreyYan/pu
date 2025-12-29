"""
比较R3 FBB在10/100/500步的详细诊断信息
"""
import re
from pathlib import Path
import numpy as np

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

    match = re.search(r'Recovery degradation:\s*(-?[\d.]+)', content)
    if match:
        data['rec_deg'] = float(match.group(1))

    return data

def analyze_experiment(exp_dir, name):
    """分析一个实验目录下所有样本"""
    exp_path = Path(exp_dir)

    all_data = []
    sample_dirs = sorted([d for d in exp_path.iterdir() if d.is_dir() and d.name.startswith('sample_')])

    for sample_dir in sample_dirs:
        diag_file = sample_dir / 'diagnostics.txt'
        if diag_file.exists():
            data = parse_diagnostics(diag_file)
            data['sample'] = sample_dir.name
            all_data.append(data)

    if not all_data:
        print(f"⚠️  {name}: 没有找到样本")
        return None

    # 统计
    metrics = ['rmsd', 'ppl_pred', 'ppl_gt', 'ppl_deg', 'rec_pred', 'rec_gt', 'rec_deg']
    stats = {}

    for metric in metrics:
        values = [d[metric] for d in all_data if metric in d]
        if values:
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }

    return {
        'name': name,
        'n_samples': len(all_data),
        'stats': stats,
        'all_data': all_data
    }

def print_comparison(results):
    """打印对比结果"""
    print("\n" + "="*80)
    print("R3 FBB: 10步 vs 100步 vs 500步 详细对比")
    print("="*80)

    # 表头
    print(f"\n{'指标':<20} {'10步':<20} {'100步':<20} {'500步':<20}")
    print("-"*80)

    # 样本数
    print(f"{'样本数':<20} {results[0]['n_samples']:<20} {results[1]['n_samples']:<20} {results[2]['n_samples']:<20}")
    print()

    # 主要指标
    metrics = [
        ('rmsd', 'Sidechain RMSD (Å)', '.4f'),
        ('ppl_pred', 'Perplexity (pred)', '.3f'),
        ('ppl_gt', 'Perplexity (GT)', '.3f'),
        ('ppl_deg', 'PPL degradation', '.3f'),
        ('rec_pred', 'Recovery (pred)', '.3f'),
        ('rec_gt', 'Recovery (GT)', '.3f'),
        ('rec_deg', 'Recovery degradation', '.3f'),
    ]

    for metric_key, metric_name, fmt in metrics:
        values = []
        for r in results:
            if metric_key in r['stats']:
                mean = r['stats'][metric_key]['mean']
                std = r['stats'][metric_key]['std']
                values.append(f"{mean:{fmt}} ± {std:.4f}")
            else:
                values.append("N/A")

        print(f"{metric_name:<20} {values[0]:<20} {values[1]:<20} {values[2]:<20}")

    # 分布信息
    print("\n" + "="*80)
    print("分布细节（Min / Median / Max）")
    print("="*80)

    for metric_key, metric_name, fmt in metrics[:4]:  # 只看关键指标
        print(f"\n{metric_name}:")
        for r in results:
            if metric_key in r['stats']:
                s = r['stats'][metric_key]
                print(f"  {r['name']:<10}: {s['min']:{fmt}} / {s['median']:{fmt}} / {s['max']:{fmt}}")

    # 关键观察
    print("\n" + "="*80)
    print("关键观察")
    print("="*80)

    rmsd_10 = results[0]['stats']['rmsd']['mean']
    rmsd_100 = results[1]['stats']['rmsd']['mean']
    rmsd_500 = results[2]['stats']['rmsd']['mean']

    print(f"\n1. RMSD变化：")
    print(f"   10步:  {rmsd_10:.4f}Å")
    print(f"   100步: {rmsd_100:.4f}Å  (变化: {(rmsd_100-rmsd_10)/rmsd_10*100:+.1f}%)")
    print(f"   500步: {rmsd_500:.4f}Å  (变化: {(rmsd_500-rmsd_10)/rmsd_10*100:+.1f}%)")

    if abs(rmsd_100 - rmsd_10) < 0.05 and abs(rmsd_500 - rmsd_10) < 0.05:
        print("   ✓ RMSD基本不变，增加步数无明显改善")

    ppl_10 = results[0]['stats']['ppl_pred']['mean']
    ppl_100 = results[1]['stats']['ppl_pred']['mean']
    ppl_500 = results[2]['stats']['ppl_pred']['mean']

    print(f"\n2. Perplexity变化：")
    print(f"   10步:  {ppl_10:.3f}")
    print(f"   100步: {ppl_100:.3f}  (变化: {(ppl_100-ppl_10)/ppl_10*100:+.1f}%)")
    print(f"   500步: {ppl_500:.3f}  (变化: {(ppl_500-ppl_10)/ppl_10*100:+.1f}%)")

    if ppl_100 > ppl_10 and ppl_500 > ppl_10:
        print("   ⚠️  Perplexity反而上升！说明：")
        print("      - 坐标质量（RMSD）没改善")
        print("      - logits受坐标误差影响更大")

    rec_10 = results[0]['stats']['rec_pred']['mean']
    rec_100 = results[1]['stats']['rec_pred']['mean']
    rec_500 = results[2]['stats']['rec_pred']['mean']

    print(f"\n3. Recovery变化：")
    print(f"   10步:  {rec_10:.3f}")
    print(f"   100步: {rec_100:.3f}  (变化: {(rec_100-rec_10)*100:+.1f}%)")
    print(f"   500步: {rec_500:.3f}  (变化: {(rec_500-rec_10)*100:+.1f}%)")

    print("\n" + "="*80)
    print("结论")
    print("="*80)
    print("\n对于R3 FBB模型：")
    print("  1. 增加采样步数（10→100→500）对RMSD几乎无影响")
    print("  2. Perplexity甚至略有上升（坐标没改善，noise accumulation）")
    print("  3. 这说明模型的velocity方向误差是限制因素")
    print("  4. 更多步数 = 在错误方向上走更多次 → 误差饱和")

def main():
    experiments = [
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step10/val_seperated_Rm0_t0_step0_20251116_210156', '10步'),
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step100/val_seperated_Rm0_t0_step0_20251116_210400', '100步'),
        ('/home/junyu/project/pu/outputs/r3fbb_atoms_cords1_step500/val_seperated_Rm0_t0_step0_20251116_211051', '500步'),
    ]

    results = []
    for exp_dir, name in experiments:
        print(f"处理 {name}...")
        result = analyze_experiment(exp_dir, name)
        if result:
            results.append(result)

    if len(results) == 3:
        print_comparison(results)
    else:
        print("⚠️  未能收集到所有实验数据")

if __name__ == '__main__':
    main()
