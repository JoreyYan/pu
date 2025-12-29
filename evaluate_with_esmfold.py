"""
用ESMFold评估不同采样方法生成的序列质量

流程:
1. 从各个方法的输出目录中提取预测序列
2. 用ESMFold折叠这些序列
3. 将折叠结果与原始GT PDB比较（scTM, scRMSD等指标）
"""
import os
import sys
import torch
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np

# 添加ESM路径
sys.path.insert(0, '/home/junyu/project/esm')
sys.path.insert(0, '/home/junyu/project/esm/genie/evaluations/pipeline')

import esm
from fold_models.esmfold import ESMFold


def extract_predicted_sequences(method_dirs):
    """
    从各个方法的输出目录中提取预测序列

    Args:
        method_dirs: dict, {"method_name": "dir_path"}

    Returns:
        dict: {"method_name": [(sample_id, gt_seq, pred_seq, pdb_path), ...]}
    """
    sequences = {}

    for method_name, dir_path in method_dirs.items():
        sequences[method_name] = []

        # 遍历所有样本目录
        for sample_dir in sorted(Path(dir_path).glob('sample_*')):
            fasta_file = sample_dir / 'sequence.fasta'
            pdb_file = sample_dir / 'predicted.pdb'

            if not fasta_file.exists() or not pdb_file.exists():
                continue

            # 读取FASTA文件
            with open(fasta_file, 'r') as f:
                lines = f.readlines()

            # 提取GT序列和预测序列
            gt_seq = None
            pred_seq = None
            sample_id = sample_dir.name

            for i, line in enumerate(lines):
                if line.startswith('>original'):
                    gt_seq = lines[i+1].strip()
                elif line.startswith('>predicted'):
                    pred_seq = lines[i+1].strip()

            if gt_seq and pred_seq:
                sequences[method_name].append({
                    'sample_id': sample_id,
                    'gt_seq': gt_seq,
                    'pred_seq': pred_seq,
                    'pdb_path': str(pdb_file)
                })

        print(f"[{method_name}] 提取了 {len(sequences[method_name])} 个样本的序列")

    return sequences


def fold_with_esmfold(sequences, output_dir):
    """
    用ESMFold折叠序列

    Args:
        sequences: dict from extract_predicted_sequences
        output_dir: 输出目录

    Returns:
        dict: {"method_name": [(sample_id, esmfold_pdb, gt_pdb_path), ...]}
    """
    # 初始化ESMFold模型
    print("初始化ESMFold模型...")
    model = ESMFold()

    results = {}

    for method_name, seq_list in sequences.items():
        method_output_dir = Path(output_dir) / method_name
        method_output_dir.mkdir(parents=True, exist_ok=True)

        results[method_name] = []

        print(f"\n{'='*80}")
        print(f"正在处理方法: {method_name}")
        print(f"{'='*80}")

        for item in tqdm(seq_list, desc=f"折叠 {method_name}"):
            sample_id = item['sample_id']
            pred_seq = item['pred_seq']
            gt_pdb = item['pdb_path']

            try:
                # 用ESMFold生成结构
                pdb_str, pae = model.predict(pred_seq)

                # 保存结构
                output_pdb = method_output_dir / f"{sample_id}_esmfold.pdb"
                with open(output_pdb, 'w') as f:
                    f.write(pdb_str)

                results[method_name].append({
                    'sample_id': sample_id,
                    'esmfold_pdb': str(output_pdb),
                    'gt_pdb': gt_pdb,
                    'pred_seq': pred_seq,
                    'pae_shape': pae.shape if pae is not None else None,
                })

            except Exception as e:
                print(f"  错误: {sample_id} 折叠失败 - {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"[{method_name}] 成功折叠 {len(results[method_name])} 个样本")

    return results


def compute_sidechain_metrics(esmfold_pdb, gt_pdb):
    """
    计算侧链指标: scTM, scRMSD

    暂时返回占位值，后续可以添加实际计算

    Args:
        esmfold_pdb: ESMFold生成的PDB路径
        gt_pdb: Ground truth PDB路径

    Returns:
        dict: {'success': bool}
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(esmfold_pdb) or not os.path.exists(gt_pdb):
            return {'success': False}

        # 检查PDB文件大小
        if os.path.getsize(esmfold_pdb) == 0:
            return {'success': False}

        return {'success': True}

    except Exception as e:
        print(f"  检查文件失败: {e}")
        return {'success': False}


def evaluate_folding_results(folding_results):
    """
    评估ESMFold折叠结果

    Args:
        folding_results: dict from fold_with_esmfold

    Returns:
        dict: {"method_name": {"success_count": int, "total_count": int}}
    """
    evaluation = {}

    for method_name, result_list in folding_results.items():
        evaluation[method_name] = {
            'success_count': 0,
            'total_count': len(result_list),
            'success_samples': [],
        }

        print(f"\n{'='*80}")
        print(f"评估方法: {method_name}")
        print(f"{'='*80}")

        for item in tqdm(result_list, desc=f"评估 {method_name}"):
            metrics = compute_sidechain_metrics(
                item['esmfold_pdb'],
                item['gt_pdb']
            )

            if metrics['success']:
                evaluation[method_name]['success_count'] += 1
                evaluation[method_name]['success_samples'].append(item['sample_id'])

    return evaluation


def print_summary(evaluation):
    """打印评估结果摘要"""
    print("\n" + "="*80)
    print("ESMFold 评估结果汇总")
    print("="*80)

    for method_name, metrics in evaluation.items():
        total = metrics['total_count']
        success = metrics['success_count']

        if total == 0:
            print(f"\n[{method_name}] 没有样本")
            continue

        success_rate = success / total * 100
        print(f"\n[{method_name}]")
        print(f"  总样本数: {total}")
        print(f"  成功折叠: {success} ({success_rate:.1f}%)")
        print(f"  失败折叠: {total - success}")


def main():
    # 定义各个方法的目录
    method_dirs = {
        'SDE': '/home/junyu/project/pu/outputs/predict_step_sde/sde',
        'ODE_1step_new': '/home/junyu/project/pu/outputs/inference_1step_fixed',
        'ODE_10step': '/home/junyu/project/pu/outputs/inference_10new_with_diagnostics',
        'ODE_1step_old': '/home/junyu/project/pu/outputs/inference',
    }

    output_dir = '/home/junyu/project/pu/outputs/esmfold_evaluation'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 提取序列
    print("="*80)
    print("步骤 1: 提取预测序列")
    print("="*80)
    sequences = extract_predicted_sequences(method_dirs)

    # 保存序列到JSON
    sequences_json = Path(output_dir) / 'extracted_sequences.json'
    with open(sequences_json, 'w') as f:
        json.dump(sequences, f, indent=2)
    print(f"\n序列已保存到: {sequences_json}")

    # 2. ESMFold折叠
    print("\n" + "="*80)
    print("步骤 2: 用ESMFold折叠序列")
    print("="*80)
    folding_results = fold_with_esmfold(sequences, output_dir)

    # 保存折叠结果
    folding_json = Path(output_dir) / 'folding_results.json'
    with open(folding_json, 'w') as f:
        json.dump(folding_results, f, indent=2)
    print(f"\n折叠结果已保存到: {folding_json}")

    # 3. 评估
    print("\n" + "="*80)
    print("步骤 3: 评估折叠质量")
    print("="*80)
    evaluation = evaluate_folding_results(folding_results)

    # 保存评估结果
    evaluation_json = Path(output_dir) / 'evaluation_results.json'
    with open(evaluation_json, 'w') as f:
        json.dump(evaluation, f, indent=2)
    print(f"\n评估结果已保存到: {evaluation_json}")

    # 4. 打印摘要
    print_summary(evaluation)


if __name__ == '__main__':
    main()
