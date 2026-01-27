"""
简单对比单步vs多步的坐标质量
"""
import torch
import os
from omegaconf import OmegaConf
from data.datasets import BaseDataset
from models.flow_module import FlowModule
import numpy as np

def compute_sidechain_rmsd(pred_atoms, gt_atoms, exists_mask):
    """
    计算侧链RMSD
    pred_atoms: [N, 11, 3] 预测的侧链坐标
    gt_atoms: [N, 11, 3] GT侧链坐标
    exists_mask: [N, 11] 原子存在mask
    """
    diff = (pred_atoms - gt_atoms) ** 2  # [N, 11, 3]
    diff = diff.sum(dim=-1)  # [N, 11]
    rmsd_per_atom = torch.sqrt(diff + 1e-8)

    mask = exists_mask.float()
    num_atoms = mask.sum()
    if num_atoms > 0:
        mean_rmsd = (rmsd_per_atom * mask).sum() / num_atoms
    else:
        mean_rmsd = torch.tensor(0.0)

    return mean_rmsd.item()

def test_sample(cfg_path, ckpt_path, sample_idx=0):
    """测试单个样本的单步vs多步"""

    # 加载配置
    cfg = OmegaConf.load(cfg_path)

    # 加载数据集
    dataset = BaseDataset(
        dataset_cfg=cfg.val_dataset,
        task=cfg.data.task,
        is_training=False,
        is_predict=True
    )

    # 获取一个样本
    batch = dataset[sample_idx]

    # 转为batch格式
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].unsqueeze(0)  # 添加batch维度

    # 加载模型
    model = FlowModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        cfg=cfg,
        map_location='cuda' if torch.cuda.is_available() else 'cpu'
    )
    model.eval()

    device = next(model.parameters()).device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    # GT坐标
    gt_atoms14_local = batch['atoms14_local']  # [1, N, 14, 3]
    gt_sidechain = gt_atoms14_local[..., 3:, :]  # [1, N, 11, 3]
    side_exists = batch['atom14_gt_exists'][..., 3:]  # [1, N, 11]

    print("="*60)
    print(f"Testing sample {sample_idx}")
    print(f"Sequence length: {batch['res_mask'].sum().item()}")
    print("="*60)

    # 准备推理batch
    interpolant = model.interpolant
    interpolant.set_device(device)

    results = {}

    # 测试不同步数
    for num_steps in [1, 10]:
        print(f"\n{'='*60}")
        print(f"Testing {num_steps}-step inference")
        print(f"{'='*60}")

        with torch.no_grad():
            prepared = interpolant.fbb_prepare_batch(batch)

            # 运行推理
            sample_out = interpolant.fbb_sample_iterative(
                prepared,
                model.model,
                num_timesteps=num_steps
            )

            # 提取预测的侧链坐标
            pred_atoms14_local = sample_out['atoms14_local_final']  # [1, N, 14, 3]
            pred_sidechain = pred_atoms14_local[..., 3:, :]  # [1, N, 11, 3]

            # 计算RMSD
            rmsd = compute_sidechain_rmsd(
                pred_sidechain[0],  # 去掉batch维度
                gt_sidechain[0],
                side_exists[0]
            )

            # 获取诊断信息
            diagnostics = sample_out.get('diagnostics', {})

            results[num_steps] = {
                'rmsd': rmsd,
                'ppl_pred': diagnostics.get('perplexity_with_pred_coords', None),
                'ppl_gt': diagnostics.get('perplexity_with_gt_coords', None),
                'recovery_pred': diagnostics.get('recovery_with_pred_coords', None),
                'recovery_gt': diagnostics.get('recovery_with_gt_coords', None),
            }

            print(f"\n结果:")
            print(f"  侧链RMSD: {rmsd:.4f} Å")
            if results[num_steps]['ppl_pred']:
                print(f"  Perplexity (pred coords): {results[num_steps]['ppl_pred']:.3f}")
                print(f"  Perplexity (GT coords):   {results[num_steps]['ppl_gt']:.3f}")
                print(f"  Recovery (pred coords):   {results[num_steps]['recovery_pred']:.3f}")
                print(f"  Recovery (GT coords):     {results[num_steps]['recovery_gt']:.3f}")

    # 对比
    print(f"\n{'='*60}")
    print("最终对比")
    print(f"{'='*60}")

    rmsd_1 = results[1]['rmsd']
    rmsd_10 = results[10]['rmsd']

    print(f"\n坐标质量 (RMSD越小越好):")
    print(f"  1步推理:  {rmsd_1:.4f} Å")
    print(f"  10步推理: {rmsd_10:.4f} Å")
    print(f"  差异:     {rmsd_10 - rmsd_1:.4f} Å ({((rmsd_10/rmsd_1 - 1)*100):.1f}%)")

    if rmsd_1 < rmsd_10:
        print(f"\n✓ 单步推理的坐标质量更好！")
    else:
        print(f"\n✓ 多步推理的坐标质量更好！")

    if results[1]['ppl_pred']:
        ppl_1 = results[1]['ppl_pred']
        ppl_10 = results[10]['ppl_pred']
        print(f"\nPerplexity (越小越好):")
        print(f"  1步推理:  {ppl_1:.3f}")
        print(f"  10步推理: {ppl_10:.3f}")
        print(f"  差异:     {ppl_10 - ppl_1:.3f} ({((ppl_10/ppl_1 - 1)*100):.1f}%)")

if __name__ == '__main__':
    cfg_path = '/home/junyu/project/pu/configs/Infer_SH.yaml'
    ckpt_path = '/home/junyu/project/pu/ckpt/se3-fm_sh/pdb__Encoder11atoms_chroma_SNR1_linearBridge/2025-10-16_21-45-09/last.ckpt'

    test_sample(cfg_path, ckpt_path, sample_idx=34)  # T1104-D1
