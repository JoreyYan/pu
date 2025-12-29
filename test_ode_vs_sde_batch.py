"""
对比 ODE 采样 vs SDE 采样（SimpleFold风格）的侧链RMSD质量
基于 test_1vs10_simple.py 修改，使用相同的初始化方式
"""
import torch
import os
from omegaconf import OmegaConf
from data.datasets import BaseDataset
from models.flow_module import FlowModule
import numpy as np
from tqdm import tqdm

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

def test_ode_vs_sde(cfg_path, ckpt_path, num_samples=50, num_timesteps=10):
    """测试 ODE vs SDE 采样的差异（批量样本）"""

    # 加载配置
    cfg = OmegaConf.load(cfg_path)

    # 加载数据集
    dataset = BaseDataset(
        dataset_cfg=cfg.val_dataset,
        task=cfg.data.task,
        is_training=False,
        is_predict=True
    )

    # 加载模型
    model = FlowModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        cfg=cfg,
        map_location='cuda' if torch.cuda.is_available() else 'cpu'
    )
    model.eval()

    device = next(model.parameters()).device

    # 设置 interpolant 的 num_timesteps
    model.interpolant.num_timesteps = num_timesteps
    model.interpolant.set_device(device)

    print("="*80)
    print(f"测试配置:")
    print(f"  模型检查点: {os.path.basename(ckpt_path)}")
    print(f"  采样步数: {num_timesteps}")
    print(f"  测试样本数: {num_samples}")
    print(f"  设备: {device}")
    print("="*80)

    results_ode = []
    results_sde = []

    # 限制样本数量
    num_samples = min(num_samples, len(dataset))

    for sample_idx in tqdm(range(num_samples), desc="处理样本"):
        # 获取样本
        batch = dataset[sample_idx]

        # 转为batch格式
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].unsqueeze(0)  # 添加batch维度

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # GT坐标
        gt_atoms14_local = batch['atoms14_local']  # [1, N, 14, 3]
        gt_sidechain = gt_atoms14_local[..., 3:, :]  # [1, N, 11, 3]
        side_exists = batch['atom14_gt_exists'][..., 3:]  # [1, N, 11]

        with torch.no_grad():
            # 准备推理batch（每次都重新准备，避免状态污染）
            prepared_ode = model.interpolant.fbb_prepare_batch(batch)

            # 测试 ODE 采样
            sample_out_ode = model.interpolant.fbb_sample_iterative(
                prepared_ode,
                model.model,
                num_timesteps=num_timesteps
            )
            pred_sidechain_ode = sample_out_ode['atoms14_local_final'][..., 3:, :]
            rmsd_ode = compute_sidechain_rmsd(
                pred_sidechain_ode[0],
                gt_sidechain[0],
                side_exists[0]
            )
            diagnostics_ode = sample_out_ode.get('diagnostics', {})
            results_ode.append({
                'rmsd': rmsd_ode,
                'ppl_pred': diagnostics_ode.get('perplexity_with_pred_coords'),
                'recovery_pred': diagnostics_ode.get('recovery_with_pred_coords'),
            })

            # 准备推理batch（SDE）
            prepared_sde = model.interpolant.fbb_prepare_batch(batch)

            # 测试 SDE 采样 (SimpleFold style)
            sample_out_sde = model.interpolant.fbb_sample_iterative_sde(
                prepared_sde,
                model.model,
                num_timesteps=num_timesteps,
                tau=0.3,
                w_cutoff=0.99,
            )
            pred_sidechain_sde = sample_out_sde['atoms14_local_final'][..., 3:, :]
            rmsd_sde = compute_sidechain_rmsd(
                pred_sidechain_sde[0],
                gt_sidechain[0],
                side_exists[0]
            )
            diagnostics_sde = sample_out_sde.get('diagnostics', {})
            results_sde.append({
                'rmsd': rmsd_sde,
                'ppl_pred': diagnostics_sde.get('perplexity_with_pred_coords'),
                'recovery_pred': diagnostics_sde.get('recovery_with_pred_coords'),
            })

    # 统计结果
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)

    # ODE结果
    rmsd_ode_list = [r['rmsd'] for r in results_ode]
    ppl_ode_list = [r['ppl_pred'] for r in results_ode if r['ppl_pred'] is not None]
    recovery_ode_list = [r['recovery_pred'] for r in results_ode if r['recovery_pred'] is not None]

    print(f"\n[ODE 采样 - {num_timesteps}步]")
    print(f"  平均 RMSD: {np.mean(rmsd_ode_list):.4f} ± {np.std(rmsd_ode_list):.4f} Å")
    print(f"  中位 RMSD: {np.median(rmsd_ode_list):.4f} Å")
    if ppl_ode_list:
        print(f"  平均 PPL: {np.mean(ppl_ode_list):.3f}")
        print(f"  平均 Recovery: {np.mean(recovery_ode_list):.3f}")

    # SDE结果
    rmsd_sde_list = [r['rmsd'] for r in results_sde]
    ppl_sde_list = [r['ppl_pred'] for r in results_sde if r['ppl_pred'] is not None]
    recovery_sde_list = [r['recovery_pred'] for r in results_sde if r['recovery_pred'] is not None]

    print(f"\n[SDE 采样 (SimpleFold风格) - {num_timesteps}步]")
    print(f"  平均 RMSD: {np.mean(rmsd_sde_list):.4f} ± {np.std(rmsd_sde_list):.4f} Å")
    print(f"  中位 RMSD: {np.median(rmsd_sde_list):.4f} Å")
    if ppl_sde_list:
        print(f"  平均 PPL: {np.mean(ppl_sde_list):.3f}")
        print(f"  平均 Recovery: {np.mean(recovery_sde_list):.3f}")

    # 对比
    print("\n" + "="*80)
    print("对比分析")
    print("="*80)

    mean_rmsd_ode = np.mean(rmsd_ode_list)
    mean_rmsd_sde = np.mean(rmsd_sde_list)
    diff_rmsd = mean_rmsd_sde - mean_rmsd_ode
    pct_change = (diff_rmsd / mean_rmsd_ode) * 100

    print(f"\nRMSD 差异:")
    print(f"  ODE: {mean_rmsd_ode:.4f} Å")
    print(f"  SDE: {mean_rmsd_sde:.4f} Å")
    print(f"  差值: {diff_rmsd:+.4f} Å ({pct_change:+.1f}%)")

    if diff_rmsd < -0.01:
        print(f"\n✓ SDE 采样的坐标质量更好！")
    elif abs(diff_rmsd) < 0.01:
        print(f"\n≈ 两种方法的坐标质量相近")
    else:
        print(f"\n✓ ODE 采样的坐标质量更好！")

    # 统计显著性 (简单t检验)
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(rmsd_ode_list, rmsd_sde_list)
    print(f"\n配对t检验: t={t_stat:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        print(f"  差异具有统计显著性 (p < 0.05)")
    else:
        print(f"  差异不具有统计显著性 (p >= 0.05)")

    return {
        'ode': results_ode,
        'sde': results_sde,
    }

if __name__ == '__main__':
    cfg_path = '/home/junyu/project/pu/configs/Infer_SH.yaml'
    ckpt_path = '/home/junyu/project/pu/ckpt/se3-fm_sh/pdb__Encoder11atoms_chroma_SNR1_linearBridge/2025-10-16_21-45-09/last.ckpt'

    # 测试不同步数
    for num_steps in [1, 10]:
        print("\n\n")
        print("█"*80)
        print(f"  测试 {num_steps} 步采样")
        print("█"*80)
        results = test_ode_vs_sde(
            cfg_path,
            ckpt_path,
            num_samples=50,
            num_timesteps=num_steps
        )
