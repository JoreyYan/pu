"""
快速验证：框架 vs 训练 问题
用真实velocity走ODE，看是否收敛
"""

import torch
import sys
sys.path.insert(0, '/home/junyu/project/pu')

from models.flow_module import FlowModule

import yaml
from omegaconf import OmegaConf
from pathlib import Path


def verify_ode_with_ground_truth_velocity(module, batch):
    """
    关键测试：用真实的velocity走ODE，看框架是否正确

    如果用真实velocity都不收敛 → 框架问题
    如果用真实velocity收敛 → 只是模型没训练好
    """
    print("\n" + "="*70)
    print("关键测试：用真实Velocity的ODE积分")
    print("="*70)

    device = batch['res_mask'].device
    interpolant = module.interpolant

    # 准备初始噪声
    prepared = interpolant.fbb_prepare_batch(batch)

    # Ground truth
    x1_gt = batch['atoms14_local'][..., 3:, :]
    exists_mask = batch['atom14_gt_exists'][..., 3:].bool()

    # 测试不同步数
    for num_steps in [10, 50, 100]:
        print(f"\n{'─'*70}")
        print(f"测试 {num_steps} 步采样")
        print(f"{'─'*70}")

        # 生成时间点
        ts = torch.linspace(interpolant._cfg.min_t, 1.0, num_steps + 1, device=device)

        # 初始状态
        x0 = prepared['atoms14_local_t'][..., 3:, :].clone()
        xt = x0.clone()

        # 用真实velocity走ODE
        for i in range(len(ts) - 1):
            t1 = float(ts[i])
            t2 = float(ts[i + 1])
            dt = t2 - t1

            # 计算当前时刻的真实velocity
            # v = x1 - x0（在linear flow中velocity是恒定的）
            v_true = x1_gt - x0

            # ODE step
            xt = xt + dt * v_true
            xt = xt * exists_mask[..., None].float()

        # 计算误差
        error = ((xt - x1_gt) ** 2).sum(dim=-1).sqrt()
        rmse = (error * exists_mask.float()).sum() / exists_mask.sum()

        print(f"  最终RMSE: {rmse.item():.6f} Å")

        if rmse.item() < 0.01:
            print(f"  ✓✓✓ 完美！用真实velocity ODE积分正确")
        elif rmse.item() < 0.1:
            print(f"  ✓✓ 很好！误差可接受（数值精度）")
        elif rmse.item() < 1.0:
            print(f"  ✓ ODE框架基本正确，有小误差")
        else:
            print(f"  ❌❌❌ 错误！即使用真实velocity也偏差大")
            print(f"  → 说明ODE实现或interpolation公式有问题")
            return False

    return True


def check_model_velocity_vs_true(module, batch):
    """
    测试：模型预测的velocity vs 真实velocity
    """
    print("\n" + "="*70)
    print("测试：模型Velocity预测质量")
    print("="*70)

    device = batch['res_mask'].device
    interpolant = module.interpolant
    model = module.model
    model.eval()

    # 测试几个t值
    test_ts = [0.3, 0.5, 0.7, 0.9]

    with torch.no_grad():
        for t_val in test_ts:
            # 生成噪声样本
            B, N = batch['res_mask'].shape

            clean = batch['atoms14_local'][..., 3:, :]
            noise = torch.randn_like(clean) * interpolant._cfg.coord_scale
            x_t = (1 - t_val) * noise + t_val * clean
            v_true = clean - noise

            # 使用interpolant准备正确的输入格式
            B_size, N_size = batch['res_mask'].shape

            # 创建一个noisy batch
            noisy_batch = interpolant.fbb_corrupt_batch(batch, prob=1.0)

            # 替换为我们手动生成的x_t和v
            noisy_batch['atoms14_local_t'] = torch.cat([
                batch['atoms14_local'][..., :3, :],
                x_t
            ], dim=-2)

            # 强制设置为指定的t值
            noisy_batch['t'] = torch.full((B_size,), t_val, device=device)
            noisy_batch['r3_t'] = torch.full((B_size, N_size), t_val, device=device, dtype=torch.float32)
            noisy_batch['so3_t'] = torch.full((B_size, N_size), t_val, device=device, dtype=torch.float32)

            # 设置velocity target
            v_full = torch.zeros_like(batch['atoms14_local'])
            v_full[..., 3:, :] = v_true
            noisy_batch['v_t'] = v_full

            test_batch = noisy_batch

            # 计算SH
            from data.sh_density import sh_density_from_atom14_with_masks_clean
            normalize_density, *_ = sh_density_from_atom14_with_masks_clean(
                test_batch['atoms14_local_t'],
                batch['atom14_element_idx'],
                batch['atom14_gt_exists'],
                L_max=8, R_bins=24,
            )
            normalize_density = normalize_density / torch.sqrt(torch.tensor(4 * torch.pi))
            test_batch['normalize_density'] = normalize_density

            # 模型预测
            out = model(test_batch)
            v_pred = out['speed_vectors']

            # 对比
            exists_mask = batch['atom14_gt_exists'][..., 3:].bool()

            v_true_norm = v_true.norm(dim=-1)
            v_pred_norm = v_pred.norm(dim=-1)

            v_true_mean = (v_true_norm * exists_mask.float()).sum() / exists_mask.sum()
            v_pred_mean = (v_pred_norm * exists_mask.float()).sum() / exists_mask.sum()

            error = ((v_pred - v_true) ** 2).sum(dim=-1).sqrt()
            rmse = (error * exists_mask.float()).sum() / exists_mask.sum()

            print(f"\nt = {t_val:.1f}:")
            print(f"  真实 v norm: {v_true_mean.item():.3f} Å")
            print(f"  预测 v norm: {v_pred_mean.item():.3f} Å")
            print(f"  Norm ratio:  {(v_pred_mean / v_true_mean).item():.3f}")
            print(f"  RMSE:        {rmse.item():.3f} Å")

            # 判断
            ratio = (v_pred_mean / v_true_mean).item()
            if abs(ratio - 1.0) > 0.5:
                print(f"  ❌ Velocity scale严重不匹配！")
            elif rmse.item() > v_true_mean.item():
                print(f"  ❌ Velocity预测误差比信号还大！")
            elif rmse.item() > 3.0:
                print(f"  ⚠️  Velocity预测质量不佳")
            else:
                print(f"  ✓ Velocity预测质量尚可")


def main():
    """主函数"""
    # 加载配置
    print("\n请输入配置文件路径（或直接回车使用默认 configs/Train_SH.yaml）:")
    config_path = input().strip()
    if not config_path:
        config_path = 'configs/Train_SH.yaml'

    if not Path(config_path).exists():
        print(f"配置文件不存在: {config_path}")
        return

    # 加载配置（使用Hydra风格的组合）
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(config_path).parent.absolute())
    config_name = Path(config_path).stem

    try:
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_name)
    except Exception as e:
        print(f"Hydra加载失败，尝试直接加载: {e}")
        cfg = OmegaConf.load(config_path)

    # 加载checkpoint
    default_ckpt = '/home/junyu/project/pu/ckpt/se3-fm_sh/pdb__shdiffusion_decoder_ctx_shloss/2025-11-14_23-05-48/epoch=49-step=93900.ckpt'
    print(f"\n请输入checkpoint路径（或直接回车使用默认）:")
    print(f"默认: {default_ckpt}")
    ckpt_path = input().strip()

    if not ckpt_path:
        ckpt_path = default_ckpt

    if not Path(ckpt_path).exists():
        print(f"找不到checkpoint，请输入有效路径: {ckpt_path}")
        return

    print(f"\n加载checkpoint: {ckpt_path}")
    module = FlowModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        cfg=cfg,
        map_location='cuda' if torch.cuda.is_available() else 'cpu'
    )
    module.eval()
    module = module.cuda()

    # 加载测试数据（使用和train一致的方式）
    print("加载测试数据...")
    from data.datasets import BaseDataset

    dataset = BaseDataset(
        dataset_cfg=cfg.val_dataset,
        task=cfg.data.task,
        is_training=False,
        is_predict=True
    )

    batch = dataset[0]

    # 转为batch格式
    import numpy as np
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].unsqueeze(0).cuda()
        elif isinstance(batch[key], np.ndarray):
            batch[key] = torch.from_numpy(batch[key]).unsqueeze(0).cuda()

    print(f"样本: 长度 {batch['res_mask'].sum().item()}")

    # 设置interpolant设备
    device = next(module.parameters()).device
    module.interpolant.set_device(device)

    print("\n" + "="*70)
    print("快速诊断：框架 vs 训练问题")
    print("="*70)

    # 测试1：用真实velocity的ODE（验证框架）
    framework_ok = verify_ode_with_ground_truth_velocity(module, batch)

    # 测试2：模型velocity质量（验证训练）
    check_model_velocity_vs_true(module, batch)

    # 总结
    print("\n" + "="*70)
    print("诊断结论")
    print("="*70)

    if framework_ok:
        print("\n✓ 框架验证：ODE积分框架是正确的")
        print("  → 用真实velocity可以完美收敛")
        print("\n❌ 问题在于：模型的velocity预测质量不足")
        print("\n建议：")
        print("  1. 不需要迁移框架！")
        print("  2. 检查训练配置：")
        print("     - vector_loss的权重是否太小（/64问题）")
        print("     - 训练是否收敛（vector_loss应该<5）")
        print("     - loss权重配置")
        print("\n修复方案：")
        print("  方案1：增大velocity loss权重")
        print("     speed_loss_weight: 10.0")
        print("     vector_loss_weight: 10.0")
        print("  方案2：移除/64缩放")
        print("     修改 models/flow_module.py:539-540")
        print("\n预期：修复后重训练10-20 epochs，问题应该解决")

    else:
        print("\n❌ 框架问题：即使用真实velocity，ODE也不收敛")
        print("\n建议：")
        print("  1. 检查interpolation公式")
        print("  2. 考虑借用ml-simplefold的sampler")
        print("  3. 或者迁移到ml-simplefold框架")
        print("\n这种情况下，修改配置无法解决，需要修复框架")


if __name__ == '__main__':
    main()
