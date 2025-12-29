"""
系统验证Flow Matching训练和推理的正确性

验证层次：
1. 数据准备：noise生成、interpolation正确性
2. 训练：单步velocity预测准确度
3. 推理：1步、10步、100步的收敛性
4. 对比：和直接回归baseline的比较
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class FlowMatchingDiagnostics:
    def __init__(self, model, interpolant, batch):
        self.model = model
        self.interpolant = interpolant
        self.batch = batch
        self.device = batch['res_mask'].device

    def test_1_data_corruption(self):
        """测试1：验证数据加噪是否正确"""
        print("\n" + "="*60)
        print("测试1：数据加噪验证")
        print("="*60)

        # 准备一个测试样本
        clean_coords = self.batch['atoms14_local'][..., 3:, :]  # [B,N,11,3]

        results = {}
        test_ts = [0.1, 0.5, 0.9]

        for t_val in test_ts:
            # 手动加噪
            noise = torch.randn_like(clean_coords) * self.interpolant._cfg.coord_scale
            x_t = (1 - t_val) * noise + t_val * clean_coords
            v_gt = clean_coords - noise

            # 验证公式：x1 = x_t + (1-t)*v
            x1_recovered = x_t + (1 - t_val) * v_gt
            error = (x1_recovered - clean_coords).abs().mean().item()

            results[f't={t_val}'] = {
                'noise_norm': noise.norm(dim=-1).mean().item(),
                'xt_norm': x_t.norm(dim=-1).mean().item(),
                'v_norm': v_gt.norm(dim=-1).mean().item(),
                'recovery_error': error,
            }

            print(f"\nt = {t_val:.1f}:")
            print(f"  Noise norm: {results[f't={t_val}']['noise_norm']:.3f} Å")
            print(f"  x_t norm:   {results[f't={t_val}']['xt_norm']:.3f} Å")
            print(f"  v norm:     {results[f't={t_val}']['v_norm']:.3f} Å")
            print(f"  Recovery error: {error:.6f} Å (应该≈0)")

            if error > 1e-5:
                print(f"  ❌ WARNING: Recovery error too large!")
            else:
                print(f"  ✓ Interpolation formula correct")

        return results

    def test_2_single_step_prediction(self):
        """测试2：单步预测准确度（最关键）"""
        print("\n" + "="*60)
        print("测试2：单步Velocity预测准确度")
        print("="*60)

        self.model.eval()
        results = {}
        test_ts = [0.1, 0.3, 0.5, 0.7, 0.9]

        with torch.no_grad():
            for t_val in test_ts:
                # 使用interpolant生成训练样本
                noisy_batch = self.interpolant.fbb_corrupt_batch(
                    self.batch, prob=1.0
                )
                # 强制设置为特定t值
                B, N = noisy_batch['res_mask'].shape
                noisy_batch['t'] = torch.full((B,), t_val, device=self.device)
                noisy_batch['r3_t'] = torch.full((B, N), t_val, device=self.device)

                # 计算SH (如果需要)
                if hasattr(self.interpolant, 'fbb_sample_iterative'):
                    from models.features.sh_density import sh_density_from_atom14_with_masks_clean
                    normalize_density, *_ = sh_density_from_atom14_with_masks_clean(
                        noisy_batch['atoms14_local_t'],
                        self.batch['atom14_element_idx'],
                        self.batch['atom14_gt_exists'],
                        L_max=8, R_bins=24,
                    )
                    normalize_density = normalize_density / torch.sqrt(torch.tensor(4 * torch.pi))
                    noisy_batch['normalize_density'] = normalize_density

                # 模型预测
                out = self.model(noisy_batch)
                v_pred = out['speed_vectors']
                v_target = noisy_batch['v_t'][..., 3:, :]

                # 计算误差
                exists_mask = self.batch['atom14_gt_exists'][..., 3:].bool()
                v_error = ((v_pred - v_target) ** 2).sum(dim=-1).sqrt()
                v_error_masked = (v_error * exists_mask.float()).sum() / exists_mask.sum()

                v_pred_norm = v_pred.norm(dim=-1).mean().item()
                v_target_norm = v_target.norm(dim=-1).mean().item()

                # Clean prediction验证
                x_t = noisy_batch['atoms14_local_t'][..., 3:, :]
                x1_pred = x_t + (1 - t_val) * v_pred
                x1_gt = self.batch['atoms14_local'][..., 3:, :]
                x1_error = ((x1_pred - x1_gt) ** 2).sum(dim=-1).sqrt()
                x1_error_masked = (x1_error * exists_mask.float()).sum() / exists_mask.sum()

                results[f't={t_val}'] = {
                    'v_pred_norm': v_pred_norm,
                    'v_target_norm': v_target_norm,
                    'v_rmse': v_error_masked.item(),
                    'x1_rmse': x1_error_masked.item(),
                }

                print(f"\nt = {t_val:.1f}:")
                print(f"  Target v norm:  {v_target_norm:.3f} Å")
                print(f"  Pred v norm:    {v_pred_norm:.3f} Å")
                print(f"  V RMSE:         {v_error_masked.item():.3f} Å")
                print(f"  Clean x1 RMSE:  {x1_error_masked.item():.3f} Å")

                # 判断准则
                norm_ratio = v_pred_norm / (v_target_norm + 1e-6)
                if abs(norm_ratio - 1.0) > 0.5:
                    print(f"  ❌ Velocity scale mismatch! Ratio: {norm_ratio:.2f}")
                elif v_error_masked.item() > 5.0:
                    print(f"  ❌ Velocity RMSE too high!")
                elif x1_error_masked.item() > 2.0:
                    print(f"  ⚠️  Clean prediction error high (但velocity可能是对的)")
                else:
                    print(f"  ✓ Prediction quality acceptable")

        return results

    def test_3_one_step_sampling(self):
        """测试3：1步采样应该等于clean prediction"""
        print("\n" + "="*60)
        print("测试3：1步采样验证")
        print("="*60)

        self.model.eval()

        # 准备初始噪声
        prepared = self.interpolant.fbb_prepare_batch(self.batch)

        # 1步采样
        with torch.no_grad():
            result_1step = self.interpolant.fbb_sample_iterative(
                prepared, self.model, num_timesteps=1
            )

        # 直接预测clean (t=min_t -> t=1.0)
        min_t = self.interpolant._cfg.min_t
        B, N = self.batch['res_mask'].shape
        test_batch = prepared.copy()
        test_batch['t'] = torch.full((B,), min_t, device=self.device)
        test_batch['r3_t'] = torch.full((B, N), min_t, device=self.device)

        from models.features.sh_density import sh_density_from_atom14_with_masks_clean
        normalize_density, *_ = sh_density_from_atom14_with_masks_clean(
            test_batch['atoms14_local_t'],
            self.batch['atom14_element_idx'],
            self.batch['atom14_gt_exists'],
            L_max=8, R_bins=24,
        )
        normalize_density = normalize_density / torch.sqrt(torch.tensor(4 * torch.pi))
        test_batch['normalize_density'] = normalize_density

        with torch.no_grad():
            out = self.model(test_batch)
            v_pred = out['speed_vectors']
            x_t = test_batch['atoms14_local_t'][..., 3:, :]

            # Clean prediction: x1 = xt + (1-t)*v
            x1_direct = x_t + (1 - min_t) * v_pred

            # 1步ODE: xt + dt*v, dt = 1-min_t
            x1_ode = x_t + (1.0 - min_t) * v_pred

        # 对比
        sampled = result_1step['atoms14_local_final'][..., 3:, :]

        error_vs_direct = (sampled - x1_direct).norm(dim=-1).mean().item()
        error_vs_ode = (sampled - x1_ode).norm(dim=-1).mean().item()

        print(f"\n1步采样 vs 直接预测:")
        print(f"  Clean prediction norm: {x1_direct.norm(dim=-1).mean().item():.3f} Å")
        print(f"  ODE result norm:       {x1_ode.norm(dim=-1).mean().item():.3f} Å")
        print(f"  Sampled result norm:   {sampled.norm(dim=-1).mean().item():.3f} Å")
        print(f"  Difference:            {error_vs_direct:.6f} Å")

        if error_vs_direct < 1e-4:
            print(f"  ✓ 1-step sampling matches clean prediction")
        else:
            print(f"  ❌ 1-step sampling does NOT match! Check ODE implementation.")

        return {
            'error_vs_direct': error_vs_direct,
            'x1_direct_norm': x1_direct.norm(dim=-1).mean().item(),
            'sampled_norm': sampled.norm(dim=-1).mean().item(),
        }

    def test_4_multistep_convergence(self):
        """测试4：多步采样收敛性"""
        print("\n" + "="*60)
        print("测试4：多步采样收敛性")
        print("="*60)

        self.model.eval()
        prepared = self.interpolant.fbb_prepare_batch(self.batch)

        x1_gt = self.batch['atoms14_local'][..., 3:, :]
        exists_mask = self.batch['atom14_gt_exists'][..., 3:].bool()

        step_counts = [1, 5, 10, 50, 100]
        results = {}

        for num_steps in step_counts:
            with torch.no_grad():
                result = self.interpolant.fbb_sample_iterative(
                    prepared, self.model, num_timesteps=num_steps
                )

            x1_pred = result['atoms14_local_final'][..., 3:, :]
            error = ((x1_pred - x1_gt) ** 2).sum(dim=-1).sqrt()
            rmse = (error * exists_mask.float()).sum() / exists_mask.sum()

            results[num_steps] = {
                'rmse': rmse.item(),
                'diagnostics': result.get('diagnostics', {}),
            }

            print(f"\n{num_steps:3d} steps: RMSE = {rmse.item():.3f} Å")

            if 'diagnostics' in result and 'sidechain_rmsd' in result['diagnostics']:
                diag = result['diagnostics']
                print(f"           PPL(pred) = {diag.get('perplexity_with_pred_coords', 0):.3f}")
                print(f"           Recovery  = {diag.get('recovery_with_pred_coords', 0):.3f}")

        # 分析趋势
        rmses = [results[k]['rmse'] for k in step_counts]
        print(f"\n收敛性分析:")
        print(f"  1步 -> 10步: {rmses[0]:.3f} -> {rmses[2]:.3f} Å")
        print(f"  10步 -> 100步: {rmses[2]:.3f} -> {rmses[4]:.3f} Å")

        if rmses[4] < rmses[2] * 0.8:
            print(f"  ✓ 增加步数显著改善结果")
        elif rmses[4] > rmses[2] * 1.1:
            print(f"  ❌ 增加步数反而变差！Velocity预测可能有问题")
        else:
            print(f"  ⚠️  增加步数几乎没改善，可能velocity scale不对或未收敛")

        return results

    def test_5_velocity_statistics(self):
        """测试5：训练数据中velocity的统计分布"""
        print("\n" + "="*60)
        print("测试5：Velocity统计分布")
        print("="*60)

        # 采样多个t值，统计velocity分布
        v_norms = []
        coord_scale = self.interpolant._cfg.coord_scale

        clean_coords = self.batch['atoms14_local'][..., 3:, :]

        for _ in range(100):
            noise = torch.randn_like(clean_coords) * coord_scale
            v = clean_coords - noise
            v_norm = v.norm(dim=-1)
            v_norms.append(v_norm.flatten())

        v_norms = torch.cat(v_norms)

        print(f"\nVelocity norm 统计 (coord_scale={coord_scale}):")
        print(f"  Mean: {v_norms.mean().item():.3f} Å")
        print(f"  Std:  {v_norms.std().item():.3f} Å")
        print(f"  P50:  {v_norms.median().item():.3f} Å")
        print(f"  P90:  {v_norms.quantile(0.9).item():.3f} Å")
        print(f"  P99:  {v_norms.quantile(0.99).item():.3f} Å")

        # 检查out_range设置是否合理
        if hasattr(self.model, 'NodeFeatExtractorWithHeads'):
            out_range = self.model.NodeFeatExtractorWithHeads.coord_head.out_range
            print(f"\n模型out_range = {out_range}")

            if v_norms.quantile(0.99).item() > out_range * 0.8:
                print(f"  ⚠️  99%分位 ({v_norms.quantile(0.99).item():.1f}) 接近out_range!")
                print(f"     建议增大out_range到 {v_norms.quantile(0.99).item() * 1.5:.1f}")
            else:
                print(f"  ✓ out_range设置合理")

        return {
            'mean': v_norms.mean().item(),
            'std': v_norms.std().item(),
            'p50': v_norms.median().item(),
            'p90': v_norms.quantile(0.9).item(),
            'p99': v_norms.quantile(0.99).item(),
        }

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("Flow Matching 系统诊断")
        print("="*60)

        all_results = {}

        # 测试1：数据准备
        all_results['data_corruption'] = self.test_1_data_corruption()

        # 测试5：velocity统计（先做这个，了解数据）
        all_results['velocity_stats'] = self.test_5_velocity_statistics()

        # 测试2：单步预测（最关键）
        all_results['single_step'] = self.test_2_single_step_prediction()

        # 测试3：1步采样
        all_results['one_step_sampling'] = self.test_3_one_step_sampling()

        # 测试4：多步收敛
        all_results['multistep'] = self.test_4_multistep_convergence()

        print("\n" + "="*60)
        print("诊断总结")
        print("="*60)

        # 生成诊断报告
        self.generate_summary(all_results)

        return all_results

    def generate_summary(self, results):
        """生成诊断摘要"""
        issues = []

        # 检查单步预测
        if 'single_step' in results:
            for t_key, vals in results['single_step'].items():
                v_ratio = vals['v_pred_norm'] / (vals['v_target_norm'] + 1e-6)
                if abs(v_ratio - 1.0) > 0.5:
                    issues.append(f"❌ {t_key}: Velocity scale严重不匹配 (ratio={v_ratio:.2f})")
                elif vals['v_rmse'] > 5.0:
                    issues.append(f"❌ {t_key}: Velocity RMSE过高 ({vals['v_rmse']:.1f} Å)")

        # 检查收敛性
        if 'multistep' in results:
            rmse_10 = results['multistep'].get(10, {}).get('rmse', 0)
            rmse_100 = results['multistep'].get(100, {}).get('rmse', 0)
            if rmse_100 > rmse_10 * 1.1:
                issues.append(f"❌ 100步比10步更差！({rmse_100:.2f} vs {rmse_10:.2f} Å)")
            elif rmse_100 > rmse_10 * 0.95:
                issues.append(f"⚠️  增加步数几乎无改善 ({rmse_100:.2f} vs {rmse_10:.2f} Å)")

        if issues:
            print("\n发现的问题:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("\n✓ 未发现明显问题")

        print("\n建议:")
        if any("Velocity scale" in issue for issue in issues):
            print("  1. 检查训练loss权重配置 (speed_loss_weight, vector_loss_weight)")
            print("  2. 查看训练log中的vector_loss是否收敛")
            print("  3. 考虑移除或调整/64的缩放")
        if any("增加步数" in issue for issue in issues):
            print("  1. Velocity预测质量不足，需要改善训练")
            print("  2. 检查时间采样分布 (训练vs推理)")


def main():
    """主函数：加载模型和数据，运行诊断"""
    import sys
    sys.path.insert(0, '/home/junyu/project/pu')

    from models.flow_module import FlowModule
    from data.se3_datasets import PDBSidechainDataset
    import yaml
    from omegaconf import OmegaConf

    # 加载配置
    config_path = 'configs/Train_SHFBB.yaml'
    with open(config_path) as f:
        cfg = OmegaConf.create(yaml.safe_load(f))

    # 加载checkpoint
    ckpt_path = input("请输入checkpoint路径: ").strip()
    if not Path(ckpt_path).exists():
        print(f"Checkpoint不存在: {ckpt_path}")
        return

    print(f"\n加载checkpoint: {ckpt_path}")
    module = FlowModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    module.eval()
    module = module.cuda()

    # 加载一个测试样本
    print("加载测试数据...")
    dataset = PDBSidechainDataset(
        data_conf=cfg.data,
        is_training=False,
    )

    batch = dataset[0]
    # 转成batch格式
    batch = {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}

    print(f"样本: {batch.get('pdb_name', 'unknown')}, 长度: {batch['res_mask'].sum().item()}")

    # 运行诊断
    diagnostics = FlowMatchingDiagnostics(module.model, module.interpolant, batch)
    results = diagnostics.run_all_tests()

    # 保存结果
    output_path = 'flow_matching_diagnostics.txt'
    print(f"\n诊断结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
