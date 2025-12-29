"""
最简单的DDP测试：直接测试模型在训练模式下的参数使用
"""
import torch
import torch.nn as nn

# 测试当前IGA模型是否有未使用参数
def test_model_ddp():
    print("=" * 80)
    print("DDP 参数使用测试")
    print("=" * 80)

    # 加载真实的模型
    import sys
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)

    from omegaconf import OmegaConf
    from models.flow_module import FlowModule

    # 加载配置
    print("\n1. 加载配置...")
    cfg = OmegaConf.load(os.path.join(script_dir, 'configs/Train_esmsd.yaml'))

    # 加载其他必需的配置文件
    base_cfg = OmegaConf.load(os.path.join(script_dir, 'configs/SH.yaml'))
    datasets_cfg = OmegaConf.load(os.path.join(script_dir, 'configs/datasets.yaml'))

    # 合并配置
    cfg = OmegaConf.merge(base_cfg, datasets_cfg, cfg)

    print("2. 创建模型...")
    model = FlowModule(cfg)
    model.cuda()

    # 创建假数据来测试
    print("3. 创建测试数据...")
    B, N = 2, 100
    device = 'cuda'

    batch = {
        'res_mask': torch.ones(B, N, device=device),
        'res_idx': torch.arange(N, device=device).unsqueeze(0).expand(B, -1),
        'chain_idx': torch.zeros(B, N, device=device, dtype=torch.long),
        'aatype': torch.randint(0, 20, (B, N), device=device),
        'diffuse_mask': torch.ones(B, N, device=device),
        'update_mask': torch.ones(B, N, device=device),
        'atoms14_local': torch.randn(B, N, 14, 3, device=device),
        'atom14_gt_exists': torch.ones(B, N, 14, device=device),
        'atom14_gt_positions': torch.randn(B, N, 14, 3, device=device),
        'rotmats_1': torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1),
        'trans_1': torch.zeros(B, N, 3, device=device),
        'sidechain_atom_mask': torch.ones(B, N, 10, device=device),
        'local_mean_1': torch.zeros(B, N, 3, device=device),
        'scaing_log_1': torch.zeros(B, N, 3, device=device),
    }

    print("4. 追踪参数使用情况...")
    model.train()

    # 记录哪些参数收到了梯度
    params_with_grad = set()

    def track_grad(name):
        def hook(grad):
            params_with_grad.add(name)
            return grad
        return hook

    # 注册hooks
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(track_grad(name))

    print("5. 执行前向+反向...")
    try:
        model.zero_grad()
        loss = model.training_step(batch, 0)

        if isinstance(loss, dict):
            print(f"   ❌ Loss是字典，无法直接backward")
            return

        loss.backward()
        print("   ✓ 反向传播成功")

    except Exception as e:
        print(f"   ❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 统计结果
    print("\n" + "=" * 80)
    print("结果")
    print("=" * 80)

    all_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    unused = {name: p for name, p in all_params.items() if name not in params_with_grad}

    print(f"\n总参数: {len(all_params)}")
    print(f"已使用: {len(params_with_grad)}")
    print(f"未使用: {len(unused)}")
    print(f"使用率: {100 * len(params_with_grad) / len(all_params):.1f}%")

    if unused:
        print("\n❌ 未使用的参数:")

        # 按模块分组
        by_module = {}
        for name in unused:
            module = '.'.join(name.split('.')[:-1])
            if module not in by_module:
                by_module[module] = []
            by_module[module].append(name)

        for module, params in sorted(by_module.items()):
            print(f"\n  {module}:")
            for p in params[:3]:
                print(f"    - {p.split('.')[-1]}")
            if len(params) > 3:
                print(f"    ... 还有 {len(params)-3} 个")
    else:
        print("\n✅ 所有参数都被使用了！")

if __name__ == '__main__':
    test_model_ddp()
