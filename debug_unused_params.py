"""
诊断DDP中未使用的参数
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from omegaconf import OmegaConf
from models.flow_module import FlowModule
import hydra
from pathlib import Path
from experiments import utils as eu
from data.datasets import PdbDataset

def find_unused_parameters(model, batch):
    """
    找出模型中哪些参数没有被使用
    """
    # 记录哪些参数产生了梯度
    used_params = set()

    def hook_fn(module, grad_input, grad_output):
        for param in module.parameters(recurse=False):
            if param.requires_grad:
                used_params.add(id(param))

    # 给所有模块注册hook
    hooks = []
    for name, module in model.named_modules():
        hook = module.register_backward_hook(hook_fn)
        hooks.append(hook)

    # 前向 + 反向
    model.train()
    model.zero_grad()

    try:
        # 执行一次训练步
        loss_dict = model.training_step(batch, 0)
        if isinstance(loss_dict, dict):
            loss = loss_dict.get('loss', list(loss_dict.values())[0])
        else:
            loss = loss_dict
        loss.backward()
    except Exception as e:
        print(f"训练步骤出错: {e}")
        for hook in hooks:
            hook.remove()
        return None, None

    # 移除hooks
    for hook in hooks:
        hook.remove()

    # 检查哪些参数没被使用
    all_params = {}
    unused_params = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            all_params[name] = param
            if id(param) not in used_params:
                unused_params[name] = param

    return unused_params, all_params


@hydra.main(version_base=None, config_path="configs", config_name="Train_esmsd.yaml")
def main(cfg):
    print("=" * 80)
    print("DDP未使用参数诊断工具")
    print("=" * 80)

    # 创建模型
    print("\n1. 加载模型...")
    model = FlowModule(cfg)
    model.cuda()
    model.train()

    # 创建数据
    print("2. 加载数据...")
    # 添加缺失的seed配置
    if 'seed' not in cfg.data:
        from omegaconf import OmegaConf
        OmegaConf.set_struct(cfg.data, False)
        cfg.data.seed = 123
        OmegaConf.set_struct(cfg.data, True)

    train_dataset, _ = eu.dataset_creation(
        PdbDataset, cfg.data, cfg.experiment.task
    )

    from data.protein_dataloader import LengthBatcher
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=LengthBatcher(
            sampler_cfg=cfg.data.sampler,
            metadata_csv=train_dataset.csv,
            rank=None,
            num_replicas=None,
        ),
        num_workers=0,
        pin_memory=False,
    )

    # 获取一个batch
    print("3. 获取测试batch...")
    batch = next(iter(train_loader))
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # 诊断
    print("4. 执行前向+反向传播，追踪参数使用情况...")
    print("   这可能需要几分钟...\n")

    unused_params, all_params = find_unused_parameters(model, batch)

    if unused_params is None:
        print("❌ 诊断失败，请检查训练步骤是否正常")
        return

    print("=" * 80)
    print("诊断结果")
    print("=" * 80)
    print(f"\n总参数数量: {len(all_params)}")
    print(f"未使用参数数量: {len(unused_params)}")
    print(f"使用率: {100 * (1 - len(unused_params) / len(all_params)):.2f}%\n")

    if len(unused_params) == 0:
        print("✅ 太好了！所有参数都被使用了，DDP应该不会报错。")
    else:
        print("❌ 发现未使用的参数:\n")

        # 按模块分组
        unused_by_module = {}
        for name in unused_params.keys():
            module_name = name.rsplit('.', 1)[0] if '.' in name else 'root'
            if module_name not in unused_by_module:
                unused_by_module[module_name] = []
            unused_by_module[module_name].append(name)

        # 打印分组结果
        for module_name, param_names in sorted(unused_by_module.items()):
            print(f"\n模块: {module_name}")
            print(f"  未使用参数数量: {len(param_names)}")
            for param_name in param_names[:5]:  # 只显示前5个
                param_shape = all_params[param_name].shape
                print(f"    - {param_name}: {param_shape}")
            if len(param_names) > 5:
                print(f"    ... 还有 {len(param_names) - 5} 个参数")

        print("\n" + "=" * 80)
        print("建议的解决方案:")
        print("=" * 80)

        # 分析未使用参数的类型
        if any('_visualize' in name or 'vis_' in name for name in unused_params.keys()):
            print("\n1. 检测到可视化相关参数未使用")
            print("   解决方法: 在IGA初始化时设置 enable_vis=False")

        if any('linear_b' in name or 'down_z' in name for name in unused_params.keys()):
            print("\n2. 检测到Pair特征相关参数未使用 (linear_b, down_z)")
            print("   可能原因: z参数为None或c_z=0")
            print("   解决方法: 确保edge_embed被正确传入IGA")

        if any('Gau_update' in name for name in unused_params.keys()):
            print("\n3. 检测到GaussianUpdateBlock参数未使用")
            print("   可能原因: update_mask全为0，或者某些条件分支没走到")
            print("   解决方法: 检查update_mask的值")

        print("\n通用解决方案:")
        print("  - 方案A: 使用 strategy='ddp_find_unused_parameters_true'")
        print("  - 方案B: 在loss中添加极小的L2正则 (train_loss += 1e-10 * param_reg)")
        print("  - 方案C: 使用单GPU训练 (devices=1)")

if __name__ == "__main__":
    main()
