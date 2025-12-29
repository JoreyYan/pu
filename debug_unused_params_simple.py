"""
简化版：诊断DDP中未使用的参数
直接从训练脚本复制数据加载逻辑
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from experiments.train import Experiment
import hydra

def find_unused_parameters(model, dataloader):
    """
    找出模型中哪些参数没有被使用
    """
    print("   正在执行前向+反向传播...")

    # 记录哪些参数产生了梯度
    used_params = set()

    def hook_fn(grad):
        # 这个hook会在反向传播时被调用
        return grad

    # 给所有参数注册hook
    hooks = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            hook = param.register_hook(lambda grad, n=name: (used_params.add(n), grad)[1])
            hooks.append(hook)

    # 获取一个batch
    batch = next(iter(dataloader))

    # 前向 + 反向
    model.train()
    model.zero_grad()

    try:
        # 执行一次训练步
        loss = model.training_step(batch, 0)
        if torch.is_tensor(loss):
            loss.backward()
        else:
            # loss可能是字典
            print(f"   Loss类型: {type(loss)}")
            return None, None
    except Exception as e:
        print(f"❌ 训练步骤出错: {e}")
        import traceback
        traceback.print_exc()
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
            if name not in used_params:
                unused_params[name] = param

    return unused_params, all_params


@hydra.main(version_base=None, config_path="configs", config_name="Train_esmsd.yaml")
def main(cfg):
    print("=" * 80)
    print("DDP未使用参数诊断工具 (简化版)")
    print("=" * 80)

    # 使用Experiment类加载所有东西
    print("\n1. 初始化训练实验...")
    exp = Experiment(cfg)

    print("2. 设置数据集...")
    exp._setup_dataset()

    print("3. 创建模型...")
    exp._create_module()

    model = exp._module
    model.cuda()

    print("4. 获取训练数据...")
    train_loader = exp._datamodule.train_dataloader()

    # 诊断
    print("5. 执行前向+反向传播，追踪参数使用情况...")
    print("   这可能需要1-2分钟...\n")

    unused_params, all_params = find_unused_parameters(model, train_loader)

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
        has_vis = any('vis' in name.lower() for name in unused_params.keys())
        has_pair = any('linear_b' in name or 'down_z' in name for name in unused_params.keys())
        has_gau = any('Gau_update' in name for name in unused_params.keys())

        if has_vis:
            print("\n1. ⚠️  检测到可视化相关参数未使用")
            print("   解决方法: 在IGA初始化时设置 enable_vis=False")
            print("   (已在 flow_model.py:626 行添加)")

        if has_pair:
            print("\n2. ⚠️  检测到Pair特征相关参数未使用 (linear_b, down_z)")
            print("   可能原因: z参数为None或c_z=0")
            print("   解决方法: 确保edge_embed被正确传入IGA")

        if has_gau:
            print("\n3. ⚠️  检测到GaussianUpdateBlock参数未使用")
            print("   可能原因: update_mask全为0，或者某些条件分支没走到")
            print("   解决方法: 检查update_mask的值")

        print("\n通用解决方案:")
        print("  - 方案A: 使用 strategy='ddp_find_unused_parameters_true' (会变慢)")
        print("  - 方案B: 在loss中添加极小的L2正则")
        print("  - 方案C: 使用单GPU训练 (devices=1)")

if __name__ == "__main__":
    main()
