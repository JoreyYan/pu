"""
简单调用 interpolant 中的批量测试函数
"""
from omegaconf import OmegaConf
from data.datasets import BaseDataset
from models.flow_module import FlowModule

cfg_path = '/home/junyu/project/pu/configs/Infer_SH.yaml'
ckpt_path = '/home/junyu/project/pu/ckpt/se3-fm_sh/pdb__Encoder11atoms_chroma_SNR1_linearBridge/2025-10-16_21-45-09/last.ckpt'

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
    map_location='cuda'
)
model.eval()

# 设置设备
interpolant = model.interpolant
interpolant.set_device(model.device)

# 测试不同步数
for num_steps in [1, 10]:
    print("\n\n")
    print("█"*80)
    print(f"  测试 {num_steps} 步采样")
    print("█"*80)

    results = interpolant.batch_test_ode_vs_sde(
        dataset=dataset,
        model=model.model,
        num_samples=50,
        num_timesteps=num_steps,
        tau=0.3,
        w_cutoff=0.99,
    )
