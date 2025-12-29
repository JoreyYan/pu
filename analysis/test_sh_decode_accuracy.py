"""
测试SH密度decode精度

实验设计：
1. 从GT结构生成SH密度
2. 用训练好的decoder decode回atom14
3. 评估decode精度：
   - 整体RMSD (所有原子)
   - 按残基类型分析 (GLY, 芳香环, 带电残基等)
   - 按原子类型分析 (主链 vs 侧链)
   - 几何质量 (键长、键角、芳香环平面性)
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

sys.path.append('/home/junyu/project/pu')
from data.sh_density import sh_density_from_atom14_with_masks
from models.shattetnion.ShDecoderSidechain import SHGeoResHead
from data import residue_constants
from data import utils as du

print("=" * 80)
print("SH密度decode精度测试")
print("=" * 80)

# ============================================================================
# 配置
# ============================================================================

# 数据
test_data_dir = '/home/junyu/project/casp15/targets/casp15.targets.TS-domains.public_12.20.2022'
output_dir = '/home/junyu/project/pu/outputs/sh_decode_accuracy'
os.makedirs(output_dir, exist_ok=True)

# SH参数
L_MAX = 8
R_MAX = 8
R_BINS = 16
C = 4  # C, N, O, S

# 模型路径
model_path = '/home/junyu/project/pu/outputs/sh_decoder_checkpoint.pt'  # 你需要提供实际路径
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# 加载模型
# ============================================================================

print("\n加载SH decoder模型...")
if os.path.exists(model_path):
    sh_decoder = SHGeoResHead(C=C, L_max=L_MAX, R_bins=R_BINS, hidden=256)
    checkpoint = torch.load(model_path, map_location=device)
    sh_decoder.load_state_dict(checkpoint['model'])
    sh_decoder.eval()
    sh_decoder.to(device)
    print(f"✓ 模型加载成功: {model_path}")
else:
    print(f"⚠️ 模型文件不存在: {model_path}")
    print("请提供训练好的SH decoder模型路径，或者我们测试一下未训练的模型作为baseline")
    sh_decoder = None

# ============================================================================
# 辅助函数
# ============================================================================

def compute_rmsd(pred, gt, mask):
    """计算RMSD (只考虑mask为True的原子)"""
    if mask.sum() == 0:
        return np.nan
    diff = (pred - gt) ** 2  # [N, 14, 3]
    diff = diff * mask.unsqueeze(-1)  # 应用mask
    msd = diff.sum() / mask.sum()
    return torch.sqrt(msd).item()

def compute_bond_length_error(atoms14, mask, residue_type):
    """计算键长误差"""
    # 这里简化：只检查CA-CB键长
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']

    has_cb = mask[:, cb_idx].bool()
    if has_cb.sum() == 0:
        return np.nan

    ca_pos = atoms14[has_cb, ca_idx]  # [M, 3]
    cb_pos = atoms14[has_cb, cb_idx]  # [M, 3]

    bond_lengths = torch.norm(cb_pos - ca_pos, dim=-1)  # [M]
    ideal_length = 1.54  # CA-CB标准键长

    error = torch.abs(bond_lengths - ideal_length).mean().item()
    return error

def analyze_by_residue_type(results_df):
    """按残基类型分析"""
    # 分组
    aromatic = ['PHE', 'TYR', 'TRP', 'HIS']
    charged = ['ARG', 'LYS', 'ASP', 'GLU']
    polar = ['SER', 'THR', 'ASN', 'GLN']
    hydrophobic = ['ALA', 'VAL', 'LEU', 'ILE', 'MET']
    special = ['GLY', 'PRO', 'CYS']

    summary = []
    for group_name, residues in [
        ('芳香环', aromatic),
        ('带电', charged),
        ('极性', polar),
        ('疏水', hydrophobic),
        ('特殊', special)
    ]:
        group_data = results_df[results_df['residue_type'].isin(residues)]
        if len(group_data) > 0:
            summary.append({
                'group': group_name,
                'count': len(group_data),
                'mean_rmsd': group_data['rmsd'].mean(),
                'std_rmsd': group_data['rmsd'].std(),
                'max_rmsd': group_data['rmsd'].max(),
            })

    return pd.DataFrame(summary)

# ============================================================================
# 主测试流程
# ============================================================================

print("\n" + "=" * 80)
print("开始测试")
print("=" * 80)

# 为了快速测试，我们先用一个简单的例子
# 你需要提供实际的数据加载函数
print("\n⚠️ 需要你提供PDB数据加载函数")
print("当前脚本使用占位符，你需要替换为实际的数据加载代码\n")

results = []

# 示例：假设你有一个函数可以加载PDB数据
# from your_data_loader import load_pdb_features
#
# pdb_files = sorted(Path(test_data_dir).glob('*.pdb'))[:10]
#
# for pdb_file in tqdm(pdb_files, desc="处理PDB文件"):
#     try:
#         # 加载GT数据
#         features = load_pdb_features(str(pdb_file))
#
#         aatype = features['aatype']  # [N]
#         atom14_gt = features['atom14_positions']  # [N, 14, 3]
#         atom14_mask = features['atom14_mask']  # [N, 14]
#         atom14_element = features['atom14_element_idx']  # [N, 14]
#
#         # 1. 计算SH密度
#         sh_density, density_mask, _, _, _ = sh_density_from_atom14_with_masks(
#             atom14_gt.unsqueeze(0),
#             atom14_element.unsqueeze(0),
#             atom14_mask.unsqueeze(0),
#             L_max=L_MAX,
#             r_max=R_MAX,
#             per_atom_norm=False
#         )
#         sh_density = sh_density / torch.sqrt(torch.tensor(4 * torch.pi))
#
#         # 2. Decode回atom14
#         if sh_decoder is not None:
#             with torch.no_grad():
#                 sh_input = sh_density.to(device)
#                 node_mask = torch.ones(1, sh_density.shape[1]).to(device)
#
#                 logits, atom14_recon = sh_decoder(sh_input, node_mask=node_mask)
#                 atom14_recon = atom14_recon.cpu()
#         else:
#             print("跳过：没有模型")
#             continue
#
#         # 3. 评估精度
#         # 3.1 整体RMSD
#         overall_rmsd = compute_rmsd(
#             atom14_recon.squeeze(0),
#             atom14_gt,
#             atom14_mask
#         )
#
#         # 3.2 主链RMSD (N, CA, C, O)
#         backbone_indices = [0, 1, 2, 3]  # N, CA, C, O
#         backbone_mask = atom14_mask.clone()
#         backbone_mask[:, 4:] = 0  # 只保留前4个原子
#         backbone_rmsd = compute_rmsd(
#             atom14_recon.squeeze(0),
#             atom14_gt,
#             backbone_mask
#         )
#
#         # 3.3 侧链RMSD
#         sidechain_mask = atom14_mask.clone()
#         sidechain_mask[:, :4] = 0  # 去掉主链
#         sidechain_rmsd = compute_rmsd(
#             atom14_recon.squeeze(0),
#             atom14_gt,
#             sidechain_mask
#         )
#
#         # 3.4 按残基评估
#         for i, aa in enumerate(aatype):
#             aa_name = residue_constants.restypes[aa]
#             res_mask = atom14_mask[i]
#
#             if res_mask.sum() == 0:
#                 continue
#
#             res_rmsd = compute_rmsd(
#                 atom14_recon.squeeze(0)[i:i+1],
#                 atom14_gt[i:i+1],
#                 res_mask.unsqueeze(0)
#             )
#
#             results.append({
#                 'pdb_name': pdb_file.stem,
#                 'residue_idx': i,
#                 'residue_type': aa_name,
#                 'rmsd': res_rmsd,
#                 'num_atoms': res_mask.sum().item(),
#             })
#
#         print(f"\n{pdb_file.stem}:")
#         print(f"  Overall RMSD: {overall_rmsd:.4f} Å")
#         print(f"  Backbone RMSD: {backbone_rmsd:.4f} Å")
#         print(f"  Sidechain RMSD: {sidechain_rmsd:.4f} Å")
#
#     except Exception as e:
#         print(f"处理失败 {pdb_file.stem}: {e}")
#         continue

# ============================================================================
# 分析结果
# ============================================================================

if len(results) > 0:
    df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("整体统计")
    print("=" * 80)
    print(f"总残基数: {len(df)}")
    print(f"平均RMSD: {df['rmsd'].mean():.4f} Å")
    print(f"标准差: {df['rmsd'].std():.4f} Å")
    print(f"中位数: {df['rmsd'].median():.4f} Å")
    print(f"最大值: {df['rmsd'].max():.4f} Å")
    print(f"90分位: {df['rmsd'].quantile(0.9):.4f} Å")

    # 按残基类型分析
    print("\n" + "=" * 80)
    print("按残基类型分析")
    print("=" * 80)
    by_type = analyze_by_residue_type(df)
    print(by_type.to_string(index=False))

    # 保存结果
    df.to_csv(f"{output_dir}/per_residue_rmsd.csv", index=False)
    by_type.to_csv(f"{output_dir}/by_residue_type.csv", index=False)

    print(f"\n✓ 结果保存到: {output_dir}/")
else:
    print("\n⚠️ 没有结果数据")

print("\n" + "=" * 80)
print("需要的下一步")
print("=" * 80)
print("1. 提供PDB数据加载函数")
print("2. 提供训练好的SH decoder模型路径")
print("3. 运行完整评估")
print("\n你有这些吗？我可以帮你集成")
