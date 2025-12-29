"""
快速计算PDB文件的侧链RMSD
"""
import torch
import numpy as np
from data.all_atom import atom37_to_atom14
from openfold.np.protein import from_pdb_string
from data.residue_constants import restype_order, STANDARD_ATOM_MASK
import sys

def load_pdb_coords(pdb_path):
    """加载PDB并转换为atom14格式"""
    with open(pdb_path, 'r') as f:
        pdb_str = f.read()

    protein = from_pdb_string(pdb_str)

    # 获取aatype
    aatype = np.array([restype_order.get(res, 0) for res in protein.sequence])

    # atom37坐标
    atom37_pos = protein.atom_positions  # [N, 37, 3]
    atom37_mask = protein.atom_mask  # [N, 37]

    # 转换为atom14
    # 需要residx_atom37_to_atom14映射
    from data.all_atom import make_new_atom14_resid
    residx_atom37_to_atom14 = make_new_atom14_resid(torch.from_numpy(aatype)).numpy()

    # 手动转换atom37 -> atom14
    N, _, _ = atom37_pos.shape
    atom14_pos = np.zeros((N, 14, 3))
    atom14_mask = np.zeros((N, 14))

    for i in range(N):
        for j in range(14):
            atom37_idx = residx_atom37_to_atom14[i, j]
            if atom37_idx < 37:
                atom14_pos[i, j] = atom37_pos[i, atom37_idx]
                atom14_mask[i, j] = atom37_mask[i, atom37_idx]

    return atom14_pos, atom14_mask, aatype

def compute_sidechain_rmsd(pred_pdb, gt_pdb):
    """计算侧链RMSD"""
    pred_atom14, pred_mask, pred_aa = load_pdb_coords(pred_pdb)
    gt_atom14, gt_mask, gt_aa = load_pdb_coords(gt_pdb)

    # 只看侧链 (index 4-13, 跳过前3个backbone + CA)
    pred_sc = pred_atom14[:, 4:, :]  # [N, 10, 3]
    gt_sc = gt_atom14[:, 4:, :]
    mask_sc = (pred_mask[:, 4:] * gt_mask[:, 4:]).astype(bool)

    # 计算RMSD
    diff = (pred_sc - gt_sc) ** 2  # [N, 10, 3]
    diff = diff.sum(axis=-1)  # [N, 10]

    rmsd_per_atom = np.sqrt(diff)

    # 只计算有效原子
    valid_count = mask_sc.sum()
    if valid_count > 0:
        mean_rmsd = (rmsd_per_atom * mask_sc).sum() / valid_count
    else:
        mean_rmsd = 0.0

    return mean_rmsd, rmsd_per_atom, mask_sc

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python compute_rmsd.py <predicted.pdb> <ground_truth.pdb>")
        sys.exit(1)

    pred_pdb = sys.argv[1]
    gt_pdb = sys.argv[2]

    rmsd, _, _ = compute_sidechain_rmsd(pred_pdb, gt_pdb)
    print(f"Sidechain RMSD: {rmsd:.4f} Å")
