"""Dynamic-K residue-aware side-chain decoder for SH densities.

Usage
-----
- Give SH coefficients [B,N,C,L+1,2L+1,R], local frames Rmats [B,N,3,3], translations tpos [B,N,3].
- Provide either `aatype` (long, AlphaFold 0..19 index) or `aatype_probs` ([B,N,20]).
- We build a per-(B,N,C) K map from residue priors, then run density->peaks decoding
  with mask & optional score threshold.

Outputs include coords_local/global [B,N,C,K,3], scores [B,N,C,K], peaks_mask [B,N,C,K].
"""
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from e3nn import o3
# from data.sh_density import SHToGridDensity,apply_l_window,visualize_density_atoms_3d,apply_l_taper
import torch
import numpy as np
import matplotlib.pyplot as plt
from openfold.model.structure_module import AngleResnetBlock,Linear

# 等值面可选依赖
try:
    from skimage.measure import marching_cubes
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False
from data.residue_constants import restype_order_with_x

# aatype vocabulary: 0-19 (standard AA) + 20 (UNK) + 21 (MASK)
aatype_vocab = 22


# def visualize_density_atoms_3d(
#     density: torch.Tensor,        # [G]
#     grid_xyz: torch.Tensor,       # [G, 3]
#     cube_shape: tuple,            # (Dx, Dy, Dz)
#     sphere_mask: torch.Tensor,    # [G] bool
#     atom_positions: torch.Tensor, # [A, 3]
#     atom_types: torch.Tensor=None,# [A]
#     select_type: int=0,           # 只显示该类型的原子（若 atom_types=None 则忽略）
#     normalize: bool=True,
#     mode: str="isosurface",       # "pointcloud" 或 "isosurface"
#     percentile: float=97.5,       # pointcloud：按分位数取高密度点
#     abs_threshold: float=None,    # pointcloud：绝对阈值（优先于 percentile）
#     max_points: int=200000,       # pointcloud：最多点数
#     add_colorbar: bool=True,      # pointcloud：是否加颜色条
#     point_size: float=2.0,        # pointcloud：点大小
#     iso_percentile: float=99.5,   # isosurface：分位阈值
#     voxel: float=1.0,             # ★ 体素物理尺寸（与 make_cube_grid 保持一致）
#     axes_order: tuple=(0,1,2)     # ★ 若 reshape 次序非 (Dx,Dy,Dz)，在这里指定，如 (0,2,1)
# ):
#     Dx, Dy, Dz = cube_shape
#     G = grid_xyz.shape[0]
#     assert density.numel() == G and grid_xyz.shape[-1] == 3
#
#     # 掩膜+归一化
#     vol = density
#     if sphere_mask is not None:
#         vol = vol * sphere_mask.to(vol.dtype)
#     vol_np = vol.detach().cpu().reshape(Dx, Dy, Dz).numpy()
#
#     if normalize:
#         vmin, vmax = float(vol_np.min()), float(vol_np.max())
#         if vmax > vmin:
#             vol_np = (vol_np - vmin) / (vmax - vmin + 1e-12)
#
#     # 原子筛选
#     atoms_np = atom_positions.detach().cpu().numpy()
#     if atom_types is not None:
#         atypes_np = atom_types.detach().cpu().numpy()
#         atoms_np = atoms_np[atypes_np == select_type]
#
#     # 世界坐标范围（用于对齐与设定坐标轴范围）
#     gx = grid_xyz.detach().cpu().numpy().reshape(-1, 3)
#     xyz_min = gx.min(axis=0)
#     xyz_max = gx.max(axis=0)
#
#     fig = plt.figure(figsize=(7, 7))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
#
#     if mode == "isosurface" and SKIMAGE_OK and vol_np.max() > 0:
#         # ★ 轴顺序对齐 marching_cubes 数据
#         vol_np_t = np.transpose(vol_np, axes_order)
#
#         # 等值面阈值
#         #level = float(np.quantile(vol_np_t, iso_percentile / 100.0))
#
#         # 改成：只看正值
#         vol_pos = np.maximum(vol_np_t, 0.0)
#         if (vol_pos > 0).any():
#             level = float(np.quantile(vol_pos[vol_pos > 0], iso_percentile / 100.0))
#         else:
#             level = float(np.quantile(vol_np_t, iso_percentile / 100.0))  # 兜底
#         level = max(level, 1e-8)
#
#         # marching_cubes 在索引坐标系（步长=1，原点=0）上返回顶点
#         verts, faces, normals, values = marching_cubes(
#             vol_np_t, level=level, spacing=(1.0, 1.0, 1.0)
#         )
#
#         # ★ 把顶点从索引坐标 → 世界坐标：
#         # 1) 轴顺序还原回 (x,y,z)
#         inv = np.argsort(axes_order)
#         verts = verts[:, inv]
#         # 2) 缩放：乘体素物理尺寸
#         verts = verts * voxel
#         # 3) 平移：加上网格原点（与 grid_xyz 对齐）
#         verts = verts + xyz_min[None, :]
#
#         from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#         mesh = Poly3DCollection(verts[faces], alpha=0.25, linewidths=0.2)
#         ax.add_collection3d(mesh)
#
#         # ★ 用世界坐标范围设定轴限，确保与原子点一致
#         ax.set_xlim(xyz_min[0], xyz_max[0])
#         ax.set_ylim(xyz_min[1], xyz_max[1])
#         ax.set_zlim(xyz_min[2], xyz_max[2])
#
#     else:
#         # 点云：取阈值以上的体素中心点，并用密度值做颜色映射
#         flat = vol_np.reshape(-1)
#         if abs_threshold is not None:
#             thr = float(abs_threshold)
#         else:
#             thr = float(np.quantile(flat, percentile / 100.0))
#         mask = flat >= thr
#         if mask.sum() == 0:
#             thr = float(np.quantile(flat, 90 / 100.0))
#             mask = flat >= thr
#
#         idx = np.nonzero(mask)[0]
#         if idx.size > max_points:
#             idx = np.random.choice(idx, size=max_points, replace=False)
#
#         pts = gx[idx]                  # 世界坐标点
#         vals = flat[idx]               # 颜色 = 密度
#         sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=point_size, c=vals)
#         if add_colorbar:
#             fig.colorbar(sc, ax=ax, shrink=0.6, label='density')
#
#         ax.set_xlim(xyz_min[0], xyz_max[0])
#         ax.set_ylim(xyz_min[1], xyz_max[1])
#         ax.set_zlim(xyz_min[2], xyz_max[2])
#
#     # 叠加原子（世界坐标）
#     if atoms_np.size > 0:
#         ax.scatter(atoms_np[:,0], atoms_np[:,1], atoms_np[:,2], s=10, c='red', marker='o')
#
#     ax.set_title('3D density vs atoms (type=={})'.format(select_type))
#     try:
#         ax.set_box_aspect(xyz_max - xyz_min)  # 等比例坐标，避免视觉拉伸
#     except Exception:
#         pass
#     plt.tight_layout()
#     plt.show()

def visualize_density_vs_atoms(
    density: torch.Tensor,      # [G] 或 [B,N,C,G] 里的某一张，已选好 b,n,c 后的 1D
    cube_shape: tuple,          # (Dx, Dy, Dz)
    grid_xyz: torch.Tensor,     # [G, 3]
    sphere_mask: torch.Tensor,  # [G] bool
    atom_positions: torch.Tensor,  # [A, 3]（单位/坐标系需与 grid_xyz 一致）
    voxel: float,               # 体素间距（和你 make_cube_grid 时一致）
    slice_tol_vox: int = 1,     # 选中切片时，允许 |index - slice_index| <= tol 的原子被绘制
    use_mask: bool = True,      # 是否应用 sphere_mask
    normalize: bool = True      # 是否把体素值归一化到 [0,1] 显示
):
    """
    可视化密度与原子坐标的正交切片叠加：
    - XY@z_center、XZ@y_center、YZ@x_center 三张切片
    - 原子：落在切片±tol体素范围内的点会被标注
    """
    device = density.device
    G = grid_xyz.shape[0]
    assert density.numel() == G, f"density长度{density.numel()}应等于G={G}"
    Dx, Dy, Dz = cube_shape

    # 1) 体素体积重排
    vol = density
    if use_mask and sphere_mask is not None:
        vol = vol * sphere_mask.to(vol.dtype)
    vol = vol.reshape(Dx, Dy, Dz).detach().cpu().numpy()

    # 2)（可选）归一化以便显示
    if normalize:
        vmin, vmax = float(vol.min()), float(vol.max())
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)

    # 3) 计算网格边界，用于将原子连续坐标 → 体素索引
    #    假设 grid_xyz 是规则立方网格（来自 make_cube_grid），求每轴最小值
    xyz = grid_xyz.detach().cpu().numpy()
    x_min, y_min, z_min = xyz[:,0].min(), xyz[:,1].min(), xyz[:,2].min()

    def world_to_index(pos):  # pos: [A,3] numpy
        idx = np.round((pos - np.array([x_min, y_min, z_min])) / voxel).astype(int)
        return idx

    A = atom_positions.shape[0]
    apos = atom_positions.detach().cpu().numpy()
    aidx = world_to_index(apos)  # [A,3] -> 体素索引
    # 过滤在体素范围内的
    inrange = (
        (aidx[:,0] >= 0) & (aidx[:,0] < Dx) &
        (aidx[:,1] >= 0) & (aidx[:,1] < Dy) &
        (aidx[:,2] >= 0) & (aidx[:,2] < Dz)
    )
    aidx = aidx[inrange]

    # 4) 选择中心切片索引
    cx, cy, cz = Dx // 2, Dy // 2, Dz // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # XY @ z=cz
    ax = axes[0]
    ax.imshow(vol[:, :, cz].T, origin='lower', interpolation='nearest')
    sel = np.abs(aidx[:,2] - cz) <= slice_tol_vox
    if sel.any():
        ax.scatter(aidx[sel,0], aidx[sel,1], s=10, edgecolors='k', facecolors='none')
    ax.set_title(f"XY @ z={cz}")
    ax.set_xlabel("x index"); ax.set_ylabel("y index")

    # XZ @ y=cy
    ax = axes[1]
    ax.imshow(vol[:, cy, :].T, origin='lower', interpolation='nearest')
    sel = np.abs(aidx[:,1] - cy) <= slice_tol_vox
    if sel.any():
        ax.scatter(aidx[sel,0], aidx[sel,2], s=10, edgecolors='k', facecolors='none')
    ax.set_title(f"XZ @ y={cy}")
    ax.set_xlabel("x index"); ax.set_ylabel("z index")

    # YZ @ x=cx
    ax = axes[2]
    ax.imshow(vol[cx, :, :].T, origin='lower', interpolation='nearest')
    sel = np.abs(aidx[:,0] - cx) <= slice_tol_vox
    if sel.any():
        ax.scatter(aidx[sel,1], aidx[sel,2], s=10, edgecolors='k', facecolors='none')
    ax.set_title(f"YZ @ x={cx}")
    ax.set_xlabel("y index"); ax.set_ylabel("z index")

    plt.tight_layout()
    plt.show()

# AlphaFold aatype order (0..19): A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V
AA_ORDER = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
AA2IDX = {aa:i for i,aa in enumerate(AA_ORDER)}

import torch

# 原子名 -> 元素编号  C/N/O/S = 0/1/2/3
ELEMENT_OF_ATOM = {
    'N':1,'CA':0,'C':0,'O':2,'OXT':2,'CB':0,'CG':0,'CG1':0,'CG2':0,
    'CD':0,'CD1':0,'CD2':0,'CE':0,'CE1':0,'CE2':0,'CE3':0,'CZ':0,'CZ2':0,'CZ3':0,'CH2':0,
    'ND1':1,'NE':1,'NE1':1,'NE2':1,'NH1':1,'NH2':1,'NZ':1,
    'OD1':2,'OD2':2,'OE1':2,'OE2':2,'OH':2,'OG':2,'OG1':2,
    'SD':3,'SG':3,
}

AA_LIST = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','UNK']

def assemble_atom14(coords_global, peaks_mask, scores, aatype_probs, restype_name_to_atom14_names):
    """
    coords_global: [B,N,4,K,3]
    peaks_mask:    [B,N,4,K]  (True=有效)
    scores:        [B,N,4,K]  (或 peak_probs)
    aatype_probs:  [B,N,20]   (one-hot也行)
    return:
        atom14_xyz:   [B,N,14,3]
        atom14_exists:[B,N,14] (bool)
    """
    B,N,C,K,_ = coords_global.shape
    device = coords_global.device
    atom14_xyz = torch.zeros(B,N,14,3, device=device, dtype=coords_global.dtype)
    atom14_exists = torch.zeros(B,N,14,   device=device, dtype=torch.bool)

    aatype_idx = aatype_probs.argmax(-1)      # [B,N], 0..19

    # 按分数对每个通道的峰排序（大到小）
    sort_scores, sort_idx = torch.sort(scores, dim=-1, descending=True)  # [B,N,4,K]
    # 重排坐标、mask
    idx_gather = sort_idx.unsqueeze(-1).expand(-1,-1,-1,-1,3)            # [B,N,4,K,3]
    coords_sorted = torch.gather(coords_global, dim=3, index=idx_gather) # [B,N,4,K,3]
    mask_sorted   = torch.gather(peaks_mask,   dim=3, index=sort_idx)    # [B,N,4,K]

    for b in range(B):
        for n in range(N):
            restype = AA_LIST[aatype_idx[b,n].item()] if aatype_probs is not None else 'UNK'
            atom_names = restype_name_to_atom14_names.get(restype, restype_name_to_atom14_names['UNK'])
            # 每个元素通道的“已用指针”
            used_ptr = [0,0,0,0]  # for C/N/O/S
            for a_i, name in enumerate(atom_names):
                if name == '':
                    continue
                elem = ELEMENT_OF_ATOM.get(name, None)
                if elem is None:   # 不识别的名字，跳过
                    continue
                # 循环找到下一个有效峰
                kptr = used_ptr[elem]
                while kptr < K and not mask_sorted[b,n,elem,kptr]:
                    kptr += 1
                if kptr < K:
                    atom14_xyz[b,n,a_i] = coords_sorted[b,n,elem,kptr]
                    atom14_exists[b,n,a_i] = True
                    used_ptr[elem] = kptr + 1
                else:
                    # 该元素通道峰不够，留空
                    pass

    return atom14_xyz, atom14_exists



# 元素通道映射：C/N/O/S -> 0/1/2/3
ELEM2IDX = {"C": 0, "N": 1, "O": 2, "S": 3}





def assemble_atom14_with_CA(
    coords_global: torch.Tensor,    # [B,N,4,K,3]   (C/N/O/S 的峰全局坐标)
    peaks_mask: torch.Tensor,       # [B,N,4,K]     (有效峰)
    scores: torch.Tensor,           # [B,N,4,K]     (峰分数)
    aatype_probs: torch.Tensor,     # [B,N,20]      (可 one-hot)
    restype_name_to_atom14_names: dict[str, list[str]],
    tpos: torch.Tensor,             # [B,N,3]       (CA 的全局坐标 = 你的输入 t)
):
    """
    返回：
      atom14_xyz:    [B,N,14,3]
      atom14_exists: [B,N,14] (bool)
    说明：
      - CA 槽位（通常是 index=1）直接用 tpos 写入，不从 peaks 取。
      - 其他原子从对应元素通道的峰里，按分数从高到低选。
    """
    B, N, C4, K, _ = coords_global.shape
    device = coords_global.device
    dtype  = coords_global.dtype

    atom14_xyz    = torch.zeros(B, N, 14, 3, device=device, dtype=dtype)
    atom14_exists = torch.zeros(B, N, 14,   device=device, dtype=torch.bool)

    # 先对每个通道把峰按分数排序
    sort_scores, sort_idx = torch.sort(scores, dim=-1, descending=True)           # [B,N,4,K]
    gather_idx = sort_idx.unsqueeze(-1).expand(-1, -1, -1, -1, 3)                 # [B,N,4,K,3]
    coords_sorted = torch.gather(coords_global, dim=3, index=gather_idx)          # [B,N,4,K,3]
    mask_sorted   = torch.gather(peaks_mask,   dim=3, index=sort_idx)             # [B,N,4,K]

    # 每个 (b,n,elem) 的已用峰指针
    used_ptr = torch.zeros(B, N, 4, dtype=torch.long, device=device)

    # 逐残基填充
    aatype_idx = aatype_probs.argmax(dim=-1)  # [B,N]

    for b in range(B):
        for n in range(N):
            restype = AA_LIST[aatype_idx[b, n].item()]
            atom_names = restype_name_to_atom14_names.get(restype, restype_name_to_atom14_names["UNK"])

            for a_i, name in enumerate(atom_names):
                if name == "":
                    continue

                if name == "CA":
                    # 1) CA 直接用 tpos
                    atom14_xyz[b, n, a_i] = tpos[b, n]
                    atom14_exists[b, n, a_i] = True
                    continue

                # 2) 其他原子按元素通道取峰
                elem_letter = name[0]  # 直接取第一个字母
                if elem_letter in ELEM2IDX:
                    elem_idx = ELEM2IDX[elem_letter]
                if elem_idx is None:
                    continue

                # 找到下一个有效峰
                kptr = used_ptr[b, n, elem_idx].item()
                while kptr < K and not mask_sorted[b, n, elem_idx, kptr]:
                    kptr += 1

                if kptr < K:
                    atom14_xyz[b, n, a_i]    = coords_sorted[b, n, elem_idx, kptr]
                    atom14_exists[b, n, a_i] = True
                    used_ptr[b, n, elem_idx] = kptr + 1
                # 峰不够就保持空（exists=False）

    return atom14_xyz, atom14_exists

# Side-chain only atom counts per element [C,N,O,S] (excludes backbone N,CA,C,O)
AA_K_PRIOR = torch.tensor([
    [1,0,0,0],  # ALA: CB
    [4,3,0,0],  # ARG: CB,CG,CD,CZ ; NE,NH1,NH2
    [2,1,1,0],  # ASN: CB,CG ; ND2,OD1
    [2,0,2,0],  # ASP: CB,CG ; OD1,OD2
    [1,0,0,1],  # CYS: CB ; SG
    [3,1,1,0],  # GLN: CB,CG,CD ; NE2,OE1
    [3,0,2,0],  # GLU: CB,CG,CD ; OE1,OE2
    [0,0,0,0],  # GLY: (no side-chain heavy atom beyond H)
    [2,1,0,0],  # HIS: CB,CG ; ND1/NE2 (1 N counted here; ring has 2 N but one overlaps? use [3,2,0,0] if counting all)
    [4,0,0,0],  # ILE: CB,CG1,CG2,CD1
    [4,0,0,0],  # LEU: CB,CG,CD1,CD2
    [4,1,0,0],  # LYS: CB,CG,CD,CE ; NZ
    [3,0,0,1],  # MET: CB,CG,CE ; SD
    [7,0,0,0],  # PHE: phenyl (CB,CG,CD1,CD2,CE1,CE2,CZ)
    [3,0,0,0],  # PRO: CB,CG,CD
    [1,0,1,0],  # SER: CB ; OG
    [2,0,1,0],  # THR: CB,CG2 ; OG1
    [9,1,0,0],  # TRP: indole (approx: many carbons + NE1)
    [7,0,1,0],  # TYR: phenyl + OH
    [3,0,0,0],  # VAL: CB,CG1,CG2
], dtype=torch.float32)
# 如果你的 SH 是“叠加了 13 个原子（= sidechain + backbone N/C/O，排除 CA）”，
# 建议把 backbone 直接并入先验：
BB = torch.tensor([1, 1, 1, 0], dtype=torch.float32)  # +C(backbone), +N, +O, +S=0

# ---------------------------
# Spherical helpers
# ---------------------------

# ---------------------------
# Grid utilities
# ---------------------------

def make_cube_grid(r_max: float, voxel: float, device):
    n = int(math.floor((2 * r_max) / voxel)) + 1
    xs = torch.linspace(-r_max, r_max, n, device=device)
    X, Y, Z = torch.meshgrid(xs, xs, xs, indexing="ij")
    grid = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
    return grid, (n, n, n)


def mask_sphere(grid_xyz: torch.Tensor, r_max: float):
    return (grid_xyz.norm(dim=-1) <= r_max)


def masked_avg_pool_1d(x, mask, eps=1e-8):
    """
    x:    [B*N, C, A]
    mask: [B*N, 1, A] (float 0/1)
    """
    if mask is None:
        return x.mean(dim=-1)
    wsum = (x * mask).sum(dim=-1)
    denom = mask.sum(dim=-1).clamp_min(eps)
    return wsum / denom
# ---------- 通用残差块 ----------

# class MLPResBlock(nn.Module):
#     def __init__(self, d, hidden=None, dropout=0.0, act=nn.SiLU, prenorm=True):
#         super().__init__()
#         hidden = hidden or d * 4
#         self.prenorm = prenorm
#         self.ln = nn.LayerNorm(d)
#         self.fc1 = nn.Linear(d, hidden)
#         self.fc2 = nn.Linear(hidden, d)
#         self.drop = nn.Dropout(dropout)
#         self.act = act()
#
#     def forward(self, x):
#         y = x
#         if self.prenorm:
#             y = self.ln(y)
#         y = self.fc1(y)
#         y = self.act(y)
#         y = self.drop(y)
#         y = self.fc2(y)
#         y = self.drop(y)
#         return x + y


class Conv1dResBlock(nn.Module):
    """
    预归一化 + Conv1d 残差；支持可选 groups 做分组卷积（默认=1，即普通卷积）。
    """
    def __init__(self, ch, kernel_size=3, groups=1, dropout=0.0, act=nn.SiLU, prenorm=True):
        super().__init__()
        pad = kernel_size // 2
        self.prenorm = prenorm
        # 对 1D 特征做 Channel-wise Norm，可用 GroupNorm(1, ch) 或 LayerNorm on T 维转置
        self.gn = nn.GroupNorm(num_groups=1, num_channels=ch)
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=pad, groups=groups)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=pad, groups=groups)
        self.act = act()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # x: [B*N, ch, R]
        y = x
        if self.prenorm:
            y = self.gn(y)
        y = self.conv1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.drop(y)
        return x + y


# ---------- 主网络：把三段都改成残差 ----------

class MLPBlock(nn.Module):
    def __init__(self, d_in, d_hidden, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_in)
        self.ln  = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = F.gelu(self.fc1(x))
        y = self.dropout(y)
        y = self.fc2(y)
        return self.ln(x + y)

class SH2Atom14(nn.Module):
    """
    输入:  feat [B,N,H]
          aatype [B,N]  (可选, 0..20) 传则做条件编码
    输出:  coords [B,N,14,3]
    """
    def __init__(
        self,
        h_in: int,          # 输入通道 H
        d_model: int = 256, # 中间宽度
        num_blocks: int = 4,
        cond_aatype: bool = False,
        dropout: float = 0.0,
        out_range: float = 8,  # tanh 缩放到 ±out_range Å
    ):
        super().__init__()
        self.cond_aatype = cond_aatype
        cond_dim = d_model if cond_aatype else 0

        # 输入投影
        self.in_proj = nn.Sequential(
            nn.LayerNorm(h_in),
            nn.Linear(h_in, d_model),
            nn.GELU(),
        )

        # aatype 条件
        if cond_aatype:
            self.aatype_emb = nn.Embedding(aatype_vocab, d_model)
            self.fuse = nn.Linear(d_model + d_model, d_model)

        # 残差 MLP 堆叠
        self.blocks = nn.ModuleList([MLPBlock(d_model, d_model*4, dropout=dropout) for _ in range(num_blocks)])

        # 输出头：一次性回归 14*3 个坐标
        self.head = nn.Linear(d_model, 14*3)

        # 输出缩放
        self.out_range = out_range

        # 简单初始化：让初始输出接近 0
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feat: torch.Tensor, aatype: torch.Tensor | None = None):
        """
        feat:   [B,N,10,3]  side atoms
        aatype: [B,N] (可选)
        return: coords [B,N,14,3]
        """
        B, N, H = feat.shape
        x = self.in_proj(feat)                    # [B,N,d]
        if self.cond_aatype and (aatype is not None):
            t = self.aatype_emb(aatype.long())    # [B,N,d]
            x = torch.cat([x, t], dim=-1)
            x = F.gelu(self.fuse(x))              # [B,N,d]

        for blk in self.blocks:
            x = blk(x)                            # [B,N,d]

        out = self.head(x)                        # [B,N,42]
        # 输出保持在缩放域：取消 Å 级别 out_range 放大
        out = torch.tanh(out)
        coords = out.view(B, N, 14, 3)            # [B,N,14,3]
        return coords



class SideAtomsFeatureHead(nn.Module):
    """
    输入:  X_sc [B,N,A(=10),3] ；可选 atom_mask [B,N,A]；可选 node_mask [B,N]
    输出:  logits [B,N,num_classes], feat [B,N,hidden]
    """
    def __init__(self, A=11, hidden=256, num_classes=20, dropout=0.1,
                 conv_blocks=4, mlp_blocks=4, fuse_blocks=4, conv_groups=1, **kwargs):
        super().__init__()
        self.A = A
        self.in_ch = 3              # 仅使用坐标通道（你已中心化，不再添加其他派生特征）
        self.hidden = hidden
        self.num_classes = num_classes

        # ---- Branch A: 原子轴 Conv1d 残差 + 掩码池化 ----
        self.branchA_in = nn.Sequential(
            nn.Conv1d(self.in_ch, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.branchA_blocks = nn.ModuleList([
            Conv1dResBlock(hidden, kernel_size=3, groups=conv_groups, dropout=dropout)
            for _ in range(conv_blocks)
        ])

        # ---- Branch B: flatten + MLP 残差 ----
        flat_dim = self.A * self.in_ch
        self.branchB_in = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, hidden),
            nn.SiLU(),
        )
        self.branchB_blocks = nn.ModuleList([
            MLPResBlock(hidden, hidden * 4, dropout=dropout)
            for _ in range(mlp_blocks)
        ])

        # ---- 融合 ----
        self.fuse_in = nn.Linear(hidden * 2, hidden)
        self.fuse_blocks = nn.ModuleList([
            MLPResBlock(hidden, hidden * 4, dropout=dropout)
            for _ in range(fuse_blocks)
        ])

        # ---- 分类头 ----
        if num_classes > 0:
            self.head = nn.Linear(hidden, num_classes)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else:
            self.head = None

    def forward(self, X_sc: torch.Tensor,
                atom_mask: torch.Tensor | None = None,
                node_mask: torch.Tensor | None = None):
        # X_sc: [B,N,A,3]
        B, N, A, D = X_sc.shape
        assert A == self.A and D == 3, f"Expect [B,N,{self.A},3], got {X_sc.shape}"
        x = X_sc

        # Branch A: [B,N,A,3] -> [B*N, 3, A] -> Conv1d ... -> masked avg pool -> [B,N,hidden]
        xa = x.view(B * N, A, 3).transpose(1, 2)           # [B*N, 3, A]
        xa = self.branchA_in(xa)                            # [B*N, hidden, A]
        for blk in self.branchA_blocks:
            xa = blk(xa)
        if atom_mask is not None:
            am = atom_mask.view(B * N, 1, A).to(xa.dtype).to(xa.device)
        else:
            am = None
        xa = masked_avg_pool_1d(xa, am)                    # [B*N, hidden]
        xa = xa.view(B, N, self.hidden)                    # [B,N,hidden]

        # Branch B: flatten per residue -> [B,N,hidden] -> residual MLP 堆叠
        xb = x.reshape(B, N, A * 3)                        # [B,N,A*3]
        # 对缺失原子补零即可；LayerNorm 已在模块内
        xb = self.branchB_in(xb)                           # [B,N,hidden]
        for blk in self.branchB_blocks:
            xb = blk(xb)

        # Fuse
        xcat = torch.cat([xa, xb], dim=-1)                 # [B,N,2*hidden]
        feat = self.fuse_in(xcat)                          # [B,N,hidden]
        for blk in self.fuse_blocks:
            feat = blk(feat)

        # Head
        if self.head is not None:
            logits = self.head(feat)                       # [B,N,num_classes]
            if node_mask is not None:
                mask = node_mask.to(dtype=logits.dtype, device=logits.device)[..., None]
                logits = logits + (1.0 - mask) * (-1e9)
        else:
            logits = None

        return logits, feat


class Atom112type(nn.Module):
    """
    输入:  feat  [B, N, H]   (默认 H=256)
    输出:  coords [B, N, 11, 3]
    说明:  纯 MLP 残差解码，不含 aatype 条件
    """
    def __init__(
        self,
        h_in: int = 33,         # 输入通道 H
        d_model: int = 256,      # 中间宽度
        num_blocks: int = 4,     # 残差 MLP 层数
        dropout: float = 0.0,
        num_classes: int = 21,

    ):
        super().__init__()


        # 输入投影
        self.in_proj = nn.Sequential(
            nn.LayerNorm(h_in),
            nn.Linear(h_in, d_model),
            nn.GELU(),
        )

        # 残差 MLP 堆叠
        self.blocks = nn.ModuleList([
            MLPBlock(d_model, d_model * 4, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # 输出头：一次性回归 11*3 个坐标
        self.head = nn.Linear(d_model, num_classes)

        # 初始化：让初始输出接近 0
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat:   [B, N, H]
        return: coords [B, N, 11, 3]
        """
        B, N, _,_ = feat.shape
        x = self.in_proj(feat.reshape(B, N, -1))           # [B, N, d_model]
        for blk in self.blocks:
            x = blk(x)                   # [B, N, d_model]

        out = self.head(x)               # [B, N, 21]



        return out


class Feat2Atom11(nn.Module):
    """
    输入:  feat  [B, N, H]   (默认 H=256)
    输出:  coords [B, N, 11, 3]
    说明:  纯 MLP 残差解码，不含 aatype 条件
    """
    def __init__(
        self,
        h_in: int = 256,         # 输入通道 H
        d_model: int = 256,      # 中间宽度
        num_blocks: int = 4,     # 残差 MLP 层数
        dropout: float = 0.0,
        out_range: float = 16.0,  # tanh 缩放到 ±out_range Å
    ):
        super().__init__()
        self.out_range = out_range

        # 输入投影
        self.in_proj = nn.Sequential(
            nn.LayerNorm(h_in),
            nn.Linear(h_in, d_model),
            nn.GELU(),
        )

        # 残差 MLP 堆叠
        self.blocks = nn.ModuleList([
            MLPBlock(d_model, d_model * 4, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # 输出头：一次性回归 11*3 个坐标
        self.head = nn.Linear(d_model, 11 * 3)

        # 初始化：让初始输出接近 0
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat:   [B, N, H]
        return: coords [B, N, 11, 3]
        """
        B, N, _ = feat.shape
        x = self.in_proj(feat)           # [B, N, d_model]
        for blk in self.blocks:
            x = blk(x)                   # [B, N, d_model]

        out = self.head(x)               # [B, N, 33]
        # 输出保持在缩放域：取消 Å 级别 out_range 放大
        out = torch.tanh(out)*self.out_range
        coords = out.view(B, N, 11, 3)   # [B, N, 11, 3]
        return coords


import torch
import torch.nn as nn
from openfold.model.primitives import Linear, LayerNorm


class SequenceHead(nn.Module):
    """
    序列预测专用头 (Deep MLP Head)。

    结构:
    Latent -> [MLP ResBlocks] -> Refined Features -> Logits

    优势:
    比简单的 Linear 层表达能力更强，能更好地解耦几何特征和序列特征。
    """

    def __init__(
            self,
            c_in: int,  # 输入维度 (Trunk output dim)
            c_hidden: int,  # 中间维度
            num_layers: int = 3,  # MLP 深度 (觉得一层太简单，这里可以加深)
            dropout: float = 0.1,
            num_classes: int = 20,  # 20种氨基酸
    ):
        super().__init__()

        # 1. 特征增强网络 (MLP)
        # 先升维或保持维度，通过残差块提取深层特征
        self.mlp = nn.Sequential(
            Linear(c_in, c_hidden),
            LayerNorm(c_hidden),
            nn.SiLU(),
            # 堆叠 ResBlock
            *[MLPResBlock(c_hidden, c_hidden * 4, dropout) for _ in range(num_layers)],
            LayerNorm(c_hidden)
        )

        # 2. 分类头 (Logits)
        self.projection = Linear(c_hidden, num_classes)

        # 初始化
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x, mask=None):
        """
        x: [B, N, C]
        mask: [B, N] (可选)
        """
        # 1. 提取深层特征
        feat = self.mlp(x)  # [B, N, C_hidden]

        # 2. 预测 Logits
        logits = self.projection(feat)  # [B, N, 20]

        # 3. Masking (可选，防止 Padding 影响 Loss)
        if mask is not None:
            # 扩展 mask 维度 [B, N, 1]
            mask = mask.unsqueeze(-1).to(dtype=logits.dtype)
            # Mask 掉的位置设为极小值 (在 Softmax 中趋近于 0)
            logits = logits * mask + (1.0 - mask) * -1e9

        return logits


# 简单的 MLP ResBlock 辅助类 (如果你没有的话)
class MLPResBlock(nn.Module):
    def __init__(self, c_in, c_mid, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            Linear(c_in, c_mid),
            nn.SiLU(),
            nn.Dropout(dropout),
            Linear(c_mid, c_in),
            nn.Dropout(dropout)
        )
        self.norm = LayerNorm(c_in)

    def forward(self, x):
        return self.norm(x + self.net(x))


class NodeFeatExtractorWithHeads(nn.Module):
    """
    逐节点特征提取 + 双头
    输入:
        x_in: [B, N, H_in]   (默认 H_in=256)
        node_mask: [B, N]    (可选; 仅用于屏蔽 logits)
    输出:
        coords: [B, N, 11, 3]
        logits: [B, N, num_classes]   (默认 21)
        feat:   [B, N, out_dim或hidden]
    """
    def __init__(
        self,
        h_in: int = 256,
        hidden: int = 256,
        mlp_blocks: int = 2,
        dropout: float = 0.1,
        # 输出给后续模块（如 Transformer）的维度；None 表示保持 hidden
        out_dim: int | None = None,
        norm_after: bool = True,
        # Feat2Atom11（坐标头）相关
        coord_blocks: int = 4,
        coord_out_range: float = 10.0,
        # 分类头
        num_classes: int = 21,
    ):
        super().__init__()
        self.num_classes = num_classes

        # === 特征提取（纯 MLP，逐节点） ===
        self.feat_in = nn.Sequential(
            nn.LayerNorm(h_in),
            nn.Linear(h_in, hidden),
            nn.SiLU(),
        )
        self.feat_blocks = nn.ModuleList([
            MLPResBlock(hidden, hidden * 4, dropout=dropout)
            for _ in range(mlp_blocks)
        ])

        # 输出投影到 out_dim（便于对接 Transformer）
        if out_dim is None or out_dim == hidden:
            self.proj_out = nn.Identity()
            self.out_dim = hidden
        else:
            self.proj_out = nn.Linear(hidden, out_dim)
            self.out_dim = out_dim

        self.out_norm = nn.LayerNorm(self.out_dim) if norm_after else nn.Identity()

        # === 头1：坐标回归（Feat2Atom11） ===
        self.coord_head = Feat2Atom11(
            h_in=self.out_dim,
            d_model=self.out_dim,
            num_blocks=coord_blocks,
            dropout=dropout,
            out_range=coord_out_range,
        )

        # === 头2：21 维分类/属性 ===
        if num_classes > 0:
            self.class_head = nn.Linear(self.out_dim, num_classes)
            nn.init.zeros_(self.class_head.weight)
            nn.init.zeros_(self.class_head.bias)
        else:
            self.class_head = None

    def forward(self, x_in: torch.Tensor, node_mask: torch.Tensor | None = None):
        """
        x_in:      [B, N, H_in]
        node_mask: [B, N] (可选; 仅屏蔽 logits)
        return:    coords [B,N,11,3], logits [B,N,num_classes] or None, feat [B,N,out_dim]
        """
        # 特征提取
        x = self.feat_in(x_in)              # [B,N,hidden]
        for blk in self.feat_blocks:
            x = blk(x)                      # [B,N,hidden]
        feat = self.out_norm(self.proj_out(x))  # [B,N,out_dim]

        # 头1：坐标
        coords = self.coord_head(feat)      # [B,N,11,3]

        # 头2：分类/属性
        if self.class_head is not None:
            logits = self.class_head(feat)  # [B,N,num_classes]
            if node_mask is not None:
                mask = node_mask.to(dtype=logits.dtype, device=logits.device)[..., None]
                logits = logits + (1.0 - mask) * (-1e9)
        else:
            logits = None

        return coords, logits


class SHFeatureHead(nn.Module):
    """
    输入:  SH [B,N,C,L+1,2L+1,R]
    输出:  logits [B,N,20], feat [B,N,hidden]
    """
    def __init__(self, C, L_max, R_bins, hidden=256, num_classes=20, dropout=0.1,
                 conv_blocks=4, mlp_blocks=4, fuse_blocks=4, conv_groups=1, **kwargs):
        super().__init__()
        self.C, self.L_max, self.R = C, L_max, R_bins
        in_ch = C * (L_max + 1) * (2 * L_max + 1)  # 把 l,m,c 合在通道维

        # ---- Branch A: 1x1 提升到 mid，再若干 Conv 残差块 + GAP ----
        self.branchA_in = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.branchA_blocks = nn.ModuleList([
            Conv1dResBlock(hidden, kernel_size=3, groups=conv_groups, dropout=dropout)
            for _ in range(conv_blocks)
        ])
        self.branchA_pool = nn.AdaptiveAvgPool1d(1)  # -> [B*N, hidden, 1]

        # ---- Branch B: 直接 flatten + 若干 MLP 残差块 ----
        self.branchB_in = nn.Sequential(
            nn.LayerNorm(in_ch * R_bins),
            nn.Linear(in_ch * R_bins, hidden),
            nn.SiLU(),
        )
        self.branchB_blocks = nn.ModuleList([
            MLPResBlock(hidden, hidden*4, dropout=dropout)
            for _ in range(mlp_blocks)
        ])

        # ---- 融合（concat 后的 MLP 残差块）----
        self.fuse_in = nn.Linear(hidden * 2, hidden)
        self.fuse_blocks = nn.ModuleList([
            MLPResBlock(hidden, hidden*4, dropout=dropout)
            for _ in range(fuse_blocks)
        ])

        self.num_classes=num_classes
        if num_classes > 0:
            self.head = nn.Linear(hidden, num_classes)
            # 初始化分类头为小值
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)



    def forward(self, SH: torch.Tensor, node_mask: torch.Tensor | None = None):
        # SH: [B,N,C,L+1,2L+1,R]
        B, N, C, Lp1, M, R = SH.shape
        assert C == self.C and Lp1 == self.L_max + 1 and R == self.R
        x = SH

        # Branch A：把 (C,L,M) 合到通道，R 作为长度
        xa = x.reshape(B, N, C * Lp1 * M, R).reshape(B * N, C * Lp1 * M, R)
        xa = self.branchA_in(xa)  # [B*N, hidden, R]
        for blk in self.branchA_blocks:
            xa = blk(xa)           # 残差堆叠
        xa = self.branchA_pool(xa).squeeze(-1)       # [B*N, hidden]
        xa = xa.view(B, N, -1)

        # Branch B：flatten 成向量，再过 MLP 残差
        xb = x.reshape(B, N, -1)                         # [B,N,in_ch*R]
        xb = self.branchB_in(xb)                      # [B,N,hidden]
        for blk in self.branchB_blocks:
            xb = blk(xb)                              # 残差堆叠

        # Fuse：concat -> 线性 -> 残差
        xcat = torch.cat([xa, xb], dim=-1)            # [B,N,2*hidden]
        feat = self.fuse_in(xcat)                     # [B,N,hidden]
        for blk in self.fuse_blocks:
            feat = blk(feat)                          # 残差堆叠

        # 分类头
        if self.num_classes > 0:
            logits = self.head(feat)                      # [B,N,num_classes]

            if node_mask is not None:
                mask = node_mask.to(dtype=logits.dtype, device=logits.device)[..., None]
                logits = logits + (1.0 - mask) * (-1e9)
        else:
            logits = None

        return logits, feat



class SHPredictionHead(nn.Module):
    """
    将标量特征 x 映射为 SH 系数。
    输入:  x [B, N, d_model]
    输出:  sh_pred [B, N, C, L+1, 2L+1, R_bins]
    """
    def __init__(
        self,
        d_model: int,
        C: int,
        L_max: int,
        R_bins: int,
        n_blocks: int = 2,
        hidden: int | None = None,
        dropout: float = 0.1,
        act = nn.SiLU,
        prenorm: bool = True,
        zero_init: bool = True,   # 将最后一层权重/偏置置零，训练更稳
        out_gain: float = 1.0,    # 可学习的整体缩放
        learnable_gain: bool = False,
    ):
        super().__init__()
        self.C = C
        self.L_max = L_max
        self.R_bins = R_bins
        self.out_dim = C * (L_max + 1) * (2 * L_max + 1) * R_bins

        # 若干个残差MLP块
        hidden = hidden or (4 * d_model)
        blocks = []
        for _ in range(n_blocks):
            blocks.append(MLPResBlock(d_model, hidden=hidden, dropout=dropout, act=act, prenorm=prenorm))
        self.blocks = nn.Sequential(*blocks) if blocks else nn.Identity()

        # 投影到目标维度
        self.proj = nn.Linear(d_model, self.out_dim)

        # 输出整体增益（可选）
        if learnable_gain:
            self.out_gain = nn.Parameter(torch.tensor(float(out_gain)))
        else:
            self.register_buffer("out_gain", torch.tensor(float(out_gain)), persistent=False)

        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, d_model]
        return: [B, N, C, L+1, 2L+1, R_bins]
        """
        y = self.blocks(x)                # [B,N,d]
        y = self.proj(y)                  # [B,N,out_dim]
        y = y * self.out_gain
        B, N, _ = y.shape
        return y.view(B, N, self.C, self.L_max + 1, 2 * self.L_max + 1, self.R_bins)

class SHTypeHybridHead(nn.Module):
    """
    输入:  SH [B,N,C,L+1,2L+1,R]
    输出:  logits [B,N,20]
    结构:
      - Branch A: 先通道混合(1x1 conv) -> 径向卷积(3x) -> GAP
      - Branch B: 直接 flatten + MLP
      - 融合: concat 两支，再 MLP 输出
    """
    def __init__(self, C, L_max, R_bins, hidden=256, num_classes=20, dropout=0.1,**kwargs):
        super().__init__()
        self.C, self.L_max, self.R = C, L_max, R_bins
        mid = hidden
        in_ch = C * (L_max + 1) * (2 * L_max + 1)  # 通道数（不含 R）

        # ---- Branch A: per-R 通道混合 + 径向卷积 ----
        # 先用 1x1 Conv 做“对每个 R 的线性混合”
        self.branchA = nn.Sequential(
            nn.Conv1d(in_ch, mid, kernel_size=1), nn.SiLU(),
            nn.Conv1d(mid, mid, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv1d(mid, mid, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv1d(mid, mid, kernel_size=3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1)  # -> [B*N, mid, 1]
        )

        # ---- Branch B: 直接 flatten + MLP ----
        self.branchB = nn.Sequential(
            nn.LayerNorm(in_ch * R_bins),
            nn.Linear(in_ch * R_bins, hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Dropout(dropout)
        )

        # ---- 融合 + 分类 ----
        self.fuse = nn.Sequential(
            nn.Linear(mid + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden)

        )
        self.head = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )

    def forward(self, SH: torch.Tensor, node_mask: torch.Tensor | None = None):
        # SH: [B,N,C,L+1,2L+1,R]
        B, N, C, Lp1, M, R = SH.shape
        assert C == self.C and Lp1 == self.L_max + 1 and R == self.R

        x = SH  # 可选：能量 x = SH.pow(2)

        # ---- Branch A ----
        # [B,N,C,L+1,2L+1,R] -> [B*N, in_ch, R]
        xa = x.view(B, N, C * Lp1 * M, R).reshape(B * N, C * Lp1 * M, R)
        xa = self.branchA(xa).squeeze(-1)            # [B*N, mid]
        xa = xa.view(B, N, -1)                       # [B,N,mid]

        # ---- Branch B ----
        xb = x.view(B, N, -1)                        # [B,N,in_ch*R]
        xb = self.branchB(xb)                        # [B,N,hidden]

        # ---- Fuse & Classify ----
        xcat = torch.cat([xa, xb], dim=-1)           # [B,N, mid+hidden]
        feat=self.fuse(xcat)
        logits = self.head(feat)                     # [B,N,num_classes]

        # 屏蔽无效节点
        if node_mask is not None:
            mask = node_mask.to(dtype=logits.dtype, device=logits.device)[..., None]
            logits = logits + (1.0 - mask) * (-1e9)

        return logits,feat

# ---------------------------
# SH -> torsion_angles
# ---------------------------

class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        super(AngleResnet, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(self.c_in, self.c_hidden)
        self.linear_initial = Linear(self.c_in, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial=s
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s ** 2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s
class SHGeoResHead(nn.Module):
    """
    输入:  SH [B,N,C,L+1,2L+1,R]
    输出:  logits [B,N,num_classes], coords [B,N,14,3]
    """

    def __init__(
        self,
        C,
        L_max,
        R_bins,
        hidden=256,
        num_classes=21,
        dropout=0.1,
        ctx_layers=8,
        ctx_heads=8,
        ctx_dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.C, self.L_max, self.R = C, L_max, R_bins
        self.num_classes = num_classes

        # 先把 SH 密度编码成 residue-level embedding（不输出 logits）
        self.SH_embedding = SHFeatureHead(
            C, L_max, R_bins, hidden=hidden, num_classes=0, dropout=dropout
        )

        self.pre_ln = nn.LayerNorm(hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=ctx_heads,
            dim_feedforward=hidden * 4,
            dropout=ctx_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.context_tfmr = nn.TransformerEncoder(
            encoder_layer, num_layers=ctx_layers, enable_nested_tensor=False
        )
        self.post_proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        if num_classes > 0:
            self.seq_head = nn.Linear(hidden, num_classes)
        else:
            self.seq_head = None

        self.atoms14_Head = SH2Atom14(h_in=hidden)

    def forward(self, SH: torch.Tensor, node_mask: torch.Tensor | None = None):
        B, N, C, Lp1, M, R = SH.shape
        assert C == self.C and Lp1 == self.L_max + 1 and R == self.R

        _, feat = self.SH_embedding(SH, node_mask)
        feat = self.pre_ln(feat)

        if node_mask is not None:
            key_padding_mask = (1 - node_mask).to(torch.bool)
        else:
            key_padding_mask = None

        feat = self.context_tfmr(feat, src_key_padding_mask=key_padding_mask)
        feat = self.post_proj(feat)

        if self.seq_head is not None:
            logits = self.seq_head(feat)
            if node_mask is not None:
                logits = logits + (1.0 - node_mask[..., None]) * (-1e9)
        else:
            logits = None

        coords = self.atoms14_Head(feat)
        return logits, coords


# ---------------------------
# Peak extraction with dynamic K & score threshold
# ---------------------------
class PeakExtractor(nn.Module):
    def __init__(self, voxel_size: float, min_distance: float, topk_per_channel=1, min_score: Optional[float]=None):
        super().__init__()
        self.voxel = float(voxel_size)
        self.min_dist = float(min_distance)
        self.topk_spec = topk_per_channel
        self.min_score = min_score

    def _expand_topk(self, B, N, C, device):
        spec = self.topk_spec
        if isinstance(spec, int):
            return torch.full((B, N, C), int(spec), device=device, dtype=torch.long)
        if isinstance(spec, (list, tuple)):
            t = torch.tensor(spec, device=device, dtype=torch.long)
        else:
            t = spec
        if t.dim() == 1:
            return t.view(1,1,-1).expand(B,N,-1).clone()
        elif t.dim() == 3:
            return t.to(device=device, dtype=torch.long)
        else:
            raise ValueError("topk_per_channel must be int, [C], or [B,N,C]")

    def forward(self, density: torch.Tensor, cube_shape, grid_xyz, sphere_mask,
                topk_map: Optional[torch.Tensor]=None, min_score: Optional[float]=None):
        """
        density: [B,N,C,G]（建议用 bf16/fp16 存）
        cube_shape: (nx,ny,nz)
        grid_xyz: [G,3]（可放 CPU）
        sphere_mask: [G] bool
        """
        B, N, C, G = density.shape
        nx, ny, nz = cube_shape
        min_score = self.min_score if min_score is None else min_score

        # 1) 原地屏蔽立方体外的体素，避免两个大临时
        #    （fp16/bf16 下用 -1e4 替代 -1e9，避免 -inf 溢出）
        neg_large = density.new_tensor(-1e9 if density.dtype == torch.float32 else -1e4)
        mask_bool = sphere_mask.to(torch.bool)
        density = density.clone()  # 如果后续还要用原密度，保留一份；否则你也可以直接 in-place
        density.masked_fill_(~mask_bool.view(1, 1, 1, -1), neg_large)

        # expand K
        if topk_map is None:
            topk_map = self._expand_topk(B, N, C, density.device)
        maxK = int(topk_map.max().item()) if topk_map.numel() > 0 else 0

        peaks_xyz = density.new_zeros(B, N, C, maxK, 3)
        peaks_score = density.new_full((B, N, C, maxK), -1e9 if density.dtype == torch.float32 else -1e4)
        peaks_mask = torch.zeros(B, N, C, maxK, dtype=torch.bool, device=density.device)

        coords = torch.stack(torch.meshgrid(
            torch.arange(nx, device=density.device),
            torch.arange(ny, device=density.device),
            torch.arange(nz, device=density.device),
            indexing="ij"), dim=-1).view(-1, 3)

        min_sep_vox = max(1, int(round(self.min_dist / self.voxel)))
        dens_cube = density.view(B, N, C, nx, ny, nz)

        for b in range(B):
            for n in range(N):
                for c in range(C):
                    Kc = int(topk_map[b, n, c].item())
                    if Kc <= 0:
                        continue
                    v = dens_cube[b, n, c].view(-1).clone()
                    got = 0
                    while got < Kc:
                        idx = torch.argmax(v)
                        score = v[idx]
                        if score < -1e8:
                            break
                        if (min_score is not None) and (score < min_score):
                            break
                        peaks_score[b, n, c, got] = score
                        peaks_xyz[b, n, c, got] = grid_xyz[idx]
                        peaks_mask[b, n, c, got] = True
                        got += 1
                        # NMS neighborhood suppression
                        ijk = coords[idx]
                        d = (coords - ijk).abs()
                        keep = (d.max(dim=-1).values > min_sep_vox)
                        v = torch.where(keep, v, torch.full_like(v, -1e9 if density.dtype == torch.float32 else -1e4))
        return peaks_xyz, peaks_score, peaks_mask

# ---------------------------
# Full decoder
# ---------------------------
class SHSidechainDecoder(nn.Module):
    def __init__(self, L_max: int, R_bins: int, r_max: float=6.0, voxel_size: float=0.3,
                 min_peak_distance: float=0.8, topk_per_channel=4, min_score: Optional[float]=None,**kwargs):
        super().__init__()
        self.sh2dens = SHToGridDensity(L_max, R_bins, r_max, voxel_size)
        self.voxel_size=voxel_size
        print('min_peak_distance:',min_peak_distance)
        self.peak = PeakExtractor(voxel_size, min_peak_distance, topk_per_channel, min_score)

    @torch.no_grad()
    def forward(self, noisy_batch,SH: torch.Tensor, Rmats: torch.Tensor, tpos: torch.Tensor,
                node_mask: Optional[torch.Tensor]=None,
                topk_map: Optional[torch.Tensor]=None, min_score: Optional[float]=None):
        # SH=apply_l_window(SH)

        tau=0.1
        sh_processed = torch.where(torch.abs(SH) >= tau, SH, torch.zeros_like(SH))


        dens, grid_xyz, cube_shape, sphere_mask = self.sh2dens(SH)
        positions = torch.where(noisy_batch['aatype'] == AA2IDX['ARG'])
        # print(noisy_batch['t'])




        b,n,c=int(positions[0][1]),int(positions[1][1]),1

        pos_bn = noisy_batch['atoms14_local'][b, n]  # [14, 3]
        types_bn = noisy_batch['atom14_element_idx'][b, n]  # [14]

        # 取出类型 == 0 的原子
        mask = (types_bn == c)
        selected_atoms = pos_bn[mask]  # [A, 3]，A是符合条件的原子数


        aatype=AA_LIST[noisy_batch['aatype'][b, n]]

        visualize_density_atoms_3d(

            density=dens[b, n, c],
            grid_xyz=grid_xyz,
            cube_shape=cube_shape,
            sphere_mask=sphere_mask,
            atom_positions=selected_atoms,
            GT_atom_positions=selected_atoms,
            t=1,#noisy_batch['t'][b],
            name=aatype,
            atom_types=None,  # atom_types  types_bn[mask]
            select_type=c,
            mode='pointcloud',  # "pointcloud", isosurface
            percentile=98.5,
            add_colorbar=True,
            point_size=2.0,
            voxel=self.voxel_size
        )




        peaks_xyz_local, peaks_score, peaks_mask = self.peak(dens, cube_shape, grid_xyz, sphere_mask,
                                                             topk_map=topk_map, min_score=min_score)

        # visualize_density_atoms_3d(
        #
        #     density=dens[b, n, c],
        #     grid_xyz=grid_xyz,
        #     cube_shape=cube_shape,
        #     sphere_mask=sphere_mask,
        #     atom_positions=selected_atoms,
        #     GT_atom_positions=selected_atoms,
        #     t=noisy_batch['t'][b],
        #     name=aatype,
        #     atom_types=None,  # atom_types  types_bn[mask]
        #     select_type=c,
        #     mode='pointcloud',  # "pointcloud", isosurface
        #     percentile=98.5,
        #     add_colorbar=True,
        #     point_size=2.0,
        #     voxel=self.voxel_size
        # )



        print(peaks_xyz_local[b,n,c])
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X');
        ax.set_ylabel('Y');
        ax.set_zlabel('Z')

        mask = peaks_mask[b, n, c]
        pts = peaks_xyz_local[b, n, c][mask].cpu().numpy()
        vals = peaks_score[b, n, c][mask].cpu().numpy()
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=vals, cmap='viridis', s=50)

        plt.tight_layout()
        plt.show()

        coords_global = torch.einsum("bnij,bnckj->bncki", Rmats, peaks_xyz_local) + tpos[..., None, None, :]

        from data import utils as du
        rigid=du.create_rigid(Rmats, tpos)
        coords_global2 = rigid[..., None, None].apply(peaks_xyz_local)  # 代替手写 einsum



        if node_mask is not None:
            m = node_mask[..., None, None, None]
            peaks_xyz_local = peaks_xyz_local * m
            coords_global = coords_global * m
            peaks_score = peaks_score * m.squeeze(-1)
            peaks_mask = peaks_mask & m.squeeze(-1)
        return dict(
            coords_local=peaks_xyz_local,
            coords_global=coords_global,
            scores=peaks_score,
            peaks_mask=peaks_mask,
            density=dens,
            grid_xyz=grid_xyz,
            cube_shape=cube_shape,
            sphere_mask=sphere_mask,
        )

# ---------------------------
# Residue priors & Dynamic-K wrapper
# ---------------------------

class DynamicKSidechainDecoder(nn.Module):
    def __init__(self, sh_decoder: SHSidechainDecoder,
                 k_prior: torch.Tensor = AA_K_PRIOR,  # [20,4]
                 entropy_thresh: float = 1.5,
                 inflate_when_uncertain: int = 1,
                 K_min: int = 0,
                 K_max_per_elem: Optional[Tuple[int,int,int,int]] = None,
                 include_backbone: bool = True):
        super().__init__()
        self.dec = sh_decoder
        self.include_backbone = bool(include_backbone)
        if self.include_backbone:
            k_prior = k_prior + BB


        self.register_buffer('k_prior', k_prior.clone())
        self.entropy_thresh = float(entropy_thresh)
        self.inflate = int(inflate_when_uncertain)
        self.K_min = int(K_min)
        self.K_max = torch.tensor(K_max_per_elem if K_max_per_elem is not None else [10,6,6,2], dtype=torch.long)



    def build_K_map(self, B: int, N: int, C: int, device,
                    aatype: Optional[torch.Tensor]=None,
                    aatype_probs: Optional[torch.Tensor]=None) -> torch.Tensor:
        assert C == 4, "Expect C=4 element channels [C,N,O,S]"
        if aatype is not None:
            onehot = F.one_hot(aatype.clamp(0, 19), num_classes=20).to(self.k_prior.dtype)
            # [B,N,20] @ [20,4] -> [B,N,4]
            K = torch.einsum('bna,af->bnf', onehot, self.k_prior.to(onehot.device))
        elif aatype_probs is not None:
            probs = aatype_probs.to(torch.float32)      # [B,N,20]
            K = torch.einsum('bna,af->bnf', probs, self.k_prior.to(probs.device))
            # 不确定性时膨胀一点
            # 熵 H = -sum p log p
            H = -(probs * (probs.clamp_min(1e-9).log())).sum(-1)  # [B,N]
            bump = (H > self.entropy_thresh).to(torch.long)[..., None] * self.inflate
            K = K + bump
        else:
            raise ValueError('Provide either aatype or aatype_probs')
        # add backbone heavy atoms (N, C, O) excluding CA
        K = K.round().long().clamp_min(self.K_min)
        K = torch.minimum(K, self.K_max.to(K.device))
        return K  # [B,N,4]

    @torch.no_grad()
    def forward(self, noisy_batch,SH: torch.Tensor, Rmats: torch.Tensor, tpos: torch.Tensor,
                aatype: Optional[torch.Tensor]=None,
                aatype_probs: Optional[torch.Tensor]=None,
                node_mask: Optional[torch.Tensor]=None,
                min_score: Optional[float]=None):
        B, N, C = SH.shape[:3]
        K_map = self.build_K_map(B, N, C, SH.device, aatype, aatype_probs)  # [B,N,4]
        out = self.dec(noisy_batch,SH, Rmats, tpos, node_mask=node_mask, topk_map=K_map, min_score=min_score)
        return out

if __name__ == '__main__':
    # Smoke test
    B, N, C = 1, 3, 4
    L_max, R_bins = 2, 8
    SH = torch.randn(B, N, C, L_max+1, 2*L_max+1, R_bins)
    Rm = o3.rand_matrix(B, N)
    t = torch.randn(B, N, 3)

    base = SHSidechainDecoder(L_max, R_bins, r_max=6.0, voxel_size=0.4, min_peak_distance=0.6, topk_per_channel=[10, 4, 3, 1] , min_score=None)
    dyn = DynamicKSidechainDecoder(base)
    # 1) 先用 SH 预测 aatype 概率
    type_head = SHTypePredictor(C=4, L_max=L_max, R_bins=R_bins, hidden=256)
    aatype_probs = type_head(SH, node_mask=node_mask)  # [B,N,20]

    # Example 1: using hard aatype labels (AlphaFold order)
    # e.g., [ASP, GLN, ARG] -> indices [3,5,1]
    aatype = torch.tensor([[3,5,1]])
    out = dyn(SH, Rm, t, aatype=aatype)
    print(out['coords_global'].shape, out['peaks_mask'].shape)

    coords = out['coords_global']  # [B,N,4,K,3]
    mask = out['peaks_mask']  # [B,N,4,K]
    score = out['scores']  # [B,N,4,K]
    atom14_xyz, atom14_exists = assemble_atom14(
        coords_global=coords,
        peaks_mask=mask,
        scores=score,  # 也可以用 out['peak_probs']
        aatype_probs=probs,  # 若是 hard label: F.one_hot(aatype,20).float()
        restype_name_to_atom14_names=restype_name_to_atom14_names
    )
    # 现在：atom14_xyz -> [B,N,14,3]，atom14_exists -> [B,N,14]

    # Example 2: using aatype probabilities [B,N,20]
    probs = torch.zeros(B, N, 20)
    probs[0, 0, AA2IDX['ASP']] = 0.6
    probs[0, 0, AA2IDX['ASN']] = 0.4
    probs[0, 1, AA2IDX['GLN']] = 1.0
    probs[0, 2, AA2IDX['TRP']] = 1.0
    out2 = dyn(SH, Rm, t, aatype_probs=probs)
    print(out2['coords_global'].shape, out2['peaks_mask'].shape, out2['peaks_mask'].sum(-1).sum(-1))
