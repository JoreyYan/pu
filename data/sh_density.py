
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
from typing import Optional
from e3nn import o3
import numpy as np
@torch.no_grad()
def _safe_acos(x: torch.Tensor) -> torch.Tensor:
    return torch.acos(x.clamp(-1.0, 1.0))

# ======== Shared utils (统一约定) ========

# 统一：返回 (r, theta, phi)
def cartesian_to_spherical(xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, z = xyz.unbind(-1)
    r = torch.sqrt(x * x + y * y + z * z).clamp_min(1e-12)
    theta = torch.acos((z / r).clamp(-1.0, 1.0))   # [0, π]
    phi = torch.atan2(y, x)                        # (-π, π]
    return r, theta, phi

# 统一：e3nn 实球谐（只依赖方向）
def real_sph_harm_all(L_max: int, points: torch.Tensor) -> torch.Tensor:
    irreps_sh = o3.Irreps.spherical_harmonics(L_max)
    return o3.spherical_harmonics(
        irreps_sh, points, normalize=True, normalization="component"
    )  # [..., sum_{l=0..L}(2l+1)]

def combo_level_from_volume(flat_vals, percentile=99.0, rel_max=0.5, abs_threshold=None):
    """
    flat_vals: ndarray, flatten 后的体素值
    percentile: 分位百分比 (0~100)
    rel_max: 相对最大值比例阈 (0~1)
    abs_threshold: 如果不为 None，则优先使用这个绝对阈值
    """
    if abs_threshold is not None:
        return float(abs_threshold)

    q_val = float(np.quantile(flat_vals, percentile / 100.0))
    r_val = float(flat_vals.max() * rel_max)

    thr = max(q_val, r_val)

    # 避免全是负或 0 时出现空集
    if not np.isfinite(thr) or thr <= 0:
        thr = float(np.quantile(flat_vals, 90 / 100.0))
    return thr

# 统一：按“居中”把扁平 Y 拆到 m 维（支持 batched 前缀）
def split_Y_by_l_centered(Y_flat: torch.Tensor, L_max: int) -> torch.Tensor:
    *prefix, Msum = Y_flat.shape
    Mmax = 2 * L_max + 1
    Y = Y_flat.new_zeros(*prefix, L_max + 1, Mmax)
    off = 0
    for l in range(L_max + 1):
        Ml = 2 * l + 1
        sl = Y_flat[..., off:off + Ml]  # [..., Ml]
        start = (Mmax - Ml) // 2        # = L_max - l
        Y[..., l, start:start + Ml] = sl
        off += Ml
    return Y  # [..., L+1, 2L+1]

# 统一：径向 RBF（高斯，等宽分箱）
def make_radial_rbf(r: torch.Tensor, r_max: float, R_bins: int, sigma: Optional[float]=None) -> torch.Tensor:
    device, dtype = r.device, r.dtype
    edges   = torch.linspace(0.0, float(r_max), R_bins + 1, device=device, dtype=dtype)
    centers = 0.5 * (edges[:-1] + edges[1:])                    # [R]
    if sigma is None:
        sigma = (edges[1] - edges[0]) * 0.5 if R_bins > 1 else max(1e-3, float(r_max) / 4)
    Rbf = torch.exp(-0.5 * ((r[..., None] - centers) / (sigma + 1e-12)) ** 2)  # [..., R]
    return Rbf
def _combo_level_from_volume(vol_np_t, iso_percentile=99.0, rel=0.5, positive_only=True):
    # vol_np_t: 3D numpy
    v = np.maximum(vol_np_t, 0.0) if positive_only else vol_np_t
    vmax = float(v.max())
    if vmax <= 0:
        return np.inf  # 没有正值就不画
    # 分位阈（只在正值上统计更稳）
    if positive_only and (v > 0).any():
        q = float(np.quantile(v[v > 0], iso_percentile / 100.0))
    else:
        q = float(np.quantile(v, iso_percentile / 100.0))
    # 相对峰值阈
    r = rel * vmax
    return max(q, r)


def sh_density_from_atom14_with_masks_e3nn(
    coords: torch.Tensor,               # [B,N,14,3] 旋转/平移已对齐（世界坐标）
    elements_idx: torch.Tensor,         # [B,N,14] in {0:C,1:N,2:O,3:S}
    atom_mask: Optional[torch.Tensor] = None,  # [B,N,14]
    L_max: int = 6,
    R_bins: int = 24,
    r_max: Optional[float] = None,
    sigma_r: float = 0.25,
    per_atom_norm: bool = False,
    l_wise_norm: bool = False,
    eps: float = 1e-8,
):
    """
    返回:
      density:        [B,N,C,L,M,R]      （与解码端一致的 L/M 排布）
      struct_mask:    [1,1,1,L,M,R]      （与数据无关，|m|<=l，沿R全1）
      data_mask_full: [B,N,C,L,M,R]      （|density|>eps）
      data_mask_lm:   [B,N,C,L,M]        （沿R any）
      data_mask_r:    [B,N,C,R]          （沿L,M any）
    """
    B, N, A, _ = coords.shape
    device, dtype = coords.device, coords.dtype

    if atom_mask is None:
        atom_mask = torch.ones(B, N, A, dtype=dtype, device=device)
    else:
        atom_mask = atom_mask.to(dtype)

    # 半径 r（用于径向基）；e3nn 的 SH 用 normalize=True，只依赖方向
    r = torch.linalg.norm(coords, dim=-1)  # [B,N,A]

    if r_max is None:
        r_max_val = max(r.max().item(), 1e-6)
    else:
        r_max_val = float(r_max)

    # 径向 RBF（与原实现一致）
    edges   = torch.linspace(0.0, r_max_val, R_bins + 1, device=device, dtype=dtype)
    centers = 0.5 * (edges[:-1] + edges[1:])                         # [R]
    w_r = torch.exp(-0.5 * ((r.unsqueeze(-1) - centers) ** 2) / (sigma_r ** 2))  # [B,N,A,R]

    # e3nn 实球谐：从笛卡尔坐标直接计算；normalize=True 去除 r^l，纯方向
    irreps_sh = o3.Irreps.spherical_harmonics(L_max)
    # 注意：输入允许为任意长度向量；normalize=True 会做方向归一，避免 r=0 需由 mask 保护
    Y_flat = o3.spherical_harmonics(
        irreps_sh, coords, normalize=True, normalization="component"
    )  # [B,N,A, sum_{l=0..L}(2l+1)]

    # 把扁平分量拆到 [B,N,A,L+1,Mmax]
    Y = _split_Y_by_l_batched(Y_flat, L_max)  # [B,N,A,L,M]

    # 原子类型 one-hot（4 通道：C/N/O/S）
    C = 4
    safe_elem_idx = elements_idx.clone()
    safe_elem_idx[safe_elem_idx >= C] = 0
    elem_oh = torch.nn.functional.one_hot(
        safe_elem_idx.clamp(0, C - 1).long(), num_classes=C
    ).to(dtype)  # [B,N,A,C]

    # 应用 mask 去掉 padding
    a_mask = atom_mask.unsqueeze(-1)               # [B,N,A,1]
    elem_oh = elem_oh * a_mask                     # [B,N,A,C]
    Y       = Y * a_mask.unsqueeze(-1)             # [B,N,A,L,M]
    w_r     = w_r * atom_mask.unsqueeze(-1)        # [B,N,A,R]

    # 汇聚：elem × Y × RBF -> [B,N,C,L,M,R]
    density = torch.einsum('bnac,bnalm,bnar->bnclmr', elem_oh, Y, w_r)

    # 可选归一化
    if per_atom_norm:
        counts = torch.einsum('bnac->bnc', elem_oh).clamp_min(eps)  # [B,N,C]
        density = density / counts.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    if l_wise_norm:
        l2 = torch.sqrt((density**2).sum(dim=(-1, -2), keepdim=True)).clamp_min(eps)  # [B,N,C,L,1,1]
        density = density / l2

    # 结构 mask（与 R 无关，沿 R 取 1）——用你现有的实现
    struct_mask = sh_structure_mask(L_max, R_bins, device=device, dtype=dtype)  # [1,1,1,L,M,R]

    # 数据 mask
    data_mask_full = (density.abs() > eps).to(dtype)                 # [B,N,C,L,M,R]
    data_mask_lm   = (data_mask_full.sum(dim=-1) > 0).to(dtype)      # [B,N,C,L,M]
    data_mask_r    = (data_mask_full.sum(dim=(-3, -2)) > 0).to(dtype)  # [B,N,C,R]

    return density, struct_mask, data_mask_full, data_mask_lm, data_mask_r



def real_spherical_harmonics(L_max: int, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    # outputs [..., L, M] with M=2*L_max+1
    x = torch.cos(theta)
    *prefix, = x.shape
    P = x.new_zeros(*prefix, L_max+1, L_max+1)
    P[..., 0, 0] = 1.0
    if L_max > 0:
        one_minus_x2 = (1.0 - x*x).clamp_min(0.0)
        fact = x.new_ones(())
        for m in range(1, L_max+1):
            fact = fact * (-(2*m - 1))
            P[..., m, m] = fact * one_minus_x2.pow(m * 0.5)
        for m in range(0, L_max):
            P[..., m+1, m] = (2*m + 1) * x * P[..., m, m]
        for m in range(0, L_max+1):
            for l in range(m+2, L_max+1):
                P[..., l, m] = ((2*l - 1) * x * P[..., l-1, m] - (l - 1 + m) * P[..., l-2, m]) / (l - m)

    Mdim = 2*L_max + 1
    Y = x.new_zeros(*prefix, L_max+1, Mdim)

    def _log_fact_ratio(a: int, b: int) -> float:
        s = 0.0
        for k in range(b+1, a+1):
            s += math.log(k)
        return s

    fourpi = 4.0 * math.pi
    for l in range(L_max+1):
        N_l0 = math.sqrt((2*l + 1)/fourpi)
        Y[..., l, L_max+0] = N_l0 * P[..., l, 0]
        for m in range(1, l+1):
            log_ratio = _log_fact_ratio(l - m, l + m) if (l+m) > 0 else 0.0
            N_lm = math.sqrt((2*l + 1)/fourpi * math.exp(log_ratio))
            common = N_lm * P[..., l, m]
            cm = torch.cos(m * phi)
            sm = torch.sin(m * phi)
            Y[..., l, L_max + m]  = math.sqrt(2.0) * common * cm
            Y[..., l, L_max - m]  = math.sqrt(2.0) * common * sm
    return Y

def sh_structure_mask(L_max: int, R_bins: int, device=None, dtype=torch.float32):
    """
    结构性合法位 (|m|<=l) 的 mask，沿 R 维度为全 1，用于广播：
    shape = [1,1,1, L, M, R]
    """
    L = L_max + 1
    M = 2*L_max + 1
    base = torch.zeros(L, M, dtype=dtype, device=device)
    for l in range(L):
        for m in range(-l, l+1):
            base[l, m + L_max] = 1.0
    struct = base.unsqueeze(-1).expand(L, M, R_bins)      # [L,M,R]
    return struct.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,L,M,R]

def sh_density_from_atom14_with_masks(
    coords: torch.Tensor,               # [B,N,14,3]
    elements_idx: torch.Tensor,         # [B,N,14] in {0:C,1:N,2:O,3:S}
    atom_mask: Optional[torch.Tensor] = None,  # [B,N,14]
    L_max: int = 8,
    R_bins: int = 24,
    r_max: Optional[float] = None,
    sigma_r: float = 0.25,
    per_atom_norm: bool = False,
    l_wise_norm: bool = False,
    eps: float = 1e-4,
):
    if not torch.isfinite(coords).all():
        raise ValueError("coords contains NaN/Inf")
    coords = torch.clamp(coords, -200.0, 200.0)

    B, N, A, _ = coords.shape
    device, dtype = coords.device, coords.dtype
    if atom_mask is None:
        atom_mask = torch.ones(B, N, A, dtype=dtype, device=device)
    else:
        atom_mask = atom_mask.to(dtype)

    # 半径 → 径向基
    r, theta, phi = cartesian_to_spherical(coords)  # 统一接口（theta/phi 这里不直接用）
    r_max_val = float(max(r.max().item(), 1e-6)) if r_max is None else float(r_max)
    r = torch.clamp(r, min=1e-4)
    w_r = make_radial_rbf(r, r_max_val, R_bins, sigma=sigma_r)        # [B,N,A,R]
    w_r = torch.nan_to_num(w_r, nan=0.0, posinf=0.0, neginf=0.0)

    # e3nn 实球谐（笛卡尔→方向）；然后“居中”拆 m
    Y_flat = real_sph_harm_all(L_max, coords)                         # [B,N,A, sum(2l+1)]
    Y = split_Y_by_l_centered(Y_flat, L_max)                          # [B,N,A,L+1,2L+1]
    Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    # 元素 one-hot（4 通道：C/N/O/S；如需 UNK，可改成 5 并同步解码端）
    C = 4
    safe_elem_idx = elements_idx.clamp(0, C - 1)
    elem_oh = torch.nn.functional.one_hot(safe_elem_idx.long(), num_classes=C).to(dtype)  # [B,N,A,C]

    # mask padding
    a_mask = atom_mask.unsqueeze(-1)             # [B,N,A,1]
    elem_oh = elem_oh * a_mask                   # [B,N,A,C]
    Y       = Y * a_mask.unsqueeze(-1)           # [B,N,A,L,M]
    w_r     = w_r * atom_mask.unsqueeze(elements_idx-1)      # [B,N,A,R]

    # w_max = w_r.max().item()
    # Y_max = Y.abs().amax(dim=(0, 1, 2)).cpu()  # -> [L+1, 2L+1]
    # poa = w_r.sum(dim=-1).mean().item()  # 理想≈1；若远>1，就会把平均值抬高
    # counts = elem_oh.sum(dim=2)  # [B,N,C]
    # counts_minmax = (counts.min().item(), counts.max().item())
    # print(f'w_max = {w_max}, Y_max = {Y_max}, poa = {poa}, counts_minmax = {counts_minmax}')

    # 一次性计算所有的
    # 聚合 → 系数
    density = torch.einsum('bnac,bnalm,bnar->bnclmr', elem_oh, Y, w_r)  # [B,N,C,L+1,2L+1,R]
    density = torch.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)

    if per_atom_norm:
        counts = elem_oh.sum(dim=2).clamp_min(eps)                       # [B,N,C]
        density = density / counts[..., None, None, None]
    if l_wise_norm:
        l2 = torch.sqrt((density**2).sum(dim=(-1, -2), keepdim=True)).clamp_min(eps)  # [B,N,C,L,1,1]
        density = density / l2

    # 结构/数据 mask（维度与解码一致）
    struct_mask = torch.zeros(1,1,1,L_max+1,2*L_max+1,R_bins, device=device, dtype=dtype)
    for l in range(L_max+1):
        Ml = 2*l+1
        start = (2*L_max+1 - Ml)//2
        struct_mask[..., l, start:start+Ml, :] = 1
    data_mask_full = (density.abs() > eps).to(dtype)
    data_mask_lm   = (data_mask_full.sum(dim=-1) > 0).to(dtype)
    data_mask_r    = (data_mask_full.sum(dim=(-3, -2)) > 0).to(dtype)

    density = torch.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    struct_mask = torch.nan_to_num(struct_mask, nan=0.0, posinf=1.0, neginf=0.0)
    data_mask_full = torch.nan_to_num(data_mask_full, nan=0.0, posinf=1.0, neginf=0.0)
    data_mask_lm = torch.nan_to_num(data_mask_lm, nan=0.0, posinf=1.0, neginf=0.0)
    data_mask_r = torch.nan_to_num(data_mask_r, nan=0.0, posinf=1.0, neginf=0.0)

    return density, struct_mask, data_mask_full, data_mask_lm, data_mask_r


def sh_density_from_atom14_with_mliasks_clean(
        coords: torch.Tensor,  # [B,N,14,3]
        elements_idx: torch.Tensor,  # [B,N,14] in {0:C,1:N,2:O,3:S}
        atom_mask: Optional[torch.Tensor] = None,  # [B,N,14]
        L_max: int = 8,
        R_bins: int = 24,
        r_max: Optional[float] = None,
        sigma_r: float = 0.25,
        per_atom_norm: bool = False,
        l_wise_norm: bool = False,
        eps: float = 1e-4,
):
    coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
    coords = torch.clamp(coords, -200.0, 200.0)

    B, N, A, _ = coords.shape
    device, dtype = coords.device, coords.dtype
    if atom_mask is None:
        atom_mask = torch.ones(B, N, A, dtype=dtype, device=device)
    else:
        atom_mask = torch.nan_to_num(atom_mask, nan=0.0, posinf=1.0, neginf=0.0).to(dtype)

    # 半径 → 径向基
    r, theta, phi = cartesian_to_spherical(coords)  # 统一接口（theta/phi 这里不直接用）
    r_max_val = float(max(r.max().item(), 1e-6)) if r_max is None else float(r_max)
    r = torch.clamp(r, min=1e-4)
    w_r = make_radial_rbf(r, r_max_val, R_bins, sigma=sigma_r)  # [B,N,A,R]
    w_r = torch.nan_to_num(w_r, nan=0.0, posinf=0.0, neginf=0.0)

    # e3nn 实球谐（笛卡尔→方向）；然后“居中”拆 m
    Y_flat = real_sph_harm_all(L_max, coords)  # [B,N,A, sum(2l+1)]
    Y = split_Y_by_l_centered(Y_flat, L_max)  # [B,N,A,L+1,2L+1]
    Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    # 元素 one-hot（4 通道：C/N/O/S；如需 UNK，可改成 5 并同步解码端）
    C = 4
    safe_elem_idx = torch.nan_to_num(elements_idx, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, C - 1)
    elem_oh = torch.nn.functional.one_hot(safe_elem_idx.long(), num_classes=C).to(dtype)  # [B,N,A,C]

    # mask padding
    a_mask = atom_mask.unsqueeze(-1)  # [B,N,A,1]
    elem_oh = elem_oh * a_mask  # [B,N,A,C]
    Y = Y * a_mask.unsqueeze(-1)  # [B,N,A,L,M]
    w_r = w_r * atom_mask.unsqueeze(-1)  # [B,N,A,R]

    # w_max = w_r.max().item()
    # Y_max = Y.abs().amax(dim=(0, 1, 2)).cpu()  # -> [L+1, 2L+1]
    # poa = w_r.sum(dim=-1).mean().item()  # 理想≈1；若远>1，就会把平均值抬高
    # counts = elem_oh.sum(dim=2)  # [B,N,C]
    # counts_minmax = (counts.min().item(), counts.max().item())
    # print(f'w_max = {w_max}, Y_max = {Y_max}, poa = {poa}, counts_minmax = {counts_minmax}')

    # 一次性计算所有的
    # 聚合 → 系数
    density = torch.einsum('bnac,bnalm,bnar->bnclmr', elem_oh, Y, w_r)  # [B,N,C,L+1,2L+1,R]
    density = torch.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)

    if per_atom_norm:
        counts = elem_oh.sum(dim=2).clamp_min(eps)  # [B,N,C]
        density = density / counts[..., None, None, None]
    if l_wise_norm:
        l2 = torch.sqrt((density ** 2).sum(dim=(-1, -2), keepdim=True)).clamp_min(eps)  # [B,N,C,L,1,1]
        density = density / l2

    # 结构/数据 mask（维度与解码一致）
    struct_mask = torch.zeros(1, 1, 1, L_max + 1, 2 * L_max + 1, R_bins, device=device, dtype=dtype)
    for l in range(L_max + 1):
        Ml = 2 * l + 1
        start = (2 * L_max + 1 - Ml) // 2
        struct_mask[..., l, start:start + Ml, :] = 1
    data_mask_full = (density.abs() > eps).to(dtype)
    data_mask_lm = (data_mask_full.sum(dim=-1) > 0).to(dtype)
    data_mask_r = (data_mask_full.sum(dim=(-3, -2)) > 0).to(dtype)

    return density, struct_mask, data_mask_full, data_mask_lm, data_mask_r
def l_window(L_max, kind="jackson", alpha=2.0, device="cpu", dtype=torch.float32):
    L = L_max
    if kind == "fejer":
        w = torch.tensor([1 - l/(L+1) for l in range(L+1)], device=device, dtype=dtype)
    elif kind == "exp":
        w = torch.exp(-alpha * torch.arange(L+1, device=device, dtype=dtype) / max(L,1))
    elif kind == "jackson":
        # 近似 Jackson 窗（平滑、振铃更小）
        import math
        w = []
        for l in range(L+1):
            x = math.pi * l / (L+1)
            w.append(((L - l + 1) * math.cos(x) + math.sin(x)/math.tan(math.pi/(L+1))) / (L + 1))
        w = torch.tensor(w, device=device, dtype=dtype)
        w = (w - w.min()) / (w.max() - w.min() + 1e-12)
    else:
        raise ValueError
    return w

def apply_l_window(coeffs, kind="jackson", alpha=2.0):
    # coeffs: [B,N,C,L+1,2L+1,R]
    B,N,C,Lp1,M,R = coeffs.shape
    w = l_window(Lp1-1, kind=kind, alpha=alpha, device=coeffs.device, dtype=coeffs.dtype)
    return coeffs * w.view(1,1,1,Lp1,1,1)


class SHToGridDensity(torch.nn.Module):
    def __init__(self, L_max: int, R_bins: int, r_max: float, voxel_size: float):
        super().__init__()
        self.L_max = int(L_max)
        self.R_bins = int(R_bins)
        self.r_max = float(r_max)
        self.voxel = float(voxel_size)
        self._cache = {}

    def _precompute(self, device):
        key = (device, self.L_max, self.R_bins, self.r_max, self.voxel)
        if key in self._cache:
            return self._cache[key]

        # 规则立方网格（世界坐标）
        n = int(math.floor((2 * self.r_max) / self.voxel)) + 1
        xs = torch.linspace(-self.r_max, self.r_max, n, device=device)
        X, Y, Z = torch.meshgrid(xs, xs, xs, indexing="ij")
        grid_xyz = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # [G,3]
        cube_shape = (n, n, n)

        # 球谐（e3nn）→ 居中拆 m
        Y_flat = real_sph_harm_all(self.L_max, grid_xyz)                  # [G, sum(2l+1)]
        Y_lmg  = split_Y_by_l_centered(Y_flat, self.L_max).permute(-2, -1, 0).contiguous()
        # Y_lmg: [L+1, 2L+1, G]

        # 径向基（用 r）
        r, _, _ = cartesian_to_spherical(grid_xyz)                         # [G]
        Rbf = make_radial_rbf(r, self.r_max, self.R_bins)                 # [G, R]
        Rbf = Rbf.transpose(0, 1).contiguous()                             # [R, G]

        sphere_mask = (grid_xyz.norm(dim=-1) <= self.r_max)                # [G]

        self._cache[key] = (grid_xyz, cube_shape, Y_lmg, Rbf, sphere_mask)
        return self._cache[key]

    def forward(self, coeffs: torch.Tensor):
        # coeffs: [B,N,C,L+1, 2L+1, R]
        grid_xyz, cube_shape, Y_lmg, Rbf, sphere_mask = self._precompute(coeffs.device)
        density = torch.einsum("bnclmr,lmg,rg->bncg", coeffs, Y_lmg, Rbf)  # [B,N,C,G]
        return density.to(dtype=torch.float16), grid_xyz, cube_shape, sphere_mask


import matplotlib.pyplot as plt
# 等值面可选依赖
try:
    from skimage.measure import marching_cubes
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False

def visualize_density_atoms_3d(

    density: torch.Tensor,        # [G]
    grid_xyz: torch.Tensor,       # [G, 3]
    cube_shape: tuple,            # (Dx, Dy, Dz)
    sphere_mask: torch.Tensor,    # [G] bool
    atom_positions: torch.Tensor, # [A, 3]
    GT_atom_positions: torch.Tensor=None,
    atom_types: torch.Tensor=None,# [A]
        t: float = 0,
        name: str = "",
    select_type: int=0,           # 只显示该类型的原子（若 atom_types=None 则忽略）
    normalize: bool=True,
    mode: str="isosurface",       # "pointcloud" 或 "isosurface"
    percentile: float=97.5,       # pointcloud：按分位数取高密度点
    abs_threshold: float=None,    # pointcloud：绝对阈值（优先于 percentile）
    max_points: int=200000,       # pointcloud：最多点数
    add_colorbar: bool=True,      # pointcloud：是否加颜色条
    point_size: float=2.0,        # pointcloud：点大小
    iso_percentile: float=99.9,   # isosurface：分位阈值
    voxel: float=1.0,             # ★ 体素物理尺寸（与 make_cube_grid 保持一致）
    axes_order: tuple=(0,1,2)     # ★ 若 reshape 次序非 (Dx,Dy,Dz)，在这里指定，如 (0,2,1)
):
    Dx, Dy, Dz = cube_shape
    G = grid_xyz.shape[0]
    assert density.numel() == G and grid_xyz.shape[-1] == 3

    # 掩膜+归一化
    vol = density
    if sphere_mask is not None:
        vol = vol * sphere_mask.to(vol.dtype)
    vol_np = vol.detach().cpu().reshape(Dx, Dy, Dz).numpy()

    if normalize:
        vmin, vmax = float(vol_np.min()), float(vol_np.max())
        if vmax > vmin:
            vol_np = (vol_np - vmin) / (vmax - vmin + 1e-12)

    # 原子筛选
    atoms_np = atom_positions.detach().cpu().numpy()
    if atom_types is not None:
        atypes_np = atom_types.detach().cpu().numpy()
        atoms_np = atoms_np[atypes_np == select_type]

    gt_atoms_np = GT_atom_positions.detach().cpu().numpy()
    if atom_types is not None:
        atypes_np = atom_types.detach().cpu().numpy()
        gt_atoms_np = gt_atoms_np[atypes_np == select_type]


    # 世界坐标范围（用于对齐与设定坐标轴范围）
    gx = grid_xyz.detach().cpu().numpy().reshape(-1, 3)
    xyz_min = gx.min(axis=0)
    xyz_max = gx.max(axis=0)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    if mode == "isosurface" and SKIMAGE_OK and vol_np.max() > 0:
        # ★ 轴顺序对齐 marching_cubes 数据
        vol_np_t = np.transpose(vol_np, axes_order)

        # 等值面阈值
        # level = float(np.quantile(vol_np_t, iso_percentile / 100.0))

        # 改成：只看正值

        # vol_pos = np.maximum(vol_np_t, 0.0)
        # if (vol_pos > 0).any():
        #     level = float(np.quantile(vol_pos[vol_pos > 0], iso_percentile / 100.0))
        # else:
        #     level = float(np.quantile(vol_np_t, iso_percentile / 100.0))  # 兜底

        level = _combo_level_from_volume(vol_np_t, iso_percentile=iso_percentile, rel=0.5, positive_only=True)

        level = max(level, 1e-8)

        # marching_cubes 在索引坐标系（步长=1，原点=0）上返回顶点
        verts, faces, normals, values = marching_cubes(
            vol_np_t, level=level, spacing=(1.0, 1.0, 1.0)
        )

        # ★ 把顶点从索引坐标 → 世界坐标：
        # 1) 轴顺序还原回 (x,y,z)
        inv = np.argsort(axes_order)
        verts = verts[:, inv]
        # 2) 缩放：乘体素物理尺寸
        verts = verts * voxel
        # 3) 平移：加上网格原点（与 grid_xyz 对齐）
        verts = verts + xyz_min[None, :]

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        mesh = Poly3DCollection(verts[faces], alpha=0.25, linewidths=0.2)
        ax.add_collection3d(mesh)

        # ★ 用世界坐标范围设定轴限，确保与原子点一致
        ax.set_xlim(xyz_min[0], xyz_max[0])
        ax.set_ylim(xyz_min[1], xyz_max[1])
        ax.set_zlim(xyz_min[2], xyz_max[2])

    else:
        # 点云：取阈值以上的体素中心点，并用密度值做颜色映射
        flat = vol_np.reshape(-1)
        # if abs_threshold is not None:
        #     thr = float(abs_threshold)
        # else:
        #     thr = float(np.quantile(flat, percentile / 100.0))

        # 组合阈值策略
        thr = combo_level_from_volume(flat, percentile=percentile, rel_max=0.5, abs_threshold=abs_threshold)

        mask = flat >= thr
        if mask.sum() == 0:
            thr = float(np.quantile(flat, 90 / 100.0))
            mask = flat >= thr

        idx = np.nonzero(mask)[0]
        if idx.size > max_points:
            idx = np.random.choice(idx, size=max_points, replace=False)

        pts = gx[idx]                  # 世界坐标点
        vals = flat[idx]               # 颜色 = 密度
        sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=point_size, c=vals)
        if add_colorbar:
            fig.colorbar(sc, ax=ax, shrink=0.6, label='density')

        ax.set_xlim(xyz_min[0], xyz_max[0])
        ax.set_ylim(xyz_min[1], xyz_max[1])
        ax.set_zlim(xyz_min[2], xyz_max[2])

    # 叠加原子（世界坐标）
    if atoms_np.size > 0:
        ax.scatter(atoms_np[:,0], atoms_np[:,1], atoms_np[:,2], s=50, c='red', marker='o', alpha=0.5)
    if gt_atoms_np.size > 0:
        ax.scatter(gt_atoms_np[:,0], gt_atoms_np[:,1], gt_atoms_np[:,2], s=50, c='blue', marker='x', alpha=0.5)

    ax.set_title('3D density vs atoms (ELEMENT={}, SELECT_TYPE={},t={:.3f})'.format(['C','N','O','S'][select_type],name,t))
    try:
        ax.set_box_aspect(xyz_max - xyz_min)  # 等比例坐标，避免视觉拉伸
    except Exception:
        pass
    plt.tight_layout()
    plt.show()


import torch, math

def l_taper(L_max, l_pass: int, l_stop: int):
    """余弦渐变窗：l<=l_pass:1；l>=l_stop:0；中间平滑过渡"""
    assert 0 <= l_pass <= l_stop <= L_max
    w = torch.ones(L_max+1, dtype=torch.float32)
    if l_stop > l_pass:
        l = torch.arange(l_pass, l_stop+1, dtype=torch.float32)
        t = (l - l_pass) / max(1e-6, (l_stop - l_pass))
        w[l.long()] = 0.5 * (1 + torch.cos(math.pi * t))  # 1→0 平滑
    if l_stop < L_max:
        w[l_stop+1:] = 0.0
    return w

def apply_l_taper(coeffs, l_pass=3, l_stop=7, preserve_l2=True):
    """coeffs: [B,N,C,L+1,2L+1,R]"""
    B,N,C,Lp1,M,R = coeffs.shape
    L_max = Lp1 - 1
    w = l_taper(L_max, l_pass, l_stop).to(coeffs.device, coeffs.dtype)  # [L+1]
    out = coeffs * w.view(1,1,1,Lp1,1,1)

    # （可选）能量补偿，避免整体幅度被压得太低或太高
    if preserve_l2:
        # 以各 l 的 (2l+1) 加权近似能量，补偿窗带来的能量变化
        num = sum((2*l+1) for l in range(L_max+1))
        den = sum((2*l+1)*(float(w[l])**2) for l in range(L_max+1))
        gain = math.sqrt(num / max(den, 1e-12))
        out = out * gain
    return out

if __name__ == '__main__':
    # 1) m 维对齐自检：编码/解码的 Y 是否只是“居中平移”
    G = 7
    pts = torch.randn(G, 3)
    # Yf = real_sph_harm_all(L_max=4, points=pts)
    # Yc = split_Y_by_l_centered(Yf, L_max=4)  # [..., L+1, 2L+1]
    # 旧的 left-pad 版本（若还有）：应当只差一个平移
    # assert torch.allclose(Yc[..., l, start:start+2*l+1], Y_left[..., l, :2*l+1])

    # 2) 端到端：单原子→编码→解码→查看原子处密度是否为峰
    #coords = torch.tensor([[[[0.0, 0.0, 0.0]]]])  # [B=1,N=1,A=1,3]
    elem = torch.tensor([[[0]]])  # C
    mask = torch.tensor([[[1.0]]])
    coef, *_ = sh_density_from_atom14_with_masks(coords, elem, mask, L_max=4, R_bins=12, r_max=3.0, sigma_r=0.3)
    decoder = SHToDensity(L_max=4, R_bins=12, r_max=3.0, voxel_size=0.2)
    dens, grid_xyz, cube_shape, sphere_mask = decoder(coef)
    # 把原子落到网格索引，查看 dens 是否在其附近取最大
    visualize_density_atoms_3d(
                density=dens[0, 0, 0],
                grid_xyz=grid_xyz,
                cube_shape=cube_shape,
                sphere_mask=sphere_mask,
                atom_positions=pts,
                atom_types=elem,  # atom_types
                select_type=0,
                mode="isosurface",
                percentile=98.5,
                add_colorbar=True,
                point_size=2.0,
                voxel=0.25

            )
