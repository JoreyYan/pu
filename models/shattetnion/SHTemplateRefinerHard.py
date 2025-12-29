
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.shattetnion.SHTemplateRefiner import build_atom14_template_tensor_from_spec,SHResEmbed,AA_ORDER
import torch
from collections import deque, defaultdict

import torch

def take_along_atoms(X, idx):
    """
    X:   [B, N, 14, 3]
    idx: [B, N] 或 [B, N, K]（K可为1）
    out: [B, N, K, 3]
    """
    if idx.dim() == 2:
        idx = idx.unsqueeze(-1)          # -> [B,N,1]
    idx = idx.clamp_min(0).long()        # -1 用 0 占位，真正有效性用 mask 控
    return torch.gather(
        X, dim=2,
        index=idx[..., None].expand(*idx.shape, X.size(-1))  # [B,N,K,3]
    )


def safe_scatter_along_atoms(X, idx, value, write_mask=None):
    """
    无副作用写回：
      X:         [B, N, 14, 3]
      idx:       [B, N, K]
      value:     [B, N, K, 3]
      write_mask:[B, N, K, 1] (bool/float)，可选；不提供则全部写
    返回: 新张量 newX（不改原 X）
    """
    if idx.dim() == 2:
        idx = idx.unsqueeze(-1)
    idx = idx.clamp_min(0).long()

    if write_mask is None:
        write_mask = torch.ones_like(value[..., :1])
    else:
        if write_mask.dim() == 3:
            write_mask = write_mask.unsqueeze(-1)
        write_mask = write_mask.to(dtype=value.dtype)

    # 1) 把 value 和 mask “散射”到与 X 同形的更新/掩码张量（对零张量做 in-place OK，不参与梯度）
    upd = torch.zeros_like(X).scatter(  # 注意：用 .scatter(…)，不是 .scatter_()
        2, idx[..., None].expand(*idx.shape, X.size(-1)), value * write_mask
    )
    m = torch.zeros_like(X[..., :1]).scatter(            # [B,N,14,1]
        2, idx[..., None], write_mask
    )
    # 2) 合成：被选中的位置用 upd，其余保留 X
    newX = X * (1.0 - m) + upd
    return newX

def rodrigues4D(V, axis, ang, eps=1e-8):
    # V:    [B,N,E,3]
    # axis: [B,N,1,3]
    # ang:  [B,N,1]
    axis = axis / (torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(eps))
    cos = torch.cos(ang)[..., None]   # [B,N,1,1]
    sin = torch.sin(ang)[..., None]
    ax  = axis                        # [B,N,1,3]
    cross = torch.cross(ax.expand_as(V), V, dim=-1)
    dot   = (V * ax).sum(-1, keepdim=True)
    return V * cos + cross * sin + ax * dot * (1.0 - cos)
def gather_meta_by_residue(meta_packed: dict, top_idx: torch.Tensor):
    """
    meta_packed: 预打包好的 AA→张量 字典：
      - 'tors_axis'     : [20, Gmax, 2]
      - 'aff_idx'       : [20, Gmax, Emax]
      - 'aff_mask'      : [20, Gmax, Emax] (bool)
      - 'G_counts'      : [20]
      - 'scale_mask'    : [20, 14] (bool)
      - 'scale_parent'  : [20, 14]
      - 以及标量 'Gmax', 'Emax'（直接原样返回）

    top_idx: [B, N]，每个残基对应的 AA 下标（0..19）

    返回: 同名字典，但所有张量都变成按残基展开的：
      - 'tors_axis'   -> [B, N, Gmax, 2]
      - 'aff_idx'     -> [B, N, Gmax, Emax]
      - 'aff_mask'    -> [B, N, Gmax, Emax]
      - 'G_counts'    -> [B, N]
      - 'scale_mask'  -> [B, N, 14]
      - 'scale_parent'-> [B, N, 14]
    """
    assert top_idx.dtype == torch.long
    out = {}
    for k, v in meta_packed.items():
        if k in ("Gmax", "Emax"):          # 标量直接带过去
            out[k] = v
        elif isinstance(v, torch.Tensor):
            # v 的第0维必须是 20（AA 维）
            assert v.size(0) == 20, f"{k} expects leading dim 20, got {v.size(0)}"
            # 高级索引：用 [B,N] 的 top_idx 去索引第0维 → 结果 [B,N,...]
            out[k] = v[top_idx]
        else:
            out[k] = v
    return out
def fk_vectorized(
    X, exists, dchi, sraw, top_idx, meta_packed, scale_clamp=0.1
):
    """
    X:       [B,N,14,3]  软模板起点（局部）
    exists:  [B,N,14]    bool
    dchi:    [B,N,Ghead] 角度增量（>=Gmax时会截取前Gmax）
    sraw:    [B,N,14]    键长缩放原始值
    top_idx: [B,N]       每个残基的 AA 下标（0..19）
    meta_packed: 预打包的 AA 元数据（含 Gmax/Emax）
    """
    B,N = top_idx.shape
    device = X.device
    Gmax = int(meta_packed["Gmax"])
    Emax = int(meta_packed["Emax"])

    # 1) 把AA维的元数据 gather 到每个残基
    # M = gather_meta_by_residue(meta_packed, top_idx)
    # M["tors_axis"]:[B,N,Gmax,2], M["aff_idx"]:[B,N,Gmax,Emax], M["aff_mask"]:[B,N,Gmax,Emax]
    # M["G_counts"]:[B,N], M["scale_mask"]:[B,N,14], M["scale_parent"]:[B,N,14]

    M=meta_packed
    # 有效组掩码（有的AA没χ3/χ4）
    g_mask = (torch.arange(Gmax, device=device)[None,None,:] < M["G_counts"][..., None])  # [B,N,Gmax]

    # 2) 逐组 χ 扭转（必须顺序）
    for gi in range(Gmax):
        valid_res = g_mask[..., gi]  # [B,N] bool
        if not bool(valid_res.any()):
            continue

        # 该组的轴 u->v
        u = M["tors_axis"][..., gi, 0]   # [B,N]
        v = M["tors_axis"][..., gi, 1]
        # 无效轴（u或v=-1）过滤
        axis_ok = (u >= 0) & (v >= 0) & valid_res

        if not bool(axis_ok.any()):
            continue

        # 取轴端点坐标
        Pu = take_along_atoms(X, u.clamp_min(0))   # [B,N,1,3]
        Pv = take_along_atoms(X, v.clamp_min(0))   # [B,N,1,3]
        axis = (Pv - Pu)                           # [B,N,1,3]

        # 该组角度（对无效残基设 0）
        ang = torch.zeros(B, N, 1, device=device, dtype=X.dtype)
        # dchi可能比Gmax多，截取第 gi 组；无效残基置 0
        ang[..., 0] = torch.where(axis_ok, dchi[..., gi], torch.zeros_like(dchi[..., gi]))

        # 受影响原子索引（填充到 Emax），用 mask 过滤
        cols = M["aff_idx"][..., gi, :]     # [B,N,Emax]
        cmask= M["aff_mask"][..., gi, :]    # [B,N,Emax]
        if not bool(cmask.any()):
            continue

        Xa  = take_along_atoms(X, cols.clamp_min(0))       # [B,N,Emax,3]
        Va  = Xa - Pu                                      # [B,N,Emax,3]
        VaR = rodrigues4D(Va, axis, ang)                   # [B,N,Emax,3]

        Xa_new = Pu + VaR                                  # [B,N,Emax,3]

        # 仅更新“存在”的原子：exists gather 并与 cmask 结合
        ex_cols = torch.gather(exists, dim=2, index=cols.clamp_min(0))  # [B,N,Emax]
        write_mask = (cmask & ex_cols).unsqueeze(-1)                    # [B,N,Emax,1]
        Xa_out = torch.where(write_mask, Xa_new, Xa)                    # [B,N,Emax,3]

        # 写回
        safe_scatter_along_atoms(X, cols.clamp_min(0), Xa_out)

    # 3) 键长缩放（完全向量化）
    scale_mask   = M["scale_mask"]         # [B,N,14] bool
    scale_parent = M["scale_parent"].clamp_min(0)  # [B,N,14]

    # 目标原子集合 S：mask 为 True 的位置
    S_mask = scale_mask
    if bool(S_mask.any()):
        # 父坐标、子坐标
        Pa = X                                  # [B,N,14,3]
        Pp = torch.gather(
            X, dim=2,
            index=scale_parent[..., None].expand(B,N,14,3)
        )
        v  = Pa - Pp
        r  = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-8)
        dir= v / r
        s_eff = torch.clamp(sraw, -1.0, 1.0) * scale_clamp   # [B,N,14]
        sf = 1.0 + s_eff[..., None]                          # [B,N,14,1]

        Pa_new = Pp + dir * r * sf                           # [B,N,14,3]

        # 只在（该原子存在 且 父存在 且 scale_mask=True）的位置更新
        pair_exist = exists & torch.gather(exists, dim=2, index=scale_parent)
        W = (S_mask & pair_exist).unsqueeze(-1)              # [B,N,14,1]
        X = torch.where(W, Pa_new, Pa)                       # 全向量化替换

    return X

# 建 meta：用 residue_constants (OpenFold/AF2) 的常量
def build_residue_meta_dict_from_rc(rc, allow_backbone_torsion: bool = False):
    """
    返回: dict[AA3] -> {
        'parent': Long[14],
        'torsion_axis_uv': Long[G,2],
        'affected_atoms': List[LongTensor],
        'scale_mask': Bool[14],
        'scale_parent': Long[14],
        # 便于调试：
        'ring_edges': set[(i,j)],
        'name2slot': Dict[str,int],
        'slot_names': List[str|None],
    }
    """
    # 取每个残基 atom14 槽位名（'' 表示无）
    restype14 = rc.restype_name_to_atom14_names  # 每 AA 的 14 槽位名
    chi_atoms = rc.chi_angles_atoms              # 每 AA 的 χ 定义: [[a,b,c,d], ..]
    res3_list = [rc.restype_1to3[x] for x in rc.restypes]  # 20 个三字母

    # 读取化学键表（实键、非虚拟），用于建立邻接与判环
    try:
        residue_bonds, _, _ = rc.load_stereo_chemical_props()
    except Exception:
        residue_bonds = None

    meta = {}
    backbone_names = {"N", "CA", "C", "O"}

    for aa in res3_list:
        slot_names = restype14[aa]                        # 长度 14 的名字列表（''=无）
        name2slot = {n:i for i,n in enumerate(slot_names) if n}

        # 存在掩码
        A = 14
        exists = torch.tensor([bool(n) for n in slot_names], dtype=torch.bool)

        # 1) 构建无向边集（优先用化学键表；没有就用 χ 角里的连续对 + 骨架）
        edges = set()
        if residue_bonds is not None and aa in residue_bonds:
            for b in residue_bonds[aa]:
                if b.atom1_name in name2slot and b.atom2_name in name2slot:
                    i, j = name2slot[b.atom1_name], name2slot[b.atom2_name]
                    if i != j:
                        edges.add((min(i,j), max(i,j)))
        else:
            # 退化兜底：骨架 + χ 角链上的 (b-c)、(c-d)
            for pair in [("N","CA"), ("CA","C"), ("CA","CB")]:
                if pair[0] in name2slot and pair[1] in name2slot:
                    i, j = name2slot[pair[0]], name2slot[pair[1]]
                    edges.add((min(i,j), max(i,j)))
            for group in chi_atoms.get(aa, []):
                if len(group) >= 3:
                    b, c = group[1], group[2]
                    if b in name2slot and c in name2slot:
                        i, j = name2slot[b], name2slot[c]
                        edges.add((min(i,j), max(i,j)))
                if len(group) >= 4:
                    c, d = group[2], group[3]
                    if c in name2slot and d in name2slot:
                        i, j = name2slot[c], name2slot[d]
                        edges.add((min(i,j), max(i,j)))

        # 仅保留存在原子的边
        edges = {(i,j) for (i,j) in edges if exists[i] and exists[j]}

        # 邻接
        adj = [set() for _ in range(A)]
        for (u,v) in edges:
            adj[u].add(v); adj[v].add(u)

        # 2) 判环：去掉边后若仍连通，则该边在环中
        def in_cycle(u, v):
            # 暂时移除
            adj[u].discard(v); adj[v].discard(u)
            seen = [False]*A
            # 从 u BFS
            q = deque([u]); seen[u] = True
            while q:
                x = q.popleft()
                for y in adj[x]:
                    if not seen[y]:
                        seen[y] = True; q.append(y)
            # 放回
            adj[u].add(v); adj[v].add(u)
            return seen[v]

        ring_edges = set()
        for (u,v) in list(edges):
            if in_cycle(u,v):
                ring_edges.add((u,v))

        # 3) 以 CA 为根建 parent 树（BFS）
        root = name2slot.get("CA", next((i for i in range(A) if exists[i]), 0))
        parent = [-1]*A
        seen = [False]*A
        q = deque([root]); seen[root] = True; parent[root] = -1
        while q:
            x = q.popleft()
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    parent[y] = x
                    q.append(y)

        # children map
        children = defaultdict(list)
        for a,p in enumerate(parent):
            if p >= 0:
                children[p].append(a)

        def collect_subtree(start):
            out = []
            dq = deque([start])
            while dq:
                z = dq.popleft()
                out.append(z)
                for w in children.get(z, []):
                    dq.append(w)
            return sorted(out)

        # 4) 扭转轴（按 χ 角定义：b→c）+ 受影响子树（以 c 为根）
        axes = []
        affected = []
        for group in chi_atoms.get(aa, []):
            if len(group) < 3:   # 防御
                continue
            a, b, c = group[0], group[1], group[2]
            if (b in name2slot) and (c in name2slot):
                i_b, i_c = name2slot[b], name2slot[c]
                # 可选：是否允许骨架相关扭转
                is_backbone_edge = (b in backbone_names) or (c in backbone_names)
                e = (min(i_b,i_c), max(i_b,i_c))
                if (not allow_backbone_torsion) and is_backbone_edge:
                    continue
                if e in ring_edges:
                    continue
                axes.append((i_b, i_c))
                affected.append(torch.tensor(collect_subtree(i_c), dtype=torch.long))

        torsion_axis_uv = torch.tensor(axes, dtype=torch.long) if axes else torch.zeros((0,2), dtype=torch.long)

        # 5) 键长缩放设置：非骨架、非环，且两端存在
        scale_mask = torch.zeros(A, dtype=torch.bool)
        scale_parent = torch.full((A,), -1, dtype=torch.long)
        for a in range(A):
            p = parent[a]
            if p < 0 or not (exists[a] and exists[p]):
                continue
            pa_name = slot_names[p] if p < len(slot_names) else None
            a_name  = slot_names[a] if a < len(slot_names) else None
            is_backbone_edge = (pa_name in backbone_names) or (a_name in backbone_names)
            e = (min(p,a), max(p,a))
            if (not is_backbone_edge) and (e not in ring_edges):
                scale_mask[a] = True
                scale_parent[a] = p

        meta[aa] = {
            "parent": torch.tensor(parent, dtype=torch.long),
            "torsion_axis_uv": torsion_axis_uv,
            "affected_atoms": affected,
            "scale_mask": scale_mask,
            "scale_parent": scale_parent,
            "ring_edges": ring_edges,
            "name2slot": name2slot,
            "slot_names": slot_names,
        }

    return meta












class TorsionScaleHead(nn.Module):
    """
    输出：每残基的 扭转角增量 Δχ[g] + 键长缩放 s[a]（只对 scale_mask=True 的原子生效）
    """
    def __init__(self, embed_dim=256, aatype_dim=20, max_groups=4, hidden=256,
                 scale_clamp=0.1):
        super().__init__()
        self.max_groups = max_groups      # 每个残基最多 G 个扭转组（χ1..χ4）
        self.scale_clamp = scale_clamp    # 键长缩放幅度（±10%）

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + aatype_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head_torsion = nn.Linear(hidden, max_groups)    # Δχ（弧度）
        self.head_scale   = nn.Linear(hidden, 14)            # 每原子一个缩放系数候选

    def forward(self, h, aatype_probs):
        z = torch.cat([h, aatype_probs], dim=-1)
        z = self.mlp(z)
        dchi = torch.tanh(self.head_torsion(z)) * 0.5  # [-0.5, 0.5] rad，先保守
        sraw = torch.tanh(self.head_scale(z))          # [-1,1]
        return dchi, sraw

def rodrigues_rot(vecs, axis, angle, eps=1e-8):
    """
    vecs: [*, 3] 需要旋转的向量（已平移到轴心系）
    axis: [*, 3] 归一化旋转轴
    angle:[*]    旋转角
    """
    axis = axis / (torch.linalg.norm(axis, dim=-1, keepdim=True) + eps)
    cos = torch.cos(angle)[..., None]
    sin = torch.sin(angle)[..., None]
    cross = torch.cross(axis, vecs, dim=-1)
    dot = (vecs * axis).sum(dim=-1, keepdim=True)
    return vecs * cos + cross * sin + axis * dot * (1.0 - cos)

def apply_torsion_groups(
    X, exists, dchi,     # X:[B,N,14,3],  exists:[B,N,14], dchi:[B,N,G]
    torsion_axis_uv,     # Long[G,2]
    affected_atoms,      # list[tensor[E_g]]，长度=G
):
    """
    只做旋转（不改长度）。每组围绕 (u->v) 轴绕 X[u] 旋转 Δχ[g]，对组的受影响子树全部应用。
    """
    B,N,A,_ = X.shape
    G = torsion_axis_uv.shape[0]
    X_out = X.clone()

    for g in range(G):
        u, v = torsion_axis_uv[g]
        if u < 0 or v < 0:
            continue
        # 轴心与轴向
        Pu = X_out[..., u, :]  # [B,N,3]
        Pv = X_out[..., v, :]
        axis = (Pv - Pu)  # [B,N,3]
        # 受影响原子集合
        idx = affected_atoms[g]  # [E_g]
        if idx.numel() == 0:
            continue
        Xa = X_out[..., idx, :]  # [B,N,E_g,3]
        # 平移到轴心系
        Va = Xa - Pu[..., None, :]
        # 旋转
        ang = dchi[..., g]  # [B,N]
        Va_rot = rodrigues_rot(Va, axis[..., None, :], ang[..., None])  # [B,N,E_g,3]
        Xa_new = Pu[..., None, :] + Va_rot

        # 回写（仅存在的原子）
        ex_mask = exists[..., idx].unsqueeze(-1).float()  # [B,N,E_g,1]
        X_out[..., idx, :] = Xa_new * ex_mask + Xa * (1.0 - ex_mask)

    return X_out

def apply_bond_scaling(
    X, exists, sraw, scale_mask, scale_parent, clamp=0.1, eps=1e-8
):
    """
    对选中的原子 a：沿 (a - parent[a]) 的方向做径向缩放： r' = r * (1 + s[a])
    s[a] 在 [-clamp, clamp] 内；其余原子忽略。
    """
    B,N,A,_ = X.shape
    scale_mask = scale_mask.bool()
    s = torch.clamp(sraw, min=-1.0, max=1.0) * clamp   # [B,N,14]

    X_out = X.clone()
    for a in torch.arange(A, device=X.device):
        if not scale_mask[a]:
            continue
        p = scale_parent[a].item()
        if p < 0:
            continue
        Pa = X_out[..., a, :]   # [B,N,3]
        Pp = X_out[..., p, :]   # [B,N,3]
        v  = Pa - Pp
        r  = torch.linalg.norm(v, dim=-1, keepdim=True) + eps
        dir = v / r
        sf = (1.0 + s[..., a][..., None])  # [B,N,1]
        Pa_new = Pp + dir * r * sf

        m = (exists[..., a] & exists[..., p]).float()[..., None]
        X_out[..., a, :] = Pa_new * m + Pa * (1.0 - m)
    return X_out


class SHTemplateRefinerHard(nn.Module):
    """
    输入：SH, aatype_probs, Rmats, tpos
    输出：局部/全局坐标，但“细化”通过  Δχ + 键长缩放 + FK 实现（几何硬约束）
    依赖：per-AA 元数据 ResidueMeta
    """
    def __init__(self, C, L_max, R_bins,
                 rigid_group_atom_positions,
                 residue_meta_dict,    # dict[AA] -> meta
                 hidden_scalar=256, max_groups=4, scale_clamp=0.1,**kwargs):
        super().__init__()
        # 模板

        T, exists,*_ = build_atom14_template_tensor_from_spec(rigid_group_atom_positions)
        self.register_buffer('template_local', T)       # [20,14,3]
        self.register_buffer('template_exists', exists) # [20,14]
        self.embed = SHResEmbed(C, L_max, R_bins, out=hidden_scalar)
        self.head  = TorsionScaleHead(embed_dim=hidden_scalar, aatype_dim=20,
                                      max_groups=max_groups, scale_clamp=scale_clamp)
        # 元数据（打包为张量形式，按 AA 顺序对齐）
        self.res_meta = self._pack_residue_meta(residue_meta_dict)
        # self.res_meta = ...  # 你已有
        packed = self.pack_residue_meta_for_batch(self.res_meta)
        # 全是小张量，注册为 buffer 方便放到同一 device
        self.register_buffer('tors_axis', packed['tors_axis'])
        self.register_buffer('aff_idx', packed['aff_idx'])
        self.register_buffer('aff_mask', packed['aff_mask'])
        self.register_buffer('G_counts', packed['G_counts'])
        self.register_buffer('scale_mask', packed['scale_mask'])
        self.register_buffer('scale_parent', packed['scale_parent'])
        self.meta_packed = {
            "tors_axis": self.tors_axis, "aff_idx": self.aff_idx, "aff_mask": self.aff_mask,
            "G_counts": self.G_counts, "scale_mask": self.scale_mask, "scale_parent": self.scale_parent,
            "Gmax": int(packed["Gmax"]), "Emax": int(packed["Emax"]),
        }


        self.Gmax = int(packed["Gmax"]);
        self.Emax = int(packed["Emax"])

    def pack_residue_meta_for_batch(self,res_meta):
        """
        输入: res_meta[AA] = { 'torsion_axis_uv':[G,2], 'affected_atoms': list[len=G] of Long[],
                               'scale_mask':[14]bool, 'scale_parent':[14]long }
        输出: 方便batch化的一组张量（按 AA 顺序对齐）
        """
        Gmax = max(m["torsion_axis_uv"].shape[0] for m in res_meta.values())
        Emax = 0
        for m in res_meta.values():
            for lst in m["affected_atoms"]:
                Emax = max(Emax, int(len(lst)))

        tors_axis = torch.full((20, Gmax, 2), -1, dtype=torch.long)
        aff_idx = torch.full((20, Gmax, Emax), 0, dtype=torch.long)  # 用0占位
        aff_mask = torch.zeros((20, Gmax, Emax), dtype=torch.bool)
        G_counts = torch.zeros(20, dtype=torch.long)
        scale_mask = torch.zeros((20, 14), dtype=torch.bool)
        scale_parent = torch.full((20, 14), -1, dtype=torch.long)

        for aa_idx, aa in enumerate(AA_ORDER):
            m = res_meta[aa]
            g = m["torsion_axis_uv"].shape[0]
            G_counts[aa_idx] = g
            if g > 0:
                tors_axis[aa_idx, :g] = m["torsion_axis_uv"]
                for gi in range(g):
                    arr = m["affected_atoms"][gi]
                    if len(arr) > 0:
                        aff_idx[aa_idx, gi, :len(arr)] = arr
                        aff_mask[aa_idx, gi, :len(arr)] = True
            scale_mask[aa_idx] = m["scale_mask"]
            scale_parent[aa_idx] = m["scale_parent"]

        return {
            "tors_axis": tors_axis,  # [20,Gmax,2]
            "aff_idx": aff_idx,  # [20,Gmax,Emax]
            "aff_mask": aff_mask,  # [20,Gmax,Emax]
            "G_counts": G_counts,  # [20]
            "scale_mask": scale_mask,  # [20,14]
            "scale_parent": scale_parent,  # [20,14]
            "Gmax": Gmax, "Emax": Emax
        }

    def rodrigues_batch(self,V, axis, angle, eps=1e-8):
        """
        V:     [K, E, 3]
        axis:  [K, 3]
        angle: [K]
        返回:   [K, E, 3]
        """
        axis = axis / (torch.linalg.norm(axis, dim=-1, keepdim=True) + eps)
        cos = torch.cos(angle)[:, None, None]  # [K,1,1]
        sin = torch.sin(angle)[:, None, None]  # [K,1,1]
        # 广播 axis -> [K,1,3]
        ax = axis[:, None, :]  # [K,1,3]
        cross = torch.cross(ax.expand_as(V), V, dim=-1)  # [K,E,3]
        dot = (V * ax).sum(-1, keepdim=True)  # [K,E,1]
        return V * cos + cross * sin + ax * dot * (1.0 - cos)

    def fk_batch_by_aa(self,
            X, exists, dchi, sraw, top_idx,  # X:[B,N,14,3] 等
            meta_packed, scale_clamp=0.1
    ):
        """
        就地返回更新后的 X（局部系）。时间复杂度 ~ O(20 * (G<=4 + |scaled_atoms|<=10))
        """
        device = X.device
        tors_axis = meta_packed["tors_axis"].to(device)  # [20,Gmax,2]
        aff_idx_all = meta_packed["aff_idx"].to(device)  # [20,Gmax,Emax]
        aff_mask_all = meta_packed["aff_mask"].to(device)  # [20,Gmax,Emax]
        G_counts = meta_packed["G_counts"].to(device)  # [20]
        scale_mask_all = meta_packed["scale_mask"].to(device)  # [20,14]
        scale_parent_all = meta_packed["scale_parent"].to(device)  # [20,14]
        Gmax = int(meta_packed["Gmax"])
        # 遍历 20 个 AA 类（组内向量化）
        for aa_idx in range(20):
            sel = (top_idx == aa_idx)
            K = int(sel.sum())
            if K == 0:
                continue
            b_idx, n_idx = torch.nonzero(sel, as_tuple=True)  # [K]
            Xk = X[b_idx, n_idx]  # [K,14,3]
            Ek = exists[b_idx, n_idx]  # [K,14]
            dk = dchi[b_idx, n_idx]  # [K,G_head]（你头里一般是 G_head=4）
            sk = sraw[b_idx, n_idx]  # [K,14]

            g_cnt = int(G_counts[aa_idx].item())
            if g_cnt > 0:
                tors = tors_axis[aa_idx, :g_cnt]  # [g_cnt,2]
                aff_idx = aff_idx_all[aa_idx, :g_cnt]  # [g_cnt,E]
                aff_mask = aff_mask_all[aa_idx, :g_cnt]  # [g_cnt,E]
                # 对每个扭转组顺序更新（顺序很重要，组之间通常是层级关系）
                for gi in range(g_cnt):
                    u = int(tors[gi, 0].item());
                    v = int(tors[gi, 1].item())
                    if u < 0 or v < 0:
                        continue
                    valid_cols = aff_mask[gi]  # [E]
                    if not bool(valid_cols.any()):
                        continue
                    cols = aff_idx[gi, valid_cols]  # [Evalid]
                    Pu = Xk[:, u, :]  # [K,3]
                    Pv = Xk[:, v, :]  # [K,3]
                    axis = Pv - Pu  # [K,3]
                    ang = dk[:, gi].contiguous()  # [K]
                    Xa = Xk[:, cols, :]  # [K,Evalid,3]
                    Va = Xa - Pu[:, None, :]  # [K,Evalid,3]
                    Va_r = self.rodrigues_batch(Va, axis, ang)  # [K,Evalid,3]
                    Xa_n = Pu[:, None, :] + Va_r  # [K,Evalid,3]
                    # 仅更新“存在”的原子
                    ex = (Ek[:, cols]).unsqueeze(-1).float()  # [K,Evalid,1]
                    Xk[:, cols, :] = Xa_n * ex + Xa * (1.0 - ex)

            # 键长缩放（对该 AA 的固定原子集合，向量化 over K）
            scale_mask = scale_mask_all[aa_idx]  # [14]
            if bool(scale_mask.any()):
                scaled_atoms = torch.nonzero(scale_mask, as_tuple=True)[0]  # [S]
                parents = scale_parent_all[aa_idx, scaled_atoms]  # [S]
                # clamp sraw 幅度
                s_eff = torch.clamp(sk[:, scaled_atoms], -1.0, 1.0) * scale_clamp  # [K,S]
                for si, a in enumerate(scaled_atoms.tolist()):
                    p = int(parents[si].item())
                    if p < 0:
                        continue
                    Pa = Xk[:, a, :]  # [K,3]
                    Pp = Xk[:, p, :]  # [K,3]
                    v = Pa - Pp
                    r = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-8)
                    dir = v / r
                    sf = (1.0 + s_eff[:, si:si + 1])  # [K,1]
                    Pa_new = Pp + dir * r * sf
                    m = (Ek[:, a] & Ek[:, p]).float().unsqueeze(-1)  # [K,1]
                    Xk[:, a, :] = Pa_new * m + Pa * (1.0 - m)

            # 写回
            X[b_idx, n_idx] = Xk
        return X

    def _pack_residue_meta(self, residue_meta_dict):
        """
        把 python 结构打包成按 AA 排列的张量/列表，便于 batch 软混合或 argmax 选择。
        这里为了简洁，推理时走 top-1 AA 的元数据；训练时你也可以软混合，但 FK 的
        组合会更复杂（通常不值得）。所以默认：用 argmax aatype 选元数据。
        """
        return residue_meta_dict  # 直接存，使用时按 top-1 AA 取

    @torch.no_grad()
    def _pick_meta(self, aatype_probs):
        # 选每个残基的 top-1 AA 字符串 Key
        idx = aatype_probs.argmax(dim=-1)  # [B,N]
        # 返回一个列表结构，后续循环处理；（也可预打包成张量字典）
        return idx

    def forward(self, SH, aatype_probs, Rmats, tpos, node_mask=None):
        B,N = SH.shape[:2]
        # 1) 残基嵌入
        h = self.embed(SH,node_mask)  # [B,N,E]

        # 2) 软模板（坐标 + 存在）
        T = (aatype_probs.unsqueeze(-1).unsqueeze(-1) * self.template_local[None,None]).sum(dim=2)  # [B,N,14,3]
        exists = (aatype_probs @ self.template_exists.float()).clamp(0,1) > 0.5                     # [B,N,14]

        # 3) 预测 Δχ + scale
        dchi, sraw = self.head(h, aatype_probs)   # dchi:[B,N,G], sraw:[B,N,14]

        # 4) 选元数据（top-1 AA）
        top_idx = aatype_probs.argmax(dim=-1)     # [B,N]

        # # 5) 逐残基 FK（为了清晰先写 loop；实际可 batch 化到一定程度）
        # X_local = []
        # for b in range(B):
        #     Xb = T[b]               # [N,14,3]
        #     Eb = exists[b]          # [N,14]
        #     db = dchi[b]            # [N,G]
        #     sb = sraw[b]            # [N,14]
        #     idxb = top_idx[b]       # [N]
        #     Xb_out = []
        #     for n in range(N):
        #         # 拿该残基的 AA 名的元数据
        #         aa_idx = idxb[n].item()
        #         aa_name = AA_ORDER[aa_idx]
        #         meta = self.res_meta[aa_name]
        #
        #         # 旋转组
        #         tors_axis = meta["torsion_axis_uv"].to(Xb.device)          # [G,2]
        #         aff_list = meta["affected_atoms"]                           # list[tensor]
        #         # 复制局部坐标
        #         x = Xb[n:n+1:, :].unsqueeze(0)                             # [1,14,3]
        #         e = Eb[n:n+1, :].unsqueeze(0)        # [1,14]
        #         d = db[n:n+1, :].unsqueeze(0)        # [1,G]
        #         # 轴旋
        #         x = apply_torsion_groups(
        #             x, e, d, tors_axis, aff_list
        #         )
        #         # 键长缩放
        #         x = apply_bond_scaling(
        #             x, e, sb[n:n+1], meta["scale_mask"].to(Xb.device),
        #             meta["scale_parent"].to(Xb.device),
        #             clamp=self.head.scale_clamp
        #         )
        #         Xb_out.append(x[0])
        #     X_local.append(torch.stack(Xb_out, dim=0))
        # X_local = torch.stack(X_local, dim=0).squeeze(2)  # [B,N,14,3]

        # 5) 批处理 FK（替代双层 for）
        # X_local = T.clone()  # [B,N,14,3] 软模板起点
        # X_local = self.fk_batch_by_aa(
        #     X_local, exists, dchi, sraw, top_idx,
        #     self.meta_packed, scale_clamp=self.head.scale_clamp
        # )  # -> [B,N,14,3]

        # X_local 初始 = 软模板 T
        M = {
            "tors_axis": self.tors_axis[top_idx],  # [B,N,Gmax,2]
            "aff_idx": self.aff_idx[top_idx],  # [B,N,Gmax,Emax]
            "aff_mask": self.aff_mask[top_idx],
            "G_counts": self.G_counts[top_idx],  # [B,N]
            "scale_mask": self.scale_mask[top_idx],  # [B,N,14]
            "scale_parent": self.scale_parent[top_idx],  # [B,N,14]
            "Gmax": self.Gmax, "Emax": self.Emax,
        }
        X_local = fk_vectorized(
            X=T.clone(),
            exists=exists,
            dchi=dchi[..., :self.meta_packed["Gmax"]],  # 截取前 Gmax
            sraw=sraw,
            top_idx=top_idx,
            meta_packed=M,
            scale_clamp=self.head.scale_clamp
        )

        # print('valid χ groups =', M['G_counts'].sum().item())  # > 0 ?
        # print('any affected atoms =', M['aff_mask'].any().item())  # True ?
        # print('any scale atoms =', M['scale_mask'].any().item())  # True ?
        # print('axis_ok any =', ((M['tors_axis'][..., 0] >= 0) & (M['tors_axis'][..., 1] >= 0)).any().item())

        # 6) node_mask
        if node_mask is not None:
            X_local = X_local * node_mask[...,None,None]
            # exists  = exists & node_mask.bool()

        # 7) 回到全局
        X_global = torch.einsum('bnij,bnaj->bnai', Rmats, X_local) + tpos[...,None,:]

        return dict(atom14_local=X_local, atom14_global=X_global, atom14_exists=exists)
if __name__ == '__main__':
    # 假设你的 residue_constants.py 以 rc 导入
    from openfold.np import residue_constants as rc

    residue_meta_dict = build_residue_meta_dict_from_rc(rc, allow_backbone_torsion=True)

    # 取 LEU 的 χ 轴、子树
    m = residue_meta_dict['LEU']
    print(m['torsion_axis_uv'])  # shape [G,2]
    print([v.tolist() for v in m['affected_atoms']])
