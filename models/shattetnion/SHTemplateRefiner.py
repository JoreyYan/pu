import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ipa_pytorch
from openfold.np.residue_constants import rigid_group_atom_positions,residue_atoms
from models.shattetnion.torsion_head import StructureModule
from data import utils as du
# ---- 准备：模板与映射 ----
AA_ORDER = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
AA2IDX = {aa:i for i,aa in enumerate(AA_ORDER)}

# 你的 rigid_group_atom_positions: dict[AA] -> list[[name, group_id, (x,y,z)]]
# 这里预处理成固定顺序的 14 槽位（与 Atom14 对齐；空的填 nan）
def build_atom14_template_tensor(rigid_group_atom_positions, atom14_names_order):
    """
    rigid_group_atom_positions: dict[AA] -> list[[atom_name, group_id, (x,y,z)]]
    atom14_names_order: ['N','CA','C','O',...14] 的固定顺序
    返回:
        T: [20,14,3] float32 模板局部坐标（NaN 表示该原子不存在）
        exists: [20,14] bool，True 表示该 AA 在此槽位有原子
    """
    T = torch.full((20, 14, 3), float('nan'), dtype=torch.float32)
    exists = torch.zeros(20, 14, dtype=torch.bool)

    for aa, aa_idx in AA2IDX.items():
        # spec 是该氨基酸的所有模板原子数据
        # 例如: ['N', 0, (-0.520, 1.363, 0.000)]
        spec = rigid_group_atom_positions[aa]

        # 建立 "原子名" -> 坐标 tensor
        mp = {atom_name: torch.tensor(coords, dtype=torch.float32)
              for atom_name, group_id, coords in spec}

        # 按 atom14_names_order 填到模板矩阵里
        for a_i, name in enumerate(atom14_names_order):
            if name in mp:
                T[aa_idx, a_i] = mp[name]
                exists[aa_idx, a_i] = True

    return T, exists


def build_atom14_template_tensor_from_spec(rigid_group_atom_positions, max_slots: int = 14, device=None):
    """
    直接按 rigid_group_atom_positions[AA] 中的原子顺序填充到前 max_slots 个槽位：
      - 若该 AA 原子数 > max_slots，则截断（保留前 max_slots）
      - 若 < max_slots，则用 NaN 填充空槽位，exists=False
    返回：
      T:        [20, max_slots, 3] float32（NaN 表示该槽位无原子）
      exists:   [20, max_slots]    bool
      slot_names_per_aa: dict[AA] -> List[str|None]（长度 max_slots）
      name2slot_per_aa:  dict[AA] -> dict[str,int]（仅包含落入槽位的原子）
    """
    dtype = torch.float32
    T = torch.full((20, max_slots, 3), float('nan'), dtype=dtype, device=device)
    exists = torch.zeros(20, max_slots, dtype=torch.bool, device=device)
    slot_names_per_aa = {}
    name2slot_per_aa  = {}

    for aa, aa_idx in AA2IDX.items():
        spec = rigid_group_atom_positions.get(aa, [])
        # spec 形如: [["N", group_id, (x,y,z)], ...]
        # 按给定顺序依次填槽位
        names  = []
        nfill = min(len(spec), max_slots)
        # 填充
        for s_i in range(nfill):
            atom_name, group_id, coords = spec[s_i]
            T[aa_idx, s_i] = torch.tensor(coords, dtype=dtype, device=device)
            exists[aa_idx, s_i] = True
            names.append(atom_name)

        # 补齐 None（占位）
        if nfill < max_slots:
            names.extend([None] * (max_slots - nfill))

        slot_names_per_aa[aa] = names
        name2slot_per_aa[aa]  = {name: i for i, name in enumerate(names) if name is not None}
    T = torch.nan_to_num(T, nan=0.0)
    return T, exists, slot_names_per_aa, name2slot_per_aa

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# 假设你已有 ipa_pytorch
# from ipa_pytorch import Linear as IPALinear, StructureModuleTransition as IPATrans
IPALinear = ipa_pytorch.Linear
IPATrans  = ipa_pytorch.StructureModuleTransition

class ResidualIPABlock(nn.Module):
    """
    单层：prenorm + Linear 残差，再 prenorm + Transition 残差
    形状：h [B, N, C]
    """
    def __init__(self, c, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(c)
        self.post = IPALinear(c, c, init="final")
        self.ln2 = nn.LayerNorm(c)
        self.trans = IPATrans(c=c)           # 内部是几层 MLP/门控
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h, node_mask=None):
        # 残差 1：Linear
        x = self.ln1(h)
        u = self.post(x)
        u = self.drop(u)
        if node_mask is not None:
            u = u * node_mask[..., None]
        h = h + u

        # 残差 2：Transition
        x = self.ln2(h)
        v = self.trans(x)      # 期望 [B,N,C]
        v = self.drop(v)
        if node_mask is not None:
            v = v * node_mask[..., None]
        h = h + v
        return h


class ResidualIPAStack(nn.Module):
    """
    多层残差堆栈，按照你的命名把每层也挂到 self.trunk 里，方便你复用原有代码风格。
    """
    def __init__(self, c, layers=6, dropout=0.0, checkpoint_layers=False):
        super().__init__()
        self.c = c
        self.layers = layers
        self.checkpoint_layers = checkpoint_layers

        # 对齐你的命名：ipa_ln_{b} / post_tfmr_{b} / node_transition_{b}
        # 同时提供一个按序执行的 blocks 列表用于 forward
        self.trunk = nn.ModuleDict()
        self.blocks = nn.ModuleList()

        for b in range(layers):
            # 独立组件（可按名字访问）
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(c)
            self.trunk[f'post_tfmr_{b}'] = IPALinear(c, c, init="default")
            self.trunk[f'node_transition_{b}'] = IPATrans(c=c)

            # 也提供合并的 Block（forward 时更简洁）
            block = ResidualIPABlock(c=c, dropout=dropout)
            # 把合并 Block 的子模块权重指向上面的命名模块，保证两种访问是一致的
            block.ln1     = self.trunk[f'ipa_ln_{b}']
            block.post    = self.trunk[f'post_tfmr_{b}']
            block.ln2     = nn.LayerNorm(c)                # 第二个 LN 单独一个
            block.trans   = self.trunk[f'node_transition_{b}']
            self.blocks.append(block)

        # 你原来的输出层
        self.trasout = nn.Linear(c, c)

    def forward(self, h, node_mask=None):
        """
        h: [B, N, C]
        node_mask: [B, N]，可选
        """
        for b, block in enumerate(self.blocks):
            if self.checkpoint_layers and self.training:
                h = checkpoint(lambda _h: block(_h, node_mask=node_mask), h, use_reentrant=False)
            else:
                h = block(h, node_mask=node_mask)
        return self.trasout(h)

# ---- SH 编码器（把 SH 压成每残基 embedding）----
class SHResEmbed(nn.Module):
    def __init__(self,C, L_max, R_bins, hidden=256, out=256, dropout=0.1):
        super().__init__()
        in_ch = C * (L_max + 1) * (2*L_max + 1)  # 每个 R 的通道
        self.branchA = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=1), nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.branchB = nn.Sequential(
            nn.LayerNorm(in_ch * R_bins),
            nn.Linear(in_ch * R_bins, hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.out = nn.Linear(hidden*2, out)


        self.C, self.L_max, self.R_bins = C, L_max, R_bins

        self.stack = ResidualIPAStack(c=out, layers=8, dropout=0.1, checkpoint_layers=True)

    def forward(self, SH,node_mask):  # [B,N,C,L+1,2L+1,R]
        B,N,C,Lp1,M,R = SH.shape
        assert C==self.C and Lp1==self.L_max+1 and R==self.R_bins
        x = SH
        xa = x.view(B*N, C*(Lp1)*M, R)     # -> Conv1d over R
        xa = self.branchA(xa).squeeze(-1)  # [B*N, hidden]
        xa = xa.view(B,N,-1)

        xb = x.view(B, N, -1)              # flatten
        xb = self.branchB(xb)              # [B,N, hidden]

        h  = torch.cat([xa, xb], dim=-1)   # [B,N, 2*hidden]
        h  = self.out(h)

        h = self.stack(h, node_mask=node_mask)   # h: [B,N,out]

        return    h              # [B,N,out]

# ---- 解码头：输出每个原子的局部偏移 Δx_hat（或参数化为扭转）----
class TemplateRefineHead(nn.Module):
    def __init__(self, embed_dim=256, aatype_dim=20, hidden=256, mode='delta'):
        """
        mode:
          'delta'  -> 直接回归 Δx_local [14,3]
          'scale'  -> 回归每原子标量 s，Δx = MLP(h)*s，或用于键长缩放
          'torsion'-> 可扩展成逐段扭转（需要树结构）
        """
        super().__init__()
        self.mode = mode
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 14*3)
        )

    def forward(self, h):
        # h:[B,N,E], aatype_probs:[B,N,20]（可 soft 或 one-hot）
           # [B,N, E+20]
        delta = self.fc(h).view(h.size(0), h.size(1), 14, 3)
        # 可选：对 Δx 做小范围限制，比如 tanh*0.8Å
        return torch.tanh(delta) * 0.8                  # [B,N,14,3]

# ---- 主模块：Template 放置 + 细化 ----
class SHTemplateRefiner(nn.Module):
    def __init__(self, C, L_max, R_bins,
                 rigid_group_atom_positions=rigid_group_atom_positions, atom14_names_order=residue_atoms,
                 hidden_scalar=256,c_z=128,**kwargs):
        super().__init__()
        # T, exists = build_atom14_template_tensor(rigid_group_atom_positions, atom14_names_order)
        T, exists,*_=build_atom14_template_tensor_from_spec(rigid_group_atom_positions)
        self.register_buffer('template_local', T)       # [20,14,3]
        self.register_buffer('template_exists', exists) # [20,14]
        self.embed = SHResEmbed(C, L_max, R_bins, out=hidden_scalar)
        self.head  = TemplateRefineHead(embed_dim=hidden_scalar, aatype_dim=20, mode='delta')

        self.Torsion=StructureModule(c_s=hidden_scalar,c_z=c_z)

    def forward(self, noisy_batch,SH,  Rmats, tpos,aatype_probs, node_mask=None):
        """
        SH:           [B,N,C,L+1,2L+1,R]
        aatype_probs: [B,N,20]  (可软/硬)
        Rmats:        [B,N,3,3]
        tpos:         [B,N,3]
        return: dict
          atom14_local:  [B,N,14,3]
          atom14_global: [B,N,14,3]
          atom14_exists: [B,N,14] (bool)
        """
        B,N = SH.shape[:2]
        # 1) 残基特征
        h = self.embed(SH,node_mask)                      # [B,N,E]

        # x=self.Torsion(
        #     evoformer_output_dict={
        #         'single':h
        #     },
        #     aatype=aatype_probs,
        #     rigids=du.create_rigid(Rmats, tpos),
        #     mask=node_mask,
        # )




        # 2) 模板放置（局部）
        # 取每个残基的“期望模板”= 对 20 个模板按 aatype_probs 做线性组合（软模板）
        # [B,N,20,1,1] * [20,14,3] -> [B,N,14,3]
        T = (aatype_probs.unsqueeze(-1).unsqueeze(-1) * self.template_local[None,None]).sum(dim=2)

        # 3) 细化偏移（局部）
       # dX = self.head(h)         # [B,N,14,3]
        X_local = T #+ dX*node_mask[...,None,None]                        # [B,N,14,3]

        # mask 无效原子（不存在的模板位置）
        # 同样按 soft-模板求 exists 概率阈值（>0.5 视为存在），或取 top1 aatype 的 exists
        exists = (aatype_probs @ self.template_exists.float()).clamp(0,1) > 0.5  # [B,N,14]
        X_local = torch.where(exists[...,None], X_local, torch.zeros_like(X_local))

        # 4) 回到全局
        X_global = torch.einsum('bnij,bnaj->bnai', Rmats, X_local)
        X_global=X_global+ tpos[..., None, :]
        # 可选：node_mask 清除 padding
        if node_mask is not None:
            X_local  = X_local  * node_mask[...,None,None]
            X_global = X_global * node_mask[...,None,None]


        return dict(atom14_local=X_local, atom14_global=X_global, atom14_exists=exists)
