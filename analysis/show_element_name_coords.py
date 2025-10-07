import torch
import numpy as np
import matplotlib.pyplot as plt

ELEM2IDX = {"C":0, "N":1, "O":2, "S":3}

AA_ORDER = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO",
            "SER", "THR", "TRP", "TYR", "VAL"]
AA2IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
def build_name_table_per_residue(target_aa_name: str,
                                 aatype: torch.Tensor,              # [B,N] 0..19
                                 atom14_element_idx: torch.Tensor,  # [B,N,14] in {0:C,1:N,2:O,3:S}
                                 rigid_group_atom_positions: dict):
    """
    返回：elem2names: dict[int, list[(slot_idx, atom_name)]]
         例如 elem2names[0] = [(3,'CB'), (6,'CG'), ...] （0=C,1=N,2=O,3=S）
    做法：对该残基类型的每个 atom14 槽位 j，统计它在选中残基上的元素类型（取众数），
          然后把该槽位的名字归到对应通道里。
    """
    target_idx = AA2IDX[target_aa_name]
    B, N = aatype.shape
    sel = (aatype == target_idx)                 # [B,N]
    assert sel.any(), f"No residues of type {target_aa_name} in this batch."

    # 该残基的 atom14 名称顺序（你的表给的顺序即 atom14 的顺序）
    names_this = [row[0] for row in rigid_group_atom_positions[target_aa_name]]
    num_slots = min(14, len(names_this))

    elem2names = {0:[], 1:[], 2:[], 3:[]}
    # 针对每个槽位 j，统计它的元素（在选中残基上取众数/第一个）
    for j in range(num_slots):
        name_j = names_this[j]
        if not name_j:  # 空名跳过
            continue
        elems_j = atom14_element_idx[:, :, j]    # [B,N]
        elems_j = elems_j[sel]                   # [K]
        if elems_j.numel() == 0:
            continue
        # 众数（如果你的 PyTorch 版本没有 torch.mode 的 values/counts，就取第一个）
        try:
            elem_mode = torch.mode(elems_j.flatten()).values.item()
        except:
            elem_mode = int(elems_j.flatten()[0].item())
        if elem_mode in (0,1,2,3):
            elem2names[elem_mode].append((j, name_j))
    return elem2names  # 每个元素通道对应哪些槽位、叫什么名


def gather_positions_by_name_for_element(target_aa_name: str,
                                         target_elem: str,            # "C"/"N"/"O"/"S"
                                         aatype: torch.Tensor,        # [B,N]
                                         atom14_positions: torch.Tensor,     # [B,N,14,3]
                                         atom14_element_idx: torch.Tensor,   # [B,N,14]
                                         rigid_group_atom_positions: dict):
    """
    返回：name2pts: dict[str, np.ndarray(M,3)]
    取出所有该残基类型中，指定元素通道的原子，按 atom name 合并坐标。
    """
    elem_idx = ELEM2IDX[target_elem]
    elem2names = build_name_table_per_residue(target_aa_name, aatype, atom14_element_idx, rigid_group_atom_positions)
    slots = elem2names.get(elem_idx, [])
    if not slots:
        return {}

    target_idx = AA2IDX[target_aa_name]
    sel = (aatype == target_idx)                 # [B,N]

    name2pts = {}
    for j, name_j in slots:
        # 严格检查首字母是否和元素符号一致
        if not name_j.startswith(target_elem):
            continue

        pos_j = atom14_positions[:, :, j, :]
        elem_j = atom14_element_idx[:, :, j]
        mask = sel & (elem_j == elem_idx)

        if mask.any():
            pts = pos_j[mask]
            arr = pts.detach().cpu().numpy()
            if name_j in name2pts:
                name2pts[name_j] = np.concatenate([name2pts[name_j], arr], axis=0)
            else:
                name2pts[name_j] = arr
    return name2pts


def plot_element_positions_by_name(name2pts: dict, title: str = ""):
    if not name2pts:
        print("No points to plot.")
        return
    fig = plt.figure(figsize=(6.2, 6.0))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7'])

    for i, (name, pts) in enumerate(sorted(name2pts.items())):
        if pts.size == 0: continue
        c = colors[i % len(colors)]
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=22, c=c, marker='o', alpha=0.85,
                   edgecolors='k', linewidths=0.3, label=name)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=9, frameon=True)
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    # 选一个残基类型 & 元素
    aa = "ASP"  # 你要看的氨基酸
    elem = "C"  # 你要看的元素

    name2pts = gather_positions_by_name_for_element(
        target_aa_name=aa,
        target_elem=elem,
        aatype=aatype,  # [B,N]
        atom14_positions=atom14_positions,  # [B,N,14,3]
        atom14_element_idx=atom14_element_idx,  # [B,N,14]
        rigid_group_atom_positions=rigid_group_atom_positions
    )

    plot_element_positions_by_name(name2pts, title=f"{aa} — {elem} atoms grouped by name")
