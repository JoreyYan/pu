import torch
from typing import Dict, Optional
from openfold.np import residue_constants as rc
import torch
import numpy as np
# 固定4通道：C,N,O,S
ELEMENTS = {"C":0, "N":1, "O":2, "S":3}
C = 4

def _atom_name_to_elem_id(name: str) -> int:
    if not name:
        return -1
    c = name[0].upper()
    return ELEMENTS.get(c, -1)

@torch.no_grad()
def build_elem_slot_maps(
    device: torch.device = torch.device("cpu"),
    X: Optional[int] = 11,     # 固定为11（TRP的C最多=11）
) -> Dict[str, torch.Tensor]:
    """
    构建“元素槽位 ↔ 14-atom”映射（20 AA + UNK），并固定 X=10。
    返回字典包含：
      - elem14:           [R,14]   每残基14原子的元素ID（C/N/O/S→0..3，其他/空为-1）
      - rank14:           [R,14]   每个14原子在其元素通道内的出现秩（0,1,2,...），无效为-1
      - slot_to_atom14:   [R,C,X]  (r,c,x) → 14原子索引（无则-1）
      - slot_needed_mask: [R,C,X]  该槽位是否应存在（True/False）
      - atom14_to_cx:     [R,14,2] 每个14原子对应 (c,x)；无则(-1,-1)
      - X:                ()       标量张量=10
      - X_min:            ()       理论最小X（检查用）
    """
    # 1) 取 20aa + UNK 的 14-atom 原子名表
    restype_3_list = [rc.restype_1to3[r] for r in rc.restypes] + ["UNK"]
    R = len(restype_3_list)

    names14 = []
    for resname in restype_3_list:
        names = rc.restype_name_to_atom14_names.get(resname, [""]*14)
        names14.append(names)

    # 2) 元素ID表：C/N/O/S→0..3，其他/空→-1
    elem14 = torch.tensor(
        [[_atom_name_to_elem_id(nm) for nm in row] for row in names14],
        dtype=torch.long, device=device
    )  # [R,14]

    # 3) 在每个元素通道内计算“出现秩”
    rank14 = torch.full((R,14), -1, dtype=torch.long, device=device)
    for r in range(R):
        for c in range(C):
            idx = (elem14[r] == c).nonzero(as_tuple=False).view(-1)
            if idx.numel() > 0:
                rank14[r, idx] = torch.arange(idx.numel(), device=device, dtype=torch.long)

    # 4) 统计最小所需 X（用于 sanity check）
    X_min = 0
    for r in range(R):
        for c in range(C):
            X_min = max(X_min, int((elem14[r] == c).sum().item()))
    X_min = torch.tensor(X_min, dtype=torch.long, device=device)

    # 校验：X 必须 >= X_min（你这里固定10，应当满足）
    if X is None:
        X = int(X_min.item())
    else:
        if X < int(X_min.item()):
            raise ValueError(f"X={X} < required minimum X_min={int(X_min.item())}")

    # 5) 槽位→14索引（slot_to_atom14）
    slot_to_atom14 = torch.full((R, C, X), -1, dtype=torch.long, device=device)  # [R,C,X]
    for r in range(R):
        for c in range(C):
            idx = (elem14[r] == c).nonzero(as_tuple=False).view(-1)
            k = min(idx.numel(), X)
            if k > 0:
                slot_to_atom14[r, c, :k] = idx[:k]

    # 6) 槽位是否应存在
    slot_needed_mask = slot_to_atom14.ge(0)  # [R,C,X]

    # 7) 反向：14原子 → (c,x)
    atom14_to_cx = torch.full((R,14,2), -1, dtype=torch.long, device=device)
    for r in range(R):
        for c in range(C):
            for x in range(X):
                a = int(slot_to_atom14[r, c, x].item())
                if a >= 0:
                    atom14_to_cx[r, a, 0] = c
                    atom14_to_cx[r, a, 1] = x

    return {
        "elem14": elem14,                       # [R,14]
        "rank14": rank14,                       # [R,14]
        "slot_to_atom14": slot_to_atom14,       # [R,C,X]
        "slot_needed_mask": slot_needed_mask,   # [R,C,X]
        "atom14_to_cx": atom14_to_cx,           # [R,14,2]
        "X_min": X_min,                         # ()
        "X": torch.tensor(X, device=device),    # () (=10)
    }


@torch.no_grad()
def atom14_to_elem_slots(
    coords14: torch.Tensor,          # [B,N,14,3]
    aatype: torch.Tensor,            # [B,N]  (0..19，UNK=20)
    maps: Dict[str, torch.Tensor],   # 来自 build_elem_slot_maps(X=10)
    mask14: Optional[torch.Tensor] = None,  # [B,N,14] (bool)，可选
):
    """
    把 Atom14 坐标映射为元素通道固定槽位坐标（C/N/O/S × X=10）。

    Returns:
      coords_by_elem: [B,N,4,10,3]
      mask_by_elem:   [B,N,4,10]  (bool)
    """
    device = coords14.device
    dtype  = coords14.dtype
    B, N, A, _ = coords14.shape
    assert A == 14, f"coords14 last-but-one dim must be 14, got {A}"

    atom14_to_cx = maps["atom14_to_cx"].to(device)  # [R,14,2]
    X = int(maps["X"].item())
    C = 4

    # 取每个 (b,n,a) 对应的 (c,x) 槽位
    cx   = atom14_to_cx[aatype]       # [B,N,14,2]
    cidx = cx[..., 0]                 # [B,N,14]
    xidx = cx[..., 1]                 # [B,N,14]

    # 有效性：必须映射存在；如提供 mask14，再与其相与
    valid = (cidx >= 0) & (xidx >= 0)
    if mask14 is not None:
        valid = valid & mask14.to(torch.bool)

    # 仅对 valid 位置做写入（避免 scatter 的冲突与无效写）
    if valid.any():
        b_idx, n_idx, a_idx = valid.nonzero(as_tuple=True)      # [K]
        c_v  = cidx[valid]                                      # [K]
        x_v  = xidx[valid]                                      # [K]
        p_v  = coords14[b_idx, n_idx, a_idx, :]                 # [K,3]

        coords_by_elem = coords14.new_zeros(B, N, C, X, 3)
        mask_by_elem   = torch.zeros(B, N, C, X, dtype=torch.bool, device=device)

        coords_by_elem[b_idx, n_idx, c_v, x_v, :] = p_v
        mask_by_elem[b_idx, n_idx, c_v, x_v]      = True
    else:
        coords_by_elem = coords14.new_zeros(B, N, C, X, 3)
        mask_by_elem   = torch.zeros(B, N, C, X, dtype=torch.bool, device=device)

    return coords_by_elem, mask_by_elem

@torch.no_grad()
def regroup_elem_to_atom14_fast(
    coords_by_elem: torch.Tensor,   # [B,N,4,10,3]  模型输出（元素槽位坐标）
    aatype: torch.Tensor,           # [B,N]         (0..19, UNK=20)
    maps: Dict[str, torch.Tensor],  # build_elem_slot_maps(...) 的输出
):
    """
    利用预计算映射，向量化回填到14-atom顺序。
    返回:
      coords14: [B,N,14,3]
      mask14:   [B,N,14] (bool)
    """
    atom14_to_cx = maps["atom14_to_cx"].to(coords_by_elem.device)  # [R,14,2]
    B,N,C,X,_ = coords_by_elem.shape

    # 取每个 (b,n,a) 的 (c,x) 槽位
    cx = atom14_to_cx[aatype]       # [B,N,14,2]
    c_idx = cx[..., 0]              # [B,N,14]
    x_idx = cx[..., 1]              # [B,N,14]
    valid = (c_idx >= 0) & (x_idx >= 0)

    # 把 (C,X) 合成一维，便于 gather
    lin = (c_idx * X + x_idx).clamp_min(0)                # [B,N,14]
    src = coords_by_elem.view(B, N, C*X, 3)               # [B,N,40,3]
    gathered = torch.gather(src, dim=2, index=lin.unsqueeze(-1).expand(-1,-1,-1,3))

    coords14 = coords_by_elem.new_zeros(B, N, 14, 3)
    coords14[valid] = gathered[valid]
    mask14 = valid
    return coords14, mask14





def compare_tensors(tensor1, tensor2, values=[0, 1, 2, 3]):
    """
    比较两个[B,N]格式tensor在特定数值位置是否一致

    Args:
        tensor1: 第一个tensor [B,N]
        tensor2: 第二个tensor [B,N]
        values: 要比较的数值列表

    Returns:
        dict: 比较结果统计
    """
    assert tensor1.shape == tensor2.shape, f"形状不匹配: {tensor1.shape} vs {tensor2.shape}"

    B, N = tensor1.shape
    results = {}

    print(f"比较两个tensor形状: {tensor1.shape}")
    print("=" * 60)

    # 整体完全一致性检查
    completely_same = torch.equal(tensor1, tensor2)
    print(f"整体完全一致: {completely_same}")

    if not completely_same:
        diff_count = (tensor1 != tensor2).sum().item()
        diff_ratio = diff_count / (B * N) * 100
        print(f"不同位置数量: {diff_count}/{B * N} ({diff_ratio:.2f}%)")

    print("\n" + "-" * 40)

    # 对每个特定数值进行比较
    for value in values:
        print(f"\n检查数值 {value}:")

        # 找到两个tensor中值为value的位置
        mask1 = (tensor1 == value)  # [B,N]
        mask2 = (tensor2 == value)  # [B,N]

        count1 = mask1.sum().item()
        count2 = mask2.sum().item()

        print(f"  tensor1中有 {count1} 个位置值为{value}")
        print(f"  tensor2中有 {count2} 个位置值为{value}")

        if count1 == 0 and count2 == 0:
            print(f"  ✅ 两个tensor都没有值为{value}的位置")
            results[value] = {
                'same_positions': True,
                'count1': count1,
                'count2': count2,
                'intersection': 0,
                'union': 0
            }
            continue

        # 检查位置是否完全一致
        positions_same = torch.equal(mask1, mask2)

        # 计算交集和并集
        intersection = (mask1 & mask2).sum().item()
        union = (mask1 | mask2).sum().item()

        if positions_same:
            print(f"  ✅ 值为{value}的位置完全一致")
        else:
            print(f"  ❌ 值为{value}的位置不完全一致")
            print(f"     交集: {intersection} 个位置")
            print(f"     并集: {union} 个位置")

            if union > 0:
                overlap_ratio = intersection / union * 100
                print(f"     重叠率: {overlap_ratio:.2f}%")

            # 找到差异位置
            only_in_1 = mask1 & (~mask2)
            only_in_2 = mask2 & (~mask1)
            only_1_count = only_in_1.sum().item()
            only_2_count = only_in_2.sum().item()

            if only_1_count > 0:
                print(f"     仅在tensor1中: {only_1_count} 个位置")
            if only_2_count > 0:
                print(f"     仅在tensor2中: {only_2_count} 个位置")

        results[value] = {
            'same_positions': positions_same,
            'count1': count1,
            'count2': count2,
            'intersection': intersection,
            'union': union,
            'overlap_ratio': intersection / union * 100 if union > 0 else 100.0
        }

    return results

if __name__ == '__main__':


    # 1) 预先构建映射表（可保存到磁盘以复用）
    maps = build_elem_slot_maps(device=torch.device("cpu"), X=10)
    print("X_min =", int(maps["X_min"].item()))  # 应 ≤ 10

    # 2) 模型输出（示例）
    B, N = 4, 128
    coords_by_elem = torch.randn(B, N, 4, 10, 3)  # 你的网络直接预测这个
    aatype = torch.randint(0, 20, (B, N))         # 0..19 + UNK=20

    # 3) 回填到14原子顺序（全向量化，无BN循环）
    coords14, mask14 = regroup_elem_to_atom14_fast(coords_by_elem, aatype, maps)
    # coords14: [B,N,14,3], mask14: [B,N,14]
