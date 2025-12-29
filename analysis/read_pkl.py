import os
import pickle
import torch
from tqdm import tqdm
import glob
from openfold.np import residue_constants as rc
from openfold.data.data_transforms import (make_atom14_masks, make_atom14_positions)  # 你的那个特征处理文件

# ===== 元素类型映射配置 =====
ELEMENTS = ["C", "N", "O", "S","X"]      # X=未知/其他
E2I = {e: i for i, e in enumerate(ELEMENTS)}

def _atom_name_to_element(atom_name: str) -> str:
    if not atom_name:
        return "X"
    c = atom_name[0].upper()
    return c if c in {"C","N","O","S"} else "X"

# ===== 构造 restype x 14 元素索引表 =====
restype_3_list = [rc.restype_1to3[r] for r in rc.restypes] + ["UNK"]
elem_idx_table = []
for resname in restype_3_list:
    names14 = rc.restype_name_to_atom14_names.get(resname, ["" for _ in range(14)])
    elem_row = [E2I[_atom_name_to_element(n)] for n in names14]
    elem_idx_table.append(elem_row)
elem_idx_table = torch.tensor(elem_idx_table, dtype=torch.long)

# ===== 路径配置 =====
src_dir = "/home/junyu/project/frameflow2data/processed_scope/"
dst_dir = "/home/junyu/project/frameflow2data/processed_scope_ele/"
os.makedirs(dst_dir, exist_ok=True)


# ======= 配置：元素集合（可按需扩展）=======
ELEMENTS = ["C", "N", "O", "S", "X"]      # X = 其他/未知
E2I = {e: i for i, e in enumerate(ELEMENTS)}

def _atom_name_to_element(atom_name: str) -> str:
    if not atom_name:
        return "X"
    c = atom_name[0].upper()
    return c if c in {"C","N","O","S"} else "X"


# ===== 处理所有 pkl =====

pkl_files = [
    os.path.basename(f)
    for f in glob.glob(os.path.join(src_dir, "*.pkl"))
    if not os.path.basename(f).startswith("._")
]

for fname in tqdm(pkl_files, desc="Processing PKLs"):
    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)

    # 1) 读取原pkl（ndarray转tensor）
    with open(src_path, "rb") as f:
        data = pickle.load(f)

    aatype = torch.tensor(data["aatype"], dtype=torch.long)
    atom_positions = torch.tensor(data["atom_positions"], dtype=torch.float32)
    atom_mask = torch.tensor(data["atom_mask"], dtype=torch.float32)

    protein = {
        "aatype": aatype,
        "all_atom_positions": atom_positions,
        "all_atom_mask": atom_mask,
    }

    # 2) 生成 14-atom 数据
    protein = make_atom14_masks(protein)
    protein = make_atom14_positions(protein)

    # 3) 生成 14-atom 元素类型索引
    atom14_element_idx = elem_idx_table[aatype]  # [N_res,14]
    protein["atom14_element_idx"] = atom14_element_idx

    # 4) 把 tensor 转回 ndarray（方便 pickle 存）
    for k, v in protein.items():
        if torch.is_tensor(v):
            protein[k] = v.cpu().numpy()
            if k in   ('atom14_gt_exists' , 'atom14_gt_positions','atom14_alt_gt_positions','atom14_alt_gt_exists','atom14_element_idx','atom14_is_ambiguous'):
                data[k] = v.cpu().numpy()

    # 5) 保存到新目录
    with open(dst_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
