# 全原子椭圆构建代码审查报告

**日期**: 2025-11-30
**审查范围**: 数据加载、扰动、模型、Loss - 使用 from_rigid_and_all_atoms

---

## 检查清单

- [ ] 数据加载 (datasets.py)
- [ ] 模型初始化 (flow_model.py)
- [ ] GaussianRigid 实现
- [ ] 坐标系转换
- [ ] 掩码逻辑
- [ ] 动态厚度传递
- [ ] Loss 计算

---

## 1. 数据加载 (data/datasets.py)

### 代码位置：Line 125-160

```python
rigids_1 = OffsetGaussianRigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
rotmats_1 = rigids_1.get_rots().get_rot_mats()
trans_1 = rigids_1.get_trans()

backbone = torch.tensor(processed_feats['atom_positions'][:, [0, 1, 2, 4], :]).float()

res_plddt = processed_feats['b_factors'][:, 1]
res_mask = torch.tensor(processed_feats['bb_mask']).int()

dynamic_thickness = torch.where(
    ~res_mask.bool(),
    torch.tensor(2.5, device=res_mask.device),
    torch.tensor(0.5, device=res_mask.device)
).unsqueeze(-1)

rigids_1 = OffsetGaussianRigid.from_rigid_and_all_atoms(
    rigids_1,
    chain_feats['atom14_gt_positions'],
    chain_feats['atom14_gt_exists'],
    base_thickness=dynamic_thickness
)
```

### ✅ 正确点

1. **Dynamic thickness 维度**: `[N, 1]` ✓
2. **全原子输入**: 使用完整的 `atom14_gt_positions` (14个原子) ✓
3. **完整掩码**: 使用完整的 `atom14_gt_exists` ✓

### ⚠️ 潜在问题

**问题 1**: `res_mask` 的语义不清
```python
dynamic_thickness = torch.where(
    ~res_mask.bool(),  # ~res_mask = mask为0的位置
    torch.tensor(2.5, device=res_mask.device),  # 这些位置用 2.5
    torch.tensor(0.5, device=res_mask.device)   # mask=1 用 0.5
)
```

**分析**:
- `bb_mask` 通常表示"残基存在" (1=存在, 0=不存在)
- `~res_mask` = 不存在的残基 → 用 2.5Å (这合理吗？)
- 这里的逻辑可能反了

**建议**:
- 确认 `bb_mask` 的语义
- 如果 `bb_mask=1` 表示存在，那么应该：
  - 存在的残基 → 0.5Å (紧致)
  - 不存在的残基 → 2.5Å (虚胖，用于 padding)

**问题 2**: 在数据加载阶段使用动态厚度

在训练时，`update_mask` 才表示哪些残基被 mask（需要预测），但数据加载阶段还没有 `update_mask`，所以这里用 `bb_mask` 控制厚度可能不对。

**正确的逻辑应该是**:
- **数据加载阶段**: 所有残基都应该用相同的 base_thickness (例如 0.5Å)
- **模型训练阶段**: 根据 `update_mask` 动态调整厚度

---

## 2. 模型初始化 (models/flow_model.py)

### 代码位置：Line 730-844

#### 2.1 动态厚度创建 (Line 730-742)

```python
is_masked = input_feats['update_mask'].bool()
dynamic_thickness = torch.where(
    is_masked,
    torch.tensor(2.5, device=is_masked.device),
    torch.tensor(0.5, device=is_masked.device)
).unsqueeze(-1)
```

### ✅ 正确
- 维度: `[B, N, 1]` ✓
- 逻辑: masked=2.5, context=0.5 ✓

---

#### 2.2 分支 1: `sideonly=True` (Line 751-789)

```python
if 'atoms14_local_t' in input_feats and sideonly:
    atoms14_local = input_feats['atoms14_local_t']
    sidechain_atoms_local = atoms14_local[..., 3:14, :]  # 索引 3-13 (11个原子)

    sidechain_atoms_global = base_rigid.unsqueeze(-1).apply(sidechain_atoms_local)

    gt_sc_exists = input_feats['atom14_gt_exists'][..., 3:14].bool()
    is_masked_residue = input_feats['update_mask'][..., None].bool()
    geom_mask = gt_sc_exists & (~is_masked_residue)

    curr_rigids = OffsetGaussianRigid.from_rigid_and_sidechain(
        base_rigid,
        sidechain_atoms_global,
        geom_mask,
        base_thickness=dynamic_thickness
    )
```

### ✅ 正确
- 使用 `from_rigid_and_sidechain` (侧链only) ✓
- 坐标转换: local → global ✓
- 掩码逻辑: context区域用GT，masked区域清零 ✓

### ❓ 疑问
这个分支和全原子方案冲突，`sideonly=True` 时还是用侧链方案？

---

#### 2.3 分支 2: `sideonly=False` (Line 793-844)

```python
if 'atoms14_local_t' in input_feats and not sideonly:
    atoms14_local = input_feats['atoms14_local_t']

    # 全局坐标转换
    all_atoms_global = base_rigid.unsqueeze(-1).apply(atoms14_local)

    # 掩码构建
    gt_exists = input_feats['atom14_gt_exists'].float()
    is_masked_broad = is_masked.unsqueeze(-1)

    mask_bb_core = gt_exists[..., :3]  # N, CA, C (0-2)
    mask_others = gt_exists[..., 3:]   # O + SC (3-13)

    mask_others_filtered = mask_others * (~is_masked_broad).float()
    geom_mask_all = torch.cat([mask_bb_core, mask_others_filtered], dim=-1)

    curr_rigids = OffsetGaussianRigid.from_rigid_and_all_atoms(
        base_rigid,
        all_atoms_global,
        geom_mask_all,
        base_thickness=dynamic_thickness
    )
```

### ✅ 正确
- 坐标转换: local → global ✓
- 掩码逻辑:
  - Context 区域: N,CA,C,O + 侧链 (全原子)
  - Masked 区域: 只有 N,CA,C (屏蔽 O 和侧链)
- 函数调用: `from_rigid_and_all_atoms` ✓

### ⚠️ 潜在问题

**问题 3**: Masked 区域只保留 N, CA, C

```python
mask_bb_core = gt_exists[..., :3]  # 始终保留
mask_others_filtered = mask_others * (~is_masked_broad).float()  # masked区域清零
```

**分析**:
- Masked 区域的 O (索引3) 和侧链 (4-13) 都被清零
- 只用 N, CA, C 构建椭圆
- 这是合理的策略（避免信息泄露），但需要确认 `atoms14_local_t` 中 masked 区域的 O 和侧链是否已经被清零或随机化

**验证**:
- 检查数据加载时，masked 区域的 `atoms14_local_t[..., 3:, :]` 是否已经是噪声/0
- 如果不是，需要在加噪函数中处理

---

## 3. GaussianRigid 实现

### 代码位置：data/GaussianRigid.py:428-501

### ✅ 正确实现

```python
def from_rigid_and_all_atoms(
    cls,
    rigid_backbone: Rigid,
    all_atoms: torch.Tensor,  # [..., 14, 3]
    all_atom_mask: torch.Tensor,  # [..., 14]
    base_thickness: torch.Tensor,  # [..., 1] 或 scalar
):
```

**关键点**:
1. **质心计算**: 正确 ✓
   ```python
   centroid_global = all_sum / atom_count
   ```

2. **Fallback 机制**: 正确 ✓
   ```python
   has_atoms = (all_atom_mask.sum(dim=-1) > 0.5)
   centroid_global = torch.where(
       has_atoms.unsqueeze(-1),
       all_sum / atom_count,
       trans_backbone  # Fallback to CA
   )
   ```

3. **Offset 计算**: 正确 ✓
   ```python
   local_mean = rigid_backbone.invert_apply(centroid_global)
   ```

4. **Scaling 计算**: 正确 ✓
   ```python
   rigid_centered = Rigid(rots_backbone, centroid_global)
   local_atoms_centered = rigid_centered_exp.invert_apply(all_atoms)
   variance = (local_atoms_masked ** 2).sum(dim=-2) / atom_count
   std_dev = torch.sqrt(variance + 1e-8)
   scaling_log = torch.log(std_dev + base_thickness + 1e-6)
   ```

### ✅ base_thickness 支持 Tensor

Line 492:
```python
scaling_log = torch.log(std_dev + base_thickness + 1e-6)
```

这里 `base_thickness` 可以是 Tensor `[B, N, 1]`，会正确广播 ✓

---

## 4. 坐标系检查

### 4.1 数据加载 (datasets.py)

```python
chain_feats['atom14_gt_positions']  # Global 坐标 ✓
```

直接传给 `from_rigid_and_all_atoms` → 正确 ✓

---

### 4.2 模型 (flow_model.py)

```python
atoms14_local = input_feats['atoms14_local_t']  # Local 坐标
all_atoms_global = base_rigid.unsqueeze(-1).apply(atoms14_local)  # → Global
```

转换正确 ✓

---

## 5. 掩码逻辑总结

### 数据加载阶段
```python
# 使用 bb_mask 控制厚度
dynamic_thickness = where(~bb_mask, 2.5, 0.5)
```
**问题**: bb_mask 语义可能不对，应该全部用固定的 0.5Å

---

### 训练阶段

**Context 区域** (`update_mask=0`):
- 原子: N,CA,C,O + 侧链 (全原子)
- 厚度: 0.5Å
- 质心: 全原子质心

**Masked 区域** (`update_mask=1`):
- 原子: N,CA,C (只有骨架核心)
- 厚度: 2.5Å
- 质心: 骨架质心

**逻辑**: ✅ 合理

---

## 6. 发现的 Bug 和建议

### 🐛 Bug 1: datasets.py 的动态厚度逻辑可能反了

**位置**: data/datasets.py:136-140

**当前代码**:
```python
dynamic_thickness = torch.where(
    ~res_mask.bool(),  # mask=0 的位置
    torch.tensor(2.5, device=res_mask.device),
    torch.tensor(0.5, device=res_mask.device)
)
```

**问题**:
- `bb_mask` 通常表示"残基存在"
- `~res_mask` = 不存在的残基 → 用 2.5Å
- 这不合理

**建议**:
```python
# 数据加载阶段应该用统一的 base_thickness
base_thickness_loading = torch.full((res_mask.shape[0], 1), 0.5, device=res_mask.device)
```

**或者** 如果确实需要动态厚度，应该：
```python
dynamic_thickness = torch.where(
    res_mask.bool(),  # 存在的残基
    torch.tensor(0.5, device=res_mask.device),  # 用紧致的 0.5
    torch.tensor(2.5, device=res_mask.device)   # 不存在的用虚胖
).unsqueeze(-1)
```

---

### 🐛 Bug 2: sideonly 分支冲突

**位置**: models/flow_model.py:751-789 vs 793-844

**问题**:
- 代码中有两个分支：`sideonly=True` 和 `sideonly=False`
- `sideonly=True` 使用 `from_rigid_and_sidechain` (侧链only)
- `sideonly=False` 使用 `from_rigid_and_all_atoms` (全原子)

**当前配置**: `experiment.task: fbb`，调用 `forward()` 时没看到 `sideonly` 参数

**建议**:
- 确认 `sideonly` 的默认值和调用位置
- 如果全面切换到全原子方案，应该删除或禁用 `sideonly=True` 分支

---

### ✅ 已验证: Masked 区域的原子坐标

**位置**: data/interpolant.py:556-612 (Gaussianatoms 模式)

**代码验证**:
```python
# Line 574-579
atoms14_masked[..., 3:, :] = torch.where(
    update_mask_exp,
    torch.zeros_like(atoms14_local[..., 3:, :]),  # Masked 区域：侧链全部置0
    atoms14_local[..., 3:, :]  # Context 区域：保留真实侧链
)
noisy_batch['atoms14_local_t'] = atoms14_masked
```

**结论**: ✅ 正确
- Masked 区域的侧链（索引 3:14）= 0
- Context 区域的侧链（索引 3:14）= 真实坐标
- Backbone (0:3) 始终保留真实坐标

**与模型掩码的配合**:
在 flow_model.py 中：
- Context 区域：使用全部 14 个原子（N,CA,C,O + SC）
- Masked 区域：由于 atoms14_local_t[..., 3:, :] = 0，配合 geom_mask 只保留 N,CA,C

**逻辑一致**: ✅

---

### ✅ 建议 4: 添加断言检查

在关键位置添加 shape 检查：

```python
# datasets.py:150 之前
assert chain_feats['atom14_gt_positions'].shape[-2] == 14, "Expected 14 atoms"
assert dynamic_thickness.shape == (res_mask.shape[0], 1), "Thickness shape mismatch"

# flow_model.py:805 之前
assert all_atoms_global.shape[-2] == 14, "Expected 14 atoms"
assert geom_mask_all.shape[-1] == 14, "Mask should cover 14 atoms"
assert dynamic_thickness.shape[-1] == 1, "Thickness should be [B,N,1]"
```

---

## 7. 检查 Loss 计算

需要检查 Loss 函数是否正确处理全原子椭圆：

```bash
grep -n "gauss_nll\|atom.*_mse" models/loss.py
```

**关键问题**:
- NLL loss 是否正确使用了全原子构建的 Gaussian？
- Per-atom MSE 的索引是否需要调整？

---

## 8. 总结和行动项

### ✅ 正确的部分
1. ✅ `from_rigid_and_all_atoms` 实现正确
2. ✅ 坐标系转换正确 (local → global)
3. ✅ 训练阶段的动态厚度逻辑正确
4. ✅ Masked 区域只用 N,CA,C 的策略合理
5. ✅ 加噪函数正确处理 masked 区域（侧链置0）
6. ✅ 掩码逻辑一致（加噪函数 + 模型初始化）

### 🐛 需要修复
1. **datasets.py:136-140** - 动态厚度逻辑可能反了，或应该用固定值
2. **flow_model.py** - 确认 `sideonly` 参数的使用，避免分支冲突（当前似乎有两个分支）

### ⚠️ 需要验证
1. ✅ **已验证**: 加噪函数正确处理 masked 区域（侧链置0）
2. 确认 `bb_mask` 的语义 (1=存在 or 0=存在?)
3. 检查 Loss 函数是否需要调整（NLL loss 是否正确使用全原子椭圆）

### 📝 建议
1. 添加 shape 断言
2. 统一数据加载和训练的厚度策略
3. 清理或禁用不用的代码分支

---

## 9. 快速验证脚本

```python
# 验证数据加载
from data.datasets import StructureDataset
ds = StructureDataset.from_csv('data/pdb_list.csv')
batch = ds[0]
print("Scaling shape:", batch['rigids_1']._scaling_log.shape)
print("Offset shape:", batch['rigids_1']._local_mean.shape)
print("Offset mean (should be small):", batch['rigids_1']._local_mean.abs().mean())

# 验证模型初始化
# (需要完整的训练脚本)
```
