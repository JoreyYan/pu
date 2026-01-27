# 高斯椭圆构建：从侧链中心切换到全原子质心

**日期**: 2025-11-30
**修改**: 将 `from_rigid_and_sidechain` 改为 `from_all_atoms`
**影响**: 椭圆质心从侧链中心变为全原子质心（N,CA,C,O + 侧链）

---

## 1. 动机

### 原方案 (from_rigid_and_sidechain)

**椭圆构建**:
- **Rotation (R)**: 主链 Frame (N-CA-C)
- **Translation (t)**: CA 坐标
- **Centroid (μ)**: **只用侧链原子**的质心
- **Scaling (S)**: 侧链原子相对质心的标准差
- **Offset**: 侧链质心相对 CA 的位移

**问题**:
1. **小氨基酸 (Gly, Ala)** 侧链原子少，椭圆很小 → overlap 弱
2. **侧链质心偏移大**: 对于长侧链 (Arg, Lys)，质心远离 CA
3. **骨架原子未参与**: N,CA,C,O 不影响椭圆形状，但它们是最稳定的部分

### 新方案 (from_all_atoms)

**椭圆构建**:
- **Rotation (R)**: 主链 Frame (N-CA-C)
- **Translation (t)**: CA 坐标
- **Centroid (μ)**: **全原子质心** (N,CA,C,O + 侧链)
- **Scaling (S)**: 全原子相对质心的标准差
- **Offset**: 全原子质心相对 CA 的位移

**优势**:
1. **Gly/Ala 也有合理尺寸**: 骨架4原子保证最小椭圆体积
2. **质心更靠近 CA**: 骨架原子拉平侧链的偏移
3. **物理更合理**: 质心是真正的"质量中心"，不只是侧链的几何中心

---

## 2. 代码修改

### 2.1 data/datasets.py (数据加载)

**位置**: Line 129-151

**旧代码** (已注释):
```python
rigids_1 = OffsetGaussianRigid.from_rigid_and_sidechain(
    rigids_1,
    chain_feats['atom14_gt_positions'][..., 3:, :],  # sidechain only
    chain_feats['atom14_gt_exists'][..., 3:],
    base_thickness=base_thickness
)
```

**新代码**:
```python
# atom14 format: 0=N, 1=CA, 2=C, 3=O/CB, 4-13=sidechain
n_coords = chain_feats['atom14_gt_positions'][..., 0, :]
ca_coords = chain_feats['atom14_gt_positions'][..., 1, :]
c_coords = chain_feats['atom14_gt_positions'][..., 2, :]
o_coords = chain_feats['atom14_gt_positions'][..., 3, :]  # O or CB
rigids_1 = OffsetGaussianRigid.from_all_atoms(
    n_coords,
    ca_coords,
    c_coords,
    o_coords,
    chain_feats['atom14_gt_positions'][..., 4:, :],  # sidechain (atoms 4-13)
    chain_feats['atom14_gt_exists'][..., 4:],
    base_thickness=base_thickness
)
```

**关键变化**:
- 侧链索引从 `[3:]` 改为 `[4:]` (因为 atom 3 是 O/CB，属于骨架)
- 提取 N, CA, C, O 作为显式参数

---

### 2.2 models/flow_model.py (模型初始化)

**位置**: Line 730-794

**旧代码** (已注释):
```python
sidechain_atoms_in = atoms14_in[..., 3:14, :]  # [B, N, 11, 3]
curr_rigids = OffsetGaussianRigid.from_rigid_and_sidechain(
    base_rigid,
    sidechain_atoms_in,
    sidechain_mask_in,
    base_thickness=self.base_thickness*5
)
```

**新代码**:
```python
# 提取骨架原子 (local 坐标)
n_atoms = atoms14_in[..., 0, :]
ca_atoms = atoms14_in[..., 1, :]
c_atoms = atoms14_in[..., 2, :]
o_atoms = atoms14_in[..., 3, :]
sidechain_atoms_in = atoms14_in[..., 4:14, :]  # [B, N, 10, 3] (atoms 4-13)

# 转换为 global 坐标 (from_all_atoms 需要 global)
base_rigid_expanded = base_rigid.unsqueeze(-1)
n_global = base_rigid_expanded.apply(n_atoms)
ca_global = base_rigid_expanded.apply(ca_atoms)
c_global = base_rigid_expanded.apply(c_atoms)
o_global = base_rigid_expanded.apply(o_atoms)
sidechain_global = base_rigid_expanded.apply(sidechain_atoms_in)

curr_rigids = OffsetGaussianRigid.from_all_atoms(
    n_global,
    ca_global,
    c_global,
    o_global,
    sidechain_global,
    sidechain_mask_in,
    base_thickness=self.base_thickness*5
)
```

**关键变化**:
- 侧链索引从 `[3:14]` (11个) 改为 `[4:14]` (10个)
- **坐标转换**: `atoms14_local_t` 是 local 坐标，需要 `base_rigid.apply()` 转为 global
- 所有原子 (骨架 + 侧链) 都转换为 global 后传给 `from_all_atoms`

---

## 3. 预期效果

### 3.1 椭圆尺寸变化

**Glycine (Gly)** - 最小氨基酸:
- **旧方案**: 只有 1-2 个侧链原子 → 非常小的椭圆 → overlap 弱
- **新方案**: 4个骨架 + 2个侧链 = 6原子 → 合理尺寸的椭圆 ✅

**Alanine (Ala)**:
- **旧方案**: CB 单原子 → 球形椭圆
- **新方案**: 4骨架 + CB → 更大椭圆 ✅

**Arginine (Arg)** - 长侧链:
- **旧方案**: 侧链质心远离 CA → 大偏移 (local_mean 大)
- **新方案**: 骨架原子拉近质心 → 偏移减小 ✅

### 3.2 Attention Geometric Bias

**期望**:
- **更多 overlap**: 相邻残基的椭圆更容易相交
- **更稳定的 bias**: 不再过度依赖侧链构象
- **小残基也有贡献**: Gly/Ala 不再是"看不见"的点

**验证方法**:
1. 可视化 `geo_bias` 矩阵: 应该看到更多负值（更多 overlap）
2. 比较 `local_attn / global_attn` 比例: 应该更高
3. 检查 Gly/Ala 的 attention 权重: 应该不再异常低

### 3.3 Loss 和指标

**可能影响**:
- **NLL loss**: 可能略微上升（椭圆更大，拟合更松）
- **Param MSE**: offset 幅度减小，MSE 可能下降
- **Atom MSE**: 特别是小残基的远端原子，应该更稳定
- **TM-score**: 结构整体质量可能提升

---

## 4. 验证步骤

### 4.1 快速测试

```bash
# 测试数据加载
python -c "
from data.datasets import StructureDataset
ds = StructureDataset.from_csv('data/pdb_list.csv')
batch = ds[0]
print('Scaling shape:', batch['rigids_1']._scaling_log.shape)
print('Offset shape:', batch['rigids_1']._local_mean.shape)
print('Offset mean:', batch['rigids_1']._local_mean.abs().mean())
"
```

**预期**:
- Offset mean 应该比旧版本**更小**（质心更靠近 CA）

### 4.2 可视化对比

保存旧版和新版的椭圆到 PDB：

```python
from data.GaussianRigid import save_gaussian_as_pdb

# 旧版 (sidechain-only)
# rigids_old = OffsetGaussianRigid.from_rigid_and_sidechain(...)
# save_gaussian_as_pdb(rigids_old, "old_sidechain.pdb")

# 新版 (all-atoms)
rigids_new = OffsetGaussianRigid.from_all_atoms(...)
save_gaussian_as_pdb(rigids_new, "new_allatoms.pdb")
```

用 PyMOL 对比：
- 椭圆是否更大？
- Gly/Ala 是否有合理尺寸？
- 质心是否更靠近 CA？

### 4.3 训练监控

**关键指标**:
- `train/gauss_nll`: 监控变化（可能略微上升）
- `train/param_mse`: offset 部分应该下降
- `train/atom03_mse` ~ `atom13_mse`: 远端原子 MSE 应该更稳定
- Attention 可视化: `geo_bias` 应该更负（更多 overlap）

---

## 5. 回滚方案

如果新方案效果不佳，可以快速回滚：

**data/datasets.py**:
```python
# 取消注释旧版代码 (line 129-135)
rigids_1 = OffsetGaussianRigid.from_rigid_and_sidechain(
    rigids_1,
    chain_feats['atom14_gt_positions'][..., 3:, :],
    chain_feats['atom14_gt_exists'][..., 3:],
    base_thickness=base_thickness
)
```

**models/flow_model.py**:
```python
# 取消注释旧版代码 (line 768-774)
sidechain_atoms_in = atoms14_in[..., 3:14, :]
curr_rigids = OffsetGaussianRigid.from_rigid_and_sidechain(
    base_rigid,
    sidechain_atoms_in,
    sidechain_mask_in,
    base_thickness=self.base_thickness*5
)
```

---

## 6. 相关文档

- **Base thickness config**: analysis/base_thickness_config.md
- **Remote atom error**: analysis/2025-11-29_remote_atom_error_analysis.md
- **Attention visualization**: analysis/2025-11-29_attention_visualization_guide.md
- **GaussianRigid API**: data/GaussianRigid.py (line 224, 309, 357)

---

## 7. 技术细节

### atom14 格式

AlphaFold2 / SimpleFold 的 atom14 格式：
- **0**: N (骨架)
- **1**: CA (骨架)
- **2**: C (骨架)
- **3**: O 或 CB (取决于氨基酸类型)
- **4-13**: 侧链原子 (最多10个)

**注意**:
- Gly 没有 CB，atom 3 是 O
- 其他氨基酸 atom 3 可能是 CB 或 O（取决于具体实现）
- 本修改将 atom 3 视为骨架的一部分（O 或 CB 都应参与质心计算）

### 坐标系转换

**datasets.py**:
- `atom14_gt_positions` 是 **global** 坐标
- 直接传给 `from_all_atoms` ✅

**flow_model.py**:
- `atoms14_local_t` 是 **local** 坐标（相对于骨架 frame）
- 需要 `base_rigid.apply()` 转换为 global ✅
- `from_all_atoms` 内部会重新计算 local offset

---

## 8. 已知问题和待办

### 待验证
- [ ] Gly/Ala 的椭圆尺寸是否合理
- [ ] Offset 幅度是否真的减小
- [ ] Geometric bias overlap 是否增强
- [ ] 对 loss 和指标的实际影响

### 潜在问题
- **O vs CB**: atom14[3] 对不同氨基酸可能是 O 或 CB，需要确认数据格式
- **Fallback 分支**: flow_model.py 的 fallback (全0初始化) 需要测试

---

## 9. 总结

这个修改将高斯椭圆的构建从"侧链中心"改为"全原子质心"，预期能：
1. 解决小氨基酸椭圆过小的问题
2. 增强 IGA 的 geometric bias（更多 overlap）
3. 使质心更靠近 CA（offset 更小，更稳定）

所有旧代码已注释保留，可随时回滚。
