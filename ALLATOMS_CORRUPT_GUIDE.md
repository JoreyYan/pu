# AllAtoms Corrupt Batch 使用指南

## 📋 概述

新增的 `allatoms_corrupt_batch` 方法同时扰动 **backbone** 和 **sidechain atoms**，用于训练 backbone + atoms14 的全原子扩散模型。

## 🎯 功能

### Backbone 定义（N, CA, C）

- **Backbone = atoms14[:3]** = N, CA, C（3个原子）
- **❌ 不包括 O**（氧原子）
- **Sidechain = atoms14[3:14]** = 11个侧链原子

### 扰动策略

| 部分 | 扰动方法 | Mask | 说明 |
|------|---------|------|------|
| **Backbone** | SE(3) flow matching | `update_mask` | 扰动 `rotmats` 和 `trans` |
| **Sidechain** | R3 flow matching | `update_mask` | 扰动侧链原子的局部坐标 |

**重要**: 主链和侧链使用**同一个 `update_mask`**，确保扰动的残基一致。

## 🔧 方法签名

```python
def allatoms_corrupt_batch(self, batch, prob=None):
    """
    Corrupt both backbone and sidechain atoms simultaneously.

    Args:
        batch: dict with required keys:
            - 'trans_1': [B, N, 3] clean translations
            - 'rotmats_1': [B, N, 3, 3] clean rotations
            - 'atoms14_local': [B, N, 14, 3] clean local coords (backbone + sidechain)
            - 'atom14_gt_exists': [B, N, 14] atom existence mask
            - 'res_mask': [B, N] residue mask
            - 'diffuse_mask': [B, N] diffusion mask
        prob: Optional mask probability for update_mask (default: random in [0.15, 1.0])

    Returns:
        noisy_batch: dict with corrupted backbone and sidechain atoms
    """
```

## 📊 扰动流程

```
Input: Clean structure
    ├─ trans_1: [B, N, 3]
    ├─ rotmats_1: [B, N, 3, 3]
    └─ atoms14_local: [B, N, 14, 3]
        ├─ [:3] = backbone (N, CA, C)
        └─ [3:] = sidechain (11 atoms)
    ↓
Sample t ~ U(min_t, 1-min_t): [B, 1]
Sample update_mask ~ BERT(mask_prob): [B, N]
    ↓
Part 1: Corrupt Backbone (SE(3)) [use update_mask]
    ├─ trans_0 ~ N(0, I) (noise)
    ├─ trans_t = (1-t) * trans_0 + t * trans_1 (linear interpolation)
    ├─ trans_v = trans_1 - trans_0 (velocity field)
    ├─ rotmats_0 ~ SO(3) (noise)
    ├─ rotmats_t = geodesic_t(t, rotmats_1, rotmats_0) (geodesic interpolation)
    └─ rot_v = calc_rot_vf(rotmats_t, rotmats_1) (rotation velocity field)
    ↓
Part 2: Corrupt Sidechain (R3) [use update_mask]
    ├─ noise_sc ~ N(0, coord_scale²)
    ├─ y_sc = (1-t) * noise_sc + t * clean_sc (linear interpolation)
    ├─ v_sc = clean_sc - noise_sc (velocity field)
    └─ Apply only to: sidechain_exists & update_mask
    ↓
Output: Noisy structure
    ├─ Backbone:
    │   ├─ trans_t: [B, N, 3]        (noisy translation)
    │   ├─ trans_0: [B, N, 3]        (noise translation)
    │   ├─ trans_v: [B, N, 3]        (translation velocity field)
    │   ├─ rotmats_t: [B, N, 3, 3]   (noisy rotation)
    │   ├─ rotmats_0: [B, N, 3, 3]   (noise rotation)
    │   └─ rot_v: [B, N, 3]          (rotation velocity field, axis-angle)
    └─ Atoms14:
        ├─ atoms14_local_t: [B, N, 14, 3]
        │   ├─ [:3] = clean backbone (in local frame)
        │   └─ [3:] = noisy sidechain (in local frame)
        └─ v_t: [B, N, 14, 3]        (velocity field for atoms14)
            ├─ [:3] = zeros (backbone clean)
            └─ [3:] = v_sc (sidechain velocity)
```

## 💡 关键设计

### 1. **统一的 y_t/v_t 结构**

Backbone 和 sidechain 现在都使用一致的 y_t (interpolated value) 和 v_t (velocity field) 结构：

**Translation (R3 space)**:
- `trans_0`: noise starting point
- `trans_t = (1-t) * trans_0 + t * trans_1`: interpolated value (y_t)
- `trans_v = trans_1 - trans_0`: velocity field (v_t)

**Rotation (SO(3) space)**:
- `rotmats_0`: noise starting point
- `rotmats_t = geodesic_t(t, rotmats_1, rotmats_0)`: interpolated value (y_t)
- `rot_v = calc_rot_vf(rotmats_t, rotmats_1)`: velocity field in axis-angle (v_t)

**Sidechain (R3 space)**:
- `noise_sc`: noise starting point
- `y_sc = (1-t) * noise_sc + t * clean_sc`: interpolated value (y_t)
- `v_sc = clean_sc - noise_sc`: velocity field (v_t)

这种设计确保了 **backbone 和 sidechain 的扰动数学形式一致**，便于训练和理解。

### 2. **Backbone 在全局坐标系扰动**
- `trans_t` 和 `rotmats_t` 定义了 noisy 的 rigid frame
- Backbone atoms (N, CA, C) 在局部坐标系保持 clean

### 3. **Sidechain 在局部坐标系扰动**
- Sidechain atoms 的局部坐标被加噪：`atoms14_local_t[..., 3:, :]`
- 使用线性插值：`y_t = (1-t) * noise + t * clean`

### 4. **组合方式**
- 在推理时，通过 `rigids_t` 将局部坐标转换为全局坐标：
  ```python
  global_coords = rigids_t.apply(atoms14_local_t)
  ```
- 这样 backbone 和 sidechain 的扰动是解耦的

## 🚀 使用示例

### 1. 基本用法

```python
from data.interpolant import Interpolant
from omegaconf import OmegaConf

# 加载配置
cfg = OmegaConf.load('your_config.yaml')
interpolant = Interpolant(cfg)

# 准备数据
batch = {
    'trans_1': trans_1,           # [B, N, 3]
    'rotmats_1': rotmats_1,       # [B, N, 3, 3]
    'atoms14_local': atoms14,     # [B, N, 14, 3]
    'atom14_gt_exists': atom_mask,# [B, N, 14]
    'res_mask': res_mask,         # [B, N]
    'diffuse_mask': diffuse_mask, # [B, N]
    'res_idx': res_idx,           # [B, N] (optional)
}

# 扰动数据
noisy_batch = interpolant.allatoms_corrupt_batch(batch)

# 输出 - Backbone
trans_t = noisy_batch['trans_t']              # [B, N, 3] noisy translation
trans_0 = noisy_batch['trans_0']              # [B, N, 3] noise translation
trans_v = noisy_batch['trans_v']              # [B, N, 3] translation velocity field
rotmats_t = noisy_batch['rotmats_t']          # [B, N, 3, 3] noisy rotation
rotmats_0 = noisy_batch['rotmats_0']          # [B, N, 3, 3] noise rotation
rot_v = noisy_batch['rot_v']                  # [B, N, 3] rotation velocity field

# 输出 - Atoms14
atoms14_local_t = noisy_batch['atoms14_local_t']  # [B, N, 14, 3] noisy atoms14
v_t = noisy_batch['v_t']                      # [B, N, 14, 3] velocity field for atoms

# 输出 - Other
t = noisy_batch['t']                          # [B, 1] time step
update_mask = noisy_batch['update_mask']      # [B, N] which residues to update
```

### 2. 训练循环

```python
for batch in dataloader:
    # Corrupt batch
    noisy_batch = interpolant.allatoms_corrupt_batch(batch)

    # Model forward
    output = model(noisy_batch)

    # Compute loss
    # Part 1: Backbone loss (trans + rotation)

    # Option A: Predict clean structure (regression to trans_1, rotmats_1)
    pred_trans = output['rigids_global'].get_trans()
    pred_rotmats = output['rigids_global'].get_rots().get_rot_mats()

    trans_loss = F.mse_loss(pred_trans, noisy_batch['trans_1'])
    rot_loss = F.mse_loss(pred_rotmats, noisy_batch['rotmats_1'])

    # Option B: Predict velocity field (flow matching)
    # If your model outputs velocity predictions:
    # pred_trans_v = output['trans_v']  # model predicts velocity
    # pred_rot_v = output['rot_v']
    # trans_v_loss = F.mse_loss(pred_trans_v, noisy_batch['trans_v'])
    # rot_v_loss = F.mse_loss(pred_rot_v, noisy_batch['rot_v'])

    # Part 2: Sidechain loss
    pred_side_atoms = output['side_atoms']  # [B, N, 11, 3]

    # Option A: Predict clean sidechain (regression)
    gt_side_atoms = batch['atoms14_local'][..., 3:, :]  # clean sidechain

    # Option B: Predict velocity field (flow matching)
    # gt_side_v = noisy_batch['v_t'][..., 3:, :]  # sidechain velocity field

    side_loss = F.mse_loss(
        pred_side_atoms,
        gt_side_atoms,
        reduction='none'
    ) * noisy_batch['sidechain_atom_mask'][..., None]

    # Total loss
    loss = trans_loss + rot_loss + side_loss.mean()
    loss.backward()
```

### 3. 与 SideAtomsFlowModel 配合使用

```python
from models.flow_model import SideAtomsFlowModel

# 创建模型
model = SideAtomsFlowModel(config.model)

# 准备 noisy batch
noisy_batch = interpolant.allatoms_corrupt_batch(batch)

# Forward
output = model(noisy_batch)

# 输出
side_atoms = output['side_atoms']           # [B, N, 11, 3] predicted sidechain
atoms_global = output['atoms_global_full']  # [B, N, 14, 3] full structure
rigids = output['rigids_global']            # [B, N, 7] rigid transforms
```

## 📝 输出字段说明

| 字段名 | 形状 | 说明 |
|-------|------|------|
| **Backbone Fields** | | |
| `trans_t` | [B, N, 3] | Noisy translation (y_t for trans) |
| `trans_0` | [B, N, 3] | Noise translation (starting point) |
| `trans_v` | [B, N, 3] | Translation velocity field (trans_1 - trans_0) |
| `rotmats_t` | [B, N, 3, 3] | Noisy rotation (y_t for rotation) |
| `rotmats_0` | [B, N, 3, 3] | Noise rotation (starting point) |
| `rot_v` | [B, N, 3] | Rotation velocity field (axis-angle representation) |
| `rigids_t` | [B, N, 7] | Noisy rigid (7D: quat + trans) |
| **Atoms14 Fields** | | |
| `atoms14_local_t` | [B, N, 14, 3] | Noisy atoms14 (local, y_t for atoms) |
| `y_t` | [B, N, 14, 3] | Alias for atoms14_local_t |
| `v_t` | [B, N, 14, 3] | Velocity field for atoms14 (target) |
| **Time Fields** | | |
| `t` | [B, 1] | Time step |
| `r3_t` | [B, N] | Broadcast time for R3 |
| `so3_t` | [B, N] | Broadcast time for SO(3) |
| **Mask Fields** | | |
| `update_mask` | [B, N] | Which residues to update |
| `sidechain_atom_mask` | [B, N, 11] | Which sidechain atoms exist |
| `diffuse_mask` | [B, N] | Diffusion mask (alias for update_mask) |
| `fixed_mask` | [B, N] | Fixed mask (alias for res_mask) |

## 🔍 与其他方法的对比

| 方法 | Backbone 扰动 | Sidechain 扰动 | 用途 |
|------|------------|--------------|------|
| `corrupt_batch` | ✅ SE(3) | ❌ | 纯 backbone 扩散 |
| `fbb_corrupt_batch` | ❌ | ✅ R3 | 固定 backbone 的侧链设计 |
| **`allatoms_corrupt_batch`** | ✅ SE(3) | ✅ R3 | **全原子扩散（backbone + sidechain）** |

## ⚙️ 配置参数

在 `config.yaml` 中设置：

```yaml
interpolant:
  min_t: 0.001               # 最小时间步
  coord_scale: 1.0           # 侧链噪声缩放
  res_idx_offset_max: 50     # 残基索引偏移范围

  trans:
    corrupt: true            # 是否扰动 translation

  rots:
    corrupt: true            # 是否扰动 rotation
```

## 🎯 推荐的训练策略

### 1. **两阶段训练**

**阶段 1: Backbone only**
```python
# 使用 corrupt_batch 训练纯 backbone
noisy_batch = interpolant.corrupt_batch(batch)
```

**阶段 2: All atoms**
```python
# 使用 allatoms_corrupt_batch 微调全原子
noisy_batch = interpolant.allatoms_corrupt_batch(batch)
```

### 2. **联合训练**

```python
# 随机选择扰动方式
if random.random() < 0.5:
    noisy_batch = interpolant.corrupt_batch(batch)
else:
    noisy_batch = interpolant.allatoms_corrupt_batch(batch)
```

## 🐛 常见问题

### Q1: Backbone atoms 在局部坐标系是 clean 的吗？

**是的。** `atoms14_local_t[:, :, :3, :]` 保持 clean，只有 `[:, :, 3:, :]` 被加噪。

Backbone 的扰动体现在 `trans_t` 和 `rotmats_t` 上。

### Q2: 如何将局部坐标转换为全局坐标？

```python
rigids_t = du.create_rigid(rotmats_t, trans_t)
global_coords = rigids_t.apply(atoms14_local_t)
```

### Q3: update_mask 的作用是什么？

`update_mask` 指定哪些残基需要被扰动（类似 BERT masking）。

**重要**: `update_mask` 同时应用于主链和侧链：
- 主链: 只有 `update_mask=True` 的残基的 `trans` 和 `rotmats` 会被扰动
- 侧链: 只有 `update_mask=True` 的残基的侧链原子会被加噪

这确保了主链和侧链的扰动是**一致的**。

### Q4: 与 SimpleFold 的区别？

SimpleFold 是全原子扩散，所有原子都在**同一坐标系**下扰动。

`allatoms_corrupt_batch` 是**混合扰动**：
- Backbone: 全局坐标系 SE(3)
- Sidechain: 局部坐标系 R3

这种设计更符合蛋白质的物理结构（backbone 定义 frame，sidechain 相对 backbone）。

## 📖 相关文档

- [ESM Integration Guide](ESM_SIDEATOMSFLOW_GUIDE.md)
- [SideAtomsFlowModel Usage](README.md)

## 🎉 总结

`allatoms_corrupt_batch` 提供了一种灵活的方式来训练 **backbone + sidechain** 的联合扩散模型：

- ✅ Backbone 使用成熟的 SE(3) flow matching
- ✅ Sidechain 使用简单的 R3 flow matching
- ✅ 两者解耦，便于训练和调试
- ✅ 支持部分 masking（BERT-style）
- ✅ 与 SideAtomsFlowModel 完美配合

Good luck with your all-atoms diffusion training! 🚀
