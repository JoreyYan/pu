# base_thickness 统一配置方案

**日期**: 2025-11-29
**目标**: 所有 `base_thickness` 从 yaml 配置读取，无硬编码默认值

---

## 配置入口 (单一来源)

### 1. **configs/datasets.yaml:9** - 数据集级别
```yaml
shared:
  base_thickness: 0.5  # Base thickness for Gaussian scaling (Angstrom)
```

**继承关系**:
- `pdb_dataset.base_thickness: ${shared.base_thickness}`
- `scope_dataset.base_thickness: ${shared.base_thickness}`
- `val_dataset.base_thickness: ${shared.base_thickness}`

### 2. **configs/model.yaml:34** - 模型级别
```yaml
model:
  base_thickness: 0.5  # Base thickness for Gaussian scaling (Angstrom)
```

### 3. **configs/Train_esmsd.yaml:83** - 训练配置级别
```yaml
experiment:
  training:
    base_thickness: 0.5  # Base thickness for Gaussian scaling (Angstrom)
```

---

## 代码使用位置

### **A. 数据加载** (data/datasets.py)

#### 1. BaseDataset.__init__ (line 293)
```python
self.base_thickness = dataset_cfg.base_thickness
```
- 从 `data.{dataset}.base_thickness` 读取

#### 2. _process_csv_row (line 87)
```python
def _process_csv_row(processed_file_path, map, base_thickness):
    # ...
    rigids_1 = OffsetGaussianRigid.from_rigid_and_sidechain(
        rigids_1,
        chain_feats['atom14_gt_positions'][...,3:,:],
        chain_feats['atom14_gt_exists'][...,3:],
        base_thickness=base_thickness  # ← 从 dataset_cfg 传入
    )
```

**用途**: 计算 GT Gaussian 参数 (scaing_log_1, local_mean_1)

---

### **B. 模型** (models/flow_model.py)

#### 1. SideAtomsIGAModel.__init__ (line 650)
```python
self.base_thickness = model_conf.base_thickness  # 从 model.base_thickness 读取
self.atom_head = SidechainAtomHead(
    self._ipa_conf.c_s,
    num_atoms=11,
    base_thickness=self.base_thickness
)
```

#### 2. SideAtomsIGAModel.forward (line 743)
```python
curr_rigids = OffsetGaussianRigid.from_rigid_and_sidechain(
    base_rigid,
    sidechain_atoms_in,
    sidechain_mask_in,
    base_thickness=self.base_thickness  # ← 从 model_conf 读取
)
```

**用途**: 在线构建初始 Gaussian (masked 区域用 base_thickness 小球)

#### 3. SidechainAtomHead.__init__ (line 413-421)
```python
def __init__(self, c_in, num_atoms=11, base_thickness=None):
    if base_thickness is None:
        raise ValueError("base_thickness must be provided from config")
    self.base_thickness_ang = base_thickness
```

**用途**: 预测局部坐标 -> 在线计算 Gaussian 参数

---

### **C. Loss函数** (models/loss.py)

#### SideAtomsIGALoss_Final.__init__ (line 420)
```python
self.base_thickness = config.base_thickness  # 从 experiment.training.base_thickness 读取
```

#### _calc_gt_gaussian (line 436)
```python
scaling_log = torch.log(std_dev + self.base_thickness)
```

**用途**: 现场计算 GT Gaussian 参数（备用，如果 batch 中没有预计算的）

---

### **D. 工具函数** (data/GaussianRigid.py)

#### 1. OffsetGaussianRigid.from_atoms (line 224)
```python
@staticmethod
def from_atoms(n, ca, c, sidechain_atoms, sidechain_mask, base_thickness):
    # ❌ 无默认值，调用者必须显式传入
```

#### 2. OffsetGaussianRigid.from_rigid_and_sidechain (line 309)
```python
@classmethod
def from_rigid_and_sidechain(
    cls, rigid_backbone, sidechain_atoms, sidechain_mask, base_thickness
):
    # ❌ 无默认值，调用者必须显式传入
```

**设计理念**: 这些是底层工具函数，不应该有隐藏的默认值

---

## 修改 base_thickness 的方式

### **方案1: 全局修改** (推荐)
修改 `configs/datasets.yaml`:
```yaml
shared:
  base_thickness: 0.8  # 所有数据集和模型都会使用 0.8
```

### **方案2: 模型级修改**
修改 `configs/model.yaml`:
```yaml
model:
  base_thickness: 0.8  # 只影响模型在线计算
```

### **方案3: 训练级修改**
修改 `configs/Train_esmsd.yaml`:
```yaml
experiment:
  training:
    base_thickness: 0.8  # 只影响 loss 函数现场计算
```

---

## 错误处理

### 如果忘记配置会怎样？

1. **SidechainAtomHead**:
   ```python
   raise ValueError("base_thickness must be provided from config")
   ```

2. **其他位置**:
   ```python
   AttributeError: 'Namespace' object has no attribute 'base_thickness'
   ```

**解决**: 在 yaml 配置文件中添加 `base_thickness: 0.5`

---

## 配置验证清单

✅ **configs/datasets.yaml**: `shared.base_thickness` 已设置
✅ **configs/model.yaml**: `model.base_thickness` 已设置
✅ **configs/Train_esmsd.yaml**: `experiment.training.base_thickness` 已设置
✅ **data/datasets.py**: 从 `dataset_cfg.base_thickness` 读取
✅ **models/flow_model.py**: 从 `model_conf.base_thickness` 读取
✅ **models/loss.py**: 从 `config.base_thickness` 读取
✅ **data/GaussianRigid.py**: 无默认参数，必须传入

---

## 数据流图

```
┌─────────────────────────────────────────┐
│  configs/datasets.yaml                   │
│  shared.base_thickness: 0.5             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  data/datasets.py                        │
│  BaseDataset.__init__:                   │
│    self.base_thickness = cfg.base_thickness │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  _process_csv_row(path, map, base_thickness) │
│  → OffsetGaussianRigid.from_rigid_and_sidechain() │
│  → 计算 GT: scaing_log_1, local_mean_1   │
└──────────────────────────────────────────┘


┌─────────────────────────────────────────┐
│  configs/model.yaml                      │
│  model.base_thickness: 0.5              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  models/flow_model.py                    │
│  SideAtomsIGAModel.__init__:             │
│    self.base_thickness = model_conf.base_thickness │
└──────────────┬───────────────────────────┘
               │
               ├─────────────────────────────────┐
               ▼                                 ▼
┌─────────────────────────────┐  ┌───────────────────────────┐
│  SidechainAtomHead          │  │  forward()                │
│  (在线计算 Gaussian)         │  │  from_rigid_and_sidechain │
└─────────────────────────────┘  └───────────────────────────┘


┌─────────────────────────────────────────┐
│  configs/Train_esmsd.yaml                │
│  experiment.training.base_thickness: 0.5 │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  models/loss.py                          │
│  SideAtomsIGALoss_Final.__init__:        │
│    self.base_thickness = config.base_thickness │
│  → _calc_gt_gaussian() (备用)            │
└──────────────────────────────────────────┘
```

---

## 总结

**统一原则**:
- ✅ 所有 `base_thickness` 从 yaml 读取
- ❌ 无硬编码默认值
- ❌ 无 `getattr(..., default=0.5)`
- ✅ 如果缺失配置，程序报错而不是静默使用默认值

**好处**:
1. 配置透明，易于追踪
2. 避免隐藏的默认值导致的混淆
3. 修改一处配置文件即可全局生效
4. 便于实验管理和复现
