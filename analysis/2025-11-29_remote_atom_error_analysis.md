# 远端原子误差过大问题分析

**日期**: 2025-11-29
**问题**: atom5 MSE ≈ 1.8, atom12 MSE ≈ 30（相差15倍）

---

## 问题现象

```
atom03_mse: 0.5   (CA, backbone)
atom04_mse: 0.8   (CB, β-carbon)
atom05_mse: 1.8   (γ-carbon)
...
atom12_mse: 30.0  (远端原子, ζ/η)
atom13_mse: 28.0  (远端原子, θ)
```

**现象**: 距离backbone越远的原子，MSE越大，呈指数增长。

---

## 根本原因分析

### 1. **高斯椭球只约束整体形状，不约束individual atoms**

#### NLL Loss实现 (models/loss.py:544-563)
```python
# 高斯质心（整个侧链的中心）
mu_pred = pred_gaussian.get_gaussian_mean()  # [B, N, 3]

# 协方差矩阵（整个侧链的形状）
sigma_pred = pred_gaussian.get_covariance()  # [B, N, 3, 3]

# 计算每个原子是否落在椭球内
delta = gt_atoms_global - mu_pred.unsqueeze(-2)  # [B, N, 11, 3]
nll_per_atom = 0.5 * (mahal_sq + log_det)
```

**问题**:
- ❌ **sigma_pred是一个3x3矩阵，描述整个侧链的协方差**
- ❌ **所有11个原子共享同一个椭球**
- ❌ **NLL只约束原子云在椭球内，不约束各原子的具体位置**

**类比**: 就像说"这11个点应该在一个椭圆里"，但没说"哪个点应该在哪"

---

### 2. **Coordinate MSE Loss 对所有原子权重相同**

#### MSE Loss实现 (models/loss.py:478-483)
```python
local_mse_loss = backbone_mse_loss(
    atoms14_gt_local,      # [B, N, 14, 3]
    atoms14_pred_local,    # [B, N, 14, 3]
    exists_full_mask,      # [B, N, 14]
    bb_atom_scale=1.0
).mean()
```

**问题**:
- ❌ **atom5和atom12的权重相同（都是1.0）**
- ❌ **但远端原子本身GT方差就大**
- ❌ **网络倾向于优先拟合简单的近端原子**

**误差传播**:
```
atom5 (γ-carbon):  离CA约4Å，GT方差≈1Å  → 易学习
atom12 (ζ/η):      离CA约8Å，GT方差≈3Å  → 难学习
```

由于权重相同，网络选择：
- 精确预测atom5 → 贡献-1.8 loss
- 粗略预测atom12 → 贡献-30 loss

但总loss = 1.8 + 30 = 31.8，平均15.9，看起来还"不错"。

---

### 3. **初始化策略可能导致椭球过大**

#### 当前设置 (flow_model.py:756)
```python
curr_rigids = OffsetGaussianRigid.from_rigid_and_sidechain(
    base_rigid,
    sidechain_atoms_in,
    sidechain_mask_in,
    base_thickness=self.base_thickness * 5  # ← 2.5 Å!
)
```

**问题**:
- ❌ **base_thickness*5 = 2.5Å**，这是个很大的球
- ❌ **所有masked原子初始化为半径2.5Å的球**
- ❌ **网络可能学到"椭球很大，远端原子可以随便放"**

**验证**: 打印训练时的scaling：
```python
print(f"Scaling (σ): {torch.exp(final_gaussian._scaling_log).mean(dim=0)}")
# 如果 σ_x=σ_y=σ_z ≈ 4-5 Å，说明椭球过大
```

---

### 4. **Pairwise Distance Loss 对远端原子约束不足**

#### Pairwise Loss实现 (models/loss.py:485-491)
```python
local_pair_loss = pairwise_distance_loss(
    atoms14_gt_local,
    atoms14_pred_local.clone(),
    exists_full_mask,
    use_huber=False
).mean()
```

**问题**:
- ✅ 约束atom5-atom6距离
- ✅ 约束atom6-atom7距离
- ❌ **但误差会累积！** atom12 = atom5 + ∑(误差)

**误差累积**:
```
atom5:  1.8 Å误差
atom6:  atom5 + 0.5 Å = 2.3 Å
atom7:  atom6 + 0.6 Å = 2.9 Å
...
atom12: atom11 + 最后一跳 ≈ 5-6 Å累积误差
```

平方后: (5-6)² ≈ 25-36 → 符合观察到的MSE≈30

---

## 验证假设

### 假设1: 远端原子GT本身方差大

**测试代码**:
```python
# 统计GT数据中各原子的方差
gt_atoms = batch['atoms14_local'][..., 3:14, :]  # [B, N, 11, 3]
atom_mask = batch['atom14_gt_exists'][..., 3:14]  # [B, N, 11]

for atom_idx in range(11):
    coords = gt_atoms[..., atom_idx, :]  # [B, N, 3]
    mask = atom_mask[..., atom_idx]      # [B, N]

    # 去中心化
    mean = (coords * mask[..., None]).sum() / mask.sum()
    var = ((coords - mean)**2 * mask[..., None]).sum() / mask.sum()
    std = var.sqrt()

    print(f"atom{atom_idx+3:02d} GT std: {std:.2f} Å")
```

**预期结果**:
```
atom03 (CA) GT std: 0.0 Å   (fixed)
atom04 (CB) GT std: 0.5 Å
atom05 (Cγ) GT std: 1.0 Å
...
atom12 (Cζ) GT std: 3.0 Å  ← 远端原子本身就发散！
atom13 (Cη) GT std: 3.5 Å
```

---

### 假设2: 高斯椭球过大

**测试代码**:
```python
# 打印训练中的scaling
final_gaussian = outs['final_gaussian']
scaling_linear = torch.exp(final_gaussian._scaling_log)  # [B, N, 3]

print(f"Gaussian scaling mean: {scaling_linear.mean(dim=(0,1))}")
print(f"Gaussian scaling std:  {scaling_linear.std(dim=(0,1))}")

# 如果均值≈4-5Å，说明椭球很大
```

**预期结果**:
```
Gaussian scaling mean: tensor([4.2, 4.5, 3.8]) Å  ← 太大了！
Ideal:                 tensor([1.5, 1.5, 2.0]) Å  ← 应该更紧凑
```

---

### 假设3: Loss权重不平衡

**当前权重**:
```python
coord_loss = 1.0 * local_mse_loss  # 所有原子平等
           + 1.0 * local_pair_loss
           + 1.0 * local_huber_loss

total_loss = 1.0 * coord_loss      # atom_loss_weight
           + 5.0 * loss_param      # w_param
           + 0.0003 * loss_nll     # w_nll ← 太小！
```

**问题**: NLL loss权重0.0003，几乎不起作用！

**验证**:
```
coord_loss: 15.0  → weighted: 15.0
gauss_param_mse: 0.5 → weighted: 2.5
gauss_nll: 1.5    → weighted: 0.00045  ← 忽略不计！
```

---

## 解决方案

### 方案1: **Per-Atom Weighted Loss** ⭐ 推荐

根据原子到CA的距离动态加权：

```python
# models/loss.py, 修改MSE计算

# 定义per-atom权重（近端原子权重高）
atom_weights = torch.tensor([
    1.0,   # atom3 (CA)
    2.0,   # atom4 (CB)
    3.0,   # atom5 (Cγ)
    4.0,   # atom6 (Cδ)
    5.0,   # atom7 (Cε)
    6.0,   # atom8 (Cζ)
    7.0,   # atom9 (Cη)
    8.0,   # atom10
    9.0,   # atom11
    10.0,  # atom12 ← 远端原子权重最高！
    11.0,  # atom13
], device=atoms14_pred_local.device)

# 应用权重
weighted_sq_error = atom_sq_error * atom_weights
local_mse_loss = (weighted_sq_error * exists_full_mask).sum() / exists_full_mask.sum()
```

**效果**:
- 强迫网络关注远端原子
- atom12的30 Å² MSE会被×10权重 → 300 loss contribution
- 网络必须降低远端误差才能降低总loss

---

### 方案2: **减小初始化椭球**

```python
# flow_model.py:756
curr_rigids = OffsetGaussianRigid.from_rigid_and_sidechain(
    base_rigid,
    sidechain_atoms_in,
    sidechain_mask_in,
    base_thickness=self.base_thickness * 2  # 从5改成2 → 1.0Å
)
```

**效果**:
- 初始椭球更紧凑
- 网络从"小球"学起，而不是"大球"
- 更符合真实侧链形状

---

### 方案3: **增加NLL Loss权重**

```python
# Train_esmsd.yaml
w_nll: 0.003  # 从0.0003提升10倍
```

**效果**:
- NLL loss从0.00045 → 0.0045（贡献增加10倍）
- 强化"原子应该在椭球内"的约束
- 但仍然不约束individual atom positions

---

### 方案4: **Per-Atom Gaussian** (根本解决) 🔥

**问题根源**: 一个椭球描述11个原子 → 信息不足

**彻底方案**: 为每个原子预测独立的高斯分布

```python
# SidechainAtomHead改造
def forward(self, s, rigid_backbone, sidechain_mask):
    # 预测per-atom参数 [B, N, 11, 9]
    # 9维: 3 (mean) + 3 (scaling_log) + 3 (rotation_log)
    per_atom_params = self.projection(s).view(B, N, 11, 9)

    # 构建11个独立的高斯
    gaussians = []
    for atom_idx in range(11):
        local_mean = per_atom_params[..., atom_idx, 0:3]
        scaling_log = per_atom_params[..., atom_idx, 3:6]
        # ... 构建per-atom Gaussian
        gaussians.append(...)

    return gaussians
```

**NLL Loss改造**:
```python
# 为每个原子单独计算NLL
for atom_idx in range(11):
    sigma_atom = gaussians[atom_idx].get_covariance()  # [B, N, 3, 3]
    mu_atom = gaussians[atom_idx].get_gaussian_mean()  # [B, N, 3]

    delta = gt_atoms_global[..., atom_idx, :] - mu_atom
    nll = 0.5 * (mahal_sq + log_det)
    loss_nll += nll
```

**效果**:
- ✅ 每个原子有自己的高斯椭球
- ✅ NLL直接约束individual atom positions
- ✅ atom12有自己的椭球，不再"随大流"

**代价**:
- 参数量增加: 11×9 = 99维 per residue
- 计算量增加: 11×NLL loss

---

## 推荐实施顺序

### 阶段1: 快速修复（1小时）
1. ✅ **方案1: Per-Atom Weighted Loss** - 立即见效
2. ✅ **方案2: 减小初始化椭球** (5→2) - 改一行代码

### 阶段2: 深度优化（1天）
3. ⚠️ **方案3: 调整NLL权重** - 需要实验找最优值
4. ⚠️ **验证假设1-3** - 统计GT分布，打印训练指标

### 阶段3: 架构升级（3天）
5. 🔥 **方案4: Per-Atom Gaussian** - 彻底解决问题

---

## 立即行动

### 快速测试: Per-Atom Weighted Loss

```python
# models/loss.py:478前添加

# Per-atom weights (远端原子权重高)
atom_distance_to_ca = torch.tensor([
    0.0,   # atom0 (N)
    0.0,   # atom1 (CA)
    0.0,   # atom2 (C)
    1.5,   # atom3 (CB)
    2.5,   # atom4
    3.5,   # atom5
    4.5,   # atom6
    5.5,   # atom7
    6.5,   # atom8
    7.5,   # atom9
    8.0,   # atom10
    8.5,   # atom11
    9.0,   # atom12
    9.5,   # atom13
], device=atoms14_pred_local.device)

# 权重 = exp(distance / 3.0) → 远端指数增长
atom_weights = torch.exp(atom_distance_to_ca / 3.0)
# atom3: exp(1.5/3)=1.6,  atom12: exp(9/3)=20.0

# 应用到sq_error
weighted_sq_error = atom_sq_error * atom_weights[None, None, :]  # broadcast
weighted_mse = (weighted_sq_error * exists_full_mask).sum() / (exists_full_mask * atom_weights[None, None, :]).sum()
```

**预期效果**:
- atom5 MSE: 1.8 → **0.8** (改善55%)
- atom12 MSE: 30 → **5.0** (改善83%)

---

**总结**: 问题的根源是**一个椭球约束11个原子**，导致远端原子"钻空子"。短期用weighted loss，长期改per-atom Gaussian。
