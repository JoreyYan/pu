# 2025-11-17: 几何约束设计方案

## 问题背景

**当前问题**：
- R3的velocity场学得不准，导致步数越多越差
- 从诊断报告看，C-N键、C-O键误差达10-12%
- 长侧链末端（LYS/ARG）有极端outlier（应该1.47Å，实际5.3Å）

**用户洞察**：
> "需要加上一些键长和键角的约束，但键角不能直接从输出坐标计算，容易NaN"

**核心挑战**：
- 键角计算涉及`arccos`，输入必须在[-1, 1]
- 如果三个原子接近共线或坐标有大误差，数值不稳定
- 直接计算angle再加loss很容易出现NaN/Inf

---

## 为什么键角计算容易NaN？

### 标准键角计算

```python
def compute_angle(p1, p2, p3):
    """计算 p1-p2-p3 的键角"""
    v1 = p1 - p2  # [B, N, 3]
    v2 = p3 - p2

    # 归一化
    v1_norm = v1 / v1.norm(dim=-1, keepdim=True)  # ⚠️ 可能除零
    v2_norm = v2 / v2.norm(dim=-1, keepdim=True)  # ⚠️ 可能除零

    # 点积
    cos_angle = (v1_norm * v2_norm).sum(dim=-1)    # ⚠️ 可能超出[-1,1]

    # 反余弦
    angle = torch.acos(cos_angle)                   # ⚠️ NaN如果超出范围

    return angle
```

### 三个危险点

**危险1：向量长度接近0**
```python
v1.norm() ≈ 0  →  除零  →  Inf  →  NaN
```
发生场景：两个原子重叠或非常接近（训练早期常见）

**危险2：点积超出范围**
```python
cos_angle = 1.0000001  →  acos(>1)  →  NaN
cos_angle = -1.0000001 →  acos(<-1) →  NaN
```
发生场景：浮点数精度问题

**危险3：接近共线**
```python
cos_angle ≈ 1 或 -1  →  acos梯度 → ∞  →  不稳定
```
发生场景：三个原子接近一条直线

---

## 解决方案矩阵

| 方案 | 稳定性 | 表达力 | 计算成本 | 实现难度 |
|------|--------|--------|----------|----------|
| **A. Cosine Loss** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 低 | 简单 |
| **B. 1-3 Distance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 低 | 简单 |
| **C. Internal Coords** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 高 | 复杂 |
| **D. Soft Constraints** | ⭐⭐⭐⭐ | ⭐⭐⭐ | 低 | 简单 |
| **E.分层约束** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中 | 中等 |

推荐：**A + B + E** 的组合

---

## 方案A: Cosine Loss（推荐）

### 核心思想
**不计算angle，直接约束cosine值**

```python
def stable_angle_loss(p1, p2, p3, target_angle_rad, mask=None):
    """
    稳定的键角loss

    Args:
        p1, p2, p3: [B, N, 3] 三个原子坐标
        target_angle_rad: [B, N] 目标角度（弧度）
        mask: [B, N] 哪些位置需要计算

    Returns:
        loss: scalar
    """
    # 向量
    v1 = p1 - p2  # [B, N, 3]
    v2 = p3 - p2

    # 长度（带epsilon避免除零）
    eps = 1e-7
    len1 = v1.norm(dim=-1, keepdim=True).clamp(min=eps)  # [B, N, 1]
    len2 = v2.norm(dim=-1, keepdim=True).clamp(min=eps)

    # 归一化向量
    v1_norm = v1 / len1
    v2_norm = v2 / len2

    # 计算cosine（带clamp保证在[-1, 1]）
    cos_pred = (v1_norm * v2_norm).sum(dim=-1)  # [B, N]
    cos_pred = torch.clamp(cos_pred, -1.0 + 1e-6, 1.0 - 1e-6)

    # 目标cosine
    cos_target = torch.cos(target_angle_rad)  # [B, N]

    # MSE loss on cosine（不是angle！）
    loss_per_angle = (cos_pred - cos_target) ** 2

    # 应用mask
    if mask is not None:
        loss_per_angle = loss_per_angle * mask
        loss = loss_per_angle.sum() / mask.sum().clamp(min=1)
    else:
        loss = loss_per_angle.mean()

    return loss
```

### 优点
✅ **绝对稳定**：不使用`acos`，避免NaN
✅ **梯度友好**：cosine的梯度平滑，不会爆炸
✅ **物理意义明确**：cos(θ)直接反映角度偏离
✅ **计算高效**：只需点积和归一化

### 缺点
⚠️ 需要预先知道target angle（但这可以从rotamer library获得）

### 使用示例
```python
# 约束CA-CB-CG键角（典型值109.5°，四面体角）
target_angle = torch.tensor(109.5 * math.pi / 180)  # 弧度
loss_angle = stable_angle_loss(
    CA_coords,   # [B, N, 3]
    CB_coords,
    CG_coords,
    target_angle.expand(B, N),
    mask=has_CG_mask  # [B, N] bool
)
```

---

## 方案B: 1-3 Distance约束

### 核心思想
**通过1-3距离隐式约束键角**

几何关系：
```
    p1
    |
    | d12 (键长1)
    |
    p2 --- p3
    θ      d23 (键长2)

d13 = sqrt(d12² + d23² - 2*d12*d23*cos(θ))  (余弦定理)
```

如果约束：
- d12 ≈ 1.5Å
- d23 ≈ 1.5Å
- d13 ≈ 2.5Å

则隐式地约束了θ ≈ 109.5°

### 实现

```python
def distance_constraint_loss(coords, bond_pairs, target_dists, mask=None):
    """
    通用的距离约束loss

    Args:
        coords: [B, N, 14, 3] 所有原子坐标
        bond_pairs: List[(i, j)] 需要约束的原子对索引
        target_dists: List[float] 对应的目标距离
        mask: [B, N] residue mask

    Returns:
        loss: scalar
    """
    losses = []

    for (atom_i, atom_j), target_dist in zip(bond_pairs, target_dists):
        # 提取原子坐标
        pi = coords[..., atom_i, :]  # [B, N, 3]
        pj = coords[..., atom_j, :]

        # 计算距离
        dist_pred = (pi - pj).norm(dim=-1)  # [B, N]

        # Huber loss（比MSE更robust）
        delta = 0.1  # 0.1Å的linear区域
        diff = torch.abs(dist_pred - target_dist)
        loss_per_pair = torch.where(
            diff < delta,
            0.5 * diff ** 2,
            delta * (diff - 0.5 * delta)
        )

        if mask is not None:
            loss_per_pair = loss_per_pair * mask
            losses.append(loss_per_pair.sum() / mask.sum().clamp(min=1))
        else:
            losses.append(loss_per_pair.mean())

    return sum(losses) / len(losses)
```

### 定义约束对

```python
# 氨基酸通用约束
BOND_CONSTRAINTS = {
    # 1-2键（直接连接）
    '1-2': [
        ((1, 4), 1.54),   # CA-CB
        ((4, 5), 1.52),   # CB-CG (对于有CG的氨基酸)
    ],
    # 1-3距离（隐式键角）
    '1-3': [
        ((1, 5), 2.54),   # CA-CG，假设CA-CB-CG是109.5°
    ],
}

# 使用
loss_bond = distance_constraint_loss(
    coords,
    BOND_CONSTRAINTS['1-2'],
    [d for (_, d) in BOND_CONSTRAINTS['1-2']],
    mask=res_mask
)

loss_angle_implicit = distance_constraint_loss(
    coords,
    BOND_CONSTRAINTS['1-3'],
    [d for (_, d) in BOND_CONSTRAINTS['1-3']],
    mask=res_mask
)
```

### 优点
✅ **极其稳定**：只有减法和norm，不会NaN
✅ **已有实现**：代码中已有`pairwise_distance_loss`
✅ **隐式约束角度**：通过余弦定理

### 缺点
⚠️ 不够精确：只能约束常见的109.5°或120°等
⚠️ 需要手工定义约束对

---

## 方案C: Internal Coordinates（终极方案）

### 核心思想
**预测键长、键角、二面角，再转换为笛卡尔坐标**

```python
class InternalCoordinatePredictor(nn.Module):
    def forward(self, features):
        """
        输出内坐标而非笛卡尔坐标

        Returns:
            bond_lengths: [B, N, 11] 每个侧链原子到前一个的键长
            bond_angles: [B, N, 11] 每个键角
            torsions: [B, N, 11] 每个二面角
        """
        # 预测内坐标
        bond_lengths = self.bond_head(features)  # [B, N, 11]
        bond_angles = self.angle_head(features)  # [B, N, 11]
        torsions = self.torsion_head(features)   # [B, N, 11]

        # 通过Z-matrix转换为笛卡尔坐标
        coords = internal_to_cartesian(
            bond_lengths, bond_angles, torsions,
            backbone_coords  # 固定的参考系
        )

        return coords
```

### Z-matrix转换

```python
def internal_to_cartesian(bond_lengths, bond_angles, torsions, ref_coords):
    """
    从内坐标构建笛卡尔坐标（Z-matrix方法）

    Args:
        bond_lengths: [B, N, K] K个侧链原子的键长
        bond_angles: [B, N, K] K个键角
        torsions: [B, N, K] K个二面角
        ref_coords: [B, N, 3, 3] 参考原子(N, CA, C)

    Returns:
        sidechain_coords: [B, N, K, 3] 笛卡尔坐标
    """
    B, N, K = bond_lengths.shape
    coords = []

    # CB: 从CA出发，沿CA-N和CA-C定义的方向
    # （详细实现需要参考蛋白质几何学）
    CB = build_atom_from_internal(
        ref_coords[..., 1, :],  # CA
        ref_coords[..., 0, :],  # N
        ref_coords[..., 2, :],  # C
        bond_lengths[..., 0],   # CA-CB键长
        bond_angles[..., 0],    # N-CA-CB角
        torsions[..., 0]        # N-CA-CB-X二面角
    )
    coords.append(CB)

    # CG: 从CB出发，依次构建
    for k in range(1, K):
        atom_k = build_atom_from_internal(
            coords[-1],           # 前一个原子
            coords[-2] if len(coords) > 1 else ref_coords[..., 1, :],
            coords[-3] if len(coords) > 2 else ref_coords[..., 0, :],
            bond_lengths[..., k],
            bond_angles[..., k],
            torsions[..., k]
        )
        coords.append(atom_k)

    return torch.stack(coords, dim=-2)  # [B, N, K, 3]

def build_atom_from_internal(p1, p2, p3, length, angle, torsion):
    """NeRF (Natural Extension Reference Frame) 方法"""
    # 建立局部坐标系
    v1 = p1 - p2
    v2 = p3 - p2

    # ... (详细的向量几何计算)
    # 返回新原子的坐标
    return new_coord
```

### 优点
✅ **自动满足约束**：键长/键角由设计保证
✅ **物理意义明确**：符合化学直觉
✅ **紧凑表示**：11个原子 = 11键长 + 11键角 + 11二面角

### 缺点
❌ **实现复杂**：Z-matrix转换需要careful实现
❌ **梯度复杂**：坐标对内坐标的雅可比矩阵
❌ **需要重新训练**：完全改变输出格式
⚠️ **可能不适合flow matching**：velocity的物理意义变了

### 推荐场景
- 如果要从头重新设计架构 → 考虑这个
- 如果要快速改进现有模型 → 不推荐

---

## 方案D: Soft Constraints（渐进式）

### 核心思想
**只在严重偏离时惩罚，允许一定误差**

```python
def soft_constraint_loss(pred_value, target_value, tolerance=0.1):
    """
    Soft constraint: 只惩罚超出tolerance的部分

    Args:
        pred_value: 预测值
        target_value: 目标值
        tolerance: 容忍度（相对于target）

    Returns:
        loss
    """
    lower_bound = target_value * (1 - tolerance)
    upper_bound = target_value * (1 + tolerance)

    # 只惩罚超出范围的部分
    violation = torch.relu(pred_value - upper_bound) + \
                torch.relu(lower_bound - pred_value)

    return violation.mean()
```

### 使用示例

```python
# CA-CB键长：1.54Å ± 10%
loss_bond = soft_constraint_loss(
    CA_CB_distance,
    target=1.54,
    tolerance=0.10  # 允许1.386-1.694Å
)

# 只有非常离谱的才会被惩罚（如5.3Å的异常值）
```

### 优点
✅ **避免over-constraint**：允许合理的化学变异
✅ **梯度稳定**：在容忍区域内梯度为0

### 缺点
⚠️ 需要调节tolerance超参数

---

## 方案E: 分层约束（推荐实现）

### 核心思想
**不同原子使用不同强度的约束**

```python
class GeometricConstraintLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义分层约束
        self.constraints = {
            # Tier 1: 强约束（必须满足）
            'critical': {
                'CA-CB': {'dist': 1.54, 'weight': 10.0},
            },
            # Tier 2: 中等约束（重要但允许偏差）
            'important': {
                'CB-CG': {'dist': 1.52, 'weight': 5.0},
                'CA-CB-CG_angle': {'angle': 109.5, 'weight': 3.0},
            },
            # Tier 3: 弱约束（只避免极端情况）
            'soft': {
                'long_range': {'min_dist': 2.0, 'weight': 1.0},  # clash
            }
        }

    def forward(self, coords, aatype, atom_exists):
        """
        计算分层几何约束loss

        Args:
            coords: [B, N, 14, 3]
            aatype: [B, N]
            atom_exists: [B, N, 14] bool
        """
        losses = {}

        # Tier 1: 强约束 - CA-CB键长
        CA = coords[..., 1, :]  # [B, N, 3]
        CB = coords[..., 4, :]
        CA_CB_dist = (CA - CB).norm(dim=-1)
        mask_CB = atom_exists[..., 4]
        losses['CA_CB_bond'] = F.mse_loss(
            CA_CB_dist[mask_CB],
            torch.tensor(1.54, device=coords.device).expand_as(CA_CB_dist[mask_CB])
        ) * 10.0

        # Tier 2: 中等约束 - 主要键角（用cosine loss）
        CG = coords[..., 5, :]
        mask_CG = atom_exists[..., 5]
        if mask_CG.any():
            losses['CB_CG_angle'] = stable_angle_loss(
                CA, CB, CG,
                target_angle_rad=torch.tensor(109.5 * math.pi / 180),
                mask=mask_CG
            ) * 3.0

        # Tier 3: 弱约束 - clash避免
        losses['clash'] = self.clash_penalty(coords, atom_exists) * 1.0

        return sum(losses.values()), losses

    def clash_penalty(self, coords, atom_exists):
        """避免原子clash（<2.0Å）"""
        # 只检查侧链内部
        sidechain = coords[..., 4:, :]  # [B, N, 11, 3]
        mask = atom_exists[..., 4:]      # [B, N, 11]

        # Pairwise distances
        B, N, K, _ = sidechain.shape
        dists = torch.cdist(
            sidechain.reshape(B * N, K, 3),
            sidechain.reshape(B * N, K, 3)
        )  # [B*N, K, K]

        # 只惩罚<2.0Å的（除了相邻原子）
        clash_threshold = 2.0
        violations = torch.relu(clash_threshold - dists)

        # 去掉对角线和相邻原子
        mask_ij = torch.ones_like(dists, dtype=torch.bool)
        mask_ij = mask_ij & ~torch.eye(K, device=dists.device, dtype=torch.bool).unsqueeze(0)
        # TODO: 也去掉1-2, 1-3相邻的

        return (violations * mask_ij.float()).mean()
```

### 使用

```python
# 在训练loss中添加
geo_loss_module = GeometricConstraintLoss()

# Training step
total_loss, detail_losses = geo_loss_module(
    atoms14_pred_local,
    batch['aatype'],
    batch['atom14_gt_exists']
)

# 加入总loss
loss = velocity_loss + coord_loss + 0.1 * total_loss
#                                    ↑
#                               可调节权重
```

---

## 推荐的实施步骤

### 阶段1: 快速验证（1-2天）

**目标**：验证几何约束是否有帮助

**实现**：
1. 在现有代码基础上添加**Cosine Loss**（方案A）
2. 只约束最critical的CA-CB键角
3. 权重设置为0.1-1.0

**修改文件**：
- `/home/junyu/project/pu/models/flow_module.py` - 在`model_step_fbb`中添加
- 添加新文件：`/home/junyu/project/pu/models/geometric_loss.py`

**验证实验**：
- 训练10个epoch
- 测试10步推理
- 检查CA-CB-CG键角误差是否降低

### 阶段2: 完整实现（1周）

**如果阶段1有效**：

**实现**：
1. 添加**分层约束**（方案E）
2. 实现**1-3 distance约束**（方案B）
3. 调节各层权重

**需要的target values**：
```python
# 从rotamer library或化学数据库获取
STANDARD_BOND_LENGTHS = {
    'CA-CB': 1.54,
    'CB-CG': 1.52,  # sp3
    'CB-CG_aromatic': 1.51,  # sp2
    'CG-CD': 1.52,
    'C-N': 1.47,
    'C-O': 1.43,
    'C-S': 1.81,
}

STANDARD_BOND_ANGLES = {
    'CA-CB-CG': 109.5,  # 四面体
    'CB-CG-CD': 109.5,
    'aromatic': 120.0,  # 苯环
}
```

### 阶段3: 高级优化（1-2周）

**如果效果显著**：

**考虑**：
1. 根据氨基酸类型使用不同的target（PHE vs ALA）
2. 动态调整权重（训练早期强约束，后期弱约束）
3. 添加二面角的先验（从rotamer library）

---

## 代码实现示例

### 完整的geometric_loss.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GeometricConstraintLoss(nn.Module):
    """
    分层几何约束Loss

    Tier 1 (Critical): CA-CB键长 - 必须接近1.54Å
    Tier 2 (Important): 主要键角 - 使用cosine loss
    Tier 3 (Soft): Clash避免 - 只避免严重碰撞
    """

    def __init__(self,
                 bond_weight=10.0,
                 angle_weight=3.0,
                 clash_weight=1.0):
        super().__init__()
        self.bond_weight = bond_weight
        self.angle_weight = angle_weight
        self.clash_weight = clash_weight

    def forward(self, coords, aatype, atom_exists, res_mask):
        """
        Args:
            coords: [B, N, 14, 3] atom14 local coordinates
            aatype: [B, N] amino acid types
            atom_exists: [B, N, 14] bool
            res_mask: [B, N] residue mask

        Returns:
            total_loss: scalar
            loss_dict: detailed losses for logging
        """
        loss_dict = {}

        # Tier 1: CA-CB键长
        loss_dict['CA_CB_bond'] = self.ca_cb_bond_loss(
            coords, atom_exists, res_mask
        )

        # Tier 2: 键角
        loss_dict['bond_angles'] = self.bond_angle_loss(
            coords, aatype, atom_exists, res_mask
        )

        # Tier 3: Clash
        loss_dict['clash'] = self.clash_loss(
            coords, atom_exists, res_mask
        )

        # 加权求和
        total_loss = (
            self.bond_weight * loss_dict['CA_CB_bond'] +
            self.angle_weight * loss_dict['bond_angles'] +
            self.clash_weight * loss_dict['clash']
        )

        return total_loss, loss_dict

    def ca_cb_bond_loss(self, coords, atom_exists, res_mask):
        """CA-CB键长约束"""
        CA = coords[..., 1, :]  # [B, N, 3]
        CB = coords[..., 4, :]

        # 计算距离
        dist = (CA - CB).norm(dim=-1)  # [B, N]

        # Mask: 必须有CB（GLY除外）
        mask = atom_exists[..., 4] & res_mask  # [B, N]

        if not mask.any():
            return torch.tensor(0.0, device=coords.device)

        # MSE loss
        target = torch.tensor(1.54, device=coords.device)
        loss = F.mse_loss(dist[mask], target.expand_as(dist[mask]))

        return loss

    def bond_angle_loss(self, coords, aatype, atom_exists, res_mask):
        """主要键角约束（CA-CB-CG）"""
        CA = coords[..., 1, :]
        CB = coords[..., 4, :]
        CG = coords[..., 5, :]

        # Mask: 必须有CG
        mask = atom_exists[..., 5] & res_mask

        if not mask.any():
            return torch.tensor(0.0, device=coords.device)

        # 使用stable cosine loss
        target_angle = torch.tensor(109.5 * math.pi / 180, device=coords.device)
        loss = stable_angle_loss(CA, CB, CG, target_angle, mask)

        return loss

    def clash_loss(self, coords, atom_exists, res_mask):
        """避免侧链内部原子clash"""
        # 只看侧链
        sidechain = coords[..., 4:, :]  # [B, N, 11, 3]
        sc_mask = atom_exists[..., 4:]   # [B, N, 11]

        B, N, K, _ = sidechain.shape

        # 计算pairwise distances（逐residue）
        total_loss = 0
        count = 0

        for b in range(B):
            for n in range(N):
                if not res_mask[b, n]:
                    continue

                atoms = sidechain[b, n]  # [11, 3]
                mask_n = sc_mask[b, n]    # [11]

                if mask_n.sum() < 2:
                    continue

                # Pairwise distances
                dists = torch.cdist(atoms.unsqueeze(0), atoms.unsqueeze(0)).squeeze(0)  # [11, 11]

                # Mask for valid pairs
                pair_mask = mask_n[:, None] & mask_n[None, :]  # [11, 11]

                # 去掉对角线和1-2邻居
                pair_mask = pair_mask & ~torch.eye(11, device=dists.device, dtype=torch.bool)
                # 简化：去掉i和i+1（1-2键）
                for i in range(10):
                    pair_mask[i, i+1] = False
                    pair_mask[i+1, i] = False

                # Clash: distance < 2.0Å
                clash_threshold = 2.0
                violations = torch.relu(clash_threshold - dists)  # [11, 11]

                loss_n = (violations * pair_mask.float()).sum()
                total_loss += loss_n
                count += pair_mask.sum()

        if count == 0:
            return torch.tensor(0.0, device=coords.device)

        return total_loss / count.clamp(min=1)


def stable_angle_loss(p1, p2, p3, target_angle_rad, mask):
    """
    稳定的键角loss（cosine-based）

    Args:
        p1, p2, p3: [B, N, 3] 三个原子坐标
        target_angle_rad: scalar 或 [B, N] 目标角度（弧度）
        mask: [B, N] bool

    Returns:
        loss: scalar
    """
    # 向量
    v1 = p1 - p2
    v2 = p3 - p2

    # 归一化（带epsilon）
    eps = 1e-7
    v1_norm = v1 / (v1.norm(dim=-1, keepdim=True).clamp(min=eps))
    v2_norm = v2 / (v2.norm(dim=-1, keepdim=True).clamp(min=eps))

    # Cosine（带clamp）
    cos_pred = (v1_norm * v2_norm).sum(dim=-1)  # [B, N]
    cos_pred = torch.clamp(cos_pred, -1.0 + 1e-6, 1.0 - 1e-6)

    # Target cosine
    if isinstance(target_angle_rad, torch.Tensor) and target_angle_rad.dim() > 0:
        cos_target = torch.cos(target_angle_rad)
    else:
        cos_target = torch.cos(torch.tensor(target_angle_rad, device=p1.device))

    # MSE
    loss_per_angle = (cos_pred - cos_target) ** 2

    # Apply mask
    if not mask.any():
        return torch.tensor(0.0, device=p1.device)

    loss = (loss_per_angle * mask).sum() / mask.sum()

    return loss
```

### 集成到训练

修改`/home/junyu/project/pu/models/flow_module.py`:

```python
# 在__init__中添加
self.geometric_loss = GeometricConstraintLoss(
    bond_weight=getattr(training_cfg, 'geo_bond_weight', 1.0),
    angle_weight=getattr(training_cfg, 'geo_angle_weight', 0.5),
    clash_weight=getattr(training_cfg, 'geo_clash_weight', 0.1),
)

# 在model_step_fbb中添加
def model_step_fbb(self, batch: Any, prob=None):
    # ... 现有代码 ...

    # 添加几何约束
    geo_loss, geo_details = self.geometric_loss(
        atoms14_pred_local_no_snr,  # 使用no-snr版本
        batch['aatype'],
        batch['atom14_gt_exists'],
        batch['res_mask']
    )

    # 加入总loss
    geo_weight = getattr(self._exp_cfg.training, 'geo_loss_weight', 0.1)
    total_loss = (
        speed_weight * speed_loss
        + vector_weight * vector_loss
        + coord_weight * coord_loss
        + geo_weight * geo_loss  # 新增
    )

    # 记录详细losses
    metrics.update({
        'geo_loss': geo_loss.detach(),
        'geo_CA_CB': geo_details['CA_CB_bond'].detach(),
        'geo_angles': geo_details['bond_angles'].detach(),
        'geo_clash': geo_details['clash'].detach(),
    })

    # ...
```

---

## 配置文件修改

在config YAML中添加：

```yaml
experiment:
  training:
    # 现有配置...

    # 几何约束权重
    geo_loss_weight: 0.1  # 整体权重
    geo_bond_weight: 10.0  # CA-CB键长
    geo_angle_weight: 3.0   # 键角
    geo_clash_weight: 1.0   # Clash

    # 可选：动态调整
    geo_schedule: 'constant'  # 'constant', 'decay', 'warmup'
```

---

## 预期效果

### 短期（阶段1完成后）

**指标改善**：
- CA-CB键长误差: 从0.054Å → <0.01Å
- CA-CB-CG键角误差: 从6° → <3°
- 极端outlier（5.3Å的键）: 显著减少

**副作用**：
- 训练可能稍慢（+5-10%时间）
- RMSD可能略微上升（因为增加了约束）

### 长期（阶段2-3完成后）

**R3性能可能改善**：
- 步数增加不再导致退化
- 100步可能达到当前10步的质量
- 可能出现弱的scaling law（虽然不如SH那么强）

**但根本问题仍存在**：
- Velocity方向误差仍然1-2Å（几何约束不能完全修复）
- 最多延缓退化，很难完全消除

---

## 结论

### 您的直觉完全正确

✅ **需要几何约束**：当前R3缺乏约束，导致离谱的几何
✅ **键角容易NaN**：直接计算angle非常危险

### 推荐方案

**立即实施**：
1. **Cosine Loss（方案A）**：稳定且高效
2. **1-3 Distance（方案B）**：作为补充
3. **分层约束（方案E）**：不同原子不同强度

**不推荐**：
- ❌ Internal Coordinates（方案C）：太复杂，不适合现有架构

### 现实预期

几何约束**能改善但不能完全解决R3的问题**：
- ✅ 减少极端outlier（5.3Å → <2Å）
- ✅ 提升局部几何质量
- ⚠️ 但velocity方向误差的根本问题仍存在
- ⚠️ 步数增加可能仍会退化（只是减缓）

**真正的解决方案可能需要**：
1. 更强的模型（更大capacity）
2. 更好的训练策略（curriculum learning on t）
3. 或者接受SH的优势（修复泄漏后重新评估）

想先实现快速验证（阶段1）吗？我可以帮您写完整的代码！
