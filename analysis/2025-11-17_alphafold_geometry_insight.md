# 2025-11-17: AlphaFold的几何约束启示

## 用户的关键反驳

> "AlphaFold也用的模板键长，也很好啊"

这个观察**完全正确**，让我重新评估之前的分析。

---

## AlphaFold的做法

### Structure Module的设计

AlphaFold2在Structure Module中：

```python
# AlphaFold的侧链构建（伪代码）
class StructureModule:
    def build_sidechain(self, backbone_frames, torsion_angles, aatype):
        """
        Args:
            backbone_frames: rigid frames (N, CA, C)
            torsion_angles: χ1, χ2, χ3, χ4 (预测的二面角)
            aatype: 氨基酸类型
        """
        # 使用固定的理想键长和键角
        ideal_bond_lengths = {
            'CA-CB': 1.54,
            'CB-CG': 1.52,
            # ... 从标准几何库获取
        }

        ideal_bond_angles = {
            'N-CA-CB': 110.5,
            'CA-CB-CG': 113.4,
            # ... 从标准几何库获取
        }

        # 只有二面角是可变的（从模型预测）
        torsion_angles_pred = self.torsion_predictor(features)

        # 用NeRF或Z-matrix组装侧链
        sidechain = build_from_internal_coords(
            backbone_frames,
            ideal_bond_lengths,  # ← 固定！
            ideal_bond_angles,   # ← 固定！
            torsion_angles_pred  # ← 可变！
        )

        return sidechain
```

### 关键特点

1. **固定键长和键角**：使用标准值（从晶体结构统计）
2. **预测二面角**：只预测χ角（chi angles），通常1-4个
3. **从backbone frame构建**：以刚性frame为参考
4. **FAPE Loss**：Frame Aligned Point Error，对齐后计算误差

---

## 为什么AlphaFold的固定键长/键角有效？

### 原因1: 键长/键角变化很小

```
键长的化学变异性：
- CA-CB: 1.54 ± 0.02Å (变化范围 ~1%)
- CB-CG: 1.52 ± 0.03Å (变化范围 ~2%)
- C-N: 1.47 ± 0.02Å

键角的化学变异性：
- N-CA-CB: 110.5 ± 2°   (变化范围 ~2%)
- CA-CB-CG: 113.4 ± 3°  (变化范围 ~3%)

二面角的变异性：
- χ1: 0-360°  (完全可变！)
- χ2: 0-360°  (完全可变！)
```

**关键洞察**：
- 键长/键角：变化<5%，接近刚性
- 二面角：变化100%，高度柔性
- **大部分构象变化来自二面角，不是键长键角！**

### 原因2: GT偏差的影响很小

我之前过分强调了GT键长可能是1.50而不是1.54的问题。

**实际计算**：
```python
# 假设侧链有5个原子
# 每个键长误差0.04Å

# 累积效应（最坏情况：误差同向）
max_error = 5 * 0.04 = 0.20Å

# 实际效应（误差随机）
typical_error = sqrt(5) * 0.04 = 0.09Å

# 对比：
# - 键长误差贡献：~0.1Å
# - R3当前RMSD：1.06Å
# - 二面角错误贡献：~1.0Å（主要来源！）
```

**结论**：键长偏差的影响被二面角误差淹没了！

### 原因3: 提供强大的归纳偏置

**固定键长/键角的好处**：

1. **减少自由度**
   - 不固定：11个原子 × 3D = 33个自由度
   - 固定后：最多4个二面角 = 4个自由度
   - 学习难度降低8倍！

2. **保证物理合理性**
   - 不会出现5.3Å的异常键长
   - 不会出现60°的四面体角（应该109°）
   - 自动满足化学约束

3. **提升泛化能力**
   - 训练数据可能有噪声（实验误差、精修误差）
   - 固定标准值避免学到噪声
   - 泛化到新结构更robust

---

## 重新评估用户的方案

### 用户的原始方案（重新理解）

```python
# 模型输出
angles_pred = model.angle_head(features)    # 预测键角
coords_pred = model.coord_head(features)    # 预测坐标

# 组装（用固定键长 + 预测键角）
coords_assembled = assemble(
    angles_pred,
    bonds=STANDARD_BONDS  # 固定标准值
)

# 三个loss
Loss1: coords_assembled vs GT
Loss2: coords_pred vs GT
Loss3: coords_assembled vs coords_pred
```

### 重新分析Loss冲突

**之前我说的冲突问题**：
```
GT键长=1.50, 但组装用1.54
→ Loss1会有0.09Å的系统误差
→ 会给angles_pred错误梯度
```

**重新评估**：
- 0.09Å的系统误差 vs 1.06Å的总RMSD
- 贡献只有8%！
- **可以接受！**

**更重要的是**：
- Loss1会迫使angles_pred学习正确的角度（主要贡献）
- Loss2会让coords_pred捕捉细节（包括键长偏差）
- Loss3会让两者保持一致性

**实际效果**：
- angles_pred学到~108°（略有调整来补偿键长）
- coords_pred学到精确的GT
- 0.09Å的不一致性可以被容忍

---

## 但有个关键问题：键角不够！

### 确定侧链需要的参数

要完全确定侧链构象，需要：

```
以PHE（苯丙氨酸）为例：

固定参数（标准值）：
- 键长: CA-CB=1.54, CB-CG=1.50, CG-CD=1.39, ...
- 键角: N-CA-CB=110°, CA-CB-CG=114°, ...

可变参数（需要预测）：
- χ1 (N-CA-CB-CG): 二面角，0-360°  ← 关键！
- χ2 (CA-CB-CG-CD1): 二面角，0-360° ← 关键！

仅靠键角无法确定：
- 知道CA-CB-CG=114°，但不知道CG在哪个方向旋转
- 需要χ1告诉你：围绕CA-CB轴旋转多少度
```

**图示**：
```
       CG
      /
CA--CB     ← 键角CA-CB-CG=114°确定了CG的"开口大小"
      \    ← 但χ1决定了CG绕CB旋转的位置（0-360°）
       ?

如果只知道键角：
CG可以在一个圆锥面上的任意位置！
```

### 用户方案的隐含假设

如果用户的方案有效，意味着：

**方案A：angles_pred实际上是二面角**
```python
angles_pred = model.angle_head(features)  # [B, N, 4]
# angles_pred实际上是[χ1, χ2, χ3, χ4]

coords_assembled = assemble_with_torsions(
    backbone_frames,
    torsions=angles_pred,
    bonds=STANDARD_BONDS,   # 固定
    angles=STANDARD_ANGLES  # 固定
)

# 这就是AlphaFold的做法！
```

**方案B：coords_pred提供额外信息**
```python
# angles_pred只预测键角（不完整）
# coords_pred提供完整的3D信息（包括隐含的二面角）

# Loss1和Loss3迫使两者一致
# 通过Loss3，angles_pred间接学到"等效的"几何信息
```

---

## 推荐的实施方案（基于AlphaFold）

### 方案：预测二面角 + 固定键长键角

```python
class TorsionAngleHead(nn.Module):
    """预测侧链二面角（AlphaFold style）"""

    def __init__(self, hidden_dim=256):
        super().__init__()

        # 预测χ角（每个氨基酸最多4个）
        self.chi_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # 4个角 × 2 (sin, cos)
        )

    def forward(self, features, aatype):
        # 预测sin和cos（避免周期性discontinuity）
        chi_sincos = self.chi_predictor(features)  # [B, N, 8]
        chi_sin = chi_sincos[..., 0::2]  # [B, N, 4]
        chi_cos = chi_sincos[..., 1::2]  # [B, N, 4]

        # 归一化
        norm = torch.sqrt(chi_sin**2 + chi_cos**2 + 1e-7)
        chi_sin = chi_sin / norm
        chi_cos = chi_cos / norm

        return chi_sin, chi_cos

def assemble_from_torsions(backbone_frames, chi_sin, chi_cos, aatype):
    """
    从二面角组装侧链（类似AlphaFold）

    Args:
        backbone_frames: [B, N, 4, 4] rigid transformation matrices
        chi_sin, chi_cos: [B, N, 4] 二面角的sin和cos
        aatype: [B, N] 氨基酸类型

    Returns:
        sidechain_coords: [B, N, 14, 3]
    """
    B, N = aatype.shape
    device = aatype.device

    # 标准几何参数（从化学数据库）
    from openfold.np import residue_constants

    coords_list = []

    for b in range(B):
        for n in range(N):
            aa = aatype[b, n].item()

            # 获取该氨基酸的标准键长/键角
            ideal_bonds = residue_constants.ideal_bond_lengths[aa]
            ideal_angles = residue_constants.ideal_bond_angles[aa]

            # 提取二面角
            chis = torch.atan2(chi_sin[b, n], chi_cos[b, n])  # [4]

            # 从backbone frame出发，逐原子构建
            frame = backbone_frames[b, n]  # [4, 4]

            # CB
            CB_local = place_atom(
                frame,
                bond_length=ideal_bonds['CA-CB'],
                bond_angle=ideal_angles['N-CA-CB'],
                torsion=chis[0]  # χ1的参考部分
            )

            # CG
            CG_local = place_atom(
                CB_local,
                bond_length=ideal_bonds['CB-CG'],
                bond_angle=ideal_angles['CA-CB-CG'],
                torsion=chis[0]  # χ1
            )

            # ... 继续构建其他原子

            coords_list.append(sidechain)

    return torch.stack(coords_list, dim=0)

def place_atom(parent_frame, bond_length, bond_angle, torsion):
    """
    在局部坐标系中放置原子（NeRF算法）

    这是标准的蛋白质几何算法
    """
    # ... (复杂的旋转矩阵计算)
    pass
```

### 三Loss实现

```python
def torsion_consistency_loss(coords_pred, chi_sin, chi_cos,
                              backbone_frames, coords_gt, aatype, mask):
    """
    三个loss（AlphaFold style）
    """
    # 组装（固定键长/键角 + 预测二面角）
    coords_assembled = assemble_from_torsions(
        backbone_frames, chi_sin, chi_cos, aatype
    )

    # Loss1: 组装 vs GT
    loss1 = fape_loss(coords_assembled, coords_gt, backbone_frames, mask)

    # Loss2: 直接预测 vs GT
    loss2 = fape_loss(coords_pred, coords_gt, backbone_frames, mask)

    # Loss3: Consistency
    loss3 = F.mse_loss(coords_assembled[mask], coords_pred[mask])

    return {
        'loss_torsion_vs_gt': loss1,
        'loss_direct_vs_gt': loss2,
        'loss_consistency': loss3,
    }

def fape_loss(pred, target, frames, mask):
    """
    Frame Aligned Point Error (AlphaFold的loss)

    先对齐到局部frame，再计算误差
    这样避免全局旋转/平移的影响
    """
    # 转换到局部frame
    pred_local = frames.invert().apply(pred)
    target_local = frames.invert().apply(target)

    # 计算误差
    error = torch.norm(pred_local - target_local, dim=-1)

    # 裁剪（类似Huber loss）
    error_clamped = torch.clamp(error, max=10.0)

    return (error_clamped * mask).sum() / mask.sum()
```

---

## 简化方案：如果只想快速验证

### 最小改动版本

**保持现有架构，只加约束**：

```python
# 现有：
coords_pred = model.coord_head(features)

# 新增：从coords_pred提取二面角
chi_extracted = extract_torsions_from_coords(
    coords_pred, backbone_frames
)

# 用提取的二面角 + 固定键长键角重新组装
coords_assembled = assemble_from_torsions(
    backbone_frames,
    chi_extracted,
    bonds=STANDARD,
    angles=STANDARD
)

# Loss
loss_coord = MSE(coords_pred, coords_gt)
loss_geometry = MSE(coords_assembled, coords_gt)  # 鼓励标准几何
loss_consistency = MSE(coords_assembled, coords_pred)  # 两者一致

total = loss_coord + 0.1*loss_geometry + 0.1*loss_consistency
```

**这个方案**：
- ✅ 不需要新的预测头
- ✅ 利用AlphaFold的固定几何思想
- ✅ 通过consistency loss实现软约束
- ⚠️ 需要实现extract_torsions和assemble函数

---

## 与之前分析的对比

### 我之前错在哪？

1. **过分强调GT偏差**：0.04Å的键长偏差实际影响很小（<10%）
2. **忽略了自由度降维的好处**：固定键长/键角大幅简化学习
3. **没有考虑AlphaFold的成功先例**：证明了这个方向的可行性

### 用户正确在哪？

1. **AlphaFold的先例**：证明固定标准值是可行且有效的
2. **归纳偏置的重要性**：物理约束帮助学习和泛化
3. **简化问题的智慧**：不是所有参数都需要学习

---

## 最终建议

### 方案对比

| 方案 | 复杂度 | 效果 | 推荐度 |
|------|--------|------|--------|
| **A. 直接约束坐标** | ⭐⭐ | ⭐⭐⭐ | ✅ 快速验证 |
| **B. 预测二面角（AlphaFold）** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅✅ 正确方向 |
| **C. 简化consistency** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ 折中方案 |

### 实施路径

**阶段1（1-2天）：方案C**
- 提取二面角 + 重组装 + consistency loss
- 验证固定几何的效果
- 实现相对简单

**阶段2（1-2周）：方案B**
- 如果有效，实施完整的torsion prediction head
- 类似AlphaFold的architecture
- 这是最正确的长期方向

**AlphaFold已经证明这个方向是对的！**

---

## 结论

### 用户的直觉完全正确！

✅ **固定键长/键角是可行的**（AlphaFold proof）
✅ **微小偏差的影响可以忽略**（<10%贡献）
✅ **提供强大的归纳偏置**（简化学习）

### 关键要素

要成功实施用户的方案，需要：
1. **预测二面角**（不只是键角）
2. **固定键长和键角**（用标准值）
3. **Frame-aligned loss**（避免旋转影响）
4. **Consistency约束**（双输出保持一致）

### 下一步

想要我实现方案C（简化版）还是直接上方案B（AlphaFold style）？

方案C可以用现有的coords_pred，只需加组装和consistency loss
方案B需要新增torsion prediction head，但更符合正确方向
