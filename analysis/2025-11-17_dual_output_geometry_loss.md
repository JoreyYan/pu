# 2025-11-17: 双输出几何约束方案对比

## 用户提出的方案

### 架构设计

```
模型输出：
1. 原子坐标 (coords_pred): [B, N, 11, 3]
2. 键角/键长 (angles_pred, bonds_pred): [B, N, 11], [B, N, 11]

处理流程：
1. 用 angles_pred + 默认键长 → 组装标准侧链 (coords_assembled)
2. 计算三个loss：
   Loss1: coords_assembled vs GT原子
   Loss2: coords_pred vs GT原子 (现有)
   Loss3: coords_assembled vs coords_pred (consistency)
```

### 核心问题

**这三个loss能起到同样的约束作用吗？**

答案：**不完全相同，但可能更好！**

---

## 详细对比分析

### 方案A: 直接约束坐标（我之前的建议）

```python
# 模型输出
coords_pred = model(input)  # [B, N, 11, 3]

# Loss
loss_coord = MSE(coords_pred, coords_gt)
loss_angle = cosine_loss(
    compute_angle_from_coords(coords_pred),  # ⚠️ 从坐标反推
    target_angle
)
loss_bond = MSE(
    compute_bond_from_coords(coords_pred),   # ⚠️ 从坐标反推
    target_bond
)

total_loss = loss_coord + λ1*loss_angle + λ2*loss_bond
```

**问题**：
- ❌ 从坐标反推角度：间接约束，梯度路径复杂
- ❌ 可能冲突：coord loss要求匹配GT，angle loss要求标准角度，但GT本身可能有偏差
- ❌ 过约束：如果GT的键角是108°（合理偏差），但我们约束到109.5°，模型困惑

**优点**：
- ✅ 实现简单，只需一个输出
- ✅ 推理时直接用coords_pred

### 方案B: 双输出+组装（用户的方案）

```python
# 模型输出
coords_pred = model.coord_head(features)   # [B, N, 11, 3]
angles_pred = model.angle_head(features)   # [B, N, 11]
bonds_pred = model.bond_head(features)     # [B, N, 11] (可选)

# 组装标准侧链
coords_assembled = assemble_from_internal(
    angles_pred,
    bonds_standard,  # 固定值：1.54, 1.52, ...
    backbone_coords  # 固定参考
)  # [B, N, 11, 3]

# 三个loss
loss1_assembled_vs_gt = MSE(coords_assembled, coords_gt)
loss2_direct_vs_gt = MSE(coords_pred, coords_gt)
loss3_consistency = MSE(coords_assembled, coords_pred)

total_loss = loss1 + loss2 + loss3
```

**关键差异**：
- ✅ 显式学习键角：模型直接输出angle，不是反推
- ✅ 解耦自由度：键长固定（标准值），只学角度
- ✅ 双路径监督：两种方式都要接近GT
- ⚠️ 但有新问题（见下文）

---

## 深入分析：三个Loss的作用

### Loss 1: coords_assembled vs coords_gt

```python
loss1 = MSE(coords_assembled, coords_gt)
```

**物理意义**：
- "用标准键长+学到的键角，能否重建GT结构？"

**作用**：
- 迫使`angles_pred`学习正确的角度
- 如果角度错误，组装出的结构会偏离GT

**问题**：
- ⚠️ GT本身可能不是"标准"键长（1.54Å可能是1.52或1.56）
- ⚠️ 如果GT键长=1.50，但我们用1.54组装，即使角度完美，loss1也不会是0
- ⚠️ 这会给`angles_pred`错误的梯度信号！

**示例**：
```
GT: CA=(0,0,0), CB=(1.50, 0, 0), CG=(1.50+1.50*cos110°, 1.50*sin110°, 0)
     键长：CA-CB=1.50, CB-CG=1.50, 角度=110°

组装: 用键长=1.54（标准），角度=110°（从angles_pred）
     CA=(0,0,0), CB=(1.54, 0, 0), CG=(1.54+1.54*cos110°, 1.54*sin110°, 0)

Loss1 ≠ 0，即使角度是完美的110°！
模型会收到错误信号：降低角度来补偿键长差异
```

### Loss 2: coords_pred vs coords_gt

```python
loss2 = MSE(coords_pred, coords_gt)
```

**物理意义**：
- "直接输出的坐标能否匹配GT？"

**作用**：
- 这是主要的监督信号
- 允许模型学习GT的真实键长和键角（包括偏差）

**优点**：
- ✅ 最灵活：可以拟合任何GT几何
- ✅ 不假设标准值

### Loss 3: coords_assembled vs coords_pred (Consistency)

```python
loss3 = MSE(coords_assembled, coords_pred)
```

**物理意义**：
- "两个输出要一致"

**作用**：
- 迫使`coords_pred`接近标准几何（通过`coords_assembled`）
- 迫使`angles_pred`解释`coords_pred`（反向）

**这是关键的约束！**

但有个问题...

---

## 问题：Loss 1 和 Loss 3 的冲突

### 场景1：GT是标准几何

```
GT键长=1.54, 角度=109.5°（完美）

理想状态：
- angles_pred = 109.5°
- coords_assembled ≈ coords_gt (loss1 ≈ 0)
- coords_pred ≈ coords_gt (loss2 ≈ 0)
- coords_assembled ≈ coords_pred (loss3 ≈ 0)

✅ 三个loss协同工作，没有冲突
```

### 场景2：GT偏离标准（常见！）

```
GT键长=1.50, 角度=108°（合理的化学偏差）

矛盾：
- Loss1说：coords_assembled要接近GT
  → 但组装用的是键长1.54！
  → 即使angles_pred=108°完美，coords_assembled也离GT有0.04Å误差
  → Loss1给出错误梯度：让angles_pred调整来补偿键长差异

- Loss2说：coords_pred要接近GT
  → coords_pred可以学到键长=1.50，角度=108°
  → Loss2 ≈ 0

- Loss3说：两者要一致
  → 但coords_assembled用1.54，coords_pred用1.50
  → 不可能一致！
  → Loss3给出错误信号

结果：三个loss互相拉扯！
```

### 数值示例

```python
# GT侧链（真实蛋白质中的合理偏差）
GT_CA = [0, 0, 0]
GT_CB = [1.50, 0, 0]        # 键长1.50（标准是1.54）
GT_CG = [1.95, 1.42, 0]     # 角度108°（标准是109.5°）

# 模型学习过程
# Iteration 1:
angles_pred = 109.5  # 初始猜测
coords_assembled = assemble([0,0,0], angles=109.5, bonds=1.54)
             = [[1.54, 0, 0], [2.06, 1.46, 0]]  # 用标准键长

loss1 = MSE([2.06, 1.46, 0], [1.95, 1.42, 0]) = 0.014  # ❌ 不为0
     → 梯度：降低angle到108°来匹配

# Iteration 2:
angles_pred = 108.0  # 根据loss1调整
coords_assembled = assemble([0,0,0], angles=108, bonds=1.54)
             = [[1.54, 0, 0], [2.01, 1.40, 0]]

loss1 = MSE([2.01, 1.40, 0], [1.95, 1.42, 0]) = 0.006  # 更好但仍不为0
     → 梯度：继续调整...

# 但同时：
coords_pred = [1.50, 0, 0], [1.95, 1.42, 0]  # 从loss2学到真实值
loss3 = MSE(coords_assembled, coords_pred) = 很大！

→ 模型困惑：angles_pred应该是108°（匹配GT）还是调整到106°（补偿键长差异）？
```

---

## 改进方案：同时预测键长和键角

### 修正的用户方案

```python
# 模型输出（添加键长预测）
coords_pred = model.coord_head(features)   # [B, N, 11, 3]
angles_pred = model.angle_head(features)   # [B, N, 11]
bonds_pred = model.bond_head(features)     # [B, N, 11] ← 新增！

# 组装：用预测的键长和键角
coords_assembled = assemble_from_internal(
    angles_pred,
    bonds_pred,      # ← 不是固定值！
    backbone_coords
)

# 三个loss
loss1_assembled_vs_gt = MSE(coords_assembled, coords_gt)
loss2_direct_vs_gt = MSE(coords_pred, coords_gt)
loss3_consistency = MSE(coords_assembled, coords_pred)

# 额外：正则化键长接近标准值
loss4_bond_regularization = MSE(bonds_pred, bonds_standard)  # soft constraint

total_loss = loss1 + loss2 + loss3 + 0.1*loss4
```

**现在三个loss协同了**：

```
场景：GT键长=1.50, 角度=108°

模型学习：
- angles_pred → 108°  (从loss1, loss2学到)
- bonds_pred → 1.50   (从loss1, loss2学到)
- coords_pred → 匹配GT (从loss2学到)
- coords_assembled = assemble(108°, 1.50) ≈ GT  (loss1 ≈ 0)
- coords_assembled ≈ coords_pred  (loss3 ≈ 0)

✅ 三个loss都接近0，没有冲突！

Loss4 (正则化)：
- bonds_pred=1.50 vs bonds_standard=1.54
- Loss4 = 0.04² = 0.0016
- 权重0.1 → 贡献很小
- 作用：鼓励接近标准值，但允许偏差
```

---

## 方案对比总结

| 方案 | 实现复杂度 | 训练稳定性 | 约束强度 | 推理时用 | 推荐 |
|------|-----------|-----------|---------|---------|------|
| **A. 直接约束坐标** | ⭐⭐ 简单 | ⭐⭐⭐ 较稳定 | ⭐⭐⭐ 中等 | coords_pred | ✅ 快速验证 |
| **B1. 双输出+固定键长** | ⭐⭐⭐ 中等 | ⭐⭐ 易冲突 | ⭐⭐⭐⭐ 强 | coords_pred? | ⚠️ 有冲突 |
| **B2. 双输出+预测键长** | ⭐⭐⭐⭐ 复杂 | ⭐⭐⭐⭐ 稳定 | ⭐⭐⭐⭐⭐ 很强 | coords_pred | ✅✅ 最佳 |
| **C. 纯内坐标** | ⭐⭐⭐⭐⭐ 很复杂 | ⭐⭐⭐⭐⭐ 很稳定 | ⭐⭐⭐⭐⭐ 完美 | coords_assembled | 🔄 重构 |

---

## 推荐实施路径

### 阶段1: 验证方案A（1-2天）

先用简单的直接约束验证**是否有帮助**：

```python
loss_coord = MSE(coords_pred, coords_gt)
loss_angle = cosine_loss(coords_pred)  # 从坐标计算
loss_bond = MSE(bond_from_coords(coords_pred), target_bonds)

total = loss_coord + 0.1*loss_angle + 0.1*loss_bond
```

**如果有改善** → 继续阶段2

### 阶段2: 实施方案B2（1周）

**添加双输出**：

```python
class SideAtomsDualOutputHead(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        # 现有的坐标head
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 11 * 3)  # 11个原子 × 3D
        )

        # 新增：键角head
        self.angle_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 11)  # 11个键角
        )

        # 新增：键长head
        self.bond_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 11)  # 11个键长
        )

    def forward(self, features):
        coords = self.coord_head(features).reshape(-1, 11, 3)
        angles = self.angle_head(features)  # [B*N, 11]
        bonds = self.bond_head(features)    # [B*N, 11]

        # 归一化到合理范围
        angles = torch.sigmoid(angles) * math.pi  # [0, π]
        bonds = torch.sigmoid(bonds) * 2.0 + 1.0  # [1.0, 3.0]Å

        return coords, angles, bonds
```

**组装函数**：

```python
def assemble_sidechain(angles, bonds, backbone_coords, aatype):
    """
    从键角和键长组装侧链

    Args:
        angles: [B, N, 11] 键角（弧度）
        bonds: [B, N, 11] 键长（Å）
        backbone_coords: [B, N, 3, 3] (N, CA, C)
        aatype: [B, N] 氨基酸类型

    Returns:
        sidechain: [B, N, 11, 3] 组装的侧链坐标
    """
    B, N, K = angles.shape
    device = angles.device

    # 提取backbone
    N_coord = backbone_coords[..., 0, :]   # [B, N, 3]
    CA_coord = backbone_coords[..., 1, :]
    C_coord = backbone_coords[..., 2, :]

    coords = []

    # CB: 从CA出发
    # 方向：沿CA-N和CA-C的叉积（指向beta碳方向）
    v_CA_N = N_coord - CA_coord
    v_CA_C = C_coord - CA_coord
    normal = torch.cross(v_CA_N, v_CA_C, dim=-1)  # [B, N, 3]
    normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-7)

    # CB在CA-N-C平面的投影方向
    v_bisect = F.normalize(v_CA_N + v_CA_C, dim=-1)

    # CB位置（简化：用固定角度）
    # 实际应该用angles[..., 0]和bonds[..., 0]
    CB_direction = F.normalize(
        v_bisect * 0.5 - normal * 0.866,  # 大致四面体角
        dim=-1
    )
    CB = CA_coord + CB_direction * bonds[..., 0:1]  # [B, N, 3]
    coords.append(CB)

    # CG: 从CB出发（需要angles[..., 1]和bonds[..., 1]）
    # 这里需要更复杂的几何计算...
    # 参考：NeRF (Natural Extension Reference Frame)

    # ... (完整实现需要递归构建)

    # 简化版本：只返回CB
    # 完整版需要实现完整的Z-matrix或NeRF算法

    return torch.stack(coords + [torch.zeros_like(CB)] * 10, dim=-2)
```

**注意**：完整的组装函数实现较复杂，需要参考：
- AlphaFold的structure module
- RoseTTAFold的coordinate building
- 或使用sidechainnet等库

**三Loss实现**：

```python
def dual_output_loss(coords_pred, angles_pred, bonds_pred,
                     coords_gt, aatype, backbone_coords, mask):
    """
    计算三个consistency losses
    """
    # 组装
    coords_assembled = assemble_sidechain(
        angles_pred, bonds_pred, backbone_coords, aatype
    )

    # Loss 1: 组装 vs GT
    loss1 = F.mse_loss(
        coords_assembled[mask],
        coords_gt[mask]
    )

    # Loss 2: 直接 vs GT (原有)
    loss2 = F.mse_loss(
        coords_pred[mask],
        coords_gt[mask]
    )

    # Loss 3: Consistency
    loss3 = F.mse_loss(
        coords_assembled[mask],
        coords_pred[mask]
    )

    # Loss 4: 键长正则化（鼓励接近标准值）
    bonds_standard = get_standard_bonds(aatype)  # [B, N, 11]
    loss4 = F.mse_loss(bonds_pred, bonds_standard)

    return {
        'loss_assembled_vs_gt': loss1,
        'loss_direct_vs_gt': loss2,
        'loss_consistency': loss3,
        'loss_bond_regularization': loss4,
    }
```

### 阶段3: 推理时的选择

**推理时用哪个输出？**

**选项1：用coords_pred**
```python
# 推理
coords, angles, bonds = model(input)
final_output = coords  # 直接用坐标输出
```

优点：
- ✅ 最灵活，可以捕捉细微偏差
- ✅ Loss2直接优化这个输出

**选项2：用coords_assembled**
```python
final_output = assemble_sidechain(angles, bonds, backbone)
```

优点：
- ✅ 保证完美几何（标准键长键角）
- ✅ 可以用高精度的组装算法（比训练时更精确）

**选项3：混合（推荐）**
```python
# 用coords_pred作为主要输出
# 但用angles/bonds来修正异常值
final_output = coords_pred.clone()

# 检测异常（如键长>2.0Å）
abnormal_mask = detect_abnormal_geometry(coords_pred)

# 对异常位置用组装版本替换
final_output[abnormal_mask] = coords_assembled[abnormal_mask]
```

---

## 实验验证计划

### 实验1: 方案A vs 方案B2

| 设置 | RMSD | CA-CB误差 | 键角误差 | 训练时间 |
|------|------|----------|---------|---------|
| Baseline (无约束) | 1.059 | 0.054Å | 6° | 1x |
| 方案A (直接约束) | ? | ? | ? | 1.1x |
| 方案B2 (双输出) | ? | ? | ? | 1.3x |

### 实验2: 消融研究

测试三个loss的必要性：

| Loss组合 | RMSD | 几何质量 |
|---------|------|---------|
| 只有Loss2 | baseline | baseline |
| Loss2 + Loss1 | ? | ? |
| Loss2 + Loss3 | ? | ? |
| Loss2 + Loss1 + Loss3 | ? | ? |
| All (包括Loss4) | ? | ? |

---

## 回答您的原始问题

> "这样三个loss能起到同样的约束作用吗？"

### 短回答

**如果用固定键长（方案B1）**：❌ 不能，会有冲突
**如果预测键长（方案B2）**：✅ 可以，甚至更好！

### 长回答

**方案B1（固定键长）的问题**：
- Loss1和Loss2会冲突（GT键长≠标准值时）
- Loss3会给出错误信号
- 不如直接约束（方案A）

**方案B2（预测键长）的优势**：
1. **显式学习几何**：模型直接学angles/bonds，不是反推
2. **解耦自由度**：可以分别监督几何参数和坐标
3. **双重保险**：两条路径都要对才能loss低
4. **推理灵活**：可以选择用坐标或组装

**但代价是**：
- 实现复杂（需要组装函数）
- 训练slower（三个loss要平衡）
- 需要调节权重

**我的建议**：
1. 先试方案A（简单快速验证）
2. 如果有效且想要更强约束→方案B2
3. 如果从头重构→考虑纯内坐标（方案C）

想先实现哪个？我可以写完整代码！
