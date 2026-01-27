# SH Density Decode精度分析

**日期**: 2025-10-18
**模型**: SH_to_atoms_decoder, Epoch 27
**任务**: GT SH密度 → atom14坐标 + 序列类型

---

## 实验配置

```yaml
Loss配置:
  atom_loss_weight: 0.2
  type_loss_weight: 1.0

Loss组成:
  atomsloss = MSE + Huber + Pairwise
  total_loss = atomsloss * 0.2 + typeloss * 1.0
```

---

## 关键结果

### 1. 整体精度

| 指标 | 数值 |
|------|------|
| **整体 atom14 RMSD** | 1.67Å |
| **侧链 RMSD (atom 4-13)** | 2.21Å |
| **序列准确率** | 98.7% |

### 2. Per-Atom RMSD分布

#### 主链原子 (0-3)
```
atom0 (N):   0.73Å
atom1 (CA):  0.54Å  ← 最准确
atom2 (C):   0.84Å
atom3 (O):   1.23Å
```

#### 侧链原子 (4-13) - 误差递增
```
atom4  (CB):  0.34Å  ← 侧链最准
atom5:        1.23Å
atom6:        1.93Å
atom7:        2.55Å
atom8:        3.62Å
atom9:        3.73Å
atom10:       4.27Å
atom11:       5.07Å  ← 最差
atom12:       3.96Å
atom13:       4.75Å
```

**趋势**：符合预期，远端原子误差更大（柔性增加）

---

## 关键发现

### ✅ 优点

1. **CB精度高** (0.34Å)
   - CB是侧链锚点，精度高说明侧链方向正确

2. **序列预测优秀** (98.7%)
   - SH密度成功编码了氨基酸类型信息

3. **主链稳定** (CA 0.54Å)
   - 主链原子重建精度可接受

### ⚠️ 问题

1. **远端原子误差大** (atom10-13: 4-5Å)
   - 侧链远端精度不足
   - 影响精细相互作用建模

2. **整体RMSD偏高** (1.67Å)
   - 距离高精度decode (< 0.5Å) 还有差距

---

## 问题诊断

### Loss配置不合理

**当前**:
```
total_loss = 0.2 * atomsloss + 1.0 * typeloss
```

**问题**:
- Type已经98.7%准确，权重1.0过高
- Atom权重0.2太小，模型不够关注坐标精度

**实际Loss值**:
```
atomsloss ≈ 6.81
typeloss ≈ 0.07
→ total_loss = 6.81*0.2 + 0.07*1.0 = 1.43
```

Atom loss虽然绝对值大，但权重低导致贡献被压制。

---

## 改进方案

### 调整Loss权重

**修改**:
```yaml
type_loss_weight: 0.1   # 1.0 → 0.1 (降低10倍)
atom_loss_weight: 1.0   # 0.2 → 1.0 (提高5倍)
```

**预期效果**:
```
total_loss = 6.81*1.0 + 0.07*0.1 ≈ 6.82
```

现在模型会更关注坐标精度。

### 优化目标

| 指标 | 当前 | 目标 |
|------|------|------|
| 整体 RMSD | 1.67Å | < 1.0Å |
| CB RMSD | 0.34Å | < 0.2Å |
| 侧链 RMSD | 2.21Å | < 1.5Å |
| 远端原子 (atom10-13) | 4-5Å | < 3Å |
| 序列准确率 | 98.7% | > 95% (保持) |

---

## 后续实验

### 已完成
- ✅ 添加per-atom RMSD监控
- ✅ 诊断loss权重问题
- ✅ 调整loss配置

### 待完成
- [ ] 重新训练观察改进效果
- [ ] 如果RMSD不再下降，考虑：
  - 分别调整MSE/Huber/Pair权重
  - 添加per-atom权重（给CB更高权重）
  - 增加训练数据或数据增强

---

## 结论

**当前状态**：SH decode可行，但精度不足

**核心问题**：Loss权重配置不合理，模型过度优化type而忽视坐标

**解决方案**：已调整loss权重，期待重新训练后RMSD显著下降

**长期目标**：达到atom14 RMSD < 1.0Å后，再考虑实现SH diffusion

---

**实验路径**: `/home/junyu/project/pu/ckpt/se3-fm_sh/pdb__SH_to_atoms_decoder/2025-10-18_17-30-42/`
**配置文件**: `/home/junyu/project/pu/configs/Train_SH.yaml`
