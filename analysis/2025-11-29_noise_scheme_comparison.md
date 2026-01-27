# 三种加噪方式对比：tokenatoms vs side_atoms vs Gaussianatoms

**日期**: 2025-11-29
**文件**: `data/interpolant.py:471-620`

---

## 快速对比表

| 特性 | `side_atoms` | `tokenatoms` | `Gaussianatoms` |
|------|-------------|-------------|-----------------|
| **任务类型** | Flow Matching | Token Masking | Inverse Folding (回归) |
| **时间步 t** | ✅ 动态采样 [0,1] | ❌ 固定 t=1 | ❌ 固定 t=1 |
| **噪声类型** | 高斯噪声 (连续) | 确定性mask (离散) | 确定性mask (离散) |
| **插值公式** | `y = (1-t)*noise + t*clean` | 不适用 | 不适用 |
| **目标预测** | Velocity `v = clean - noise` | 直接预测坐标 | 直接预测坐标 + 高斯参数 |
| **训练范式** | Diffusion/Flow | BERT-style | Regression |
| **Context支持** | ✅ update_mask=0保留clean | ✅ update_mask=0保留clean | ✅ update_mask=0保留clean |
| **适用模型** | SideAtomsFlowModel | SideAtomsFlowModel | SideAtomsIGAModel |

---

## 详细对比

### 1. **side_atoms** - Flow Matching加噪

#### 实现 (line 471-513)

```python
# 采样时间步
t = self.sample_t(batch_size)[:, None]  # [B, 1] in [0, 1]

# 生成高斯噪声
noise_sc = torch.randn_like(atoms14_local[..., 3:, :]) * coord_scale

# Flow Matching插值
y_sc = (1.0 - t) * noise_sc + t * clean_sc  # y_t
v_sc = clean_sc - noise_sc                    # velocity

# 应用update_mask
y_sc = torch.where(update_mask, y_sc, clean_sc)  # mask=0保留clean
v_sc = torch.where(update_mask, v_sc, 0)
```

#### 特点

**优势**:
- ✅ **连续时间**: t ∈ [0,1]，覆盖从纯噪声到干净数据的全过程
- ✅ **理论保证**: Flow Matching有完整的理论框架
- ✅ **多样性**: 可以通过采样不同t获得不同难度的训练样本
- ✅ **ODE/SDE采样**: 推理时可以用ODE solver或SDE获得高质量样本

**劣势**:
- ❌ **训练复杂**: 需要学习velocity field
- ❌ **采样慢**: 需要多步ODE integration
- ❌ **数值敏感**: 对t的分布、coord_scale敏感

**输出**:
```python
noisy_batch['atoms14_local_t'] = y_full  # 加噪坐标
noisy_batch['v_t'] = v_full              # velocity目标
noisy_batch['t'] = t                     # 时间步
noisy_batch['r3_t'] = t_broadcast        # 用于SNR weighting
```

**训练Loss**:
- Velocity MSE: `||v_pred - v_gt||^2`
- 坐标MSE: `||x_pred - x_gt||^2` (预测的clean坐标)

**适用场景**:
- 需要生成多样性样本
- 追求SOTA质量
- 有计算资源做多步采样

---

### 2. **tokenatoms** - BERT风格Token Masking

#### 实现 (line 517-554)

```python
# 创建masked版本
atoms14_masked = atoms14_local.clone()

# 目标区域 (update_mask=1): 侧链坐标置0
atoms14_masked[..., 3:, :] = torch.where(
    update_mask_exp,
    torch.zeros_like(atoms14_local[..., 3:, :]),  # Mask → 0
    atoms14_local[..., 3:, :]                     # Context → clean
)

# 固定t=1 (确定性)
t = torch.ones(batch_size, device=device)
```

#### 特点

**优势**:
- ✅ **简单直观**: 类似BERT的[MASK]机制，易于理解
- ✅ **快速推理**: 单次前向传播，无需多步采样
- ✅ **稳定训练**: 无需调节t分布、noise scale等超参
- ✅ **Context明确**: 0坐标明确表示"需要预测"

**劣势**:
- ❌ **信息丢失**: 0坐标丢失了侧链的先验信息
- ❌ **单一模式**: 只有一种mask模式，没有渐进式训练
- ❌ **冷启动**: 从0坐标直接预测到真实坐标，跨度大

**输出**:
```python
noisy_batch['atoms14_local_t'] = atoms14_masked  # 0坐标输入
noisy_batch['v_t'] = v_full                      # clean target
noisy_batch['t'] = torch.ones(B)                 # t=1占位符
```

**训练Loss**:
- 直接坐标MSE: `||x_pred - x_gt||^2`

**适用场景**:
- 快速原型验证
- Inverse Folding (给定backbone预测侧链)
- 不需要采样多样性

---

### 3. **Gaussianatoms** - 高斯回归模式

#### 实现 (line 556-612)

```python
# 与tokenatoms类似：masked输入
atoms14_masked[..., 3:, :] = torch.where(
    update_mask_exp,
    torch.zeros_like(atoms14_local[..., 3:, :]),  # Mask → 0
    atoms14_local[..., 3:, :]                     # Context → clean
)

# Target: clean sidechains
v_target_sc = atoms14_local[..., 3:, :] * effective_mask
v_full[..., 3:, :] = v_target_sc

# 固定t=1
noisy_batch['t'] = torch.ones(batch_size, device=device)
```

#### 特点

**优势**:
- ✅ **概率框架**: 输出高斯参数(offset, scaling)，建模不确定性
- ✅ **额外监督**: Gaussian parameter MSE + NLL loss
- ✅ **物理意义**: 高斯椭球反映侧链体积和形状
- ✅ **快速推理**: 单次前向传播
- ✅ **Context明确**: 0坐标 + 高斯先验

**劣势**:
- ❌ **需要GT高斯**: 数据集需要预计算或在线计算高斯参数
- ❌ **Loss复杂**: 需要同时优化坐标、高斯参数、NLL
- ❌ **数值敏感**: NLL loss对权重非常敏感 (见测试报告)

**输出**:
```python
noisy_batch['atoms14_local_t'] = atoms14_masked  # 0坐标输入
noisy_batch['v_t'] = v_full                      # clean target
noisy_batch['sidechain_atom_mask'] = effective_mask
noisy_batch['t'] = torch.ones(B)                 # 占位符
```

**训练Loss** (SideAtomsIGALoss_Final):
- 坐标MSE: `||x_pred - x_gt||^2`
- Pairwise distance: `||d_pred - d_gt||^2`
- Huber loss: robust regression
- Gaussian parameter MSE: `||offset_pred - offset_gt||^2 + ||scale_pred - scale_gt||^2`
- NLL loss: `-log p(x_gt | Gaussian_pred)` (权重0.0003)
- Sequence loss: CE loss for amino acid type

**适用场景**:
- 需要不确定性估计
- 侧链形状和体积重要（如分子对接）
- Inverse Folding + Structure Quality Assessment

---

## 实验建议

### 三种模式的配置

#### 1. side_atoms (Flow Matching)

```yaml
# configs/Train_SH.yaml
experiment:
  task: shfbb
  noise_scheme: side_atoms

interpolant:
  coord_scale: 1.0
  sampling:
    num_timesteps: 10  # ODE steps
    do_sde: True       # SDE采样

training:
  use_snr_weight: True  # SNR weighting
  atom_loss_weight: 1.0
```

**预期结果**:
- 训练时间: 最长 (需要学习velocity)
- 推理时间: 最长 (10-100步ODE)
- 质量: SOTA
- 多样性: 高

#### 2. tokenatoms (BERT Masking)

```yaml
# 新配置
experiment:
  task: fbb
  noise_scheme: tokenatoms

training:
  atom_loss_weight: 1.0
  type_loss_weight: 0.01
```

**预期结果**:
- 训练时间: 快
- 推理时间: 极快 (1步)
- 质量: 中等
- 多样性: 低 (deterministic)

#### 3. Gaussianatoms (IGA)

```yaml
# configs/Train_esmsd.yaml
experiment:
  task: fbb
  noise_scheme: Gaussianatoms

training:
  atom_loss_weight: 1.0
  pair_loss_weight: 1.0
  huber_loss_weight: 1.0
  w_param: 5.0
  w_nll: 0.0003  # ⚠️ 关键：必须调小
  type_loss_weight: 0.01
```

**预期结果**:
- 训练时间: 中等
- 推理时间: 极快 (1步)
- 质量: 高 (概率监督)
- 多样性: 中 (可通过Gaussian采样)
- 额外: 输出不确定性估计

---

## 三者的关系

```
                    ┌────────────────────┐
                    │   Protein Backbone │
                    │   (Fixed/Known)    │
                    └──────────┬─────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
         side_atoms      tokenatoms     Gaussianatoms
              │                │                │
              ▼                ▼                ▼
    ┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
    │ Flow Matching   │ │ BERT Masking │ │ Gaussian Regress │
    │                 │ │              │ │                  │
    │ • t ∈ [0,1]    │ │ • t = 1      │ │ • t = 1          │
    │ • Noisy → Clean│ │ • 0 → Clean  │ │ • 0 → Gaussian   │
    │ • Multi-step   │ │ • One-step   │ │ • One-step       │
    │ • High quality │ │ • Fast       │ │ • Probabilistic  │
    └─────────────────┘ └──────────────┘ └──────────────────┘
              │                │                │
              └────────────────┴────────────────┘
                               │
                        Output: Sidechain
                        Coordinates (+ Gaussian)
```

---

## 性能对比 (预估)

| Metric | side_atoms | tokenatoms | Gaussianatoms |
|--------|-----------|-----------|---------------|
| **训练速度** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **推理速度** | ⭐ (10-100步) | ⭐⭐⭐⭐⭐ (1步) | ⭐⭐⭐⭐⭐ (1步) |
| **坐标精度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **样本多样性** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| **不确定性** | ❌ | ❌ | ✅ (Gaussian) |
| **实现复杂度** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **调参难度** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ (NLL权重) |

---

## 选择建议

### 使用 **side_atoms** 如果:
- ✅ 追求最高质量
- ✅ 需要多样性采样
- ✅ 有充足的计算资源
- ✅ 愿意调节t分布、coord_scale等超参

### 使用 **tokenatoms** 如果:
- ✅ 快速原型验证
- ✅ 只需要单一预测（不需要多样性）
- ✅ Inverse Folding任务
- ✅ 计算资源有限

### 使用 **Gaussianatoms** 如果:
- ✅ 需要不确定性估计
- ✅ 侧链体积/形状重要
- ✅ 希望额外的概率监督
- ✅ 愿意仔细调节NLL权重（见测试报告）

---

## 当前代码状态

根据 `Train_esmsd.yaml`:

```yaml
experiment:
  task: fbb
  noise_scheme: Gaussianatoms  # ← 当前使用
```

当前你的配置使用 **Gaussianatoms** + **SideAtomsIGAModel** + **SideAtomsIGALoss_Final**。

这是最先进的配置，结合了：
- ✅ 快速推理 (1步)
- ✅ 概率监督
- ✅ 不确定性估计
- ✅ 已修复dtype问题
- ✅ 已校准NLL权重 (0.0003)

**推荐**: 保持当前配置，直接训练观察效果！

---

## 代码位置总结

| 加噪方式 | 代码位置 | 适配模型 | Loss函数 |
|---------|---------|---------|---------|
| `side_atoms` | `interpolant.py:471-513` | SideAtomsFlowModel | Velocity MSE + Coord MSE |
| `tokenatoms` | `interpolant.py:517-554` | SideAtomsFlowModel | Coord MSE |
| `Gaussianatoms` | `interpolant.py:556-612` | SideAtomsIGAModel | SideAtomsIGALoss_Final |

---

**报告生成时间**: 2025-11-29
