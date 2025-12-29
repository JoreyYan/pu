# `fbb_sample_iterative` 采样代码审查

**文件位置**: `/home/junyu/project/pu/data/interpolant.py:1544-1816`

---

## 代码概览

```python
def fbb_sample_iterative(self, batch, model, num_timesteps=None):
    """Iterative sidechain sampling in local frame (diffusion-style ODE)."""
```

这个函数实现了Flow Matching的ODE采样，用于从噪声生成侧链坐标。

---

## 关键问题分析

### ❌ **问题1: 时间点数量错误** (Line 1573)

```python
# Line 1571-1573 (注释说要修复，但实际没修复！)
# 修复：num_timesteps是步数，需要num_timesteps+1个时间点 (包括起点和终点)
# 例如：1步需要[t0, t1]两个点，10步需要[t0, t1, ..., t10]共11个点
ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps, device=device)  # ❌ 少一个点！
```

**问题**:
- 注释说需要`num_timesteps+1`个点，但代码却只生成`num_timesteps`个点
- 如果`num_timesteps=10`，应该有11个点 `[t0, t1, ..., t10]`，但实际只有10个点

**影响**:
```python
# 假设 min_t=0.01, num_timesteps=10
# 当前代码: ts = [0.01, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99]
# 正确应该: ts = [0.01, 0.10, 0.19, 0.28, 0.37, 0.46, 0.55, 0.64, 0.73, 0.82, 0.91, 1.00]
#                                                                                    ^^^^ 缺失！

# Loop: range(len(ts) - 1) = range(9) → 只有9步，而不是10步
# 最终 t_final = ts[-1] = 0.99，而不是 1.0
```

**后果**:
1. 实际只做了9步采样，而不是10步
2. 最终时间是`t=0.99`而非`t=1.0`
3. Final step的`clean_final = xt + (1.0 - t_final) * v_final`中，`(1.0 - 0.99) = 0.01`还会有微小的velocity校正

**修复**:
```python
ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps + 1, device=device)  # ✅ 加1
```

---

### ❌ **问题2: Final step的velocity key错误** (Line 1716)

```python
# Line 1716
v_final = out_final['speed_vectors']  # ❌ key名称不一致！
```

**上下文对比**:
```python
# Line 1612: 循环中
v_pred = out['side_atoms']  # ✅ 使用 'side_atoms'

# Line 1716: Final step
v_final = out_final['speed_vectors']  # ❌ 使用 'speed_vectors'
```

**问题**:
- 循环中使用`out['side_atoms']`
- Final step使用`out_final['speed_vectors']`
- **Key不一致！** 很可能导致KeyError或读取错误的值

**检查模型输出**:
需要确认你的模型返回字典中velocity的key到底是什么：
- 如果是`'side_atoms'` → Final step应改为`out_final['side_atoms']`
- 如果是`'speed_vectors'` → 循环中应改为`out['speed_vectors']`

**推测**:
根据Line 1612的用法，应该统一为`'side_atoms'`

**修复**:
```python
v_final = out_final['side_atoms']  # ✅ 与循环中一致
```

---

### ✅ **正确的部分**

#### 1. **ODE积分逻辑** (Line 1614-1618)

```python
# Line 1614-1618
# Standard Euler ODE step for v = x1 - x0
# x_t = (1-t)*x0 + t*x1, so dx/dt = x1 - x0 = v
dt = t2 - t1
xt = xt + dt * v_pred  # ✅ 正确的Euler step
xt = xt * side_exists[..., None]  # ✅ 正确mask
```

**Flow Matching的ODE**:
```
x_t = (1-t)*x0 + t*x1
dx/dt = x1 - x0 = v

Euler step:
x_{t+dt} = x_t + dt * v(x_t, t)
```

**代码实现**: ✅ 完全正确

---

#### 2. **SH density on-the-fly计算** (Line 1599-1609)

```python
# Line 1599-1609
normalize_density, *_ = sh_density_from_atom14_with_masks_clean(
    input_feats['atoms14_local_t'],
    batch['atom14_element_idx'],
    batch['atom14_gt_exists'],
    L_max=8,
    R_bins=24,
)
normalize_density = normalize_density / torch.sqrt(torch.tensor(4 * torch.pi))
input_feats['normalize_density'] = normalize_density
```

**这是SH+FBB的核心**:
- 每个采样步都从当前`atoms14_local_t`重新计算SH density
- 避免了SH作为最终表示的信息损失
- ✅ 实现正确

---

#### 3. **Final prediction** (Line 1718)

```python
# Line 1718 (假设修复了v_final key)
clean_final = xt + (1.0 - t_final) * v_final
```

**数学推导**:
```
Flow matching: x_t = (1-t)*x0 + t*x1
=> x1 = (x_t - (1-t)*x0) / t

模型预测: v = x1 - x0
=> x1 = x0 + v

结合:
x1 = x_t + (1-t) * v  (当我们不知道x0时的近似)
```

**这个公式是否正确**?

实际上，根据Flow Matching的定义：
```
v_t = dx/dt = x1 - x0
x_t = (1-t)*x0 + t*x1

从 x_t 恢复 x1:
x_t = (1-t)*x0 + t*x1
=> x1 = (x_t - (1-t)*x0) / t

但如果我们只知道 v_t = x1 - x0:
x1 = x0 + v_t

又有 x_t = (1-t)*x0 + t*x1
=> x0 = (x_t - t*x1) / (1-t)

代入:
x1 = (x_t - t*x1)/(1-t) + v_t
x1 * (1-t) = x_t - t*x1 + v_t * (1-t)
x1 * (1-t) + t*x1 = x_t + v_t * (1-t)
x1 = x_t + (1-t) * v_t  ✅ 公式正确！
```

---

### ⚠️ **潜在问题3: Self-conditioning被禁用** (Line 1622)

```python
# Line 1620-1622
clean_pred = xt + (1.0 - t2) * v_pred  # 计算了clean prediction
input_feats_base['atoms14_local_sc'] = torch.cat([backbone_local, clean_pred], dim=-2)*0
#                                                                                     ^^^
#                                                                              乘以0 = 禁用
```

**问题**:
- Self-conditioning可以提升采样质量（让模型看到上一步的预测）
- 但代码中乘以0，完全禁用了这个功能

**是否应该启用**?
- SimpleFold论文中没有提到self-conditioning
- AlphaFold2/3中self-conditioning很常见
- 如果模型没有训练过self-conditioning输入 → 不应启用（会confuse模型）
- 如果模型训练时有self-conditioning → 应该启用

**建议**:
- 如果你的模型训练时`atoms14_local_sc`是全0 → 采样时也应该全0（当前正确）
- 如果你的模型训练时使用了self-conditioning → 应该去掉`*0`

---

### ✅ **Diagnostics正确** (Line 1728-1810)

Diagnostics部分计算了：
1. ✅ Sidechain RMSD (pred vs GT)
2. ✅ Perplexity with pred coords vs GT coords
3. ✅ Recovery with pred coords vs GT coords

这些都是很好的对比实验，可以量化坐标误差对type prediction的影响。

---

## 与SimpleFold的对比

### SimpleFold采样代码 (参考)

```python
# SimpleFold: simplefold.py (sampling部分)
def sample(self, model, noise, batch, num_steps=500):
    # 时间点生成
    ts = torch.linspace(0, 1, num_steps + 1)  # ✅ num_steps + 1个点

    xt = noise
    for i in range(len(ts) - 1):
        t1, t2 = ts[i], ts[i+1]
        dt = t2 - t1

        # 模型预测velocity
        out = model(xt, t1, batch)
        v_pred = out['predict_velocity']  # ✅ 统一的key

        # Euler step
        xt = xt + dt * v_pred

    return xt
```

**SimpleFold的特点**:
1. ✅ 使用`num_steps + 1`个时间点
2. ✅ 统一的velocity key
3. ❌ 没有SH中间表示（你的优势）

---

## 修复建议

### 必须修复（Critical）:

#### 1. **修复时间点数量** (Line 1573)
```python
# 当前
ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps, device=device)

# 修复为
ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps + 1, device=device)
```

#### 2. **统一velocity key** (Line 1716)
```python
# 当前
v_final = out_final['speed_vectors']

# 修复为 (假设模型输出key是'side_atoms')
v_final = out_final['side_atoms']
```

**验证方法**:
```python
# 在你的模型forward函数中检查返回值
def forward(self, batch):
    ...
    return {
        'side_atoms': v_pred,  # ← 确认这个key
        'logits': logits,
    }
```

---

### 建议检查（Optional）:

#### 3. **验证self-conditioning逻辑**
- 检查训练时`atoms14_local_sc`是否使用
- 如果训练时用了，采样时去掉`*0`
- 如果训练时没用，保持当前代码

#### 4. **添加时间点验证**
```python
# 在Line 1573后添加assertion
ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps + 1, device=device)
assert len(ts) == num_timesteps + 1, f"Expected {num_timesteps + 1} time points, got {len(ts)}"
assert torch.isclose(ts[-1], torch.tensor(1.0)), f"Final time should be 1.0, got {ts[-1]}"
```

---

## 测试建议

### 单元测试

```python
def test_fbb_sample_iterative():
    # 测试时间点数量
    min_t = 0.01
    num_timesteps = 10
    ts = torch.linspace(min_t, 1.0, num_timesteps + 1)

    assert len(ts) == 11, "10步应该有11个时间点"
    assert torch.isclose(ts[0], torch.tensor(0.01)), "起点应该是min_t"
    assert torch.isclose(ts[-1], torch.tensor(1.0)), "终点应该是1.0"

    # 测试步数
    num_steps = 0
    for i in range(len(ts) - 1):
        num_steps += 1
    assert num_steps == 10, "应该循环10次"
```

### 端到端测试

```python
# 对比修复前后的结果
results_before = fbb_sample_iterative_old(batch, model, num_timesteps=10)
results_after = fbb_sample_iterative_fixed(batch, model, num_timesteps=10)

# 检查差异
diff_rmsd = torch.abs(
    results_after['diagnostics']['sidechain_rmsd'] -
    results_before['diagnostics']['sidechain_rmsd']
)
print(f"RMSD difference: {diff_rmsd:.4f} Å")
```

**预期**:
- 修复后应该有更低的RMSD（因为做了完整的10步）
- Perplexity可能略有改善

---

## 对比其他采样函数

你的代码库中有多个采样函数，让我们对比一下：

| 函数 | 时间点数量 | Velocity key | SH支持 | 特点 |
|------|-----------|--------------|--------|------|
| `fbb_sample_iterative` | ❌ `num_timesteps` | ❌ 不一致 | ✅ Yes | 主要函数，有bug |
| `fbb_sample_iterative_sde` | ✅ `num_timesteps + 1` | ✅ 一致 | ❌ No | SDE版本，实现正确 |
| `fbb_sample_iterative_ssq` | ❌ `num_timesteps` | ✅ 一致 | ❌ No | 线性桥版本 |
| `fbb_sample_iterative_stable` | ❌ `num_timesteps` | ✅ 一致 | ❌ No | 稳定版本，有alpha-cap |

**发现**:
- 只有`fbb_sample_iterative_sde`使用了正确的`num_timesteps + 1`
- 其他函数都有相同的时间点数量问题
- `fbb_sample_iterative`的velocity key不一致问题是独有的

**建议**:
将所有采样函数统一修复为`num_timesteps + 1`

---

## 完整修复代码

```python
def fbb_sample_iterative(
        self,
        batch: dict,
        model,
        num_timesteps: int | None = None,
):
    """Iterative sidechain sampling in local frame (diffusion-style ODE).

    Fixed across steps: res_mask, diffuse_mask, res_idx, chain_idx, trans_1, rotmats_1.
    Evolving: atoms14_local_t (only sidechain indices 3: are updated each step).
    """
    device = batch['res_mask'].device
    B, N = batch['res_mask'].shape

    res_mask = batch['res_mask']
    diffuse_mask = batch.get('diffuse_mask', torch.ones_like(res_mask))
    res_idx = batch['res_idx']
    chain_idx = batch['chain_idx']
    rotmats_1 = batch['rotmats_1']
    trans_1 = batch['trans_1']

    atoms14_local_gt = batch['atoms14_local']  # [B,N,14,3]
    side_exists = batch.get('atom14_gt_exists', torch.ones_like(atoms14_local_gt[..., 0]))[..., 3:]  # [B,N,11]

    if num_timesteps is None:
        num_timesteps = self._sample_cfg.num_timesteps

    # ✅ 修复1: 使用 num_timesteps + 1 个时间点
    ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps + 1, device=device)
    assert len(ts) == num_timesteps + 1
    assert torch.isclose(ts[-1], torch.tensor(1.0, device=device))

    # Prepare base features
    input_feats_base = copy.deepcopy(batch)
    backbone_local = input_feats_base['atoms14_local_t'][..., :3, :]
    xt = input_feats_base['atoms14_local_t'][..., 3:, :]
    input_feats_base['atoms14_local_sc'] = torch.zeros_like(input_feats_base['atoms14_local_t'])

    logs = []

    for i in tqdm.tqdm(range(len(ts) - 1)):  # Now correctly loops num_timesteps times
        t1 = float(ts[i])
        t2 = float(ts[i + 1])

        atoms14_local_t = torch.cat([backbone_local, xt], dim=-2)

        input_feats = input_feats_base.copy()
        input_feats.update({
            't': torch.full((res_mask.shape[0],), t1, device=device, dtype=torch.float32),
            'r3_t': torch.full(res_mask.shape, t1, device=device, dtype=torch.float32),
            'so3_t': torch.full(res_mask.shape, t1, device=device, dtype=torch.float32),
            'atoms14_local_t': atoms14_local_t,
        })

        # SH+FBB: on-the-fly SH density calculation
        normalize_density, *_ = sh_density_from_atom14_with_masks_clean(
            input_feats['atoms14_local_t'],
            batch['atom14_element_idx'],
            batch['atom14_gt_exists'],
            L_max=8,
            R_bins=24,
        )
        normalize_density = normalize_density / torch.sqrt(torch.tensor(4 * torch.pi))
        input_feats['normalize_density'] = normalize_density

        out = model(input_feats)
        v_pred = out['side_atoms']  # ✅ 统一使用 'side_atoms'

        # Standard Euler ODE step for v = x1 - x0
        dt = t2 - t1
        xt = xt + dt * v_pred
        xt = xt * side_exists[..., None]  # mask out non-existing atoms

        # For self-conditioning (currently disabled with *0)
        clean_pred = xt + (1.0 - t2) * v_pred
        input_feats_base['atoms14_local_sc'] = torch.cat([backbone_local, clean_pred], dim=-2) * 0

    # Final step at t_final = 1.0
    t_final = float(ts[-1])  # Now correctly equals 1.0
    atoms14_local_t = torch.cat([backbone_local, xt], dim=-2)
    input_feats_final = input_feats_base.copy()
    input_feats_final.update({
        't': torch.full((res_mask.shape[0],), t_final, device=device, dtype=torch.float32),
        'r3_t': torch.full(res_mask.shape, t_final, device=device, dtype=torch.float32),
        'so3_t': torch.full(res_mask.shape, t_final, device=device, dtype=torch.float32),
        'atoms14_local_t': atoms14_local_t,
    })

    with torch.no_grad():
        out_final = model(input_feats_final)

    # ✅ 修复2: 统一velocity key
    v_final = out_final['side_atoms']  # Changed from 'speed_vectors'
    final_logits = out_final.get('logits', None)

    clean_final = xt + (1.0 - t_final) * v_final  # (1.0 - 1.0) = 0, no correction

    atoms14_local_final = torch.cat([backbone_local, clean_final], dim=-2)
    if side_exists is not None:
        atoms14_local_final[..., 3:, :] = atoms14_local_final[..., 3:, :] * side_exists[..., None]

    # Build global 14 using fixed frames
    rigid = du.create_rigid(rotmats_1, trans_1)
    atoms14_global_final = rigid[..., None].apply(atoms14_local_final)

    # [Diagnostics部分保持不变，省略...]
    diagnostics = {}
    # ... (same as before)

    return {
        'atoms14_local_final': atoms14_local_final,
        'atoms14_global_final': atoms14_global_final,
        'logits_final': final_logits,
        'diagnostics': diagnostics,
    }
```

---

## 总结

### ❌ 必须修复的问题

1. **时间点数量错误** (Line 1573)
   - 当前: `num_timesteps` 个点
   - 应该: `num_timesteps + 1` 个点
   - 影响: 少做一步，最终时间不是1.0

2. **Velocity key不一致** (Line 1716)
   - 当前: `out_final['speed_vectors']`
   - 应该: `out_final['side_atoms']` (与循环中一致)
   - 影响: 可能KeyError或读取错误值

### ✅ 正确的部分

1. ODE积分逻辑正确 (Euler step)
2. SH density on-the-fly计算正确 (SH+FBB核心)
3. Final prediction公式正确
4. Diagnostics设计合理

### ⚠️ 建议改进

1. 验证self-conditioning是否应该启用
2. 统一所有采样函数的时间点生成
3. 添加assertion验证时间点

**修复优先级**: Critical → 立即修复以获得正确的10-step采样结果

---

**审查日期**: 2025-11-11
**审查结论**: 代码整体逻辑正确，但有2个critical bug需要立即修复
