# 训练与推理一致性检查

## 对比总结表

| 维度 | 训练代码 | 推理代码 | 是否一致 | 问题 |
|------|---------|---------|---------|------|
| **Velocity定义** | `v_t = x1 - x0` | `v_pred` | ✅ 一致 | 无 |
| **ODE积分** | 训练target是`v_t` | `xt = xt + dt*v_pred` | ✅ 一致 | 无 |
| **SH计算** | From `atoms14_local_t` | From `atoms14_local_t` | ✅ 一致 | 无 |
| **Self-conditioning** | `atoms14_local_sc = zeros` | `atoms14_local_sc = zeros` | ✅ 一致 | 无 |
| **Backbone处理** | Clean (0:3不加噪) | Clean (0:3不加噪) | ✅ 一致 | 无 |
| **时间点数量** | N/A | `num_timesteps` | ❌ **Bug** | 应该是`num_timesteps+1` |
| **Velocity key** | `speed_vectors` | `side_atoms` / `speed_vectors` | ❌ **不一致** | Key混用！ |
| **Final prediction** | N/A | `x1 = xt + (1-t)*v` | ✅ 正确 | 公式对 |

---

## 详细分析

### ✅ 1. Velocity定义 - 一致

#### 训练代码 (flow_module.py:413-448)

```python
def model_step_shfbb(self, batch, prob=None):
    # 1. 加噪
    noisy_batch = self.interpolant.fbb_corrupt_batch(batch, prob)

    # fbb_corrupt_batch内部 (interpolant.py:486-489):
    # y_sc = (1.0 - t) * noise_sc + t * clean_sc
    # v_sc = clean_sc - noise_sc  ← velocity = x1 - x0

    # 2. 模型预测
    outs = self.model(noisy_batch)
    speed_vectors = outs['speed_vectors']  # [B,N,11,3]

    # 3. Target
    target_vectors = noisy_batch['v_t'][..., 3:, :]  # v_t = x1 - x0

    # 4. Loss
    vector_loss = F.mse_loss(speed_vectors, target_vectors, ...)
```

**训练的velocity**: `v_t = clean_sc - noise_sc = x1 - x0`

#### 推理代码 (interpolant.py:1614-1618)

```python
# ODE step
dt = t2 - t1
xt = xt + dt * v_pred  # Euler step: dx = v*dt
```

**推理的velocity**: 假设模型预测的是`v = x1 - x0`

**结论**: ✅ 定义一致，都是`v = x1 - x0`

---

### ✅ 2. SH Density计算 - 一致

#### 训练代码 (flow_module.py:421-430)

```python
normalize_density, *_ = sh_density_from_atom14_with_masks_clean(
    noisy_batch['atoms14_local_t'],  # ← 使用noisy坐标
    batch['atom14_element_idx'],
    batch['atom14_gt_exists'],
    L_max=self._model_cfg.sh.L_max,
    R_bins=self._model_cfg.sh.R_bins,
)
normalize_density = normalize_density / torch.sqrt(torch.tensor(4 * torch.pi))
noisy_batch['normalize_density'] = normalize_density
```

#### 推理代码 (interpolant.py:1600-1609)

```python
normalize_density, *_ = sh_density_from_atom14_with_masks_clean(
    input_feats['atoms14_local_t'],  # ← 使用当前迭代的坐标
    batch['atom14_element_idx'],
    batch['atom14_gt_exists'],
    L_max=8,
    R_bins=24,
)
normalize_density = normalize_density / torch.sqrt(torch.tensor(4 * torch.pi))
input_feats['normalize_density'] = normalize_density
```

**结论**: ✅ 完全一致，都是on-the-fly从noisy坐标计算SH

---

### ✅ 3. Self-conditioning - 一致（都禁用了）

#### 训练代码 (flow_module.py:436-439)

```python
if 'atoms14_local_t' in noisy_batch:
    noisy_batch['atoms14_local_sc'] = torch.zeros_like(noisy_batch['atoms14_local_t'])
else:
    noisy_batch['atoms14_local_sc'] = torch.zeros_like(noisy_batch['atoms14_local'])
```

#### 推理代码 (interpolant.py:1620-1622)

```python
clean_pred = xt + (1.0 - t2) * v_pred
input_feats_base['atoms14_local_sc'] = torch.cat([backbone_local, clean_pred], dim=-2) * 0
#                                                                                        ^^^
#                                                                                     乘以0 = 禁用
```

**结论**: ✅ 一致，训练和推理都禁用了self-conditioning

---

### ✅ 4. Backbone处理 - 一致

#### 训练代码 (interpolant.py:494-497)

```python
# fbb_corrupt_batch内部
y_full = atoms14_local.clone()  # ← 从clean atoms14开始
v_full = torch.zeros_like(atoms14_local)
y_full[..., 3:, :] = y_sc  # ← 只有侧链(3:)加噪
v_full[..., 3:, :] = v_sc  # ← 只有侧链有velocity
# Backbone (0:3)保持clean！
```

#### 推理代码 (interpolant.py:1579-1580)

```python
backbone_local = input_feats_base['atoms14_local_t'][..., :3, :]  # ← Backbone不变
xt = input_feats_base['atoms14_local_t'][..., 3:, :]  # ← 只更新侧链
```

**结论**: ✅ 一致，训练和推理都只对侧链加噪/更新，backbone始终保持clean

---

### ❌ 5. 时间点数量 - Bug

#### 训练代码

训练时每个batch采样一个随机的`t`，没有"步数"概念。

#### 推理代码 (interpolant.py:1573)

```python
# 修复：num_timesteps是步数，需要num_timesteps+1个时间点 (包括起点和终点)
# 例如：1步需要[t0, t1]两个点，10步需要[t0, t1, ..., t10]共11个点
ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps, device=device)  # ❌ 少一个点！
```

**问题**:
- 注释说需要`num_timesteps+1`个点
- 代码却只生成`num_timesteps`个点
- **这是之前发现的bug！**

**修复**:
```python
ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps + 1, device=device)  # ✅
```

---

### ❌ 6. Velocity Key不一致 - Bug

#### 训练代码 (flow_module.py:442-443)

```python
outs = self.model(noisy_batch)
speed_vectors = outs['speed_vectors']  # ← 训练时用这个key
speed_pred = outs['speed_pred']  # ← 这是速度标量
```

#### 推理代码

**循环中** (interpolant.py:1611-1612):
```python
out = model(input_feats)
v_pred = out['side_atoms']  # ← 推理时用这个key (正确)
```

**Final step** (interpolant.py:1716):
```python
v_final = out_final['speed_vectors']  # ← 这里又用了训练的key (错误！)
```

**问题**:
- 训练时模型返回`speed_vectors`
- 推理循环中用`side_atoms`
- 推理Final step用`speed_vectors`
- **Key不统一！**

**检查模型输出的key**:

让我查看模型返回什么：

## 🔴 关键发现：模型返回值分析

### 模型Forward函数 (flow_model.py:1117-1125)

```python
return {
    'speed_vectors': speed_vectors,  # ← Velocity向量 [B,N,11,3]
    'speed_pred': speed_pred,        # ← Velocity标量 [B,N,11]
    'side_atoms': side_atoms,        # ← Clean prediction x1 = xt + (1-t)*v
    'side_atoms_local_full': local_full,
    'atoms_global_full': global_full,
    'rigids_global': curr_rigids_ang,
    'logits': logits,
}
```

**计算逻辑** (flow_model.py:1084-1108):
```python
# 1. 模型预测velocity
speed_vectors, _ = self.NodeFeatExtractorWithHeads(node_embed, node_mask)

# 2. 计算clean prediction
t_factor = (1.0 - r3_t)[..., None, None]
side_atoms = xt_side + t_factor * speed_vectors  # x1 = xt + (1-t)*v
```

---

## ❌ 严重问题：推理代码中velocity vs clean prediction混淆

### 训练代码使用 - 正确

```python
# flow_module.py:442
speed_vectors = outs['speed_vectors']  # ← 取velocity
target_vectors = noisy_batch['v_t'][..., 3:, :]  # ← target也是velocity
loss = MSE(speed_vectors, target_vectors)  # ✅ 匹配
```

### 推理代码循环 - **错误！**

```python
# interpolant.py:1611-1617
out = model(input_feats)
v_pred = out['side_atoms']  # ❌ 取了clean prediction，不是velocity！

dt = t2 - t1
xt = xt + dt * v_pred  # ❌ 用clean prediction做ODE step！
```

**问题**:
```
side_atoms = xt + (1-t)*v  ← 这是clean prediction x1

如果用它做ODE step:
xt_new = xt + dt * side_atoms
       = xt + dt * (xt + (1-t)*v)
       = xt + dt*xt + dt*(1-t)*v  ← 完全错误！

应该用:
xt_new = xt + dt * speed_vectors
       = xt + dt * v  ← 正确的Euler step
```

### 推理Final step - 部分正确

```python
# interpolant.py:1716
v_final = out_final['speed_vectors']  # ✅ 取了velocity（对的key）

clean_final = xt + (1.0 - t_final) * v_final  # ✅ 公式正确
```

---

## 🚨 关键Bug总结

### Bug 1: 推理循环使用错误的key

**位置**: `interpolant.py:1612`

**当前代码**:
```python
v_pred = out['side_atoms']  # ❌ 这是x1，不是v！
```

**应该修复为**:
```python
v_pred = out['speed_vectors']  # ✅ 这是velocity
```

### Bug 2: 时间点数量错误

**位置**: `interpolant.py:1573`

**当前代码**:
```python
ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps, device=device)
```

**应该修复为**:
```python
ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps + 1, device=device)
```

### Bug 3: Final step key不一致（但碰巧对了）

**位置**: `interpolant.py:1716`

**当前代码**:
```python
v_final = out_final['speed_vectors']  # ✅ 碰巧用对了
```

**建议统一**:
所有地方都用`speed_vectors`作为velocity的key。

---

## 🔧 修复方案

### 修复后的推理代码

```python
def fbb_sample_iterative(self, batch, model, num_timesteps=None):
    device = batch['res_mask'].device
    B, N = batch['res_mask'].shape

    res_mask = batch['res_mask']
    # ... 其他初始化 ...

    if num_timesteps is None:
        num_timesteps = self._sample_cfg.num_timesteps

    # ✅ 修复1: 正确的时间点数量
    ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps + 1, device=device)

    # Prepare base features
    input_feats_base = copy.deepcopy(batch)
    backbone_local = input_feats_base['atoms14_local_t'][..., :3, :]
    xt = input_feats_base['atoms14_local_t'][..., 3:, :]
    input_feats_base['atoms14_local_sc'] = torch.zeros_like(input_feats_base['atoms14_local_t'])

    for i in tqdm.tqdm(range(len(ts) - 1)):
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

        # ✅ 修复2: 使用正确的velocity key
        v_pred = out['speed_vectors']  # ← 改用speed_vectors

        # Standard Euler ODE step
        dt = t2 - t1
        xt = xt + dt * v_pred  # ✅ 现在是正确的Euler step
        xt = xt * side_exists[..., None]

        # Self-conditioning (disabled)
        clean_pred = xt + (1.0 - t2) * v_pred
        input_feats_base['atoms14_local_sc'] = torch.cat([backbone_local, clean_pred], dim=-2) * 0

    # Final step
    t_final = float(ts[-1])  # 现在正确等于1.0
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

    # ✅ 统一使用speed_vectors
    v_final = out_final['speed_vectors']
    final_logits = out_final.get('logits', None)

    clean_final = xt + (1.0 - t_final) * v_final  # (1.0 - 1.0) = 0，无校正

    atoms14_local_final = torch.cat([backbone_local, clean_final], dim=-2)
    if side_exists is not None:
        atoms14_local_final[..., 3:, :] = atoms14_local_final[..., 3:, :] * side_exists[..., None]

    # Build global coordinates
    rigid = du.create_rigid(rotmats_1, trans_1)
    atoms14_global_final = rigid[..., None].apply(atoms14_local_final)

    return {
        'atoms14_local_final': atoms14_local_final,
        'atoms14_global_final': atoms14_global_final,
        'logits_final': final_logits,
        'diagnostics': diagnostics,
    }
```

---

## 🎯 为什么之前能work（但结果不对）？

### 误用side_atoms的后果

```python
# 错误的实现
v_pred = out['side_atoms']  # = xt + (1-t)*v
xt_new = xt + dt * v_pred
       = xt + dt * (xt + (1-t)*v)
       = xt * (1 + dt) + dt*(1-t)*v

# 正确的实现
v_pred = out['speed_vectors']  # = v
xt_new = xt + dt * v
```

**为什么没有完全崩溃？**

当`dt`很小时（例如10步，dt=0.1）:
```
错误: xt_new ≈ xt * 1.1 + 0.1*(1-t)*v
正确: xt_new = xt + 0.1*v

差异: 多了0.1*xt项
```

这个额外的`0.1*xt`项会导致：
1. **发散**: xt会被"放大"1.1倍
2. **累积误差**: 每步都多加一点当前位置
3. **最终偏离**: 10步后累积效应明显

**这可能就是为什么你的TM-score是0.660而不是更高！**

---

## 📊 预期修复后的改进

| 指标 | 修复前 (错误的side_atoms) | 修复后 (正确的speed_vectors) | 预期改进 |
|------|--------------------------|----------------------------|---------|
| TM-score | 0.660 ± 0.267 | ? | +5-10% |
| CA RMSD | 10.92 ± 11.76 Å | ? | -10-20% |
| Sidechain RMSD | ? | ? | -15-25% |
| Recovery | 0.907 | ? | +1-2% |

**原因**:
1. 正确的ODE积分路径
2. 完整的10步采样（而不是9步）
3. 最终时间真正到达t=1.0

---

## 📝 总结

### 训练与推理的一致性

| 维度 | 一致性 | 问题 |
|------|--------|------|
| Velocity定义 | ✅ | 无 |
| SH计算 | ✅ | 无 |
| Backbone处理 | ✅ | 无 |
| Self-conditioning | ✅ | 无 |
| **Velocity key** | ❌ | **训练用speed_vectors，推理用side_atoms** |
| **时间点数量** | ❌ | **少一个时间点** |
| Final prediction | ✅ | 公式正确 |

### 必须修复的Bug

1. ❌ **Bug 1 (Critical)**: 推理循环使用`out['side_atoms']`应改为`out['speed_vectors']`
2. ❌ **Bug 2 (Critical)**: 时间点数量应为`num_timesteps + 1`
3. ⚠️ **建议**: 统一所有地方使用`speed_vectors`作为velocity的key

### 修复后预期

- **更准确的ODE积分路径**
- **更低的RMSD**（可能降低10-20%）
- **更高的TM-score**（可能提升5-10%）
- **更好的几何质量**

---

**生成日期**: 2025-11-11
**结论**: 发现了关键的train-test mismatch bug，推理代码错误地使用了clean prediction而不是velocity
