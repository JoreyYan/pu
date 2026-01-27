# Velocity vs Clean Prediction - 最终确认

## SimpleFold的实现

### 训练 (simplefold.py:395-420)

```python
# 1. 生成噪声和插值
noise = torch.randn_like(batch['coords'])  # x0
_, y_t, v_t = self.path.interpolant(t, noise, batch["coords"])
#                                       ^^^ x0      ^^^^^^^^^^^ x1

# path.interpolant返回:
# y_t = (1-t)*x0 + t*x1
# v_t = x1 - x0  (来自compute_ut)

# 2. 模型预测
out_dict = self.model(y_t, t, batch)

# 3. Loss
target = v_t  # velocity = x1 - x0
loss = MSE(out_dict['predict_velocity'], target)
```

**SimpleFold训练的target**: `v_t = x1 - x0` (velocity)

---

### 推理 (sampler.py:68-77)

```python
# 模型预测
velocity = model_fn(
    noised_pos=y,
    t=batched_t,
    feats=batch,
)['predict_velocity']  # ← 取velocity

# ODE step
dt = t_next - t
mean_y = y + velocity * dt  # ← 用velocity做Euler step
```

**SimpleFold推理使用**: `velocity`直接做ODE积分

**结论**: SimpleFold预测velocity，推理用velocity ✅

---

## 你的代码

### 模型返回 (flow_model.py:1084-1125)

```python
# 1. 模型预测velocity
speed_vectors, _ = self.NodeFeatExtractorWithHeads(node_embed, node_mask)

# 2. 计算clean prediction
t_factor = (1.0 - r3_t)[..., None, None]
side_atoms = xt_side + t_factor * speed_vectors  # x1 = xt + (1-t)*v

# 3. 返回
return {
    'speed_vectors': speed_vectors,  # ← velocity
    'side_atoms': side_atoms,        # ← clean prediction
    ...
}
```

---

### 训练 (flow_module.py:441-473)

```python
outs = self.model(noisy_batch)
speed_vectors = outs['speed_vectors']  # ← 取velocity

target_vectors = noisy_batch['v_t'][..., 3:, :]  # ← target是velocity

vector_loss = F.mse_loss(speed_vectors, target_vectors, ...)
```

**你的训练使用**: `speed_vectors` (velocity) ✅

---

### 推理 (interpolant.py:1611-1617)

```python
out = model(input_feats)
v_pred = out['side_atoms']  # ❌ 取了clean prediction！

dt = t2 - t1
xt = xt + dt * v_pred  # ❌ 用clean prediction做ODE step！
```

**你的推理使用**: `side_atoms` (clean prediction) ❌

---

## 数学分析

### 正确做法 (SimpleFold)

```python
v_pred = out['speed_vectors']  # v = x1 - x0
xt_new = xt + dt * v_pred      # Euler step
```

**ODE**: `dx/dt = v = x1 - x0`

**积分**: `x(t+dt) = x(t) + v*dt`

---

### 你的错误做法

```python
v_pred = out['side_atoms']  # = xt + (1-t)*v
xt_new = xt + dt * v_pred
       = xt + dt * (xt + (1-t)*v)
       = xt * (1 + dt) + dt*(1-t)*v
```

**问题**:
1. 多了`xt * dt`项 → 放大当前位置
2. velocity被缩放了`(1-t)`倍 → 方向错误

---

## 为什么你的代码没有完全崩溃？

### 数值分析

假设10步，dt=0.1，从t=0到t=1：

| Step | t | 正确: xt + v*dt | 错误: xt + (xt + (1-t)*v)*dt | 差异 |
|------|---|----------------|----------------------------|------|
| 1 | 0.0 | xt + 0.1*v | xt + 0.1*(xt + 1.0*v) = 1.1*xt + 0.1*v | +0.1*xt |
| 2 | 0.1 | xt + 0.1*v | xt + 0.1*(xt + 0.9*v) = 1.1*xt + 0.09*v | +0.1*xt, -0.01*v |
| ... | ... | ... | ... | ... |
| 10 | 0.9 | xt + 0.1*v | xt + 0.1*(xt + 0.1*v) = 1.1*xt + 0.01*v | +0.1*xt, -0.09*v |

**累积效应**:
- 每步都放大xt 1.1倍
- 后期velocity贡献越来越小
- 导致最终结果偏离真实轨迹

---

## 实验验证

### 方法1: 打印对比

```python
# 在推理循环中添加
out = model(input_feats)
v_correct = out['speed_vectors']
v_wrong = out['side_atoms']

print(f"t={t1:.3f}")
print(f"  velocity norm: {v_correct.norm().item():.4f}")
print(f"  side_atoms norm: {v_wrong.norm().item():.4f}")
print(f"  ratio: {(v_wrong.norm() / v_correct.norm()).item():.4f}")
```

**预期输出**:
```
t=0.000
  velocity norm: 2.5000
  side_atoms norm: 2.5000  # xt≈0, side_atoms≈v
  ratio: 1.0000

t=0.500
  velocity norm: 2.5000
  side_atoms norm: ~4.0    # side_atoms = xt + 0.5*v > v
  ratio: ~1.6

t=0.900
  velocity norm: 2.5000
  side_atoms norm: ~10.0   # side_atoms ≈ xt (主导)
  ratio: ~4.0
```

---

### 方法2: A/B测试

```python
# 用正确的velocity
results_correct = fbb_sample_iterative_fixed(batch, model, num_timesteps=10)

# 用错误的side_atoms (当前代码)
results_wrong = fbb_sample_iterative_current(batch, model, num_timesteps=10)

# 对比
print(f"Correct TM-score: {compute_tm(results_correct):.3f}")
print(f"Wrong TM-score: {compute_tm(results_wrong):.3f}")
```

**预期**:
- 正确版本应该提升5-10%

---

## 最终答案

### SimpleFold预测什么？

**预测**: `velocity = x1 - x0`

**推理时使用**: `velocity`直接做ODE step

---

### 你应该用什么？

**训练时**: `speed_vectors` (velocity) ✅ 正确

**推理时应该用**: `speed_vectors` (velocity)

**当前错误地用了**: `side_atoms` (clean prediction)

---

## 修复代码

### 修改1: interpolant.py:1612

```python
# 当前 (错误)
v_pred = out['side_atoms']

# 修复为
v_pred = out['speed_vectors']
```

### 修改2: interpolant.py:1716

```python
# 当前 (碰巧对了)
v_final = out_final['speed_vectors']  # ✅ 已经对了

# 保持不变
```

### 修改3: interpolant.py:1573

```python
# 当前 (错误)
ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps, device=device)

# 修复为
ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps + 1, device=device)
```

---

## 总结

| | SimpleFold | 你的训练 | 你的推理 | 是否一致 |
|---|-----------|---------|---------|---------|
| **预测** | velocity | velocity | ❌ clean prediction | ❌ 不一致 |
| **使用** | velocity | velocity | ❌ clean prediction | ❌ 不一致 |

**结论**: 你的推理代码确实用错了！应该用`speed_vectors`而不是`side_atoms`

---

**生成日期**: 2025-11-11
**确认**: SimpleFold预测velocity，你的代码应该也用velocity
