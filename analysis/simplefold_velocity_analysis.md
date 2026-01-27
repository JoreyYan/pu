# SimpleFold预测的是 `x1 - x0` (Velocity) 还是 `x1 - xt`？

**答案**: SimpleFold预测的是 **`v_t = dx/dt`**，这在Linear Flow的情况下**等价于 `x1 - x0`**

---

## 详细分析

### 1. SimpleFold的Linear Path定义

**文件**: `ml-simplefold/src/simplefold/model/flow.py:81-102`

```python
class LinearPath(BasePath):
    """
    Linear flow process:
    x0: noise, x1: data
    In inference, we sample data from 0 -> 1
    """

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return t, 1  # alpha(t) = t, d_alpha/dt = 1

    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return 1 - t, -1  # sigma(t) = 1-t, d_sigma/dt = -1
```

**插值路径**:
```
x_t = alpha(t) * x1 + sigma(t) * x0
    = t * x1 + (1-t) * x0
```

**Velocity**:
```python
def compute_ut(self, t, x0, x1):
    """Compute the vector field corresponding to p_t"""
    _, d_alpha_t = self.compute_alpha_t(t)  # d_alpha/dt = 1
    _, d_sigma_t = self.compute_sigma_t(t)  # d_sigma/dt = -1
    return d_alpha_t * x1 + d_sigma_t * x0  # 1*x1 + (-1)*x0 = x1 - x0
```

**结论1**: SimpleFold定义的velocity `v_t = x1 - x0` (与时间`t`无关，是常数！)

---

### 2. SimpleFold的训练代码

**文件**: `ml-simplefold/src/simplefold/model/simplefold.py:381-420`

```python
def flow_matching_train_step(self, batch, batch_idx):
    # 1. 采样时间t
    t = 0.98 * logit_normal_sample(...) + 0.02 * torch.rand(...)

    # 2. 生成噪声x0
    noise = torch.randn_like(batch['coords'])  # x0

    # 3. 计算interpolant: x_t和真实velocity v_t
    _, y_t, v_t = self.path.interpolant(t, noise, batch["coords"])
    #                                         ^^^^^ x0    ^^^^^^^^^^^^^ x1
    #   返回: (t, x_t, v_t)
    #   其中: x_t = t*x1 + (1-t)*x0
    #        v_t = x1 - x0  (来自compute_ut)

    # 4. 模型预测
    out_dict = self.model(y_t, t, batch)  # 输入x_t, 预测velocity

    # 5. Loss (with Rigid Alignment)
    if self.use_rigid_align:
        # 用预测的velocity计算denoised coords
        v_t = out_dict['predict_velocity']
        denoised_coords = y_t + v_t * (1.0 - t)

        # 对GT做rigid alignment
        coords_aligned = weighted_rigid_align(batch["coords"], denoised_coords, ...)

        # 重新计算aligned的velocity target
        _, _, v_t_aligned = self.path.interpolant(t, noise, coords_aligned)
        target = v_t_aligned  # v_t_aligned = coords_aligned - noise
    else:
        target = v_t  # v_t = x1 - x0

    # 6. MSE Loss
    loss = F.mse_loss(out_dict['predict_velocity'], target)
```

**关键**:
- Target是`v_t = x1 - x0`
- 模型输入`x_t = t*x1 + (1-t)*x0`，预测`v = x1 - x0`

**结论2**: SimpleFold训练时的target是 `x1 - x0`

---

### 3. SimpleFold的采样代码

**文件**: `ml-simplefold/src/simplefold/model/torch/sampler.py:83-108`

```python
@torch.no_grad()
def sample(self, model_fn, flow, noise, batch):
    steps = torch.linspace(t_start, 1.0, num_timesteps + 1)  # ✅ 正确的时间点数量
    y_sampled = noise  # 初始化为x0

    for i in range(num_timesteps):
        t = steps[i]
        t_next = steps[i + 1]

        y_sampled = self.euler_maruyama_step(
            model_fn, flow, y_sampled, t, t_next, batch
        )

    return {"denoised_coords": y_sampled}
```

**Euler-Maruyama步骤** (Line 48-80):
```python
def euler_maruyama_step(self, model_fn, flow, y, t, t_next, batch):
    dt = t_next - t

    # 模型预测velocity
    velocity = model_fn(noised_pos=y, t=t, feats=batch)['predict_velocity']

    # 从velocity计算score (用于SDE)
    score = flow.compute_score_from_velocity(velocity, y, t)

    # Drift term (ODE + diffusion correction)
    diff_coeff = self.diffusion_coefficient(t)
    drift = velocity + diff_coeff * score

    # Euler-Maruyama step
    mean_y = y + drift * dt  # 确定性部分
    y_sample = mean_y + sqrt(2*dt*diff_coeff*tau) * noise  # 随机部分

    return y_sample
```

**关键**: `y_new = y + velocity * dt`

**但这里有个问题**: 如果`velocity = x1 - x0`（常数），那么ODE积分应该是什么？

---

## 数学推导：为什么 SimpleFold 的采样是对的？

### Linear Flow的ODE

给定：
```
x_t = t*x1 + (1-t)*x0
```

求导：
```
dx_t/dt = x1 - x0 = v  (常数velocity!)
```

**ODE积分**（从t到t+dt）:
```
x_{t+dt} = x_t + ∫[t, t+dt] v ds
         = x_t + v * dt
         = x_t + (x1 - x0) * dt
```

**这就是SimpleFold采样代码中的**:
```python
y_new = y + velocity * dt
```

其中`velocity`是模型预测的`x1 - x0`。

---

### 为什么看起来"违反直觉"？

**直觉错误**: "velocity应该是`(x1 - x_t) / (1 - t)`，因为从`x_t`到`x1`还需要时间`1-t`"

**实际情况**: Flow Matching的velocity是**tangent vector field** `v_t(x_t)`，不是"到终点的方向"！

**类比物理**:
- 想象一个粒子沿着轨迹`x_t = t*x1 + (1-t)*x0`运动
- 它的速度`dx/dt = x1 - x0`是**恒定的**（匀速直线运动）
- 不管粒子在哪（x_t），速度都是`x1 - x0`

---

## 对比你的代码

### 你的代码中的velocity

**你的训练代码** (假设你也用Linear Flow):
```python
# 训练时
x_t = t*x1 + (1-t)*x0
v_target = x1 - x0  # 真实velocity
v_pred = model(x_t, t)  # 模型预测
loss = MSE(v_pred, v_target)
```

**你的采样代码** (`fbb_sample_iterative`):
```python
# Line 1614-1618
# Standard Euler ODE step for v = x1 - x0
# x_t = (1-t)*x0 + t*x1, so dx/dt = x1 - x0 = v
dt = t2 - t1
xt = xt + dt * v_pred  # ✅ 正确！
```

**结论**: 你的ODE积分逻辑是**正确的**！

---

### 但是！你的Final step有问题

**你的Final step** (Line 1718):
```python
clean_final = xt + (1.0 - t_final) * v_final
```

这个公式假设`v = (x1 - x_t) / (1 - t)`，但实际上：

**如果v是`x1 - x0`** (SimpleFold的定义):
```
x1 = x_t + (1-t) * (x1 - x0)  ❌ 这是错的！

正确的推导:
x_t = t*x1 + (1-t)*x0
x1 = (x_t - (1-t)*x0) / t

但我们不知道x0，所以从velocity恢复x1:
v = x1 - x0
=> x0 = x1 - v

代入:
x_t = t*x1 + (1-t)*(x1 - v)
x_t = t*x1 + (1-t)*x1 - (1-t)*v
x_t = x1 - (1-t)*v
=> x1 = x_t + (1-t)*v  ✅ 这才是对的！
```

**所以你的Final step公式居然是对的！** 只要确保`v_final`是模型预测的`x1 - x0`。

---

## 总结：SimpleFold vs 你的代码

| 项目 | SimpleFold | 你的代码 | 是否一致 |
|------|-----------|---------|---------|
| **Velocity定义** | `v = x1 - x0` | `v = x1 - x0` (假设) | ✅ 一致 |
| **Interpolant** | `x_t = t*x1 + (1-t)*x0` | `x_t = t*x1 + (1-t)*x0` | ✅ 一致 |
| **ODE Step** | `x_new = x + v*dt` | `xt = xt + dt*v_pred` | ✅ 一致 |
| **Final Pred** | (在loop中完成) | `x1 = x_t + (1-t)*v` | ✅ 正确 |
| **时间点数量** | `num_steps + 1` | `num_steps` ❌ | ❌ **你的bug** |
| **Velocity key** | `predict_velocity` | `side_atoms` / `speed_vectors` 不一致 | ❌ **你的bug** |

---

## 关键洞察

### 1. **Linear Flow的velocity是常数**

```
v_t(x_t, t) = x1 - x0  (与t无关！)
```

这意味着：
- 不管在哪个时间点`t`
- 不管当前位置`x_t`是什么
- Velocity永远是`x1 - x0`

**物理直觉**: 匀速直线运动，速度恒定。

---

### 2. **为什么Final step公式看起来不同？**

**SimpleFold采样**: 在loop中一步步更新到`t=1.0`，最后一步后`x_final ≈ x1`

**你的代码**:
- Loop到`t=t_final`（应该是1.0但你bug成0.99）
- Final step: `x1 = x_t + (1-t)*v`

两种方式在`t=1.0`时**数学上等价**:
```
当 t=1.0:
x_1.0 = x_0.99 + dt * v  (SimpleFold)
x_1 = x_0.99 + (1-0.99) * v = x_0.99 + 0.01*v  (你的公式)

如果dt = 0.01，两者完全一样！
```

**但你的bug导致**:
```
t_final = 0.99 (因为时间点少一个)
x1 = x_0.99 + (1-0.99)*v = x_0.99 + 0.01*v  ← 只前进了0.01的距离
```

**修复后**:
```
t_final = 1.0
x1 = x_1.0 + (1-1.0)*v = x_1.0  ← 不需要额外校正！
```

所以修复后，Final step的velocity校正项变成0，这是**正确的**。

---

## 你需要确认的问题

### ❓ 你的模型到底预测什么？

请检查你的训练代码中的target：

**Option 1**: Velocity (与SimpleFold一样)
```python
# 训练时
x_t = t*x1 + (1-t)*x0
v_target = x1 - x0
loss = MSE(model(x_t, t), v_target)

# 采样时
xt = xt + dt * v_pred  # ✅ 正确
```

**Option 2**: Denoised coordinates (x1直接预测)
```python
# 训练时
x_t = t*x1 + (1-t)*x0
x1_target = x1
loss = MSE(model(x_t, t), x1_target)

# 采样时
x1_pred = model(x_t, t)
v = (x1_pred - x_t) / (1 - t)  # 计算implicit velocity
xt = xt + dt * v  # ✅ 也对
```

**如何确认**:
1. 检查你的training loss计算
2. 看target是`x1`还是`x1 - x0`
3. 如果target shape = input shape，可能是预测x1
4. 如果训练时有`v = x1 - x0`的计算，那就是velocity

---

## 最终答案

**SimpleFold预测的是**: `v = x1 - x0` (Constant Velocity)

**你的代码应该**:
1. 确认你的模型也预测`v = x1 - x0`
2. 如果是，你的ODE step是对的
3. 修复时间点数量bug (num_timesteps → num_timesteps + 1)
4. 修复velocity key不一致bug
5. Final step公式已经是对的

**修复后效果**:
- 做完整的10步采样（而不是9步）
- 最终时间是1.0（而不是0.99）
- Final step的校正项变成0（因为(1-1.0)=0）

---

**生成日期**: 2025-11-11
**结论**: SimpleFold和你的代码都预测`x1 - x0`，ODE积分逻辑正确，只需修复2个bug即可
