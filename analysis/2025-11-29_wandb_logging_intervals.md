# Wandb Logging 间隔不一致问题分析

**日期**: 2025-11-29
**问题**: 某些指标每1步上传wandb，某些每10步上传

---

## 问题根源

PyTorch Lightning 的 `log_every_n_steps` 设置 **只对没有明确设置 `on_step=True` 的 log 调用起作用**。

### 关键代码位置

#### 1. **Config 设置** (configs/Train_esmsd.yaml:131)
```yaml
trainer:
  log_every_n_steps: 10  # ← 期望每10步记录一次
```

#### 2. **_log_scalar 默认参数** (models/flow_module.py:1385-1386)
```python
def _log_scalar(
    self, key, value,
    on_step=True,     # ← 默认值！会绕过 log_every_n_steps
    on_epoch=False,
    ...
):
```

#### 3. **Training Step 中的不同调用方式**

**方式A: 每1步记录** (line 1440-1442)
```python
# Stratified losses (没有条件判断)
for k, v in stratified_losses.items():
    self._log_scalar(
        f"train/{k}", v, prog_bar=False, batch_size=num_batch
    )  # ← 使用默认 on_step=True → 每1步都记录
```

**方式B: 每10步记录** (line 1491-1503)
```python
# FBB task losses (手动控制频率)
if self.global_step % 10 == 0:  # ← 手动条件判断
    for k, v in total_losses.items():
        self._log_scalar(
            f"train/{k}", float(v.detach().cpu().item()),
            prog_bar=True, batch_size=num_batch
        )  # ← 虽然 on_step=True，但外层有条件控制
```

---

## PyTorch Lightning 的 Logging 行为

### 规则
```python
self.log('metric', value, on_step=True)
# ↓
# 行为: 每个step都记录，忽略 log_every_n_steps 设置

self.log('metric', value, on_step=False)
# ↓
# 行为: 遵循 log_every_n_steps=10，每10步记录一次

self.log('metric', value)  # 默认 on_step=None
# ↓
# 行为: 在 training 中默认为 on_step=True
#       遵循 log_every_n_steps（但只对未显式设置的有效）
```

### 当前代码的实际行为

| 指标类型 | 代码位置 | on_step | 条件判断 | 实际频率 |
|---------|---------|---------|---------|---------|
| stratified losses (atom_mse_t0.2, etc.) | 1440-1442 | `True` (默认) | ❌ 无 | **每1步** |
| total losses (loss, coord_loss, etc.) | 1491-1503 | `True` (默认) | ✅ `global_step % 10 == 0` | **每10步** |
| validation losses | 1187-1213 | `False` | ✅ `on_epoch=True` | **每epoch** |

---

## 解决方案

### 方案1: **修改 _log_scalar 默认参数** (推荐)

**目标**: 让所有 log 默认遵循 `log_every_n_steps`

```python
# models/flow_module.py:1385-1386
def _log_scalar(
    self, key, value,
    on_step=None,     # ← 改成 None，让 Lightning 自动决定
    on_epoch=False,
    ...
):
```

**效果**:
- Lightning 会根据 `log_every_n_steps=10` 自动控制频率
- 所有使用 `_log_scalar` 的地方都统一为每10步记录
- **但需要移除手动的 `if self.global_step % 10 == 0` 判断**（否则重复控制）

---

### 方案2: **手动添加条件判断** (当前部分实现)

**目标**: 在每个 log 调用前添加 `if self.global_step % 10 == 0`

```python
# models/flow_module.py:1440-1442
if self.global_step % 10 == 0:  # ← 添加条件
    for k, v in stratified_losses.items():
        self._log_scalar(
            f"train/{k}", v, prog_bar=False, batch_size=num_batch
        )
```

**优点**:
- 精确控制每个指标的频率
- 不依赖 Lightning 的默认行为

**缺点**:
- 需要在每个地方手动添加
- 代码冗余

---

### 方案3: **创建两个 log 函数** (最灵活)

```python
# models/flow_module.py

def _log_scalar_frequent(self, key, value, **kwargs):
    """每step都记录（on_step=True）"""
    self.log(key, value, on_step=True, on_epoch=False, **kwargs)

def _log_scalar_sparse(self, key, value, **kwargs):
    """遵循 log_every_n_steps（on_step=None）"""
    self.log(key, value, on_step=None, on_epoch=False, **kwargs)

# 使用
self._log_scalar_sparse(f"train/{k}", v)  # 每10步
self._log_scalar_frequent(f"debug/{k}", v)  # 每1步
```

---

## 推荐修复步骤

### 步骤1: 修改 _log_scalar 默认参数

```python
# models/flow_module.py:1381-1403
def _log_scalar(
    self,
    key,
    value,
    on_step=None,      # 从 True 改成 None
    on_epoch=False,
    prog_bar=True,
    batch_size=None,
    sync_dist=False,
    rank_zero_only=True
):
    if sync_dist and rank_zero_only:
        raise ValueError('Unable to sync dist when rank_zero_only=True')
    self.log(
        key,
        value,
        on_step=on_step,
        on_epoch=on_epoch,
        prog_bar=prog_bar,
        batch_size=batch_size,
        sync_dist=sync_dist,
        rank_zero_only=rank_zero_only
    )
```

### 步骤2: 移除手动的频率控制

```python
# models/flow_module.py:1488-1503
# 移除外层的 if self.global_step % 10 == 0:

elif self._exp_cfg.task in ('SHdecode', 'shfbb', 'fbb', 'sh_to_atoms', 'shdiffusion'):
    train_loss = total_losses['loss']

    # 直接记录，让 Lightning 控制频率
    if 'speed_loss' in total_losses:
        self._log_scalar(
            "train/speed_loss",
            total_losses['speed_loss'],
            prog_bar=True,
            batch_size=num_batch,
        )
    for k, v in total_losses.items():
        if k == 'speed_loss':
            continue
        self._log_scalar(
            f"train/{k}", float(v.detach().cpu().item()),
            prog_bar=True, batch_size=num_batch
        )
```

### 步骤3: 验证行为

训练后检查 wandb：
- 所有 `train/*` 指标应该每10步记录一次
- 所有 `valid/*` 指标应该每epoch记录一次

---

## 特殊情况处理

### 如果确实需要某些指标每步都记录

```python
# 显式指定 on_step=True
self._log_scalar(
    "debug/gradient_norm",
    grad_norm,
    on_step=True,    # ← 明确指定每步记录
    on_epoch=False,
    prog_bar=False
)
```

### 如果需要同时记录 step 和 epoch

```python
# PyTorch Lightning 支持同时记录
self.log(
    "train/loss",
    loss,
    on_step=True,   # wandb 上显示为 "train/loss"
    on_epoch=True,  # wandb 上显示为 "train/loss_epoch"
    prog_bar=True
)
```

---

## 总结

**问题原因**:
- `_log_scalar` 默认 `on_step=True`，绕过了 `log_every_n_steps=10`
- 部分代码手动用 `if self.global_step % 10 == 0` 控制频率，部分没有

**解决方案**:
1. ✅ **改 `on_step=None`** - 统一遵循 `log_every_n_steps`
2. ✅ **移除手动频率控制** - 避免重复逻辑

**效果**:
- 所有训练指标统一每10步记录一次
- Wandb 更清爽，不会因为过于频繁的记录而卡顿
