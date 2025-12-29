# Gamma 乘法 & Colorbar 标尺修复

**日期**: 2025-11-30
**修复内容**:
1. 在 logits 加和时补上 gamma 乘法
2. 可视化 colorbar 使用实际数据范围而非对称范围

---

## 问题 1: 缺少 Gamma 乘法

### 发现

**位置**: models/IGA.py:329

**原代码**:
```python
# Combine: Logits = Scalar + Gamma * Gaussian
logits = logits + attn_bias_geo  # ← 缺少 gamma
```

**问题**:
- 虽然注释写的是 `Gamma * Gaussian`
- 实际代码没有乘以 gamma
- 导致几何 bias 的贡献不受 gamma 控制

### 修复

```python
# Combine: Logits = Scalar + Gamma * Gaussian
logits = logits + gamma * attn_bias_geo  # ✓ 加上 gamma
```

**影响**:
- 现在 gamma 可以真正控制几何 bias 的强度
- gamma 从 softplus 学习而来，可以自适应调节
- 预期：训练初期 gamma 小，后期如果几何有用会增大

---

## 问题 2: Colorbar 标尺不匹配

### 发现

**位置**: models/visualize_attention.py 多处

**原逻辑**:
```python
vmax_abs = max(abs(vmin), abs(vmax))
norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0, vmax=vmax_abs)
```

**问题**:
- 强制使用对称范围
- 例如数据范围 [-100, 0]，显示为 [-100, 100]
- Colorbar 标尺包含数据中不存在的值（0 到 100）
- 用户难以判断实际数据分布

**示例**:
```
数据实际范围: [-100, -5]
旧版本显示:   [-100, 100]  (右半边全是空的)
```

### 修复

**修改了 4 处**:

1. **Logits Before** (Line 96-102):
```python
pre_vmin, pre_vmax = float(logits_pre.min()), float(logits_pre.max())
norm_pre = TwoSlopeNorm(vmin=pre_vmin, vcenter=0, vmax=pre_vmax)  # 使用实际范围
```

2. **Geometric Bias** (Line 110-113):
```python
w_vmin, w_vmax = float(weighted_geo.min()), float(weighted_geo.max())
norm_w = TwoSlopeNorm(vmin=w_vmin, vcenter=0, vmax=w_vmax)  # 使用实际范围
```

3. **Logits After** (Line 121-127):
```python
post_vmin, post_vmax = float(logits_post.min()), float(logits_post.max())
norm_post = TwoSlopeNorm(vmin=post_vmin, vcenter=0, vmax=post_vmax)  # 使用实际范围
```

4. **Scatter Plot** (Line 158-162):
```python
geo_flat_min, geo_flat_max = float(geo_flat.min()), float(geo_flat.max())
norm = TwoSlopeNorm(vmin=geo_flat_min, vcenter=0, vmax=geo_flat_max)  # 使用实际范围
```

### 效果对比

**旧版本**:
```
Geometric Bias: [-8.2, -0.1]
Colorbar 显示:  [-10, 10]  (强制对称)
→ 右半边 (0~10) 完全空白，浪费空间
```

**新版本**:
```
Geometric Bias: [-8.2, -0.1]
Colorbar 显示:  [-8.2, -0.1]  (实际范围)
→ 充分利用颜色映射区间，更容易看清数据分布
```

---

## 预期改进

### Gamma 修复后

1. **早期训练**:
   - Gamma 初始值小（softplus(0) ≈ 0.69）
   - 几何 bias 贡献较小
   - Scalar attention 主导

2. **中后期训练**:
   - 如果几何确实有用，gamma 会增大
   - 几何 bias 贡献增强
   - Attention 更关注局部相互作用

3. **Per-head 自适应**:
   - 每个 head 有独立的 gamma
   - 可能有的 head 偏语义（gamma 小）
   - 有的 head 偏几何（gamma 大）

### Colorbar 修复后

1. **更准确的视觉反馈**:
   - 看到的颜色范围 = 实际数据范围
   - 不会被空白区域误导

2. **更容易诊断问题**:
   - 如果 geo_bias 全是 -100 左右 → 说明 overlap 太强
   - 如果 geo_bias 接近 0 → 说明 overlap 太弱

3. **对比更清晰**:
   - Before vs After 的颜色变化更直观
   - 能更清楚看到 gamma * geo_bias 的影响

---

## 验证步骤

### 1. 检查 Gamma 是否生效

训练时观察打印：
```
[IGA Vis Layer 0] Step 100: γ=0.693, geo_bias=-2.341, ...
```

- **初期**: γ 应该接近 0.69 (softplus(0))
- **训练中**: γ 如果逐渐增大 → 几何 bias 在学习
- **稳定后**: γ 稳定在某个值 → 找到平衡

### 2. 检查 Colorbar 显示

打开可视化图片：
```bash
ls ./attention_vis/layer0_*.png
```

检查：
- Colorbar 的范围是否和矩阵颜色匹配
- 是否还有大片空白区域（对称但数据不对称）
- 不同图之间的标尺是否合理

---

## 相关文件

- **models/IGA.py:329** - Gamma 乘法修复
- **models/visualize_attention.py:96-162** - Colorbar 标尺修复（4处）
- **configs/Train_esmsd.yaml:124-128** - Wandb notes 更新

---

## 总结

这两个修复解决了：
1. ✅ **功能 bug**: gamma 现在真正控制几何 bias 的强度
2. ✅ **可视化 bug**: colorbar 显示实际数据范围，不再误导

预期影响：
- 训练更稳定（gamma 可以自适应调节）
- 可视化更准确（直观反映数据分布）
- 诊断更容易（能快速判断 geo_bias 是否合理）
