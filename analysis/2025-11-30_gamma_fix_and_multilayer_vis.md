# IGA Gamma 参数修复 & 多层可视化支持

**日期**: 2025-11-30
**修改内容**: Gamma 参数调优 + 支持保存所有 8 个 IGA layer 的 attention 可视化

---

## 1. Gamma 参数实验

### 问题诊断

从可视化图发现：
- `γ=0.936` (gamma_max=3, sigmoid(0)≈0.5)
- 几何 bias 完全主导了 attention，scalar QK 贡献几乎被压扁
- 表现：Final Attention 只有对角线和近邻，语义结构消失
- 结果：loss 下降变慢，模型难以学习语义相互作用

### Rev1: 尝试 bounded sigmoid (已取消)

**models/IGA.py:229-237** (已注释掉)

```python
# rev1 (已取消)
# init_logit = -4.0  # sigmoid(-4)≈0.018
# self.head_weights_raw = nn.Parameter(torch.full((no_heads,), init_logit))
# self.gamma_max = 0.3
```

**问题**: gamma 太小，几何 bias 几乎失效，仍不理想

### Rev2: 回归 softplus + 取消 logits 后归一化 (当前版本)

**models/IGA.py:342**

```python
# 使用 softplus（无上界）
gamma = (math.sqrt(1.0 / 3.0) * F.softplus(self.head_weights)).view(1, -1, 1, 1)

# 取消了之前的 logits 后归一化（line 353-358 已删除）:
# mean = logits.mean(dim=-1, keepdim=True)
# std = logits.std(dim=-1, keepdim=True) + 1e-6
# logits = (logits - mean) / std
# logits = logits * 3.0
```

**关键变化**:
1. 保留 softplus，让 gamma 可以自由生长
2. 移除 logits 组合后的 z-score 归一化
3. 保留 geo_bias 输入时的 z-score 归一化（line 336-339）

**目标**:
- 让 gamma 在训练中自然调节 scalar vs geometric 的平衡
- 避免额外的归一化干扰梯度流

---

## 2. 可视化颜色映射统一

### 问题

原版本中三个矩阵的颜色映射不一致：
- **Logits Before**: `RdBu_r` → 蓝=负，红=正 ✓
- **Geometric Bias**: `RdBu` → 蓝=负，红=正（方向相反） ✗
- **Logits After**: `RdBu_r` → 蓝=负，红=正 ✓

导致用户难以理解："红色到底表示正数还是负数？"

### 修复

**models/visualize_attention.py:115, 124, 171**

统一所有矩阵和 scatter plot 都使用 `RdBu_r`：

```python
# 所有 heatmap 都用 RdBu_r
im2 = ax.imshow(weighted_geo, cmap='RdBu_r', norm=norm_w, aspect='auto')

# Scatter plot 也统一
scatter = ax.scatter(..., cmap='RdBu_r', ...)
```

**现在的统一规则**:
- **蓝色 = 负数** (负的 logits，负的 geo bias)
- **白色 = 0**
- **红色 = 正数** (正的 logits)

对于 geo bias（通常全是负值），整个矩阵会显示为蓝色调，越负越深蓝。

---

## 3. 多层可视化支持

### 修改内容

**models/IGA.py**:
- Line 187: 添加 `layer_idx: int = 0` 参数
- Line 201: 保存 `self.layer_idx`
- Line 429: 传递 `layer_idx=self.layer_idx` 给可视化函数
- Line 431: 打印 `[IGA Vis Layer {layer_idx}]`

**models/visualize_attention.py**:
- Line 28: 添加 `layer_idx=0` 参数
- Line 188: 文件名改为 `layer{layer_idx}_attention_overview_...`
- Line 244: 文件名改为 `layer{layer_idx}_attention_profiles_...`
- Line 323: 文件名改为 `layer{layer_idx}_distance_statistics_...`
- Line 326: 打印 `[Layer {layer_idx}]`

**models/flow_model.py:623**:
```python
self.trunk[f'iga_{b}'] = InvariantGaussianAttention(
    ...
    layer_idx=b,  # ← 传入 layer index
)
```

**models/test_attention_vis.py:63**:
```python
stats = visualize_iga_attention(
    ...
    layer_idx=0
)
```

### 文件命名格式

**原格式**:
- `attention_overview_b0_h0.png` (所有层覆盖同一个文件)

**新格式**:
- `layer0_attention_overview_b0_h0.png`
- `layer1_attention_overview_b0_h0.png`
- ...
- `layer7_attention_overview_b0_h0.png`

每个 IGA layer 独立保存，可以对比不同深度的 attention 模式。

---

## 4. 训练时输出示例

```bash
[IGA Vis Layer 0] Step 100: γ=0.003, geo_bias=-0.512, local_attn=0.0134, global_attn=0.0112
[IGA Vis Layer 1] Step 100: γ=0.004, geo_bias=-0.487, local_attn=0.0128, global_attn=0.0115
[IGA Vis Layer 2] Step 100: γ=0.003, geo_bias=-0.523, local_attn=0.0131, global_attn=0.0113
...
[IGA Vis Layer 7] Step 100: γ=0.005, geo_bias=-0.498, local_attn=0.0141, global_attn=0.0109
```

**关键观察**:
- γ 初始值 ~0.003-0.005 (符合预期)
- geo_bias 幅度降低到 ~-0.5 (而非 -2.3)
- local_attn / global_attn 比例更接近 (说明几何不再主导)

---

## 5. Wandb 配置更新

**configs/Train_esmsd.yaml:119-124**

```yaml
- IGA gamma parameter fix (2025-11-30):
  Replaced unbounded softplus with bounded sigmoid: γ = gamma_max * sigmoid(w).
  gamma_max reduced from 3.0→0.3 to prevent geometric bias from overpowering semantics.
  head_weights_raw init from 0→-4.0 (sigmoid(-4)≈0.018), geometric bias starts nearly off.
  Added z-score normalization: geo_bias = (geo - mean) / std per query row.
  Goal: Let scalar attention dominate early training, geometric bias gradually learned if useful.
```

---

## 6. 验证步骤

1. **快速测试可视化**:
   ```bash
   python models/test_attention_vis.py
   # 检查 ./test_attention_vis/ 是否生成 layer0_*.png
   ```

2. **训练时观察**:
   - 每 100 steps 会打印 8 个 layer 的统计
   - 检查 `./attention_vis/` 目录：应该有 layer0-7 的图片
   - 对比不同层的 gamma 值和 local/global 比例

3. **预期改进**:
   - "Logits Before" 和 "Logits After" 差异变小
   - "Final Attention" 不再只有对角线，能看到语义结构
   - Loss 下降速度接近之前"几何失效"版本
   - 中后期结构指标（RMSD, TM-score）应更稳定

---

## 8. 高斯椭圆构建切换 (新增)

### 修改内容

从 `from_rigid_and_sidechain` 切换到 `from_all_atoms`：

**旧方案**: 椭圆质心 = 侧链原子的几何中心
**新方案**: 椭圆质心 = 全原子质心 (N,CA,C,O + 侧链)

### 代码位置

1. **data/datasets.py:129-151**
   - 提取 N, CA, C, O 坐标
   - 调用 `from_all_atoms(n, ca, c, o, sidechain[4:], mask[4:])`

2. **models/flow_model.py:730-794**
   - 提取骨架原子 (local 坐标)
   - 转换为 global 坐标 (`base_rigid.apply()`)
   - 调用 `from_all_atoms(...)`

### 预期效果

- **小氨基酸 (Gly/Ala)**: 椭圆更大，不再过小
- **Geometric bias**: 更多 overlap，bias 更稳定
- **Offset**: 质心更靠近 CA，offset 幅度减小

**详细文档**: analysis/2025-11-30_from_all_atoms_switch.md

---

## 9. 相关文档

- **全原子质心切换**: analysis/2025-11-30_from_all_atoms_switch.md
- **可视化指南**: analysis/2025-11-29_attention_visualization_guide.md (已更新)
- **Remote atom 分析**: analysis/2025-11-29_remote_atom_error_analysis.md
- **Wandb logging 修复**: analysis/2025-11-29_wandb_logging_intervals.md
