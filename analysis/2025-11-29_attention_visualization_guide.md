# IGA Attention 可视化使用指南

**日期**: 2025-11-29
**功能**: 可视化 InvariantGaussianAttention 的 attention 分布和几何bias

---

## 快速测试

```bash
cd /home/junyu/project/pu
python models/test_attention_vis.py
```

**输出**: `./test_attention_vis/` 目录下会生成3张图片（文件名前缀为 `layer0_`）

---

## 训练时启用可视化

### 方法1: 修改 IGA 初始化 (推荐)

在 `models/flow_model.py:594` 修改 IGA 初始化：

```python
self.trunk[f'iga_{b}'] = InvariantGaussianAttention(
    c_s=self._ipa_conf.c_s,
    c_hidden=self._ipa_conf.c_hidden,
    no_heads=self._ipa_conf.no_heads,
    no_qk_gaussians=self._ipa_conf.no_qk_points,
    no_v_points=self._ipa_conf.no_v_points,
    enable_vis=True,        # ← 启用可视化
    vis_interval=100,       # ← 每100步可视化一次
    vis_dir="./attention_vis",  # ← 保存目录
)
```

### 方法2: 通过 YAML 配置 (需要添加支持)

```yaml
# configs/model.yaml
model:
  ipa:
    enable_attention_vis: True
    attention_vis_interval: 100
    attention_vis_dir: "./attention_vis"
```

---

## 可视化输出

每次可视化会生成3张图片：

### 1. **layer{L}_attention_overview_b0_h0.png** (全景图，2×3布局)

**第一行 (计算流程)**：展示 attention 如何从标量 + 几何bias → 最终权重

| 子图 | 内容 | 用途 |
|-----|------|------|
| ① (0,0) | **Logits Before** (Scalar Q·K) | 只用标量attention，看基础的相似性 |
| ② (0,1) | **Geometric Bias** (γ × overlap) | 几何项，**负值=高overlap** |
| ③ (0,2) | **Logits After** (① + ②) | 加上几何bias后，softmax之前的logits |

**第二行 (分析对比)**：展示最终结果和真实结构的关系

| 子图 | 内容 | 用途 |
|-----|------|------|
| ④ (1,0) | **Final Attention Weights** | Softmax后的最终attention权重 |
| ⑤ (1,1) | **CA Distance Matrix** | 真实空间距离（ground truth） |
| ⑥ (1,2) | **Local Interaction Analysis** | **关键图**: attention vs distance scatter |

**如何解读**:
1. **① → ② → ③**: 看几何bias如何调整标量attention
   - 如果③和①差别不大 → γ太小，几何bias没起作用
   - 如果③的局部区域变亮 → 几何bias增强了局部相互作用 ✅
2. **④ vs ⑤**: 对比最终attention和真实距离
   - attention应该和距离呈反相关（近距离高attention）
3. **⑥**: 定量分析局部vs全局
   - 局部(<8Å)的点应该有更高的attention
   - 颜色(geo bias)应该显示近距离更负

---

### 2. **layer{L}_attention_profiles_b0_h0.png** (单个Query的Attention曲线)

**4个子图**: 显示4个不同位置的query residue对所有key的attention

**解读**:
- **绿色填充**: Attention权重（正值）
- **蓝色虚线**: Geometric bias（负值）
- **橙色区域**: 8Å局部邻域
- **红色虚线**: Query位置

**期望行为**:
- Attention应该在query附近有峰值（局部邻域）
- Geo bias应该在query附近更负（更强的overlap）

---

### 3. **layer{L}_distance_statistics_b0_h0.png** (统计分析，1×3布局)

| 子图 | 内容 | 关键问题 |
|-----|------|----------|
| (0) | Average Attention vs Distance | **Attention是否随距离衰减？** |
| (1) | Geometric Bias vs Distance | **Geo bias是否随距离变正？** |
| (2) | Local vs Global Attention Distribution | **局部邻域attention是否更高？** |

**健康的IGA应该显示**:
- ✅ Attention随距离快速衰减（8Å内最高）
- ✅ Geo bias随距离变正（远距离overlap小）
- ✅ Local attention mean >> Global attention mean

---

## 问题诊断

### 问题1: Attention 完全均匀（无局部性）

**症状**:
- Attention vs Distance: 平坦的线
- Local vs Global: 两个分布重叠

**可能原因**:
- γ (gamma) 太小 → 几何bias被忽略
- Geo bias范围太小 → 没有区分度

**解决**:
- 检查 `gamma` 值（应该在0.5-2.0之间）
- 检查 `geo_bias_std` (应该>0.1)

---

### 问题2: Attention 只关注对角线

**症状**:
- Attention权重矩阵只有对角线亮
- Distance scatter图显示只有0Å有attention

**可能原因**:
- Geometric bias过于惩罚远距离
- Softmax温度太低

**解决**:
- 减小 `gamma` 权重
- 检查 overlap score 计算

---

### 问题3: 负值过多（TwoSlopeNorm 错误）

**症状**:
```
vmin, vcenter, and vmax must be in ascending order
```

**已修复**: 当前代码会自动检测并使用单色colormap

**验证**:
- 如果geo_bias全是负值 → 用 `Blues_r` colormap
- 如果有正有负 → 用 `RdBu` diverging colormap

---

## 命令行打印输出

训练时每次可视化会打印（每个 layer 独立）：

```
[IGA Vis Layer 0] Step 100: γ=0.812, geo_bias=-2.341, local_attn=0.0234, global_attn=0.0089
[IGA Vis Layer 1] Step 100: γ=0.756, geo_bias=-2.108, local_attn=0.0221, global_attn=0.0095
...
[IGA Vis Layer 7] Step 100: γ=0.893, geo_bias=-2.512, local_attn=0.0248, global_attn=0.0082
```

**解读**:
- **γ (gamma)**: 可学习权重，控制几何bias的强度
- **geo_bias**: 平均几何bias（负值，绝对值越大=overlap越强）
- **local_attn**: 8Å内的平均attention
- **global_attn**: 8Å外的平均attention

**健康比例**: `local_attn / global_attn ≈ 2-5`

---

## 可视化频率控制

### 训练期间

```python
# models/IGA.py:184-186
enable_vis=True,        # 启用
vis_interval=100,       # 每100步
vis_dir="./attention_vis",
```

**建议**:
- 初期训练: `vis_interval=50` (观察attention如何形成)
- 稳定训练: `vis_interval=500` (减少开销)
- 调试: `vis_interval=10` (密集观察)

### 保存位置

默认: `./attention_vis/`

**文件命名** (每个 IGA layer 独立保存):
- `layer{L}_attention_overview_b{batch}_h{head}.png`
- `layer{L}_attention_profiles_b{batch}_h{head}.png`
- `layer{L}_distance_statistics_b{batch}_h{head}.png`

其中 `{L}` 是 layer index (0-7 for 8 layers)

---

## 性能影响

**CPU时间**: 每次可视化约2-5秒
**磁盘占用**: 每组图片约1-2MB
**GPU影响**: 无（数据已detach到CPU）

**建议**:
- 不要在每个step都可视化（太慢）
- 使用 `vis_interval >= 50`
- 定期清理旧图片

---

## 高级用法

### 可视化特定 head

默认只可视化 `head_idx=0`，如果想看其他head：

```python
# models/IGA.py:402
def _visualize_attention(self, ...):
    from models.visualize_attention import visualize_iga_attention

    # 可视化所有head
    for h in range(self.no_heads):
        stats = visualize_iga_attention(
            ...,
            head_idx=h,  # ← 遍历所有head
            ...
        )
```

### 可视化更多residues

```python
# models/IGA.py:404
num_vis_res=100  # 从50改到100
```

**注意**: residue太多会导致图片难以阅读

---

## 代码修复记录

**问题**: `TwoSlopeNorm` 要求 vmin < vcenter < vmax，但geo_bias全是负值时违反

**修复** (visualize_attention.py):
1. 检测数据范围
2. 如果全负值 → 使用 `Blues_r` 单色
3. 如果有正有负 → 使用 `RdBu` diverging
4. 添加NaN/Inf处理

**其他改进**:
- 边界检查（num_res < 2）
- 空数组处理（local_mask为空）
- 自动处理xlim/ylim的0值

---

## 总结

✅ **启用可视化**: `enable_vis=True` in IGA.__init__
✅ **调整频率**: `vis_interval=100` (推荐)
✅ **查看输出**: `./attention_vis/*.png`
✅ **诊断问题**: 看 distance_statistics 图
✅ **监控训练**: 看命令行打印的 local_attn/global_attn 比例

**关键问题**: Attention能否看到邻点相互作用？
- 看 `attention_vs_distance` scatter plot
- 看 `local_attn / global_attn` 比例
- 看 `geo_bias` 是否随距离变化
