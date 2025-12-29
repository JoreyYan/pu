# 修复 SH 信息泄漏的方案

## 问题诊断

### 当前泄漏
在训练和推理时，`sh_density_from_atom14_with_masks_clean` 使用了 GT 信息：
1. `atom14_element_idx` - GT的元素类型
2. `atom14_gt_exists` - GT的原子存在mask

这导致模型即使在噪声坐标上也能通过SH密度的通道模式推断氨基酸类型。

## 解决方案

### 方案 1：不区分元素类型（最简单）

所有原子都当作同一种元素（如C）来计算SH密度：

```python
# 推理时
element_idx_generic = torch.zeros_like(batch['atom14_element_idx'])  # 全0 = 全C
atom_exists_predicted = torch.ones_like(batch['atom14_gt_exists'])   # 假设所有14个原子都可能存在

normalize_density, *_ = sh_density_from_atom14_with_masks_clean(
    input_feats['atoms14_local_t'],
    element_idx_generic,           # 通用元素
    atom_exists_predicted,         # 通用mask
    L_max=8,
    R_bins=24,
)
```

**优点**：
- 完全消除泄漏
- 实现简单
- SH只编码几何信息，不编码化学信息

**缺点**：
- 丢失了元素类型信息
- 可能降低性能

### 方案 2：从预测的aatype获取元素信息

使用模型自己预测的序列来获取元素类型：

```python
# 推理时需要先预测aatype
# 可以用一个轻量级的序列预测头（基于backbone几何）
aatype_pred = self.sequence_predictor(backbone_features)  # 新增模块

# 从预测的aatype构建元素索引
element_idx_pred = get_element_idx_from_aatype(aatype_pred)
atom_exists_pred = get_atom_exists_from_aatype(aatype_pred)

normalize_density, *_ = sh_density_from_atom14_with_masks_clean(
    input_feats['atoms14_local_t'],
    element_idx_pred,              # 从预测获得
    atom_exists_pred,              # 从预测获得
    L_max=8,
    R_bins=24,
)
```

**优点**：
- 保留元素信息的益处
- 模型学会自己推断化学类型

**缺点**：
- 需要额外的序列预测模块
- 训练复杂度增加

### 方案 3：混合策略（推荐）

训练时逐渐减少泄漏，推理时完全不泄漏：

```python
# 训练时
if self.training:
    # 以概率 p 使用GT元素，概率 (1-p) 使用通用元素
    p_use_gt = max(0.0, 1.0 - self.current_epoch / self.total_epochs)
    if random.random() < p_use_gt:
        element_idx = batch['atom14_element_idx']
        atom_exists = batch['atom14_gt_exists']
    else:
        element_idx = torch.zeros_like(batch['atom14_element_idx'])
        atom_exists = torch.ones_like(batch['atom14_gt_exists'])
else:
    # 推理时永远用通用元素
    element_idx = torch.zeros_like(batch['atom14_element_idx'])
    atom_exists = torch.ones_like(batch['atom14_gt_exists'])
```

**优点**：
- 训练早期有GT帮助，容易收敛
- 训练后期学会不依赖GT
- 推理时无泄漏

## 验证实验

### 实验1：对比泄漏 vs 不泄漏
```bash
# A. 当前方法（有泄漏）
python run.py --use_gt_elements=True

# B. 不区分元素（无泄漏）
python run.py --use_gt_elements=False
```

预期结果：
- Recovery从88.5%降到60-70%（合理范围）
- TM-score可能略微下降或保持不变
- Perplexity可能上升到3-5

### 实验2：分析SH通道的信息量
计算不同氨基酸的SH密度的可区分性：

```python
# 对每种氨基酸，计算其典型SH密度
sh_signatures = {}
for aa in amino_acids:
    coords = sample_coords_from_rotamer_library(aa)
    element_idx_gt = get_element_idx(aa)
    element_idx_generic = zeros_like(element_idx_gt)

    sh_with_gt = compute_sh(coords, element_idx_gt)
    sh_generic = compute_sh(coords, element_idx_generic)

    sh_signatures[aa] = {
        'with_gt': sh_with_gt,
        'generic': sh_generic
    }

# 计算氨基酸之间的SH相似度矩阵
similarity_with_gt = compute_pairwise_similarity(sh_signatures, 'with_gt')
similarity_generic = compute_pairwise_similarity(sh_signatures, 'generic')

print(f"平均区分度（有GT）: {similarity_with_gt.mean()}")
print(f"平均区分度（无GT）: {similarity_generic.mean()}")
```

预期：有GT时相似度很低（易区分），无GT时相似度更高（难区分）

## 修改文件清单

1. `/home/junyu/project/pu/data/interpolant.py`
   - `fbb_sample_iterative` (line ~1150)
   - `fbb_sample_iterative_sde` (line ~1286)

2. `/home/junyu/project/pu/models/flow_module.py`
   - `model_step_shfbb` (line ~422)

3. 可能还需要修改：
   - `data/sh_density.py` - 添加通用元素选项
   - 配置文件 - 添加开关控制

## 建议行动

**立即执行**：
1. 实现方案1（不区分元素），快速验证是否是泄漏问题
2. 运行小规模实验（10个样本）
3. 对比有泄漏 vs 无泄漏的recovery

**如果recovery确实下降**：
- 证明了泄漏存在
- 可以接受60-70%的recovery（这是正常水平）
- MPNN的结果应该仍然是最好的

**后续优化**：
- 如果性能下降太多，尝试方案2或方案3
- 研究是否可以从backbone几何预测序列

## 为什么MPNN不受影响

MPNN是逆向折叠：
- 输入：完美的GT backbone坐标
- 任务：预测序列
- 它本来就应该看到完整的结构信息

您的方法是端到端预测：
- 输入：噪声
- 任务：预测结构+序列
- 不应该看到GT的化学信息

## 预期结果修正

修复后的合理结果应该是：

| 方法 | TM | pLDDT | Recovery | 解读 |
|------|-----|-------|----------|------|
| MPNN | 0.709 | 79.5 | - | 逆向折叠SOTA（有GT结构） |
| SH SDE 1000（修复后） | 0.68 | 76.8 | **60-70%** | 端到端预测SOTA |
| R3 ODE 10 | 0.644 | 71.1 | 68% | 快速方法 |

60-70%的recovery是合理的，因为：
1. 同一个结构可以有多个合理序列
2. 没有GT化学信息的帮助
3. 完全依靠学到的序列-结构关系

这才是真实的性能！
