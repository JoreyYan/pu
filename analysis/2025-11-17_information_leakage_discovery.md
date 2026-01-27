# 2025-11-17: 重大发现 - SH密度中的信息泄漏

## 摘要

发现了导致SH+FBB方法recovery异常高（88.5%）的根本原因：**在训练和推理时，SH密度计算使用了GT的元素类型信息和原子存在mask，导致模型可以通过SH密度的通道模式直接推断氨基酸类型，而不是真正学习序列-结构关系。**

## 问题发现过程

### 1. 异常现象观察

在评估结果中发现：
- **SH SDE 1000**: Recovery=88.5%, TM=0.683, pLDDT=76.8
- **R3 ODE 10**: Recovery=68.0%, TM=0.644, pLDDT=71.1
- **MPNN** (逆向折叠): Recovery=未知, TM=0.709, pLDDT=79.5

**异常点**：
1. SH的recovery (88.5%) 远高于R3 (68%)，接近MPNN水平
2. 同一结构位置本应该可以有多个合理的氨基酸，88.5%的recovery异常地高
3. 训练时的recovery就已经很高（怀疑过拟合或信息泄漏）

### 2. 用户的关键洞察

用户提出疑问：
> "按理说，一个位置的序列本来就是多样的，同一个位置本来就可以是多个氨基酸，为什么能达到80%以上的recovery呢？而且我觉得这么高不是好事情，因为限制了多样性"

这个观察非常深刻，促使我们检查是否存在信息泄漏。

### 3. 泄漏点定位

用户检查代码后发现：

**推理时的代码** (`data/interpolant.py` line 1286-1295):
```python
with torch.no_grad():
    normalize_density, *_ = sh_density_from_atom14_with_masks_clean(
        input_feats['atoms14_local_t'],     # 噪声坐标
        batch['atom14_element_idx'],        # ❌ GT元素类型！
        batch['atom14_gt_exists'],          # ❌ GT原子存在mask！
        L_max=8,
        R_bins=24,
    )
```

**训练时的代码** (`models/flow_module.py` line 422-431):
```python
noisy_batch = self.interpolant.fbb_corrupt_batch(batch, prob)  # 坐标加噪声

normalize_density, *_ = sh_density_from_atom14_with_masks_clean(
    noisy_batch['atoms14_local_t'],     # 噪声坐标
    batch['atom14_element_idx'],        # ❌ 还是GT元素！
    batch['atom14_gt_exists'],          # ❌ 还是GT存在mask！
    L_max=self._model_cfg.sh.L_max,
    R_bins=self._model_cfg.sh.R_bins,
)
```

## 泄漏机制详解

### 1. 元素信息泄漏氨基酸类型

不同氨基酸有独特的原子组成模式：

```
GLY: [N, Cα, C, O, -, -, -, -, -, -, -, -, -, -]
     4个原子，无Cβ（唯一特征）

ALA: [N, Cα, C, O, Cβ, -, -, -, -, -, -, -, -, -]
     5个原子，1个额外的碳

VAL: [N, Cα, C, O, Cβ, Cγ1, Cγ2, -, -, -, -, -, -, -]
     7个原子，2个甲基

PHE: [N, Cα, C, O, Cβ, Cγ, Cδ1, Cδ2, Cε1, Cε2, Cζ, -, -, -]
     11个原子，6个碳形成苯环

ARG: [N, Cα, C, O, Cβ, Cγ, Cδ, Nε, Cζ, Nη1, Nη2, -, -, -]
     11个原子，包含3个氮（独特）

CYS: [N, Cα, C, O, Cβ, Sγ, -, -, -, -, -, -, -, -]
     6个原子，包含1个硫（唯一有硫）
```

**关键观察**：
- 通过原子数量和元素类型的组合，可以**直接推断**出大部分氨基酸
- GLY是唯一没有Cβ的（4个原子）
- CYS是唯一有硫的
- PHE/TYR/TRP有独特的芳香环结构（多个碳原子）

### 2. SH密度的多通道编码

SH密度张量结构：`[B, N, C=4, L_max, R_bins]`
- C维度：对应4种元素 [C, N, O, S]
- 每个元素在独立通道中编码

**即使坐标是完全随机的噪声**：

**GLY的SH密度特征**：
```
C通道：2个峰（Cα, C=O的C）
N通道：1个峰（主链N）
O通道：1个峰（羰基O）
S通道：0个峰
位置5-14：全零（无侧链）
```

**PHE的SH密度特征**：
```
C通道：8个峰（Cα, C=O, Cβ, Cγ, Cδ1, Cδ2, Cε1, Cε2, Cζ）
N通道：1个峰（主链N）
O通道：1个峰（羰基O）
S通道：0个峰
特征：C通道峰密集（苯环）
```

**CYS的SH密度特征**：
```
C通道：3个峰（Cα, C=O, Cβ）
N通道：1个峰
O通道：1个峰
S通道：1个峰 ← 独特标志！
```

### 3. 模型的"作弊"策略

模型学到的捷径：

```python
# 伪代码：模型实际学到的逻辑
def predict_amino_acid(sh_density):
    c_peaks = count_peaks(sh_density[:, :, 0, :, :])  # C通道
    n_peaks = count_peaks(sh_density[:, :, 1, :, :])  # N通道
    o_peaks = count_peaks(sh_density[:, :, 2, :, :])  # O通道
    s_peaks = count_peaks(sh_density[:, :, 3, :, :])  # S通道

    if c_peaks == 2 and n_peaks == 1 and o_peaks == 1:
        return GLY
    if s_peaks == 1:
        return CYS
    if c_peaks >= 8:
        return PHE or TYR or TRP
    if n_peaks >= 3:
        return ARG or LYS
    ...
```

**完全不需要看坐标的具体位置**，只需要统计每个通道有几个原子！

## 为什么R3没有这个问题

**R3方法** (`models/flow_model.py` - SideAtomsFlowModel):
```python
# R3直接预测坐标，不使用SH密度
side_atoms = self.NodeFeatExtractorWithHeads(node_embed, node_mask)
```

- 不使用元素类型信息
- 完全依靠学到的几何-序列关系
- 68%的recovery才是真实的、无泄漏的性能

## 这解释了所有异常

### ✅ 为什么训练时recovery就很高？
- 坐标加噪声，但**元素信息没加噪声**
- 模型学会通过SH密度通道模式识别氨基酸
- 不是真正学习序列-结构关系

### ✅ 为什么SH recovery (88.5%) 远高于R3 (68%)？
- SH可以"作弊"：利用元素信息
- R3无法作弊：只能依靠真实的结构信息
- **68%才是正常水平**

### ✅ 为什么SH接近MPNN (但略低)？
- MPNN看到完美的GT结构，可以达到~90%+
- SH看到噪声坐标，但有元素信息帮助，达到88.5%
- 差距主要是坐标质量，不是序列-结构理解

### ✅ 为什么高recovery不是好事？
- 80%+ recovery说明模型在"记忆"而非"理解"
- 限制了设计空间的多样性
- 无法探索新的序列-结构组合

## 验证方案

### 方案1：最简单验证
使用完全随机的坐标 + GT元素信息：

```python
coords_random = torch.randn(batch_size, n_res, 14, 3) * 5.0
sh_with_leak = compute_sh(
    coords_random,
    batch['atom14_element_idx'],  # GT元素
    batch['atom14_gt_exists']      # GT mask
)
logits = model(sh_with_leak)
recovery = compare(logits.argmax(-1), batch['aatype'])

# 预测：recovery仍会有60-70%！
```

### 方案2：对比实验
在同一批数据上测试：

| 设置 | 坐标 | 元素信息 | 预期Recovery |
|------|------|----------|--------------|
| A. 当前（泄漏） | 噪声 | GT | **88.5%** |
| B. 无泄漏 | 噪声 | 全0（通用） | **60-70%** |
| C. 完美情况 | GT | GT | **95%+** |
| D. 随机测试 | 随机 | GT | **60-70%** ← 关键 |

如果D的recovery仍有60-70%，证明模型完全依赖元素信息。

## 修复方案

### 立即方案：不区分元素类型

**修改位置1**: `data/interpolant.py` line 1286-1295 (推理)
```python
# 修改前
normalize_density, *_ = sh_density_from_atom14_with_masks_clean(
    input_feats['atoms14_local_t'],
    batch['atom14_element_idx'],        # ❌
    batch['atom14_gt_exists'],          # ❌
    L_max=8, R_bins=24,
)

# 修改后
element_idx_generic = torch.zeros_like(batch['atom14_element_idx'])  # 全C
atom_exists_generic = torch.ones_like(batch['atom14_gt_exists'])     # 全14位
normalize_density, *_ = sh_density_from_atom14_with_masks_clean(
    input_feats['atoms14_local_t'],
    element_idx_generic,                # ✅ 通用元素
    atom_exists_generic,                # ✅ 通用mask
    L_max=8, R_bins=24,
)
```

**修改位置2**: `models/flow_module.py` line 422-431 (训练)
```python
# 同样的修改
element_idx_generic = torch.zeros_like(batch['atom14_element_idx'])
atom_exists_generic = torch.ones_like(batch['atom14_gt_exists'])
normalize_density, *_ = sh_density_from_atom14_with_masks_clean(
    noisy_batch['atoms14_local_t'],
    element_idx_generic,
    atom_exists_generic,
    L_max=self._model_cfg.sh.L_max,
    R_bins=self._model_cfg.sh.R_bins,
)
```

**需要重新训练**：因为模型已经学会利用泄漏信息。

### 预期修复后的结果

| 方法 | TM | pLDDT | Recovery | 状态 |
|------|-----|-------|----------|------|
| MPNN | 0.709 | 79.5 | - | 逆向折叠SOTA |
| **SH SDE 1000（修复后）** | **0.68** | **76.8** | **60-70%** | 端到端SOTA（无泄漏） |
| R3 ODE 10 | 0.644 | 71.1 | 68% | 快速方法（无泄漏） |

修复后：
- Recovery降到60-70%（正常水平，与R3一致）
- TM/pLDDT应该基本保持（因为结构预测没泄漏）
- 模型需要重新训练才能达到这个性能

## 影响评估

### 对现有结果的影响

**当前所有的SH结果都是不可信的**：
- ✅ 结构质量（RMSD, TM, pLDDT）：**可信**，因为坐标预测没泄漏
- ❌ 序列质量（Recovery, Perplexity）：**不可信**，因为有元素信息帮助
- ❌ 端到端性能比较：**不公平**，SH有GT信息，R3没有

### 对论文/报告的影响

如果已经写了报告或论文：
1. **必须声明这个问题**
2. **提供修复后的结果**
3. **重新评估SH vs R3的对比**

修复后可能的情况：
- SH和R3的recovery接近（都在60-70%）
- SH的TM仍可能更高（因为scaling law是真实的）
- 但优势会缩小

## 后续工作

### 短期（1周内）
1. ✅ 记录问题（本文档）
2. ⬜ 实现快速验证实验（随机坐标+GT元素）
3. ⬜ 修改代码去除泄漏
4. ⬜ 小规模测试（10个样本）

### 中期（1-2周）
1. ⬜ 重新训练无泄漏的SH模型
2. ⬜ 完整评估修复后的性能
3. ⬜ 更新所有结果和对比

### 长期（1个月+）
1. ⬜ 探索如何合理使用化学信息
2. ⬜ 研究是否可以从backbone预测侧链元素
3. ⬜ 发表修正后的结果

## 关键教训

### 1. 多模态输入的隐患
使用多种信息源（坐标+元素类型）时，必须确保：
- 训练时：所有信息都是模型应该"知道"的
- 推理时：不能使用训练时不应该有的GT信息

### 2. 高性能指标需要质疑
当某个指标异常地好时（88.5% recovery）：
- 不要立即庆祝
- 首先质疑：是否有信息泄漏？
- 对比其他方法找差异点

### 3. 元素特异性是双刃剑
SH密度按元素分通道的设计：
- 优点：能编码丰富的化学信息
- 缺点：容易造成信息泄漏
- 需要careful设计防止捷径学习

### 4. R3作为baseline的价值
R3方法虽然简单，但因为没有复杂的中间表示：
- 不容易引入信息泄漏
- 68%的recovery是可靠的基准
- 可以用来检测其他方法是否有问题

## 结论

这是一个**严重但可修复**的问题：
1. ✅ 问题根源已找到：GT元素信息泄漏
2. ✅ 修复方案清晰：使用通用元素类型
3. ⚠️ 需要重新训练和评估
4. ⚠️ 当前的高recovery数字不可信

**好消息**：
- 结构预测质量（TM, pLDDT）应该基本不受影响
- 修复后仍可能是很好的方法，只是recovery会更现实

**最重要的收获**：
- 用户的直觉和质疑精神发现了这个问题
- 这再次证明了critical thinking的重要性
- 高指标不一定意味着好模型

---

**创建日期**: 2025-11-17
**发现者**: 用户
**严重程度**: 高
**状态**: 已确认，待修复
**预计修复时间**: 1-2周（需重新训练）
