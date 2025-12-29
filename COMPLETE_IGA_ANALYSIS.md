# IGA vs IPA 完整性能分析

## 总结

根据现有评估结果，以下是IGA和IPA的完整对比：

---

## 1. CASP15 Test Set (Out-of-Distribution, 45 samples)

### 整体性能对比

| 指标 | IGA | IPA | Δ (IGA-IPA) | Winner |
|------|-----|-----|-------------|--------|
| **TM-score** | 0.508 ± 0.270 | 0.548 ± 0.274 | **-0.040 (-7.3%)** | IPA |
| **RMSD** | 16.476 ± 14.949 Å | 14.202 ± 14.007 Å | +2.274 Å | IPA |
| **pLDDT** | 53.089 ± 16.981 | 57.639 ± 16.952 | -4.550 | IPA |
| **pAE** | 16.116 ± 5.702 | 14.646 ± 5.833 | +1.470 | IPA |
| **Recovery** | 0.278 ± 0.062 | 0.294 ± 0.059 | -0.015 | IPA |
| **Perplexity** | 76.400 ± 77.489 | 25.929 ± 15.496 | +50.471 | IPA |

### 胜负统计
- **IGA更好**: 6 samples (13.3%)
- **差不多**: 25 samples (55.6%)
- **IPA更好**: 14 samples (31.1%)

### 关键发现 (CASP15)

1. **IGA略差于IPA，但差异不大** (TM-score差异仅-0.040)
2. **两者在困难样本上都表现不佳**:
   - IGA: 35.6%样本TM-score < 0.3 (very bad)
   - IPA: 26.7%样本TM-score < 0.3
3. **Perplexity差异显著**: IGA的perplexity是IPA的3倍
   - 这可能表明IGA生成的序列更加"surprising"
   - 需要进一步分析这是否是问题

---

## 2. Regular PDB Validation - IGA Performance

### IGA on ffvaldata (17 samples)
- **TM-score**: 0.893 ± 0.076 (vs native PDB)
- **Predicted pLDDT**: 78.367 ± 5.880
- **RMSD**: 3.097 ± 2.624 Å

### IGA on epoch163 (40 samples)
- **TM-score**: 0.897 ± 0.122 (vs ESMFold native)
- **Predicted pLDDT**: 78.581 ± 7.157
- **RMSD**: 3.157 ± 4.005 Å

### 一致性验证
✓ 两个验证集上的性能高度一致：
  - TM-score: 0.893 vs 0.897 (差异0.004)
  - Predicted pLDDT: 78.4 vs 78.6 (差异0.2)
  - 说明模型在训练分布内表现稳定

---

## 3. 泛化性能分析

### IGA: Regular PDB → CASP15

| 指标 | Regular PDB | CASP15 | Δ | % Change |
|------|-------------|---------|---|----------|
| TM-score | 0.893 | 0.508 | **-0.385** | **-43.2%** |
| Pred pLDDT | 78.4 | 53.1 | **-25.3** | **-32.3%** |
| RMSD | 3.1 Å | 16.5 Å | +13.4 Å | +432% |

### IPA: 没有Regular PDB数据
⚠️ **缺失IPA在Regular PDB validation set上的结果**
- 无法直接对比IGA和IPA在训练集内的性能
- 无法判断IGA在训练集内是否优于IPA

---

## 4. 核心结论

### 4.1 泛化能力对比 (CASP15)

```
IGA TM-score: 0.508  ┃
IPA TM-score: 0.548  ┃  差异: -7.3%
                     ┃
结论: IGA略差于IPA，但差异不显著 (< 0.05)
      两者都在CASP15上表现不佳
```

### 4.2 训练集内性能

**已知:**
- IGA在训练集内表现优秀 (TM=0.89)

**未知:**
- IPA在训练集内的表现（缺少数据）
- 无法判断IGA是否在训练集内优于IPA

### 4.3 问题诊断

#### 问题1: 泛化性能差 ⭐ 主要问题

**证据:**
```
IGA性能下降:  0.89 → 0.51 (-43%)
IPA性能下降:  未知 → 0.55 (下降比例未知)
```

**结论:**
- **IGA和IPA都在CASP15上失效**
- 这更像是**训练数据问题**，而非IGA架构特有问题
- CASP15包含novel folds，训练集中罕见

#### 问题2: IGA略差于IPA (CASP15)

**证据:**
```
TM-score:   0.508 vs 0.548 (-7.3%)
Perplexity: 76.4 vs 25.9 (+195%)
```

**可能原因:**
1. **Perplexity过高**: IGA生成的序列更不常见
   - 可能是IGA的采样策略问题
   - 需要检查temperature、top-k等参数

2. **IGA架构的轻微劣势**:
   - Invariant Gaussian Attention可能在困难样本上不如IPA稳定
   - 需要检查attention权重分布

3. **训练不足**:
   - IGA可能需要更多训练数据或更长训练时间
   - IPA的训练可能更充分

---

## 5. 改进建议

### 优先级1: 解决泛化问题 ⭐⭐⭐

**这是主要问题！** 两个模型都在CASP15上崩溃。

**行动计划:**
```python
# A. 数据层面
1. 增加CASP历年数据到训练集
2. 识别训练集中的"困难样本"，增加采样权重
3. 数据增强：
   - 序列突变
   - 结构扰动
   - 合成困难样本

# B. 训练策略
1. Hard example mining
2. Progressive difficulty training
3. Domain adaptation on CASP data
```

### 优先级2: 改进IGA相对IPA的劣势 ⭐⭐

**目标:** 将IGA的CASP15 TM-score从0.508提升到至少0.548（与IPA持平）

**行动计划:**
```python
# A. 降低Perplexity
1. 检查IGA的采样参数
2. 考虑使用nucleus sampling或top-k
3. 调整temperature

# B. 改进IGA架构
1. 检查attention是否正确实现
2. 考虑增加残差连接
3. 尝试不同的Gaussian parameterization

# C. 训练改进
1. 增加正则化
2. 更长的训练时间
3. 更大的batch size
```

### 优先级3: 获取更多对比数据 ⭐

**Missing data:**
- IPA在Regular PDB validation set上的表现
- 其他baseline (SimpleFold) 在两个数据集上的表现

**建议:**
```bash
# 运行IPA在Regular PDB上的评估
cd /home/junyu/project/esm/genie/evaluations/pipeline

python evaluate_val_sequences.py \
  --val_dir /path/to/IPA/val_samples_epoch163 \
  --output_csv /path/to/IPA/sequence_evaluation.csv
```

---

## 6. 最终回答

### 你的问题: "只考虑训练集内，那么IGA 和 IPA 对比怎么样？"

**回答:**

**无法完整回答，因为缺少IPA在训练集内的结果。**

但根据现有数据：

1. **IGA在训练集内表现优秀**:
   - TM-score: 0.89-0.90
   - Predicted pLDDT: 78.4-78.6
   - 性能稳定，两个验证集一致

2. **IGA在CASP15上略差于IPA**:
   - TM-score: 0.508 vs 0.548 (-7.3%)
   - 但差异不显著（55.6%的样本差不多）

3. **推测:**
   如果IPA在训练集内也达到TM~0.90，那么：
   ```
   训练集内:  IGA ≈ IPA  (都~0.90)
   CASP15:    IGA < IPA  (0.51 vs 0.55)

   结论: IGA的泛化能力略差于IPA
   ```

4. **更大的问题是泛化，而非训练集内性能**:
   - 两者都在CASP15上大幅下降
   - 训练集内谁更好不是关键
   - **关键是如何提升CASP15性能**

---

## 7. 下一步行动

### 立即行动 (本周)

1. ✅ **获取IPA在Regular PDB上的结果**
   - 如果已有checkpoint，直接运行评估
   - 对比IGA vs IPA在训练集内的表现

2. **分析IGA的perplexity过高问题**
   - 检查生成的序列
   - 对比IGA和IPA的序列分布
   - 调整采样参数

3. **分析13个pLDDT<40的失败案例**
   - 找出共同特征
   - 确定失败模式

### 中期计划 (本月)

4. **数据增强实验**
   - 添加CASP14数据到训练集
   - 实现hard example mining
   - 测试性能提升

5. **IGA架构改进**
   - 尝试降低perplexity的方法
   - 优化attention机制
   - A/B测试不同配置

### 长期目标 (下个月)

6. **系统性泛化能力改进**
   - Progressive difficulty training
   - Domain adaptation
   - Ensemble methods

---

## 附录: 数据文件位置

- IGA CASP15: `/home/junyu/project/pu/outputs/IGA_xlocal=μ+u⊙σ/val_seperated_Rm0_t0_step0_20251210_165253_eval/fbb_results/fbb_scores.csv`
- IPA CASP15: `/home/junyu/project/pu/outputs/IPA/val_seperated_Rm0_t0_step0_20251208_130502_eval/fbb_results/fbb_scores.csv`
- IGA ffvaldata: `/home/junyu/project/pu/outputs/IGA_xlocal=μ+u⊙σ_ffvaldata/val_seperated_Rm0_t0_step0_20251210_183038_fbb_eval/fbb_results/fbb_scores.csv`
- IGA epoch163: `/media/junyu/DATA/pu5090weight/pdb__fbb_iga_simplified_attention_xlocal=μ + u ⊙ σ_2025-12-09_13-52-05/val_samples_epoch163/sequence_evaluation.csv`
- 对比分析: `/home/junyu/project/pu/iga_vs_ipa_casp15_comparison.csv`
