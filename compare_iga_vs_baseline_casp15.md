# IGA vs Baseline on CASP15 - 对比实验计划

## 目标
确定CASP15性能差是因为：
- A. IGA架构问题（相比baseline更差）
- B. 通用问题（所有方法在CASP15都差）

## 实验设置

### 需要对比的模型：
1. **IGA** (你的模型) - 已完成
   - TM-score: 0.508
   - pLDDT: 53.1

2. **IPA Baseline** (需要运行)
   - 使用相同的CASP15输入
   - 相同的评估流程

3. **SimpleFold** (如果有)
   - 作为额外参考

## 如何运行对比

### Step 1: 找到IPA baseline的CASP15输出
```bash
# 查找IPA在CASP15上的输出
find /home/junyu/project/pu/outputs -name "*casp*" -o -name "*T11*" | grep -v IGA
```

### Step 2: 运行FBB评估
```bash
# 假设IPA输出在这里 (需要你确认实际路径):
cd /home/junyu/project/esm/genie/evaluations/pipeline

python evaluate_fbb.py \
  --fbb_output_dir /path/to/IPA/casp15/output \
  --output_dir /path/to/IPA/casp15/output_eval \
  --native_dir /home/junyu/project/casp15/targets/casp15.targets.TS-domains.public_12.20.2022 \
  --verbose
```

### Step 3: 比较结果
```python
import pandas as pd

# IGA结果
iga = pd.read_csv('/home/junyu/project/pu/outputs/IGA_xlocal=μ+u⊙σ/val_seperated_Rm0_t0_step0_20251210_165253_eval/fbb_results/fbb_scores.csv')

# IPA结果 (需要你运行后得到)
ipa = pd.read_csv('/path/to/IPA/casp15/eval/fbb_scores.csv')

print(f"IGA  TM-score: {iga['TM_score'].mean():.3f}")
print(f"IPA  TM-score: {ipa['TM_score'].mean():.3f}")
print(f"Difference:    {iga['TM_score'].mean() - ipa['TM_score'].mean():.3f}")
```

## 预期结果分析

### 情况A: IGA << IPA (IGA明显更差)
```
IGA TM-score: 0.508
IPA TM-score: 0.750
→ 说明IGA架构有问题，泛化能力不如IPA
→ 需要改进IGA模型设计
```

### 情况B: IGA ≈ IPA (差不多)
```
IGA TM-score: 0.508
IPA TM-score: 0.520
→ 说明是训练数据问题，不是IGA特有的
→ 需要改进训练数据分布
```

### 情况C: IGA > IPA (IGA反而更好)
```
IGA TM-score: 0.508
IPA TM-score: 0.450
→ 说明IGA实际上改进了泛化
→ 但absolute性能仍然不够，需要更好的训练数据
```

## 下一步行动

根据对比结果：
- 如果是架构问题 → 改进IGA设计
- 如果是数据问题 → 增加困难样本、数据增强
- 如果是通用问题 → 可能需要根本性的方法改进
