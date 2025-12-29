# 分析文档主索引 (Master Index)

本文档汇总了所有分析、实验日志和技术报告。按时间顺序排列，便于追踪项目演进历程。

---

## 📋 使用说明

- **日期格式**: YYYY-MM-DD
- **文档分类**:
  - 🔬 实验日志 (Experiment Log)
  - 📊 技术报告 (Technical Report)
  - 🐛 问题分析 (Bug Analysis)
  - 📚 总结文档 (Summary)
  - ⚠️ 重要发现 (Critical Discovery)

---

## 2025年11月

### 2025-11-17 ⚠️ [信息泄漏重大发现](2025-11-17_information_leakage_discovery.md)
**类型**: 重大Bug发现
**摘要**: 发现SH+FBB方法在训练和推理时使用了GT的元素类型信息（`atom14_element_idx`）和原子存在mask（`atom14_gt_exists`），导致模型可以通过SH密度的通道模式直接推断氨基酸类型，而不是学习真正的序列-结构关系。这解释了为什么SH recovery异常高（88.5%），远超R3 (68%)。
**关键发现**:
- 不同氨基酸有独特的元素组成模式（GLY无Cβ，CYS有S等）
- SH密度按元素分4个通道，即使坐标是噪声，通道模式也能识别氨基酸
- R3的68% recovery才是无泄漏的真实水平
- 需要重新训练使用通用元素类型的模型

**影响**: 所有SH的序列质量指标（Recovery, Perplexity）不可信；结构质量指标（RMSD, TM, pLDDT）仍可信

---

### 2025-11-16 📊 [最终对比报告](FINAL_COMPARISON_REPORT.md)
**类型**: 综合报告
**摘要**: 完整评估了R3 vs SH、ODE vs SDE在10-1000步的性能。SH SDE 1000达到最佳端到端性能（TM=0.683, pLDDT=76.8, Recovery=88.5%），仅比MPNN低3.7%。发现明显的scaling law：步数从10→1000，RMSD从2.76Å降至0.84Å，异常键从93%降至0%。
**关键结果**:
- SH SDE 1000: 最佳端到端方法
- R3 ODE 10: 最快方法（10步，几何完美）
- Scaling law验证：更多步数→更好结果
- 过滤11个缺失残基样本后，所有方法提升~5% TM

**注**: ⚠️ 本报告的Recovery数字因信息泄漏而不可信（见2025-11-17发现）

---

### 2025-11-15 📊 [快速参考指南](QUICK_REFERENCE.md)
**类型**: 快速查询文档
**摘要**: 一页纸总结所有实验结果，方便快速查询各方法的RMSD、TM-score、Recovery等指标。包含所有ODE/SDE、不同步数的对比。

---

### 2025-11-14 📊 [完整诊断报告](DIAGNOSTIC_REPORT.md)
**类型**: 详细实验记录
**摘要**: 逐步记录了所有实验的详细诊断信息，包括每个step的velocity统计、坐标质量、异常键比例等。涵盖R3 FBB (10/100/500步)、SH ODE/SDE (10/100/200/300/400/1000步)的完整数据。

---

### 2025-11-13 📚 [SH测试检查清单](FINAL_SH_TEST_CHECKLIST.md)
**类型**: 测试计划
**摘要**: 为验证SH方法的correctness和性能制定的系统性测试清单，包括单元测试、集成测试、端到端测试的详细步骤。

---

### 2025-11-10 🔬 [SH实验日志](2025-11-10_sh_experiments.md)
**类型**: 实验日志
**摘要**: 记录SH方法的调试和优化过程，包括SH density计算的正确性验证、与R3方法的对比实验。

---

### 2025-11-08 🔬 [SH实验日志](2025-11-08_sh_experiments.md)
**类型**: 实验日志
**摘要**: SH方法的早期实验，探索SH作为中间表示的可行性，对比不同L_max和R_bins参数的影响。

---

### 2025-11-07 📊 [SH vs R3 原始讨论](2025-11-07_sh_raw_discussion.md)
**类型**: 技术讨论
**摘要**: 关于SH和R3两种表示方法优劣的原始讨论记录，分析各自的理论基础和实现难点。

---

### 2025-11-07 📚 [SH实验计划](2025-11-07_sh_plan.md)
**类型**: 实验规划
**摘要**: 制定SH方法的系统性测试计划，包括baseline建立、变量控制、指标定义等。

---

### 2025-11-04 📊 [SH vs R3 备忘录](2025-11-04_sh_vs_r3_memo.md)
**类型**: 技术备忘
**摘要**: 总结SH和R3两种方法的核心差异、各自优势和适用场景。讨论了为什么SH可能比直接预测原子坐标更有效。

---

## 2025年10月

### 2025-10-18 🔬 [SH方法初探](sh_1018.md)
**类型**: 实验记录
**摘要**: 首次引入SH (Spherical Harmonics) 作为侧链表示的实验记录，包括SH计算方法、特征提取和初步结果。

---

## 2025年9月

### 2025-09-29 🔬 [实验日志](experiment_log_2025-09-29.md)
**类型**: 实验日志
**摘要**: 早期实验记录，包括不同采样策略、步数和超参数的探索。

---

### 2025-09-28 🔬 [FBB实验日志](fbb_experiment_log_2025-09-28.md)
**类型**: 实验日志
**摘要**: Fixed Backbone (FBB) 方法的初始实验，建立baseline和基础框架。

---

## 技术专题文档（无明确日期）

### 🐛 [Backbone泄漏修复方案](backbone_leakage_fix_proposal.md)
**类型**: Bug修复方案
**摘要**: 详细的信息泄漏修复方案，包括3种不同策略（不区分元素、从预测获取、混合策略）及其优缺点分析。提供了具体的代码修改位置和验证实验设计。
**关联**: 2025-11-17泄漏发现的修复方案

---

### 🐛 [Backbone泄漏澄清](backbone_leakage_clarification.md)
**类型**: 问题澄清
**摘要**: 早期对backbone信息使用的澄清讨论，区分backbone coordinates和sidechain predictions的不同处理方式。

---

### 🐛 [训练推理一致性检查](train_inference_consistency_check.md)
**类型**: Bug分析
**摘要**: 检查训练和推理时的实现一致性，确保没有train-test mismatch。发现并修复了一些实现细节差异。

---

### 📊 [Velocity vs Clean Prediction分析](velocity_vs_clean_prediction.md)
**类型**: 技术分析
**摘要**: 分析flow matching中velocity prediction和clean sample prediction两种输出方式的关系和差异，讨论何时应该使用哪种。

---

### 📊 [SimpleFold速度分析](simplefold_velocity_analysis.md)
**类型**: 技术分析
**摘要**: 分析SimpleFold论文中的velocity field设计，与本项目的实现进行对比，找出差异和改进点。

---

### 📊 [SimpleFold对比](simplefold_comparison.md)
**类型**: 对比分析
**摘要**: 详细对比本项目与SimpleFold的方法论、架构设计和性能差异。

---

### 📊 [FBB采样迭代审查](fbb_sample_iterative_review.md)
**类型**: 代码审查
**摘要**: 详细审查`fbb_sample_iterative`函数的实现，确保ODE采样的数学正确性和数值稳定性。

---

### 📊 [SH作为中间表示](sh_as_intermediate_representation.md)
**类型**: 理论分析
**摘要**: 论证为什么SH density是比原子坐标更好的中间表示，从理论上分析其优势（旋转不变性、平滑性、信息密度等）。

---

### 📊 [Flow vs Latent Binder模型对比](flow_vs_latent_binder_model.md)
**类型**: 方法对比
**摘要**: 对比Flow Matching和Latent Diffusion两种生成模型范式在蛋白质设计中的应用，分析各自适用场景。

---

### 📊 [Perplexity采样实验](perplexity_sampling_experiments.md)
**类型**: 实验记录
**摘要**: 研究不同采样策略对序列perplexity的影响，探索如何平衡序列质量和多样性。

---

### 🐛 [Perplexity修复历史](perplexity_fix_history.md)
**类型**: Bug修复记录
**摘要**: 记录修复perplexity计算错误的完整历程，包括问题发现、根因分析和解决方案。

---

### 📚 [项目历史总结](project_history_summary.md)
**类型**: 项目总结
**摘要**: 从项目启动到主要里程碑的完整历史回顾，梳理技术演进路线。

---

### 📊 [SH解码准确度 (Epoch 27)](sh_decode_accuracy_epoch27.md)
**类型**: 模型评估
**摘要**: 评估SH decoder在epoch 27时的解码准确度，分析SH→atoms的重建误差。

---

### 📊 [SH解码v5结果](sh_decode_v5_results.md)
**类型**: 实验结果
**摘要**: SH decoder第5版的详细测试结果，包括不同超参数配置下的性能。

---

## 📊 统计总结

- **总文档数**: 30+
- **时间跨度**: 2025年9月 - 2025年11月
- **主要主题**:
  - Flow Matching方法开发
  - R3 vs SH表示对比
  - ODE vs SDE采样策略
  - 信息泄漏问题发现与修复
  - Scaling law验证

---

## 🎯 关键里程碑

1. **2025-09-28**: Fixed Backbone (FBB) 框架建立
2. **2025-10-18**: 引入Spherical Harmonics (SH) 表示
3. **2025-11-04**: 开始系统性对比R3 vs SH
4. **2025-11-10**: 完成ODE vs SDE对比实验
5. **2025-11-14**: 发现并验证Scaling Law (10→1000步)
6. **2025-11-16**: 完成与MPNN baseline对比
7. **2025-11-17**: 🔴 **发现信息泄漏问题** - 项目转折点

---

## 📝 文档命名规范

为保持一致性，新文档请遵循以下命名规范：

### 实验日志
```
YYYY-MM-DD_experiment_name.md
例: 2025-11-17_sde_scaling_test.md
```

### 技术报告
```
descriptive_name.md
例: backbone_attention_mechanism.md
```

### Bug分析
```
issue_description.md 或 YYYY-MM-DD_issue_description.md
例: 2025-11-17_information_leakage_discovery.md
```

### 总结文档
```
TOPIC_SUMMARY.md 或 FINAL_TOPIC_REPORT.md
例: FINAL_COMPARISON_REPORT.md
```

---

## 🔍 快速查找指引

### 想了解...

**项目整体情况** → [项目历史总结](project_history_summary.md)

**最新性能对比** → [最终对比报告](FINAL_COMPARISON_REPORT.md) ⚠️ 注意Recovery数字有问题

**快速查指标** → [快速参考指南](QUICK_REFERENCE.md)

**详细实验数据** → [完整诊断报告](DIAGNOSTIC_REPORT.md)

**SH方法原理** → [SH作为中间表示](sh_as_intermediate_representation.md)

**R3 vs SH对比** → [2025-11-04备忘录](2025-11-04_sh_vs_r3_memo.md)

**ODE vs SDE差异** → [SimpleFold对比](simplefold_comparison.md)

**信息泄漏问题** → [2025-11-17重大发现](2025-11-17_information_leakage_discovery.md) 🔴

**修复泄漏方案** → [Backbone泄漏修复方案](backbone_leakage_fix_proposal.md)

---

## ⚠️ 重要声明

**关于2025-11-17之前的所有SH结果**:

由于发现了信息泄漏问题（GT元素类型和原子存在mask被使用），2025-11-17之前所有涉及SH方法的序列质量指标（Recovery, Perplexity）均**不可信**。

**仍然可信的指标**:
- ✅ 结构质量: RMSD, TM-score, pLDDT
- ✅ 几何质量: 异常键比例
- ✅ Scaling law趋势

**需要重新评估的对比**:
- ❌ SH vs R3 的Recovery对比
- ❌ SH vs MPNN的序列质量对比
- ❌ 所有声称SH序列质量好的结论

**后续工作**:
- 修复泄漏后重新训练
- 重新评估所有序列质量指标
- 更新所有涉及的报告和文档

---

## 📅 更新日志

- **2025-11-17**: 创建主索引，汇总所有现有文档
- **2025-11-17**: 添加信息泄漏发现和重要声明

---

**维护者**: Claude + 用户
**最后更新**: 2025-11-17
**文档版本**: v1.0
