# 早期Perplexity爆炸问题修复记录

**时间**: 2025年初期实验

---

## 问题现象

**症状**: 多步采样(10步/100步)导致序列质量崩溃
- Perplexity: 1.84 → 654.23 (爆炸)
- Recovery: 0.739 → 0.062 (崩溃)

**表现**: 步数越多，ESM-2序列困惑度越高，氨基酸序列越不合理

---

## 根本原因

**时间步不平衡**: 训练时未使用SNR加权，导致模型对不同时间步t的学习不均匀
- 早期t (noise多): 信号弱，难学习
- 晚期t (noise少): 信号强，过拟合

**采样失配**: 多步采样累积了早期时间步的误差

---

## 解决方案

### 1. SNR加权 (Signal-to-Noise Ratio)
```python
snr = torch.sqrt(snr)  # √SNR weighting
snr_weight = snr_weight.clamp(min=1.0, max=10.0)
```

### 2. Linear Bridge重建
```
x_pred = noise * (1-t) + clean * t
```

结合噪声和干净信号进行渐进式重建

---

## 修复结果

| 指标 | 修复前 | 修复后 |
|------|-------|--------|
| Perplexity | 654.23 | 1.84 |
| Recovery | 0.062 | 0.739 |

**模型**: `pdb__Encoder11atoms_chroma_SNR1_linearBridge`

---

**结论**: SNR加权平衡了训练时间步，Linear Bridge改善了采样稳定性，彻底解决了多步采样下的序列崩溃问题。

---

**参考**: `/home/junyu/project/pu/analysis/perplexity_sampling_experiments.md`
