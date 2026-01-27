# 蛋白质侧链生成项目历史记录

## Timeline

### Phase 1: R3坐标Flow Matching (Epoch 129)
**目标**: 直接扩散atom14坐标生成侧链

**实验**: 对比ODE采样步数 (1步 / 10步 / 100步)

**结果**:
- 100步: 97.2%芳香环平面性, 键长偏离<0.03Å
- 1步: pLDDT 67.44 (最高) 但芳香环平面性仅9%
- pLDDT不随步数单调变化

**误区**: 曾认为1步pLDDT更高=更好 (错误)

**详细报告**: `outputs/evaluation_129_complete/SUMMARY.md`

---

### Phase 2: SH密度Decoder (当前)
**目标**: 验证SH密度→atom14 decode精度

**动机**: SH密度包含元素信息(C/N/O/S通道)，提供更强type-geometry耦合

**进展** (Epoch 27+):
- 序列准确率: 98.7%
- 整体atom14 RMSD: 1.67Å
- CB RMSD: 0.34Å, 远端原子: 4-5Å

**优化**:
- Loss调整: `atom_loss_weight=1.0`, `type_loss_weight=0.0`
- 移除tau=0.2阈值 (保留完整密度)

**分析**: `analysis/sh_decode_accuracy_epoch27.md`

---

## 关键结论

1. **多步采样至关重要**: 1→10步几何质量从9%→96.3%，100步达97.2%
2. **pLDDT不反映精细几何**: 主要看序列模式，不关心亚埃级侧链精度
3. **评估优先级**: 几何准确性(键长/键角/平面性) > TM-score > pLDDT
4. **10步性价比最高**: 96.3%质量，10倍速度
5. **R3 diffusion局限**: 几何质量优秀但type-geometry耦合弱
6. **SH密度潜力**: 元素通道可能改善type-geometry一致性，待验证

---

## Next Steps

- [ ] SH decoder达到atom14 RMSD < 1.0Å
- [ ] 实现SH density diffusion
- [ ] 对比SH vs R3生成序列的化学合理性
- [ ] 探索SH-VAE/SH-LM预训练模型

---

**Last Updated**: 2025-10-30
