### 2025-11-08 SH 实验记录与下一步

1. **实验 1：SH decoder（静态）**
   - 配置：SHFeatureHead + Transformer context（ctx_layers=8），type loss 0，atom loss 1.0。
   - 现象：训练未完全收敛时，远端原子 RMSD 从旧版 4–5 Å 降到 <0.5 Å（例如 atom11: 4.8 Å → 0.18 Å），整体均值 ≈0.8–0.9 Å；数值反演脚本也验证 SH 表示上限 ≈1 Å。
   - 结论：旧版 4–5 Å 的问题在于解码器，SH 表示本身信息充足。

2. **实验 2：SideAtomsSHFlowModel（FlowModel 分支）**
   - 实现：把 SideAtomsFlowModel 的侧链特征换成 SH embedding（normalize_density → SHFeatureHead → Transformer → SH2Atom14）。
   - 目标：在 FlowModel 的骨干上验证 SH 特征融合的效果，为含噪训练做准备。
   - 当前结果：初步训练表明 SH 分支能与 backbone GNN 成功融合，末端原子 RMSD 显著下降；仍在跑更长训练以汇总指标。

3. **总结**
   - 静态解码阶段已验证 SH 表示可逆、误差可控；关键改进在于解码结构和跨残基建模。
   - FlowModel 分支也完成 SH 替换，下一步可直接在含噪 flow/fbb 任务中测试。

4. **下一步计划**
   1. **含噪 SH 扩散/Flow**：在 SH 空间定义噪声桥，训练 flow/fbb 模型，比较 R³ vs SH 的 TM/pLDDT/perplexity 等指标。
   2. **多模态 VAE/预训练（准备阶段）**：把 frame + SH 压缩成 latent，规划 SH-LM/CLIP，对接序列/实验密度等，将来与扩散结合。
