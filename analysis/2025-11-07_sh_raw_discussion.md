### 2025-11-07 SH 讨论原始记录

1. **当前进展**
   - 新的 SH decoder（SHFeatureHead + Transformer）在未完全收敛前，末端原子 RMSD 已 <0.5 Å，表明原先 4–5 Å 是解码结构太弱，SH 表示本身有足够信息。

2. **扩散 vs VAE 的讨论片段**
   - 「扩散和 VAE 是两条路，扩散更适合高维连续空间、多约束生成；VAE 适合作 latent 压缩、预训练、多模态对齐。可以先做 VAE 得到 SH latent，再在 latent 上做扩散。」
   - 「要把 SH 真正用于多模态，需要 frame + SH latent 的联合 VAE：frame 捕捉 backbone，SH latent 把局部化学压缩，两个 latent 拼起来，再用解码器还原 atom14。」
   - 「FBB 目前假设 backbone 已知，如果要从序列直接生成结构，要在 FlowModel 前面加一个 backbone 预测器，输出 rigids，再把 SH 模块接上。」
   - 「噪声过程应该定义在 SH 上，训练时加噪 SH、预测 atom14；推理时模型输出 atom14，再映射回 SH，用同一个桥继续更新，训练/推理才一致。」
   - 「也可以在 atom14 上加噪后映射回 SH，这样噪声定义和推理保持一致，SH 只作为状态表示。」
   - 「只要整体端到端，先输出 frame 再输出 SH 也可以，只要 SH 解码器考虑 frame 反馈就行。」

3. **多模态想法片段**
   - 「Frame 提供全局刚体，SH latent 提供局部化学，两者结合才能处理蛋白/RNA/小分子/实验密度。」
   - 「借鉴 NeurIPS 2024 VAR（next-scale autoregressive）思想，可在 SH 频谱里做逐尺度预测：先低阶 SH，再高阶 refinement。」
   - 「计划把 frame + SH 压缩成统一 latent，支持多模态预训练（SH-CLIP/SH-LM），然后在 latent 上跑扩散或 next-scale autoregression。」
