# Claude: 椭球参数解码氨基酸类型 — 验证与实现

> **日期**: 2026-02-22
> **参与**: Human + Claude (Opus)
> **状态**: Phase A1 验证通过，Phase B 代码就绪待训练
> **背景**: Phase 4 已验证（12D flow matching 生成合理 backbone + 椭球），本轮验证椭球→序列假设并构建解码器

---

## 1. 核心假设

椭球参数 `(scaling_log[3] + local_mean[3])` = 6D **隐式编码了氨基酸类型**。

如果成立：
- 不需要 ProteinMPNN 做序列设计
- 12D flow matching 一次生成 backbone + 椭球 → 直接解码出 aatype + 全原子坐标
- 实现真正的 end-to-end 蛋白质生成

---

## 2. Phase A1 验证结果 — 线性基线

**方法**: 从 3042 个蛋白（608,498 个残基）提取 GT 椭球参数，用 `LogisticRegression(multinomial)` 分类 20 种氨基酸。

| 特征集 | Accuracy |
|--------|----------|
| 随机猜 (1/20) | 5.0% |
| 频率基线 (总是猜最多的) | 9.2% |
| **scaling_log 单独 (3D)** | **37.7%** |
| **local_mean 单独 (3D)** | **50.6%** |
| **两者合并 (6D)** | **60.2%** |

**结论**: **假设强成立**。

- 60.2% 远超"强信号"阈值（50%），仅用线性模型、6 个特征就能区分 20 类
- `local_mean`（侧链质心偏移方向）是主信号源，单独 50.6%
- `scaling_log`（椭球形状/大小）提供互补信息，合并后 +10%
- Phase A2 (MLP) 跳过，直接进入 Phase B

**判断**: accuracy 60.2% >> 40% 阈值 → **进入 Phase B**

---

## 3. Phase B 实现 — 已完成的代码改动

### 新建文件

| 文件 | 用途 |
|------|------|
| `models/ellipsoid_decoder_v2.py` | 两头解码器：`node_embed[256] + scaling_log[3] + local_mean[3]` → `aa_logits[20]` + `atom14_local[14,3]` |
| `analysis/validate_ellipsoid_aatype.py` | Phase A1 线性基线脚本（已跑完） |
| `analysis/train_ellipsoid_decoder_standalone.py` | Phase A2 独立 MLP 训练脚本（可选） |

### 修改文件

| 文件 | 改动 |
|------|------|
| `models/flow_model.py` | FlowModelIGA 加 `logits_head`（SequenceHead），forward 输出加 `aa_logits` + `node_embed` |
| `models/flow_module.py` | t-gated aa_loss + EllipsoidDecoderV2 集成 + optimizer 加 decoder 参数 + sample_12d/predict_step_12d 传播 aa_logits |
| `data/interpolant.py` | `sample_12d` 返回第 5 个值 `model_out`（包含 aa_logits、node_embed） |
| `configs/Train_fm.yaml` | 新增 `enable_aa_head`、`aa_head`、`ellipsoid_decoder` v2 配置、loss 权重与 t-gate 阈值 |

---

## 4. 架构设计

### 4.1 Trunk 序列头 (logits_head)

```
FlowModelIGA trunk output: node_embed [B, N, 256]
  ↓ SequenceHead (3-layer MLP, dropout=0.1)
  ↓ aa_logits [B, N, 20]
```

- 直接从 trunk 的 node_embed 预测序列
- 经过 6 层 IGA，已包含丰富空间上下文
- Config: `model.enable_aa_head: true`

### 4.2 EllipsoidDecoderV2 (两头解码器)

```
node_embed[256] + scaling_log[3] + local_mean[3] = 262D
  ↓ Shared stem: LayerNorm → Linear → GELU → 256D
  ↓ 2× MLPResBlock(256, 1024)
  ├─→ aa_head: LayerNorm → Linear → GELU → Linear → 20 logits
  └─→ atom14_head: concat(shared_feat, aatype_embed[64]) → 2× MLPResBlock → Linear → 14×3
      训练: teacher forcing (GT aatype)
      推理: aa_head 预测的 aatype
```

- 与 V1 的关键区别：**加了 node_embed 作为输入**
- node_embed 帮助区分形状相似但类型不同的氨基酸（如 LEU vs ILE）
- Config: `model.ellipsoid_decoder.version: 2`

### 4.3 t-gated Loss

| Loss | t 阈值 | 理由 |
|------|--------|------|
| trunk aa_loss | t > 0.25 | t≈0 时结构是纯噪声，预测 aatype 等于猜频率 prior |
| decoder aa_loss | t > 0.25 | 同上 |
| decoder atom14_loss | t > 0.50 | atom14 需要精确局部几何，backbone 在 t<0.5 时还太模糊 |

### 4.4 node_embed detach 策略

初期 `decoder_detach_node_embed: true`：
- decoder loss 不反传到 trunk，防止不稳定的 decoder 梯度干扰已收敛的 backbone 生成
- 等 decoder 稳定后可改为 false 做 end-to-end 微调

---

## 5. 新增配置项 (Train_fm.yaml)

```yaml
model:
  enable_aa_head: true
  aa_head:
    c_hidden: 256
    num_layers: 3
    dropout: 0.1
  ellipsoid_decoder:
    enable: true
    version: 2
    c_in: 256
    d_model: 256
    num_shared_blocks: 2
    num_atom14_blocks: 2
    aatype_embed_dim: 64

experiment.training:
  aa_loss_weight: 1.0
  aa_loss_t_threshold: 0.25
  decoder_loss_weight: 1.0
  decoder_aa_loss_weight: 0.5
  decoder_use_gt_ellipsoid: true
  decoder_atom14_t_threshold: 0.5
  decoder_detach_node_embed: true
```

---

## 6. Checkpoint 兼容性

- Warm-start 加载已有 checkpoint 时，`load_partial_state_dict` 跳过不存在的 key
- 新模块 `logits_head.*` 和 `ellipsoid_decoder_v2.*` 从随机初始化开始
- 已有 trunk 参数（IGA blocks、node/edge feature net）正常加载，不受影响

---

## 7. 运行命令

```bash
# Phase B 训练（warm-start from curriculum checkpoint）
python experiments/train_se3_flows.py
```

---

## 8. 训练后验证清单

- [ ] 加载 checkpoint 无报错（新模块 key 被跳过的 log）
- [ ] 跑 1 步 training_step，所有 loss 非 NaN
- [ ] t=0.1 时 aa_loss = 0（被 gate 掉），t=0.5 时正常计算
- [ ] W&B 出现 `train/aa_loss`、`train/aa_recovery`、`train/decoder_v2_aa_loss`、`train/decoder_v2_atom14_loss`
- [ ] sample_12d 输出包含 aa_logits，argmax 可生成 aatype
- [ ] predict_step_12d 输出 PDB 包含预测的残基名称（不再全是 GLY/UNK）

---

## 9. 关键设计决策 Q&A

**Q: 为什么 decoder 用 node_embed + ellipsoid，不是只用 ellipsoid？**
A: Phase A1 证明 6D 线性就有 60%。但 Phase B 的 decoder 要达到 >80% 并预测精确原子坐标，需要 trunk 的 256D embedding（经过 6 层 IGA 的空间上下文）来区分 LEU/ILE、ASP/GLU 等形状相似的氨基酸对。

**Q: 为什么 aa loss 要 t-gate？**
A: t≈0 时结构是纯噪声。如果让模型在噪声中猜 aatype，它只会学到数据集中 aa 频率的 prior（ALA 最多 → 总是猜 ALA），而不是从结构推断序列。t > 0.25 保证结构已开始成形。

**Q: 为什么 atom14 loss 的 t 阈值更高（0.5）？**
A: atom14 是 local frame 里精确到 0.1A 级的原子坐标。backbone 在 t=0.3 时 trans_loss 仍然很大（~2A RMSD），local frame 本身就不准。只有 t > 0.5 时 backbone 足够准确，在 local frame 预测原子位置才有意义。

**Q: decoder 的 node_embed 要不要 detach？**
A: 初期 detach。decoder 从零初始化，早期梯度是垃圾信号，反传到已收敛的 trunk 会造成 catastrophic forgetting。等 decoder 的 aa_loss 和 atom14_loss 稳定后（~2k steps），可以关掉 detach 做 end-to-end 微调。

**Q: 为什么不在 ODE 过程中做 aa self-conditioning？**
A: 需要改 `node_feature_net` 的输入通道（256 → 256+20），破坏 checkpoint 结构，增加训练复杂度。先用 post-hoc（最后一步预测）验证假设。如果 aatype accuracy 足够高，Phase C 再做 SC。

**Q: 为什么有两个 aa 预测头（trunk logits_head + decoder aa_head）？**
A: 冗余设计。trunk 的 logits_head 只用 node_embed（纯空间上下文），decoder 的 aa_head 额外用了 ellipsoid 参数（显式几何信息）。两者可以互相验证，最终取更好的那个。
