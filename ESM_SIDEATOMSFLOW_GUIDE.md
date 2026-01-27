# ESM Integration Guide for SideAtomsFlowModel

ESM (Evolutionary Scale Modeling) 序列特征已成功集成到 `SideAtomsFlowModel` 中，用于 **backbone + atoms14 原子扩散**。

## 📦 已安装的模块

1. **frozen_esm.py** - ESM 编码器
   - 位置: `models/components/frozen_esm.py`
   - 功能: 提取 ESM single (residue-level) 和 pair (attention map) 表示

2. **sequence_adapters.py** - 序列到主干网络的适配器
   - 位置: `models/components/sequence_adapters.py`
   - 功能: 将 ESM 特征投影到模型维度并添加位置编码

## 🎯 SideAtomsFlowModel 架构

```
Input: noisy atoms14 (backbone + sidechain atoms)
    ↓
1. Extract sidechain atom features (SideAtomsFeatureHead)
    ↓
2. Extract structure features (BackboneEncoderGNN)
    ↓
3. [NEW] Extract sequence features (ESM)
   ├─ single: [B, L, nLayers, C_esm]
   └─ pair (attention map): [B, L, L, nHeads*nLayers]
    ↓
4. Fuse all features:
   node_h = structure_features + esm_single
   edge_h = structure_edge_features + esm_pair
   combined = [node_features, sidechain_features, node_h]
    ↓
5. IPA Trunk + Transformer
    ↓
6. Predict atoms14 (backbone + sidechain)
```

## 🚀 如何使用

### 1. 配置文件设置

```yaml
model:
  use_esm: true                # 启用 ESM
  esm_model: esm2_650M         # ESM 模型大小
  # 可选: esm2_8M_270K, esm2_35M_270K, esm2_650M, esm2_3B, esm2_15B

  ipa:
    c_s: 256                   # node feature dim
    c_z: 128                   # edge/pair feature dim (match ESM pair projection)

  edge_embed_size: 128         # must match c_z

  sidechain_atoms:
    A: 11                      # number of sidechain atoms
    hidden: 256                # sidechain feature dim
    conv_blocks: 4
    mlp_blocks: 4
```

### 2. 数据准备

确保你的输入数据包含以下字段：

```python
input_feats = {
    # 必需字段
    'aatype': aatype,                    # [B, N] 氨基酸类型 (AlphaFold2 格式)
    'res_mask': res_mask,                # [B, N] 残基 mask
    'diffuse_mask': diffuse_mask,        # [B, N] 扩散 mask
    'res_idx': res_idx,                  # [B, N] 残基索引
    'chain_idx': chain_idx,              # [B, N] 链索引

    # Noisy atoms
    'atoms14_local_t': atoms14_local_t,  # [B, N, 14, 3] noisy atoms14 (local frame)
    'atom14_gt_exists': atom14_exists,   # [B, N, 14] atom存在性
    'rotmats_1': rotmats_t,              # [B, N, 3, 3] noisy rotation
    'trans_1': trans_t,                  # [B, N, 3] noisy translation

    # Time step
    'r3_t': t,                           # float or [B, 1] 时间步

    # 可选：self-conditioning
    'atoms14_local_sc': atoms14_local_sc, # [B, N, 14, 3] previous prediction
}
```

### 3. 代码使用示例

```python
from models.flow_model import SideAtomsFlowModel
from omegaconf import OmegaConf

# 加载配置
config = OmegaConf.load('your_config.yaml')

# 创建模型（ESM 会自动初始化）
model = SideAtomsFlowModel(config.model)

# Forward pass
output = model(input_feats)

# 输出
side_atoms = output['side_atoms']              # [B, N, 11, 3] 预测的侧链原子 (local)
atoms_global = output['atoms_global_full']     # [B, N, 14, 3] 全局坐标
rigids = output['rigids_global']               # [B, N, 7] 刚体变换
logits = output['logits']                      # [B, N, 20] 氨基酸类型预测 (可选)
```

### 4. 不使用 ESM

如果不想使用 ESM：

```yaml
model:
  use_esm: false  # 或者不添加这个字段
```

模型会自动跳过 ESM 处理，仅使用结构和侧链特征。

## 🔍 ESM 集成的工作流程

### 在 SideAtomsFlowModel.forward 中：

```python
# 1. 提取侧链原子特征
sidechain_features = sidechain_head(atoms14_local_t[..., 3:14, :])
    ↓
# 2. 提取结构特征 (BackboneEncoderGNN)
node_h, edge_h = feature_graph(atoms14_local_for_graph)
    ↓
# 3. [NEW] 提取并融合 ESM 特征
if use_esm:
    seq_emb_s, seq_emb_z = seq_encoder(aatype, chain_idx, node_mask)
    seq_emb_s, seq_emb_z = sequence_to_trunk(seq_emb_s, seq_emb_z, ...)
    node_h = node_h + seq_emb_s  # 融合 sequence 到 structure
    edge_h = edge_h + seq_emb_z  # 融合 attention map 到 edge features
    ↓
# 4. 组合所有特征
combined = [node_features, sidechain_features, node_h]
fused_node = feature_fusion(combined)
    ↓
# 5. IPA Trunk 处理
...
```

## 💡 为什么这个设计对 atoms14 扩散特别好？

### 1. **Sequence 信息指导侧链构象**
- ESM 的 single representation 编码了序列偏好
- 某些氨基酸（如 Pro, Gly）有特定的构象限制
- ESM 帮助模型学习这些序列-构象关系

### 2. **Attention map 捕获残基间相互作用**
- ESM 的 pair representation (attention map) 编码了共进化信息
- 对于侧链-侧链接触预测很有帮助
- 例如：盐桥 (Arg-Glu)、疏水相互作用 (Leu-Val-Ile)

### 3. **三重特征融合**
```
Combined Features =
    ├─ Node features (time, mask, index)        ← 扩散条件
    ├─ Sidechain features (atoms geometry)      ← 局部几何
    ├─ Structure features (backbone GNN)        ← 全局结构
    └─ [NEW] ESM features (sequence context)    ← 进化信息
```

这种设计充分利用了：
- **结构约束**（backbone GNN）
- **局部几何**（sidechain atoms）
- **序列进化**（ESM）

## 📊 特性对比

| 特性 | **无 ESM** | **有 ESM** |
|------|-----------|-----------|
| 输入信息 | 结构 + 侧链几何 | 结构 + 侧链几何 + 序列进化 |
| Pair 表示 | 仅结构 pair | 结构 pair + ESM attention |
| 侧链预测 | 基于几何 | 几何 + 序列偏好 |
| 蛋白质设计 | 结构优先 | 结构 + 可设计性 |
| 训练稳定性 | 中等 | 更好（ESM 正则化） |

## 🧪 验证 ESM 是否正常工作

```python
import torch
from models.flow_model import SideAtomsFlowModel

# 创建模型
model = SideAtomsFlowModel(config.model)

# 检查 ESM 是否启用
if model.use_esm:
    print("✓ ESM is enabled")
    print(f"  Model: {model.seq_encoder.esm}")
    print(f"  Single dim: {model.seq_encoder.single_dim}")
    print(f"  Num layers: {model.seq_encoder.num_layers}")
else:
    print("✗ ESM is disabled")

# 测试 forward pass
B, N = 2, 50
input_feats = {
    'aatype': torch.randint(0, 20, (B, N)),
    'res_mask': torch.ones(B, N),
    'diffuse_mask': torch.ones(B, N),
    'res_idx': torch.arange(N).unsqueeze(0).repeat(B, 1),
    'chain_idx': torch.ones(B, N, dtype=torch.long),
    'atoms14_local_t': torch.randn(B, N, 14, 3),
    'atom14_gt_exists': torch.ones(B, N, 14),
    'rotmats_1': torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1),
    'trans_1': torch.zeros(B, N, 3),
    'r3_t': torch.tensor([0.5]),
}

output = model(input_feats)
print("✓ Forward pass successful!")
print(f"  Side atoms shape: {output['side_atoms'].shape}")
print(f"  Global atoms shape: {output['atoms_global_full'].shape}")
```

## ⚠️ 注意事项

1. **aatype 格式**：必须是 AlphaFold2 格式（0-20），会自动转换为 ESM 格式
2. **内存占用**：ESM-650M 需要约 2.5GB，ESM-3B 需要约 6GB GPU 内存
3. **推理速度**：ESM forward 增加约 20-30% 的推理时间
4. **ESM 参数冻结**：ESM 参数不会在训练中更新，只训练投影层
5. **Batch size**：使用 ESM 时可能需要减小 batch size

## 📝 下一步

1. 在配置文件中启用 `use_esm: true`
2. 准备包含 `aatype` 的训练数据
3. 开始训练 backbone + atoms14 扩散模型
4. 观察 ESM 是否提升侧链构象预测质量

## 🎯 预期效果

启用 ESM 后，你应该看到：
- ✅ 更准确的侧链方向（尤其是 aromatic 残基）
- ✅ 更合理的侧链-侧链接触
- ✅ 更稳定的训练（ESM 作为正则化）
- ✅ 更好的序列-结构一致性

Good luck with your backbone + atoms14 diffusion model! 🎉
