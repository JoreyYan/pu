# SimpleFold 参考项目分析

**项目来源**: Apple Research - SimpleFold (Arxiv 2025)
**论文**: "SimpleFold: Folding Proteins is Simpler than You Think"
**关键创新**: 首个纯Transformer + Flow Matching的蛋白质折叠模型，无需triangle attention或pair representation

---

## 核心架构对比

### SimpleFold 架构

| 组件 | 实现 | 说明 |
|------|------|------|
| **表示空间** | **R3 (atom-level coordinates)** | 直接在3D坐标空间diffusion |
| **Flow Matching** | Linear Interpolant | `x_t = t*x_1 + (1-t)*x_0` |
| **网络结构** | DiT (Diffusion Transformer) | 纯Transformer，无domain-specific模块 |
| **条件信息** | ESM-2 embeddings (3B) | 序列表示来自预训练语言模型 |
| **Atom Encoder** | Local Attention (queries=32, keys=128) | 编码原子级特征 → 残基级latent |
| **Residue Trunk** | 8层DiT Block (100M model) | 全局Self-Attention处理残基级信息 |
| **Atom Decoder** | Local Attention (queries=32, keys=128) | 残基级latent → 原子级坐标 |
| **Loss** | MSE + smooth LDDT loss | Flow matching velocity + 几何约束 |

**数据规模**: 8.6M distilled结构 (PDB + SwissProt + AFESM)

---

## 与你的工作对比

### 1. **表示空间选择**

| 项目 | 表示空间 | 优劣 |
|------|----------|------|
| **SimpleFold** | R3 atom coordinates (3N维) | ✅ 无损表示<br>✅ 几何质量高<br>❌ 序列-几何解耦弱 |
| **你的SH Decoder** | Spherical Harmonics密度 (9792维结构化) | ❌ 有损压缩<br>❌ 远端原子精度差 (4-5Å)<br>✅ (理论上)元素通道编码type信息 |
| **你的R3 Diffusion** | R3 atom14 coordinates | ✅ 无损表示<br>✅ 几何质量97.2%<br>✅ 与SimpleFold一致！ |

**结论**: SimpleFold验证了R3直接坐标diffusion的有效性，你的R3 diffusion方向是正确的！

---

### 2. **Flow Matching实现**

#### SimpleFold (model/flow.py:81-102)
```python
class LinearPath(BasePath):
    def compute_alpha_t(self, t):
        return t, 1  # alpha_t = t, d_alpha_t = 1

    def compute_sigma_t(self, t):
        return 1 - t, -1  # sigma_t = 1-t, d_sigma_t = -1

    # Interpolant: x_t = alpha_t * x1 + sigma_t * x0
    #            x_t = t*x1 + (1-t)*x0
```

#### 你的实现 (interpolant配置)
```yaml
rots:
  sample_schedule: exp
  exp_rate: 10
trans:
  sample_schedule: linear
  vpsde_bmin: 0.1
  vpsde_bmax: 20.0
```

**差异**:
- SimpleFold: 纯线性插值，简单直接
- 你的工作: 旋转用exp schedule，平移支持VPSDE，更复杂的噪声设计

**建议**: 如果回归R3 diffusion，可以先尝试SimpleFold的简单线性插值作为baseline

---

### 3. **网络架构对比**

#### SimpleFold: Atom Encoder → Residue Trunk → Atom Decoder

```python
# 1. Atom-level特征编码
atom_feat = [ref_pos_emb, atom_type, atom_res_pos, charge, element, ...]
atom_latent = AtomEncoder(atom_feat)  # Local attention

# 2. Pooling到残基级
latent = bmm(atom_to_token_mean, atom_latent)  # [B,N,D] -> [B,M,D]

# 3. 与ESM embeddings融合
latent = cat([latent, esm_emb]) + DiT_Trunk(latent)

# 4. Broadcast回原子级
output = bmm(atom_to_token, latent)  # [B,M,D] -> [B,N,D]

# 5. Atom Decoder输出坐标
coords = AtomDecoder(output + skip_connection)
```

#### 你的SH Decoder架构

```python
# 1. SH密度特征提取
sh_feat = SHFeatureHead(sh_density)  # [B,N,C,L,M,R] -> [B,N,H]

# 2. Contextual Transformer
ctx_out = TransformerEncoder(sh_feat + aatype_emb)

# 3. 解码到atom14
coords, logits = SH2Atom14(ctx_out)  # [B,N,H] -> [B,N,14,3]
```

**关键差异**:
1. **SimpleFold**: Atom → Residue → Atom 的U-Net式结构
2. **你的SH**: SH density → Residue → Atom 的单向解码

**SimpleFold的优势**:
- Atom Encoder保留原子级细节 → Pooling后处理更高效
- Skip connection确保原子级信息不丢失
- 你的SH方法在第一步就损失了信息（SH压缩瓶颈）

---

### 4. **Loss设计**

#### SimpleFold (simplefold.py:420-464)
```python
# 1. Flow Matching Loss (MSE on velocity)
loss = mse_loss(pred_velocity, target_velocity)

# 2. Smooth LDDT Loss (几何约束)
if use_smooth_lddt_loss:
    denoised_coords = y_t + pred_velocity * (1.0 - t)
    smooth_lddt_loss = compute_smooth_lddt(
        denoised_coords, true_coords, mask, t
    )
    loss += smooth_lddt_loss * weight
```

**LDDT计算**: 基于距离差的sigmoid平滑版本，可微分

#### 你的Loss设计
```yaml
trans_loss_weight: 1.0
rot_loss_weight: 0.5
bb_atom_loss_weight: 1
dist_mat_loss_weight: 1.0
aux_loss_weight: 0.25
chil_loss_weight: 10
type_loss_weight: 0.01
atom_loss_weight: 1.0
SH_loss_weight: 0.01
```

**差异**:
- SimpleFold: 简单MSE + 几何约束，权重固定
- 你的工作: 多个loss项，需要手动调权重（很难平衡）

---

### 5. **训练细节**

| 项目 | SimpleFold | 你的工作 |
|------|-----------|----------|
| **时间采样** | Logit-Normal: `t ~ 0.98*sigmoid(N(0.8, 1.7)) + 0.02*uniform` | 你的: `min_t=1e-4`, 按schedule采样 |
| **Rigid Alignment** | ✅ 训练时对齐GT和预测 (weighted_rigid_align) | ❌ 未明确使用 |
| **Gradient Clipping** | ✅ clip_grad_norm=2.0 | 你的配置未明确 |
| **EMA** | ✅ EMA decay=0.999 | 你的实现中有 |
| **数据增强** | Center + random augmentation | 你的: align to frame |

**SimpleFold的Rigid Alignment** (simplefold.py:404-418):
```python
# 训练时对GT做rigid alignment，使loss更稳定
with torch.no_grad():
    denoised_coords = y_t + pred_velocity * (1.0 - t)
    coords_aligned = weighted_rigid_align(
        coords, denoised_coords, weights, mask
    )
    _, _, v_t_aligned = path.interpolant(t, noise, coords_aligned)
target = v_t_aligned  # 用对齐后的target计算loss
```

**这可能是关键技巧！** 确保模型预测的velocity在正确的参考系下。

---

## SimpleFold的成功要素分析

### ✅ **为什么SimpleFold能work？**

1. **简单的表示空间**: R3坐标，无信息损失
2. **纯Transformer**: 避免domain-specific模块的复杂性，依赖scaling law
3. **大规模数据**: 8.6M distilled结构 (你的: ~PDB规模)
4. **强大的序列表示**: ESM-2 3B预训练embeddings (你的: ESM-1b?)
5. **分层处理**: Atom-level细节 + Residue-level全局信息
6. **Rigid Alignment**: 训练技巧确保几何一致性

### ❌ **你的SH Decoder失败原因 (回顾)**

1. **表示瓶颈**: SH密度编码inherently有损，远端原子精度受限
2. **信息早期损失**: SH编码阶段就丢失了原子位置精度
3. **调参地狱**: 多个loss权重难以平衡
4. **缺乏理论优势**: 元素通道的type-geometry耦合未能体现

---

## 对你的建议

### 🟢 **推荐方向1: 回归R3 Diffusion + 借鉴SimpleFold设计**

**实施步骤**:
1. **简化架构**: 参考SimpleFold的Atom Encoder → Residue Trunk → Atom Decoder
2. **添加Rigid Alignment**: 在训练时对齐GT和预测（这可能是你缺失的关键技巧）
3. **简化Loss**: 主要用MSE + LDDT，减少多loss调参负担
4. **优化时间采样**: 使用Logit-Normal采样（SimpleFold论文证明有效）
5. **检查ESM版本**: 如果可能，升级到ESM-2 3B获得更强序列表示

**优势**:
- 有Apple的3B参数模型作为参考benchmark
- R3方向已被SimpleFold验证为SOTA
- 你的R3 diffusion已有97.2%几何质量的基础

---

### 🟡 **推荐方向2: 评估type-geometry一致性问题是否真实存在**

SimpleFold论文中**没有提及type prediction**，只做坐标预测。这暗示：
- **可能1**: 序列由ProteinMPNN等单独处理，坐标生成与type解耦
- **可能2**: ESM embeddings已隐式包含type信息，无需显式预测

**实验**:
1. 用你的R3 diffusion生成backbone
2. ProteinMPNN设计序列
3. AlphaFold2检查侧链clash/空腔合理性
4. **如果质量足够好 → type-geometry耦合不是问题**
5. **如果有明显type错配 → 再考虑如何改进**

---

### 🔴 **停止SH Density方向**

SimpleFold的成功进一步证明：
- **简单的R3表示足够好**，无需复杂的密度编码
- 大规模训练 + 纯Transformer scaling > 精巧的domain知识设计
- SH的信息瓶颈无法通过调参或更深网络解决

---

## 可直接借鉴的代码模块

### 1. Rigid Alignment (utils/boltz_utils.py中应该有)
```python
def weighted_rigid_align(coords_gt, coords_pred, weights, mask):
    """在训练时对齐GT和预测，stabilize loss"""
    # 可以直接移植到你的代码
```

### 2. Smooth LDDT Loss (simplefold.py:152-207)
```python
def smooth_lddt_loss(pred_coords, true_coords, coords_mask, t):
    """可微分的LDDT作为几何约束"""
    # 比你的dist_mat_loss更principled
```

### 3. Logit-Normal时间采样 (simplefold.py:36-40)
```python
def logit_normal_sample(n=1, m=0.0, s=1.0):
    u = torch.randn(n) * s + m
    t = 1 / (1 + torch.exp(-u))
    return t
```

### 4. Local Attention Mask (architecture.py:125-149)
```python
def create_local_attn_bias(n, n_queries, n_keys):
    """创建sliding window attention mask"""
    # 降低计算复杂度，处理长序列蛋白
```

---

## SimpleFold vs 你的R3 Diffusion详细对比

| 组件 | SimpleFold | 你的R3 Diffusion | 建议改进 |
|------|-----------|-----------------|----------|
| **Backbone** | DiT (pure Transformer) | IPA + Transformer | 可简化为纯Transformer |
| **Conditioning** | ESM-2 3B + time + length | ESM-1b? + time | 升级ESM-2 |
| **训练技巧** | Rigid alignment | 未使用？ | **添加这个！** |
| **时间采样** | Logit-Normal | Uniform/Exponential? | 尝试Logit-Normal |
| **Loss** | MSE + smooth LDDT | MSE + dist_mat + ... | 简化到2项 |
| **数据规模** | 8.6M distilled | PDB (~100K?) | 如可能增加数据 |
| **Atom-level处理** | Local attention (Q=32,K=128) | 全局？ | 添加local attention |

---

## 核心洞察

**SimpleFold的成功本质**:
> "Scaling simple methods with powerful models (ESM) and large data beats hand-crafted domain knowledge."

这与你的SH实验结论一致：
- SH密度编码的"domain knowledge"（元素通道，旋转等变）**没有带来收益**
- R3直接坐标 + 大模型 (Transformer) + 大数据才是正道

**你的下一步应该是**:
1. 放弃SH density
2. 回到R3 diffusion
3. 添加SimpleFold的训练技巧（especially rigid alignment）
4. 简化loss设计
5. 系统评估type-geometry问题是否真实存在

---

## 参考资源

- **论文**: [SimpleFold: Folding Proteins is Simpler than You Think](https://arxiv.org/abs/2509.18480)
- **代码**: `/home/junyu/project/ml-simplefold/ml-simplefold/`
- **关键文件**:
  - `src/simplefold/model/simplefold.py` - 主模型 + 训练逻辑
  - `src/simplefold/model/flow.py` - Flow matching实现
  - `src/simplefold/model/torch/architecture.py` - DiT架构
  - `configs/model/simplefold.yaml` - 训练配置

---

**生成日期**: 2025-11-11
**状态**: SimpleFold验证了R3 diffusion的正确性，建议放弃SH并借鉴SimpleFold的训练技巧
