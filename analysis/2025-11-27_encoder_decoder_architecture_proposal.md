# FBB侧链生成：从Flow Matching到Encoder-Decoder架构的思考

**日期**: 2025-11-27
**讨论核心**: 为什么要做FBB，当前Flow Matching方法的问题，以及为什么Encoder-Decoder可能更适合

---

## 一、项目动机：为什么要做FBB侧链生成？

### 最终目标
**Binder Design** - 设计能与target蛋白精确结合的binder蛋白

### 核心能力需求
Binder成功的关键 = **深刻理解氨基酸间相互作用**
- 界面氢键网络
- 疏水packing
- 静电互补
- 形状互补

### FBB作为代理任务的逻辑

```
最终目标: Binder Design
    ↓ 依赖
关键能力: 理解氨基酸间相互作用
    ↓ 如何验证？
代理任务: FBB侧链生成
    ↓ 推理
如果连FBB都做不好 → 对相互作用理解不到位 → Binder肯定做不好
```

**核心假设**：
- 如果模型连单个蛋白内部的侧链相互作用（packing、避免clash、氢键）都理解不好
- 那它不可能设计好两个蛋白界面的复杂相互作用
- **FBB能力是Binder能力的必要条件（但非充分条件）**

---

## 二、当前方法的问题诊断

### 当前结果（ESM + Flow Matching, step100, 随机mask训练）

```
TM-score:  0.519   ← 整体结构质量中等
Recovery:  64.0%   ← 序列识别准确率可以
pLDDT:     56.7    ← 结构置信度/物理合理性低 ⚠️
Perplexity: 4.40   ← 序列合理性尚可
```

### 核心问题：pLDDT低意味着什么？

**pLDDT不是坐标准确度，而是物理合理性**：

pLDDT = 56.7 说明：
- ❌ 侧链之间有steric clash（不理解空间排斥）
- ❌ 疏水残基packing不紧密（不理解疏水效应）
- ❌ 氢键网络不optimal（不理解静电相互作用）
- ❌ 能量不是最小化的（不理解热力学）

**这些正是做Binder需要理解的物理原理！**

### Recovery 64%已经够好，为什么pLDDT还是低？

**关键洞察**："侧链原子对 → 序列对"的逻辑是正确的（化学决定性）

但当前情况：
```
TM = 0.519 + Recovery = 64%
        ↓
说明：原子位置"大致对"，氨基酸类型也能识别
        ↓
但 pLDDT = 56.7 说明：
        ↓
原子位置虽然"大致对"，但物理上不够精确
```

**模型学到了什么 vs 没学到什么**：

✅ 学到了：
- "Trp的侧链大致在哪个方向"
- "吲哚环大致长什么样"
- "这个位置应该是疏水/极性残基"

❌ 没学到：
- "这个Trp的吲哚环应该如何**精确朝向**，才能和旁边的Arg形成cation-π作用"
- "这个Leu应该如何旋转chi角，才能和疏水核心的其他残基**紧密packing**"
- "这个Ser的侧链羟基应该指向哪里，才能和backbone形成**氢键**"

**结论**：模型学到了统计规律（"Trp一般在这"），但没学到物理原因（"为什么Trp要这样摆"）

---

## 三、为什么当前的Flow Matching没学到相互作用？

### 原因1：Loss函数只关心坐标，不关心物理

```python
loss = MSE(pred_coords, true_coords)
```

模型学习过程：
```
模型: 我把Trp侧链放在(x, y, z)
Loss: 不对，应该在(x', y', z')
模型: 好的，我记住了在这类backbone环境下Trp要放在(x', y', z')
     ↑ 死记硬背，不理解为什么
```

**应该学的是**：
```
模型: 我把Trp侧链放在(x, y, z)
Loss: 不对！
     - clash_penalty = 5.0 （和Leu有overlap）
     - interaction_loss = 3.0 （和Arg距离太远，损失cation-π作用）
     - energy = 100 REU （能量太高）
模型: 哦，我应该旋转chi2角，避免clash，同时靠近Arg
     ↑ 理解了物理原因
```

### 原因2：Flow Matching的本质是"去噪"，不是"优化相互作用"

```python
# Flow matching训练过程
for t in timesteps:
    x_t = x_0 + noise * t
    pred_noise = model(x_t, t)
    loss = MSE(pred_noise, true_noise)
```

模型学到的是：
- "在噪声水平t下，应该往哪个方向更新坐标来去噪"
- **不是**："如何优化氨基酸间相互作用"

### 原因3：缺乏显式的相互作用建模

需要检查：
- Pair representation是否动态更新？
- 模型能否学到"调整residue A会影响residue B的最优构象"？
- 是否有协同优化机制？

如果pair representation只是静态的ESM attention，没有随着生成过程动态更新，模型很难学到相互作用的协同性。

### 原因4：训练信号分散

Flow matching的loss分散在100个timestep：
- 每个timestep学一点点
- 难以建立"整体相互作用"的概念
- 推理时100步累积误差

---

## 四、提议的新方向：Encoder-Decoder架构

### 核心思想

**用一步直接生成，而不是迭代去噪**

```python
# Flow Matching（当前）
for t in [100, 99, ..., 1]:
    x = x - f(x, t) * dt  # 100步，每步修正一点

# Encoder-Decoder（提议）
all_sidechains = decoder(encoder(backbone, known_sidechains))  # 1步
```

### 为什么Encoder-Decoder更适合我们的目标？

#### 优势1：直接优化相互作用推理

Flow matching学的是：
```
"在噪声水平t下，应该往哪个方向去噪"
```

Encoder-decoder学的是：
```
"给定backbone和部分已知侧链，其他残基应该如何排列来优化相互作用"
```

**后者更接近我们的目标**。

#### 优势2：训练信号集中

- Flow matching: 损失分散在100个timestep
- Encoder-decoder: 所有信号集中在最终输出
- 更容易学到"整体最优"的概念

#### 优势3：推理快50倍

```
Flow matching: 100 steps × 50ms = 5秒/结构
Encoder-decoder: 1 step × 100ms = 0.1秒/结构
                                 ↑ 50倍快！
```

快50倍 = 更多实验 + 更快迭代 + 更容易调试

#### 优势4：更容易加入物理约束

```python
# 在decoder中直接做能量优化
for ipa_layer in decoder:
    coords = ipa_layer(features, coords)

    # 可微的能量优化
    if training:
        energy = differentiable_rosetta_energy(coords)
        coords = coords - lr * grad(energy, coords)
```

#### 优势5：支持conditional generation

训练时random mask部分侧链：
```python
mask_ratio = random.uniform(0.3, 0.9)
known_sidechains = mask * true_sidechains

pred_sidechains = model(backbone, known_sidechains, mask)
```

这样模型学会：
- 根据已知侧链推断未知侧链
- **这正是相互作用推理！**
- 也更接近binder design的场景（给定target侧链，设计binder侧链）

---

## 五、具体架构设计

### 推荐方案：AlphaFold-style架构

```python
class SidechainPredictor(nn.Module):
    def __init__(self):
        # Encoder: 处理backbone + 已知侧链
        self.evoformer = Evoformer(
            num_blocks=8,
            single_dim=256,
            pair_dim=128
        )

        # Decoder: 预测所有侧链
        self.structure_module = StructureModule(
            num_layers=8,
            ipa_heads=12
        )

        # 可选：ESM特征融合
        self.esm_encoder = FrozenESM()
        self.esm_adapter = SequenceToTrunkNetwork()

    def forward(self, backbone, known_sidechains=None, mask=None, aatype=None):
        """
        Args:
            backbone: [B, N, 4, 3] (N, CA, C, O)
            known_sidechains: [B, N, 14, 3] 部分已知侧链（训练时random mask）
            mask: [B, N] 哪些位置的侧链已知
            aatype: [B, N] 氨基酸类型（可选，用于ESM）

        Returns:
            pred_sidechains: [B, N, 14, 3]
            pred_aa_logits: [B, N, 20] 副产品
            aux_outputs: dict with intermediate features
        """
        B, N = backbone.shape[:2]

        # ============ Encoder ============

        # 1. 初始化single/pair representations
        single_repr = self.init_single_repr(backbone)  # 从backbone几何特征
        pair_repr = self.init_pair_repr(backbone)      # 距离、角度等

        # 2. 融合ESM特征（如果需要）
        if aatype is not None:
            esm_single, esm_pair = self.esm_encoder(aatype)
            esm_single, esm_pair = self.esm_adapter(esm_single, esm_pair, ...)
            single_repr = single_repr + esm_single
            pair_repr = pair_repr + esm_pair

        # 3. 融合已知侧链信息
        if known_sidechains is not None:
            known_features = self.encode_sidechains(known_sidechains)
            single_repr = single_repr + mask.unsqueeze(-1) * known_features

        # 4. Evoformer: 推理相互作用
        for block in self.evoformer:
            single_repr, pair_repr = block(single_repr, pair_repr)
            # pair_repr[i,j] 逐渐编码"residue i和j应该如何相互作用"

        # ============ Decoder ============

        # 5. Structure module: 迭代生成/refine侧链坐标
        coords = backbone  # 初始只有backbone

        for layer_idx, ipa_layer in enumerate(self.structure_module):
            # IPA: 用pair representation指导坐标更新
            coords_update = ipa_layer(
                single_repr,
                pair_repr,
                coords,
                mask=mask  # 不更新已知侧链
            )

            coords = coords + coords_update

            # 可选：每层后做物理约束投影
            if self.use_constraint_projection:
                coords = self.project_to_physical_constraints(coords)

        # 6. 输出
        pred_sidechains = coords[:, :, 4:]  # 除去backbone原子
        pred_aa_logits = self.aa_classifier(single_repr)  # 副产品

        return pred_sidechains, pred_aa_logits, {
            'single_repr': single_repr,
            'pair_repr': pair_repr,
            'intermediate_coords': coords  # 用于可视化
        }
```

### 训练策略：Random Mask

```python
def training_step(batch):
    backbone = batch['backbone_coords']  # [B, N, 4, 3]
    true_sidechains = batch['sidechain_coords']  # [B, N, 14, 3]
    true_aatype = batch['aatype']  # [B, N]

    # ========== Random Masking ==========
    # 随机决定mask比例（类似BERT，也类似你的ESM随机mask策略）
    mask_ratio = np.random.uniform(0.3, 0.9)

    # 生成mask: True=已知, False=需要预测
    mask = torch.rand(B, N) > mask_ratio

    # 对于已知位置，给真实侧链；未知位置给零或噪声
    known_sidechains = torch.where(
        mask.unsqueeze(-1).unsqueeze(-1),
        true_sidechains,
        torch.zeros_like(true_sidechains)  # 或者加入噪声
    )

    # ========== Forward ==========
    pred_sidechains, pred_aa_logits, aux = model(
        backbone=backbone,
        known_sidechains=known_sidechains,
        mask=mask,
        aatype=true_aatype  # 用于ESM（可选，或者也mask掉）
    )

    # ========== Loss ==========
    loss = compute_loss(
        pred_sidechains, true_sidechains,
        pred_aa_logits, true_aatype,
        mask
    )

    return loss


def compute_loss(pred_sc, true_sc, pred_aa, true_aa, mask):
    """综合loss函数"""

    # 1. 坐标重建loss
    coord_loss = F.mse_loss(pred_sc, true_sc)

    # 2. 物理约束losses
    clash_loss = compute_clash_penalty(pred_sc)
    packing_loss = compute_packing_density_loss(pred_sc)
    dihedral_loss = compute_dihedral_regularization(pred_sc)

    # 3. 序列预测loss（副产品，帮助学习序列-结构对应）
    aa_loss = F.cross_entropy(
        pred_aa.reshape(-1, 20),
        true_aa.reshape(-1)
    )

    # 4. 可选：能量loss
    if use_energy_loss:
        energy = differentiable_energy_function(pred_sc)
        energy_loss = F.mse_loss(energy, target_energy)
    else:
        energy_loss = 0.0

    # 加权组合
    total_loss = (
        1.0 * coord_loss +
        0.1 * clash_loss +
        0.05 * packing_loss +
        0.02 * dihedral_loss +
        0.1 * aa_loss +
        0.05 * energy_loss
    )

    return total_loss
```

---

## 六、验证流程

### 阶段1：pLDDT验证（快速筛选）

```python
def validate_with_plddt(model, test_set):
    """
    用ESMFold的pLDDT评估结构合理性
    目标：pLDDT > 70
    """
    results = []

    for sample in test_set:
        # 生成侧链（不给任何已知信息）
        pred_sidechains, pred_aa, _ = model(
            backbone=sample['backbone'],
            known_sidechains=None,  # 全部预测
            mask=None
        )

        # 构建完整结构
        structure = build_full_structure(
            sample['backbone'],
            pred_sidechains
        )

        # ESMFold评估
        with torch.no_grad():
            esmfold_out = esmfold.infer(structure)
            plddt = esmfold_out['plddt'].mean()

        # 计算recovery
        recovery = compute_sequence_recovery(
            pred_aa.argmax(-1),
            sample['true_aatype']
        )

        results.append({
            'sample_name': sample['name'],
            'plddt': plddt.item(),
            'recovery': recovery,
            'tm_score': compute_tm(structure, sample['native'])
        })

    return pd.DataFrame(results)
```

**目标指标**：
- pLDDT > 70：结构合理
- Recovery > 60%：序列合理
- TM > 0.6：整体准确

### 阶段2：Rosetta能量验证（精确评估）

```python
def validate_with_rosetta(model, test_set):
    """
    用Rosetta能量函数评估物理合理性
    目标：energy_gap < 10 REU
    """
    from pyrosetta import init, pose_from_pdb, get_score_function
    from pyrosetta.rosetta.core.scoring import ScoreType

    init()
    scorefxn = get_score_function()

    results = []

    for sample in test_set:
        # 生成结构
        pred_structure = model.generate(sample['backbone'])
        native_structure = sample['native']

        # 保存为PDB
        save_pdb(pred_structure, 'pred.pdb')
        save_pdb(native_structure, 'native.pdb')

        # Rosetta打分
        pred_pose = pose_from_pdb('pred.pdb')
        native_pose = pose_from_pdb('native.pdb')

        pred_energy = scorefxn(pred_pose)
        native_energy = scorefxn(native_pose)

        # 能量分解
        pred_breakdown = {
            'total': pred_energy,
            'fa_atr': pred_pose.energies().total_energies()[ScoreType.fa_atr],
            'fa_rep': pred_pose.energies().total_energies()[ScoreType.fa_rep],
            'fa_sol': pred_pose.energies().total_energies()[ScoreType.fa_sol],
            'hbond_sc': pred_pose.energies().total_energies()[ScoreType.hbond_sc],
            'rama': pred_pose.energies().total_energies()[ScoreType.rama],
            'omega': pred_pose.energies().total_energies()[ScoreType.omega],
            'p_aa_pp': pred_pose.energies().total_energies()[ScoreType.p_aa_pp],
        }

        native_breakdown = {
            'total': native_energy,
            # ... 同上
        }

        results.append({
            'sample_name': sample['name'],
            'pred_energy': pred_energy,
            'native_energy': native_energy,
            'energy_gap': pred_energy - native_energy,
            'pred_breakdown': pred_breakdown,
            'native_breakdown': native_breakdown
        })

    return pd.DataFrame(results)
```

**关键指标**：
- `energy_gap < 10 REU`：接近native能量
- `fa_rep < 50`：没有严重clash
- `hbond_sc` 接近native：氢键网络合理
- `fa_atr` 接近native：疏水packing合理

### 阶段3：物理验证（最严格，可选）

```python
def validate_with_md(structure, time_ns=10):
    """
    用分子动力学模拟验证稳定性
    目标：结构在MD中保持稳定（RMSD < 2Å）
    """
    from openmm.app import *
    from openmm import *
    from openmm.unit import *

    # 加载结构
    pdb = PDBFile(structure_file)

    # 设置力场
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0*nanometer,
        constraints=HBonds
    )

    # 能量最小化
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 2*femtoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=1000)

    # 运行MD
    simulation.reporters.append(
        DCDReporter('trajectory.dcd', 1000)
    )
    simulation.step(int(time_ns * 1e6 / 2))  # time_ns纳秒

    # 分析轨迹
    trajectory = mdtraj.load('trajectory.dcd', top='structure.pdb')
    rmsd_to_initial = mdtraj.rmsd(trajectory, trajectory[0])

    return {
        'stable': rmsd_to_initial[-1] < 0.2,  # 2Å
        'mean_rmsd': rmsd_to_initial.mean(),
        'final_energy': simulation.context.getState(getEnergy=True).getPotentialEnergy()
    }
```

---

## 七、实施路线图

### Phase 1: 最小可行原型（1-2周）

**目标**：验证encoder-decoder范式是否可行

```python
# 最简单的实现
class MinimalDecoder(nn.Module):
    def __init__(self):
        self.ipa_blocks = nn.ModuleList([
            InvariantPointAttention(c_s=256, c_z=128, n_heads=12)
            for _ in range(8)
        ])
        self.sidechain_head = nn.Linear(256, 14*3)

    def forward(self, backbone):
        # 不用Evoformer，不用ESM，只用IPA
        features = init_features_from_backbone(backbone)

        for ipa in self.ipa_blocks:
            features = ipa(features, backbone)

        sidechains = self.sidechain_head(features)
        return sidechains.reshape(B, N, 14, 3)

# 训练
loss = F.mse_loss(pred_sidechains, true_sidechains)
```

**验证指标**：
- pLDDT能到多少？（希望 > 60）
- 推理速度多快？（应该 < 1秒/结构）
- 训练稳定性如何？

### Phase 2: 加入Conditional Generation（2-3周）

**目标**：验证模型能否学到相互作用推理

```python
# 支持已知侧链输入
def forward(self, backbone, known_sidechains, mask):
    features = init_features(backbone)
    features += encode_known_sc(known_sidechains) * mask
    # ...
```

**实验**：
- 给定30%侧链，预测70%：pLDDT多少？
- 给定50%侧链，预测50%：pLDDT多少？
- 给定70%侧链，预测30%：pLDDT多少？

**期望**：随着已知信息增加，pLDDT应该提高（说明模型在利用context）

### Phase 3: 加入物理约束（3-4周）

**目标**：提升pLDDT到70+

```python
loss = (
    coord_loss +
    0.1 * clash_penalty +
    0.05 * packing_loss +
    0.02 * dihedral_regularization
)
```

**实验**：
- 每加入一个约束，看pLDDT提升多少
- 找到最优的权重组合

**目标指标**：pLDDT > 70

### Phase 4: Rosetta能量验证（4-5周）

**目标**：确认物理合理性

用Rosetta能量函数评估：
- Energy gap < 10 REU
- Clash分数 < 50
- 氢键、packing等分项接近native

### Phase 5: 可选扩展

如果Phase 1-4成功（pLDDT > 70, energy gap < 10）：

1. **加入Evoformer**：更强的全局推理能力
2. **加入ESM特征**：利用进化信息（如果有帮助）
3. **Self-conditioning/Recycling**：迭代refine（类似AlphaFold）
4. **Energy-guided decoding**：显式优化Rosetta能量
5. **迁移到Binder design任务**：用学到的相互作用理解设计界面

---

## 八、与Flow Matching的对比总结

| 维度 | Flow Matching（当前） | Encoder-Decoder（提议） |
|------|---------------------|----------------------|
| **训练目标** | 学习去噪方向 | 学习相互作用推理 |
| **推理速度** | 5秒/结构（100步） | 0.1秒/结构（1步） |
| **训练信号** | 分散在100个timestep | 集中在最终输出 |
| **物理约束** | 难以加入 | 容易加入（直接优化能量） |
| **条件生成** | 不自然 | 自然（训练时random mask） |
| **调试难度** | 高（100步都可能有问题） | 低（1步forward） |
| **理论上限** | 高（生成多样性） | 中（确定性预测） |
| **当前阶段适用性** | 较低（目标是学相互作用，不是生成多样性） | 较高（直接优化相互作用） |

**结论**：
- Flow matching理论上限更高，适合追求生成多样性
- 但**当前阶段**，我们的目标是"验证能否学到相互作用"
- Encoder-decoder更适合这个目标：快速、直接、易于加入物理约束
- **建议**：先用encoder-decoder验证核心假设（能否学到相互作用）
- 如果成功（pLDDT > 70），再考虑用flow matching追求更高生成质量

---

## 九、关键问题和风险

### Q1: Encoder-decoder能处理生成任务吗？

**A**: 可以，但有限制
- Encoder-decoder适合"确定性预测"（给定input，output唯一）
- FBB任务：给定backbone，最优侧链构象基本确定（能量最小）
- 不适合"多样性生成"（同一个backbone生成100种不同侧链）
- 但我们的目标就是找到"最优"的侧链，不需要多样性
- 所以encoder-decoder足够

### Q2: 如果encoder-decoder也做不好呢？

**A**: 那说明问题更根本
- 可能是数据量不够
- 可能是模型容量不够
- 可能是任务本身就很难
- 但至少我们可以快速验证这些假设（因为推理快50倍）

### Q3: 会不会过拟合？

**A**: 有风险，需要注意
- Random mask策略可以作为数据增强
- 可以用更多正则化（dropout, weight decay）
- 监控validation pLDDT，early stopping

### Q4: 如何确保学到的是物理规律，不是统计规律？

**A**: 用Rosetta能量验证
- 如果模型只是记住训练数据，在测试集上能量会很高
- 如果学到了物理规律，能量应该接近native
- 这是最终的判断标准

---

## 十、总结

### 核心洞察

1. **目标明确**：我们要学习氨基酸间相互作用，为binder design做准备
2. **代理任务合理**：FBB能力是binder能力的必要条件
3. **当前问题清晰**：pLDDT低说明没学到物理合理性
4. **方案调整必要**：Flow matching不适合当前阶段目标

### 行动计划

1. **立即开始**：实现Phase 1最小原型（1-2周）
2. **快速验证**：看pLDDT能否突破60
3. **迭代优化**：逐步加入conditional generation和物理约束
4. **严格验证**：用Rosetta能量确认物理合理性
5. **最终目标**：pLDDT > 70, energy gap < 10 REU

### 成功标志

如果encoder-decoder方法能达到：
- ✅ pLDDT > 70
- ✅ Energy gap < 10 REU
- ✅ Clash score < 50
- ✅ 条件生成测试：给定50%侧链，能准确预测另外50%

**那么我们就真正学到了氨基酸间相互作用**，可以进入binder design阶段。

---

## 附录：参考代码位置

相关现有代码：
- Flow model: `/home/junyu/project/pu/models/flow_model.py`
- IPA实现: `/home/junyu/project/pu/models/components/` (需确认)
- ESM集成: `/home/junyu/project/pu/models/components/frozen_esm.py`
- 训练脚本: 需新建

需要新增：
- `models/encoder_decoder_model.py` - 主架构
- `models/losses/physical_constraints.py` - 物理约束loss
- `evaluation/rosetta_scoring.py` - Rosetta能量评估
- `train_encoder_decoder.py` - 训练脚本
