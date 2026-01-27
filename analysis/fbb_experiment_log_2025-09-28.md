## 2025-09-28（晚间更新）：诊断、采样变体与修复（中文版）

### 为什么要做这些实验（假设）
- 判定多步退化到底来自时间方向/符号，还是来自“迭代把状态 xt 带离训练桥分布（off-manifold）”。
- 排除“t 条件失效”的可能，聚焦在“xt 与 t 不匹配（噪声大但 t 大）”导致的 OOD 前向。
- 尝试更稳定的多步：α 上限（alpha-cap）、Heun 预测-校正、位移限幅、早期回桥混合，抑制超步与曝光偏差。

### 批量评测（单步与多步）
- 单步（新 loss 微调）：`inference_outputs/2025-09-28_16-57-46`
  - TM 0.603 ± 0.279；RMSD 13.501 ± 14.867 Å；pLDDT 64.131 ± 19.175；pAE 13.148 ± 6.355；Recovery 0.695 ± 0.044；PPL 2.029 ± 0.182
- 10 步：`2025-09-28_17-09-05`
  - TM 0.157 ± 0.035；RMSD 24.958 ± 10.911；pLDDT 32.304 ± 4.623；pAE 23.926 ± 2.785；Recovery 0.135 ± 0.033；PPL 69.825 ± 14.074
- DDIM 风格：`2025-09-28_17-19-33`
  - TM 0.144 ± 0.029；RMSD 26.032 ± 11.628；pLDDT 32.180 ± 6.545；pAE 24.446 ± 2.485；Recovery 0.114 ± 0.032；PPL 92.400 ± 20.979
- “最后一步直接用模型输出”：`2025-09-28_17-49-10`
  - TM 0.156 ± 0.031；RMSD 25.511 ± 10.840；pLDDT 31.972 ± 4.480；pAE 23.946 ± 2.805；Recovery 0.131 ± 0.036；PPL 77.583 ± 15.488
- 迭代中“用上一步 x1_pred 当下一步输入”：`2025-09-28_19-46-58`
  - TM 0.132 ± 0.022；RMSD 28.446 ± 11.210；pLDDT 37.491 ± 6.258；pAE 24.885 ± 2.584；Recovery 0.104 ± 0.027；PPL 98.846 ± 16.845

结论：所有多步变体都远逊于单步，指向“迭代使 xt 离桥”而非 “t 条件失效”。

### 离桥诊断（固定 t2=0.5，对比桥上 xt 与一步更新后的 xt）
- 基线（桥上样本）：`2025-09-28_21-16-45`
  - TM 0.638 ± 0.275；RMSD 11.695 ± 12.004；pLDDT 69.518 ± 19.047；pAE 12.301 ± 6.729；Recovery 0.815 ± 0.039；PPL 1.587 ± 0.146
- 一步更新（min_t→0.5）：`2025-09-28_21-27-12`
  - TM 0.191 ± 0.064；RMSD 24.801 ± 11.170；pLDDT 39.216 ± 11.158；pAE 22.639 ± 4.674；Recovery 0.157 ± 0.040；PPL 79.984 ± 23.252

结论：仅一次 Euler 更新已显著离桥 → 第二次前向 OOD，PPL 断崖式上升，结构信心同步崩溃。

### 其它多步更新规则（仍然不稳）
- xt = (1 − t1)·x0 + t1·x1_pred：`2025-09-28_21-54-58`
  - TM 0.144 ± 0.023；RMSD 26.435 ± 11.490；pLDDT 31.911 ± 6.491；pAE 24.599 ± 2.221；Recovery 0.108 ± 0.030；PPL 113.775 ± 27.022
- xt = (1 − t2)·x0 + t2·x1_pred：`2025-09-28_22-04-31`
  - TM 0.143 ± 0.023；RMSD 26.382 ± 11.056；pLDDT 32.611 ± 6.136；pAE 24.686 ± 2.194；Recovery 0.109 ± 0.033；PPL 108.086 ± 25.036

结论：步步插值到“预测桥均值”也无法修复曝光偏差；PPL 甚至更高。

### 只改 t 的敏感性（固定同一桥上 xt_fixed@0.5）
相同 xt_fixed = (1−0.5)·noise0 + 0.5·x1_gt，不改坐标，只改 t_eval：
- t_eval=min_t → `2025-09-28_22-55-24`：pLDDT 66.94；PPL 1.85
- t_eval=0.3 → `2025-09-28_22-56-51`：pLDDT 70.16；PPL 1.65
- t_eval=0.5 → `2025-09-28_22-57-14`：pLDDT 69.52；PPL 1.59
- t_eval=0.7 → `2025-09-28_22-57-32`：pLDDT 70.30；PPL 1.67
- t_eval=1.0 → `2025-09-28_22-57-52`：pLDDT 70.01；PPL 1.65

结论：当 xt 在桥上时，PPL 随 t 基本稳定（~1.6–1.9）。之前“t 越大 PPL 越差”的现象来自“xt 是纯噪声但 t 很大”的矛盾输入，而非 t 条件本身问题。

### 代码修改（中文归纳）
- 评测管线：`esm/genie/evaluations/pipeline/pipeline_pdb.py`
  - 关闭三级多样性统计 `_compute_tertiary_diversity`（加速；摘要不再包含多样性）。注意：聚合末尾 `_process_results` 存在轻微 UnboundLocalError，不影响 `fbb_scores.csv` 的生成与读取。
- 采样与诊断：`data/interpolant.py`
  - 增加每步 recovery/PPL 打印与 `self.loss(...)` 原子级误差诊断。
  - 新增稳定版多步 `fbb_sample_iterative_stable`：
    - α 上限（`_alpha_raw`/`_apply_alpha_cap`）收缩有效步长；
    - Heun 预测-校正（`_heun_step_R3`），第二次评估改用真实 `t_next=t+dt_eff`；
    - 形状修正：`t:[B]`，`r3_t/so3_t:[B,N]`；
    - 位移限幅 `_displacement_cap`（默认 0.8 Å/atom）；
    - 早期回桥混合（前 K 步，`γ0≈0.7` 线性衰减）；
    - 外层时间按 `dt_eff` 推进（不再硬跳网格 t2），桥混合与 Heun 一致使用 `t_next`。
  - Heun 步数默认增至 10（可配）。
  - 新增诊断前向：`fbb_sample_diag_baseline_t2` 与 `fbb_sample_diag_step_t1_to_t2`。

### 关键信息与建议
- 多步退化主因：曝光偏差/离桥 → 第二次前向 OOD → PPL 飙升与 pLDDT 下降。
- t 条件：在桥上样本下表现稳定；问题不在 t 本身。
- 当前策略：默认单步（t≈min_t/0.3），得到最佳稳定性；
  若继续探索多步，建议：alpha_max∈[0.3,0.5]、heun_steps=3–10、disp_cap_ang∈[0.4,0.8]、桥混合 γ 前 3 步（如 0.85→0.6→0.4）。


### Key takeaways
- Multi-step degradation is driven by exposure bias: iterative xt leaves the training bridge; second forward becomes OOD; PPL skyrockets; structure confidence collapses.
- t-conditioning is fine when xt is on-bridge; PPL stable across t.
- Simple interpolations to predicted bridge or naive DDIM/Euler do not fix the issue; require stabilized updates and/or training-time consistency.

### Next steps (recommended)
- Use single-step (t≈min_t/0.3) as default for now (best pLDDT ~64–70 with new loss).
- Try the stabilized sampler `fbb_sample_iterative_stable` with:
  - alpha_max ∈ [0.3, 0.5], heun_steps=3–10, disp_cap_ang ∈ [0.4, 0.8], bridge_gamma: γ0≈0.85→0.6→0.4 for first 3 steps.
- Training-side: add consistency/teacher-forcing so the model sees rollout xt; consider log-SNR grid and matched loss weighting.

FBB 实验与排障日志（截至 2025-09-28）

一、目标与范围
- 任务：侧链全掩码 FBB 推理与评估，修复 t 条件问题与迭代采样退化；对接 ESMFold/评估管线；建立基线（MPNN）。
- 评估：CASP15 原生结构为对齐参照；输出 FASTA+PDB；统计 TM-score/RMSD/pLDDT、sequence recovery/perplexity。

二、主要代码改动（按文件）
- models/flow_module.py
  - predict_step：重构为 FBB 推理单一入口；输出 config.yaml；生成 FASTA 与 PDB；计算 recovery/perplexity；输出目录命名含原名。
  - fbb_sample：接入 Interpolant 的推理函数，当前默认单步 t=min_t（见后述）以避免多步退化。
  - model_step_fbb：增加基于时间的 SNR 风格缩放（r3_t→r3_norm_scale=1−clip(t)）以对齐训练目标在不同 t 的权重，缓解“高噪声主导梯度”。

- models/flow_model.py（SideAtomsFlowModel）
  - 引入 BackboneEncoderGNN 并融合 node/edge 图特征；feature_fusion/edge_feature_fusion 融合 node/sidechain/graph 表示。
  - 前向将预测侧链（局部）与原始主链（局部）拼接为 14 原子局部坐标，并以 curr_rigids 映射到全局坐标返回。
  - 时间条件使用 r3_t/so3_t 进入 NodeFeatureNet（确认 dtype=float32）。

- data/interpolant.py
  - fbb_corrupt_batch（训练）：侧链 (3:) 按 (1−t)·noise0 + t·clean 的方式添加噪声；仅在 update_mask∧atom_exists 处扰动；写入 atoms14_local_t；同步写入 t/r3_t/so3_t（float32）。
  - fbb_prepare_batch（推理前处理）：以 min_t 初始化（侧链为纯噪，主链干净），统一 dtype=float32。
  - fbb_sample_iterative：迭代采样（Euler），每步仅更新侧链局部；确认 t/r3_t/so3_t 按步更新。
  - fbb_sample_consistent：新增（DDIM/一致性风格），固定 noise0，步步将 xt 回投到 (1−t)·noise0 + t·x1_pred。
  - fbb_sample_single：新增（单步），以 t_eval（默认用 min_t）一次前向得到结果。

- data/datasets.py
  - Validation/Predict 使用全量数据；在 __getitem__ 注入 source_name 便于输出命名。

- data/protein_dataloader.py
  - predict_dataloader：单进程推理避免错误地初始化 DistributedSampler。

- analysis/retag_inference_samples.py（新增）
  - 将旧格式输出目录重命名为 sample_<protein_name>_<idx>，更新 FASTA 头并生成映射 CSV。

- esm/genie/evaluations/pipeline/
  - pipeline_pdb.py：
    - evaluate_fbb：新增 FBB 评估工作流（只折叠 FASTA 第二条 predicted 序列，过滤 *_metadata.txt）。
    - _aggregate_fbb_scores：修复缩进、fallback 读取 seqlen；解析 pLDDT/pAE；残基重编号（1..N）以适配 TM-score；修复“native 对齐/重复折叠”等问题。
  - evaluate_fbb.py：新增评估脚本（仅折叠 predicted 序列，支持 native_only 基准）。
  - 其余：引入 TMscore/TMalign 绝对路径；补充 shutil import；解决 seqlen=0 与未生成结构的情况。

三、关键问题与修复
1）Default process group 未初始化（单进程推理）
- 现象：predict_dataloader 启动 DistributedSampler 抛错。
- 处理：仅在 dist.is_initialized() 时使用 DistributedSampler，否则 shuffle=False 普通 DataLoader。已修复。

2）atom14_to_atom37 报 numpy×tensor 类型错误
- 现象：STANDARD_ATOM_MASK 与 Tensor 相乘报错。
- 处理：确保 STANDARD_ATOM_MASK 保持 numpy 数组，不转为 torch.Tensor。已修复。

3）OmegaConf.save 配置保存失败
- 现象：直接保存 self.hparams 报类型不支持。
- 处理：构造时缓存完整 cfg（DictConfig）到 self._full_cfg；predict_step 保存该对象。已修复。

4）FBB 推理多步退化（步数越多越差）
- 现象：2 步优于 5/10/50/100/1000 步；pLDDT/TM 随步数下降明显。
- 定位：
  - t 在训练/推理中 dtype 已修复（float32），敏感性对比 t=0.1/0.9 的 logits 有差异（cos≈0.968，KL≈0.145），说明“非失效但偏弱”。
  - 多步迭代更像分布漂移：Euler 叠代使 xt 偏离训练桥接分布，步越多误差累积越大，分类头更不稳。
- 措施：
  - 新增 DDIM/一致性回投（fbb_sample_consistent）：未见显著提升（见四、实验）。
  - 新增单步采样（fbb_sample_single，t=min_t）：显著提升（见四、实验）。
  - 训练侧加入 SNR 风格的 t 加权，缓解“高噪声主导梯度”，用于后续再训验证。

5）TM-score 无法对齐（无公共残基）
- 现象：部分样本 native 残基编号为 662..1022，预测为 1..N，TM-score 报 no alignment。
- 处理：聚合阶段对 native 生成重编号副本（1..N），再对齐。已修复。

6）seqlen=0 与结构重复折叠/文件筛选
- 现象：聚合 seqlen=0；折叠数量翻倍。
- 处理：聚合时 seqlen 失败回退从序列文件读取；折叠时过滤 *_metadata.txt。已修复。

四、实验与结果（重点）
注：以下均为 CASP15 45 个目标的整体统计。

1）FBB（多步 Euler，2 步示例，早期版本）
- valid≈38/45；pLDDT≈56.1±19.8；TM≈0.294±0.221；RMSD≈15.1±7.5。

2）FBB（10 步 Euler/或均匀 t）
- pLDDT≈34.1±9.1；TM≈0.193±0.066；RMSD≈23.8±11.4（显著退化）。

3）FBB（DDIM/一致性回投：fbb_sample_consistent）
- 与 10 步 Euler 基本一致：pLDDT≈33.6±7.5；TM≈0.199±0.058；RMSD≈23.9±11.5。
- 结论：积分器本身不是主因，分布漂移+分类头后期退化更关键。

4）FBB（单步 t=min_t：fbb_sample_single，当前默认）
- N=45；pLDDT≈54.4±18.4；TM≈0.501±0.281；RMSD≈16.7±14.1；TM 范围[0.149, 0.934]。
- 多个样本 TM>0.9（如 T1106s2-D1、T1137s9-D1），与 2 步版本最好的情况相当甚至更稳。

5）MPNN+ESMFold 基线（CASP15，每个1条序列）
- N=45；TM≈0.650±0.295；RMSD≈11.1±12.2；pLDDT≈70.9±17.2（从 PDB B-factor 平均）。
- 输出目录：/home/junyu/project/protein-frame-flow-u/mpnn_evaluation/casp15_1sample

五、诊断与结论
- t 条件：已确认有效但单点注入偏弱；多步采样的持续退化主要来自分布漂移与后期 t 段分类头退化。
- 单步最优：单步 t=min_t 与训练起点一致，不会发生步步偏移，当前显著优于多步/一致性方案。
- SNR 加权：训练损失加入基于 t 的缩放，有望在再训练后提升后期（大 t）的鲁棒性，缓解“步数越多越差”。
- 评估流程：已稳定生成 FASTA/PDB；解决重编号、seqlen、分布式、文件过滤等问题；支持 ESMFold 与 MPNN。

六、后续建议
- 训练侧：
  - 合入 SNR 加权后再训；可考虑 log-SNR 均匀采样；或切换 v-预测头提升一致性。
  - 在主干多层注入 time embedding（FiLM/加法），增强 t 的可达性与影响范围。
- 采样侧：
  - 维持单步 t=min_t 为当前默认；
  - 输出时加入 logits 早停/多步集成（固定结构，用最佳 logits 写 FASTA）。
- 评估侧：
  - 持续对比 FBB 单步 vs. 多步（如 SNR 再训后）与 MPNN 基线；关注 pLDDT/TM 的稳定性。

七、关键命令与路径（参考）
- FBB 评估：/home/junyu/project/esm/genie/evaluations/pipeline/evaluate_fbb.py
  - 输入：FBB 输出目录（sample_* 含 FASTA+PDB）；native_dir 指向 CASP15 目录。
- MPNN 基线：/home/junyu/project/protein-frame-flow-u/mpnn_evaluation/casp15_1sample
  - sequences/（生成的序列），structures/（ESMFold 结构），results/mpnn_scores.csv。

八、里程碑与时间线（简略）
- 修复分布式/类型错误/配置保存（早期）。
- 引入 FBB 迭代采样 + ESMFold 评估；修复 TM-score 重编号/文件过滤/seqlen 回退。
- 发现“步数小更好”；确认 t dtype、做 logits t-敏感性对比；尝试 Heun/一致性无明显提升。
- 切换单步 t=min_t，显著提升指标；补回 SNR 加权用于后续再训。

（完）

