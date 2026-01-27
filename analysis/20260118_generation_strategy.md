# 全原子生成策略记录
**日期**: 2026年1月18日

## 核心思想
为了解决全原子生成中“序列-结构”联合生成的矛盾（即噪声阶段不应泄露侧链原子数量），采用 **Coarse-to-Fine (由粗到细)** 的策略。

## 1. 生成阶段 (Coarse / Gaussian Proxy)
在 Flow Matching / Diffusion 过程中，模型**不直接操作原子坐标**，而是操作一组固定维度的、与氨基酸类型无关的**高斯刚体参数**。

**状态变量**:
*   **Backbone**:
    *   `Trans` ($T \in \mathbb{R}^3$): 主链平移（如 CA 位置）。
    *   `Rot` ($R \in SO(3)$): 主链旋转（Frame 方向）。
*   **Sidechain (Gaussian Proxy)**:
    *   `Local_Mean` ($\mu_{local} \in \mathbb{R}^3$): 侧链质心相对于主链 Frame 的偏移量。
    *   `Scaling_log` ($\log S \in \mathbb{R}^3$): 侧链高斯分布的形状/大小（对数空间）。

**优势**:
*   **信息隐藏**: 无论是什么氨基酸，这组参数的维度都是固定的。在 $t=1$ 时，可以从通用分布中采样，完全不泄露序列信息（如原子数量）。
*   **物理意义**: `OffsetGaussianRigid` 能够很好地抽象侧链的空间占据情况。

## 2. 参数化与加噪策略 (Parameterization & Corruption)

### A. $\alpha$ 参数化 (Alpha Parameterization)
为了利用数据中的强相关性（质心偏移量 $\mu$ 通常被 Scaling $S$ 包裹），我们引入相对系数 $\alpha$：

$$
\mu_{local} = \alpha \odot S = \alpha \odot \exp(\text{Scaling\_log})
$$

*   **统计依据**: 实测数据显示，全原子构建的高斯分布中，绝大多数残基的 $|\mu| / S < 1.6$，且从未超过 2.0。
*   **模型预测**: 模型预测 $\alpha$ 和 $\log S$。
    *   $\alpha$ 的取值范围约为 $[-2.5, 2.5]$。
    *   $\log S$ 的取值范围约为 $[-1, 1]$。
*   **优势**: 解耦了“尺寸”与“方向”，并归一化了数值范围，利于神经网络学习。

### B. 噪声分布 (Noise Distribution)
*   **$\alpha$**: 使用标准正态分布 $\mathcal{N}(0, 1)$。
    *   自然覆盖 $[-3, 3]$，完美匹配真实数据的分布范围。
*   **$\text{Scaling\_log}$**: 使用标准正态分布 $\mathcal{N}(0, 1)$。
    *   覆盖范围对应物理尺寸约 $[0.13 \text{Å}, 7.3 \text{Å}]$，包容了所有天然氨基酸侧链。

### C. 实现细节 (Implementation)
在 `interpolant.py` 中：
1.  **通用扰动函数**: `_corrupt_parameter`，执行不带单位缩放的线性插值。
2.  **流程**:
    *   计算 GT $\alpha_1 = \mu_1 / (S_1 + \epsilon)$。
    *   采样噪声 $\alpha_0, \log S_0 \sim \mathcal{N}(0, 1)$。
    *   插值得到 $\alpha_t, \log S_t$。
    *   恢复 $\mu_t = \alpha_t \odot \exp(\log S_t)$。
    *   构建包含噪声参数的 `OffsetGaussianRigid` 作为模型输入。

## 3. 特征构建与模型输入 (Feature Construction)
*   **Backbone 几何**: 使用 **Chroma 风格** 的 Rotation (Frame) 和 Translation。
*   **原子化特征 (Atomization)**:
    *   在模型内部（或数据预处理阶段），使用 `OpenFoldFrameBuilder` 将扰动后的 `rotmats_t` 和 `trans_t` 转换为 **Backbone 原子坐标 (N, CA, C)**。
    *   这些原子坐标作为几何特征（如 GNN 的节点位置、边距离等）输入到模型中。
*   **侧链特征**:
    *   **禁止输入**: 侧链原子坐标、侧链 Mask。
    *   **允许输入**: 扰动后的 Gaussian 参数 ($\alpha_t, \log S_t$) 可以作为节点标量特征输入。

## 4. 模型输出与解码 (Model Output & Decoding)
*   **模型输出**: 预测完整的 **OffsetGaussianRigid** 参数更新：
    *   `Trans` & `Rot` (Backbone)
    *   `Alpha` & `Scaling_log` (Sidechain Gaussian)
*   **解码 (Fine Stage)**:
    *   当生成过程结束（或在训练计算 Loss 时），利用预测出的粗粒度参数和序列信息，恢复全原子结构。
    *   使用内坐标 (Internal Coordinates) 方法或局部解码网络，将 Gaussian 参数映射回具体的原子坐标 `x_hat`。

## 5. 训练目标
*   **Flow Matching Loss**:
    *   针对 `Trans`, `Rot` 的 SE(3) 匹配损失。
    *   针对 $\alpha$, $\log S$ 的 $\mathbb{R}^3$ 向量场匹配损失 (MSE)。
*   **Reconstruction Loss**: 解码后的原子坐标 `x_hat` 与 Ground Truth 原子的重构损失（如 MSE, RMSD）。
*   **Sequence Loss**: 序列预测的 Cross Entropy Loss。

---
## 2026年1月24日更新
*   **Dataset 更新**: 在 `datasets.py` 中，将 Rotation 的定义替换为 **Chroma** 的定义方法 (`chroma.layers.structure.backbone.FrameBuilder`)。
    *   **目的**: 实现 Rotation + Translation 到 Backbone 原子 (N, CA, C, O) 之间的随意转化 (Invertible)，确保从骨架参数能精确恢复原子坐标。

---
## 2026年1月24日更新 (关于 Scaling 捷径的警示)
**核心风险**: `Scaling_log` 是一个低维、强分簇（与氨基酸类型强相关）的变量。在简单的线性插值桥接下，它可能在中等时间步（t=0.5~0.7）就过早泄露了氨基酸类型信息，成为模型绕过几何推断的“捷径”。

**应对策略**:
1.  **监控**: 训练时关注模型是否过度依赖 `Scaling_log`。
2.  **验证**: 可以尝试在 diffuse 区域对 `Scaling_log` 进行 batch 内 shuffle，观察性能变化。如果性能大幅下降，说明模型在走捷径。
3.  **干预手段**:
    *   **Scaling Dropout/Shuffle**: 训练时以一定概率打乱或噪声化 `Scaling_log`。
    *   **Schedule 调整**: 让 `Scaling_log` 的去噪进度慢于其他变量（例如 $t_{scale} = t^\gamma, \gamma > 1$）。
    *   **解耦**: 考虑解除 `Local_Mean` 对 `Scaling` 的直接乘法依赖，避免 Scaling 只要准了就把 Mean 带准了。

---
*记录人: AI Assistant*
