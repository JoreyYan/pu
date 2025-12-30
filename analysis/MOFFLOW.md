

# MOFDIFF

**Coarse-Grained Diffusion with Invariant Gaussian Attention for MOF Structure Generation**

---

## 1. 任务背景与核心动机

金属有机框架（MOF）的结构生成，本质上面临两个高度耦合却尺度不同的问题：

1. **拓扑问题**：

   > 不同金属簇（SBU）与有机配体如何在三维空间中组合，形成稳定、合理的整体骨架？

2. **组分问题**：

   > 在给定的拓扑与空间约束下，使用了哪些具体的基本单元（配体类型、金属簇类型），以及它们的精细原子构型？

现有的 MOF 生成方法往往在两个极端之间摇摆：

* **原子级生成**：表达力强，但搜索空间巨大、拓扑难学；
* **离散拓扑枚举 + 模板拼接**：高效但不可微，难以探索新结构。

### 核心思想

我们提出：
**将 MOF 生成问题分解为“空间骨架生成（coarse）→ 结构细化（decode）”两个阶段。**

其中，最关键、最困难的一步是：

> **如何生成一个合理的 coarse-grained 空间骨架，使其天然蕴含 MOF 的拓扑与几何约束。**

---

## 2. Coarse 表示：高斯体作为基本生成单元

### 2.1 为什么不用点、而用“球/椭圆”

在 MOF 中：

* 一个金属簇 ≠ 一个点
* 一个配体 ≠ 一个点

它们都有：

* 空间体积
* 排斥
* 可连接方向
* 尺度差异

因此，我们用 **高斯体（Gaussian / Ellipsoid）** 表示每一个 coarse 单元：

[
\mathcal{G}_i = (\mu_i, \Sigma_i, s_i)
]

其中：

* (\mu_i \in \mathbb{R}^3)：空间中心
* (\Sigma_i \in \mathbb{R}^{3\times3})：形状与尺度（球/椭球）
* (s_i \in \mathbb{R}^C)：语义 latent（对应“这是什么单元”）

**直观理解**：

* (\mu)：放在哪里
* (\Sigma)：占多大空间、朝向如何
* (s)：可能是什么（配体？金属簇？）

---

## 3. Invariant Gaussian Attention（IGA）

### 3.1 IGA 的定义

Invariant Gaussian Attention 是一种 **以高斯体为一等公民的注意力机制**。

对于两个节点 (i, j)，其几何相关性由高斯重叠决定：

[
\text{Attn}^{geo}_{ij}
======================

-\frac{1}{2}
(\mu_i-\mu_j)^\top
(\Sigma_i+\Sigma_j)^{-1}
(\mu_i-\mu_j)
]

该分数满足：

* 平移不变
* 旋转不变
* 尺度感知
* 连续可微

### 3.2 IGA 的核心更新

IGA 同时更新 **语义与几何**：

[
\begin{aligned}
s_i' &= \sum_j w_{ij}, V(s_j) \
\mu_i' &= \mu_i + \Delta \mu_i(s, \mu, \Sigma) \
\Sigma_i' &= \Sigma_i \oplus \Delta \Sigma_i(s, \mu, \Sigma)
\end{aligned}
]

其中 attention 权重：

[
w_{ij} = \text{softmax}_j
\big(
\text{scalar}(s_i,s_j)
+
\text{overlap}(\mu_i,\Sigma_i,\mu_j,\Sigma_j)
\big)
]

**关键点**：

* 几何不是 condition，而是网络状态
* 语义与空间是强耦合演化的

---

## 4. Coarse-Grained 扩散生成（MOFDIFF）

### 4.1 状态定义

在扩散时间 (t)，系统状态为一组高斯体：

[
X_t = {\mathcal{G}*i^t}*{i=1}^{N}
]

每个 (\mathcal{G}_i^t = (\mu_i^t, \Sigma_i^t, s_i^t))

---

### 4.2 Forward Diffusion（加噪）

我们在高斯体空间中定义 forward 过程：

* 位置噪声：
  [
  \mu_t = \alpha_t \mu_0 + \sigma_t \epsilon
  ]
* 尺度扰动：
  [
  \Sigma_t = \Sigma_0 \cdot \exp(\eta_t)
  ]
* 语义扰动：
  [
  s_t = \alpha_t s_0 + \sigma_t \epsilon_s
  ]

---

### 4.3 Reverse Diffusion（去噪）

反向过程由 **IGA-based denoiser** 实现：

[
(\hat{\mu}*{t-1}, \hat{\Sigma}*{t-1}, \hat{s}_{t-1})
====================================================

\text{IGA}_\theta(X_t, t)
]

其作用是：

* 推动高斯体分离/聚集
* 调整尺度以避免碰撞
* 逐步形成稳定拓扑结构

---

## 5. 拓扑如何“自然涌现”

本方法 **不显式预测 bond 或 graph**。

拓扑来自三个因素的共同作用：

1. **高斯重叠**：
   空间上可连接的单元自然靠近

2. **多 parent influence（soft attention）**：
   一个单元可同时受多个邻居影响
   → 符合 MOF 中多中心配位现实

3. **扩散去噪的稳定点**：
   低能态对应合理的周期/网格结构

---

## 6. Coarse → Fine 解码

当扩散结束，我们得到一组 coarse 高斯体：

[
{\mu_i, \Sigma_i, s_i}_{i=1}^N
]

### 6.1 类型解码（What）

使用 (s_i) 预测：

* 金属簇类型
* 配体类型

### 6.2 几何对齐（Where）

对每个 coarse 单元：

* 取对应的 reference 单元结构（原子坐标）
* 通过刚体变换 + 局部优化，使其匹配 ((\mu_i, \Sigma_i))

### 6.3 原子级 Refinement（可选）

可接入：

* 原子级 diffusion
* 能量最小化
* DFT / force field refinement

---

## 7. 训练目标（Loss）

总损失为多项之和：

1. **扩散去噪损失**
   [
   \mathcal{L}*{\mu}, \mathcal{L}*{\Sigma}, \mathcal{L}_{s}
   ]

2. **几何合理性**

   * 高斯 overlap 正则（避免不合理穿插）
   * 尺度约束

3. **类型监督**

   * coarse 单元类型分类 loss

4. **原子级对齐误差（可选）**

---

## 8. 方法优势与创新点

**MOFDIFF 的核心创新在于：**

1. **以高斯体而非原子/点作为生成单元**
2. **通过 IGA 在连续空间中学习拓扑**
3. **联合生成位置、尺度与语义**
4. **允许多 parent influence，符合 MOF 物理现实**
5. **将复杂 MOF 生成拆解为“空间骨架 → 结构细化”**

---

## 9. 总结一句话

> **我们提出了一种基于 Invariant Gaussian Attention 的 coarse-grained 扩散模型，在连续空间中生成 MOF 的几何骨架与拓扑结构，为后续精细化建模提供稳定、可控的基础。**

---


