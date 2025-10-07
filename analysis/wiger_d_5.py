import torch
import torch.nn as nn
import math


class WignerTransformMatrix(nn.Module):
    """
    预构造Wigner变换矩阵，使得整个变换变成简单的矩阵乘法

    核心思想：
    R[3,3] -> 通过预构造的变换矩阵 -> expanded_features[N,N]
    然后可以直接用于attention计算
    """

    def __init__(self, max_l=2, target_dim=16):
        super().__init__()
        self.max_l = max_l
        self.target_dim = target_dim

        # 计算各阶维度
        self.wigner_dims = [(2 * l + 1) for l in range(max_l + 1)]
        self.total_wigner_dim = sum(self.wigner_dims)

        # 预构造从旋转矩阵到扩展特征的变换矩阵
        self.register_buffer('transform_matrix', self._build_transform_matrix())

        print(f"Wigner变换: R[3,3] -> features[{target_dim},{target_dim}]")
        print(f"各阶维度: {self.wigner_dims}, 总计: {self.total_wigner_dim}")

    def _build_transform_matrix(self):
        """
        构造变换矩阵，使得：
        vec(expanded_features) = transform_matrix @ vec(R)

        其中 vec() 是矩阵向量化操作
        """
        # 输入：旋转矩阵 R[3,3] -> 9维向量
        # 输出：扩展特征矩阵 [target_dim, target_dim] -> target_dim^2 维向量

        input_dim = 9  # vec(R)
        output_dim = self.target_dim * self.target_dim  # vec(expanded_features)

        # 初始化变换矩阵
        transform = torch.zeros(output_dim, input_dim)

        # 为不同的Wigner阶构造对应的变换
        output_start = 0

        for l in range(self.max_l + 1):
            dim_l = self.wigner_dims[l]

            if l == 0:
                # l=0: 标量，映射到对角线元素
                self._fill_scalar_transform(transform, output_start, dim_l)
            elif l == 1:
                # l=1: 向量，直接对应旋转矩阵
                self._fill_vector_transform(transform, output_start, dim_l)
            elif l == 2:
                # l=2: 二阶张量，更复杂的映射
                self._fill_tensor_transform(transform, output_start, dim_l)

            output_start += dim_l * dim_l

        return transform

    def _fill_scalar_transform(self, transform, start_idx, dim_l):
        """
        填充标量(l=0)的变换：标量在旋转下不变
        映射到单位矩阵
        """
        # l=0对应1x1矩阵，总是单位元
        end_idx = start_idx + dim_l * dim_l  # 1
        # 不依赖于R，总是1
        # 这里我们可以让它依赖于R的迹（旋转的"强度"）
        for i in range(3):
            transform[start_idx, i * 3 + i] = 1.0 / 3.0  # trace(R)/3

    def _fill_vector_transform(self, transform, start_idx, dim_l):
        """
        填充向量(l=1)的变换：直接使用旋转矩阵
        """
        # l=1对应3x3矩阵，直接复制旋转矩阵
        end_idx = start_idx + dim_l * dim_l  # 9

        for i in range(dim_l):
            for j in range(dim_l):
                output_pos = start_idx + i * dim_l + j
                input_pos = i * 3 + j  # R[i,j] 在 vec(R) 中的位置
                transform[output_pos, input_pos] = 1.0

    def _fill_tensor_transform(self, transform, start_idx, dim_l):
        """
        填充二阶张量(l=2)的变换：基于R构造5x5矩阵

        这里用简化的方法：让5x5矩阵的某些元素依赖于R的元素
        实际中可以使用更精确的Wigner公式
        """
        end_idx = start_idx + dim_l * dim_l  # 25

        # 简化的映射：基于旋转矩阵的元素构造5x5矩阵
        # 这里使用一些启发式的映射规则

        # 对角线元素：基于R的对角线
        for i in range(min(dim_l, 3)):
            diag_pos = start_idx + i * dim_l + i
            transform[diag_pos, i * 3 + i] = 1.0

        # 其他元素：基于R的非对角线元素的组合
        mapping_rules = [
            # (output_i, output_j, input_combinations)
            (0, 1, [(0, 1, 0.5), (1, 0, 0.5)]),  # 对称元素
            (0, 2, [(0, 2, 0.7), (2, 0, 0.3)]),
            (1, 2, [(1, 2, 0.6), (2, 1, 0.4)]),
            (0, 4, [(0, 1, 0.3), (1, 0, 0.3), (0, 2, 0.2), (2, 0, 0.2)]),
            # 可以添加更多映射规则...
        ]

        for out_i, out_j, input_combos in mapping_rules:
            if out_i < dim_l and out_j < dim_l:
                output_pos = start_idx + out_i * dim_l + out_j
                for inp_i, inp_j, weight in input_combos:
                    input_pos = inp_i * 3 + inp_j
                    transform[output_pos, input_pos] = weight

                    # 对称填充
                    if out_i != out_j:
                        sym_output_pos = start_idx + out_j * dim_l + out_i
                        transform[sym_output_pos, input_pos] = weight

    def forward(self, R):
        """
        应用预构造的变换矩阵

        R: [B, N, 3, 3] 旋转矩阵
        返回: [B, N, target_dim, target_dim] 扩展特征矩阵
        """
        batch_shape = R.shape[:-2]
        device = R.device

        # 将旋转矩阵向量化
        R_vec = R.view(*batch_shape, 9)  # [B, N, 9]

        # 应用变换矩阵
        expanded_vec = torch.matmul(R_vec, self.transform_matrix.t())  # [B, N, target_dim^2]

        # 重塑为矩阵形式
        expanded_matrix = expanded_vec.view(*batch_shape, self.target_dim, self.target_dim)

        return expanded_matrix


class WignerEnhancedAttention(nn.Module):
    """
    使用Wigner变换增强的注意力机制
    保持原有attention的计算结构
    """

    def __init__(self, hidden_dim=16, num_heads=8, max_l=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_l = max_l

        # 原有的注意力参数
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Wigner变换矩阵
        self.wigner_transform = WignerTransformMatrix(max_l=max_l, target_dim=hidden_dim)

        # 融合权重
        self.fusion_weight = nn.Parameter(torch.ones(1))

    def forward(self, x, rotations, mask=None):
        """
        x: [B, N, H, G, hidden_dim] 输入特征
        rotations: [B, N, 3, 3] 旋转矩阵
        mask: [B, N] 注意力掩码
        """
        B, N, H, G, D = x.shape

        # 标准注意力计算
        q = self.q_proj(x)  # [B, N, H, G, D]
        k = self.k_proj(x)  # [B, N, H, G, D]
        v = self.v_proj(x)  # [B, N, H, G, D]

        # 重塑为注意力计算的格式
        q = q.view(B, N, H * G, D).transpose(1, 2)  # [B, H*G, N, D]
        k = k.view(B, N, H * G, D).transpose(1, 2)  # [B, H*G, N, D]
        v = v.view(B, N, H * G, D).transpose(1, 2)  # [B, H*G, N, D]

        # 计算标准注意力分数
        scale = math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H*G, N, N]

        # 计算Wigner增强项
        wigner_matrices = self.wigner_transform(rotations)  # [B, N, D, D]

        # 将Wigner矩阵用于增强attention
        # 方法1：直接加到attention scores上
        wigner_scores = self._compute_wigner_attention_scores(q, k, wigner_matrices)

        # 融合标准attention和Wigner attention
        enhanced_scores = scores + self.fusion_weight * wigner_scores

        # 应用掩码
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
            mask_2d = mask_expanded * mask_expanded.transpose(-2, -1)  # [B, 1, N, N]
            enhanced_scores = enhanced_scores.masked_fill(~mask_2d, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(enhanced_scores, dim=-1)

        # 应用到values
        out = torch.matmul(attn_weights, v)  # [B, H*G, N, D]

        # 重塑回原始格式
        out = out.transpose(1, 2).contiguous().view(B, N, H, G, D)

        # 输出投影
        out = self.out_proj(out)

        return out

    def _compute_wigner_attention_scores(self, q, k, wigner_matrices):
        """
        计算基于Wigner变换的注意力分数

        q: [B, H*G, N, D]
        k: [B, H*G, N, D]
        wigner_matrices: [B, N, D, D]
        """
        B, HG, N, D = q.shape

        # 扩展Wigner矩阵到所有头
        wigner_expanded = wigner_matrices.unsqueeze(1).expand(B, HG, N, D, D)  # [B, H*G, N, D, D]

        # 对query和key应用Wigner变换
        q_transformed = torch.matmul(q.unsqueeze(-2), wigner_expanded).squeeze(-2)  # [B, H*G, N, D]
        k_transformed = torch.matmul(k.unsqueeze(-2), wigner_expanded).squeeze(-2)  # [B, H*G, N, D]

        # 计算变换后的注意力分数
        wigner_scores = torch.matmul(q_transformed, k_transformed.transpose(-2, -1))  # [B, H*G, N, N]

        return wigner_scores / math.sqrt(D)


# 使用示例
def example_usage():
    # 模型参数
    batch_size, seq_len = 2, 100
    hidden_dim, num_heads, num_groups = 16, 8, 4

    # 创建模型
    attention = WignerEnhancedAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        max_l=2
    )

    # 模拟输入
    x = torch.randn(batch_size, seq_len, num_heads, num_groups, hidden_dim)
    rotations = torch.randn(batch_size, seq_len, 3, 3)

    # 确保旋转矩阵是正交的（简化处理）
    U, _, Vt = torch.svd(rotations)
    rotations = torch.matmul(U, Vt)

    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    print(f"输入特征: {x.shape}")
    print(f"旋转矩阵: {rotations.shape}")

    # 前向传播
    output = attention(x, rotations, mask)

    print(f"输出特征: {output.shape}")

    # 验证Wigner变换
    wigner_transform = attention.wigner_transform
    wigner_matrix = wigner_transform(rotations[:1, :1])  # 测试单个样本
    print(f"Wigner变换矩阵: {wigner_matrix.shape}")

    return output


if __name__ == "__main__":
    result = example_usage()