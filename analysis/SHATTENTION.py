import torch
import torch.nn.functional as F
from e3nn.o3 import spherical_harmonics, FullyConnectedTensorProduct
import math
from openfold.utils.rigid_utils import Rigid,Rotation

class SphericalHarmonicClebschGordanAttention(torch.nn.Module):
    def __init__(self, l_max, num_heads, group_size):
        super(SphericalHarmonicClebschGordanAttention, self).__init__()
        self.l_max = l_max
        self.num_heads = num_heads
        self.group_size = group_size

        # Fully connected tensor product for Clebsch-Gordan product
        self.tp = FullyConnectedTensorProduct("1e + 2e + 3e", "1e + 2e + 3e","1e + 2e + 3e")  # Assuming the input irrep is 'l_max'

    def forward(self, q, k, r_q, r_k):
        """
        Args:
            q: Query tensor, [B, N, H, G, 3]
            k: Key tensor, [B, N, H, G, 3]
            r_q: Rotation matrix for q, [B, N, H, G, 3, 3]
            r_k: Rotation matrix for k, [B, N, H, G, 3, 3]
        """

        # Step 1: Apply rotation to q and k
        q_rot = self.apply_rotation(q, r_q)  # [B, N, H, G, 3]
        k_rot = self.apply_rotation(k, r_k)  # [B, N, H, G, 3]

        # Step 2: Spherical harmonics expansion on the direction vectors
        Y_q = self.spherical_harmonic_expansion(q_rot)  # [B, N, H, G, l_max]
        Y_k = self.spherical_harmonic_expansion(k_rot)  # [B, N, H, G, l_max]

        # 2. 正确的维度重排用于attention计算
        # [B, N, H, G, l_max] -> [B, H, G, N, l_max]
        Y_q_attn = Y_q.permute(0, 2, 3, 1, 4)  # [B, H, G, N_res, l_max]
        Y_k_attn = Y_k.permute(0, 2, 3, 1, 4)  # [B, H, G, N_res, l_max]

        # 3. 计算attention分数
        # [B, H, G, N_res, l_max] @ [B, H, G, l_max, N_res] -> [B, H, G, N_res, N_res]
        attention_scores = torch.matmul(Y_q_attn, Y_k_attn.transpose(-2, -1))

        # 4. 缩放
        attention_scores = attention_scores / math.sqrt(Y_q_attn.size(-1))  # sqrt(l_max)

        return attention_scores

    def apply_rotation(self, features, rotation_matrix):
        """
        Apply the rotation matrix to the feature vectors.
        Args:
            features: [B, N, H, G, 3]
            rotation_matrix: [B, N, H, G, 3, 3]
        Returns:
            Rotated features: [B, N, H, G, 3]
        """
        # Step 1: Expand rotation_matrix to match features' shape
        # Adding two dimensions to rotation_matrix so that it can multiply with features
        rotation_matrix_expanded = rotation_matrix[:, :, None, None, :, :]  # [B, N, 1, 1, 3, 3]

        # Step 2: Perform batch matrix multiplication (rotation) with features
        rotated_features = torch.matmul(rotation_matrix_expanded, features.unsqueeze(-1))  # [B, N, H, G, 3, 1]

        # Remove the last dimension
        rotated_features = rotated_features.squeeze(-1)  # [B, N, H, G, 3]
        rot=Rotation(rotation_matrix)
        rf=rot[..., None, None].apply(features)


        return rotated_features
    def spherical_harmonic_expansion(self, features):
        """
        Perform spherical harmonic expansion on 3D vectors.
        Args:
            features: [B, N, H, G, 3]
        Returns:
            spherical_harmonics: [B, N, H, G, l_max]
        """
        # Perform spherical harmonics expansion on each 3D vector in the batch
        # Using e3nn's spherical_harmonics function to compute the spherical harmonic coefficients
        Y = spherical_harmonics(self.l_max, features, normalize="component")
        return Y


# Example usage
B, N, H, G = 2, 5, 8, 4  # Batch size, N_elements, N_heads, N_groups
l_max = [1,2,3]  # Max l value for spherical harmonics
group_size = 8  # Number of features per group (here 3 for 3D vector)

# Dummy tensors for q, k, and their rotations
q = torch.randn(B, N, H, G, 3)
k = torch.randn(B, N, H, G, 3)
r_q = torch.randn(B, N, 3, 3)  # Rotation matrix for q
r_k = torch.randn(B, N,  3, 3)  # Rotation matrix for k

# Initialize the attention module
attn_module = SphericalHarmonicClebschGordanAttention(l_max, H, group_size)

# Compute the output
output = attn_module(q, k, r_q, r_k)
print(output.shape)  # Expected shape: [B, N, H, G, 3]
