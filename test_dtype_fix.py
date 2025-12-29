"""
测试dtype修复是否有效
"""
import torch
import sys
sys.path.append('/home/junyu/project/pu')
sys.path.append('/home/junyu/project/pu/data')

from data.GaussianRigid import OffsetGaussianRigid
from openfold.utils.rigid_utils import Rotation

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 创建测试数据 (float32)
B, N = 2, 5
identity_rot = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(B, N, 3, 3)
rots = Rotation(rot_mats=identity_rot)
trans = torch.randn(B, N, 3, device=device, dtype=torch.float32)
scaling_log = torch.randn(B, N, 3, device=device, dtype=torch.float32)
local_mean = torch.randn(B, N, 3, device=device, dtype=torch.float32)

print("\n创建OffsetGaussianRigid...")
gaussian = OffsetGaussianRigid(rots, trans, scaling_log, local_mean)

print(f"  trans.dtype: {gaussian.get_trans().dtype}")
print(f"  scaling_log.dtype: {gaussian._scaling_log.dtype}")
print(f"  local_mean.dtype: {gaussian._local_mean.dtype}")

print("\n计算协方差矩阵...")
try:
    cov = gaussian.get_covariance()
    print(f"  ✓ 协方差矩阵计算成功")
    print(f"  cov.dtype: {cov.dtype}")
    print(f"  cov.shape: {cov.shape}")

    if cov.dtype == torch.float64:
        print(f"  ✗ 错误！dtype仍然是float64")
        sys.exit(1)
    elif cov.dtype == torch.float32:
        print(f"  ✓ 正确！dtype是float32")

    # 测试Cholesky分解
    print("\n测试Cholesky分解...")
    L = torch.linalg.cholesky(cov)
    print(f"  ✓ Cholesky分解成功")
    print(f"  L.dtype: {L.dtype}")

    if L.dtype != torch.float32:
        print(f"  ✗ 错误！Cholesky结果的dtype是{L.dtype}")
        sys.exit(1)

    # 测试log_det计算
    print("\n测试log_det计算...")
    log_det = torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1).mul_(2.0)
    print(f"  ✓ log_det计算成功")
    print(f"  log_det.dtype: {log_det.dtype}")
    print(f"  log_det值范围: [{log_det.min():.4f}, {log_det.max():.4f}]")

    if log_det.dtype != torch.float32:
        print(f"  ✗ 错误！log_det的dtype是{log_det.dtype}")
        sys.exit(1)

    print("\n" + "="*60)
    print("✓✓✓ 所有dtype检查通过！")
    print("="*60)

except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
