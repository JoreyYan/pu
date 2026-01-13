import torch

def _safe_norm(v: torch.Tensor, eps: float = 1e-12):
    return torch.sqrt(torch.clamp((v * v).sum(dim=-1, keepdim=True), min=eps))

def _orthonormal_basis_from_v1(v1: torch.Tensor, eps: float = 1e-12):
    """
    给定单位向量 v1，构造与其正交的 v2,v3（右手系）。
    用一个与 v1 最不平行的坐标轴作为锚点，避免 cross 接近 0。
    """
    # 选择与 v1 最不平行的轴：|dot(v1, e_i)| 最小的那个
    # dot with ex, ey, ez is just abs(component)
    ax = v1[..., 0].abs()
    ay = v1[..., 1].abs()
    az = v1[..., 2].abs()

    # choose the smallest component axis as anchor
    # if x is smallest -> use ex, else if y smallest -> ey, else ez
    use_ex = (ax <= ay) & (ax <= az)
    use_ey = (~use_ex) & (ay <= az)
    # else use ez
    ex = torch.tensor([1.0, 0.0, 0.0], device=v1.device, dtype=v1.dtype).expand_as(v1)
    ey = torch.tensor([0.0, 1.0, 0.0], device=v1.device, dtype=v1.dtype).expand_as(v1)
    ez = torch.tensor([0.0, 0.0, 1.0], device=v1.device, dtype=v1.dtype).expand_as(v1)

    a = torch.where(use_ex[..., None], ex, torch.where(use_ey[..., None], ey, ez))

    v2 = torch.cross(v1, a, dim=-1)
    v2 = v2 / _safe_norm(v2, eps=eps)
    v3 = torch.cross(v1, v2, dim=-1)
    v3 = v3 / _safe_norm(v3, eps=eps)
    return v2, v3

def sym3_eigvals_closed_form(S: torch.Tensor, eps: float = 1e-12):
    """
    对称 3x3 矩阵特征值闭式解（三角公式）。
    输入 S: [...,3,3] (应为对称)
    输出 w: [...,3] (降序)
    """
    assert S.shape[-2:] == (3, 3)

    # trace / mean
    tr = S[..., 0, 0] + S[..., 1, 1] + S[..., 2, 2]
    q = tr / 3.0

    # B = S - qI
    B00 = S[..., 0, 0] - q
    B11 = S[..., 1, 1] - q
    B22 = S[..., 2, 2] - q
    B01 = S[..., 0, 1]
    B02 = S[..., 0, 2]
    B12 = S[..., 1, 2]

    # tr(B^2) for symmetric: sum diag^2 + 2*sum offdiag^2
    trB2 = (B00 * B00 + B11 * B11 + B22 * B22) + 2.0 * (B01 * B01 + B02 * B02 + B12 * B12)
    p2 = trB2 / 6.0
    p = torch.sqrt(torch.clamp(p2, min=eps))

    # near-spherical: p ~ 0 => all eigenvalues ~ q
    spherical = p < 1e-6  # 这个阈值在 float32 下通常够用

    # C = (1/p) B, but compute det(C) robustly
    invp = 1.0 / p
    c00 = B00 * invp
    c11 = B11 * invp
    c22 = B22 * invp
    c01 = B01 * invp
    c02 = B02 * invp
    c12 = B12 * invp

    # det(C) for symmetric 3x3:
    # |c00 c01 c02|
    # |c01 c11 c12|
    # |c02 c12 c22|
    detC = (
        c00 * (c11 * c22 - c12 * c12)
        - c01 * (c01 * c22 - c12 * c02)
        + c02 * (c01 * c12 - c11 * c02)
    )

    r = detC * 0.5
    # 用 atan2 形式比 acos 更抗数值误差：acos(r)=atan2(sqrt(1-r^2), r)
    r = torch.clamp(r, -1.0 + 1e-7, 1.0 - 1e-7)
    s = torch.sqrt(torch.clamp(1.0 - r * r, min=0.0))
    phi = torch.atan2(s, r) / 3.0

    two_p = 2.0 * p
    lam1 = q + two_p * torch.cos(phi)
    lam2 = q + two_p * torch.cos(phi + 2.0 * torch.pi / 3.0)
    lam3 = q + two_p * torch.cos(phi + 4.0 * torch.pi / 3.0)

    w = torch.stack([lam1, lam2, lam3], dim=-1)

    # spherical fallback
    if spherical.any():
        w = torch.where(spherical[..., None], q[..., None].expand_as(w), w)

    # sort descending
    w, _ = torch.sort(w, dim=-1, descending=True)
    return w

def eigvec_from_lambda_robust(S: torch.Tensor, lam: torch.Tensor, eps: float = 1e-12):
    """
    给定对称 3x3 S 和特征值 lam，构造一个特征向量（单位化），不依赖 eigh。
    方法：A = S - lam I，对 A 的行向量做叉乘找零空间方向；简并时 fallback。
    """
    assert S.shape[-2:] == (3, 3)

    I = torch.eye(3, device=S.device, dtype=S.dtype)
    A = S - lam[..., None, None] * I

    r0 = A[..., 0, :]
    r1 = A[..., 1, :]
    r2 = A[..., 2, :]

    c01 = torch.cross(r0, r1, dim=-1)
    c02 = torch.cross(r0, r2, dim=-1)
    c12 = torch.cross(r1, r2, dim=-1)

    n01 = (c01 * c01).sum(dim=-1)
    n02 = (c02 * c02).sum(dim=-1)
    n12 = (c12 * c12).sum(dim=-1)

    n_stack = torch.stack([n01, n02, n12], dim=-1)
    val, idx = torch.max(n_stack, dim=-1)

    v = torch.where((idx == 0)[..., None], c01,
         torch.where((idx == 1)[..., None], c02, c12))

    # 简并/近简并：叉乘都很小 => v ~ 0
    deg = val < (1e-10)  # 对 float32 的经验阈值；你也可按场景调
    if deg.any():
        # fallback：选一个固定非零向量即可（后面 GS / basis 会处理正交）
        fb = torch.zeros_like(v)
        fb[..., 0] = 1.0
        v = torch.where(deg[..., None], fb, v)

    v = v / _safe_norm(v, eps=eps)
    return v

def cov_to_R_scale_no_eigh_robust(
    Sigma: torch.Tensor,
    jitter: float = 1e-6,
    eps: float = 1e-12,
):
    """
    从协方差 Sigma 得到：
      Sigma = R diag(eigvals) R^T
      scale = sqrt(eigvals)
    不使用 torch.linalg.eigh。

    Sigma: [...,3,3] (建议 PSD；内部会对称化 + jitter)
    返回:
      R:     [...,3,3]  (正交，det=+1，列为主轴方向，按大->小)
      scale: [...,3]    (半轴尺度 = sqrt(eigvals)，按大->小)
      eigvals: [...,3]  (按大->小，clamp>=eps)
    """
    assert Sigma.shape[-2:] == (3, 3)

    orig_dtype = Sigma.dtype
    # 半精度下，闭式 trig/sqrt 建议 float32 算
    work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
    S = Sigma.to(work_dtype)

    # 1) 对称化 + jitter
    S = 0.5 * (S + S.transpose(-1, -2))
    I = torch.eye(3, device=S.device, dtype=work_dtype)
    S = S + jitter * I

    # 2) 闭式特征值（降序）
    w = sym3_eigvals_closed_form(S, eps=eps)
    w = torch.clamp(w, min=eps)
    scale = torch.sqrt(w)

    # 3) 特征向量（v1,v2），并做稳健正交化
    v1 = eigvec_from_lambda_robust(S, w[..., 0], eps=eps)
    v2 = eigvec_from_lambda_robust(S, w[..., 1], eps=eps)

    # Gram-Schmidt: v2 <- v2 - proj(v2,v1)*v1
    v2 = v2 - (v2 * v1).sum(dim=-1, keepdim=True) * v1
    v2n = (v2 * v2).sum(dim=-1)  # [...]
    bad_v2 = v2n < 1e-10  # 简并导致 v2 ~ 0

    if bad_v2.any():
        v2_fb, v3_fb = _orthonormal_basis_from_v1(v1, eps=eps)
        v2 = torch.where(bad_v2[..., None], v2_fb, v2)
        v2 = v2 / _safe_norm(v2, eps=eps)
        v3 = torch.where(bad_v2[..., None], v3_fb, torch.cross(v1, v2, dim=-1))
    else:
        v2 = v2 / _safe_norm(v2, eps=eps)
        v3 = torch.cross(v1, v2, dim=-1)

    v3 = v3 / _safe_norm(v3, eps=eps)

    # 4) 组装 R（列向量为主轴），并强制右手系 det=+1
    R = torch.stack([v1, v2, v3], dim=-1)  # [...,3,3]
    det = torch.det(R)
    flip = (det < 0).to(work_dtype)[..., None, None]
    R = torch.cat([R[..., :, 0:2], R[..., :, 2:3] * (1.0 - 2.0 * flip)], dim=-1)

    # cast 回原 dtype
    R = R.to(orig_dtype)
    scale = scale.to(orig_dtype)
    w = w.to(orig_dtype)
    return R, scale, w
