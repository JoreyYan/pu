import torch
import torch.nn as nn
import torch.nn.functional as F

class CoarseEdgeCoarsenAndFuse(nn.Module):
    """
    Build coarse pair embedding:
      Z_sem_c = A^T Z A
      Z_geo_c = geo(r_c)  (recomputed from coarse geometry)
      Z_c = fuse([Z_sem_c, Z_geo_c])
    """

    def __init__(
        self,
        c_z_in: int,          # input Z channel (fine)
        c_z_out: int,         # output coarse Z channel
        geo_rbf_bins: int = 16,
        geo_rbf_max: float = 10.0,   # in your geometry unit (nm if r in nm)
        use_local_dir: bool = False, # if you have stable rotations
        use_sigma_stats: bool = False,
            mode:str = "down",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.geo_rbf_bins = geo_rbf_bins
        self.geo_rbf_max = geo_rbf_max
        self.use_local_dir = use_local_dir
        self.use_sigma_stats = use_sigma_stats
        self.mode=mode

        geo_dim = geo_rbf_bins
        if use_local_dir:
            geo_dim += 3  # local direction (R_a^T (mu_b-mu_a)) normalized
        if use_sigma_stats:
            geo_dim += 6  # simple sigma invariants for pair: (logdet_a, logtr_a, aniso_a, logdet_b, logtr_b, aniso_b)

        self.fuse = nn.Sequential(
            nn.Linear(c_z_in + geo_dim, c_z_out),
            nn.LayerNorm(c_z_out),
            nn.SiLU(),
            nn.Linear(c_z_out, c_z_out),
        )

    @staticmethod
    def _rbf(d, bins, dmax, eps=1e-8):
        # d: [...], produce [..., bins]
        # centers in [0, dmax]
        centers = torch.linspace(0.0, dmax, bins, device=d.device, dtype=d.dtype)
        # width: spacing
        if bins > 1:
            width = (dmax / (bins - 1))
        else:
            width = dmax
        width = max(float(width), 1e-6)
        return torch.exp(-0.5 * ((d.unsqueeze(-1) - centers) / width) ** 2)

    @staticmethod
    def _sigma_invariants(S, eps=1e-8):
        # S: [...,3,3] SPD-ish
        # invariants: logdet, logtrace, aniso ~ log(lambda_max) - log(lambda_min)
        # avoid eigvalsh instability: use cholesky for logdet; trace is easy; aniso use approx via diagonal ratio (cheap) or eig if stable
        S = 0.5 * (S + S.transpose(-1, -2))
        I = torch.eye(3, device=S.device, dtype=S.dtype)
        S = S + 1e-6 * I
        L = torch.linalg.cholesky(S)
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-12)).sum(dim=-1)
        tr = torch.diagonal(S, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(eps)
        logtr = torch.log(tr)

        # cheap anisotropy proxy: log(max(diag)) - log(min(diag))
        diag = torch.diagonal(S, dim1=-2, dim2=-1).clamp_min(eps)
        aniso = torch.log(diag.max(dim=-1).values) - torch.log(diag.min(dim=-1).values)
        return logdet, logtr, aniso

    def coarsen_sem_dense(self, A, Z, mask_f=None):
        """
        Z_sem_c = (A^T Z A) / (A^T 1 A)
        A: [B,N,K], Z: [B,N,N,Cz]
        return: [B,K,K,Cz]
        """
        if mask_f is not None:
            A = A * mask_f.unsqueeze(-1)  # [B,N,K]
            m2 = mask_f[:, :, None] * mask_f[:, None, :]  # [B,N,N]
            Z = Z * m2.unsqueeze(-1)

        # numerator: [B,K,K,Cz]
        Z_num = torch.einsum("bik,bijc,bjl->bklc", A, Z, A)

        # denom: [B,K,K]
        ones = torch.ones_like(Z[..., 0])  # [B,N,N]
        Z_den = torch.einsum("bik,bij,bjl->bkl", A, ones, A).clamp_min(self.eps)

        return Z_num / Z_den.unsqueeze(-1)

    def lift_sem_dense(self, w, Z_parent, mask_child=None, mask_parent=None):
        """
        Z_child = (w Z_parent w^T) / (w 1 w^T)
        w: [B,N,K]  child<-parent
        Z_parent: [B,K,K,Cz]
        return: [B,N,N,Cz]
        """
        if mask_child is not None:
            w = w * mask_child.unsqueeze(-1)
        if mask_parent is not None:
            Z_parent = Z_parent * (mask_parent[:, :, None] * mask_parent[:, None, :]).unsqueeze(-1)

        Z_num = torch.einsum("bik,bklc,bjl->bijc", w, Z_parent, w)

        ones = torch.ones_like(Z_parent[..., 0])  # [B,K,K]
        Z_den = torch.einsum("bik,bkl,bjl->bij", w, ones, w).clamp_min(self.eps)

        Z_child = Z_num / Z_den.unsqueeze(-1)

        if mask_child is not None:
            m2 = mask_child[:, :, None] * mask_child[:, None, :]
            Z_child = Z_child * m2.unsqueeze(-1)

        return Z_child

    def geo_from_rc(self, r_c, mask_c=None):
        """
        Build Z_geo_c: [B,K,K,geo_dim] from coarse geometry.
        Uses mu_c, and optionally rotations / sigma stats.
        """
        mu_c = r_c.get_gaussian_mean()  # [B,K,3]
        B, K, _ = mu_c.shape
        # pairwise delta: [B,K,K,3]
        delta = mu_c[:, :, None, :] - mu_c[:, None, :, :]
        d = torch.sqrt((delta ** 2).sum(dim=-1) + self.eps)  # [B,K,K]

        rbf = self._rbf(d, self.geo_rbf_bins, self.geo_rbf_max, eps=self.eps)  # [B,K,K,Bins]
        feats = [rbf]

        if self.use_local_dir:
            # if your r_c has rotations: local_dir = R_a^T (mu_b - mu_a)
            # NOTE: you must adapt these two lines to your Rotation class API
            R = r_c.get_rotation_mats()  # [B,K,3,3]  <-- you need to implement/get this
            # compute dir in a-local frame for each (a,b)
            # dir_ab = R_a^T (mu_b - mu_a)
            dir_ab = torch.einsum("bkij,bkkj->bkki", R.transpose(-1, -2), (-delta))  # [B,K,K,3]
            dir_ab = F.normalize(dir_ab, dim=-1, eps=self.eps)
            feats.append(dir_ab)

        if self.use_sigma_stats:
            S = r_c.get_covariance()  # [B,K,3,3]
            logdet, logtr, aniso = self._sigma_invariants(S, eps=self.eps)  # each [B,K]
            # expand to pair
            a = torch.stack([logdet, logtr, aniso], dim=-1)  # [B,K,3]
            a_i = a[:, :, None, :].expand(B, K, K, 3)
            a_j = a[:, None, :, :].expand(B, K, K, 3)
            feats.append(torch.cat([a_i, a_j], dim=-1))  # [B,K,K,6]

        Z_geo_c = torch.cat(feats, dim=-1)  # [B,K,K,geo_dim]

        if mask_c is not None:
            m2 = mask_c[:, :, None] * mask_c[:, None, :]
            Z_geo_c = Z_geo_c * m2.unsqueeze(-1)

        return Z_geo_c

    def forward(self, A, Z_in, r_target, mask_f=None, mask_c=None):
        """
        A: [B, N, K] 分配矩阵
        Z_in: 输入的 pair 特征
        r_target: 目标层级的 GaussianRigid
        mask_f: [B, N] (Fine 层的 mask)
        mask_c: [B, K] (Coarse 层的 mask)
        """
        if self.mode == 'down':
            # --- 下采样: N -> K ---
            # A: [B, N, K], Z_in: [B, N, N, C]
            # 聚合公式: Z_out = A.T @ Z_in @ A
            Z_sem_out = self.coarsen_sem_dense(A, Z_in, mask_f=mask_f)
            # 此时目标是 Coarse 层，用 mask_c
            Z_geo_out = self.geo_from_rc(r_target, mask_c=mask_c)
            current_mask = mask_c
        else:
            # --- 上采样: K -> N ---
            # A: [B, N, K], Z_in: [B, K, K, C]
            # 分发公式: Z_out = A @ Z_in @ A.T
            Z_sem_out = self.lift_sem_dense(A, Z_in, mask_parent=mask_c)
            # 此时目标是 Fine 层，用 mask_f
            Z_geo_out = self.geo_from_rc(r_target, mask_c=mask_f)
            current_mask = mask_f

        Z_cat = torch.cat([Z_sem_out, Z_geo_out], dim=-1)
        Z_out = self.fuse(Z_cat)

        if current_mask is not None:
            # 2D Mask: [B, L, L, 1]
            m2 = current_mask[:, :, None] * current_mask[:, None, :]
            Z_out = Z_out * m2.unsqueeze(-1)

        return Z_out, Z_sem_out, Z_geo_out
