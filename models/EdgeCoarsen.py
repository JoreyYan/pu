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
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
        self.geo_rbf_bins = geo_rbf_bins
        self.geo_rbf_max = geo_rbf_max
        self.use_local_dir = use_local_dir
        self.use_sigma_stats = use_sigma_stats

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
        """
        if mask_f is not None:
            A = A * mask_f.unsqueeze(-1)
            # mask Z
            m2 = mask_f[:, :, None] * mask_f[:, None, :]
            Z = Z * m2.unsqueeze(-1)

        # numerator: [B,K,K,Cz]
        Z_num = torch.einsum("bna,bmj,bnmz->bamz", A, A, Z)
        # denom: [B,K,K]
        ones = torch.ones_like(Z[..., 0])
        Z_den = torch.einsum("bna,bmj,bnm->bam", A, A, ones).clamp_min(self.eps)
        Z_sem_c = Z_num / Z_den.unsqueeze(-1)
        return Z_sem_c

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

    def forward(self, A, Z_fine, r_c, mask_f=None, mask_c=None):
        Z_sem_c = self.coarsen_sem_dense(A, Z_fine, mask_f=mask_f)   # [B,K,K,c_z_in]
        Z_geo_c = self.geo_from_rc(r_c, mask_c=mask_c)               # [B,K,K,geo_dim]
        Z_cat = torch.cat([Z_sem_c, Z_geo_c], dim=-1)
        Z_c = self.fuse(Z_cat)
        if mask_c is not None:
            m2 = mask_c[:, :, None] * mask_c[:, None, :]
            Z_c = Z_c * m2.unsqueeze(-1)
        return Z_c, Z_sem_c, Z_geo_c
