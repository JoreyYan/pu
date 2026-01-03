import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import math

# -----------------------------
# utils: safe cholesky + mahalanobis + projection
# -----------------------------
def _safe_cholesky(A: torch.Tensor, jitter: float = 1e-6, max_tries: int = 4):
    I = torch.eye(3, device=A.device, dtype=A.dtype)
    for t in range(max_tries):
        try:
            return torch.linalg.cholesky(A + (jitter * (10 ** t)) * I)
        except RuntimeError:
            continue
    A = 0.5 * (A + A.transpose(-1, -2))
    return torch.linalg.cholesky(A + (jitter * (10 ** (max_tries - 1))) * I)

def _mahalanobis2(delta: torch.Tensor, Sigma: torch.Tensor, jitter: float = 1e-6):
    L = _safe_cholesky(Sigma, jitter=jitter)
    y = torch.linalg.solve_triangular(L, delta.unsqueeze(-1), upper=False)
    return (y.squeeze(-1) ** 2).sum(dim=-1)

def _project_inside_ellipsoid(mu_child, mu_parent, Sig_parent, tau2: float = 9.0, jitter: float = 1e-6):
    d = mu_child - mu_parent
    d2 = _mahalanobis2(d, Sig_parent, jitter=jitter)
    scale = torch.sqrt((tau2 / d2.clamp_min(1e-12))).clamp_max(1.0)
    return mu_parent + d * scale.unsqueeze(-1)

# -----------------------------
# allocation + sobol normal + repulsion
# -----------------------------
def _allocate_counts(pi: torch.Tensor, mask_parent: torch.Tensor, N: int,
                     min_per_parent: int = 1, max_per_parent: int = None):
    """
    pi: [B,K] normalized over valid parents
    """
    B, K = pi.shape
    device = pi.device
    valid = (mask_parent > 0.5)
    m = torch.zeros((B, K), device=device, dtype=torch.long)

    for b in range(B):
        idx = torch.nonzero(valid[b], as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        m[b, idx] = min_per_parent
        rem = N - int(min_per_parent * idx.numel())
        if rem < 0:
            m[b].zero_()
            m[b, idx[:N]] = 1
            continue
        pb = pi[b, idx]
        pb = pb / pb.sum().clamp_min(1e-9)
        add = torch.floor(pb * rem).to(torch.long)
        m[b, idx] += add
        rem2 = rem - int(add.sum().item())
        if rem2 > 0:
            frac = (pb * rem) - add.to(pb.dtype)
            _, order = torch.sort(frac, descending=True)
            m[b, idx[order[:rem2]]] += 1

        if max_per_parent is not None:
            m[b, idx] = torch.clamp(m[b, idx], max=int(max_per_parent))
            total = int(m[b, idx].sum().item())
            if total != N:
                diff = N - total
                if diff > 0:
                    _, order = torch.sort(pb, descending=True)
                    for t in range(diff):
                        m[b, idx[order[t % order.numel()]]] += 1
                else:
                    _, order = torch.sort(pb, descending=False)
                    t = 0
                    while diff < 0 and t < 10_000:
                        ksel = idx[order[t % order.numel()]]
                        if m[b, ksel] > min_per_parent:
                            m[b, ksel] -= 1
                            diff += 1
                        t += 1
    return m

def _sobol_normal(n: int, device, dtype, scramble: bool = True):
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=scramble)
    u = engine.draw(n).to(device=device, dtype=dtype).clamp(1e-6, 1 - 1e-6)
    z = torch.erfinv(2 * u - 1) * math.sqrt(2.0)
    return z

def _one_step_repulsion(mu: torch.Tensor, parent_idx: torch.Tensor, eta: float = 0.02, eps: float = 1e-4):
    B, N, _ = mu.shape
    mu2 = mu.clone()
    for b in range(B):
        pid = parent_idx[b]
        for k in pid.unique():
            sel = torch.nonzero(pid == k, as_tuple=False).squeeze(-1)
            if sel.numel() <= 1:
                continue
            x = mu2[b, sel]
            diff = x[:, None, :] - x[None, :, :]
            dist2 = (diff ** 2).sum(dim=-1) + eps
            dist2.fill_diagonal_(1e9)
            force = (diff / dist2[..., None]).sum(dim=1)
            mu2[b, sel] = x + eta * force
    return mu2

# -----------------------------
# your old-style tools (toy)
# -----------------------------
def sample_from_mixture(mu_p, Sig_p, pi, M, mask_parent, eps=1e-6):
    B, K, _ = mu_p.shape
    pi2 = pi.clone()
    if mask_parent is not None:
        pi2 = pi2 * (mask_parent > 0.5).to(pi2.dtype)
        pi2 = pi2 / pi2.sum(dim=-1, keepdim=True).clamp_min(eps)

    choices = torch.multinomial(pi2 + eps, M, replacement=True)  # [B,M]
    L = torch.linalg.cholesky(Sig_p + eps * torch.eye(3, device=Sig_p.device, dtype=Sig_p.dtype))

    z = torch.randn(B, M, 3, device=mu_p.device, dtype=mu_p.dtype)
    cand_x = torch.zeros(B, M, 3, device=mu_p.device, dtype=mu_p.dtype)
    for b in range(B):
        L_selected = L[b, choices[b]]
        mu_selected = mu_p[b, choices[b]]
        cand_x[b] = mu_selected + torch.bmm(L_selected, z[b].unsqueeze(-1)).squeeze(-1)
    return cand_x, choices

def fps_points_batch(x, n_points):
    B, M, _ = x.shape
    indices = torch.zeros(B, n_points, dtype=torch.long, device=x.device)
    for b in range(B):
        dist = torch.ones(M, device=x.device, dtype=x.dtype) * 1e10
        farthest = 0
        for i in range(n_points):
            indices[b, i] = farthest
            centroid = x[b, farthest, :].view(1, 3)
            d2 = torch.sum((x[b] - centroid) ** 2, dim=1)
            dist = torch.min(dist, d2)
            farthest = torch.max(dist, dim=0)[1]
    return indices

def init_sigma_from_child_spacing(mu0, node_mask, k_nn, alpha, sigma_floor, sigma_ceil):
    B, N, _ = mu0.shape
    dist2 = torch.sum((mu0[:, :, None, :] - mu0[:, None, :, :]) ** 2, dim=-1)
    val, _ = torch.topk(dist2, k=min(k_nn + 1, N), largest=False)
    avg_dist = torch.sqrt(val[:, :, 1:].mean(dim=-1).clamp_min(1e-12))
    sig_val = (avg_dist * alpha).clamp(sigma_floor, sigma_ceil)
    eye = torch.eye(3, device=mu0.device, dtype=mu0.dtype)
    Sig = sig_val[:, :, None, None] * eye
    Sig = Sig * node_mask[:, :, None, None]
    return Sig

# -----------------------------
# cover up-init (yours+mine)
# -----------------------------
@torch.no_grad()
def cover_upsample_init(
    mu_p, Sig_p, pi, mask_parent, node_mask,
    jitter=1e-6, tau2_inside=9.0, min_per_parent=1,
    sigma_floor=0.03, sigma_ceil=2.0,
    mix_cover_alpha=0.2, k_nn_spacing=4,
    repulse_eta=0.02,
):
    B, K, _ = mu_p.shape
    N = node_mask.shape[1]
    device, dtype = mu_p.device, mu_p.dtype
    I = torch.eye(3, device=device, dtype=dtype)[None, None]

    # (1) allocate
    m = _allocate_counts(pi, mask_parent, N, min_per_parent=min_per_parent)

    mu0 = torch.zeros((B, N, 3), device=device, dtype=dtype)
    Sig0_split = torch.zeros((B, N, 3, 3), device=device, dtype=dtype)
    parent_idx = torch.zeros((B, N), device=device, dtype=torch.long)

    for b in range(B):
        cursor = 0
        for k in range(K):
            if mask_parent[b, k] < 0.5:
                continue
            mk = int(m[b, k].item())
            if mk <= 0:
                continue
            z = _sobol_normal(mk, device=device, dtype=dtype, scramble=True)
            Lk = _safe_cholesky(Sig_p[b, k], jitter=jitter)
            x = mu_p[b, k].unsqueeze(0) + (z @ Lk.transpose(0, 1))
            x = _project_inside_ellipsoid(x, mu_p[b, k].unsqueeze(0), Sig_p[b, k].unsqueeze(0),
                                          tau2=tau2_inside, jitter=jitter)

            mu0[b, cursor:cursor + mk] = x
            parent_idx[b, cursor:cursor + mk] = k

            scale = float(max(mk, 1)) ** (2.0 / 3.0)
            Sig_child = Sig_p[b, k] / scale
            Sig0_split[b, cursor:cursor + mk] = Sig_child.unsqueeze(0).expand(mk, 3, 3)
            cursor += mk

        if cursor < N:
            kk = int(torch.nonzero(mask_parent[b] > 0.5, as_tuple=False)[0].item())
            mk = N - cursor
            z = _sobol_normal(mk, device=device, dtype=dtype, scramble=True)
            Lk = _safe_cholesky(Sig_p[b, kk], jitter=jitter)
            x = mu_p[b, kk].unsqueeze(0) + (z @ Lk.transpose(0, 1))
            x = _project_inside_ellipsoid(x, mu_p[b, kk].unsqueeze(0), Sig_p[b, kk].unsqueeze(0),
                                          tau2=tau2_inside, jitter=jitter)
            mu0[b, cursor:] = x
            parent_idx[b, cursor:] = kk
            Sig0_split[b, cursor:] = Sig_p[b, kk].unsqueeze(0).expand(mk, 3, 3)

    mu0 = mu0 * node_mask[..., None]

    if repulse_eta and repulse_eta > 0:
        mu0 = _one_step_repulsion(mu0, parent_idx, eta=float(repulse_eta))
        mu0 = mu0 * node_mask[..., None]

    Sig0_cover = init_sigma_from_child_spacing(
        mu0, node_mask, k_nn=int(k_nn_spacing),
        alpha=0.6, sigma_floor=float(sigma_floor), sigma_ceil=float(sigma_ceil)
    )
    Sig0_cover = Sig0_cover + jitter * I * node_mask[:, :, None, None]

    a = float(mix_cover_alpha)
    Sig0 = (1.0 - a) * Sig0_split + a * Sig0_cover
    Sig0 = 0.5 * (Sig0 + Sig0.transpose(-1, -2)) + jitter * I

    # clamp diag
    diag = torch.diagonal(Sig0, dim1=-2, dim2=-1)
    diag = diag.clamp_min(float(sigma_floor) ** 2).clamp_max(float(sigma_ceil) ** 2)
    Sig0 = Sig0.clone()
    Sig0[..., 0, 0] = diag[..., 0]
    Sig0[..., 1, 1] = diag[..., 1]
    Sig0[..., 2, 2] = diag[..., 2]
    return mu0, Sig0, parent_idx, m


# -----------------------------
# NEW: parent assignment for old methods (for fair visualization)
# -----------------------------
@torch.no_grad()
def assign_parent_by_maha(mu_child, mu_p, Sig_p, pi=None, jitter=1e-6):
    """
    hard assign each child to a parent (MAP under gaussian kernels).
    mu_child: [B,N,3], mu_p:[B,K,3], Sig_p:[B,K,3,3]
    """
    B, N, _ = mu_child.shape
    K = mu_p.shape[1]
    delta = mu_child[:, :, None, :] - mu_p[:, None, :, :]  # [B,N,K,3]
    # maha using parent covariance (cheap K small)
    maha = []
    for k in range(K):
        d = delta[:, :, k, :]
        maha_k = _mahalanobis2(d, Sig_p[:, k, :, :], jitter=jitter)  # [B,N]
        maha.append(maha_k)
    maha = torch.stack(maha, dim=-1)  # [B,N,K]
    logits = -0.5 * maha
    if pi is not None:
        logits = logits + torch.log(pi.clamp_min(1e-9))[:, None, :]
    pid = torch.argmax(logits, dim=-1)  # [B,N]
    return pid


# -----------------------------
# NEW: metrics (no voxel)
# -----------------------------
@torch.no_grad()
def coverage_rate(mu_child, parent_idx, mu_p, Sig_p, tau2=9.0, jitter=1e-6):
    """
    fraction of children inside their assigned parent's tau-sigma ellipsoid
    """
    B, N, _ = mu_child.shape
    ok_all = []
    for b in range(B):
        pid = parent_idx[b]
        mu_pb = mu_p[b]
        Sig_pb = Sig_p[b]
        # gather parent params per child
        mu_assigned = mu_pb[pid]           # [N,3]
        Sig_assigned = Sig_pb[pid]         # [N,3,3]
        d = mu_child[b] - mu_assigned
        d2 = _mahalanobis2(d, Sig_assigned, jitter=jitter)
        ok = (d2 <= tau2).to(mu_child.dtype)
        ok_all.append(ok.mean())
    return torch.stack(ok_all).mean().item()

@torch.no_grad()
def overlap_proxy(mu_child, Sig_child, parent_idx, jitter=1e-6, max_pairs=4096):
    """
    crude overlap proxy: average exp(-0.5 * maha in pooled sigma) for pairs within same parent.
    Larger -> more overlap/clumping. Smaller -> more separated.
    """
    B, N, _ = mu_child.shape
    vals = []
    for b in range(B):
        pid = parent_idx[b]
        # sample pairs
        idx = torch.arange(N, device=mu_child.device)
        if N * (N - 1) // 2 > max_pairs:
            # random pairs
            i = torch.randint(0, N, (max_pairs,), device=mu_child.device)
            j = torch.randint(0, N, (max_pairs,), device=mu_child.device)
        else:
            # full pairs
            ii, jj = torch.triu_indices(N, N, offset=1, device=mu_child.device)
            i, j = ii, jj
        same = (pid[i] == pid[j])
        if same.sum() == 0:
            vals.append(torch.tensor(0.0, device=mu_child.device))
            continue
        i, j = i[same], j[same]
        d = mu_child[b, i] - mu_child[b, j]  # [P,3]
        # pooled sigma (rough)
        Sig = Sig_child[b, i] + Sig_child[b, j]
        d2 = _mahalanobis2(d, Sig, jitter=jitter)
        score = torch.exp(-0.5 * d2).mean()
        vals.append(score)
    return torch.stack(vals).mean().item()

@torch.no_grad()
def density_fit_proxy(mu_p, Sig_p, pi, mu_child, parent_idx, jitter=1e-6):
    """
    proxy: child log-likelihood under parent mixture (MAP-assigned version)
    Higher is better (children sit in high density).
    """
    B, N, _ = mu_child.shape
    K = mu_p.shape[1]
    ll_all = []
    for b in range(B):
        pid = parent_idx[b]
        mu_pb, Sig_pb, pib = mu_p[b], Sig_p[b], pi[b]
        mu_assigned = mu_pb[pid]
        Sig_assigned = Sig_pb[pid]
        d = mu_child[b] - mu_assigned
        d2 = _mahalanobis2(d, Sig_assigned, jitter=jitter)
        # ignore normalization constants (compare relative)
        ll = (torch.log(pib[pid].clamp_min(1e-9)) - 0.5 * d2).mean()
        ll_all.append(ll)
    return torch.stack(ll_all).mean().item()


# -----------------------------
# visualization helpers
# -----------------------------
def get_ellipsoid_surface(mu, sig, n=10, scale=1.5):
    vals, vecs = torch.linalg.eigh(sig)
    radii = scale * torch.sqrt(torch.clamp(vals, min=1e-9)).cpu().numpy()
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ellipsoid = np.stack([x * radii[0], y * radii[1], z * radii[2]], axis=-1)
    ellipsoid = ellipsoid @ vecs.cpu().numpy().T + mu.cpu().numpy()
    return ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2]

def plot_case(ax, mu_p, Sig_p, mu_c, Sig_c, parent_idx, title, colors=None):
    B = mu_c.shape[0]
    assert B == 1, "toy plot assumes B=1"
    mu_p = mu_p[0]; Sig_p = Sig_p[0]
    mu_c = mu_c[0]; Sig_c = Sig_c[0]
    pid = parent_idx[0]

    K = mu_p.shape[0]
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(k) for k in range(K)]

    # Parents
    for k in range(K):
        X, Y, Z = get_ellipsoid_surface(mu_p[k], Sig_p[k], n=16, scale=2.5)
        ax.plot_wireframe(X, Y, Z, color=colors[k], alpha=0.10, linewidth=0.6)

    # Children (colored by assigned parent)
    for n in range(mu_c.shape[0]):
        k = int(pid[n].item())
        X, Y, Z = get_ellipsoid_surface(mu_c[n], Sig_c[n], n=8, scale=1.5)
        ax.plot_wireframe(X, Y, Z, color=colors[k], alpha=0.30, linewidth=0.7)

    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_zlim(-3, 3)
    ax.view_init(elev=35, azim=-45)


# -----------------------------
# main: 3-way comparison
# -----------------------------
@torch.no_grad()
def visualize_3d_comparison_three_way():
    B, K, N = 1, 2, 80
    dtype = torch.float32
    device = "cpu"

    # Parents: disk + cigar (same as your figure)
    mu_p = torch.tensor([[[-2.5, 0.0, 0.0], [2.5, 0.0, 0.0]]], dtype=dtype, device=device)
    Sig_p = torch.zeros(1, 2, 3, 3, dtype=dtype, device=device)
    Sig_p[0, 0] = torch.tensor([[4.0, 0.0, 0.0],
                                [0.0, 4.0, 0.0],
                                [0.0, 0.0, 0.1]], dtype=dtype, device=device)

    angle = np.pi / 4
    rot = torch.tensor([[np.cos(angle), -np.sin(angle), 0.0],
                        [np.sin(angle),  np.cos(angle), 0.0],
                        [0.0,            0.0,           1.0]], dtype=dtype, device=device)
    Sig_p[0, 1] = rot @ torch.diag(torch.tensor([5.0, 0.2, 0.2], dtype=dtype, device=device)) @ rot.T

    pi = torch.tensor([[0.5, 0.5]], dtype=dtype, device=device)
    mask_parent = torch.ones(1, 2, dtype=dtype, device=device)
    node_mask = torch.ones(1, N, dtype=dtype, device=device)

    # -------- Case A: Old (global mixture + FPS + isotropic spacing sigma)
    M = 6 * N
    cand_x, _ = sample_from_mixture(mu_p, Sig_p, pi, M, mask_parent)
    idx = fps_points_batch(cand_x, N)
    mu_A = cand_x.gather(1, idx[..., None].expand(B, N, 3))
    Sig_A = init_sigma_from_child_spacing(mu_A, node_mask, k_nn=4, alpha=0.6, sigma_floor=0.03, sigma_ceil=2.0)
    pid_A = assign_parent_by_maha(mu_A, mu_p, Sig_p, pi=pi)

    # -------- Case B: Old improved (per-parent quota sampling + FPS + isotropic spacing sigma)
    # Step1: ensure quota by sampling per parent (still isotropic sigma later)
    m = _allocate_counts(pi, mask_parent, N, min_per_parent=1)
    mu_B = torch.zeros_like(mu_A)
    pid_B = torch.zeros_like(pid_A)
    cur = 0
    for k in range(K):
        mk = int(m[0, k].item())
        z = torch.randn(mk, 3, dtype=dtype, device=device)
        Lk = _safe_cholesky(Sig_p[0, k])
        x = mu_p[0, k].unsqueeze(0) + (z @ Lk.T)
        x = _project_inside_ellipsoid(x, mu_p[0, k].unsqueeze(0), Sig_p[0, k].unsqueeze(0), tau2=9.0)
        mu_B[0, cur:cur+mk] = x
        pid_B[0, cur:cur+mk] = k
        cur += mk
    # then FPS within all quota points? (to keep same spirit, oversample within each parent then FPS)
    # simplest: add extra global candidates then FPS
    cand2, _ = sample_from_mixture(mu_p, Sig_p, pi, M, mask_parent)
    cand_all = torch.cat([mu_B, cand2], dim=1)  # [B, N+M,3]
    idx2 = fps_points_batch(cand_all, N)
    mu_B = cand_all.gather(1, idx2[..., None].expand(B, N, 3))
    Sig_B = init_sigma_from_child_spacing(mu_B, node_mask, k_nn=4, alpha=0.6, sigma_floor=0.03, sigma_ceil=2.0)
    pid_B = assign_parent_by_maha(mu_B, mu_p, Sig_p, pi=pi)

    # -------- Case C: New cover (split + fill, anisotropic sigma inherited)
    mu_C, Sig_C, pid_C, _ = cover_upsample_init(
        mu_p, Sig_p, pi, mask_parent, node_mask,
        mix_cover_alpha=0.2, sigma_floor=0.03, sigma_ceil=2.0, tau2_inside=9.0
    )

    # -------- Metrics
    def metrics(name, mu, Sig, pid):
        cov = coverage_rate(mu, pid, mu_p, Sig_p, tau2=9.0)
        ov  = overlap_proxy(mu, Sig, pid)
        ll  = density_fit_proxy(mu_p, Sig_p, pi, mu, pid)
        return f"{name}\ncoverage={cov:.3f}  overlap≈{ov:.3f}  ll≈{ll:.3f}"

    tA = metrics("A Old: GMM+FPS+IsoSigma", mu_A, Sig_A, pid_A)
    tB = metrics("B Old+Quota: quota+FPS+IsoSigma", mu_B, Sig_B, pid_B)
    tC = metrics("C New: Cover split+fill+AnisoSigma", mu_C, Sig_C, pid_C)

    # -------- Plot
    fig = plt.figure(figsize=(22, 7))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    cmap = plt.get_cmap("tab10")
    colors = [cmap(k) for k in range(K)]

    plot_case(ax1, mu_p, Sig_p, mu_A, Sig_A, pid_A, tA, colors=colors)
    plot_case(ax2, mu_p, Sig_p, mu_B, Sig_B, pid_B, tB, colors=colors)
    plot_case(ax3, mu_p, Sig_p, mu_C, Sig_C, pid_C, tC, colors=colors)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_3d_comparison_three_way()
