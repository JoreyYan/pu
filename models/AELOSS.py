import torch
import torch.nn.functional as F
from models.finnalup import extract_parent_params

def compute_xi_gt_from_parents(
    x_ca: torch.Tensor,        # [B,N,3]
    node_mask: torch.Tensor,   # [B,N]
    a_idx: torch.Tensor,       # [B,N]
    r_parent,                  # OffsetGaussianRigid [B,K]
    eps: float = 1e-8,
):
    """
    xi_gt = R^T (x - mu) / s
    这里假设你有 extract_parent_params(r_parent)->(mu_k,R_k,s_k)
    """
    B, N, _ = x_ca.shape
    dtype = x_ca.dtype
    m = node_mask.to(dtype=dtype)

    mu_k, R_k, s_k = extract_parent_params(r_parent)  # [B,K,3], [B,K,3,3], [B,K,3]
    mu_i = mu_k.gather(1, a_idx[..., None].expand(B, N, 3))
    s_i  = s_k.gather(1, a_idx[..., None].expand(B, N, 3))
    R_i  = R_k.gather(1, a_idx[..., None, None].expand(B, N, 3, 3))

    delta = (x_ca - mu_i) * m.unsqueeze(-1)
    local = torch.einsum("bnji,bnj->bni", R_i, delta)   # R^T (x-mu)
    xi_gt = local / s_i.clamp_min(eps)
    xi_gt = xi_gt * m.unsqueeze(-1)
    return xi_gt
def pairwise_dist(x: torch.Tensor, mask: torch.Tensor | None = None, eps: float = 1e-8):
    """
    x:    [B,N,3]
    mask: [B,N] 0/1
    return:
      d:   [B,N,N]  (euclidean distance)
      m2:  [B,N,N]  pair mask
    """
    # squared distance
    # (x_i - x_j)^2 = x^2_i + x^2_j - 2 x_i·x_j
    x2 = (x * x).sum(dim=-1, keepdim=True)  # [B,N,1]
    dist2 = x2 + x2.transpose(1, 2) - 2.0 * torch.matmul(x, x.transpose(1, 2))
    dist2 = dist2.clamp_min(0.0)
    d = torch.sqrt(dist2 + eps)

    if mask is None:
        m2 = None
    else:
        m = mask.to(dtype=x.dtype)
        m2 = (m[:, :, None] * m[:, None, :])  # [B,N,N]
    return d, m2

# ============================================================
# 2) break loss：BCE +（可选）链断点强约束
# ============================================================
def break_supervision_loss(
    break_logits: torch.Tensor,     # [B,N-1]
    break_gt: torch.Tensor,         # [B,N-1] 0/1
    break_mask: torch.Tensor,       # [B,N-1] 0/1
    pos_weight: float = 3.0,        # teacher break 通常偏少，给正样本加权更稳
    eps: float = 1e-8,
):
    """
    BCEWithLogitsLoss with mask.
    """
    # 手写 masked BCE
    # w_pos 只对正类加权：pos_weight
    w = torch.ones_like(break_gt)
    w = w + (pos_weight - 1.0) * break_gt

    loss = F.binary_cross_entropy_with_logits(
        break_logits, break_gt, weight=w, reduction="none"
    )
    denom = break_mask.sum().clamp_min(eps)
    return (loss * break_mask).sum() / denom


def segmenter_aux_losses(
    break_logits: torch.Tensor,        # [B,N-1]
    a_idx_teacher: torch.Tensor,       # [B,N]
    node_mask: torch.Tensor,           # [B,N]
    chain_idx: torch.Tensor | None,    # [B,N]
    pos_weight: float = 3.0,
):
    break_gt, break_mask = breaks_from_a_idx(a_idx_teacher, node_mask, chain_idx)
    loss_break = break_supervision_loss(
        break_logits=break_logits,
        break_gt=break_gt,
        break_mask=break_mask,
        pos_weight=pos_weight,
    )
    metrics = {
        "break_sup_loss": loss_break.detach(),
        "break_gt_rate": (break_gt.sum() / break_mask.sum().clamp_min(1.0)).detach(),
        "break_pred_rate": (
            ((break_logits > 0.0).float() * break_mask).sum() / break_mask.sum().clamp_min(1.0)
        ).detach(),
    }
    return loss_break, metrics


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None, eps: float = 1e-8):
    if mask is None:
        return x.mean()
    denom = mask.sum().clamp_min(eps)
    return (x * mask).sum() / denom


def seg_up_losses_angstrom(
    out: dict,
    x_ca_gt_A: torch.Tensor,        # [B,N,3] GT CA in Å
    node_mask: torch.Tensor,        # [B,N] 0/1
    *,
    # weights
    w_ca_mse: float = 1.0,
    w_pair: float = 0.2,
    w_xi_reg: float = 0.0,
    w_xi_sup: float = 0.1,
    # pair config
    pair_mode: str = "dist_l1",     # "dist_l1" or "dist_mse"
    pair_k: int | None = 32,
    # xi
    xi_gt: torch.Tensor | None = None,  # [B,N,3] optional
    # unit conversion
    x_hat_is_nm: bool = True,       # 你的 out["x_hat"] 目前是 nm
    nm_to_A: float = 10.0,
):
    """
    out needs:
      out["x_hat"]  [B,N,3] (nm or Å)
      out["xi_hat"] [B,N,3]
    """
    x_hat = out["x_hat"]
    xi_hat = out["xi_hat"]
    m = node_mask.to(dtype=x_hat.dtype)

    # --- convert x_hat to Å if needed ---
    if x_hat_is_nm:
        x_hat_A = x_hat * nm_to_A
    else:
        x_hat_A = x_hat

    # -------- (1) CA point MSE in Å --------
    ca_mse = ((x_hat_A - x_ca_gt_A) ** 2).sum(dim=-1)  # [B,N]
    ca_mse = masked_mean(ca_mse, m)

    # RMSD in Å (直接就是 Å)
    ca_rmsd_A = torch.sqrt(ca_mse.clamp_min(0.0))

    # -------- (2) pairwise distance loss in Å --------
    pair_loss = x_hat_A.new_zeros(())

    if w_pair > 0:
        with torch.no_grad():
            d_gt, _ = pairwise_dist(x_ca_gt_A, mask=None)  # [B,N,N]
            B, N, _ = d_gt.shape
            eye = torch.eye(N, device=d_gt.device, dtype=d_gt.dtype)[None]
            d_gt = d_gt + eye * 1e6

            if pair_k is not None and pair_k < N:
                nn_idx = torch.topk(d_gt, k=pair_k, dim=-1, largest=False).indices  # [B,N,k]
            else:
                nn_idx = None

        d_hat, _ = pairwise_dist(x_hat_A, mask=None)  # [B,N,N]

        if nn_idx is None:
            diff = d_hat - d_gt
            m2 = (node_mask[:, :, None] * node_mask[:, None, :]).to(dtype=diff.dtype)
            if pair_mode == "dist_mse":
                pair_loss = masked_mean(diff * diff, m2)
            else:
                pair_loss = masked_mean(diff.abs(), m2)
        else:
            d_hat_ik = d_hat.gather(-1, nn_idx)  # [B,N,k]
            d_gt_ik = d_gt.gather(-1, nn_idx)    # [B,N,k]
            diff = d_hat_ik - d_gt_ik

            m_i = node_mask.to(dtype=diff.dtype).unsqueeze(-1)  # [B,N,1]
            m_j = node_mask.to(dtype=diff.dtype).gather(1, nn_idx.reshape(B, -1)).reshape(B, N, -1)  # [B,N,k]
            m_ik = m_i * m_j

            if pair_mode == "dist_mse":
                pair_loss = masked_mean(diff * diff, m_ik)
            else:
                pair_loss = masked_mean(diff.abs(), m_ik)

    # -------- (3) xi loss (optional supervise) + reg --------
    xi_sup_loss = x_hat_A.new_zeros(())
    if (xi_gt is not None) and (w_xi_sup > 0):
        xi_sup_loss = ((xi_hat - xi_gt) ** 2).sum(dim=-1)  # [B,N]
        xi_sup_loss = masked_mean(xi_sup_loss, m)

    xi_reg = x_hat_A.new_zeros(())
    if w_xi_reg > 0:
        denom = m.sum().clamp_min(1.0)
        xi = xi_hat * m.unsqueeze(-1)
        mean = xi.sum(dim=(0, 1)) / denom          # [3]
        var = (xi * xi).sum(dim=(0, 1)) / denom - mean * mean
        xi_reg = (mean * mean).mean() + ((var - 1.0) ** 2).mean()

    total = (
        w_ca_mse * ca_mse +
        w_pair * pair_loss +
        w_xi_reg * xi_reg +
        w_xi_sup * xi_sup_loss
    )

    metrics = {
        "loss": total,
        "loss_ca_mse_A2": ca_mse.detach(),     # 单位 Å^2 (每点的平方和再均值)
        "ca_rmsd_A": ca_rmsd_A.detach(),       # 单位 Å
        "loss_pair_A": pair_loss.detach(),     # 单位 Å 或 Å^2（看 pair_mode）
        "loss_xi_reg": xi_reg.detach(),
        "loss_xi_sup": xi_sup_loss.detach(),
        "xi_abs_mean": xi_hat.abs().mean().detach(),
        "xi_abs_max": xi_hat.abs().max().detach(),
    }
    return  metrics


def segment_intra_pair_loss(
    x_hat: torch.Tensor,
    x_gt: torch.Tensor,
    node_mask: torch.Tensor,
    a_idx: torch.Tensor,
    clamp: float | None = None,
    eps: float = 1e-8,
):
    B, N, _ = x_hat.shape
    dtype = x_hat.dtype
    device = x_hat.device

    m = node_mask.to(dtype=dtype)
    pair_mask = m[:, :, None] * m[:, None, :]
    same_seg = (a_idx[:, :, None] == a_idx[:, None, :]).to(dtype=dtype)
    pair_mask = pair_mask * same_seg

    eye = torch.eye(N, device=device, dtype=dtype)[None]
    pair_mask = pair_mask * (1.0 - eye)

    D_hat = torch.cdist(x_hat, x_hat, p=2)
    D_gt  = torch.cdist(x_gt,  x_gt,  p=2)
    if clamp is not None:
        D_hat = D_hat.clamp_max(clamp)
        D_gt  = D_gt.clamp_max(clamp)

    err2 = (D_hat - D_gt) ** 2
    denom = pair_mask.sum().clamp_min(eps)
    return (err2 * pair_mask).sum() / denom

def seg_ae_losses(
    *,
    x_hat: torch.Tensor,        # [B,N,3]
    x_gt: torch.Tensor,         # [B,N,3]
    xi_hat: torch.Tensor,       # [B,N,3]
    node_mask: torch.Tensor,    # [B,N]
    a_idx: torch.Tensor,        # [B,N]
    r_parent,
    w_x: float = 1.0,
    w_xi: float = 0.0,
    w_pair_intra: float = 1.0,
    clamp_pair: float | None = 50.0,  # Å 上常用 clamp，防长距离主导
    eps: float = 1e-8,
        **kwargs
):
    dtype = x_hat.dtype
    m = node_mask.to(dtype=dtype)

    # (1) x mse
    x_mse = ((x_hat - x_gt) ** 2).sum(dim=-1)
    x_mse = (x_mse * m).sum() / m.sum().clamp_min(eps)

    # (2) xi supervised (from GT)
    xi_gt = compute_xi_gt_from_parents(x_gt, node_mask, a_idx, r_parent, eps=eps)
    xi_sup = ((xi_hat - xi_gt) ** 2).sum(dim=-1)
    xi_sup = (xi_sup * m).sum() / m.sum().clamp_min(eps)

    # (3) intra-seg pair loss
    pair_intra = segment_intra_pair_loss(x_hat, x_gt, node_mask, a_idx, clamp=clamp_pair, eps=eps)

    total = w_x * x_mse + w_xi * xi_sup + w_pair_intra * pair_intra

    return {
        "loss": total,
        "x_mse": x_mse.detach(),
        "xi_sup": xi_sup.detach(),
        "pair_intra": pair_intra.detach(),
        "xi_gt_max":xi_gt.detach().max()
    }