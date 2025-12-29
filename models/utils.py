import math
import torch
from torch.nn import functional as F
import numpy as np
from data import utils as du
from typing import Callable, Literal, Optional, Tuple, Union
from torch import nn


class RelativePosition(nn.Module):
    def __init__(self, bins=32, pairwise_state_dim=128):
        super().__init__()
        self.bins = bins

        # Note an additional offset is used so that the 0th position
        # is reserved for masked pairs.
        self.embedding = torch.nn.Embedding(2 * bins + 2, pairwise_state_dim)

    def forward(self, residue_index, mask=None):
        """
        Input:
          residue_index: B x L tensor of indices (dytpe=torch.long)
          mask: B x L tensor of booleans

        Output:
          pairwise_state: B x L x L x pairwise_state_dim tensor of embeddings
        """

        assert residue_index.dtype == torch.long
        if mask is not None:
            assert residue_index.shape == mask.shape

        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # Add 1 to adjust for padding index.

        if mask is not None:
            mask = mask[:, None, :] * mask[:, :, None]
            diff[mask == False] = 0

        output = self.embedding(diff)
        return output
def expand(x, tgt=None, dim=1):
    if tgt is None:
        for _ in range(dim):
            x = x[..., None]
    else:
        while len(x.shape) < len(tgt.shape):
            x = x[..., None]
    return x
def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb



def pad_input_feats_like(x_, input_feats, ref_key="res_mask"):
    """
    Pad input_feats to match x_ along dim=1 (sequence length N),
    using the shape of input_feats[ref_key] to infer M.

    Args:
        x_ : [B, N, D] target tensor
        input_feats : dict of [B, M, ...] or [B, M] tensors
        ref_key : key used to infer original M

    Returns:
        padded_feats : dict of [B, N, ...] or [B, N] tensors
    """
    B, N, *_ = x_.shape
    assert ref_key in input_feats, f"{ref_key} not found in input_feats"

    ref_feat = input_feats[ref_key]
    M = ref_feat.shape[1]  # 第二维度是 M

    pad_len = N - M
    if pad_len < 0:
        raise ValueError(f"N={N} is smaller than M={M} from {ref_key}")
    elif pad_len == 0:
        return input_feats,-1
    else:

        padded_feats = {}
        for key, feat in input_feats.items():
            if feat.shape[1] != M:
                continue

            pad_shape = list(feat.shape)
            pad_shape[1] = pad_len
            pad_tensor = torch.zeros(*pad_shape, dtype=feat.dtype, device=feat.device)
            feat_padded = torch.cat([feat, pad_tensor], dim=1)
            padded_feats[key] = feat_padded

        return padded_feats,M

def t_stratified_loss(batch_t, batch_loss, num_bins=4, loss_name=None):
    """Stratify loss by binning t."""
    batch_t = du.to_numpy(batch_t)
    batch_loss = du.to_numpy(batch_loss)
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins+1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = 'loss'
    for t_bin in np.unique(bin_idx).tolist():
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin+1]
        t_range = f'{loss_name} t=[{bin_start:.2f},{bin_end:.2f})'
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses
def t_stratified_mean_loss(batch_t, batch_loss, loss_name=None, bins=None):
    """
    Compute mean loss in different t-bins.
    """
    batch_t = du.to_numpy(batch_t)
    batch_loss = du.to_numpy(batch_loss)

    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()

    if loss_name is None:
        loss_name = 'loss'

    if bins is None:
        bins = [0.0, 0.25, 0.5, 0.75, 1.0]

    stratified_losses = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i < len(bins) - 2:
            mask = (flat_t >= lo) & (flat_t < hi)
        else:
            mask = (flat_t >= lo) & (flat_t <= hi)
        if mask.any():
            stratified_losses[f'{loss_name}_{lo:.2f}_{hi:.2f}'] = float(np.mean(flat_losses[mask]))
    stratified_losses[f'{loss_name}_mean'] = float(np.mean(flat_losses))
    return stratified_losses


def compute_fape(
        pred_frames: du.Rigid,
        target_frames: du.Rigid,
        frames_mask: torch.Tensor,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        positions_mask: torch.Tensor,
        length_scale: float,
        l1_clamp_distance: Optional[float] = None,
        eps: float = 1e-8,
) -> torch.Tensor:
    local_pred_pos = pred_frames.invert()[..., None].apply(pred_positions[..., None, :, :])
    local_target_pos = target_frames.invert()[..., None].apply(target_positions[..., None, :, :])

    error_dist = torch.sqrt(torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps)

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale

    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


def fbb_backbone_loss(
        pred_trans: torch.Tensor,
        pred_rot: torch.Tensor,
        gt_trans: torch.Tensor,
        gt_rot: torch.Tensor,
        mask: torch.Tensor,
        clamp_distance: float = 10.0,
        loss_unit_distance: float = 10.0,
        eps: float = 1e-4,
) -> torch.Tensor:
    pred_aff = du.create_rigid(pred_rot, pred_trans)
    gt_aff = du.create_rigid(gt_rot, gt_trans)



    fape_loss = compute_fape(
        pred_aff,
        gt_aff,
        mask,
        pred_aff.get_trans(),
        gt_aff.get_trans(),
        mask,
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )

    return fape_loss

def report_atom14_rmsd(pred_local, gt_local, exists_mask, names, topk=10):
    # pred_local, gt_local: [B,N,14,3], exists_mask: [B,N,14] (bool)
    B,N,A,_ = gt_local.shape
    m = exists_mask.bool()
    diff = (pred_local - gt_local).pow(2).sum(-1).sqrt()  # [B,N,14]
    diff = torch.where(m, diff, torch.full_like(diff, float('nan')))

    # per-atom
    means, counts = [], []
    for i in range(A):
        v = diff[..., i].flatten()
        v = v[~torch.isnan(v)]
        c = v.numel()
        means.append(v.mean().item() if c>0 else float('nan'))
        counts.append(int(c))
    print("\n=== Atom14 RMSD Report ===")
    print(f"Overall RMSD: {torch.nanmean(diff).item():.3f} Å\n")
    print("## Per-atom RMSD (Å) | Count\n")
    for i,(mu,ct) in enumerate(zip(means,counts)):
        print(f"{i:2d} {names[i]:>4s}: {mu:6.3f}  |  {ct}")

    # worst residues
    res_rmsd = torch.nanmean(diff, dim=-1)  # [B,N]
    flat = res_rmsd.flatten()
    vals, idx = torch.topk(torch.nan_to_num(flat, nan=-1.0), k=min(topk, flat.numel()))
    print("\nWorst residues (RMSD):\n")
    for rank,(v,i) in enumerate(zip(vals.tolist(), idx.tolist()), 1):
        print(f"{rank}. {v:.3f} Å (flat idx={i})")
    print("==================================")

@torch.no_grad()
def rmsd_analysis_atom14(
    pred: torch.Tensor,             # [B, N, 14, 3]
    gt: torch.Tensor,               # [B, N, 14, 3]
    exists: torch.Tensor,           # [B, N, 14]  bool or 0/1
    atom_names = ('N','CA','C','O','CB','CG','CD','NE','CZ','NH1','NH2','OG','SG','OH'),
    eps: float = 1e-8,
):
    """
    返回:
      report: dict
        - overall_rmsd: 标量
        - per_atom_rmsd: [14]，每个位点 RMSD
        - per_atom_count: [14]，各位点有效原子数量
        - per_residue_rmsd: [B,N]，每个残基（对其存在的原子）RMSD
        - atom_names: tuple[str] 长度14
    """
    assert pred.shape == gt.shape and pred.shape[-1] == 3
    assert pred.shape[:-1] == exists.shape
    B, N, A, _ = pred.shape
    assert A == 14, "expect 14-atom convention"

    exists = exists.to(dtype=pred.dtype)
    # [B,N,14] -> [B,N,14,1] for broadcasting
    mask = exists.unsqueeze(-1)

    diff = (pred - gt) * mask                        # [B,N,14,3]
    sq  = (diff ** 2).sum(dim=-1)                    # [B,N,14]  squared distance per atom
    # --- overall RMSD ---
    overall_rmsd = torch.sqrt((sq.sum() / (exists.sum() + eps))).item()

    # --- per-atom (14位点) RMSD ---
    per_atom_sum = sq.sum(dim=(0,1))                 # [14]
    per_atom_cnt = exists.sum(dim=(0,1)).clamp_min(1.0)  # [14]
    per_atom_rmsd = torch.sqrt(per_atom_sum / per_atom_cnt)  # [14]

    # --- per-residue RMSD（对每个残基，平均其“存在”的原子）---
    per_residue_cnt = exists.sum(dim=-1).clamp_min(1.0)     # [B,N]
    per_residue_rmsd = torch.sqrt( (sq.sum(dim=-1) / per_residue_cnt) )  # [B,N]

    report = {
        "overall_rmsd": overall_rmsd,
        "per_atom_rmsd": per_atom_rmsd.detach().cpu(),
        "per_atom_count": per_atom_cnt.detach().cpu(),
        "per_residue_rmsd": per_residue_rmsd.detach().cpu(),
        "atom_names": tuple(atom_names) if atom_names is not None else tuple(range(14)),
    }
    return report


def print_rmsd_report(report, topk_residues: int = 10):
    atom_names = report["atom_names"]
    pa = report["per_atom_rmsd"].tolist()
    pc = report["per_atom_count"].tolist()

    print(f"\n=== Atom14 RMSD Report ===")
    print(f"Overall RMSD: {report['overall_rmsd']:.3f} Å")
    print("\nPer-atom RMSD (Å)  |  Count")
    print("-"*34)
    for i, name in enumerate(atom_names):
        print(f"{i:2d} {name:>4s}: {pa[i]:6.3f}  |  {int(pc[i])}")

    # 每残基 worst K
    prs = report["per_residue_rmsd"].view(-1)
    vals, idx = torch.topk(prs, k=min(topk_residues, prs.numel()))
    print("\nWorst residues (RMSD):")
    for rank, (v, linear_idx) in enumerate(zip(vals.tolist(), idx.tolist()), 1):
        print(f"{rank:2d}. {v:.3f} Å  (flat idx={linear_idx})")
    print("="*34 + "\n")


ELEM2IDX = {"C": 0, "N": 1, "O": 2, "S": 3}
AA_LIST = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','UNK']
def _hungarian(cost: torch.Tensor):
    """简易匈牙利：优先用 scipy，没装就退化为贪心（还挺好用）"""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost.cpu().numpy())
        return torch.as_tensor(r, device=cost.device), torch.as_tensor(c, device=cost.device)
    except Exception:
        # 贪心 fallback：依次挑最小代价且不冲突
        K, M = cost.shape
        used_r = torch.zeros(K, dtype=torch.bool, device=cost.device)
        used_c = torch.zeros(M, dtype=torch.bool, device=cost.device)
        pairs = []
        # 展平成 (K*M) 排序
        flat = cost.view(-1)
        order = torch.argsort(flat)
        for idx in order:
            r = (idx // M).item()
            c = (idx %  M).item()
            if not used_r[r] and not used_c[c]:
                pairs.append((r,c))
                used_r[r] = True
                used_c[c] = True
        if len(pairs) == 0:
            return torch.empty(0, dtype=torch.long, device=cost.device), torch.empty(0, dtype=torch.long, device=cost.device)
        rr = torch.tensor([p[0] for p in pairs], device=cost.device)
        cc = torch.tensor([p[1] for p in pairs], device=cost.device)
        return rr, cc

@torch.no_grad()
def rmsd_atom14_elementwise_assignment(
    coords_global,   # [B,N,4,K,3]  预测峰（C/N/O/S）
    peaks_mask,      # [B,N,4,K]    有效峰
    scores,          # [B,N,4,K]    峰分数（可选用作tie-break）
    gt_atom14,       # [B,N,14,3]   GT 坐标（全局或同一参照）
    gt_exists,       # [B,N,14]     GT 是否存在
    aatype,          # [B,N]        0..19
    restype_name_to_atom14_names: dict[str, list[str]],
    tpos=None,       # [B,N,3]      若传，则 CA 直接取 tpos
    score_weight: float = 0.1,      # 成本里加入 -w*score（单位约等于 Å）
    dist_cut: float = 2.5,          # 超过此距离的匹配禁用
):
    device = coords_global.device
    B, N, _, K, _ = coords_global.shape

    # 收集每个槽位的误差
    per_atom_sum = torch.zeros(14, device=device)
    per_atom_cnt = torch.zeros(14, dtype=torch.long, device=device)

    for b in range(B):
        for n in range(N):
            restype = AA_LIST[aatype[b, n].item()]
            names = restype_name_to_atom14_names.get(restype, restype_name_to_atom14_names['UNK'])

            # 预先把“CA”的误差记掉（若提供 tpos）；否则按存在性忽略
            for a_i, name in enumerate(names):
                if name == 'CA' and tpos is not None and gt_exists[b, n, a_i]:
                    d = torch.linalg.norm(gt_atom14[b,n,a_i] - tpos[b,n])
                    per_atom_sum[a_i] += d
                    per_atom_cnt[a_i] += 1

            # 按元素做最优匹配
            for elem, eidx in ELEM2IDX.items():
                # GT 侧：这个元素的所有槽位 index（排除 CA）
                tgt_idx = [i for i, nm in enumerate(names) if (nm != '' and nm != 'CA' and nm[0] == elem and gt_exists[b,n,i])]
                if len(tgt_idx) == 0:
                    continue

                tgt_xyz = gt_atom14[b, n, tgt_idx]              # [T,3]

                # Pred 侧：该元素通道的可用峰
                mask = peaks_mask[b, n, eidx]                   # [K]
                if not mask.any():
                    continue
                cand_xyz = coords_global[b, n, eidx, mask]      # [M,3]
                cand_sco = scores[b, n, eidx, mask]             # [M]

                # 距离代价
                # cost[t, m] = ||tgt_xyz[t] - cand_xyz[m]|| - w*norm_score
                dmat = torch.cdist(tgt_xyz[None, ...], cand_xyz[None, ...]).squeeze(0)  # [T,M]
                # 归一化分数（0..1），防止单位不匹配
                if cand_sco.numel() > 0:
                    sco = (cand_sco - cand_sco.min()) / (cand_sco.max() - cand_sco.min() + 1e-9)
                    dmat = dmat - score_weight * sco[None, :]

                # 超过阈值的禁用（设成很大）
                dmat = torch.where(dmat <= dist_cut, dmat, torch.full_like(dmat, 1e6))

                # 最优指派
                rr, cc = _hungarian(dmat)
                for r, c in zip(rr.tolist(), cc.tolist()):
                    if dmat[r, c] >= 1e5:  # 都太远了，视为匹配失败
                        continue
                    slot = tgt_idx[r]
                    d = torch.linalg.norm(gt_atom14[b,n,slot] - cand_xyz[c])
                    per_atom_sum[slot] += d
                    per_atom_cnt[slot] += 1

    # 输出统计
    rmsd = torch.full((14,), float('nan'), device=device)
    ok = per_atom_cnt > 0
    rmsd[ok] = per_atom_sum[ok] / per_atom_cnt[ok].to(per_atom_sum.dtype)
    overall = torch.nanmean(rmsd[ok]) if ok.any() else torch.tensor(float('nan'), device=device)
    return overall.item(), rmsd.tolist(), per_atom_cnt.tolist()
