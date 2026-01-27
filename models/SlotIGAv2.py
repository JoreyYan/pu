import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 你工程里已有的：
from models.IGA import fused_gaussian_overlap_score




def get_gini_coefficient(occ_k):
    """
    occ_k: [B, K] 每个 Slot 的占用残基数
    """
    # 排序
    occ_sorted, _ = torch.sort(occ_k, dim=-1)
    B, K = occ_k.shape
    index = torch.arange(1, K + 1, device=occ_k.device, dtype=occ_k.dtype)
    # 基尼系数公式
    gini = (torch.sum((2 * index - K - 1) * occ_sorted, dim=-1)) / (K * torch.sum(occ_sorted, dim=-1) + 1e-8)
    return gini.mean() # [0, 1] 越小越均匀

def get_perplexity(A):
    # A_avg: 每个 Slot 的平均占用概率 [B, K]
    occ_prob = A.sum(dim=1) / A.sum(dim=(1, 2), keepdim=True)
    # 计算信息熵
    entropy = -torch.sum(occ_prob * torch.log(occ_prob + 1e-10), dim=-1)
    # 困惑度
    perplexity = torch.exp(entropy)
    return perplexity.mean() # 理想值应该接近 K_use


@torch.no_grad()
def compute_pooling_metrics(A, idx, k_max):
    # 1. 样本内均匀度 (Sample Level)
    occ = A.sum(dim=1)  # [B, K]

    # Perplexity (越多越好，最大为 K_use)
    occ_dist = occ / occ.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    ent = -(occ_dist * torch.log(occ_dist + 1e-8)).sum(dim=-1)
    perp = ent.exp().mean()

    # Active Ratio (越多越好)
    active_slots = (occ > 0.5).float().sum(dim=-1).mean()

    # 2. Bank 覆盖率 (Global Level)
    # 统计这一 batch 里用了多少个不同的 Bank ID
    unique_ids = torch.unique(idx).numel()
    bank_coverage = unique_ids / k_max

    bank_coverage=torch.tensor(bank_coverage,device=active_slots.device)

    # =======================================================
    # [新增] 3. 分配尖锐度 (Assignment Sharpness)
    # =======================================================
    # A: [B, N, K]
    # (1) Max Probability: 越接近 1.0 表示分配越自信 (Hard assignment)
    #     越接近 1/K 表示越均匀 (Blurry)
    max_prob = A.max(dim=-1).values.mean()  # [B, N] -> scalar

    # (2) Assignment Entropy: 越接近 0 表示越自信
    #     注意：这是对 K 维度的熵，不是上面的 occ 熵
    assign_ent = -(A * torch.log(A.clamp_min(1e-8))).sum(dim=-1).mean()

    return {
        "met_perp": perp,  # 有效槽位数
        "met_active": active_slots,  # 活跃槽位绝对数
        "met_bank_cov": bank_coverage,  # Bank 利用率
        "met_max_prob": max_prob,  # [新增] 监控这个！如果 < 0.5 说明很糊
        "met_assign_ent": assign_ent  # [新增] 监控这个！如果很大说明很糊
    }

def merge_gaussians_soft(
    mu: torch.Tensor,          # [B,N,3]
    Sigma: torch.Tensor,       # [B,N,3,3]
    A: torch.Tensor,           # [B,N,K]
    mask: Optional[torch.Tensor] = None,  # [B,N]
    eps: float = 1e-8,
    jitter: float = 1e-6,
):
    """
    Moment matching with membership weights A (soft assignment).
    Returns:
      mu_k: [B,K,3]
      Sig_k:[B,K,3,3]
      occ:  [B,K]
    """
    B, N, _ = mu.shape
    K = A.shape[-1]
    device, dtype = mu.device, mu.dtype
    I = torch.eye(3, device=device, dtype=dtype)[None, None]

    if mask is not None:
        A = A * mask.unsqueeze(-1)

    occ = A.sum(dim=1).clamp_min(eps)  # [B,K]

    mu_k = torch.einsum("bnk,bni->bki", A, mu) / occ.unsqueeze(-1)

    d = mu[:, :, None, :] - mu_k[:, None, :, :]  # [B,N,K,3]
    outer = d.unsqueeze(-1) * d.unsqueeze(-2)    # [B,N,K,3,3]

    Sig_k = (
        torch.einsum("bnk,bnij->bkij", A, Sigma) +
        torch.einsum("bnk,bnkij->bkij", A, outer)
    ) / occ.view(B, K, 1, 1)

    Sig_k = 0.5 * (Sig_k + Sig_k.transpose(-1, -2)) + jitter * I
    return mu_k, Sig_k, occ

# ----------------------------
# Loss container
# ----------------------------
@dataclass
class PoolLoss:
    usage_kl: torch.Tensor
    usage_ent: torch.Tensor
    occ_mse: torch.Tensor
    collapse: torch.Tensor
    rep: torch.Tensor
    total: torch.Tensor


# ----------------------------
# Small utils
# ----------------------------
def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8):
    """
    x: [B,N,...], mask: [B,N]
    """
    assert mask.dim() == 2, f"mask must be [B,N], got {mask.shape}"
    m = mask
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    num = (x * m).sum(dim=dim)
    den = m.sum(dim=dim).clamp_min(eps)
    return num / den


def token_geo_invariants(mu: torch.Tensor, Sigma: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """
    旋转不变的 token 形状描述 (很便宜)：
      - logdet(Sigma)
      - log(tr(Sigma))
      - sqrt(diag) 三个尺度（或它们的 log）
    输出: [B,N,5] 你也可以改成更少/更多
    """
    assert mu.shape[-1] == 3 and Sigma.shape[-2:] == (3, 3)
    # sym
    S = 0.5 * (Sigma + Sigma.transpose(-1, -2))
    s00, s01, s02 = S[..., 0, 0], S[..., 0, 1], S[..., 0, 2]
    s11, s12 = S[..., 1, 1], S[..., 1, 2]
    s22 = S[..., 2, 2]

    det = s00 * (s11 * s22 - s12 * s12) - s01 * (s01 * s22 - s02 * s12) + s02 * (s01 * s12 - s02 * s11)
    det = det.clamp_min(eps)
    logdet = torch.log(det)

    tr = (s00 + s11 + s22).clamp_min(eps)
    logtr = torch.log(tr)

    diag = torch.stack([s00, s11, s22], dim=-1).clamp_min(eps)
    logdiag = torch.log(diag)

    feat = torch.cat([logdet.unsqueeze(-1), logtr.unsqueeze(-1), logdiag], dim=-1)  # [B,N,5]
    feat = feat * mask.unsqueeze(-1)
    return feat


def collapse_loss(A: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    """
    penalize similarity among columns of A: A^T A off-diagonal
    A: [B,N,K], mask: [B,N]
    """
    B, N, K = A.shape
    A = A * mask.unsqueeze(-1)
    X = A.transpose(1, 2)  # [B,K,N]
    X = X / X.norm(dim=-1, keepdim=True).clamp_min(eps)
    G = torch.einsum("bkn,bln->bkl", X, X)  # [B,K,K]
    eye = torch.eye(K, device=A.device, dtype=A.dtype).unsqueeze(0)
    off = (G - eye) * (1.0 - eye)
    return (off ** 2).mean()


@torch.no_grad()
def get_residue_to_slot_map(A, mask, idx):
    """
    分析分配矩阵 A，查看每个 Slot (K) 负责了哪些原始残基 (N)。

    A: [B, N, K] - 分配概率
    mask: [B, N] - 残基掩码
    idx: [B, K] - 选中的 Bank 索引
    """
    B, N, K = A.shape
    # 1. 硬分配：对每个残基 n，找相似度最大的那个 slot k
    # assignment: [B, N]
    assignment = A.argmax(dim=-1)

    # 屏蔽掉无效的 mask 位置
    assignment = assignment.masked_fill(mask < 0.5, -1)

    all_batches_map = []

    for b in range(B):
        batch_map = {}
        for k in range(K):
            # 找到属于第 k 个 slot 的残基索引
            residue_indices = (assignment[b] == k).nonzero(as_tuple=True)[0]

            # 记录信息
            batch_map[k] = {
                "bank_id": idx[b, k].item(),  # 它在 1024 里的 ID
                "res_indices": residue_indices.tolist(),  # 领走了哪些残基
                "num_res": len(residue_indices)  # 领走了多少个
            }
        all_batches_map.append(batch_map)

    return all_batches_map
# ----------------------------
# Greedy coverage selection from precomputed sim
# ----------------------------
@torch.no_grad()
def greedy_coverage_from_sim(sim: torch.Tensor, mask: torch.Tensor, K_use: int):
    """
    sim:  [B,N,Kmax] (higher is better)
    mask: [B,N]
    returns idx: [B,K_use]
    """
    B, N, Kmax = sim.shape
    K_use = min(K_use, Kmax)

    # current best per token
    covered = torch.full((B, N), -1e9, device=sim.device, dtype=sim.dtype)

    idx_out = []
    picked_mask = torch.zeros((B, Kmax), device=sim.device, dtype=torch.bool)

    for _ in range(K_use):
        # gain = max(covered, sim_k) - covered
        gain = torch.maximum(covered.unsqueeze(-1), sim) - covered.unsqueeze(-1)  # [B,N,Kmax]
        gain = gain.masked_fill((mask < 0.5).unsqueeze(-1), 0.0)
        gain_sum = gain.sum(dim=1)  # [B,Kmax]

        # prevent repick
        gain_sum = gain_sum.masked_fill(picked_mask, -1e9)

        next_idx = gain_sum.argmax(dim=-1)  # [B]
        idx_out.append(next_idx)

        picked_mask.scatter_(1, next_idx[:, None], True)

        # update covered
        best_new = sim.gather(-1, next_idx[:, None, None].expand(B, N, 1)).squeeze(-1)
        covered = torch.maximum(covered, best_new)

    idx = torch.stack(idx_out, dim=1)  # [B,K_use]
    return idx

def slot_repulsion(mu_k, mask_k, sigma=2.0):
    # mu_k: [B,K,3], mask_k: [B,K]
    diff = mu_k[:, :, None, :] - mu_k[:, None, :, :]
    dist2 = (diff**2).sum(-1)  # [B,K,K]

    eye = torch.eye(mu_k.size(1), device=mu_k.device)[None]
    dist2 = dist2 + eye * 1e6  # ignore self

    rep = torch.exp(-dist2 / (2 * sigma**2))
    rep = rep * mask_k[:, :, None] * mask_k[:, None, :]
    return rep.mean()
# ----------------------------
# Main module
# ----------------------------
class IGASlotPoolingV2(nn.Module):
    """
    结构闭环版（推荐）：

    Step 0: 计算 sim_all[n,k]（只算一次）
      sim_all = sim_sem + lam_geo * sim_geo   (geo 可选)
    Step 1: 用 sim_all 做 greedy coverage 选 idx（K_use 个）
    Step 2: 用同一份 sim_all 的子矩阵 sim_sel 构造 A0 (softmax over K_use)
    Step 3: 用 A0 做 moment matching 初始化 (mu_k, Sig_k)
    Step 4: refine iters：语义 logits + IGA 几何 bias -> A -> moment matching -> slot GRU

    输出：
      A, s_c, mu_c, Sigma_c, idx, losses
    """

    def __init__(
        self,
        c_s: int,
        k_max: int = 1024,
        ratio: float = 6.0,
        iters: int = 3,

        # gating sim
        tau_gate: float = 0.7,
        gate_noise: float = 0.02,

        # geo sim for gating（可选）
        use_geo_gating: bool = True,
        lam_geo_gating: float = 0.3,
        c_geo: int = 64,            # geo embedding dim（投到 c_s 也行，这里用独立 dim）

        # refine
        tau_refine: float = 1.0,
        w_geo_init: float = 1.0,
        geo_heads: int = 4,
        sigma_floor: float = 0.03,
        jitter: float = 1e-6,

        # losses
        w_usage_kl: float = 1.0,
        w_usage_ent: float = 0.0,   # 你也可以不用 ent，只用 KL
        w_occ_mse: float = 1.0,
        w_collapse: float = 1.0,
            w_rep: float = 1.0,
            w_KL_A: float = 1.0,

        usage_ema_momentum: float = 0.99,
    ):
        super().__init__()
        self.c_s = c_s
        self.k_max = k_max
        self.ratio = ratio
        self.iters = iters

        self.tau_gate = float(tau_gate)
        self.gate_noise = float(gate_noise)

        self.use_geo_gating = bool(use_geo_gating)
        self.lam_geo_gating = float(lam_geo_gating)

        self.tau_refine = float(tau_refine)
        self.w_geo = nn.Parameter(torch.tensor(float(w_geo_init)))
        self.geo_heads = int(geo_heads)
        self.sigma_floor = float(sigma_floor)
        self.jitter = float(jitter)

        # bank
        self.slot_embed_pool = nn.Parameter(torch.randn(k_max, c_s) * 0.02)

        # gating projections
        self.gating_proj = nn.Linear(c_s, c_s, bias=False)

        # optional geo-gating bank + proj
        if self.use_geo_gating:
            # 存一个“形状原型 bank”（旋转不变）
            self.geo_proto_pool = nn.Parameter(torch.randn(k_max, 5) * 0.02)  # 5 = token_geo_invariants 输出维
            self.geo_proj_tok = nn.Linear(5, c_geo, bias=False)
            self.geo_proj_bank = nn.Linear(5, c_geo, bias=False)
            self.sem_proj_tok = nn.Linear(c_s, c_geo, bias=False)  # 让语义/几何都在 c_geo 空间比 cos
            self.sem_proj_bank = nn.Linear(c_s, c_geo, bias=False)
        else:
            self.geo_proto_pool = None

        # refine semantic q/k
        self.proj_q = nn.Linear(c_s, c_s, bias=False)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)

        self.gru = nn.GRUCell(c_s, c_s)
        self.mlp = nn.Sequential(
            nn.LayerNorm(c_s),
            nn.Linear(c_s, 4 * c_s),
            nn.GELU(),
            nn.Linear(4 * c_s, c_s),
        )

        # geo calibration (a*G + b) per head
        self.geo_scale_raw = nn.Parameter(torch.full((self.geo_heads,), -4.0))
        self.geo_bias = nn.Parameter(torch.full((self.geo_heads,), -6.0))

        # usage ema over bank
        self.register_buffer("usage_ema", torch.full((k_max,), 1.0 / k_max))
        self.usage_ema_momentum = float(usage_ema_momentum)

        # loss weights
        self.w_usage_kl = float(w_usage_kl)
        self.w_usage_ent = float(w_usage_ent)
        self.w_occ_mse = float(w_occ_mse)
        self.w_collapse = float(w_collapse)
        self.w_rep = float(w_rep)
        self.w_KL_A=float(w_KL_A)

    @staticmethod
    def _choose_K(N: int, ratio: float):
        return max(1, int(math.ceil(N / max(ratio, 1e-6))))

    def _sim_all(self, s: torch.Tensor, mu: torch.Tensor, Sigma: torch.Tensor, mask: torch.Tensor):
        """
        return sim_all: [B,N,Kmax]
        """
        B, N, C = s.shape
        mask_f = mask

        # semantic cos sim (token-level, 不是 global)
        if self.use_geo_gating:
            # project to shared dim c_geo
            S_sem = F.normalize(self.sem_proj_tok(s), dim=-1)                     # [B,N,c_geo]
            K_sem = F.normalize(self.sem_proj_bank(self.slot_embed_pool), dim=-1) # [Kmax,c_geo]
        else:
            S_sem = F.normalize(self.gating_proj(s), dim=-1)                      # [B,N,C]
            K_sem = F.normalize(self.slot_embed_pool, dim=-1)                     # [Kmax,C]

        sim_sem = torch.einsum("bnc,kc->bnk", S_sem, K_sem)  # [B,N,Kmax]

        if self.use_geo_gating:
            g_tok = token_geo_invariants(mu, Sigma, mask_f)                        # [B,N,5]
            g_bank = self.geo_proto_pool[None, :, :].expand(B, -1, -1)             # [B,Kmax,5]
            G_tok = F.normalize(self.geo_proj_tok(g_tok), dim=-1)                  # [B,N,c_geo]
            G_bank = F.normalize(self.geo_proj_bank(g_bank), dim=-1)               # [B,Kmax,c_geo]
            sim_geo = torch.einsum("bnc,bkc->bnk", G_tok, G_bank)                  # [B,N,Kmax]
            sim_all = sim_sem + self.lam_geo_gating * sim_geo
        else:
            sim_all = sim_sem

        sim_all = sim_all.masked_fill((mask_f < 0.5).unsqueeze(-1), -1e9)
        return sim_all

    def _a0_from_sim_and_idx(self, sim_all: torch.Tensor, idx: torch.Tensor, mask: torch.Tensor):
        """
        sim_all: [B,N,Kmax]
        idx:     [B,K_use]
        return A0: [B,N,K_use]
        """
        B, N, Kmax = sim_all.shape
        K_use = idx.size(1)
        sim_sel = sim_all.gather(-1, idx[:, None, :].expand(B, N, K_use))  # [B,N,K_use]

        logits0 = sim_sel / max(self.tau_gate, 1e-6)
        logits0 = logits0.masked_fill((mask < 0.5).unsqueeze(-1), -1e9)
        A0 = torch.softmax(logits0, dim=-1) * mask.unsqueeze(-1)
        return A0

    def _update_usage_ema(self, idx: torch.Tensor):
        """
        idx: [B,K_use]
        """
        if not self.training:
            return
        flat = idx.reshape(-1)
        curr = torch.zeros(self.k_max, device=idx.device, dtype=self.usage_ema.dtype)
        ones = torch.ones_like(flat, dtype=curr.dtype)
        curr.scatter_add_(0, flat, ones)
        curr = curr / curr.sum().clamp_min(1.0)
        m = self.usage_ema_momentum
        self.usage_ema.copy_(m * self.usage_ema + (1.0 - m) * curr)

    def forward(
        self,
        s: torch.Tensor,              # [B,N,C]
        mu: torch.Tensor,             # [B,N,3]
        Sigma: torch.Tensor,          # [B,N,3,3]
        mask: torch.Tensor,           # [B,N] 0/1

    ):
        B, N, C = s.shape
        device, dtype = s.device, s.dtype
        assert mask.shape == (B, N), f"mask must be [B,N], got {mask.shape}"
        mask = mask.to(device=device, dtype=dtype)

        K_use = self._choose_K(N, self.ratio)
        K_use = min(K_use, self.k_max)

        # ----------------------------
        # (A) compute sim_all ONCE
        # ----------------------------
        sim_all = self._sim_all(s, mu, Sigma, mask)  # [B,N,Kmax]
        if self.training and self.gate_noise > 0:
            sim_all = sim_all + torch.randn_like(sim_all) * self.gate_noise

        # ----------------------------
        # (B) select idx by greedy coverage
        # ----------------------------
        idx = greedy_coverage_from_sim(sim_all, mask, K_use)  # [B,K_use]
        self._update_usage_ema(idx)

        # gather active slot embeddings
        slots = self.slot_embed_pool[idx]  # [B,K_use,C]

        # ----------------------------
        # (C) build A0 from SAME sim_all
        # ----------------------------
        A = self._a0_from_sim_and_idx(sim_all, idx, mask)  # [B,N,K_use]
        # 在你的代码外调用打印：
        # mapping = get_residue_to_slot_map(A, mask, idx)
        # for k, info in mapping[0].items():
        #     print(f"Slot {k} (Bank #{info['bank_id']}) 负责了 {info['num_res']} 个残基: {info['res_indices']}")

        # ----------------------------
        # (D) init geo by moment matching from A0 (no template offsets)
        # ----------------------------
        mu_k, Sig_k, occ = merge_gaussians_soft(
            mu=mu, Sigma=Sigma, A=A, mask=mask, jitter=max(self.jitter, 1e-6)
        )

        # sigma floor + sym
        I = torch.eye(3, device=device, dtype=dtype)[None, None]
        Sig_k = 0.5 * (Sig_k + Sig_k.transpose(-1, -2)) + self.jitter * I
        diag = torch.diagonal(Sig_k, dim1=-2, dim2=-1).clamp_min(self.sigma_floor ** 2)
        Sig_k = Sig_k - torch.diag_embed(torch.diagonal(Sig_k, dim1=-2, dim2=-1)) + torch.diag_embed(diag)

        # ----------------------------
        # (E) refine loop
        # ----------------------------
        k_sem = self.proj_k(s)  # [B,N,C]

        kl_A=0

        for _ in range(self.iters):
            q_sem = self.proj_q(slots)  # [B,K,C]
            logits_sem = torch.einsum("bnc,bkc->bnk", k_sem, q_sem) / math.sqrt(C)

            delta = mu[:, :, None, :] - mu_k[:, None, :, :]                    # [B,N,K,3]
            sigma_sum = Sigma[:, :, None, :, :] + Sig_k[:, None, :, :, :]      # [B,N,K,3,3]

            G = fused_gaussian_overlap_score(delta, sigma_sum)                 # [B,N,K] <=0

            a = F.softplus(self.geo_scale_raw).view(1, self.geo_heads, 1, 1)
            b = self.geo_bias.view(1, self.geo_heads, 1, 1)
            geo_bias = (a * G.unsqueeze(1) + b).sum(dim=1)                     # [B,N,K]

            logits = (logits_sem + self.w_geo * geo_bias) / max(self.tau_refine, 1e-6)
            logits = logits.masked_fill((mask < 0.5).unsqueeze(-1), -1e9)
            A = torch.softmax(logits, dim=-1) * mask.unsqueeze(-1)             # [B,N,K]

            p = (A * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B,K]
            p = p.clamp_min(1e-8)
            kl = (p * (p * K_use).log()).sum(dim=-1).mean()  # KL(p || 1/K)
            kl_A=kl_A+kl

            # 在你的代码外调用打印：
            # mapping = get_residue_to_slot_map(A, mask, idx)
            # for k, info in mapping[0].items():
            #     print(f"Slot {k} (Bank #{info['bank_id']}) 负责了 {info['num_res']} 个残基: {info['res_indices']}")

            # geo update
            mu_k, Sig_k, occ = merge_gaussians_soft(
                mu=mu, Sigma=Sigma, A=A, mask=mask, jitter=max(self.jitter, 1e-6)
            )
            Sig_k = 0.5 * (Sig_k + Sig_k.transpose(-1, -2)) + self.jitter * I
            diag = torch.diagonal(Sig_k, dim1=-2, dim2=-1).clamp_min(self.sigma_floor ** 2)
            Sig_k = Sig_k - torch.diag_embed(torch.diagonal(Sig_k, dim1=-2, dim2=-1)) + torch.diag_embed(diag)

            # slot semantic update
            denom = A.sum(dim=1).clamp_min(1e-8)                               # [B,K]
            slot_in = torch.einsum("bnk,bnc->bkc", A, s) / denom.unsqueeze(-1) # [B,K,C]
            slots = self.gru(slot_in.reshape(-1, C), slots.reshape(-1, C)).view(B, K_use, C)
            slots = slots + self.mlp(slots)

        # pooled outputs
        s_c, mu_c, Sigma_c = slots, mu_k, Sig_k

        # ----------------------------
        # (F) losses: usage diversity + occ balance + collapse
        # ----------------------------
        # 1) usage over BANK (which idx chosen)
        if self.training:
            usage = self.usage_ema.clamp_min(1e-12)
        else:
            usage = self.usage_ema.clamp_min(1e-12)

        uniform = torch.full_like(usage, 1.0 / self.k_max)
        usage_kl = (usage * (usage / uniform).log()).sum()  # KL(usage || uniform)
        usage_ent = -(usage * usage.log()).sum()

        # 2) occ balance over ACTIVE K (from A)
        occ_k = A.sum(dim=1)  # [B,K]
        mask_k = (occ > 1e-4).to(mu_k.dtype)  # [B,K]
        occ_k = occ_k / occ_k.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        target = torch.full_like(occ_k, 1.0 / max(K_use, 1))
        occ_mse = F.mse_loss(occ_k, target)
        rep=slot_repulsion(mu_k,mask_k)

        # 3) collapse
        col = collapse_loss(A, mask)

        total = (
            self.w_usage_kl * usage_kl +
            self.w_usage_ent * (-usage_ent) +   # 你如果想“更高熵更好”，就对 -ent 加权；不想就置 w_usage_ent=0
            self.w_occ_mse * occ_mse +
            self.w_collapse * col+
            self.w_rep*rep+self.w_KL_A+kl_A
        )

        losses = PoolLoss(
            usage_kl=usage_kl,
            usage_ent=-usage_ent,
            occ_mse=occ_mse,
            collapse=col,
            rep=rep,
            total=total,
        )

        metric=compute_pooling_metrics(A,idx,K_use)

        return A, s_c, mu_c, Sigma_c, idx, losses,metric
