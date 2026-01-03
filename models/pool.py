import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
from data import utils as du
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.GaussianRigid import OffsetGaussianRigid,save_gaussian_as_pdb
from openfold.utils.rigid_utils import Rotation
from models.IGA import fused_gaussian_overlap_score
def gaussian_geo_features(Sigma: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Sigma: [B, N, 3, 3] (对称/近似SPD)
    return g: [B, N, 6]  (平移不变，只依赖形状)
      g = [logdet, logtrace, aniso, logλ1, logλ2, logλ3]
    """
    # eigenvalues: [B,N,3]  升序 λ0<=λ1<=λ2
    lam = torch.linalg.eigvalsh(Sigma)  # stable for symmetric
    lam = lam.clamp_min(eps)

    loglam = torch.log(lam)  # [B,N,3]
    logdet = loglam.sum(dim=-1, keepdim=True)  # [B,N,1]

    tr = lam.sum(dim=-1, keepdim=True).clamp_min(eps)
    logtr = torch.log(tr)  # [B,N,1]

    # anisotropy: logλ_max - logλ_min
    aniso = (loglam[..., 2:3] - loglam[..., 0:1])  # [B,N,1]

    g = torch.cat([logdet, logtr, aniso, loglam], dim=-1)  # [B,N,6]
    return g


# =========================
# 1) Learn-only pooling (你原来的 그대로)
# =========================
def choose_K(N: int, ratio: float = 12.0, k_min: int = 1, k_max: Optional[int] = None) -> int:
    k = int(math.ceil(N / ratio))
    k = max(k, k_min)
    if k_max is not None:
        k = min(k, k_max)
    return k


def merge_gaussians_soft(mu, Sigma, A, mask=None, eps=1e-8, jitter=1e-6):
    """
    mu:    [B,N,3]
    Sigma: [B,N,3,3]
    A:     [B,N,K]
    mask:  [B,N] in {0,1} (optional)
    Return:
      mu_c:    [B,K,3]
      Sigma_c: [B,K,3,3]  (SPD-ish)
    """
    if mask is not None:
        A = A * mask.unsqueeze(-1)

    denom = A.sum(dim=1, keepdim=True).clamp_min(eps)  # [B,1,K]
    w = A / denom                                       # [B,N,K]

    mu_c = torch.einsum("bnk,bnd->bkd", w, mu)          # [B,K,3]

    # intra: E[Sigma_i]
    intra = torch.einsum("bnk,bnij->bkij", w, Sigma)    # [B,K,3,3]

    # inter: Cov(mu_i)
    diff = mu.unsqueeze(2) - mu_c.unsqueeze(1)          # [B,N,K,3]
    inter = torch.einsum("bnk,bnkd,bnke->bkde", w, diff, diff)  # [B,K,3,3]

    Sigma_c = intra + inter
    Sigma_c = 0.5 * (Sigma_c + Sigma_c.transpose(-1, -2))

    I = torch.eye(3, device=Sigma_c.device, dtype=Sigma_c.dtype)
    Sigma_c = Sigma_c + jitter * I  # broadcast [3,3] -> [B,K,3,3]
    return mu_c, Sigma_c



def gaussian_overlap_score(delta: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    I = torch.eye(3, device=sigma.device, dtype=sigma.dtype)
    inv = torch.linalg.inv(sigma + eps * I)
    dist2 = torch.einsum("...i,...ij,...j->...", delta, inv, delta)
    return -0.5 * dist2


@dataclass
class PoolLoss:
    occ: torch.Tensor
    rep: torch.Tensor
    ent: torch.Tensor
    collapse: torch.Tensor
    total: torch.Tensor



def _symm_jitter(S, jitter=1e-6):
    S = 0.5 * (S + S.transpose(-1, -2))
    I = torch.eye(3, device=S.device, dtype=S.dtype)
    return S + jitter * I


def gaussian_sigma_invariants(Sigma: torch.Tensor, eps: float = 1e-8):
    """
    Sigma: [..., 3, 3] (SPD-ish)
    Return: [..., 6] invariants
      [logdet, logtr, aniso_proxy, log_diag0, log_diag1, log_diag2]
    用 Cholesky 求 logdet，避免 eigvalsh 不稳。
    aniso 用对角 proxy（稳、便宜）。
    """
    Sigma = _symm_jitter(Sigma, jitter=1e-6)
    L = torch.linalg.cholesky(Sigma)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-12)).sum(dim=-1)
    tr = torch.diagonal(Sigma, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(eps)
    logtr = torch.log(tr)

    diag = torch.diagonal(Sigma, dim1=-2, dim2=-1).clamp_min(eps)
    log_diag = torch.log(diag)                       # [...,3]
    aniso = log_diag.max(dim=-1).values - log_diag.min(dim=-1).values

    feat = torch.cat([logdet.unsqueeze(-1), logtr.unsqueeze(-1), aniso.unsqueeze(-1), log_diag], dim=-1)
    return feat  # [...,6]


def merge_gaussians_soft(mu, Sigma, A, mask=None, eps=1e-8, jitter=1e-6):
    """
    Moment matching:
      mu_c = E[mu_i]
      Sigma_c = E[Sigma_i] + Cov(mu_i)
    """
    if mask is not None:
        A = A * mask.unsqueeze(-1)

    denom = A.sum(dim=1, keepdim=True).clamp_min(eps)   # [B,1,K]
    w = A / denom                                       # [B,N,K]
    mu_c = torch.einsum("bnk,bnd->bkd", w, mu)          # [B,K,3]

    intra = torch.einsum("bnk,bnij->bkij", w, Sigma)    # [B,K,3,3]
    diff = mu.unsqueeze(2) - mu_c.unsqueeze(1)          # [B,N,K,3]
    inter = torch.einsum("bnk,bnkd,bnke->bkde", w, diff, diff)  # [B,K,3,3]

    Sigma_c = intra + inter
    Sigma_c = 0.5 * (Sigma_c + Sigma_c.transpose(-1, -2))
    I = torch.eye(3, device=Sigma_c.device, dtype=Sigma_c.dtype)
    Sigma_c = Sigma_c + jitter * I
    return mu_c, Sigma_c


class LearnOnlyGaussianPoolingV2(nn.Module):
    """
    Learnable pooling with:
      - semantic dot-product term (s vs slot_embed)
      - spatial distance term (mu vs mu_slot)  [soft k-means / GMM-like]
      - shape term (Sigma invariants vs geo_slot)
      - weak slot identity bias (symmetry breaking)
      - anti-collapse regularizer (column correlation of A)

    Returns:
      A: [B,N,K], s_c: [B,K,C], mu_c: [B,K,3], Sigma_c: [B,K,3,3], losses: PoolLoss
    """

    def __init__(
        self,
        c_s: int,
        ratio: float = 12.0,
        k_max_cap: Optional[int] = None,
        tau_init: float = 1.0,
        slots_init_scale: float = 0.02,
        eps: float = 1e-8,

        # weights for logits terms
        w_sem_init: float = 1.0,
        w_mu_init: float = 1.0,
        w_shape_init: float = 0.3,

        # spatial kernel
        sigma_mu_init: float = 1.0,   # in nm if mu is nm
        learn_mu_sigma: bool = True,

        # shape branch
        geo_dim: int = 16,

        # symmetry breaking (weak!)
        id_bias_init: float = 0.02,   # small
        learn_id_bias: bool = True,

        # regularizers
        w_occ: float = 1.0,
        w_ent: float = 0.0,
        w_collapse: float = 0.1,
    ):
        super().__init__()
        self.ratio = ratio
        self.k_max_cap = k_max_cap
        self.tau = tau_init
        self.eps = eps

        self.w_occ = w_occ
        self.w_ent = w_ent
        self.w_collapse = w_collapse

        # learnable weights to combine terms (you can schedule externally too)
        self.w_sem = nn.Parameter(torch.tensor(float(w_sem_init)))
        self.w_mu = nn.Parameter(torch.tensor(float(w_mu_init)))
        self.w_shape = nn.Parameter(torch.tensor(float(w_shape_init)))

        # semantic projection + slot prototypes
        self.proj = nn.Linear(c_s, c_s, bias=False)
        self.slot_embed: Optional[nn.Parameter] = None
        self._slot_K: Optional[int] = None
        self._slots_init_scale = slots_init_scale

        # spatial prototypes: mu_slot
        self.mu_slot: Optional[nn.Parameter] = None

        # sigma for spatial distances (log form)
        if learn_mu_sigma and sigma_mu_init > 0:
            self.log_sigma_mu = nn.Parameter(torch.log(torch.tensor(float(sigma_mu_init))))
        else:
            self.register_buffer("log_sigma_mu", torch.log(torch.tensor(float(max(sigma_mu_init, 1e-3)))))

        # shape branch: Sigma invariants -> geo_dim, slot geo prototypes
        self.geo_proj = nn.Linear(6, geo_dim, bias=True)
        self.geo_ln = nn.LayerNorm(geo_dim)
        self.geo_slot: Optional[nn.Parameter] = None

        # weak slot identity bias
        if learn_id_bias:
            self.id_bias: Optional[nn.Parameter] = None
            self._id_bias_init = float(id_bias_init)
        else:
            self.id_bias = None
            self.register_buffer("_fixed_id_bias", torch.tensor(float(id_bias_init)))

    def _ensure_params(self, K: int, C: int, device, dtype):
        # slot_embed
        if (self.slot_embed is None) or (self._slot_K != K) or (self.slot_embed.shape[-1] != C):
            self._slot_K = K
            self.slot_embed = nn.Parameter(torch.randn(K, C, device=device, dtype=dtype) * self._slots_init_scale)

        # mu_slot
        if (self.mu_slot is None) or (self.mu_slot.shape[0] != K):
            # init small random; you may prefer init from uniform anchors later
            self.mu_slot = nn.Parameter(torch.randn(K, 3, device=device, dtype=dtype) * 0.5)

        # geo_slot
        if (self.geo_slot is None) or (self.geo_slot.shape[0] != K):
            self.geo_slot = nn.Parameter(torch.randn(K, self.geo_proj.out_features, device=device, dtype=dtype) * self._slots_init_scale)

        # id_bias
        if self.id_bias is None:
            # create if learnable
            if isinstance(getattr(self, "_id_bias_init", None), float):
                self.id_bias = nn.Parameter(torch.randn(K, device=device, dtype=dtype) * self._id_bias_init)

    @staticmethod
    def _choose_K(N: int, ratio: float, k_min: int = 1, k_max: Optional[int] = None):
        K = max(k_min, int(math.ceil(N / max(ratio, 1e-6))))
        if k_max is not None:
            K = min(K, int(k_max))
        return max(K, 1)

    @staticmethod
    def collapse_loss(A: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
        """
        Penalize column similarity of A:
          compute normalized occupancy vectors per slot, take Gram matrix,
          penalize off-diagonal.
        A: [B,N,K], mask: [B,N]
        """
        B, N, K = A.shape
        # masked A
        A = A * mask.unsqueeze(-1)
        # occupancy vectors per slot: [B,N,K] -> [B,K,N]
        X = A.transpose(1, 2)  # [B,K,N]
        # normalize each slot vector
        X = X / (X.norm(dim=-1, keepdim=True).clamp_min(eps))
        # Gram: [B,K,K]
        G = torch.einsum("bkn,bln->bkl", X, X)
        eye = torch.eye(K, device=A.device, dtype=A.dtype).unsqueeze(0)
        off = (G - eye)  # off-diagonals + maybe small diag noise
        return (off * (1.0 - eye)).pow(2).mean()

    def forward(
        self,
        s: torch.Tensor,                 # [B,N,C]
        mu: torch.Tensor,                # [B,N,3] (nm)
        Sigma: torch.Tensor,             # [B,N,3,3]
        mask: Optional[torch.Tensor] = None,  # [B,N] {0,1}
        rep_topk: int = 4,               # (unused here; you can still add overlap repulsion if you want)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PoolLoss]:

        B, N, C = s.shape
        device, dtype = s.device, s.dtype

        if mask is None:
            mask = torch.ones(B, N, device=device, dtype=dtype)
        else:
            mask = mask.to(device=device, dtype=dtype)

        # choose K
        K = self._choose_K(N, ratio=self.ratio, k_min=1, k_max=self.k_max_cap)
        self._ensure_params(K, C, device, dtype)

        # ---- (1) semantic logits ----
        s_proj = self.proj(s)  # [B,N,C]
        slots = self.slot_embed.unsqueeze(0).expand(B, -1, -1)  # [B,K,C]
        logits_sem = torch.einsum("bnc,bkc->bnk", s_proj, slots) / max(float(self.tau), 1e-6)

        # ---- (2) spatial logits (distance to mu_slot) ----
        # mu_slot: [K,3] -> [1,1,K,3]
        # c_mu = self.mu_slot.unsqueeze(0).unsqueeze(0)           # [1,1,K,3]

        idx = fps_anchors_mu(mu, mask, K)  # [B,K]
        mu_anchor = mu.gather(1, idx[..., None].expand(B, K, 3))  # [B,K,3]
        c_mu = mu_anchor.unsqueeze(1)  # [B,1,K,3]
        dist2 = ((mu.unsqueeze(2) - c_mu) ** 2).sum(dim=-1)  # [B,N,K]

        sigma2 = torch.exp(self.log_sigma_mu).clamp_min(1e-4) ** 2
        logits_mu = -dist2 / (2.0 * sigma2)

        # # ---- (3) shape logits (Sigma invariants) ---- “局部形状相似 → 归属同一 domain” 没有生物物理意义
        # g = gaussian_sigma_invariants(Sigma, eps=self.eps)       # [B,N,6]
        # g = self.geo_ln(self.geo_proj(g))                        # [B,N,geo_dim]
        # geo_slots = self.geo_slot.unsqueeze(0).expand(B, -1, -1) # [B,K,geo_dim]
        # logits_shape = torch.einsum("bnd,bkd->bnk", g, geo_slots)

        # ---- (4) weak slot identity bias (symmetry breaking) ----
        # if self.id_bias is not None:
        #     idb = self.id_bias.view(1, 1, K)
        # else:
        #     idb = getattr(self, "_fixed_id_bias", torch.tensor(0.0, device=device, dtype=dtype)).view(1, 1, 1)

        # ---- combine logits ----
        logits = (
            self.w_sem * logits_sem +
            self.w_mu * logits_mu #+idb
            # self.w_shape * logits_shape +

        )

        # mask invalid residues
        logits = logits.masked_fill((mask < 0.5).unsqueeze(-1), -1e9)

        A = torch.softmax(logits, dim=-1)          # [B,N,K]
        A = A * mask.unsqueeze(-1)

        # ---- pooled semantic ----
        denom = A.sum(dim=1).clamp_min(1e-8)       # [B,K]
        s_c = torch.einsum("bnk,bnc->bkc", A, s) / denom.unsqueeze(-1)

        # ---- pooled gaussian (moment matching) ----
        mu_c, Sigma_c = merge_gaussians_soft(mu, Sigma, A, mask=None, eps=self.eps, jitter=1e-6)

        # ---- losses ----
        # occupancy: encourage balanced usage
        occ = A.sum(dim=1)  # [B,K]
        occ = occ / occ.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        target = torch.full_like(occ, 1.0 / max(K, 1))
        loss_occ = F.mse_loss(occ, target)

        # entropy: optional (higher entropy = softer)
        ent = -(A.clamp_min(1e-8) * A.clamp_min(1e-8).log()).sum(dim=-1)  # [B,N]
        ent = (ent * mask).sum() / mask.sum().clamp_min(1.0)

        # anti-collapse: penalize similar columns
        loss_collapse = self.collapse_loss(A, mask, eps=self.eps)

        total = self.w_occ * loss_occ + self.w_ent * ent + self.w_collapse * loss_collapse
        losses = PoolLoss(
            occ=loss_occ,
            rep=torch.zeros((), device=device, dtype=dtype),
            ent=ent,
            collapse=loss_collapse,
            total=total
        )

        return A, s_c, mu_c, Sigma_c, losses



@torch.no_grad()
def uniform_anchors(mask: torch.Tensor, K: int) -> torch.Tensor:
    """
    序列均匀采样 anchor 索引（batch版，支持 mask）。
    mask: [B,N]  (0/1 or bool)
    return idx: [B,K] long, 每个 k 对应一个 valid residue 的原始 index
    """
    device = mask.device
    mask_bool = mask.to(dtype=torch.bool)
    B, N = mask_bool.shape

    lengths = mask_bool.sum(dim=1).clamp_min(1)  # [B]

    base = torch.linspace(0, 1, steps=K, device=device)  # [K]
    pos_in_valid = (base[None, :] * (lengths[:, None] - 1).to(base.dtype)).round().long()  # [B,K]
    pos_in_valid = pos_in_valid.clamp_min(0)

    key = (~mask_bool).to(torch.int32)  # valid=0 invalid=1
    valid_order = torch.argsort(key, dim=1, stable=True)  # [B,N] 前 lengths[b] 个是 valid 的 index

    idx = valid_order.gather(1, pos_in_valid)
    return idx


import torch


@torch.no_grad()
def uniform_anchors_Sem(N: int, K: int, mask: torch.Tensor) -> torch.Tensor:
    """
    序列均匀采样 anchor 索引（纯 Batch 矩阵化版本）。

    Args:
        N: 原始序列长度。
        K: 需要采样的 anchor 数量。
        mask: [B, N] (0/1 或 bool)，表示哪些位置是有效的。

    Returns:
        idx: [B, K] long, 采样得到的原始序列索引。
    """
    device = mask.device
    B = mask.shape[0]
    mask_bool = mask.to(dtype=torch.bool)

    # 1. 计算每个 Batch 的有效长度
    # clamp_min(1) 用于处理全无效的特殊情况，防止除以 0
    lengths = mask_bool.sum(dim=1).to(dtype=torch.long).clamp_min(1)  # [B]

    # 2. 生成均匀步长映射 [B, K]
    # base 从 0 到 1 均匀分布
    base = torch.linspace(0, 1, steps=K, device=device).view(1, K)  # [1, K]

    # 映射到每个 batch 的有效索引范围 [0, lengths-1]
    # 使用 round() 模拟你原代码中的 linspace 采样逻辑
    pos_in_valid = (base * (lengths.view(B, 1) - 1).float()).round().long()  # [B, K]

    # 3. 将有效索引“挤”到左侧
    # 原理：对 (1 - mask) 进行稳定排序。
    # valid 位置(0) 会排在前面，invalid 位置(1) 会排在后面
    # stable=True 保证了 valid 位置内部的相对顺序不变
    sort_key = (~mask_bool).to(torch.int16)
    valid_indices_sorted = torch.argsort(sort_key, dim=1, stable=True)  # [B, N]

    # 4. 根据计算出的相对位置 gather 原始索引
    # pos_in_valid 对应的是在“所有有效点”里的第几个
    # 比如某 batch 有 5 个有效点，pos 分布在 [0, 4]，从 sorted 前 5 位里取值
    idx = torch.gather(valid_indices_sorted, 1, pos_in_valid)  # [B, K]

    # 5. 处理全无效的情况（兜底逻辑）
    # 如果某 batch 全为 0，lengths=1，pos_in_valid 全为 0，会取到原始索引 0
    # 这与你原代码中“退化为 0..K-1”略有不同，但更符合采样逻辑且不报错。

    return idx
@torch.no_grad()
def fps_anchors_mu(mu: torch.Tensor, mask: torch.Tensor, K: int) -> torch.Tensor:
    """
    更快的 FPS 实现：Batch 维度并行，矩阵化距离更新。
    mu:   [B, N, 3]
    mask: [B, N] (0/1 or bool)
    K:    采样点数
    """
    B, N, _ = mu.shape
    device = mu.device
    mask_bool = mask.to(torch.bool)

    # 1. 预计算：为了计算距离，无效点的距离初始化为无穷大
    # 在计算 min_d2 时，我们希望 invalid 点永远不被选中
    dist_inf = 1e10

    # 结果容器
    idx = torch.zeros(B, K, device=device, dtype=torch.long)

    # 2. 确定第一个种子点 (Seed)
    # 逻辑：距离 valid 均值最远的点
    denom = mask_bool.sum(dim=1, keepdim=True).clamp_min(1)
    mu_masked = mu * mask_bool.unsqueeze(-1)
    mean = mu_masked.sum(dim=1, keepdim=True) / denom.unsqueeze(-1)  # [B, 1, 3]

    d2_to_mean = torch.sum((mu - mean) ** 2, dim=-1)  # [B, N]
    # 排除 invalid 点：将其距离设为极小值，确保 argmax 选不到它
    d2_to_mean = d2_to_mean.masked_fill(~mask_bool, -1.0)

    first_idx = torch.argmax(d2_to_mean, dim=-1)  # [B]
    idx[:, 0] = first_idx

    # 3. 初始化最小距离场 (min_d2)
    # 存储每个点到当前已选集合的最小距离
    # [B, 1, 3]
    last_mu = torch.gather(mu, 1, first_idx.view(B, 1, 1).expand(-1, -1, 3))
    min_d2 = torch.sum((mu - last_mu) ** 2, dim=-1)  # [B, N]

    # 关键点：将 invalid 点的距离设为很小的值（如 -1），保证 argmax 选不到
    # 同时在更新时，它们也不会影响有效点的距离场
    min_d2 = min_d2.masked_fill(~mask_bool, -1.0)

    # 4. 迭代采样 (必须保留 K 的循环，但内部全矩阵化)
    #
    for t in range(1, K):
        # 选取当前距离场中最大的点
        selected_idx = torch.argmax(min_d2, dim=-1)  # [B]
        idx[:, t] = selected_idx

        # 提取新选点的坐标 [B, 1, 3]
        last_mu = torch.gather(mu, 1, selected_idx.view(B, 1, 1).expand(-1, -1, 3))

        # 计算所有点到这个新选点的距离 [B, N]
        new_d2 = torch.sum((mu - last_mu) ** 2, dim=-1)
        new_d2 = new_d2.masked_fill(~mask_bool, -1.0)

        # 更新全局最小距离场：取旧场和新距离的最小值
        # 注意：因为 invalid 是 -1，这里的 minimum 会保留 -1
        min_d2 = torch.where(min_d2 == -1.0, new_d2, torch.minimum(min_d2, new_d2))

    return idx
# 你已有：
# - uniform_anchors(mask, K) -> idx [B,K]
# - choose_K(N, ratio, k_min=1, k_max=...) -> int K
# - merge_gaussians_soft(mu, Sigma, A) -> (mu_c, Sigma_c)
# - gaussian_overlap_score(delta, sigma_sum) -> score  (或 fused_gaussian_overlap_score)
# - PoolLoss dataclass/NamedTuple: PoolLoss(occ, rep, ent, total)


import torch

@torch.no_grad()
def fps_points_batch(
    x: torch.Tensor,                 # [B,P,3]
    M: int,
    mask: torch.Tensor | None = None,# [B,P] 1/0
    init: str = "centroid_farthest", # "random" | "centroid_farthest"
) -> torch.Tensor:
    """
    Batch-parallel Farthest Point Sampling (FPS).
    Runs an O(M*P) loop (FPS is iterative) but fully parallel over batch on GPU.

    Returns:
      idx: [B,M] long
    """
    assert x.dim() == 3 and x.size(-1) == 3
    B, P, _ = x.shape
    device = x.device

    if mask is None:
        mask_bool = torch.ones((B, P), device=device, dtype=torch.bool)
    else:
        mask_bool = mask.to(device=device).to(torch.bool)

    # if some batch has no valid points, fall back to all-valid to avoid crash
    valid_cnt = mask_bool.sum(dim=1)  # [B]
    all_invalid = (valid_cnt == 0)
    if all_invalid.any():
        mask_bool = mask_bool.clone()
        mask_bool[all_invalid] = True
        valid_cnt = mask_bool.sum(dim=1)

    idx = torch.empty((B, M), device=device, dtype=torch.long)

    # ---- init pick ----
    if init == "random":
        # sample a valid index per batch
        # trick: add big negative to invalid, then argmax over random
        r = torch.rand((B, P), device=device)
        r = r.masked_fill(~mask_bool, -1.0)
        idx0 = torch.argmax(r, dim=1)  # [B]
    elif init == "centroid_farthest":
        # pick farthest valid point from masked centroid (more stable)
        w = mask_bool.to(x.dtype)
        denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
        centroid = (x * w[..., None]).sum(dim=1) / denom   # [B,3]
        d2 = ((x - centroid[:, None, :]) ** 2).sum(dim=-1) # [B,P]
        d2 = d2.masked_fill(~mask_bool, -1.0)
        idx0 = torch.argmax(d2, dim=1)
    else:
        raise ValueError(f"Unknown init={init}")

    idx[:, 0] = idx0

    # ---- running min distance to selected set ----
    # dists: [B,P] stores min_{selected} ||x - x_sel||^2
    sel = x.gather(1, idx0[:, None, None].expand(B, 1, 3)).squeeze(1)  # [B,3]
    dists = ((x - sel[:, None, :]) ** 2).sum(dim=-1)                   # [B,P]
    dists = dists.masked_fill(~mask_bool, -1.0)                        # invalid never chosen

    for t in range(1, M):
        far = torch.argmax(dists, dim=1)         # [B]
        idx[:, t] = far

        sel = x.gather(1, far[:, None, None].expand(B, 1, 3)).squeeze(1)
        d2 = ((x - sel[:, None, :]) ** 2).sum(dim=-1)
        # update min-dist
        dists = torch.minimum(dists, d2)
        dists = dists.masked_fill(~mask_bool, -1.0)

    return idx

@torch.no_grad()
def sample_from_gaussian_mixture(mu_p, Sig_p, pi, M, mask_parent=None, eps=1e-6):
    """
    mu_p: [B,K,3]
    Sig_p:[B,K,3,3]
    pi:   [B,K] normalized
    return:
      parent_id: [B,M]
      x:         [B,M,3]  samples
      Sig_k:     [B,M,3,3] sampled parent's Sigma (for later Sigma0 init)
    """
    B, K, _ = mu_p.shape
    device, dtype = mu_p.device, mu_p.dtype
    I = torch.eye(3, device=device, dtype=dtype)[None, None]

    if mask_parent is not None:
        pi = pi * (mask_parent > 0.5).to(pi.dtype)
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    parent_id = torch.multinomial(pi, num_samples=M, replacement=True)  # [B,M]

    mu_k = mu_p.gather(1, parent_id[..., None].expand(B, M, 3))                 # [B,M,3]
    Sig_k = Sig_p.gather(1, parent_id[..., None, None].expand(B, M, 3, 3))      # [B,M,3,3]
    Sig_k = 0.5 * (Sig_k + Sig_k.transpose(-1, -2)) + eps * I

    L = torch.linalg.cholesky(Sig_k)
    z = torch.randn((B, M, 3), device=device, dtype=dtype)
    x = mu_k + torch.einsum("bmij,bmj->bmi", L, z)

    return parent_id, x, Sig_k

class UniformAnchorSemGeoAssign(nn.Module):
    """
    用均匀 anchor (idx) 锁死 slot 身份，避免对称塌缩；
    A 由 “几何距离到 anchor center” + “语义相似到 anchor 语义” 得到。

    相比 LearnOnlyGaussianPooling：
    - 不再使用 learnable slot_embed 来做分配（可以保留 proj 做语义投影）
    - slot 语义原型 = anchor residue 的语义（s_anchor），天然绑定空间与语义
    - 几何项使用 mu 距离（提供空间结构信息）
    - 仍返回 A / s_c / mu_c / Sigma_c / losses，后续逻辑不变
    """
    def __init__(
        self,
        c_s: int,
        ratio: float = 12.0,
        k_max_cap: Optional[int] = None,

        # 语义投影维度（建议比 c_s 小，稳定尺度）
        sem_dim: int = 128,

        # A 中几何/语义的权重（你也可以外部 schedule）
        w_geo_init: float = 1.0,
        w_sem_init: float = 0.2,

        # 几何距离 softmax 的尺度（nm）
        sigma_nm_init: float = 1.0,

        # 温度（用于语义项可选缩放；几何项已经有 sigma）
        tau_sem_init: float = 1.0,

        eps: float = 1e-8,
    ):
        super().__init__()
        self.ratio = ratio
        self.k_max_cap = k_max_cap
        self.eps = eps

        # 可学习权重（方便你后面做 schedule；也可改成普通 float）
        self.w_geo = nn.Parameter(torch.tensor(float(w_geo_init)))
        self.w_sem = nn.Parameter(torch.tensor(float(w_sem_init)))
        self.sigma_nm = nn.Parameter(torch.tensor(float(sigma_nm_init)))
        self.tau_sem = nn.Parameter(torch.tensor(float(tau_sem_init)))

        # 语义投影：q 对 token，k 对 anchor token
        self.sem_q = nn.Linear(c_s, sem_dim, bias=False)
        self.sem_k = nn.Linear(c_s, sem_dim, bias=False)

        # 可选 LayerNorm，让尺度更稳
        self.sem_ln_q = nn.LayerNorm(sem_dim)
        self.sem_ln_k = nn.LayerNorm(sem_dim)

    def forward(
        self,
        s: torch.Tensor,                 # [B,N,C]
        mu: torch.Tensor,                # [B,N,3] (nm)
        Sigma: torch.Tensor,             # [B,N,3,3]
        mask: Optional[torch.Tensor] = None,  # [B,N] {0,1}
        w_occ: float = 1.0,
        w_rep: float = 0.1,
        w_ent: float = 0.0,
        rep_topk: int = 4,
        rep_margin: float = -1.0,
    ):
        B, N, C = s.shape
        device, dtype = s.device, s.dtype

        if mask is None:
            mask = torch.ones(B, N, device=device, dtype=dtype)
        else:
            mask = mask.to(device=device, dtype=dtype)

        # 1) choose K
        K = choose_K(N, ratio=self.ratio, k_min=1, k_max=self.k_max_cap)

        # 2) 均匀 anchor idx: [B,K]
        #    你已复制 uniform_anchors 函数
        idx = uniform_anchors(mask, K)

        # 3) gather anchor centers / anchor semantics
        mu_a = mu.gather(1, idx[..., None].expand(B, K, 3))           # [B,K,3]
        s_a  = s.gather(1, idx[..., None].expand(B, K, C))            # [B,K,C]

        # 4) 几何 logits：距离到 anchor（提供空间结构）
        #    dist2: [B,N,K]
        dist2 = ((mu[:, :, None, :] - mu_a[:, None, :, :]) ** 2).sum(dim=-1)

        sigma_nm = self.sigma_nm.clamp_min(1e-4)
        logits_geo = -dist2 / (2.0 * (sigma_nm ** 2))

        # 5) 语义 logits：与 anchor 语义的相似度（绑定语义，不再对称）
        q = self.sem_q(s)          # [B,N,D]
        k = self.sem_k(s_a)        # [B,K,D]
        q = self.sem_ln_q(q)
        k = self.sem_ln_k(k)

        q = F.normalize(q, dim=-1, eps=self.eps)
        k = F.normalize(k, dim=-1, eps=self.eps)

        logits_sem = torch.einsum("bnd,bkd->bnk", q, k)  # [-1,1] roughly

        # 可选温度缩放（只对语义项）
        tau_sem = self.tau_sem.clamp_min(1e-4)
        logits_sem = logits_sem / tau_sem

        # 6) 合并 logits，mask，再 softmax 得到 A
        w_geo_eff = self.w_geo
        w_sem_eff = self.w_sem
        logits = w_geo_eff * logits_geo + w_sem_eff * logits_sem
        logits = logits.masked_fill((mask < 0.5).unsqueeze(-1), -1e9)

        A = torch.softmax(logits, dim=-1)              # [B,N,K]
        A = A * mask.unsqueeze(-1)                     # [B,N,K] 保证 invalid residue 不贡献

        # 7) pooled semantic
        denom = A.sum(dim=1).clamp_min(1e-8)           # [B,K]
        s_c = torch.einsum("bnk,bnc->bkc", A, s) / denom.unsqueeze(-1)

        # 8) pooled gaussian
        mu_c, Sigma_c = merge_gaussians_soft(mu, Sigma, A)

        # --------------------------
        # losses（保留你原来的）
        # --------------------------

        # occupancy loss：鼓励每个 slot 有类似负载（可保留/可关）
        occ = A.sum(dim=1)  # [B,K]
        occ = occ / (occ.sum(dim=-1, keepdim=True).clamp_min(1e-8))
        target = torch.full_like(occ, 1.0 / max(K, 1))
        loss_occ = F.mse_loss(occ, target)

        # repulsion loss：防止 coarse 高斯重叠（你已有 overlap_score）
        if K <= 1:
            loss_rep = torch.zeros((), device=device, dtype=dtype)
        else:
            delta = mu_c.unsqueeze(2) - mu_c.unsqueeze(1)            # [B,K,K,3]
            sigma_sum = Sigma_c.unsqueeze(2) + Sigma_c.unsqueeze(1)  # [B,K,K,3,3]
            score = gaussian_overlap_score(delta, sigma_sum)         # [B,K,K]
            eye = torch.eye(K, device=device, dtype=torch.bool).unsqueeze(0)
            score = score.masked_fill(eye, -1e9)
            top = torch.topk(score, k=min(rep_topk, K - 1), dim=-1).values
            loss_rep = F.relu(top - rep_margin).mean()

        # entropy（可选；注意：你原来把 ent 当 loss_ent=ent，本质是鼓励更“散”/更“软”）
        ent = -(A.clamp_min(1e-8) * A.clamp_min(1e-8).log()).sum(dim=-1)  # [B,N]
        ent = (ent * mask).sum() / mask.sum().clamp_min(1.0)
        loss_ent = ent

        total = w_occ * loss_occ + w_rep * loss_rep + w_ent * loss_ent
        losses = PoolLoss(occ=loss_occ, rep=loss_rep, ent=loss_ent, total=total)

        return A, s_c, mu_c, Sigma_c, losses
# -------------------------
# 关键模块：UniformAnchorSemGeoAssign
# -------------------------
class UniformAnchorSemAssign(nn.Module):
    """
    稳定可训练的下采样 pooling：
    - Anchor/window 只做破对称（限定竞争范围）
    - 语义 logits 决定 A，所以不会固定比例（6/8/3/0 自然发生）
    - 可选 mu-geo bias（建议先很小 or 先关）
    """

    def __init__(
        self,
        c_s: int,
        ratio: float = 12.0,
        k_max_cap: Optional[int] = None,

        # 语义分支
        tau_init: float = 0.2,
        slots_init_scale: float = 0.02,

        # window：每个 residue 只允许分给附近多少个 anchor
        # 推荐：R=1~2（越大越接近全连接，越容易塌缩）
        neighbor_R: int = 2,

        # 几何分支（用 mu 距离更直接、更有结构信息）
        use_mu_geo: bool = True,
        sigma2_init: float = 0.25,   # nm^2，控制距离logits尺度
        gamma_init: float = 0.0,     # 先 0；跑通后再慢慢加到 0.1~0.5

        eps: float = 1e-8,
    ):
        super().__init__()
        self.c_s = c_s
        self.ratio = ratio
        self.k_max_cap = k_max_cap

        self.tau = tau_init
        self.neighbor_R = neighbor_R
        self.eps = eps

        # semantic
        self.proj = nn.Linear(c_s, c_s)
        self.slot_embed: Optional[nn.Parameter] = None
        self._slot_K: Optional[int] = None
        self._slots_init_scale = slots_init_scale

        # geo (mu)
        self.use_mu_geo = use_mu_geo
        self.log_sigma2 = nn.Parameter(torch.log(torch.tensor(float(sigma2_init))))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _ensure_slots(self, K: int, C: int, device, dtype):
        if (self.slot_embed is None) or (self._slot_K != K) or (self.slot_embed.shape[-1] != C):
            self._slot_K = K
            slots = torch.randn(K, C, device=device, dtype=dtype) * self._slots_init_scale
            self.slot_embed = nn.Parameter(slots)

    @staticmethod
    def _make_local_slot_mask_from_anchors(
        N: int, K: int, anchor_idx: torch.Tensor, neighbor_R: int
    ) -> torch.Tensor:
        """
        构造 local allow mask: [B,N,K] bool
        对每个 residue i，只允许分配给“距离最近的 R 个 anchor slot”
        """
        B = anchor_idx.shape[0]
        device = anchor_idx.device

        i_idx = torch.arange(N, device=device)[None, :, None].expand(B, N, K)          # [B,N,K]
        a_idx = anchor_idx[:, None, :].expand(B, N, K)                                  # [B,N,K]
        dist = (i_idx - a_idx).abs()                                                    # [B,N,K]

        # 取最近 R 个 slot
        R = min(max(neighbor_R, 1), K)
        top = torch.topk(dist, k=R, dim=-1, largest=False).indices                      # [B,N,R]

        allow = torch.zeros((B, N, K), device=device, dtype=torch.bool)
        allow.scatter_(dim=-1, index=top, value=True)
        return allow

    def forward(
        self,
        s: torch.Tensor,                 # [B,N,C]
        mu: torch.Tensor,                # [B,N,3]
        Sigma: torch.Tensor,             # [B,N,3,3]  (这里不强依赖，保留给 merge)
        mask: Optional[torch.Tensor] = None,  # [B,N] {0,1}
        w_occ: float = 1.0,
        w_rep: float = 0.1,
        w_ent: float = 0.0,
        rep_topk: int = 4,
        rep_margin: float = -1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PoolLoss, torch.Tensor]:
        """
        Returns:
          A:      [B,N,K]
          s_c:    [B,K,C]
          mu_c:   [B,K,3]
          Sigma_c:[B,K,3,3]
          losses: PoolLoss
          anchor_idx: [B,K]
        """
        B, N, C = s.shape
        device, dtype = s.device, s.dtype
        if mask is None:
            mask = torch.ones((B, N), device=device, dtype=dtype)

        # ---- choose K ----
        K = choose_K(N, ratio=self.ratio, k_min=1, k_max=self.k_max_cap)
        self._ensure_slots(K, C, device, dtype)

        # ---- anchors + local window allow mask ----
        anchor_idx = uniform_anchors( mask, K)                        # [B,K]
        allow = self._make_local_slot_mask_from_anchors(N, K, anchor_idx, self.neighbor_R)  # [B,N,K]

        # ---- semantic logits ----
        s_proj = self.proj(s)                                           # [B,N,C]
        slots = self.slot_embed.unsqueeze(0).expand(B, -1, -1)          # [B,K,C]
        logits_sem = torch.einsum("bnc,bkc->bnk", s_proj, slots) / max(self.tau, 1e-6)

        # ---- geo logits (mu-distance to anchor points) ----
        if self.use_mu_geo:
            # anchor_mu: [B,K,3]
            anchor_mu = mu.gather(1, anchor_idx[..., None].expand(-1, -1, 3))
            dist2 = ((mu[:, :, None, :] - anchor_mu[:, None, :, :]) ** 2).sum(dim=-1)  # [B,N,K]
            sigma2 = torch.exp(self.log_sigma2).clamp_min(1e-6)
            logits_geo = -dist2 / (2.0 * sigma2)
        else:
            logits_geo = torch.zeros((B, N, K), device=device, dtype=dtype)

        # ---- combine + masks ----
        logits = logits_sem + self.gamma * logits_geo

        # 1) padding mask
        logits = logits.masked_fill((mask < 0.5).unsqueeze(-1), -1e9)
        # 2) local window mask（关键：破对称 + 稳定）
        logits = logits.masked_fill(~allow, -1e9)

        A = F.softmax(logits, dim=-1)                                   # [B,N,K]
        A = A * mask.unsqueeze(-1)

        # ---- aggregate s ----
        denom = A.sum(dim=1).clamp_min(1e-8)                            # [B,K]
        s_c = torch.einsum("bnk,bnc->bkc", A, s) / denom.unsqueeze(-1)

        # ---- merge Gaussians (用你的实现) ----
        # 你应当已有：mu_c, Sigma_c = merge_gaussians_soft(mu, Sigma, A)
        mu_c, Sigma_c = merge_gaussians_soft(mu, Sigma, A)

        # ---- losses ----
        # occupancy：鼓励不要全挤一个，也不要平均死（建议用 MSE 到均匀分布，权重小一点）
        occ = A.sum(dim=1)                                              # [B,K]
        occ = occ / (occ.sum(dim=-1, keepdim=True).clamp_min(1e-8))
        target = torch.full_like(occ, 1.0 / max(K, 1))
        loss_occ = F.mse_loss(occ, target)

        # repulsion：用你已有的 overlap score（建议用 log overlap，且只topk）
        if K <= 1:
            loss_rep = torch.zeros((), device=device, dtype=dtype)
        else:
            delta = mu_c.unsqueeze(2) - mu_c.unsqueeze(1)               # [B,K,K,3]
            sigma_sum = Sigma_c.unsqueeze(2) + Sigma_c.unsqueeze(1)     # [B,K,K,3,3]

            # 你如果有 fused_gaussian_overlap_score 就用它
            # score 越接近 0 越重叠，越负越不重叠
            score = fused_gaussian_overlap_score(delta, sigma_sum)      # [B,K,K]

            eye = torch.eye(K, device=device, dtype=torch.bool).unsqueeze(0)
            score = score.masked_fill(eye, -1e9)

            top = torch.topk(score, k=min(rep_topk, K - 1), dim=-1).values
            # 惩罚“太重叠”（score 接近 0）
            loss_rep = F.relu(top - rep_margin).mean()

        # entropy：早期可 0，后期开一点让 A 更“硬”
        ent = -(A.clamp_min(1e-8) * A.clamp_min(1e-8).log()).sum(dim=-1) # [B,N]
        ent = (ent * mask).sum() / mask.sum().clamp_min(1.0)
        loss_ent = ent

        total = w_occ * loss_occ + w_rep * loss_rep + w_ent * loss_ent
        losses = PoolLoss(occ=loss_occ, rep=loss_rep, ent=loss_ent, total=total)

        return A, s_c, mu_c, Sigma_c, losses, anchor_idx
# =========================
# 2) 打包：fine 后“下采样一切相关”模块
# =========================
@dataclass
class DownsampleOut:
    s: torch.Tensor                 # [B,K,C]
    rigids: "OffsetGaussianRigid"   # [B,K]
    mask: torch.Tensor              # [B,K]
    A: torch.Tensor                 # [B,N,K] (可选用于debug/跨尺度)
    pool_loss: PoolLoss




def _make_so3(R: torch.Tensor) -> torch.Tensor:
    """
    Force det(R)=+1 by flipping the 3rd column if needed.
    R: [...,3,3]
    """
    det = torch.linalg.det(R)  # [...]
    flip = (det < 0).to(R.dtype)  # [...]
    # flip last column
    R = R.clone()
    R[..., :, 2] = R[..., :, 2] * (1.0 - 2.0 * flip).unsqueeze(-1)
    return R

def _sign_fix_with_prev(R: torch.Tensor, prev_R: torch.Tensor) -> torch.Tensor:
    """
    Reduce eigenvector sign ambiguity by aligning columns with prev_R.
    R, prev_R: [...,3,3]
    """
    # column-wise dot
    dots = (R * prev_R).sum(dim=-2)  # [...,3]  (sum over rows)
    sgn = torch.where(dots >= 0, torch.ones_like(dots), -torch.ones_like(dots))  # [...,3]
    return R * sgn.unsqueeze(-2)

def coarse_rigids_from_mu_sigma(mu_c, Sigma_c, OffsetGaussianRigid_cls, eps=1e-6):
    """
    mu_c:    [B, K, 3]
    Sigma_c: [B, K, 3, 3]
    return:  OffsetGaussianRigid [B, K]
    """
    B, K, _ = mu_c.shape
    device, dtype = mu_c.device, mu_c.dtype

    # 1) R = Identity
    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(B, K, 3, 3)
    rots = du.Rotation(rot_mats=eye) if hasattr(du, "Rotation") else Rotation(rot_mats=eye)  # 兼容你的 Rotation 定义

    # 2) scaling = sqrt(diag(Sigma))
    tr = torch.diagonal(Sigma_c, dim1=-2, dim2=-1)  # [B,K,3]
    var = (tr / 3.0).clamp_min(eps)
    scale = torch.sqrt(var)  # [B,K]
    scaling_log = torch.log(scale)

    # 3) local_mean = 0 (因为 trans 就是 mu)
    local_mean = torch.zeros_like(mu_c)

    return OffsetGaussianRigid_cls(
        rots=rots,
        trans=mu_c,
        scaling_log=scaling_log,
        local_mean=local_mean,
    )