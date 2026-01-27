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
from models.kernel.kernel import cov_to_R_scale_no_eigh_robust

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


# -------------------------
# 1) teacher 分段：变长 4~10
#    支持链断点（chain_idx变了就强制新段）
# -------------------------
@torch.no_grad()
def teacher_segment_variable_length(
    node_mask: torch.Tensor,          # [B,N] 0/1
    chain_idx: Optional[torch.Tensor] = None,  # [B,N] int (可选)
    min_len: int = 4,
    max_len: int = 10,
    Kmax: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回:
      a_idx: [B,N]       residue -> parent id (0..Kmax-1)
      mask_parent:[B,Kmax]  启用的 parent
      seg_lens: [B,Kmax] 段长
    """
    B, N = node_mask.shape
    device = node_mask.device

    a_idx = torch.zeros((B, N), device=device, dtype=torch.long)
    mask_parent = torch.zeros((B, Kmax), device=device, dtype=torch.float32)
    seg_lens = torch.zeros((B, Kmax), device=device, dtype=torch.long)

    for b in range(B):
        n = int(node_mask[b].sum().item())
        if n <= 0:
            continue

        pos = 0
        k = 0
        while pos < n and k < Kmax:
            # 链断点：如果 chain_idx 有变化，强制从这里开新段
            if chain_idx is not None and pos > 0:
                if chain_idx[b, pos].item() != chain_idx[b, pos - 1].item():
                    # 直接开新段（不改变 pos，只是 k++)
                    pass

            L = torch.randint(low=min_len, high=max_len + 1, size=(1,), device=device).item()
            L = min(L, n - pos)

            # 如果有 chain_idx，别跨链
            if chain_idx is not None:
                c0 = chain_idx[b, pos].item()
                # 找到最长不跨链长度
                maxL = 1
                for t in range(pos + 1, pos + L):
                    if chain_idx[b, t].item() != c0:
                        break
                    maxL += 1
                L = maxL

            a_idx[b, pos:pos + L] = k
            mask_parent[b, k] = 1.0
            seg_lens[b, k] = L
            pos += L
            k += 1

        # 如果 n 没覆盖完（Kmax 太小）：把剩余塞到最后一个
        if pos < n:
            last = Kmax - 1
            a_idx[b, pos:n] = last
            mask_parent[b, last] = 1.0
            seg_lens[b, last] += (n - pos)

    return a_idx, mask_parent, seg_lens


@dataclass
class ParentParams:
    mu: torch.Tensor
    R: torch.Tensor
    s: torch.Tensor
    mask_parent: torch.Tensor



def build_parents_from_A_soft(
    x: torch.Tensor,          # [B,N,3]
    A: torch.Tensor,          # [B,N,K]
    node_mask: torch.Tensor,  # [B,N]
    jitter: float = 1e-6,
    eps: float = 1e-8,
        # ---- mask_parent threshold knobs ----
        occ_thresh_abs: float = 1e-4,  # 绝对阈值（适合 hard-cutoff 后的“严格 0”列）
        occ_thresh_rel: float = 0.05,  # 相对阈值：相对 (n_valid/K) 的比例
):
    """
    返回：
      mu:        [B,K,3]
      cov:       [B,K,3,3]  (仍返回，方便 debug/兼容旧代码)
      R:         [B,K,3,3]  (主轴旋转，det=+1)
      scale:     [B,K,3]    (半轴尺度 = sqrt(eigvals))
      eigvals:   [B,K,3]    (方差特征值，降序)
      occ:       [B,K]
      mask_soft: [B,K]
    """
    B, N, _ = x.shape
    _, _, K = A.shape
    dtype, device = x.dtype, x.device

    # mask padded residues
    m = node_mask.to(dtype=dtype).unsqueeze(-1)      # [B,N,1]
    A = A.to(dtype=dtype) * m                        # [B,N,K]
    # occ = A.sum(dim=1).clamp_min(eps)                # [B,K]

    # ---- occupancy ----
    occ_raw = A.sum(dim=1)  # [B,K]  (不 clamp)
    occ = occ_raw.clamp_min(eps)  # [B,K]  (仅用于除法安全)

    # ---- build hard mask_parent ----
    # 每个样本的有效残基数
    n_valid = node_mask.to(dtype=dtype).sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
    # “均匀分配”下每个 parent 的期望长度
    occ_expect = (n_valid / max(K, 1))               # [B,1]

    # 阈值：abs + rel * expect
    thr = (occ_thresh_abs + occ_thresh_rel * occ_expect)  # [B,1]
    mask_parent = (occ_raw > thr).to(dtype=dtype)         # [B,K] 0/1

    # 可选：保证至少有 1 个 parent 激活（极端情况防炸）
    # 如果全 0，就强行把最大 occ 的那个置 1
    all_zero = (mask_parent.sum(dim=-1, keepdim=True) < 0.5)  # [B,1] bool
    if all_zero.any():
        k_best = occ_raw.argmax(dim=-1)  # [B]
        mask_parent = mask_parent.clone()
        mask_parent[torch.arange(B, device=device), k_best] = 1.0

    # mean
    mu = torch.einsum("bnk,bnd->bkd", A, x) / occ.unsqueeze(-1)  # [B,K,3]

    # covariance (raw, no jitter here)
    xc = x.unsqueeze(2) - mu.unsqueeze(1)            # [B,N,K,3]
    cov = torch.einsum("bnk,bnkd,bnke->bkde", A, xc, xc) / occ.unsqueeze(-1).unsqueeze(-1)
    cov = 0.5 * (cov + cov.transpose(-1, -2))        # 强制对称（jitter 交给下游）

    # 从 cov 得到 R 和 scale（不调用 eigh）
    # 注意：你传的 eps=1e-8 对闭式/归一化来说略大但没问题；这里做个下限，避免 eps 太大影响精度
    eps_work = max(float(eps), 1e-12)
    R, scale, eigvals = cov_to_R_scale_no_eigh_robust(
        cov, jitter=jitter, eps=eps_work
    )  # R:[B,K,3,3], scale:[B,K,3], eigvals:[B,K,3]

    # 再做一次 mask（避免未使用列被 jitter 撑起来）
    scale = scale * mask_parent[:, :, None]
    # 对 unused 列，给一个安全的最小 scale（否则 log(0)）
    scale = scale + (1.0 - mask_parent[:, :, None]) * 1e-3

    # ---- soft usedness（0~1） ----
    occ_max = occ_raw.max(dim=-1, keepdim=True).values.clamp_min(eps)
    mask_soft = (occ_raw / occ_max) * mask_parent  # [B,K] 仅在 hard mask 内才有 soft 值

    r_parent = OffsetGaussianRigid(
        rots=Rotation(rot_mats=R),
        trans=mu,
        scaling_log=torch.log(scale.clamp_min(1e-6)),
        local_mean=torch.zeros_like(mu),
    )

    return r_parent, occ,mask_parent, mask_soft


def build_parents_from_segments_v3_debug(
        x_ca: torch.Tensor,
        node_mask: torch.Tensor,
        a_idx: torch.Tensor,
        mask_parent: torch.Tensor,
        Kmax: int,
        eps: float = 1e-6,
) -> ParentParams:
    # -------------------------------------------------------------------------
    # PART 1: 计算逻辑 (保持你要求的 Exact Fit 算法不变)
    # -------------------------------------------------------------------------
    B, N, _ = x_ca.shape
    device, dtype = x_ca.device, x_ca.dtype
    K = Kmax

    x = x_ca * node_mask[..., None]
    oh = F.one_hot(a_idx.clamp(0, K - 1).long(), num_classes=K).to(dtype)
    oh = oh * node_mask[..., None]
    w = oh.permute(0, 2, 1)  # [B, K, N]
    cnt = w.sum(dim=2).clamp_min(1.0)

    # Center
    mu = torch.einsum("bkn,bnj->bkj", w, x) / cnt[..., None]

    # Covariance & Scatter
    delta = x.unsqueeze(1) - mu.unsqueeze(2)
    delta = delta * w.unsqueeze(-1)
    scatter = torch.einsum("bkni,bknj->bkij", delta, delta)
    denom = (cnt - 1.0).clamp_min(1.0)
    cov = scatter / denom[..., None, None]

    # Eigendecomposition (Ascending: s[0]=Small, s[2]=Large)
    I = torch.eye(3, device=device, dtype=dtype)[None, None]
    cov_safe = cov + I * eps

    with torch.no_grad():
        eigvals, eigvecs = torch.linalg.eigh(cov_safe)
        eigvals = eigvals.detach()
        eigvecs = eigvecs.detach()

    Rk = eigvecs
    det = torch.det(Rk)
    flip = (det < 0).float().unsqueeze(-1).unsqueeze(-1)
    # Fix handedness
    col0 = Rk[..., 0] * (1.0 - 2.0 * flip.squeeze(-1))
    Rk = torch.stack([col0, Rk[..., 1], Rk[..., 2]], dim=-1)

    # Exact Fit Scaling
    local_coords = torch.einsum("bkni,bkij->bknj", delta, Rk)
    std_devs_sq = eigvals.clamp_min(eps).unsqueeze(2)
    norm_dist_sq = (local_coords ** 2 / std_devs_sq).sum(dim=-1)

    # Mask invalid points for max calculation
    valid_dist = torch.where(w > 0.5, norm_dist_sq, torch.tensor(-1.0, device=device))
    max_sq_val = valid_dist.max(dim=2).values.clamp_min(eps)
    scale_factor = torch.sqrt(max_sq_val)

    # Final Radii
    s = torch.sqrt(eigvals.clamp_min(eps)) * scale_factor.unsqueeze(-1)

    # Apply Masks
    valid_mask = mask_parent[..., None, None]
    mu_final = mu * mask_parent[..., None]
    s_final = s * mask_parent[..., None]
    Rk_final = Rk * valid_mask + I * (1.0 - valid_mask)

    # -------------------------------------------------------------------------
    # PART 2: 内部可视化 (DEBUG PLOTTING - Batch 0 All Parents)
    # -------------------------------------------------------------------------
    # 只有在非训练模式或者想强制看的时候才跑这个
    if False:
        print(f"\n[DEBUG] Plotting all parents for Batch 0 (Total K={K})...")

        # 准备数据 (转 numpy)
        b_idx = 0
        points_all = x[b_idx].detach().cpu().numpy()  # [N, 3]
        labels = a_idx[b_idx].detach().cpu().numpy()  # [N]

        mu_np = mu_final[b_idx].detach().cpu().numpy()  # [K, 3]
        R_np = Rk_final[b_idx].detach().cpu().numpy()  # [K, 3, 3]
        s_np = s_final[b_idx].detach().cpu().numpy()  # [K, 3]
        mask_p_np = mask_parent[b_idx].detach().cpu().numpy()  # [K]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 1. 绘制所有原始点 (按归属上色)
        # 使用 colormap 区分不同的 parent
        cmap = plt.get_cmap('tab10')
        for n_i in range(len(points_all)):
            if node_mask[b_idx, n_i] > 0.5:
                cluster_id = int(labels[n_i])
                ax.scatter(points_all[n_i, 0], points_all[n_i, 1], points_all[n_i, 2],
                           color=cmap(cluster_id % 10), s=40, alpha=0.6)

        # 2. 遍历每一个 Parent 进行绘制
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        x_unit = np.outer(np.cos(u), np.sin(v))
        y_unit = np.outer(np.sin(u), np.sin(v))
        z_unit = np.outer(np.ones_like(u), np.cos(v))

        print(mu_np)
        print(s_np)
        print(R_np)

        for k in range(K):
            if mask_p_np[k] < 0.5:
                continue  # 跳过无效 Parent

            center = mu_np[k]
            radii = s_np[k]  # [s0, s1, s2] (从小到大)
            rot = R_np[k]  # [col0, col1, col2]

            # --- A. 画椭球 Wireframe ---
            # 缩放: local_x * s0, local_y * s1, local_z * s2
            x_loc = x_unit * radii[0]
            y_loc = y_unit * radii[1]
            z_loc = z_unit * radii[2]

            # 旋转 & 平移
            # shape: [3, num_points]
            coords = np.stack([x_loc.flatten(), y_loc.flatten(), z_loc.flatten()])
            coords_world = (rot @ coords).T + center

            Xw = coords_world[:, 0].reshape(x_unit.shape)
            Yw = coords_world[:, 1].reshape(y_unit.shape)
            Zw = coords_world[:, 2].reshape(z_unit.shape)

            color = cmap(k % 10)
            ax.plot_wireframe(Xw, Yw, Zw, color=color, alpha=0.3)

            # --- B. 画中心点 ---
            ax.scatter(center[0], center[1], center[2], color='k', marker='x', s=100)

            # --- C. 画最长轴 (Direction) ---
            # s[2] 是最大的特征值 (长轴), R的第2列 (index 2) 是对应的方向
            long_axis_vec = rot[:, 2] * radii[2] * 1.1
            ax.quiver(center[0], center[1], center[2],
                      long_axis_vec[0], long_axis_vec[1], long_axis_vec[2],
                      color=color, linewidth=2, linestyle='dashed')

        # 3. 调整视角
        # 简单的 Box Aspect 修正
        all_coords = points_all[node_mask[b_idx].bool().cpu().numpy()]
        if len(all_coords) > 0:
            max_range = (all_coords.max(0) - all_coords.min(0)).max() / 2.0
            mid = (all_coords.max(0) + all_coords.min(0)) * 0.5
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(f"In-Function Visualization (Batch 0, {K} Parents)")
        plt.show()  # 阻塞住，直到你关闭窗口
        save_path = "debug_ellipse_v3.png"
        plt.savefig(save_path, dpi=300)

    return ParentParams(mu=mu_final, R=Rk_final, s=s_final, mask_parent=mask_parent)


import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 1) 你现成的 mean-pool（原样保留）
# =========================
def segment_mean_pool_s(
    s: torch.Tensor,              # [B,N,C]
    node_mask: torch.Tensor,      # [B,N] 0/1
    a_idx: torch.Tensor,          # [B,N] long in [0..Kmax-1]
    mask_parent: torch.Tensor,    # [B,Kmax] 0/1
    Kmax: int,
    eps: float = 1e-8,
):
    """
    returns:
      s_parent: [B,Kmax,C]
      occ:      [B,Kmax]   (#res in each seg, masked)
      A:        [B,N,Kmax] hard one-hot (masked)
    """
    B, N, C = s.shape
    dtype = s.dtype

    m = node_mask.to(dtype=dtype)
    oh = F.one_hot(a_idx.clamp_min(0).clamp_max(Kmax - 1), num_classes=Kmax).to(dtype)  # [B,N,K]
    A = oh * m.unsqueeze(-1)  # [B,N,K]  (hard assignment + pad masked)

    occ = A.sum(dim=1)  # [B,K]
    denom = occ.clamp_min(eps).unsqueeze(-1)  # [B,K,1]

    s_parent = torch.einsum("bnk,bnc->bkc", A, s) / denom  # [B,K,C]

    # mask unused parents
    s_parent = s_parent * mask_parent.to(dtype=dtype).unsqueeze(-1)
    return s_parent, occ, A


# =========================
# 2) 段内相对位置 pos01（段内从 0 开始，归一化到 [0,1]）
#    假设 teacher 分段是沿序号连续的
# =========================
@torch.no_grad()
def segment_pos01_from_assignment(
    a_idx: torch.Tensor,       # [B,N] long
    node_mask: torch.Tensor,   # [B,N] 0/1
    Kmax: int,
    eps: float = 1e-8,
):
    """
    返回:
      pos01:   [B,N] float in [0,1]
      seg_len: [B,Kmax] long (unused parents = 0)
    """
    B, N = a_idx.shape
    device = a_idx.device

    pos01 = torch.zeros((B, N), device=device, dtype=torch.float32)
    seg_len = torch.zeros((B, Kmax), device=device, dtype=torch.long)

    m = node_mask.bool()
    for b in range(B):
        n = int(m[b].sum().item())
        if n <= 0:
            continue

        ids = a_idx[b, :n]  # [n]

        # change points where segment id changes
        change = torch.ones(n, device=device, dtype=torch.bool)
        change[1:] = ids[1:] != ids[:-1]
        starts = torch.nonzero(change, as_tuple=False).squeeze(-1)          # [S]
        ends = torch.cat([starts[1:], torch.tensor([n], device=device)])    # end-exclusive

        for s0, e0 in zip(starts.tolist(), ends.tolist()):
            k = int(ids[s0].item())
            L = e0 - s0
            if 0 <= k < Kmax:
                seg_len[b, k] = L

            if L <= 1:
                pos01[b, s0:e0] = 0.0
            else:
                t = torch.arange(L, device=device, dtype=torch.float32) / (L - 1.0 + eps)
                pos01[b, s0:e0] = t

    return pos01, seg_len


# =========================
# 3) 轻量 Fourier 位置编码：pos01 -> [B,N,C]
# =========================
class Pos01FourierEncoder(nn.Module):
    """
    输入 pos01 ∈ [0,1]，输出 [B,N,C]
    """
    def __init__(self, c_s: int, n_freq: int = 16, trainable: bool = False):
        super().__init__()
        self.c_s = int(c_s)
        self.n_freq = int(n_freq)

        # 2^k 频率（稳定、足够表达“段内形状”）
        freq = 2.0 ** torch.arange(self.n_freq, dtype=torch.float32)  # [F]
        self.register_buffer("freq", freq, persistent=False)

        feat_dim = 2 * self.n_freq
        self.proj = nn.Linear(feat_dim, self.c_s, bias=False)

        self.log_scale = nn.Parameter(torch.zeros(())) if trainable else None
        self.ln = nn.LayerNorm(self.c_s)

    def forward(self, pos01: torch.Tensor, node_mask: torch.Tensor):
        """
        pos01: [B,N] float
        node_mask: [B,N] 0/1
        return: [B,N,C]
        """
        device = pos01.device
        m = node_mask.to(dtype=pos01.dtype)

        x = pos01.clamp(0.0, 1.0).unsqueeze(-1)  # [B,N,1]
        freq = self.freq.to(device=device)       # [F]
        if self.log_scale is not None:
            freq = freq * torch.exp(self.log_scale)

        ang = 2.0 * torch.pi * x * freq.view(1, 1, -1)  # [B,N,F]
        feat = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [B,N,2F]

        out = self.proj(feat)
        out = out * m.unsqueeze(-1)
        out = self.ln(out)
        return out


# =========================
# 4) 组合：Segment pooling + 段内相对位置编码注入
# =========================
class SegmentPoolingWithPosEnc(nn.Module):
    """
    用法：
      pool = SegmentPoolingWithPosEnc(c_s=C, Kmax=Kmax, n_freq=16, pos_weight=1.0)
      s_parent, occ, A, aux = pool(s, node_mask, a_idx, mask_parent)

    说明：
      - pos01 是“每段从 0 开始”的段内相对位置（归一化到 0..1）
      - 我们把 pos_emb 加到 s 上再做 mean-pool
    """
    def __init__(self, c_s: int, Kmax: int, n_freq: int = 16, pos_weight: float = 1.0, trainable_pos_scale: bool = False):
        super().__init__()
        self.Kmax = int(Kmax)
        self.pos_weight = float(pos_weight)
        self.pos_enc = Pos01FourierEncoder(c_s=c_s, n_freq=n_freq, trainable=trainable_pos_scale)

    def forward(
        self,
        s: torch.Tensor,              # [B,N,C]
        node_mask: torch.Tensor,      # [B,N]
        a_idx: torch.Tensor,          # [B,N]
        mask_parent: torch.Tensor,    # [B,Kmax]
        eps: float = 1e-8,
    ):
        # 1) 段内 pos01
        pos01, seg_len = segment_pos01_from_assignment(
            a_idx=a_idx,
            node_mask=node_mask,
            Kmax=mask_parent.shape[-1],
        )
        pos01 = pos01.to(device=s.device)  # float32

        # 2) pos embedding -> 注入到语义
        pos_emb = self.pos_enc(pos01, node_mask).to(dtype=s.dtype)  # [B,N,C]
        s_aug = s + (self.pos_weight * pos_emb)

        # 3) mean pool
        s_parent, occ, A = segment_mean_pool_s(
            s=s_aug,
            node_mask=node_mask,
            a_idx=a_idx,
            mask_parent=mask_parent,
            Kmax=mask_parent.shape[-1],
            eps=eps,
        )

        aux = {
            "pos01": pos01,     # [B,N] float32
            "seg_len": seg_len, # [B,K] long
        }
        return s_parent, occ, A, aux


# ============================================================
# 0) 从 a_idx 生成 teacher break 标签（最省事、最稳）
#    break[i]=1 表示 i 和 i+1 之间切段
# ============================================================



# ============================================================
# 1) 可学习 segmenter head：预测 break_logits [B,N-1]
#    输入只用 s_out (以及可选 z_local)
# ============================================================
class SegmentBreakHead(nn.Module):
    """
    预测每个 i,i+1 的切分概率。默认只用语义 s。
    你也可以把 z_local[i,i+1] 拼进去。
    """
    def __init__(self, c_s: int, c_z: int = 0, hidden: int = 256, use_z: bool = False):
        super().__init__()
        self.use_z = bool(use_z)
        in_dim = 2 * c_s + (c_z if self.use_z else 0)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        s: torch.Tensor,                 # [B,N,C]
        node_mask: torch.Tensor,         # [B,N]
        z_f: torch.Tensor | None = None, # [B,N,N,Cz]
    ):
        B, N, C = s.shape
        dtype = s.dtype

        m_pair = node_mask[:, :-1] * node_mask[:, 1:]  # [B,N-1]

        s_i = s[:, :-1, :]
        s_j = s[:, 1:, :]
        feat = [s_i, s_j]

        if self.use_z:
            assert z_f is not None, "use_z=True but z_f is None"
            # 取局部边 (i,i+1)
            z_ij = z_f[:, :-1, 1:, :]            # [B,N-1,N-1,Cz] 不是我们要的
            # 正确取对角 (i,i+1)
            # z_f: [B,N,N,Cz] -> z_local[b,i] = z_f[b,i,i+1]
            z_local = z_f[:, torch.arange(N-1), torch.arange(1, N), :]  # [B,N-1,Cz]
            feat.append(z_local)

        x = torch.cat(feat, dim=-1)  # [B,N-1,dim]
        logits = self.mlp(x).squeeze(-1)  # [B,N-1]
        logits = logits * m_pair.to(dtype=dtype)  # mask 掉无效
        return logits


def soft_pos01_from_A(
    A: torch.Tensor,          # [B,N,K] row-stochastic (masked already ok)
    node_mask: torch.Tensor,  # [B,N] 0/1
    eps: float = 1e-8,
):
    """
    returns:
      pos01: [B,N] in [0,1]  (soft segment-relative position)
      Lk:    [B,K]          (soft lengths)
    """
    dtype = A.dtype
    m = node_mask.to(dtype=dtype).unsqueeze(-1)            # [B,N,1]
    A = A * m                                              # [B,N,K]

    # soft length per segment
    Lk = A.sum(dim=1)                                      # [B,K]

    # cum along sequence (inclusive)
    cum = torch.cumsum(A, dim=1)                           # [B,N,K]
    rank = cum - A                                         # [B,N,K]  ~ sum_{t<i} A[t,k]

    denom = (Lk - 1.0).clamp_min(eps).unsqueeze(1)         # [B,1,K]
    pos_k = rank / denom                                   # [B,N,K] in [0,1] approximately

    # expected pos under A(i,k)
    pos01 = (A * pos_k).sum(dim=-1)                        # [B,N]
    pos01 = pos01 * node_mask.to(dtype=dtype)
    return pos01, Lk

def soft_pool_s(
    s: torch.Tensor,          # [B,N,C]
    node_mask: torch.Tensor,  # [B,N]
    A: torch.Tensor,          # [B,N,K]
    eps: float = 1e-8,
):
    dtype = s.dtype
    m = node_mask.to(dtype=dtype).unsqueeze(-1)
    A = A.to(dtype=dtype) * m                               # [B,N,K]
    occ = A.sum(dim=1)                                      # [B,K]
    denom = occ.clamp_min(eps).unsqueeze(-1)                # [B,K,1]
    s_parent = torch.einsum("bnk,bnc->bkc", A, s) / denom   # [B,K,C]
    return s_parent, occ, A

class SoftSegmentPoolingWithPosEnc(nn.Module):
    """
    输入 A_soft（可导）而不是 a_idx（离散）。
    做：
      pos01_soft(A_soft) -> pos_emb -> s_aug -> soft_pool
    """
    def __init__(self, c_s: int, n_freq: int = 16, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.pos_enc = Pos01FourierEncoder(c_s=c_s, n_freq=n_freq)

    def forward(self, s: torch.Tensor, node_mask: torch.Tensor, A_soft: torch.Tensor, eps: float = 1e-8):
        pos01, Lk = soft_pos01_from_A(A_soft, node_mask, eps=eps)        # [B,N], [B,K]
        pos_emb = self.pos_enc(pos01.to(torch.float32), node_mask).to(s.dtype)
        s_aug = s + self.pos_weight * pos_emb

        s_parent, occ, A_used = soft_pool_s(s_aug, node_mask, A_soft, eps=eps)

        aux = {"pos01": pos01, "Lk": Lk}
        return s_parent, occ, A_used, aux


# ============================================================
# 3) 把 break_logits -> a_idx（推理/切换 teacher 时用）
#    先给一个最简单的贪心版本（带 min/max_len 和 chain 断点）
#    ⚠️ 训练阶段建议先不用它：先拟合 teacher break
# ============================================================
@torch.no_grad()
def a_idx_from_break_logits_greedy_budget(
    break_logits: torch.Tensor,     # [B, N-1]
    node_mask: torch.Tensor,        # [B, N]
    chain_idx: torch.Tensor | None,
    min_len: int,
    max_len: int,
    Kmax: int,
    threshold: float = 0.0,         # logits > threshold => candidate
):
    """
    改进点：
      1) 在窗口内选“logit 最大”的 cut（而不是第一个 True）
      2) budget-aware：保证剩余长度能被剩余段数以[min_len,max_len]填满
      3) 仍然保持 chain_break 强制断开
    """
    B, N_ = node_mask.shape
    device = node_mask.device
    a_idx = torch.zeros((B, N_), device=device, dtype=torch.long)
    mask_parent = torch.zeros((B, Kmax), device=device, dtype=torch.float32)
    seg_lens = torch.zeros((B, Kmax), device=device, dtype=torch.long)

    m = node_mask.bool()

    for b in range(B):
        n = int(m[b].sum().item())
        if n <= 0:
            continue

        # chain break
        chain_break = None
        if chain_idx is not None:
            chain_break = (chain_idx[b, 1:n] != chain_idx[b, 0:n-1])  # [n-1]

        logits = break_logits[b, :n-1].clone()                         # [n-1]
        pred_break = (logits > threshold)
        if chain_break is not None:
            pred_break = pred_break | chain_break

        k = 0
        start = 0

        while start < n and k < Kmax:
            # 如果已经只剩最后一个 segment 名额，直接吞掉剩余
            if k == Kmax - 1:
                a_idx[b, start:n] = k
                mask_parent[b, k] = 1.0
                seg_lens[b, k] = n - start
                start = n
                break

            # 允许的 cut 范围
            end_min = min(start + min_len, n)
            end_max = min(start + max_len, n)

            # 若连 min_len 都不够了，直接吞掉
            if end_min >= n:
                a_idx[b, start:n] = k
                mask_parent[b, k] = 1.0
                seg_lens[b, k] = n - start
                start = n
                break

            # 候选 cut：pos 表示在 pos 和 pos+1 之间断开，所以 cut=pos+1
            # pos ∈ [end_min-1, end_max-2]
            pos_lo = end_min - 1
            pos_hi = end_max - 1  # python range hi is exclusive, so use pos_hi

            # 枚举窗口内所有候选（包含 forced chain break）
            cand_pos = []
            cand_score = []
            for pos in range(pos_lo, pos_hi):
                if pred_break[pos].item():
                    cut = pos + 1

                    # -------- budget feasibility check --------
                    remain_len = n - cut
                    remain_slots = (Kmax - 1) - k  # 剩余“还可以创建的新段数”（不含当前段）
                    # 这些 remain_slots 段必须能覆盖 remain_len
                    min_need = remain_slots * min_len
                    max_can  = remain_slots * max_len
                    if remain_len < min_need or remain_len > max_can:
                        continue  # 这个 cut 会导致后面无解，禁止

                    cand_pos.append(pos)
                    cand_score.append(logits[pos].item())

            if len(cand_pos) > 0:
                # 选 logit 最大的候选 cut
                best_i = int(torch.tensor(cand_score).argmax().item())
                cut = cand_pos[best_i] + 1
            else:
                # 没有合格 break，则尽量取 end_max，同时也要保证预算可行
                cut = end_max
                # budget 修正：如果 cut 太靠后导致后面不够分，就往前收
                remain_slots = (Kmax - 1) - k
                while True:
                    remain_len = n - cut
                    if remain_len >= remain_slots * min_len and remain_len <= remain_slots * max_len:
                        break
                    cut -= 1
                    if cut <= end_min:
                        cut = end_min
                        break

            L = cut - start
            a_idx[b, start:cut] = k
            mask_parent[b, k] = 1.0
            seg_lens[b, k] = L

            start = cut
            k += 1

    return a_idx, mask_parent, seg_lens


# ============================================================
# 4) 你要接到现有结构里时，训练阶段最推荐的用法：
#    - 仍然用 teacher a_idx 构 parents
#    - seg_tower 更新 s -> break_head 预测 break_logits
#    - break_sup_loss 拟合 teacher break
# ============================================================


# =========================
# quick test
# =========================


import torch
import torch.nn.functional as F

# ============================================================
# 1) 从 break_logits 构造 p_break 和 A_soft （完全可导）
# ============================================================

def build_p_break(
    break_logits: torch.Tensor,   # [B, N-1]
    node_mask: torch.Tensor,      # [B, N]
    chain_idx: torch.Tensor | None = None,  # [B,N] or None
    temp: float = 1.0,
):
    """
    returns:
      p_break: [B, N-1] in [0,1]  (masked; chain boundary forced to 1)
      edge_mask: [B, N-1] 0/1 (valid edges within valid residues)
    """
    B, N = node_mask.shape
    device = node_mask.device
    dtype = break_logits.dtype

    m = node_mask.to(dtype=dtype)
    edge_mask = (m[:, 1:] * m[:, :-1])  # [B,N-1] valid adjacent edges

    # sigmoid prob
    p = torch.sigmoid(break_logits / max(temp, 1e-6)) * edge_mask

    # force chain breaks (hard)
    if chain_idx is not None:
        # chain boundary between i-1 and i => must break
        chain_break = (chain_idx[:, 1:] != chain_idx[:, :-1]).to(dtype=dtype) * edge_mask
        # set to 1 where chain_break==1
        p = torch.where(chain_break > 0.5, torch.ones_like(p), p)

    return p, edge_mask


def build_A_soft_from_p_break(
    p_break: torch.Tensor,    # [B, N-1]
    node_mask: torch.Tensor,  # [B, N]
    Kmax: int,
    alpha: float = 16.0,
    eps: float = 1e-8,
):
    """
    核心：s_i = sum_{t<i} p_break[t]  (monotonic increasing)
          A_{i,k} ∝ exp(-alpha*(s_i - k)^2)

    returns:
      A_soft: [B, N, Kmax]   rows sum to 1 on valid residues
      s_id:   [B, N]         soft segment coordinate
    """
    B, N = node_mask.shape
    device = node_mask.device
    dtype = p_break.dtype

    m = node_mask.to(dtype=dtype)

    # s_id: [B,N], s_id[:,0]=0, s_id[:,i]=sum_{t< i} p_break[t]
    s_id = torch.zeros((B, N), device=device, dtype=dtype)
    if N > 1:
        s_id[:, 1:] = torch.cumsum(p_break, dim=1)  # [B,N-1] -> placed into [B,1:]

    # centers k = 0..Kmax-1
    k = torch.arange(Kmax, device=device, dtype=dtype).view(1, 1, Kmax)  # [1,1,K]    s = s_id.unsqueeze(-1)  # [B,N,1]
    s = s_id.unsqueeze(-1)  # [B,N,1]
    logits = -alpha * (s - k) ** 2  # [B,N,K]
    logits = logits.masked_fill((m < 0.5).unsqueeze(-1), -1e9)

    A = torch.softmax(logits, dim=-1) * m.unsqueeze(-1)  # [B,N,K]
    return A, s_id


# ============================================================
# 2) 用 A_soft 做 pooling（替代硬 one-hot）
# ============================================================

def segment_soft_pool_s(
    s: torch.Tensor,              # [B,N,C]
    A_soft: torch.Tensor,         # [B,N,K]
    node_mask: torch.Tensor,      # [B,N]
    eps: float = 1e-8,
):
    """
    returns:
      s_parent: [B,K,C]
      occ:      [B,K] soft lengths / occupancy
    """
    dtype = s.dtype
    m = node_mask.to(dtype=dtype)

    A = A_soft.to(dtype=dtype)
    occ = A.sum(dim=1)  # [B,K]
    denom = occ.clamp_min(eps).unsqueeze(-1)
    s_parent = torch.einsum("bnk,bnc->bkc", A, s) / denom
    return s_parent, occ


# ============================================================
# 3) 你要的正则：lambda_K * E[K] + min/max_len hinge
#    +（可选）A 连续性平滑
# ============================================================

def segmentation_regularizers(
    A_soft: torch.Tensor,         # [B,N,K]
    p_break: torch.Tensor,        # [B,N-1]
    edge_mask: torch.Tensor,      # [B,N-1]
    Kmax_limit,
    s_id,
    min_len: float = 2.0,
    max_len: float = 4.0,
    lambda_K: float = 0.0,       # 段数惩罚（越大越少切）
    w_minlen: float = 0.1,       # 短段惩罚（避免全是2）
    w_maxlen: float = 0.1,       # 长段惩罚（避免一段吞太长）
    w_smoothA: float = 0.1,       # 可选：增强连续性
compression_ratio: float = 2.0,
    eps: float = 1e-8,
):
    """
    returns dict with:
      E_K, loss_K, loss_minlen, loss_maxlen, loss_smoothA, total
    """
    B, N, K = A_soft.shape
    dtype = A_soft.dtype

    # ---- E[K] ≈ 1 + sum p_break over valid edges
    E_K = 1.0 + (p_break * edge_mask).sum(dim=1)  # [B]
    loss_K = (lambda_K * E_K).mean()

    # ---- soft segment length L_k = sum_i A_{i,k}
    Lk = A_soft.sum(dim=1)  # [B,K]

    # 只对“非空段”施加 min/max（否则空段会被 min_len 罚爆）
    # active_mask: Lk > tiny
    active = (Lk > 0.5).to(dtype=dtype)  # 0/1

    # min_len hinge:  (min_len - Lk)+
    h_min = F.relu(min_len - Lk)
    loss_minlen = (w_minlen * (h_min ** 2) * active).sum() / (active.sum().clamp_min(1.0))

    # max_len hinge:  (Lk - max_len)+
    h_max = F.relu(Lk - max_len)
    loss_maxlen = (w_maxlen * (h_max ** 2) * active).sum() / (active.sum().clamp_min(1.0))



    # ---- 可选：A 的邻接平滑（只有当不 break 的地方才强制相近）
    # sum_i ||A_i - A_{i+1}||_1 * (1 - p_break[i])
    loss_smoothA = torch.zeros((), device=A_soft.device, dtype=dtype)
    if w_smoothA > 0.0 and N > 1:
        w = (1.0 - p_break).clamp(0.0, 1.0) * edge_mask  # [B,N-1]
        diff = (A_soft[:, 1:, :] - A_soft[:, :-1, :]).abs().sum(dim=-1)  # [B,N-1]
        denom = w.sum().clamp_min(eps)
        loss_smoothA = w_smoothA * (diff * w).sum() / denom

    # 1. 计算总切分段数 (最后一个点的累积路径)
    # s_id: [B, N]
    total_segments = s_id[:, -1]  # [B]

    # 2. 【核心】溢出惩罚 (Overflow Barrier)
    # 如果 total_segments > Kmax_limit，产生巨大的梯度
    # 给一点余量 (buffer)，比如 limit - 2
    # limit = float(Kmax_limit) - 2.0
    #
    # # ReLU(total - limit) 表示：只要没超标，Loss=0；一旦超标，Loss 随距离线性/平方增长
    # overflow = F.relu(total_segments - limit)
    #
    # # 权重给大一点，因为这是硬约束
    # loss_overflow = (overflow ** 2).mean() * 1.0
    #
    # # 2.
    # # 设定你的“红线” (Budget)
    # # 比如 N=200, ratio=10 => target=20
    # # 你希望 K <= 20
    # target_budget = float(N) / compression_ratio
    #
    # # 3. 计算超支 (Over Budget)
    # # 只有当 current_K > target_budget 时才有值
    # over_budget = F.relu(E_K - target_budget)
    #
    # # 4. 暴力惩罚 (Quadratic Penalty)
    # # 用平方！如果超了 1 个 K，罚 1；超了 5 个 K，罚 25。
    # # 这会逼疯模型，让它必须砍掉多余的 K。
    # loss_budget = (over_budget ** 2).mean() * 1
    #
    # 固定 2x 合并时，关闭预算/溢出软约束
    loss_overflow = torch.zeros((), device=A_soft.device, dtype=dtype)
    loss_budget = torch.zeros((), device=A_soft.device, dtype=dtype)




    total = loss_K + loss_minlen + loss_maxlen + loss_smoothA + loss_overflow + loss_budget

    return {
        "E_K": E_K.mean(),
        "loss_K": loss_K,
        "loss_minlen": loss_minlen,
        "loss_maxlen": loss_maxlen,
        "loss_smoothA": loss_smoothA,
        "loss_overflow":loss_overflow,
        "loss_budget": loss_budget,

        "total": total,
        "Lk_mean": Lk.mean(),
        "Lk_min": (Lk + (1.0 - active) * 1e9).min(),  # min over active-ish
        "Lk_max": Lk.max(),
    }


# ============================================================
# 4) 一站式：break_logits -> A_soft + reg losses
# ============================================================

def build_soft_segments_and_loss(
    break_logits: torch.Tensor,   # [B,N-1]
    node_mask: torch.Tensor,      # [B,N]
    chain_idx: torch.Tensor | None,
    Kmax: int,
    min_len: float,
    max_len: float,
    temp: float = 1.0,
    alpha: float = 16.0,
    lambda_K: float = 0.01,
    w_minlen: float = 0.1,
    w_maxlen: float = 0.1,
    w_smoothA: float = 0.01,
):
    p_break, edge_mask = build_p_break(
        break_logits=break_logits, node_mask=node_mask, chain_idx=chain_idx, temp=temp
    )
    # A_soft, s_id = build_A_soft_from_p_break(
    #     p_break=p_break, node_mask=node_mask, Kmax=Kmax, alpha=alpha
    # )

    A_soft, s_id =build_A_soft_dynamic(
        p_break=p_break, node_mask=node_mask, Kmax_limit=Kmax, alpha=alpha
    )

    regs = segmentation_regularizers(
        A_soft=A_soft, p_break=p_break, edge_mask=edge_mask,Kmax_limit=Kmax,s_id=s_id,
        min_len=min_len, max_len=max_len,
        lambda_K=lambda_K, w_minlen=w_minlen, w_maxlen=w_maxlen, w_smoothA=w_smoothA
    )
    aux = {"p_break": p_break, "edge_mask": edge_mask, "s_id": s_id}
    return A_soft, regs, aux


def calculate_uniformity_metrics(seg_lens, mask_parent):
    """
    输入:
        seg_lens: [B, Kmax] 每个片段的长度
        mask_parent: [B, Kmax] 0/1 mask，表示该片段是否真实存在
    输出:
        metrics: 包含 cv, min_max_ratio 等指标的字典
    """
    # 1. 转换为 float 以进行统计计算
    lens = seg_lens.float()

    # 2. 计算每个样本的真实片段数量 K_real
    K_real = mask_parent.sum(dim=1).clamp(min=1)  # [B]

    # 3. 计算均值 (Mean)
    # 注意：不能直接对 lens 求 mean，因为后面有 padding 的 0
    # 公式: sum(lengths) / K_real
    mean_len = (lens * mask_parent).sum(dim=1) / K_real  # [B]

    # 4. 计算方差和标准差 (Std)
    # 只计算 mask 为 1 的部分
    # var = sum((x - mean)^2) / K
    diff_sq = (lens - mean_len.unsqueeze(-1)) ** 2
    # 这里的 mask_parent 很重要，要过滤掉 padding 部分产生的误差
    variance = (diff_sq * mask_parent).sum(dim=1) / K_real
    std_dev = torch.sqrt(variance)  # [B]

    # --- 指标 1: 变异系数 (CV) ---
    # CV = std / mean
    cv = std_dev / mean_len.clamp(min=1e-6)

    # --- 指标 2: Max/Min Ratio ---
    # 为了计算 min，我们需要把 padding 的 0 变成无穷大，否则 min 永远是 0
    lens_for_min = lens.clone()
    lens_for_min[mask_parent == 0] = float('inf')

    max_l = (lens * mask_parent).max(dim=1)[0]
    min_l = lens_for_min.min(dim=1)[0]

    # 避免 min 为 0 导致除零
    ratio = max_l / min_l.clamp(min=1e-6)

    return {
        "mean_len": mean_len,  # 平均长度
        "std_dev": std_dev,  # 标准差
        "cv": cv,  # 变异系数 (越低越好)
        "max_min_ratio": ratio  # 最长/最短比 (越接近1越好)
    }




def build_A_soft_dynamic(
        p_break: torch.Tensor,  # [B, N-1]
        node_mask: torch.Tensor,  # [B, N]
        Kmax_limit: int,  # 这是一个"硬上限"，比如 128，防止模型疯了导致显存爆炸
        alpha: float = 16.0,
        radius_factor: float = 3.0,  # 依然建议加上这个截断，保证 0 就是 0
):
    B, N = node_mask.shape
    device = node_mask.device
    dtype = p_break.dtype

    # 1. 计算 sid (累积路程)
    # s_id: [B, N]
    s_id = torch.zeros((B, N), device=device, dtype=dtype)
    if N > 1:
        s_id[:, 1:] = torch.cumsum(p_break, dim=1)

    # =========================================================
    # 固定 2x 合并：K = ceil(N/2)，不再动态决定
    # =========================================================
    with torch.no_grad():
        K_fixed = int(torch.ceil(torch.tensor(float(N) / 2.0, device=device)).item())
        K_curr = min(max(K_fixed, 1), int(Kmax_limit))

        # ===== 原动态 K 逻辑（保留注释） =====
        # # 找出当前 batch 里走得最远的 sid 是多少
        # # 比如 max_s = 4.3，说明最大用到 K=5 (索引 0~4)
        # # 我们取整并 +1，作为当前的 K_curr
        # max_s = (s_id * node_mask).max()
        # K_needed = int(torch.ceil(max_s).item()) + 1
        #
        # # 加上一点余量(buffer)，比如 +2，防止边界效应切断了最后一个高斯的尾巴
        # K_curr = K_needed + 2
        #
        # # 但不能超过硬件允许的上限
        # K_curr = min(K_curr, Kmax_limit)
        #
        # # 至少要保留 1 个 K
        # K_curr = max(K_curr, 1)

    # =========================================================

    # 2. 生成动态的 K 网格
    # 注意：这里我们只生成到 K_curr，而不是 Kmax_limit
    k = torch.arange(K_curr, device=device, dtype=dtype).view(1, 1, K_curr)  # [1, 1, K_curr]
    s = s_id.unsqueeze(-1)  # [B, N, 1]

    # 3. 计算距离 (只计算必要的 K)
    # logits: [B, N, K_curr] <--- 维度变小了！
    dist_sq = (s - k) ** 2

    # --- 加上之前的 Hard Radius 截断 (建议保留，为了彻底的 0) ---
    sigma = 1.0 / (alpha ** 0.5)
    radius = sigma * radius_factor
    cutoff_mask = (dist_sq > radius ** 2)

    logits = -alpha * dist_sq

    # Mask 处理
    # 1. Padding 点设为 -inf
    # 2. 距离太远的点设为 -inf
    final_mask = (node_mask.unsqueeze(-1) < 0.5) | cutoff_mask
    logits = logits.masked_fill(final_mask, -1e9)

    # 4. Softmax
    A = torch.softmax(logits, dim=-1)

    # 再次清洗微小值
    A = A * (~final_mask).to(dtype)

    # 归一化
    denom = A.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    A = A / denom * node_mask.unsqueeze(-1)

    return A, s_id
if __name__ == "__main__":
    B, N, C = 2, 32, 64
    Kmax = 16
    s = torch.randn(B, N, C)
    node_mask = torch.ones(B, N)
    # toy: segments [0..3]=0, [4..9]=1, [10..]=2
    a_idx = torch.zeros(B, N, dtype=torch.long)
    a_idx[:, 4:10] = 1
    a_idx[:, 10:] = 2
    mask_parent = torch.zeros(B, Kmax)
    mask_parent[:, :3] = 1

    pool = SegmentPoolingWithPosEnc(c_s=C, Kmax=Kmax, n_freq=16, pos_weight=1.0)
    s_parent, occ, A, aux = pool(s, node_mask, a_idx, mask_parent)

    print("s_parent:", s_parent.shape)
    print("occ:", occ[0, :5])
    print("pos01[0]:", aux["pos01"][0, :12])
    print("seg_len[0,:5]:", aux["seg_len"][0, :5])
