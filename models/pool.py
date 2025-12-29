import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
from data import utils as du
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.GaussianRigid import OffsetGaussianRigid,save_gaussian_as_pdb
from openfold.utils.rigid_utils import Rotation
from models.loss import HierarchicalGaussianLoss
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


def merge_gaussians_soft(mu: torch.Tensor, Sigma: torch.Tensor, A: torch.Tensor, eps: float = 1e-8):
    B, N, K = A.shape
    w = A / (A.sum(dim=1, keepdim=True) + eps)         # [B, N, K]
    mu_c = torch.einsum("bnk,bnd->bkd", w, mu)         # [B, K, 3]
    intra = torch.einsum("bnk,bnij->bkij", w, Sigma)   # [B, K, 3, 3]
    diff = mu.unsqueeze(2) - mu_c.unsqueeze(1)         # [B, N, K, 3]
    outer = diff.unsqueeze(-1) * diff.unsqueeze(-2)    # [B, N, K, 3, 3]
    inter = torch.einsum("bnk,bnkij->bkij", w, outer)  # [B, K, 3, 3]
    Sigma_c = intra + inter
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
    total: torch.Tensor


class LearnOnlyGaussianPooling(nn.Module):
    def __init__(
        self,
        c_s: int,
        ratio: float = 12.0,
        k_max_cap: Optional[int] = None,
        tau_init: float = 2.0,
        slots_init_scale: float = 0.02,
            geo_dim=8, gamma_init=0.5, eps=1e-8
    ):
        super().__init__()
        self.ratio = ratio
        self.k_max_cap = k_max_cap
        self.tau = tau_init
        self.proj = nn.Linear(c_s, c_s)
        self.slot_embed: Optional[nn.Parameter] = None
        self._slot_K: Optional[int] = None
        self._slots_init_scale = slots_init_scale
        self.eps=eps

        # ---------- 新增：几何分支 ----------
        self.geo_dim = geo_dim
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))  # 可学习或你手动 schedule

        # U: 把几何特征 g_i 投影到 geo_dim
        self.geo_proj = nn.Linear(self._geo_feat_dim(), geo_dim, bias=True)

        # h_k: 每个 slot 的几何 prototype（不含坐标）
        self.geo_slot = nn.Parameter(torch.randn(k_max_cap, geo_dim) * slots_init_scale)

        # 可选：让几何 term 有更稳定尺度
        self.geo_ln = nn.LayerNorm(geo_dim)



    def _ensure_slots(self, K: int, C: int, device, dtype):
        if (self.slot_embed is None) or (self._slot_K != K) or (self.slot_embed.shape[-1] != C):
            self._slot_K = K
            slots = torch.randn(K, C, device=device, dtype=dtype) * self._slots_init_scale
            self.slot_embed = nn.Parameter(slots)
    def _geo_feat_dim(self):
        # 你要的特征：logdet, logtrace, aniso(=logλ1-logλ3), (logλ1,logλ2,logλ3)
        # => 1 + 1 + 1 + 3 = 6
        return 6
    def forward(
        self,
        s: torch.Tensor,                 # [B,N,C]
        mu: torch.Tensor,                # [B,N,3]
        Sigma: torch.Tensor,             # [B,N,3,3]
        mask: Optional[torch.Tensor] = None,  # [B,N] {0,1}
        w_occ: float = 1.0,
        w_rep: float = 0.1,
        w_ent: float = 0.0,
        rep_topk: int = 4,
        rep_margin: float = -1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PoolLoss]:
        B, N, C = s.shape
        device, dtype = s.device, s.dtype
        if mask is None:
            mask = torch.ones(B, N, device=device, dtype=dtype)

        K = choose_K(N, ratio=self.ratio, k_min=1, k_max=self.k_max_cap)
        self._ensure_slots(K, C, device, dtype)

        s_proj = self.proj(s)  # [B,N,C]
        slots = self.slot_embed.unsqueeze(0).expand(B, -1, -1)  # [B,K,C]
        logits_sem = torch.einsum("bnc,bkc->bnk", s_proj, slots) / max(self.tau, 1e-6)

        # --------- (2) 几何 logits（平移不变） ----------
        # g = gaussian_geo_features(Sigma, eps=self.eps)  # [B,N,6]
        # g = self.geo_proj(g)  # [B,N,geo_dim]
        # g = self.geo_ln(g)
        #
        # geo_slots = self.geo_slot[:K].unsqueeze(0).expand(B, -1, -1)  # [B,K,geo_dim]
        # logits_geo = torch.einsum("bnd,bkd->bnk", g, geo_slots)  # [B,N,K]

        #A.最快、最稳（我最推荐先用）：序列均匀anchor


        #上面的效果不好 方案 B（可学习中心）：B 用“可学习 center”，但仍用距离而不是点积
        # self.mu_slot = nn.Parameter(torch.randn(Kmax, 3) * 0.5)  # nm
        # c = self.mu_slot[:K][None, None, :, :]  # [1,1,K,3]
        # dist2 = ((mu[:, :, None, :] - c) ** 2).sum(-1)  # [B,N,K]
        # logits_mu = -dist2 / (2 * sigma2)

        #FPS（再考虑优化）


        # --------- (3) 合并 ----------
        logits = logits_sem + self.gamma * logits_geo




        logits = logits.masked_fill((mask < 0.5).unsqueeze(-1), -1e9)
        A = F.softmax(logits, dim=-1)  # [B,N,K]

        # # A: [B,N,K]
        # print("A col std:", A.sum(dim=1).std(dim=-1).mean().item())  # slot占用差异
        # print("A diff:", (A[:, :, 0] - A[:, :, 1]).abs().mean().item())  # 两列是否一样

        A = A * mask.unsqueeze(-1)

        denom = A.sum(dim=1).clamp_min(1e-8)          # [B,K]
        s_c = torch.einsum("bnk,bnc->bkc", A, s) / denom.unsqueeze(-1)

        mu_c, Sigma_c = merge_gaussians_soft(mu, Sigma, A)

        # occupancy
        occ = A.sum(dim=1)  # [B,K]
        occ = occ / (occ.sum(dim=-1, keepdim=True).clamp_min(1e-8))
        target = torch.full_like(occ, 1.0 / K)
        loss_occ = F.mse_loss(occ, target)

        # repulsion
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

        # entropy
        ent = -(A.clamp_min(1e-8) * A.clamp_min(1e-8).log()).sum(dim=-1)  # [B,N]
        ent = (ent * mask).sum() / mask.sum().clamp_min(1.0)
        loss_ent = ent

        total = w_occ * loss_occ + w_rep * loss_rep + w_ent * loss_ent
        losses = PoolLoss(occ=loss_occ, rep=loss_rep, ent=loss_ent, total=total)


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



# 你已有：
# - uniform_anchors(mask, K) -> idx [B,K]
# - choose_K(N, ratio, k_min=1, k_max=...) -> int K
# - merge_gaussians_soft(mu, Sigma, A) -> (mu_c, Sigma_c)
# - gaussian_overlap_score(delta, sigma_sum) -> score  (或 fused_gaussian_overlap_score)
# - PoolLoss dataclass/NamedTuple: PoolLoss(occ, rep, ent, total)

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


class GaussianHierarchicalDownsampler(nn.Module):
    """
    ✅ 你要的：把 fine 之后所有下采样相关逻辑打包为一个类

    输入（来自 fine trunk 结束）：
      - s_f: [B,N,C]
      - r_f: OffsetGaussianRigid [B,N]
      - mask_f: [B,N]

    输出：
      - levels: List[DownsampleOut]  (每一次下采样后的 coarse)
      - reg_total: 所有 pooling 正则的和（直接加到总 loss）
    """

    def __init__(
        self,
        c_s: int,
        num_downsample: int = 2,
        ratio: float = 12.0,
        k_max_cap: Optional[int] = None,
        tau_start: float = 2.0,
        tau_end: float = 0.5,
        # pooling loss weights (默认工程稳)
        w_occ: float = 1.0,
        w_rep: float = 0.1,
        w_ent_start: float = 0.0,
        w_ent_end: float = 0.01,
        rep_topk: int = 4,
        rep_margin: float = -1.0,
        # Sigma->rigid
        sigma_jitter: float = 1e-6,
        var_floor: float = 1e-6,  # variance floor (nm^2)
        empty_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_downsample = num_downsample
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.w_ent_start = w_ent_start
        self.w_ent_end = w_ent_end

        self.w_occ = w_occ
        self.w_rep = w_rep
        self.rep_topk = rep_topk
        self.rep_margin = rep_margin

        self.sigma_jitter = sigma_jitter
        self.var_floor = var_floor
        self.empty_eps = empty_eps

        self.pools = nn.ModuleList([
            LearnOnlyGaussianPooling(
                c_s=c_s,
                ratio=ratio,
                k_max_cap=k_max_cap,
                tau_init=tau_start,
            )
            for _ in range(num_downsample)
        ])

    @staticmethod
    def _fix_rotation_matrix(R: torch.Tensor) -> torch.Tensor:
        det = torch.det(R)
        flip = (det < 0).to(R.dtype)[..., None, None]
        R2 = R.clone()
        R2[..., :, 2:3] = R2[..., :, 2:3] * (1.0 - 2.0 * flip)
        return R2

    def _coarse_rigids_from_mu_sigma(
        self,
        mu_c: torch.Tensor,     # [B,K,3]
        Sigma_c: torch.Tensor,  # [B,K,3,3]
    ) -> "OffsetGaussianRigid":
        device, dtype = mu_c.device, mu_c.dtype

        Sigma = 0.5 * (Sigma_c + Sigma_c.transpose(-1, -2))
        I = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
        Sigma = Sigma + self.sigma_jitter * I

        evals, evecs = torch.linalg.eigh(Sigma)     # stable for 3x3
        evals = evals.clamp_min(self.var_floor)
        scaling = torch.sqrt(evals).clamp_min(1e-6)  # std (nm)

        R = self._fix_rotation_matrix(evecs)

        # 下面两行依赖你项目里的 Rotation / OffsetGaussianRigid
        rots = Rotation(rot_mats=R)  # 如果你 Rotation 构造不是 rot_mats=，改这里即可
        trans = mu_c
        local_mean = torch.zeros_like(mu_c)
        scaling_log = torch.log(scaling + 1e-6)

        return OffsetGaussianRigid(
            rots=rots,
            trans=trans,
            scaling_log=scaling_log,
            local_mean=local_mean,
        )

    def _schedule(self, i: int, step: Optional[int], total_steps: Optional[int]) -> Tuple[float, float]:
        """
        返回 (tau, w_ent)；你不传 step/total_steps 就固定用起始值。
        """
        if (step is None) or (total_steps is None) or (total_steps <= 0):
            return self.tau_start, self.w_ent_start

        t = float(step) / float(max(total_steps, 1))
        tau = self.tau_start + (self.tau_end - self.tau_start) * t
        w_ent = self.w_ent_start + (self.w_ent_end - self.w_ent_start) * t
        return tau, w_ent

    def forward(
        self,
        s_f: torch.Tensor,                # [B,N,C]
        r_f: "OffsetGaussianRigid",        # [B,N]
        mask_f: torch.Tensor,             # [B,N]
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> Tuple[List[DownsampleOut], torch.Tensor]:
        """
        Returns:
          levels: list of DownsampleOut, length = num_downsample
          reg_total: scalar tensor (sum of pool losses)
        """
        s = s_f
        r = r_f
        mask = mask_f

        levels: List[DownsampleOut] = []
        reg_total = torch.zeros((), device=s.device, dtype=s.dtype)

        for i in range(self.num_downsample):
            pool = self.pools[i]
            tau, w_ent = self._schedule(i, step, total_steps)
            pool.tau = float(tau)

            mu = r.get_gaussian_mean()     # [B,N,3]
            Sigma = r.get_covariance()     # [B,N,3,3]

            A, s_c, mu_c, Sigma_c, pool_loss = pool(
                s=s,
                mu=mu,
                Sigma=Sigma,
                mask=mask,
                w_occ=self.w_occ,
                w_rep=self.w_rep,
                w_ent=float(w_ent),
                rep_topk=self.rep_topk,
                rep_margin=self.rep_margin,
            )

            denom = A.sum(dim=1)  # [B,K]
            mask_c = (denom > self.empty_eps).to(mask.dtype)
            s_c = s_c * mask_c[..., None]

            r_c = self._coarse_rigids_from_mu_sigma(mu_c, Sigma_c)

            levels.append(DownsampleOut(
                s=s_c, rigids=r_c, mask=mask_c, A=A, pool_loss=pool_loss
            ))
            reg_total = reg_total + pool_loss.total

            # 下一层继续下采样（把 coarse 当 fine 用）
            s, r, mask = s_c, r_c, mask_c

        return levels, reg_total


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
    diag = torch.diagonal(Sigma_c, dim1=-2, dim2=-1)  # [B,K,3]
    scaling = torch.sqrt(torch.clamp(diag, min=eps))
    scaling_log = torch.log(scaling + eps)

    # 3) local_mean = 0 (因为 trans 就是 mu)
    local_mean = torch.zeros_like(mu_c)

    return OffsetGaussianRigid_cls(
        rots=rots,
        trans=mu_c,
        scaling_log=scaling_log,
        local_mean=local_mean,
    )