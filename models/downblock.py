import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Sequence, Tuple
from data import utils as du
from models.pool import SoftSegmentPoolingWithPosEnc,calculate_uniformity_metrics,coarse_rigids_from_mu_sigma,UniformAnchorSemGeoAssign,UniformAnchorSemAssign,SegmentBreakHead,a_idx_from_break_logits_greedy_budget
from models.IGA import InvariantGaussianAttention,CoarseIGATower,GaussianUpdateBlock,FastTransformerTower
from models.SlotIGA import  IGASlotPoolingV1
from models.SlotIGAv2 import IGASlotPoolingV2
from data.GaussianRigid import save_gaussian_as_pdb,OffsetGaussianRigid
from models.loss import HierarchicalGaussianLoss,SymmetricGaussianLoss
from models.EdgeCoarsen import CoarseEdgeCoarsenAndFuse
from models.pool import teacher_segment_variable_length,build_parents_from_segments_v3_debug,build_parents_from_A_soft,SegmentPoolingWithPosEnc,build_soft_segments_and_loss,build_A_soft_from_p_break
from openfold.utils.rigid_utils import Rigid,Rotation


import torch
import matplotlib.pyplot as plt
import numpy as np
class HierarchicalDownsampleIGAModule(nn.Module):
    """
    做 K 次:
      mu,Sigma <- r.get_mean/cov
      A,s_c,mu_c,Sigma_c, pool_loss, mask_c = pool(...)
      r_c = coarse_rigids_from_mu_sigma(mu_c,Sigma_c)
      (s_c,r_c) = coarse_iga_tower(s_c,r_c,mask_c)
    然后用 (s_c,r_c,mask_c) 作为下一次输入

    返回每层的 levels，后面你上采样一定用得上。
    """

    def __init__(
            self,
            c_s: int,
            iga_conf,
            OffsetGaussianRigid_cls,
            num_downsample: int = 2,
            ratio: float = 6.0,
            k_max_cap: int = 1024,
            coarse_iga_layers: int | list[int] = 6,  # <--- 现在可配置
    ):
        super().__init__()
        self.num_downsample = num_downsample
        self.OffsetGaussianRigid_cls = OffsetGaussianRigid_cls

        # --- normalize coarse_iga_layers to per-level list ---
        if isinstance(coarse_iga_layers, int):
            per_level_layers = [coarse_iga_layers] * num_downsample
        else:
            assert len(coarse_iga_layers) == num_downsample, \
                "coarse_iga_layers list length must equal num_downsample"
            per_level_layers = list(coarse_iga_layers)

        # pool per level
        # self.pools = nn.ModuleList([
        #     LearnOnlyGaussianPoolingV2(
        #         c_s=c_s, ratio=ratio, k_max_cap=k_max_cap,
        #         tau_init=0.2, slots_init_scale=0.2,
        #     )
        #     for _ in range(num_downsample)
        # ])

        self.pools = nn.ModuleList(
            [IGASlotPoolingV2(c_s=c_s, ratio=ratio, k_max=k_max_cap) for _ in range(num_downsample)])

        # self.pools = nn.ModuleList([
        #     UniformAnchorSemGeoAssign(
        #         c_s=c_s,
        #         ratio=ratio,
        #         k_max_cap=k_max_cap,
        #
        #         # -------- 关键超参数（强烈建议这样起步） --------
        #         sem_dim=128,  # 语义投影维度（稳定，不要太大）
        #         w_geo_init=1.0,  # 几何主导（先保证不重叠）
        #         w_sem_init=0.1,  # 语义弱绑定（防止 slot 对称）
        #         sigma_nm_init=1.0,  # nm 级几何 soft 半径
        #         tau_sem_init=1.0,  # 语义温度
        #     )
        #     for _ in range(num_downsample)
        # ])

        self.edge_fusers = nn.ModuleList([
            CoarseEdgeCoarsenAndFuse(
                c_z_in=getattr(iga_conf, "hgfc_z", 0),  # 你 IGA 用的 Cz
                c_z_out=getattr(iga_conf, "hgfc_z", 0),  # 你 IGA 用的 Cz
                # 这里填你类需要的其它配置，比如:
                # geo_dim=..., sem_dim=..., fuse_dim=..., use_geo=True ...
            )
            for _ in range(num_downsample)
        ])

        # coarse tower per level (num_layers taken from per_level_layers[lv])
        self.coarse_towers = nn.ModuleList()
        for lv in range(num_downsample):
            iga = InvariantGaussianAttention(
                c_s=c_s,
                c_z=getattr(iga_conf, "hgfc_z", 0),
                c_hidden=iga_conf.c_hidden,
                no_heads=iga_conf.no_heads,
                no_qk_gaussians=iga_conf.no_qk_points,
                no_v_points=iga_conf.no_v_points,
                layer_idx=1000 + lv,
                enable_vis=False,
            )
            gau_update = GaussianUpdateBlock(c_s)

            self.coarse_towers.append(
                CoarseIGATower(
                    iga=iga,
                    gau_update=gau_update,
                    c_s=c_s,
                    hgfc_z=iga_conf.hgfc_z,
                    num_layers=per_level_layers[lv],  # <--- 每层不同
                )
            )

        # 初始化 Loss 模块
        self.hier_loss = HierarchicalGaussianLoss(w_sep=10.0, w_compact=1)  # compact 权重别太大
        # 实例化
        self.parent_child_loss = SymmetricGaussianLoss(w_center_p=1.0, w_center_c=10.0, w_shape=0.0, eps=1e-6)

    def forward(self, s_f, z,r_f, mask_f, step: int, total_steps: int):
        """
        s_f: [B,N,C]
        r_f: OffsetGaussianRigid [B,N]
        mask_f: [B,N]
        """
        levels = []
        pool_reg_total = 0.0

        s, r, mask = s_f, r_f, mask_f

        for lv in range(self.num_downsample):
            # 1) 取几何
            mu = r.get_gaussian_mean()        # [B,N,3]

            # print("mu spread (std over N):", mu.std(dim=1).mean().item())
            # print("mu range:", (mu.max(dim=1).values - mu.min(dim=1).values).mean(dim=0))


            Sigma = r.get_covariance()        # [B,N,3,3]

            # 在每个 lv pooling 前
            # ang_rigids = r.scale_translation(10.0)  # 0.1
            # save_gaussian_as_pdb(
            #     gaussian_rigid=ang_rigids,
            #     filename=f"debug_lv{lv}_prepool_gaussian_mean.pdb",
            #     mask=mask,
            #     center_mode="gaussian_mean",
            # )



            # 2) schedule（退火）
            # 你可以把策略放这：tau 0.8->0.5
            t = float(step) / max(int(total_steps), 1)
            self.pools[lv].tau = 0.8 - 0.5 * t

            # 3) pooling
            A, s_c, mu_c, Sigma_c, idx, pool_loss,downmetric = self.pools[lv](
                s=s, mu=mu, Sigma=Sigma, mask=mask,
                # w_occ=1.0,
                # w_rep=0.1,
                # w_ent=0.0 if t < 0.5 else 0.01,   # 你要的“早软后硬”
            )

            # --- 新增: 动态计算 mask_c ---
            # 既然这个节点都没分到原子，它就不该参与几何 Loss
            curr_occ = (A * mask.unsqueeze(-1)).sum(dim=1)  # [B, K]
            mask_c = (curr_occ > 1e-4).float()  # [B, K]
            # ---------------------------
            # 计算 Loss
            geo_stats = self.hier_loss(mu_c, Sigma_c, mask_c)
            pool_reg_total += geo_stats["total_hier"]

            # 3. 计算父子一致性 Loss
            pc_stats = self.parent_child_loss(
                mu_child=mu,
                Sigma_child=Sigma,
                mu_parent=mu_c,
                Sigma_parent=Sigma_c,
                A=A,
                mask=mask
            )
            pool_reg_total += pc_stats["loss"]

            pool_reg_total = pool_reg_total + pool_loss.total

            # 4) mask_c：哪些 coarse token 有人分配
            # A: [B,N,K]
            occ = A.sum(dim=1)                        # [B,K]
            mask_c = (occ > 1e-6).to(mask.dtype)       # [B,K]

            # 5) build coarse rigids
            r_c = coarse_rigids_from_mu_sigma(
                mu_c, Sigma_c, self.OffsetGaussianRigid_cls
            )
            # ang_rigids = r_c.scale_translation(10.0)  # 0.1
            #
            # # pooling 得到 r_c / mask_c 后
            # save_gaussian_as_pdb(
            #     gaussian_rigid=ang_rigids,
            #     filename=f"debug_lv{lv}_r_down_slotv2.pdb",
            #     mask=mask_c,
            #     center_mode="gaussian_mean",
            # )

            # ---- build z_c for this level ----
            if z is not None and getattr(self, "edge_fusers", None) is not None:
                z_c, Z_sem_c, Z_geo_c = self.edge_fusers[lv](
                    A=A,Z_in=z, r_target=r_c,
                    mask_f=mask, mask_c=mask_c
                )
            else:
                z_c = None

            # 6) coarse IGA × (coarse_iga_layers)
            s_c, r_c, z_c = self.coarse_towers[lv](s_c, r_c, mask_c,z=z_c)

            # ang_rigids = r_c.scale_translation(10.0)  # 0.1
            #
            # # pooling 得到 r_c / mask_c 后
            # save_gaussian_as_pdb(
            #     gaussian_rigid=ang_rigids,
            #     filename=f"debug_lv{lv}_r_down_aftertowner.pdb",
            #     mask=mask_c,
            #     center_mode="gaussian_mean",
            # )


            # 记录 level
            levels.append({
                "A": A,
                "s": s_c,
                "z": z_c,
                "r": r_c,
                "mask": mask_c,
                'curr_occ':curr_occ,
                "pool_loss": pool_loss,
                "downmetric":downmetric
            })

            # 下一层输入
            s,z, r, mask = s_c,z_c, r_c, mask_c

        return levels, pool_reg_total

class HierarchicalDownsampleModuleFast(nn.Module):
    """
    每层:
      mu,Sigma <- r.get_gaussian_mean/cov   (如果你还保留椭圆)
      A,s_c,mu_c,Sigma_c, pool_loss, anchor_idx = pool(...)
      mask_c = occ>thr
      (可选) r_c = coarse_rigids_from_mu_sigma(mu_c,Sigma_c)
      tower:
        - transformer: s_c <- Transformer(s_c)
        - iga: (s_c,r_c) <- CoarseIGATower(s_c,r_c)

    返回 levels, pool_reg_total
    """

    def __init__(
        self,
        c_s: int,
        iga_conf,
        OffsetGaussianRigid_cls,
        num_downsample: int = 2,
        ratio: float = 6.0,
        k_max_cap: int = 64,
        coarse_layers: int | list[int] = 6,

        tower_mode: str = "transformer",  # "transformer" or "iga"
        tf_heads: int = 8,
        tf_mlp_ratio: float = 4.0,
    ):
        super().__init__()
        assert tower_mode in ("transformer", "iga")
        self.num_downsample = num_downsample
        self.OffsetGaussianRigid_cls = OffsetGaussianRigid_cls
        self.tower_mode = tower_mode

        # per level layer count
        if isinstance(coarse_layers, int):
            per_level_layers = [coarse_layers] * num_downsample
        else:
            assert len(coarse_layers) == num_downsample
            per_level_layers = list(coarse_layers)

        # ---- pools ----
        self.pools = nn.ModuleList([
            UniformAnchorSemAssign(
                c_s=c_s,
                ratio=ratio,
                k_max_cap=k_max_cap,
                neighbor_R=6,
                use_mu_geo=False,     # 可以先 True，但 gamma_init=0
                gamma_init=0.0,      # 先语义跑通，再加
                sigma2_init=0.25,
                tau_init=0.2,
                slots_init_scale=0.02,
            )
            for _ in range(num_downsample)
        ])

        # ---- towers ----
        if tower_mode == "iga":
            self.coarse_towers = nn.ModuleList()
            for lv in range(num_downsample):
                iga = InvariantGaussianAttention(
                    c_s=c_s,
                    c_z=getattr(iga_conf, "hgfc_z", 0),
                    c_hidden=iga_conf.c_hidden,
                    no_heads=iga_conf.no_heads,
                    no_qk_gaussians=iga_conf.no_qk_points,
                    no_v_points=iga_conf.no_v_points,
                    layer_idx=1000 + lv,
                    enable_vis=False,
                )
                gau_update = GaussianUpdateBlock(c_s)
                self.coarse_towers.append(
                    CoarseIGATower(
                        iga=iga,
                        gau_update=gau_update,
                        c_s=c_s,
                        num_layers=per_level_layers[lv],
                    )
                )
        else:
            self.tf_towers = nn.ModuleList([
                FastTransformerTower(c_s=c_s, num_layers=per_level_layers[lv], n_heads=tf_heads, mlp_ratio=tf_mlp_ratio)
                for lv in range(num_downsample)
            ])

        # 你的层内正则（可留着）
        self.hier_loss = HierarchicalGaussianLoss(w_sep=10.0, w_compact=1.0)

    def forward(self, s_f, r_f, mask_f, step: int, total_steps: int):
        levels = []
        pool_reg_total = 0.0

        s, r, mask = s_f, r_f, mask_f

        t = float(step) / max(int(total_steps), 1)

        for lv in range(self.num_downsample):
            # ---- schedule ----
            # 语义温度退火（你之前写的OK）
            self.pools[lv].tau = 0.8 - 0.5 * t

            # ---- 取几何（如果你还保留 r）----
            mu = r.get_gaussian_mean()     # [B,N,3]
            Sigma = r.get_covariance()     # [B,N,3,3]

            # ---- pooling ----
            A, s_c, mu_c, Sigma_c, pool_loss, anchor_idx = self.pools[lv](
                s=s, mu=mu, Sigma=Sigma, mask=mask,
                w_occ=1.0,
                w_rep=0.1,
                w_ent=0.0 if t < 0.5 else 0.01,
            )

            # ---- mask_c from occ ----
            occ = (A * mask.unsqueeze(-1)).sum(dim=1)        # [B,K]
            mask_c = (occ > 1e-6).to(mask.dtype)             # [B,K]

            # ---- hier loss（可选）----
            geo_stats = self.hier_loss(mu_c, Sigma_c, mask_c)
            pool_reg_total = pool_reg_total + geo_stats["total_hier"] + pool_loss.total

            # ---- tower ----
            if self.tower_mode == "iga":
                r_c = coarse_rigids_from_mu_sigma(mu_c, Sigma_c, self.OffsetGaussianRigid_cls)
                s_c, r_c = self.coarse_towers[lv](s_c, r_c, mask_c)
            else:
                # transformer 不需要 r_c
                s_c = self.tf_towers[lv](s_c, mask_c)
                # 但为了后续上采样你可能仍想保留 r_c（否则 up 用不了几何）
                r_c = coarse_rigids_from_mu_sigma(mu_c, Sigma_c, self.OffsetGaussianRigid_cls)


            levels.append({
                "A": A,
                "anchor_idx": anchor_idx,
                "s": s_c,
                "r": None,
                "mu": mu_c,
                "Sigma": Sigma_c,
                "mask": mask_c,
                "pool_loss": pool_loss,
                "geo_stats": geo_stats,
            })

            # next level
            s, r, mask = s_c, r_c, mask_c

        return levels, pool_reg_total


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List


# ============================================================
# Helper: hard segment pooling for node features s
# ============================================================
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


# ============================================================
# Helper: hard segment pooling for edge features z
# ============================================================
def segment_mean_pool_z(
    z: torch.Tensor,              # [B,N,N,Cz]
    node_mask: torch.Tensor,      # [B,N] 0/1
    a_idx: torch.Tensor,          # [B,N] long
    mask_parent: torch.Tensor,    # [B,Kmax] 0/1
    Kmax: int,
    eps: float = 1e-8,
):
    """
    z_parent[p,q] = mean_{i in p, j in q} z[i,j]
    returns:
      z_parent: [B,Kmax,Kmax,Cz]
    """
    if z is None:
        return None

    B, N, N2, Cz = z.shape
    assert N2 == N

    dtype = z.dtype
    m = node_mask.to(dtype=dtype)

    oh = F.one_hot(a_idx.clamp_min(0).clamp_max(Kmax - 1), num_classes=Kmax).to(dtype)  # [B,N,K]
    # apply node mask to both ends
    oh_i = oh * m.unsqueeze(-1)  # [B,N,K]
    oh_j = oh * m.unsqueeze(-1)  # [B,N,K]

    # numerator: sum_{i,j} oh[i,p] * oh[j,q] * z[i,j]
    # -> [B,K,K,Cz]
    num = torch.einsum("bik,bjq,bijc->bkqc", oh_i, oh_j, z)

    # denom: (#i in p) * (#j in q)
    cnt = oh_i.sum(dim=1).clamp_min(eps)  # [B,K]
    denom = (cnt[:, :, None] * cnt[:, None, :]).unsqueeze(-1)  # [B,K,K,1]

    z_parent = num / denom

    # mask unused parents
    mp = mask_parent.to(dtype=dtype)
    z_parent = z_parent * (mp[:, :, None, None] * mp[:, None, :, None])

    return z_parent


# ============================================================
# Main: Simple Segment Downsample Module
# ============================================================
class SimpleSegmentDownIGAModule_v1(nn.Module):
    """
    简化版 Down：
      - teacher_segment_variable_length -> (a_idx, mask_parent, seg_lens)
      - build_parents_from_segments_v3_debug -> r_parent (你已有的“父椭圆/rigid”构造)
      - s,z 用 hard assignment 聚合到 parent level
    不做任何 coarse IGA tower（先跑通 AE / segAE 约束）。
    """

    def __init__(
        self,
        c_s: int,
            iga_conf,
        Kmax: int = 64,
        min_len: int = 4,
        max_len: int = 10,
        use_pos_emb: bool = True,

        eps: float = 1e-8,
    ):
        super().__init__()
        self.c_s = c_s
        self.Kmax = Kmax
        self.min_len = min_len
        self.max_len = max_len
        self.use_pos_emb = use_pos_emb
        # self.pos_emb = pos_emb
        self.eps = eps


        self.pool=SoftSegmentPoolingWithPosEnc(c_s=c_s,  n_freq=16, pos_weight=1.0)

        self.seg_towers = nn.ModuleList()

        iga = InvariantGaussianAttention(
            c_s=c_s,
            c_z=getattr(iga_conf, "hgfc_z", 0),
            c_hidden=iga_conf.c_hidden,
            no_heads=iga_conf.no_heads,
            no_qk_gaussians=iga_conf.no_qk_points,
            no_v_points=iga_conf.no_v_points,
            layer_idx=1000 ,
            enable_vis=False,
        )


        self.seg_towers.append(
            CoarseIGATower(
                iga=iga,
                gau_update=None,
                c_s=c_s,
                hgfc_z=iga_conf.hgfc_z,
                num_layers=4,  # <--- 每层不同
            )
        )


        self.break_head=SegmentBreakHead(c_s=c_s,hidden=iga.c_hidden)


        # self.edge_fusers = nn.ModuleList([
        #     CoarseEdgeCoarsenAndFuse(
        #         c_z_in=getattr(iga_conf, "hgfc_z", 0),  # 你 IGA 用的 Cz
        #         c_z_out=getattr(iga_conf, "hgfc_z", 0),  # 你 IGA 用的 Cz
        #
        #     )
        #
        # ])

    @torch.no_grad()
    def _teacher_segments(self, node_mask, chain_idx):
        # 你已有的 teacher 分段函数
        a_idx, mask_parent, seg_lens = teacher_segment_variable_length(
            node_mask=node_mask,
            chain_idx=chain_idx,
            min_len=self.min_len,
            max_len=self.max_len,
            Kmax=self.Kmax,
        )
        return a_idx, mask_parent, seg_lens

    def forward(
        self,
        s_f: torch.Tensor,                 # [B,N,C]
        z_f: Optional[torch.Tensor],        # [B,N,N,Cz] or None
        r_f,                                 # OffsetGaussianRigid [B,N] (fine)
        node_mask: torch.Tensor,            # [B,N]
        chain_idx: Optional[torch.Tensor] = None, # [B,N]
        step: int = 0,
        total_steps: int = 1,
            log_img_every: int = 500,
    ):
        B, N, C = s_f.shape
        Kmax = self.Kmax
        device = s_f.device

        if chain_idx is None:
            chain_idx = torch.zeros((B, N), device=device, dtype=torch.long)

        # ----------------------------------------------------
        # 1) teacher segments -> a_idx/mask_parent/seg_lens
        # ----------------------------------------------------
        # a_idx, mask_parent, seg_lens = self._teacher_segments(node_mask=node_mask, chain_idx=chain_idx)

        s_out,_,_ = self.seg_towers[0](s_f, z_f, r_f, node_mask)
        break_logits = self.break_head(s_out, node_mask, z_f=None )

        A_soft, regs, aux=build_soft_segments_and_loss(break_logits,node_mask,chain_idx,Kmax=int(N/2),min_len=2,max_len=4)
        # a_idx, mask_parent, seg_lens=a_idx_from_break_logits_greedy_budget(break_logits,node_mask,chain_idx,min_len=2,max_len=12,Kmax=int((N/2)))

        if step% log_img_every == 0:
            fig_dwon=visualize_asoft_diagnosis(A_soft,save_path='asoft.png')
        else:
            fig_dwon=None
        # ----------------------------------------------------
        # 2) build parents geometry (keep your impl)
        #    NOTE: x_ca comes from your r_f fine gaussian mean (or CA trans)
        # ----------------------------------------------------
        # 你现在的约定：x_ca 是 CA 坐标（Å / nm 你自己保证一致）
        # 如果你希望 x_ca=rigid trans(=CA)，而不是 gaussian_mean，也可以换成 r_f.get_trans()
        x_ca = r_f.get_trans() if hasattr(r_f, "get_trans") else r_f.get_gaussian_mean()

        r_parent, occ_p, mask_parent,mask_soft = build_parents_from_A_soft(
            x=x_ca, A=A_soft, node_mask=node_mask
        )



        # ----------------------------------------------------
        # 3) (optional) add positional embedding before pooling
        # ----------------------------------------------------
        s_parent, occ_parent, A, aux = self.pool(s_f, node_mask, A_soft)
        # ----------------------------------------------------
        # 4) pool s to parent
        # ----------------------------------------------------


        # ----------------------------------------------------
        # 5) pool z to parent
        # ----------------------------------------------------
        z_parent = None
        # if z_f is not None:
        #     z_parent, Z_sem_c, Z_geo_c = self.edge_fusers(
        #         A=A, Z_in=z, r_target=r_parent,
        #         mask_f=node_mask, mask_c=mask_parent
        #     )

        # active K (debug)
        active_K = mask_parent.sum(dim=1).float().mean()

        # metrics_seglens = calculate_uniformity_metrics(seg_lens, mask_parent)
        # num_unique_clusters = len(torch.unique(a_idx))
        # strict_ratio = N / num_unique_clusters
        downmetric = {
            "seg_down_active_K": active_K.detach(),
            "seg_down_occ_mean": occ_parent.mean().detach(),
            "seg_down_occ_max": occ_parent.max().detach(),
            "seg_down_occ_min": occ_parent.min().detach(),
            "seg_down_ratio": N/active_K,
        }
        downmetric=downmetric

        levels = [{
            "A_soft": A_soft,                        # [B,N,K]
            "s": s_parent,                 # [B,K,C]
            "z": z_parent,                 # [B,K,K,Cz] or None
            "r": r_parent,                 # OffsetGaussianRigid [B,K]
            "mask_parent": mask_parent,           # [B,K]
            "curr_occ": occ_parent,         # [B,K]
            # "aux": {
            #     "a_idx": a_idx,
            #     "seg_lens": seg_lens,
            # },
            "downmetric": downmetric,
            "fig_dwon":fig_dwon
        }]

        # 你以前 forward 返回 (levels, reg_total)

        return levels, regs



def visualize_asoft_diagnosis(A_soft, node_mask=None, save_path=None, title="A_soft Diagnosis"):
    """
    专门诊断 Soft Segmentation 是否健康的函数

    Args:
        A_soft: Tensor [B, N, K] 或 [N, K]
        node_mask: Tensor [B, N] 或 [N] (可选，用于过滤 padding)
        save_path: str (可选，保存路径)
    """
    # 1. 数据预处理：取 Batch 中第一个非空样本
    if A_soft.dim() == 3:
        idx = 0
        # 如果有 mask，找一个长度适中的样本，而不是空的
        if node_mask is not None:
            lengths = node_mask.sum(dim=1)
            # 找最接近平均长度的样本，或者直接取第一个
            idx = torch.argmax(lengths).item()  # 取最长的那个看，最清晰
            mask = node_mask[idx].bool().cpu()
            A = A_soft[idx].detach()[mask].cpu().numpy()  # [N_valid, K]
        else:
            A = A_soft[0].detach().cpu().numpy()
    else:
        A = A_soft.detach().cpu().numpy()

    # 转置一下：X轴=原子序列(N)，Y轴=父节点ID(K)
    # 这样符合直觉：从左到右随着序列延伸，父节点ID逐渐增加
    A_map = A.T  # [K, N]
    K, N = A_map.shape

    # 2. 计算硬切分路径 (Argmax)
    hard_path = np.argmax(A_map, axis=0)  # [N]

    # 3. 计算每个簇的占用率 (Occupancy)
    occupancy = np.sum(A_map, axis=1)  # [K]
    active_k_indices = np.where(occupancy > 0.1)[0]
    active_k_count = len(active_k_indices)
    max_k_used = active_k_indices.max() if active_k_count > 0 else 0

    # ================= 绘图 =================
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)

    # --- 左图：Heatmap (A_soft) ---
    ax1 = fig.add_subplot(gs[0])
    # 使用 log 刻度可以让极小值显形，但这会夸大噪声。
    # 这里使用线性刻度，但 vmax 设小一点(0.8)让方块更亮
    im = ax1.imshow(A_map, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=1.0)

    # 画出红色的硬切分线
    ax1.plot(np.arange(N), hard_path, color='red', linewidth=1.5, alpha=0.7, label='Hard Path (Argmax)')

    # 装饰
    ax1.set_title(f"{title}\nPoints(N)={N}, Active K={active_k_count}, Used K range=0~{max_k_used}")
    ax1.set_xlabel("Sequence Index (N)")
    ax1.set_ylabel("Parent Cluster ID (K)")
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', color='white', alpha=0.1)

    # 只显示用到的 K 范围，避免上面留一大片空白
    if max_k_used < K - 1:
        ax1.set_ylim(-0.5, max_k_used + 5)

    # --- 右图：Occupancy Bar (负载均衡情况) ---
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    y_pos = np.arange(K)
    ax2.barh(y_pos, occupancy, color='teal', alpha=0.8)
    ax2.set_xlabel("Mass (Num Points)")
    ax2.set_title("Cluster Occupancy")
    ax2.grid(True, axis='x', alpha=0.3)

    # 在条形图上标数值
    for i, v in enumerate(occupancy):
        if v > 0.1 and i <= max_k_used + 5:
            ax2.text(v, i, f" {v:.1f}", va='center', fontsize=8)

    # 隐藏右图 Y 轴标签（共享轴）
    plt.setp(ax2.get_yticklabels(), visible=False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

    return fig

# ================= 使用示例 =================
# 在你的 forward 或者 validation loop 里：
# A_soft 是模型输出的 [B, N, K]
# visualize_asoft_diagnosis(A_soft, node_mask, save_path="debug_asoft.png")