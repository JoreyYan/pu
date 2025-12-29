import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Sequence, Tuple
from data import utils as du
from models.pool import LearnOnlyGaussianPooling,coarse_rigids_from_mu_sigma,UniformAnchorSemGeoAssign
from models.IGA import InvariantGaussianAttention,CoarseIGATower,GaussianUpdateBlock
from data.GaussianRigid import save_gaussian_as_pdb
from models.loss import HierarchicalGaussianLoss,SymmetricGaussianLoss

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
            k_max_cap: int = 64,
            coarse_iga_layers: int | list[int] = 4,  # <--- 现在可配置
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
        #     LearnOnlyGaussianPooling(
        #         c_s=c_s, ratio=ratio, k_max_cap=k_max_cap,
        #         tau_init=0.2, slots_init_scale=0.2,
        #     )
        #     for _ in range(num_downsample)
        # ])

        self.pools = nn.ModuleList([
            UniformAnchorSemGeoAssign(
                c_s=c_s,
                ratio=ratio,
                k_max_cap=k_max_cap,

                # -------- 关键超参数（强烈建议这样起步） --------
                sem_dim=128,  # 语义投影维度（稳定，不要太大）
                w_geo_init=1.0,  # 几何主导（先保证不重叠）
                w_sem_init=0.1,  # 语义弱绑定（防止 slot 对称）
                sigma_nm_init=1.0,  # nm 级几何 soft 半径
                tau_sem_init=1.0,  # 语义温度
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
                    num_layers=per_level_layers[lv],  # <--- 每层不同
                )
            )

        # 初始化 Loss 模块
        self.hier_loss = HierarchicalGaussianLoss(w_sep=10.0, w_compact=1)  # compact 权重别太大
        # 实例化
       # self.parent_child_loss = SymmetricGaussianLoss(w_center_p=1.0, w_center_c=10.0, w_shape=0.0, eps=1e-6)

    def forward(self, s_f, r_f, mask_f, step: int, total_steps: int):
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
            A, s_c, mu_c, Sigma_c, pool_loss = self.pools[lv](
                s=s, mu=mu, Sigma=Sigma, mask=mask,
                w_occ=1.0,
                w_rep=0.1,
                w_ent=0.0 if t < 0.5 else 0.01,   # 你要的“早软后硬”
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
            # pc_stats = self.parent_child_loss(
            #     mu_child=mu,
            #     Sigma_child=Sigma,
            #     mu_parent=mu_c,
            #     Sigma_parent=Sigma_c,
            #     A=A,
            #     mask=mask
            # )
            # pool_reg_total += pc_stats["loss"]

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
            #     filename=f"debug_lv{lv}_r_cang_postpool_gaussian_mean_anchor.pdb",
            #     mask=mask_c,
            #     center_mode="gaussian_mean",
            # )



            # 6) coarse IGA × (coarse_iga_layers)
            s_c, r_c = self.coarse_towers[lv](s_c, r_c, mask_c)


            # 记录 level
            levels.append({
                "A": A,
                "s": s_c,
                "r": r_c,
                "mask": mask_c,
                "pool_loss": pool_loss,
            })

            # 下一层输入
            s, r, mask = s_c, r_c, mask_c

        return levels, pool_reg_total
