"""
Orchestrated, stagewise-structured version of flow_model_HGF_clean.py

原则：
- 100% 忠于原始逻辑
- 不引入任何新模块
- 不改变任何计算
- 仅通过“再包装一层 orchestration”让主结构一眼可读
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from data import utils as du

# ===== 原始依赖 =====
from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models.features.backbone_gnn_feature import BackboneEncoderGNN
from models import ipa_pytorch
from models.shattetnion.ShDecoderSidechain import SideAtomsFeatureHead, SequenceHead
from openfold.model.primitives import Linear
from openfold.utils.rigid_utils import Rigid,Rotation
from data.GaussianRigid import OffsetGaussianRigid

from models.components.frozen_esm import FrozenEsmModel
from models.components.sequence_adapters import SequenceToTrunkNetwork

from models.IGA import (
    InvariantGaussianAttention,
    GaussianUpdateBlock,
    BottleneckIGAModule,
FastTransformerBlock,
BottleneckSemanticModule
)
from models.downblock import HierarchicalDownsampleIGAModule,HierarchicalDownsampleModuleFast
from models.upsample_block import HierarchicalUpsampleIGAModule,HierarchicalUpsampleSemanticModule
from models.finnalup import FinalCoarseToFineIGAModule
from models.FinalSemUp import FinalCoarseToFineSemanticUpModule


# ============================================================
# Sidechain Atom Head（与你 clean 版一致）
# ============================================================
class SidechainAtomHead(nn.Module):
    def __init__(self, c_in: int, num_atoms: int = 10, base_thickness_ang: float = 0.5):
        super().__init__()
        self.num_atoms = num_atoms
        self.base_thickness_ang = float(base_thickness_ang)
        self.ang_to_nm = 0.1

        self.proj = nn.Sequential(
            Linear(c_in, c_in),
            nn.LayerNorm(c_in),
            nn.SiLU(),
            Linear(c_in, num_atoms * 3),
        )
        nn.init.normal_(self.proj[-1].weight, std=1e-4)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, s, gaussian_rigid, atom_mask, thickness_nm=None):
        B, N, _ = s.shape
        mu = gaussian_rigid._local_mean
        sigma = torch.exp(gaussian_rigid._scaling_log)

        if thickness_nm is None:
            thickness_nm = torch.full(
                (B, N, 1),
                self.base_thickness_ang * self.ang_to_nm,
                device=s.device,
                dtype=s.dtype,
            )
        if thickness_nm.shape[-1] == 1:
            thickness_nm = thickness_nm.expand(B, N, 3)

        sigma_eff = torch.maximum(sigma, thickness_nm)

        u_raw = self.proj(s).view(B, N, self.num_atoms, 3)
        u = torch.tanh(u_raw) * (1.0 + 0.2 * torch.abs(u_raw))

        local = mu[:, :, None, :] + u * sigma_eff[:, :, None, :]
        local = local #* atom_mask[..., None]
        global_xyz = gaussian_rigid.unsqueeze(-1).apply(local)
        return global_xyz, local, gaussian_rigid


# ============================================================
# 主模型（Orchestrated Stagewise）
# ============================================================
class HierarchicalGaussianFieldModel(nn.Module):
    """
    Orchestrated + Stagewise 版本

    顶层 forward 只表达“结构顺序”
    具体实现全部来自你原始 clean 版
    """

    def __init__(self, model_conf):
        super().__init__()
        self.conf = model_conf
        self.ipa = model_conf.ipa

        # ========== Stage 0: 特征模块 ==========
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.node_feature_ln = nn.LayerNorm(self.ipa.c_s)
        self.graph_feature_ln = nn.LayerNorm(self.ipa.c_s)




        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)
        self.feature_graph = BackboneEncoderGNN(dim_nodes=self.ipa.c_s)

        self.use_side=False
        self.SEM_only = False

        if  self.use_side:

            self.sidechain_hidden = model_conf.sidechain_atoms.get("hidden", 256)
            self.sidechain_feature_ln = nn.LayerNorm(self.sidechain_hidden)
            self.sidechain_head = SideAtomsFeatureHead(
                A=model_conf.sidechain_atoms.get("A", 10),
                hidden=self.sidechain_hidden,
                num_classes=0,
                dropout=model_conf.sidechain_atoms.get("dropout", 0.1),
                conv_blocks=model_conf.sidechain_atoms.get("conv_blocks", 4),
                mlp_blocks=model_conf.sidechain_atoms.get("mlp_blocks", 4),
                fuse_blocks=model_conf.sidechain_atoms.get("fuse_blocks", 4),
                conv_groups=model_conf.sidechain_atoms.get("conv_groups", 1),
            )
        else:
            self.sidechain_hidden=0

        self.use_esm = getattr(model_conf, "use_esm", False)
        if self.use_esm:
            self.esm = FrozenEsmModel(model_key=model_conf.esm_model, use_esm_attn_map=True)
            self.seq_to_trunk = SequenceToTrunkNetwork(
                esm_single_dim=self.esm.single_dim,
                num_layers=self.esm.num_layers,
                d_single=self.ipa.c_s,
                esm_attn_dim=self.esm.attn_head * self.esm.num_layers,
                d_pair=model_conf.edge_embed_size,
                position_bins=32,
                pairwise_state_dim=model_conf.edge_embed_size,
            )

        node_in = self.ipa.c_s + self.sidechain_hidden + self.ipa.c_s
        if self.use_esm:
            node_in += self.ipa.c_s

        self.node_fusion = nn.Sequential(
            nn.Linear(node_in, self.ipa.c_s),
            nn.LayerNorm(self.ipa.c_s),
            nn.SiLU(),
            nn.Linear(self.ipa.c_s, self.ipa.c_s),
        )

        edge_in = self.ipa.c_z + self.ipa.c_z
        if self.use_esm:
            edge_in += model_conf.edge_embed_size
        self.edge_init_ln = nn.LayerNorm(self.conf.edge_embed_size)
        self.edge_graph_ln = nn.LayerNorm(self.ipa.c_z)
        self.edge_fusion = nn.Sequential(
            nn.Linear(edge_in, self.ipa.c_z),
            nn.LayerNorm(self.ipa.c_z),
            nn.SiLU(),
            nn.Linear(self.ipa.c_z, self.ipa.c_z),
        )

        # ========== Stage 2: IGA Trunk ==========
        self.trunk = nn.ModuleDict()
        for b in range(self.ipa.num_blocks):
            self.trunk[f"iga_{b}"] = InvariantGaussianAttention(
                c_s=self.ipa.c_s,
                c_z=self.ipa.c_z,
                c_hidden=self.ipa.c_hidden,
                no_heads=self.ipa.no_heads,
                no_qk_gaussians=self.ipa.no_qk_points,
                no_v_points=self.ipa.no_v_points,
                layer_idx=b,
            )
            self.trunk[f"iga_ln_{b}"] = nn.LayerNorm(self.ipa.c_s)

            # Seq Transformer
            tfmr_in = self.ipa.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self.ipa.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,  # or *2
                batch_first=True, dropout=0.0, norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self.ipa.seq_tfmr_num_layers)

            self.trunk[f'post_tfmr_{b}'] = Linear(tfmr_in, self.ipa.c_s, init="final")
            # Transition
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(c=self.ipa.c_s)


            edge_in = self.conf.edge_embed_size
            self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                node_embed_size=self.ipa.c_s,
                edge_embed_in=edge_in,
                edge_embed_out=self.conf.edge_embed_size,
            )
            self.trunk[f"gau_update_{b}"] = GaussianUpdateBlock(self.ipa.c_s)

        # ========== Stage 3: HGF ==========
        if not self.SEM_only:
            self.down = HierarchicalDownsampleIGAModule(
                c_s=self.ipa.c_s,
                iga_conf=self.ipa,
                num_downsample=self.conf.num_downsample,
                OffsetGaussianRigid_cls=OffsetGaussianRigid,
            )
            self.bottleneck = BottleneckIGAModule(
                c_s=self.ipa.c_s,
                iga_conf=self.ipa,
            )


            self.up = HierarchicalUpsampleIGAModule(
                c_s=self.ipa.c_s,
                iga_conf=self.ipa,
                num_upsample=self.conf.num_downsample-1,
                OffsetGaussianRigid_cls=OffsetGaussianRigid,
            )


            self.final_up = FinalCoarseToFineIGAModule(
                c_s=self.ipa.c_s,
                iga_conf=self.ipa,
                OffsetGaussianRigid_cls=OffsetGaussianRigid,
            )

        else:
            ####sem

            # self.down = HierarchicalDownsampleModuleFast(
            #     c_s=self.ipa.c_s,
            #     iga_conf=self.ipa,
            #     num_downsample=self.conf.num_downsample,
            #     OffsetGaussianRigid_cls=OffsetGaussianRigid,
            # )
            #
            # self.bottleneck = BottleneckSemanticModule(
            #     c_s=self.ipa.c_s,
            #     bottleneck_layers=6,
            #     n_heads=self.ipa.no_heads ,
            #     mlp_ratio=4.0,
            #     dropout=0.0,
            # )
            # self.up = HierarchicalUpsampleSemanticModule(
            #     c_s=self.ipa.c_s,
            #     num_upsample=self.conf.num_downsample - 1,
            #     M_max=8,
            #     up_ratio=6.0,  # Kt = active_parent * 6（期望）
            #     K_target=None,  # None => 用 up_ratio 自动预算
            #     tower_layers=4,  # 先小后大都行：也支持 list
            #     tower_heads=4,
            #     use_cross_attn=True,  # 你这套里 cross-attn 是 up-init 内部那一下
            #     tau_init=1.0,
            #     beta_init=0.0,
            #     mlp_ratio=4.0,
            #     dropout=0.0,
            # )

            self.final_up = FinalCoarseToFineSemanticUpModule(
                conf=self.ipa,
                c_s=self.ipa.c_s,
                OffsetGaussianRigid_cls=OffsetGaussianRigid,
                RotationCls=Rotation,  # 你工程里 Rotation 类
                num_tf_layers=6,  # 语义 refine
                neighbor_R=3,  # 每个 residue top2 coarse 父
                use_chain_emb=False,  # 你有 chain_id 再开
            )

        # ========== Stage 4: Heads ==========
        self.seq_head = SequenceHead(self.ipa.c_s, self.ipa.c_s, num_classes=21)
        self.atom_head = SidechainAtomHead(
            c_in=self.ipa.c_s,
            num_atoms=4,
            base_thickness_ang=getattr(model_conf, "base_thickness", 0.5),
        )

    # ======================================================
    # Top-level forward（只表达结构）
    # ======================================================
    def forward(self, input_feats, step: int, total_steps: int):
        node_mask = input_feats["res_mask"]
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        res_index = input_feats['res_idx']
        chain_idx = input_feats['chain_idx']

        # Stage 0
        node_embed, edge_embed, sideatom_mask = self.forward_stage0_features(
            input_feats, node_mask, edge_mask,diffuse_mask,res_index,chain_idx
        )

        # Stage 1
        rigids_nm, thickness_nm = self.forward_stage1_geometry(input_feats)

        # Stage 2
        node_embed,edge_embed, rigids_nm = self.forward_stage2_trunk(
            node_embed, edge_embed, rigids_nm, node_mask, edge_mask,input_feats
        )

        # Stage 3
        s_res, r_res, reg_hgf = self.forward_stage3_hgf(
            node_embed, edge_embed,rigids_nm, node_mask, input_feats, step, total_steps
        )

        # Stage 4
        return self.forward_stage4_heads(
            s_res, r_res, sideatom_mask, thickness_nm, None
        )

    # ======================================================
    # 各 Stage 的真实实现（原样搬运）
    # ======================================================
    def forward_stage0_features(self, input_feats, node_mask, edge_mask,diffuse_mask, res_index, chain_idx):
        return self._extract_features(input_feats, node_mask, edge_mask,diffuse_mask, res_index, chain_idx)

    def forward_stage1_geometry(self, input_feats):
        base_rigid = du.create_rigid(
            input_feats["rotmats_1"], input_feats["trans_1"]
        )
        return self._init_gaussian_rigids_nm(input_feats, base_rigid)

    def forward_stage2_trunk(self, node_embed, edge_embed, rigids_nm, node_mask,edge_mask, input_feats):
        for b in range(self.ipa.num_blocks):
            iga_out = self.trunk[f"iga_{b}"](s=node_embed, z=edge_embed, r=rigids_nm, mask=node_mask)
            iga_out = iga_out * node_mask[..., None]
            node_embed = self.trunk[f"iga_ln_{b}"](node_embed + iga_out)

            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool)
            )
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)

            node_embed = self.trunk[f"node_transition_{b}"](node_embed) * node_mask[..., None]


            edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed) * edge_mask[..., None]

            rigids_nm = self.trunk[f"gau_update_{b}"](
                node_embed, rigids_nm, mask=input_feats["update_mask"]
            )


        return node_embed,edge_embed, rigids_nm

    def forward_stage3_hgf(self, node_embed,edge_embed, rigids_nm, node_mask, input_feats, step, total_steps):
        levels_down, reg_down = self.down(
            node_embed,edge_embed, rigids_nm, node_mask, step, total_steps
        )
        sL, zL,rL, mL = levels_down[-1]["s"], levels_down[-1]["z"], levels_down[-1]["r"], levels_down[-1]["mask"]
        sL,zL, rL = self.bottleneck(sL,zL, rL, mL)
        # levels_up, reg_up = self.up(sL, rL, mL, step, total_steps)

        # last = levels_up[-1]





        if not self.SEM_only:
            final_levels, reg_final = self.final_up(
                s_parent=sL,
                r_parent=rL,
                mask_parent=mL,
                node_mask=node_mask,
                res_idx=input_feats["res_idx"],
            )

            # final_levels, reg_final = self.final_up(
            #     s_parent=node_embed,
            #     r_parent=rigids_nm,
            #     mask_parent=node_mask,
            #     node_mask=node_mask,
            #     res_idx=input_feats["res_idx"],
            # )

        else:
            # final_levels, reg_final = self.final_up(
            #     s_parent=last["s"],
            #     mask_parent=last["mask"],
            #     node_mask=node_mask,
            #     res_idx=input_feats["res_idx"],
            # )
            final_levels, reg_final = self.final_up(
                s_parent=sL,
                mask_parent=node_mask,
                node_mask=node_mask,
                res_idx=input_feats["res_idx"],
            )



        s_res = final_levels[-1]["s"]
        r_res = final_levels[-1]["r"]
        return s_res, r_res, None


    def _make_noise_t(self, input_feats, node_mask):
        if "r3_t" in input_feats:
            return input_feats["r3_t"]
        return torch.ones((node_mask.shape[0], 1), device=node_mask.device, dtype=node_mask.dtype)
    def forward_stage4_heads(self, s_res, r_res, atom_mask, thickness_nm, reg_hgf):
        logits = self.seq_head(s_res)
        pred_g_nm, pred_l_nm, _ = self.atom_head(
            s_res, r_res, atom_mask, thickness_nm
        )

        r_res = r_res.scale_translation(du.NM_TO_ANG_SCALE)  # 10.

        return {
            "logits": logits,
            "pred_atoms": pred_l_nm * du.NM_TO_ANG_SCALE,
            "pred_atoms_global": pred_g_nm * du.NM_TO_ANG_SCALE,
            "reg_total": reg_hgf,
            "final_gaussian": r_res,  # <- 直接透传
            "rigids": r_res,  # <- 你喜欢也可留一个别名

        }

    # ======================================================
    # ===== 以下两个函数：直接复制你 clean 版的实现 =====
    # ======================================================
    def _extract_features(self, input_feats, node_mask, edge_mask, diffuse_mask, res_index, chain_idx):
        """
        Returns:
          node_embed [B,N,Cs]
          edge_embed [B,N,N,Cz]
          sidechain_atom_mask [B,N,A] (if present else ones)
          seq_emb_s/seq_emb_z maybe
        """
        noise_t = self._make_noise_t(input_feats, node_mask)

        # node feature
        init_node_embed = self.node_feature_net(noise_t, node_mask, diffuse_mask, res_index)

        # sidechain CNN feature (optional)
        sidechain_features = None
        sidechain_atom_mask = input_feats.get("sidechain_atom_mask", None)

        if  self.use_side:

            if "atoms14_local_t" in input_feats:
                atoms14_local_t = input_feats["atoms14_local_t"]
                sidechain_atoms = atoms14_local_t[..., 4:14, :]  # (10 atoms)
                if sidechain_atom_mask is None:
                    sidechain_atom_mask = torch.ones((*atoms14_local_t.shape[:2], 10), device=atoms14_local_t.device)
                _, sidechain_features = self.sidechain_head(sidechain_atoms, atom_mask=sidechain_atom_mask, node_mask=node_mask)

        # backbone GNN features
        node_h, edge_h, *_ = self.feature_graph(input_feats["atoms14_local_t"][..., :4, :], chain_idx)

        # ESM features (optional)
        if self.use_esm and self.seq_encoder is not None:
            seq_emb_s, seq_emb_z = self.seq_encoder(input_feats["aatype"], chain_idx, attn_mask=node_mask)
            seq_emb_s, seq_emb_z = self.sequence_to_trunk(seq_emb_s, seq_emb_z, res_index, node_mask)
        else:
            seq_emb_s, seq_emb_z = None, None

        # normalize + fuse node
        init_node_embed_n = self.node_feature_ln(init_node_embed)
        node_h_n = self.graph_feature_ln(node_h)

        if self.use_side:
            if sidechain_features is None:
                # FIX: if missing, inject zeros so fusion shape is stable.
                sidechain_features = torch.zeros((init_node_embed.shape[0], init_node_embed.shape[1], self.sidechain_hidden),
                                                device=init_node_embed.device, dtype=init_node_embed.dtype)
            sidechain_features_n = self.sidechain_feature_ln(sidechain_features)

            parts = [init_node_embed_n, sidechain_features_n, node_h_n]
        else:
            parts = [init_node_embed_n, node_h_n]

        if self.use_esm and seq_emb_s is not None:
            parts.append(self.esm_feature_ln(seq_emb_s))
        node_embed = self.node_fusion(torch.cat(parts, dim=-1)) * node_mask[..., None]

        # edge feature net
        trans_t = input_feats["trans_1"]
        init_edge_embed = self.edge_feature_net(node_embed, trans_t, edge_mask, diffuse_mask)
        init_edge_embed_n = self.edge_init_ln(init_edge_embed)
        edge_h_n = self.edge_graph_ln(edge_h)

        if self.use_esm and seq_emb_z is not None:
            seq_emb_z_n = self.edge_esm_ln(seq_emb_z)
            edge_embed = self.edge_fusion(torch.cat([init_edge_embed_n, edge_h_n, seq_emb_z_n], dim=-1))
        else:
            edge_embed = self.edge_fusion(torch.cat([init_edge_embed_n, edge_h_n], dim=-1))
        edge_embed = edge_embed * edge_mask[..., None]

        if sidechain_atom_mask is None:
            sidechain_atom_mask = torch.ones((*node_embed.shape[:2], 10), device=node_embed.device, dtype=node_embed.dtype)

        return node_embed, edge_embed, sidechain_atom_mask

    def _init_gaussian_rigids_nm(self, input_feats, base_rigid):
        """
        Build OffsetGaussianRigid from available atoms + masks, and scale to nm.
        Also returns thickness_nm tensor for atom_head stabilization.
        """
        node_mask = input_feats["res_mask"]
        is_masked = input_feats["update_mask"].bool()  # 1=masked (to predict)

        # Dynamic thickness in Angstrom then to nm
        thickness_ang = torch.where(
            is_masked,
            torch.tensor(2.5, device=node_mask.device, dtype=torch.float32),
            torch.tensor(0.5, device=node_mask.device, dtype=torch.float32),
        ).unsqueeze(-1)  # [B,N,1] ang
        thickness_nm = thickness_ang.to(node_mask.dtype) * du.ANG_TO_NM_SCALE  # [B,N,1] nm

        atoms14_local = input_feats["atoms14_local_t"][..., :3]
        all_atoms_global = base_rigid.unsqueeze(-1).apply(atoms14_local)  # [B,N,14,3] ang

        gt_exists = input_feats["atom14_gt_exists"].float()  # [B,N,14]
        is_masked_b = is_masked.unsqueeze(-1)  # [B,N,1]

        # masked residues: keep only N,CA,C (0..2), drop O+SC
        mask_bb = gt_exists[..., :3]
        mask_others = gt_exists[..., 3:] * (~is_masked_b).float()
        geom_mask_all = torch.cat([mask_bb, mask_others], dim=-1)

        curr_rigids = OffsetGaussianRigid.from_rigid_and_all_atoms(
            base_rigid,
            all_atoms_global,
            geom_mask_all,
            base_thickness=thickness_ang,  # in ang; constructor expects ang in your original codepath
        )
        # Scale to nm for neural stability
        curr_rigids = curr_rigids.scale_translation(du.ANG_TO_NM_SCALE)  # 0.1

        return curr_rigids, thickness_nm
