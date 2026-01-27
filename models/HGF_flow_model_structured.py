"""
Orchestrated, stagewise-structured version of flow_model_HGF_clean.py

原则：
- 100% 忠于原始逻辑
- 不引入任何新模块
- 不改变任何计算
- 仅通过“再包装一层 orchestration”让主结构一眼可读
"""
# 强制切换到交互式后端
# Windows/Linux 桌面通常用 'TkAgg' 或 'Qt5Agg'
# MacOS 通常用 'MacOSX'
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from analysis.draw import plot_mse_analysis
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
import matplotlib

import matplotlib.pyplot as plt
plt.ion()

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
from data.GaussianRigid import OffsetGaussianRigid,save_gaussian_as_pdb,save_parents_as_ply,save_parents_as_pdb_manual_cov,save_parents_as_pdb_explicit
from chroma.layers.structure.backbone import FrameBuilder
from models.components.frozen_esm import FrozenEsmModel
from models.components.sequence_adapters import SequenceToTrunkNetwork

from models.IGA import (
    InvariantGaussianAttention,
    GaussianUpdateBlock,
    BottleneckIGAModule,
FastTransformerBlock,
BottleneckSemanticModule
)
from models.downblock import HierarchicalDownsampleIGAModule,HierarchicalDownsampleModuleFast,SimpleSegmentDownIGAModule_v1
from models.upsample_block import HierarchicalUpsampleIGAModule,HierarchicalUpsampleSemanticModule
from models.finnalup import UpXiPredictorAsoftPos,UpXiPredictorAttnPos_Query,FinalCoarseToFineDensenSampleIGAModulev3_2,FinalCoarseToFineDensenSampleIGAModulev2,FinalCoarseToFineDensenSampleIGAModulev3
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

class HierarchicalGaussianFieldModel_Generate(nn.Module):
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

        self.FrameBuilder=FrameBuilder()

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
        rigids_nm = self.forward_stage1_geometry(input_feats)

        # Stage 2
        node_embed,edge_embed, rigids_nm = self.forward_stage2_trunk(
            node_embed, edge_embed, rigids_nm, node_mask, edge_mask,input_feats
        )

        result=self.forward_stage4_heads(node_embed,rigids_nm)


        return result

    # ======================================================
    # 各 Stage 的真实实现（原样搬运）
    # ======================================================
    def forward_stage0_features(self, input_feats, node_mask, edge_mask,diffuse_mask, res_index, chain_idx):
        return self._extract_features(input_feats, node_mask, edge_mask,diffuse_mask, res_index, chain_idx)

    def forward_stage1_geometry(self, input_feats):
        # 1. Rigids (Ang -> NM)
        if "rigids_t" in input_feats:
            rigids_ang = input_feats["rigids_t"]
        else:
            # Fallback: construct from trans_t/rotmats_t if rigids_t missing
            # This handles inference or legacy cases
            B, N = input_feats["res_mask"].shape
            device = input_feats["res_mask"].device
            dummy_s = torch.zeros(B, N, 3, device=device)
            dummy_m = torch.zeros(B, N, 3, device=device)
            rigids_ang = OffsetGaussianRigid(
                ru.Rotation(rot_mats=input_feats["rotmats_t"]),
                input_feats["trans_t"],
                dummy_s,
                dummy_m
            )

        rigids_nm = rigids_ang.scale_translation(du.ANG_TO_NM_SCALE)


        return rigids_nm

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
                node_embed, rigids_nm, mask=input_feats["diffuse_mask"]
            )


        return node_embed,edge_embed, rigids_nm





    def _make_noise_t(self, input_feats, node_mask):
        if "t" in input_feats:
            return input_feats["t"]
        return torch.ones((node_mask.shape[0], 1), device=node_mask.device, dtype=node_mask.dtype)
    def forward_stage4_heads(self, s_res, r_res):
        logits = self.seq_head(s_res)


        r_res = r_res.scale_translation(du.NM_TO_ANG_SCALE)  # 10.

        return {
            "logits": logits,
            # "pred_atoms": pred_l_nm * du.NM_TO_ANG_SCALE,
            # "pred_atoms_global": pred_g_nm * du.NM_TO_ANG_SCALE,
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


        # backbone GNN features
        noisy_backbone=self.FrameBuilder(input_feats['rotmats_t'],input_feats['trans_t'],chain_idx)
        node_h, edge_h, *_ = self.feature_graph(noisy_backbone, chain_idx)

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
        trans_t = input_feats["trans_t"]
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

        # self.trunk_out = nn.ModuleDict()
        # for b in range(self.ipa.num_blocks):
        #     self.trunk_out[f"iga_{b}"] = InvariantGaussianAttention(
        #         c_s=self.ipa.c_s,
        #         c_z=self.ipa.c_z,
        #         c_hidden=self.ipa.c_hidden,
        #         no_heads=self.ipa.no_heads,
        #         no_qk_gaussians=self.ipa.no_qk_points,
        #         no_v_points=self.ipa.no_v_points,
        #         layer_idx=b,
        #     )
        #     self.trunk_out[f"iga_ln_{b}"] = nn.LayerNorm(self.ipa.c_s)
        #
        #     # Seq Transformer
        #     tfmr_in = self.ipa.c_s
        #     tfmr_layer = torch.nn.TransformerEncoderLayer(
        #         d_model=tfmr_in,
        #         nhead=self.ipa.seq_tfmr_num_heads,
        #         dim_feedforward=tfmr_in,  # or *2
        #         batch_first=True, dropout=0.0, norm_first=False
        #     )
        #     self.trunk_out[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
        #         tfmr_layer, self.ipa.seq_tfmr_num_layers)
        #
        #     self.trunk_out[f'post_tfmr_{b}'] = Linear(tfmr_in, self.ipa.c_s, init="final")
        #     # Transition
        #     self.trunk_out[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(c=self.ipa.c_s)
        #
        #     edge_in = self.conf.edge_embed_size
        #     self.trunk_out[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
        #         node_embed_size=self.ipa.c_s,
        #         edge_embed_in=edge_in,
        #         edge_embed_out=self.conf.edge_embed_size,
        #     )
        #     self.trunk_out[f"gau_update_{b}"] = GaussianUpdateBlock(self.ipa.c_s)

        # ========== Stage 3: HGF ==========
        if not self.SEM_only:
            self.down = SimpleSegmentDownIGAModule_v1(
                c_s=self.ipa.c_s,
                iga_conf=self.ipa,

            )
            # self.bottleneck = BottleneckIGAModule(
            #     c_s=self.ipa.c_s,
            #     iga_conf=self.ipa,
            # )
            #
            #
            # self.up = HierarchicalUpsampleIGAModule(
            #     c_s=self.ipa.c_s,
            #     iga_conf=self.ipa,
            #     num_upsample=self.conf.num_downsample-1,
            #     OffsetGaussianRigid_cls=OffsetGaussianRigid,
            # )
            #
            #
            # self.final_up = FinalCoarseToFineDensenSampleIGAModulev3_2(
            #     c_s=self.ipa.c_s,
            #     iga_conf=self.ipa,
            #     OffsetGaussianRigid_cls=OffsetGaussianRigid,
            # )

            self.final_up=UpXiPredictorAttnPos_Query(c_s=self.ipa.c_s, use_trunk_query=False)

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

        # 在 forward_stage3_hgf 里，拿得到 rigids_nm / node_mask / chain_idx
        # if (step % 50 == 0):  # 你自己调频率
        #     dbg = segment_ae_debug_hook(
        #         rigids_nm=rigids_nm,
        #         node_mask=node_mask,
        #         chain_idx=input_feats.get("chain_idx", None),
        #         Kmax=64,
        #         min_len=4,
        #         max_len=10,
        #         s_floor=1e-3,
        #         base_point_thickness_nm=0.2,
        #         save_pdb=True,
        #         filename_prefix=f"debug__segAE_step{step}",
        #         OffsetGaussianRigid_cls=OffsetGaussianRigid,
        #         save_gaussian_as_pdb_fn=save_gaussian_as_pdb,  # 你工程已有
        #     )
        #     downmetric =  {k: v for k, v in dbg.items()}

        # Stage 3
        # s_res,z_res, r_res, reg_hgf,downmetric = self.forward_stage3_hgf(
        #     node_embed, edge_embed,rigids_nm, chain_idx,node_mask, input_feats, step, total_steps
        # )

        result,reg,downmetric = self.forward_stage3_hgf(
            node_embed, edge_embed,rigids_nm, chain_idx,node_mask, input_feats, step, total_steps
        )

        # Stage 3.5
        # s_res,edge_embed, r_res = self.forward_stage3_half_trunk(
        #     s_res, z_res, r_res, node_mask, edge_mask,input_feats
        # )
        # Stage 4
        # result=self.forward_stage4_heads(
        #     s_res, r_res, sideatom_mask, thickness_nm, reg_hgf
        # )
        return result,reg,downmetric

    # ======================================================
    # 各 Stage 的真实实现（原样搬运）
    # ======================================================
    def forward_stage0_features(self, input_feats, node_mask, edge_mask,diffuse_mask, res_index, chain_idx):
        return self._extract_features(input_feats, node_mask, edge_mask,diffuse_mask, res_index, chain_idx)

    def forward_stage1_geometry(self, input_feats):
        # 1. Rigids (Ang -> NM)
        if "rigids_t" in input_feats:
            rigids_ang = input_feats["rigids_t"]
        else:
            # Fallback: construct from trans_t/rotmats_t if rigids_t missing
            # This handles inference or legacy cases
            B, N = input_feats["res_mask"].shape
            device = input_feats["res_mask"].device
            dummy_s = torch.zeros(B, N, 3, device=device)
            dummy_m = torch.zeros(B, N, 3, device=device)
            rigids_ang = OffsetGaussianRigid(
                ru.Rotation(rot_mats=input_feats["rotmats_t"]),
                input_feats["trans_t"],
                dummy_s,
                dummy_m
            )

        rigids_nm = rigids_ang.scale_translation(du.ANG_TO_NM_SCALE)

        # 2. Thickness (Ang -> NM)
        node_mask = input_feats["res_mask"]
        if "update_mask" in input_feats:
            is_masked = input_feats["update_mask"].bool()
        else:
            is_masked = torch.zeros_like(node_mask).bool()

        thickness_ang = torch.where(
            is_masked,
            torch.tensor(2.5, device=node_mask.device, dtype=torch.float32),
            torch.tensor(0.5, device=node_mask.device, dtype=torch.float32),
        ).unsqueeze(-1)
        thickness_nm = thickness_ang.to(node_mask.dtype) * du.ANG_TO_NM_SCALE

        return rigids_nm, thickness_nm

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


    def forward_stage3_half_trunk(self, node_embed, edge_embed, rigids_nm, node_mask,edge_mask, input_feats):
        for b in range(self.ipa.num_blocks):
            iga_out = self.trunk_out[f"iga_{b}"](s=node_embed, z=edge_embed, r=rigids_nm, mask=node_mask)
            iga_out = iga_out * node_mask[..., None]
            node_embed = self.trunk_out[f"iga_ln_{b}"](node_embed + iga_out)

            seq_tfmr_out = self.trunk_out[f"seq_tfmr_{b}"](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool)
            )
            node_embed = node_embed + self.trunk_out[f"post_tfmr_{b}"](seq_tfmr_out)

            node_embed = self.trunk_out[f"node_transition_{b}"](node_embed) * node_mask[..., None]


            edge_embed = self.trunk_out[f"edge_transition_{b}"](node_embed, edge_embed) * edge_mask[..., None]

            rigids_nm = self.trunk_out[f"gau_update_{b}"](
                node_embed, rigids_nm, mask=input_feats["update_mask"]
            )


        return node_embed,edge_embed, rigids_nm


    def forward_stage3_hgf(self, node_embed,edge_embed, rigids_nm, chain_idx,node_mask, input_feats, step, total_steps):
        levels_down, reg_down = self.down(
            node_embed,edge_embed, rigids_nm, node_mask,chain_idx, step, total_steps
        )
        A_soft,sL, zL,rL, mask_parent ,curr_occ,downmetric= levels_down[-1]["A_soft"],levels_down[-1]["s"], levels_down[-1]["z"], levels_down[-1]["r"], levels_down[-1]["mask_parent"], levels_down[-1]["curr_occ"],levels_down[-1]['downmetric']
        # sL,zL, rL = self.bottleneck(sL,zL, rL, mL)
        # levels_up, reg_up = self.up(sL, rL, mL, step, total_steps)

        # last = levels_up[-1]





        if not self.SEM_only:
            # final_levels, reg_final = self.final_up(
            #     s_parent=sL,
            #     z_parent=zL,
            #     r_parent=rL,
            #     mask_parent=mL,
            #     node_mask=node_mask,
            #     s_trunk=node_embed,
            #     occ_parent=curr_occ,
            #     res_idx=input_feats["res_idx"],
            # )

            # final_levels, reg_final = self.final_up(
            #     s_parent=node_embed,
            #     r_parent=rigids_nm,
            #     mask_parent=node_mask,
            #     node_mask=node_mask,
            #     res_idx=input_feats["res_idx"],
            # )

            out2 = self.final_up(s_parent=sL, r_parent=rL, A_soft=None, mask_parent=mask_parent,node_mask=node_mask, s_trunk=node_embed)

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



        # s_res = final_levels[-1]["s"]
        # z_res = final_levels[-1]["z"]
        # r_res = final_levels[-1]["r"]
        #
        # debug = final_levels[-1]["aux"].debug
        #
        # downmetric=downmetric|debug
        #
        # reg=reg_down+reg_final

        out2["fig_dwon"]=levels_down[-1]["fig_dwon"]
        return out2,reg_down,downmetric

        # return s_res,z_res, r_res, reg,downmetric


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
            # "pred_atoms": pred_l_nm * du.NM_TO_ANG_SCALE,
            # "pred_atoms_global": pred_g_nm * du.NM_TO_ANG_SCALE,
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
        node_h, edge_h, *_ = self.feature_graph(input_feats['atom14_gt_positions'][..., :4, :], chain_idx)

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

# =========================
# 你工程里应该已有：
#   - OffsetGaussianRigid
#   - Rotation
#   - save_gaussian_as_pdb(gaussian_rigid, filename, mask, center_mode=...)
# 如果路径不同自己改 import
# =========================





# -------------------------
# 2) 段内旋转平均：平均 rot mats -> SVD 投影回 SO(3)
#    这比 PCA 更贴合蛋白 backbone frame
# -------------------------
def _project_to_so3(M: torch.Tensor) -> torch.Tensor:
    """
    M: [...,3,3]
    return R in SO(3) via SVD projection
    """
    U, _, Vt = torch.linalg.svd(M)
    R = U @ Vt
    # enforce det +1
    det = torch.det(R)
    # 若 det<0，翻 U 的最后一列
    flip = (det < 0).to(R.dtype)[..., None, None]
    U_fix = torch.cat([U[..., :, :2], U[..., :, 2:3] * (1.0 - 2.0 * flip)], dim=-1)
    R = U_fix @ Vt
    return R


def segment_avg_rotation(
    rotmats: torch.Tensor,      # [B,N,3,3]
    node_mask: torch.Tensor,    # [B,N]
    a_idx: torch.Tensor,        # [B,N]
    Kmax: int,
    mask_parent: torch.Tensor,  # [B,K]
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    返回 R_k: [B,K,3,3]
    """
    B, N = node_mask.shape
    device, dtype = rotmats.device, rotmats.dtype
    K = Kmax

    # one-hot [B,N,K]
    oh = F.one_hot(a_idx, num_classes=K).to(dtype) * node_mask[..., None]  # [B,N,K]
    w_sum = oh.sum(dim=1).clamp_min(1.0)  # [B,K]
    # 加权平均 rotmat（线性空间）
    # M_k = Σ_i w_i * R_i
    M = torch.einsum("bnk,bnij->bkij", oh, rotmats) / w_sum[..., None, None].clamp_min(eps)  # [B,K,3,3]
    Rk = _project_to_so3(M)

    # mask unused parents -> identity
    I = torch.eye(3, device=device, dtype=dtype)[None, None]
    Rk = Rk * mask_parent[..., None, None] + I * (1.0 - mask_parent[..., None, None])
    return Rk


# -------------------------
# 3) Down：从 CA 统计 μk, Rk, sk（对角）
# -------------------------
@dataclass
class ParentParams:
    mu: torch.Tensor         # [B,K,3]
    R: torch.Tensor          # [B,K,3,3]
    s: torch.Tensor          # [B,K,3]
    mask_parent: torch.Tensor# [B,K]


def build_parents_from_segments(
    x_ca: torch.Tensor,                # [B,N,3]  (nm)
    rotmats: torch.Tensor,             # [B,N,3,3] backbone frames
    node_mask: torch.Tensor,           # [B,N]
    a_idx: torch.Tensor,               # [B,N]
    mask_parent: torch.Tensor,         # [B,K]
    Kmax: int,
    s_floor: float = 1e-3,
    eps: float = 1e-6,
) -> ParentParams:
    B, N, _ = x_ca.shape
    device, dtype = x_ca.device, x_ca.dtype
    K = Kmax

    x = x_ca * node_mask[..., None]
    oh = F.one_hot(a_idx, num_classes=K).to(dtype) * node_mask[..., None]  # [B,N,K]
    cnt = oh.sum(dim=1).clamp_min(1.0)  # [B,K]

    mu = torch.einsum("bnk,bnd->bkd", oh, x) / cnt[..., None]  # [B,K,3]

    Rk = segment_avg_rotation(rotmats, node_mask, a_idx, Kmax=K, mask_parent=mask_parent, eps=eps)  # [B,K,3,3]

    # local coords: R^T (x - mu)
    # local[b,k,i,:] = Rk^T @ (x_i - mu_k)
    delta = (x[:, None, :, :] - mu[:, :, None, :])  # [B,K,N,3]
    local = torch.einsum("bkji,bknj->bkni", Rk, delta)  # [B,K,N,3]

    w = oh.permute(0, 2, 1)  # [B,K,N]
    local = local * w[..., None]  # mask
    denom = w.sum(dim=2).clamp_min(1.0)  # [B,K]
    var = (local ** 2).sum(dim=2) / denom[..., None]  # [B,K,3]
    s = torch.sqrt(var.clamp_min(eps)).clamp_min(s_floor)  # [B,K,3]

    # mask unused
    mu = mu * mask_parent[..., None]
    s = s * mask_parent[..., None]

    return ParentParams(mu=mu, R=Rk, s=s, mask_parent=mask_parent)


def build_parents_from_segments_mahalanobis_quantile(
        x_ca: torch.Tensor,  # [B,N,3]
        rotmats: torch.Tensor,  # [B,N,3,3]
        node_mask: torch.Tensor,  # [B,N]
        a_idx: torch.Tensor,  # [B,N]
        mask_parent: torch.Tensor,  # [B,K]
        Kmax: int,
        s_floor: float = 1e-3,
        eps: float = 1e-6,
        coverage_factor: float = 2.0,  # 新增：覆盖因子 (2.0 ~ 95% 覆盖)
) -> ParentParams:
    B, N, _ = x_ca.shape
    device, dtype = x_ca.device, x_ca.dtype
    K = Kmax

    x = x_ca * node_mask[..., None]
    oh = F.one_hot(a_idx, num_classes=K).to(dtype) * node_mask[..., None]  # [B,N,K]
    cnt = oh.sum(dim=1).clamp_min(1.0)  # [B,K]

    # 1. 计算中心 mu (不变)
    mu = torch.einsum("bnk,bnd->bkd", oh, x) / cnt[..., None]

    # 2. 计算旋转 R (不变)
    Rk = segment_avg_rotation(rotmats, node_mask, a_idx, Kmax=K, mask_parent=mask_parent, eps=eps)

    # 3. 投影到局部坐标 (不变)
    # local: [B,K,N,3]
    delta = (x[:, None, :, :] - mu[:, :, None, :])
    local = torch.einsum("bkji,bknj->bkni", Rk, delta)

    # -------------------------------------------------------------
    # 【核心修改】：从 "标准差" 改为 "最大包围/分位数包围"
    # -------------------------------------------------------------

    # mask 掉不属于该段的点 (设为 0)
    w = oh.permute(0, 2, 1)  # [B,K,N]
    local_abs = local.abs() * w[..., None]  # [B,K,N,3] 取绝对值

    # 方案 A: 绝对最大值 (Max Wrapping) - 保证 100% 包裹，但对离群点敏感
    # s_max = local_abs.max(dim=2)[0] # [B,K,3]
    # s = s_max

    # 方案 B (推荐): 软最大值 (Soft Max / Quantile) - 类似 densitypeak 的逻辑
    # 使用 power mean 近似 max，或者直接取 std * factor
    # 这里用 std * 2.0 也就是 2-Sigma，能覆盖 95%，比之前的 1-Sigma 强得多

    denom = w.sum(dim=2).clamp_min(1.0)
    var = (local ** 2).sum(dim=2) / denom[..., None]
    s_std = torch.sqrt(var.clamp_min(eps))

    # 关键点：乘以 coverage_factor (通常取 2.0 或 2.5)
    # 这样生成的椭圆半径是 2倍标准差，能包裹住绝大多数点
    s = s_std * coverage_factor

    # -------------------------------------------------------------

    # 保证最小尺寸
    s = s.clamp_min(s_floor)

    # mask unused
    mu = mu * mask_parent[..., None]
    s = s * mask_parent[..., None]

    return ParentParams(mu=mu, R=Rk, s=s, mask_parent=mask_parent)
# -------------------------
# 4) Up：模板 ξ（段内线性） decode -> x_hat
# -------------------------
def build_template_xi(
    a_idx: torch.Tensor,           # [B,N]
    seg_lens: torch.Tensor,        # [B,K]
    node_mask: torch.Tensor,       # [B,N]
    Kmax: int,
) -> torch.Tensor:
    """
    每个段内按顺序放在 [-1,1] 的 t，ξ=[t,0,0]
    返回 xi: [B,N,3]
    """
    B, N = a_idx.shape
    device = a_idx.device
    xi = torch.zeros((B, N, 3), device=device, dtype=torch.float32)

    for b in range(B):
        n = int(node_mask[b].sum().item())
        if n <= 0:
            continue
        # 逐段处理
        for i in range(n):
            k = int(a_idx[b, i].item())
            L = int(seg_lens[b, k].item())
            if L <= 1:
                t = 0.0
            else:
                # 段内位置：需要知道 i 在段内的相对序号
                # 简单做法：回扫找到段起点
                j = i
                while j > 0 and int(a_idx[b, j - 1].item()) == k:
                    j -= 1
                pos_in_seg = i - j  # 0..L-1
                t = -1.0 + 2.0 * (pos_in_seg / (L - 1))
            xi[b, i, 0] = t

    xi = xi * node_mask[..., None].float()
    return xi


def decode_from_parents(
    parents: ParentParams,
    a_idx: torch.Tensor,          # [B,N]
    xi: torch.Tensor,             # [B,N,3]
    node_mask: torch.Tensor,      # [B,N]
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    x_hat = mu_k + R_k * diag(s_k) * xi
    """
    B, N = a_idx.shape
    mu = parents.mu
    Rk = parents.R
    s = parents.s

    # gather per residue
    mu_i = mu.gather(1, a_idx[..., None].expand(B, N, 3))             # [B,N,3]
    s_i  = s.gather(1, a_idx[..., None].expand(B, N, 3)).clamp_min(eps)# [B,N,3]
    R_i  = Rk.gather(1, a_idx[..., None, None].expand(B, N, 3, 3))    # [B,N,3,3]

    local = xi * s_i
    x_hat = mu_i + torch.einsum("bnij,bnj->bni", R_i, local)
    x_hat = x_hat * node_mask[..., None]
    return x_hat


# -------------------------
# 5) 可视化：把点云做成“小球高斯”存 PDB（如果你有 save_gaussian_as_pdb）
# -------------------------
def build_point_gaussians_from_ca(
    OffsetGaussianRigid_cls,
    x_ca: torch.Tensor,         # [B,N,3]
    rots: torch.Tensor,         # [B,N,3,3] or None
    node_mask: torch.Tensor,    # [B,N]
    base_thickness: float = 0.2,  # nm
):
    """
    用你 OffsetGaussianRigid(local_mean=0) 表示每个点一个小球
    """
    B, N, _ = x_ca.shape
    device, dtype = x_ca.device, x_ca.dtype

    if rots is None:
        rots = torch.eye(3, device=device, dtype=dtype)[None, None].expand(B, N, 3, 3)

    # scaling_log: log([t,t,t])
    s = torch.full((B, N, 3), float(base_thickness), device=device, dtype=dtype)
    scaling_log = torch.log(s.clamp_min(1e-6))
    local_mean = torch.zeros((B, N, 3), device=device, dtype=dtype)

    rot_obj = Rotation(rot_mats=rots) if Rotation is not None else None
    # 你工程里的 OffsetGaussianRigid 构造签名：OffsetGaussianRigid(rots, trans, scaling_log, local_mean)
    return OffsetGaussianRigid_cls(rot_obj, x_ca, scaling_log, local_mean)


# -------------------------
# 6) 主入口：你在 forward_stage3_hgf 里调用它即可
# -------------------------
@torch.no_grad()
def segment_ae_debug_hook(
    rigids_nm,                    # OffsetGaussianRigid [B,N]
    node_mask: torch.Tensor,      # [B,N]
    chain_idx: Optional[torch.Tensor] = None,  # [B,N]
    Kmax: int = 64,
    min_len: int = 4,
    max_len: int = 10,
    s_floor: float = 1e-3,
    base_point_thickness_nm: float = 0.2,
    save_pdb: bool = True,
    filename_prefix: str = "debug__segAE",
    OffsetGaussianRigid_cls=None,
    save_gaussian_as_pdb_fn=None,
) -> Dict[str, Any]:
    """
    返回 debug dict，并可选写 PDB：
      - parents 椭圆
      - x_hat 重建点云
      - x_gt 原点云
    """
    device = node_mask.device

    # 1) 取 CA（nm）
    x_ca = rigids_nm.get_trans()  # [B,N,3] nm

    # 2) 取 backbone rotation（nm 空间无所谓）
    rotmats = rigids_nm.get_rots().get_rot_mats()  # [B,N,3,3]

    # 3) teacher 分段
    a_idx, mask_parent, seg_lens = teacher_segment_variable_length(
        node_mask=node_mask,
        chain_idx=chain_idx,
        min_len=min_len,
        max_len=max_len,
        Kmax=Kmax,
    )

    print(x_ca)
    print(node_mask)
    print(a_idx)
    print(mask_parent)

    # 4) Down：构建 parents
    parents = build_parents_from_segments_v3_debug(
        x_ca=x_ca,
        node_mask=node_mask,
        a_idx=a_idx,
        mask_parent=mask_parent,
        Kmax=Kmax,


    )

    # 5) Up：模板 xi decode
    xi = build_template_xi(a_idx=a_idx, seg_lens=seg_lens, node_mask=node_mask, Kmax=Kmax).to(x_ca.dtype)
    x_hat = decode_from_parents(parents=parents, a_idx=a_idx, xi=xi, node_mask=node_mask)

    # 6) 误差/统计
    mse = ((x_hat - x_ca) ** 2).sum(dim=-1)
    denom = node_mask.sum().clamp_min(1.0)
    mse_mean = (mse * node_mask).sum() / denom
    plot_mse_analysis(mse, xi, node_mask)
    # 父体尺度统计
    s_active = parents.s[parents.mask_parent > 0.5]  # [n_active,3] 可能为空
    if s_active.numel() > 0:
        s_mean = s_active.mean().item()
        s_min = s_active.min().item()
        s_max = s_active.max().item()
    else:
        s_mean, s_min, s_max = 0.0, 0.0, 0.0

    debug = {
        "segAE_mse_mean": mse_mean.detach().cpu(),
        "segAE_s_mean": torch.tensor(s_mean),
        "segAE_s_min": torch.tensor(s_min),
        "segAE_s_max": torch.tensor(s_max),
        "segAE_active_K": parents.mask_parent.sum(dim=-1).mean().detach().cpu(),  # batch mean
    }

    # 7) 可选写 PDB（你已有 save_gaussian_as_pdb）
    if save_pdb and (save_gaussian_as_pdb_fn is not None) and (OffsetGaussianRigid_cls is not None):
        # parents 椭圆：local_mean=0, trans=mu_k, rots=R_k, scaling=s_k
        B, K = parents.mu.shape[:2]
        parent_rot = Rotation(rot_mats=parents.R.transpose(-1,-2))
        parent_scaling_log = torch.log(parents.s.clamp_min(1e-6))
        parent_local_mean = torch.zeros((B, K, 3), device=device, dtype=parents.mu.dtype)
        parents_rigid = OffsetGaussianRigid_cls(parent_rot, parents.mu, parent_scaling_log, parent_local_mean)



        # print("max|R_in - R_out|:", (parents.R - parents_rigid.get_rots().get_rot_mats()).abs().max().item())
        # print("max|R_in - R_out^T|:",
        #       (parents.R - parents_rigid.get_rots().get_rot_mats().transpose(-1, -2)).abs().max().item())

        # 点云：GT 和 recon 都做成“小球”
        gt_pts = build_point_gaussians_from_ca(
            OffsetGaussianRigid_cls=OffsetGaussianRigid_cls,
            x_ca=x_ca,
            rots=None,
            node_mask=node_mask,
            base_thickness=base_point_thickness_nm,
        )
        rec_pts = build_point_gaussians_from_ca(
            OffsetGaussianRigid_cls=OffsetGaussianRigid_cls,
            x_ca=x_hat,
            rots=None,
            node_mask=node_mask,
            base_thickness=base_point_thickness_nm,
        )

        # 写文件
        save_parents_as_pdb_explicit(
            parents=parents,
            filename=f"{filename_prefix}parents_data.pdb",
        )

        save_parents_as_ply(
            parents=parents,
            filename=f"{filename_prefix}parents_data.ply",
        )

        save_gaussian_as_pdb_fn(
            gaussian_rigid=parents_rigid,
            filename=f"{filename_prefix}__parents_v1envelope_v4.pdb",
            mask=parents.mask_parent,
            center_mode="gaussian_mean",
        )
        save_gaussian_as_pdb_fn(
            gaussian_rigid=gt_pts,
            filename=f"{filename_prefix}__gt_pts.pdb",
            mask=node_mask,
            center_mode="gaussian_mean",
        )
        save_gaussian_as_pdb_fn(
            gaussian_rigid=rec_pts,
            filename=f"{filename_prefix}__rec_pts.pdb",
            mask=node_mask,
            center_mode="gaussian_mean",
        )

    return debug
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

# 你工程里应该已有这个 dataclass；没有就用这个
@dataclass
class ParentParams:
    mu: torch.Tensor          # [B,K,3]
    R: torch.Tensor           # [B,K,3,3]
    s: torch.Tensor           # [B,K,3]
    mask_parent: torch.Tensor # [B,K]


def _ensure_right_handed(R: torch.Tensor) -> torch.Tensor:
    # R: [...,3,3]
    det = torch.det(R)
    flip = (det < 0).to(R.dtype)[..., None, None]
    # flip last column if det<0
    R_fix = torch.cat([R[..., :2], R[..., 2:3] * (1.0 - 2.0 * flip)], dim=-1)
    return R_fix


def _sigma_from_p95_p05(width: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    width = p95 - p05（覆盖 90% 区间）
    对正态分布：p95 - p05 = 2 * 1.64485 * sigma ≈ 3.2897 * sigma
    => sigma = width / 3.2897
    """
    return width / (3.289707253902945 + eps)


# ============================================================
# V1: mu=段中点, R=avg rot, s=p95-p05(robust sigma)
# ============================================================
@torch.no_grad()
def build_parents_from_segments_v1(
    x_ca: torch.Tensor,                # [B,N,3]  (nm)
    rotmats: torch.Tensor,             # [B,N,3,3] backbone frames
    node_mask: torch.Tensor,           # [B,N]
    a_idx: torch.Tensor,               # [B,N]
    mask_parent: torch.Tensor,         # [B,K]
    Kmax: int,
    s_floor: float = 1e-3,
    eps: float = 1e-6,
):
    """
    V1 思路：
      - mu_k 用“段中点 residue”的 CA（更像骨架点，不会被弯曲段均值拖到空洞里）
      - R_k 用 segment_avg_rotation 的平均 frame
      - s_k 用 local coords 的 p95-p05 转换成 robust sigma（避免离群点拉爆 var）
    """
    B, N, _ = x_ca.shape
    device, dtype = x_ca.device, x_ca.dtype
    K = Kmax

    # 1) Rk: 平均旋转（你工程已有）
    Rk = segment_avg_rotation(
        rotmats, node_mask, a_idx, Kmax=K, mask_parent=mask_parent, eps=eps
    )  # [B,K,3,3]
    Rk = _ensure_right_handed(Rk)

    mu = torch.zeros((B, K, 3), device=device, dtype=dtype)
    s  = torch.zeros((B, K, 3), device=device, dtype=dtype)

    # 2) per segment: midpoint mu & robust scale
    for b in range(B):
        n_valid = int(node_mask[b].sum().item())
        if n_valid <= 0:
            continue

        # 只遍历 active parents
        active_ks = torch.nonzero(mask_parent[b] > 0.5, as_tuple=False).reshape(-1)
        for kk in active_ks.tolist():
            # indices in this segment (and valid)
            idxs = torch.nonzero((a_idx[b] == kk) & (node_mask[b] > 0.5), as_tuple=False).reshape(-1)
            if idxs.numel() == 0:
                continue

            # midpoint index (按序列位置取中点)
            mid = idxs[idxs.numel() // 2].item()
            mu_k = x_ca[b, mid]  # [3]
            mu[b, kk] = mu_k

            # local coords: R^T (x - mu_k)
            Xk = x_ca[b, idxs]  # [L,3]
            delta = Xk - mu_k[None, :]
            # Rk[b,kk] is global rotation, local = R^T * delta
            local = (Rk[b, kk].transpose(-1, -2) @ delta.unsqueeze(-1)).squeeze(-1)  # [L,3]

            # robust width: p95 - p05 per axis
            # torch.quantile works on 1D along dim=0
            p05 = torch.quantile(local, 0.05, dim=0)
            p95 = torch.quantile(local, 0.95, dim=0)
            width = (p95 - p05).abs()

            sigma = _sigma_from_p95_p05(width, eps=eps).clamp_min(s_floor)
            s[b, kk] = sigma

    # mask unused
    mu = mu * mask_parent[..., None]
    s  = s  * mask_parent[..., None]

    # 对于没用的 parent，R 回退成 I，避免后续 gather 产生 nan
    I = torch.eye(3, device=device, dtype=dtype)[None, None]
    R = Rk * mask_parent[..., None, None] + I * (1.0 - mask_parent[..., None, None])

    return ParentParams(mu=mu, R=R, s=s, mask_parent=mask_parent)


# ============================================================
# V2: 段内 2-comp GMM (local) -> moment match -> eig 得到 (R,s)
# ============================================================
@torch.no_grad()
def build_parents_from_segments_v2(
        x_ca: torch.Tensor,
        node_mask: torch.Tensor,
        a_idx: torch.Tensor,
        mask_parent: torch.Tensor,
        Kmax: int,
        coverage_factor: float = 2.0,  # 2.0 ~ 95% 包裹
        eps: float = 1e-6,
) -> ParentParams:
    B, N, _ = x_ca.shape
    device, dtype = x_ca.device, x_ca.dtype
    K = Kmax

    x = x_ca * node_mask[..., None]
    oh = F.one_hot(a_idx, num_classes=K).to(dtype) * node_mask[..., None]
    cnt = oh.sum(dim=1).clamp_min(1.0)

    # 1. 计算位置重心 (mu)
    mu = torch.einsum("bnk,bnd->bkd", oh, x) / cnt[..., None]

    # 2. 【修正】计算对齐形状的旋转 (R)
    # 将 x 整理为 [B, K, N, 3] 以便批量计算 Cov
    # 这一步会消耗显存，如果 K*N 很大要注意优化
    x_expanded = x.unsqueeze(1).expand(B, K, N, 3)
    w_expanded = oh.permute(0, 2, 1)  # [B, K, N]

    Rk = get_shape_aligned_rotation(mu, x_expanded, w_expanded)

    # 3. 投影到局部坐标
    # local: [B,K,N,3]
    # 此时 local[..., 0] 是沿着主轴(X)的坐标
    delta = (x_expanded - mu.unsqueeze(2))
    local = torch.einsum("bkji,bknj->bkni", Rk, delta)

    # 4. 计算包围尺度 (s)
    # 既然 R 已经对齐了主轴，s[0] 就是长轴半径，s[1] 是中轴，s[2] 是短轴
    w = oh.permute(0, 2, 1)
    denom = w.sum(dim=2).clamp_min(1.0)

    # 用 2-Sigma 包裹
    var = ((local ** 2) * w.unsqueeze(-1)).sum(dim=2) / denom.unsqueeze(-1)
    s = torch.sqrt(var.clamp_min(eps)) * coverage_factor

    # 5. Mask
    mu = mu * mask_parent[..., None]
    s = s * mask_parent[..., None]
    # R 保持单位阵如果不激活
    I = torch.eye(3, device=device, dtype=dtype)[None, None]
    Rk = Rk * mask_parent[..., None, None] + I * (1.0 - mask_parent[..., None, None])

    return ParentParams(mu=mu, R=Rk, s=s, mask_parent=mask_parent)
@torch.no_grad()
def build_parents_from_segments_v1_envelope(
    rigids_nm,                 # OffsetGaussianRigid-like, provides get_gaussian_mean() and get_covariance()
    node_mask: torch.Tensor,   # [B,N] 0/1
    a_idx: torch.Tensor,       # [B,N] in [0..K-1]
    mask_parent: torch.Tensor, # [B,K]
    Kmax: int,
    k_sigma: float = 1.0,      # 画/包络用 1σ or 3σ: shape term uses (k_sigma^2 * Σ_i)
    eps: float = 1e-6,
    alpha_floor: float = 1.0,  # 防止反向缩小（通常设 1）
) -> ParentParams:
    """
    V1 (你上面 A+B 那套)：child-ellipsoid envelope (AE 验证用，不训练)
    - 子椭圆来自 rigids_nm 的 gaussian_mean / covariance
    - Step1: moment-match 得到 Σ_mm
    - Step2: 只膨胀不旋转：Σ_parent = α^2 Σ_mm，α^2 = max_i (t_i^2 + e_i)
      where:
        t_i^2 = (μ_i-μ_k)^T Σ_mm^{-1} (μ_i-μ_k)
        e_i   = λ_max( Σ_mm^{-1/2} (k^2 Σ_i) Σ_mm^{-1/2} )
    输出仍是 (mu, R, s)，其中 Σ_parent = R diag(s^2) R^T
    """
    device, dtype = node_mask.device, node_mask.dtype
    B, N = node_mask.shape
    K = Kmax

    # --- child ellipsoids ---
    mu_i = rigids_nm.get_trans() * node_mask[..., None]         # [B,N,3]
    Sig_i = rigids_nm.get_covariance()                                  # [B,N,3,3]
    Sig_i = _sym(Sig_i)

    # --- one-hot weights per parent ---
    oh = F.one_hot(a_idx.clamp(0, K-1), num_classes=K).to(mu_i.dtype)    # [B,N,K]
    oh = oh * node_mask[..., None]                                      # mask padding
    cnt = oh.sum(dim=1).clamp_min(1.0)                                  # [B,K]

    # --- parent mean of centers: μ_k ---
    mu_k = torch.einsum("bnk,bnd->bkd", oh, mu_i) / cnt[..., None]       # [B,K,3]

    # --- moment-matched Σ_mm = E[Σ_i] + Cov(μ_i) ---
    # E[Σ_i]
    E_Sig = torch.einsum("bnk,bnij->bkij", oh, Sig_i) / cnt[..., None, None]  # [B,K,3,3]

    # Cov(μ_i)
    delta = mu_i[:, None, :, :] - mu_k[:, :, None, :]                   # [B,K,N,3]
    # outer(delta)
    outer = delta.unsqueeze(-1) @ delta.unsqueeze(-2)                    # [B,K,N,3,3]
    w = oh.permute(0, 2, 1)                                              # [B,K,N]
    Cov_mu = (outer * w[..., None, None]).sum(dim=2) / cnt[..., None, None]  # [B,K,3,3]

    I = torch.eye(3, device=device, dtype=mu_i.dtype)[None, None]
    Sig_mm = _sym(E_Sig + Cov_mu) + eps * I                              # [B,K,3,3]

    # --- inflate alpha^2 per parent to (approximately) contain all child ellipsoids ---
    alpha2 = torch.ones((B, K), device=device, dtype=mu_i.dtype) * alpha_floor

    # Precompute per-residue parent membership mask for max
    # (w: [B,K,N] already)
    for k in range(K):
        mk = (mask_parent[:, k] > 0.5).to(mu_i.dtype)                    # [B]
        if mk.max().item() < 0.5:
            continue

        Sigk = Sig_mm[:, k, :, :]                                        # [B,3,3]
        # Cholesky for Σ^{-1} and Σ^{-1/2} operations
        Sigk = _sym(Sigk) + eps * torch.eye(3, device=device, dtype=Sigk.dtype)
        L = torch.linalg.cholesky(Sigk)                                  # [B,3,3]

        # t_i^2 = || L^{-1} (μ_i - μ_k) ||^2
        dk = (mu_i - mu_k[:, k, :].unsqueeze(1))                          # [B,N,3]
        y = torch.linalg.solve_triangular(L.unsqueeze(1), dk.unsqueeze(-1), upper=False).squeeze(-1)  # [B,N,3]
        t2 = (y * y).sum(dim=-1)                                          # [B,N]

        # e_i = λ_max( L^{-1} (k^2 Σ_i) L^{-T} )   (generalized eig upper bound)
        Sig_i_scaled = (k_sigma * k_sigma) * Sig_i                        # [B,N,3,3]
        X = torch.linalg.solve_triangular(L.unsqueeze(1), Sig_i_scaled, upper=False)  # [B,N,3,3] = L^{-1} Σ
        A = torch.linalg.solve_triangular(L.unsqueeze(1), X.transpose(-1, -2), upper=False).transpose(-1, -2)  # L^{-1} Σ L^{-T}
        A = _sym(A)
        e = torch.linalg.eigvalsh(A)[..., -1]                              # [B,N]

        val = t2 + e                                                      # [B,N]

        # mask residues not in this parent segment
        wk = w[:, k, :]                                                   # [B,N]
        val = torch.where(wk > 0.5, val, torch.tensor(-1e9, device=device, dtype=val.dtype))

        maxv = val.max(dim=1).values                                      # [B]
        # only apply for active batch entries (mk)
        alpha2[:, k] = torch.where(mk > 0.5, torch.maximum(alpha2[:, k], maxv), alpha2[:, k])

    # --- final parent covariance: Σ_parent = α^2 Σ_mm ---
    Sig_parent = Sig_mm * alpha2[..., None, None]
    Sig_parent = _sym(Sig_parent) + eps * I

    # mask unused parents (keep identity-ish so downstream不炸)
    Sig_parent = Sig_parent * mask_parent[..., None, None] + (1.0 - mask_parent[..., None, None]) * I

    # --- convert to (R, s) ---
    Rk, sk = _eig_R_s_from_cov(Sig_parent, eps=eps)                        # [B,K,3,3], [B,K,3]

    # mask mu/s
    mu_k = mu_k * mask_parent[..., None]
    sk = sk * mask_parent[..., None]

    return ParentParams(mu=mu_k, R=Rk, s=sk, mask_parent=mask_parent)

def _sym(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))
def _eig_R_s_from_cov(Sigma: torch.Tensor, eps: float = 1e-8):
    """
    Sigma: [B,K,3,3] SPD-ish
    return:
      R: [B,K,3,3] (cols are principal axes, descending)
      s: [B,K,3]   (std, descending)
    """
    Sigma = _sym(Sigma)
    # eigh returns ascending eigenvalues
    evals, evecs = torch.linalg.eigh(Sigma)  # evals [B,K,3], evecs [B,K,3,3]
    idx = torch.argsort(evals, dim=-1, descending=True)  # [B,K,3]
    # gather eigenvalues
    evals_sorted = torch.gather(evals, dim=-1, index=idx)
    # gather eigenvectors by columns
    R = torch.gather(
        evecs, dim=-1,
        index=idx.unsqueeze(-2).expand(*evecs.shape[:-2], 3, 3)
    )

    # enforce right-handed (det = +1)
    det = torch.det(R)  # [B,K]
    flip = (det < 0).to(R.dtype).view(*det.shape, 1, 1)
    R = torch.cat([R[..., :2], R[..., 2:3] * (1.0 - 2.0 * flip)], dim=-1)

    s = torch.sqrt(evals_sorted.clamp_min(eps))
    return R, s


def get_shape_aligned_rotation(mu, x, mask, eps=1e-6):
    """
    计算点云的协方差，并通过 PCA 得到 R。
    强制把长轴排在第一列 (X轴)。

    Args:
        mu:   [B, K, 3]
        x:    [B, K, N, 3] (已扩展)
        mask: [B, K, N]
    """
    # 1. 中心化
    # 【修正点】：使用 unsqueeze(2) 而不是 (1)
    # mu: [B, K, 3] -> [B, K, 1, 3]
    # x:  [B, K, N, 3]
    # 结果 delta: [B, K, N, 3]
    delta = (x - mu.unsqueeze(2)) #* mask.unsqueeze(-1)

    # 2. 计算协方差 Cov = E[XX^T]
    # mask.sum: [B, K] -> [B, K, 1, 1]
    denom = mask.sum(dim=-1).clamp_min(1.0).unsqueeze(-1).unsqueeze(-1)

    # einsum: 沿着 N (dim=2) 求和
    # [B, K, N, 3] x [B, K, N, 3] -> [B, K, 3, 3]
    cov = torch.einsum("bkni,bknj->bkij", delta, delta) / denom

    # 3. 特征分解
    # eigh 返回的 eigenvalues 是从小到大: λ0 <= λ1 <= λ2
    I = torch.eye(3, device=x.device)[None, None]
    evals, evecs = torch.linalg.eigh(cov + I * eps)

    # 4. 重排轴：强制 X 轴为主轴 (最长)
    # evecs: [B, K, 3, 3]
    v_short = evecs[..., 0]  # λ0 (最短)
    v_mid = evecs[..., 1]  # λ1
    v_long = evecs[..., 2]  # λ2 (最长)

    # 重新组装 R = [Long, Mid, Short]
    # 此时 R @ [1, 0, 0]^T = v_long
    R_new = torch.stack([v_long, v_mid, v_short], dim=-1)

    # 5. 确保右手系 (Det = 1)
    det = torch.det(R_new)
    flip = (det < 0).float().unsqueeze(-1).unsqueeze(-1)
    # 翻转最后一列 (Z轴)
    R_new[..., 2] = R_new[..., 2] * (1.0 - 2.0 * flip.squeeze(-1))

    return R_new




def build_parents_from_segments_v3(
        x_ca: torch.Tensor,  # [B, N, 3] 原子坐标
        node_mask: torch.Tensor,  # [B, N]    有效原子Mask
        a_idx: torch.Tensor,  # [B, N]    每个原子属于哪个Parent (0~K-1)
        mask_parent: torch.Tensor,  # [B, K]    有效Parent Mask
        Kmax: int,
        eps: float = 1e-6,
) -> ParentParams:
    """
    根据原子簇直接计算精确包裹的椭球 (Exact Fit Ellipsoid).
    逻辑完全复刻 NumPy 的 analyze_and_plot_atoms 流程。
    """
    B, N, _ = x_ca.shape
    device, dtype = x_ca.device, x_ca.dtype
    K = Kmax

    # ---------------------------------------------------------
    # 0. 数据准备: One-hot Mask
    # ---------------------------------------------------------
    # x: [B, N, 3]
    x = x_ca * node_mask[..., None]

    # oh: [B, N, K] -> [B, K, N] 用于广播
    # 表示第 k 个 Parent 包含了哪些原子
    oh = F.one_hot(a_idx.clamp(0, K - 1).long(), num_classes=K).to(dtype)
    oh = oh * node_mask[..., None]  # 只有有效原子的分配才算数
    w = oh.permute(0, 2, 1)  # [B, K, N]

    # 每个 parent 有多少个点
    cnt = w.sum(dim=2).clamp_min(1.0)  # [B, K]

    # ---------------------------------------------------------
    # 1. 计算中心 (Mean)
    # ---------------------------------------------------------
    # mu: [B, K, 3]
    mu = torch.einsum("bkn,bnj->bkj", w, x) / cnt[..., None]

    # ---------------------------------------------------------
    # 2. 计算协方差矩阵 (Covariance)
    # ---------------------------------------------------------
    # 对应 NumPy: cov = np.cov(points, rowvar=False)
    # 公式: Cov = E[(x-mu)(x-mu)^T]

    # delta: [B, K, N, 3]
    delta = x.unsqueeze(1) - mu.unsqueeze(2)

    # Mask 掉不属于该 Parent 的点 (置为 0，防止干扰求和)
    delta = delta * w.unsqueeze(-1)

    # 计算未归一化的散度矩阵 (Scatter Matrix)
    # einsum: [B,K,N,3] x [B,K,N,3] -> [B,K,3,3]
    # 这里不需要额外乘权重 w，因为 delta 已经被 mask 过了
    scatter = torch.einsum("bkni,bknj->bkij", delta, delta)

    # 归一化得到协方差
    # 注意：np.cov 默认是用 (N-1)，为了完全对齐，我们用 cnt - 1
    # 但如果点数极少(1个)，分母为0，所以 clamp_min(1.0)
    denom = (cnt - 1.0).clamp_min(1.0)
    cov = scatter / denom[..., None, None]

    # ---------------------------------------------------------
    # 3. 特征分解 (Eigen Decomposition) => 得到 R 和 基础特征值
    # ---------------------------------------------------------
    # 加一点微小噪声防止全0矩阵导致 NaN
    I = torch.eye(3, device=device, dtype=dtype)[None, None]
    cov_safe = cov + I * eps

    # eigh 返回特征值升序排列: val[0] <= val[1] <= val[2]
    # evals: [B, K, 3]
    # evecs: [B, K, 3, 3] -> 列向量是特征向量
    eigvals, eigvecs = torch.linalg.eigh(cov_safe)

    # 旋转矩阵 R 直接就是特征向量矩阵 (列向量为轴)
    # R: [B, K, 3, 3]
    # Col 0: 短轴方向, Col 1: 中轴方向, Col 2: 长轴方向
    Rk = eigvecs

    # 确保右手系 (Determinant = +1)
    # 如果 det < 0，翻转第一列 (X轴/短轴)
    det = torch.det(Rk)  # [B, K]
    flip = (det < 0).float().unsqueeze(-1).unsqueeze(-1)
    # Rk[..., 0] *= (1 - 2*flip)
    # 这种写法在 inplace 上可能有问题，用 clone 或者是 cat
    col0 = Rk[..., 0] * (1.0 - 2.0 * flip.squeeze(-1))
    Rk = torch.stack([col0, Rk[..., 1], Rk[..., 2]], dim=-1)

    # ---------------------------------------------------------
    # 4. 计算包裹系数 (Exact Fit Scale) - 关键步骤
    # ---------------------------------------------------------
    # 我们的目标是找到一个缩放系数 S，使得所有点都在椭球内
    # 标准椭球方程: sum( (x_local / sqrt(lambda))^2 ) <= S^2

    # 4.1 投影到局部主轴坐标系 (Project to PCA space)
    # delta: [B, K, N, 3] (已经中心化并Mask过)
    # Rk:    [B, K, 3, 3]
    # local: [B, K, N, 3]
    # local = (x - mu) @ R
    local_coords = torch.einsum("bkni,bkij->bknj", delta, Rk)

    # 4.2 计算归一化距离 (Mahalanobis Distance squared)
    # eigvals: [B, K, 3] -> 广播到 [B, K, 1, 3]
    # 注意防除零 eps
    std_devs_sq = eigvals.clamp_min(eps).unsqueeze(2)

    # normalized_dist_sq: [B, K, N]
    # 也就是代码里的: points_pca**2 / eigvals
    norm_dist_sq = (local_coords ** 2 / std_devs_sq).sum(dim=-1)

    # 4.3 找到每个 Cluster 内部最远的点 (Max Distance)
    # 需要再次应用 Mask，因为非该 Cluster 的点在 delta 归零后位于中心，距离为0，
    # 但为了保险起见，我们只在 mask 范围内取 max
    # 设无效点的距离为 -1 (反正距离>=0)
    valid_dist = torch.where(w > 0.5, norm_dist_sq, torch.tensor(-1.0, device=device))

    # max_sq_val: [B, K] -> 最大的马氏距离平方
    max_sq_val = valid_dist.max(dim=2).values.clamp_min(eps)

    # scale_factor: [B, K] -> S
    scale_factor = torch.sqrt(max_sq_val)

    # ---------------------------------------------------------
    # 5. 计算最终轴长 (s)
    # ---------------------------------------------------------
    # 原始标准差 sqrt(eigvals) * 缩放系数 scale_factor
    # s: [B, K, 3]
    # 顺序：s[0]对应短轴, s[2]对应长轴 (与 R 的列对应)
    s = torch.sqrt(eigvals.clamp_min(eps)) * scale_factor.unsqueeze(-1)

    # ---------------------------------------------------------
    # 6. Mask 和 输出清理
    # ---------------------------------------------------------
    # 对于无效的 Parent (mask_parent=0)，将 R 设为单位阵，s 设为 0，mu 设为 0
    valid_mask = mask_parent[..., None, None]  # [B, K, 1, 1]

    mu = mu * mask_parent[..., None]
    s = s * mask_parent[..., None]
    Rk = Rk * valid_mask + I * (1.0 - valid_mask)  # 保持无效位为 Identity

    return ParentParams(mu=mu, R=Rk, s=s, mask_parent=mask_parent)









# =========================================================================
# 测试入口：构造你那4个点的数据并运行
# =========================================================================
