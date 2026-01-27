
"""
Cleaned & fixed version of flow_model_HGF.py

Key fixes vs original:
  - Fix undefined s_last/r_last/mask_last before final_up (take from levels_up[-1]).
  - Feed residue-level outputs (s_res, r_res) into SequenceHead and SidechainAtomHead.
  - Make SidechainAtomHead respect (dynamic) thickness via a sigma floor (stabilizes early training).
  - Robust handling when sidechain_features are missing (set to zeros instead of crashing).
  - Clearer structure: feature extraction -> gaussian init -> trunk -> HGF (down/bottleneck/up/final) -> heads.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from data import utils as du
from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models.features.backbone_gnn_feature import BackboneEncoderGNN

from models.shattetnion.ShDecoderSidechain import SideAtomsFeatureHead, SequenceHead
from openfold.model.primitives import Linear

from data.GaussianRigid import OffsetGaussianRigid

from models.components.frozen_esm import FrozenEsmModel
from models.components.sequence_adapters import SequenceToTrunkNetwork

from models.IGA import InvariantGaussianAttention, GaussianUpdateBlock, BottleneckIGAModule
from models.downblock import HierarchicalDownsampleIGAModule
from models.upsample_block import HierarchicalUpsampleIGAModule
from models.finnalup import FinalCoarseToFineIGAModule


# -----------------------------------------------------------------------------
# Output head: Differentiable sidechain atom decoder (nm internal)
# -----------------------------------------------------------------------------
class SidechainAtomHead(nn.Module):
    """
    Predict sidechain atom coordinates in residue-local frame, then apply GaussianRigid to global.
    Internal units: nanometers (nm).

    Decoder:
        u = f(s) in R^{A x 3}
        x_local = mu + u ⊙ sigma

    Stabilization:
        sigma_eff = max(sigma, thickness_nm)  (per-residue floor)
    """

    def __init__(self, c_in: int, num_atoms: int = 10, base_thickness_ang: float = 0.5):
        super().__init__()
        self.num_atoms = int(num_atoms)
        self.base_thickness_ang = float(base_thickness_ang)
        self.ang_to_nm = 0.1

        self.projection = nn.Sequential(
            Linear(c_in, c_in),
            nn.LayerNorm(c_in),
            nn.SiLU(),
            Linear(c_in, self.num_atoms * 3),
        )
        self._init_weights()

    def _init_weights(self):
        # Start with very small u so early training doesn't explode.
        nn.init.normal_(self.projection[-1].weight, std=1e-4)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(
        self,
        s: torch.Tensor,                         # [B, N, C]
        gaussian_rigid: OffsetGaussianRigid,      # [B, N] (nm)
        sidechain_mask: torch.Tensor,             # [B, N, A]
        thickness_nm: torch.Tensor | None = None  # [B, N, 1] or [B, N, 3]
    ):
        B, N, C = s.shape
        assert sidechain_mask.shape[:2] == (B, N), "sidechain_mask batch/length mismatch"
        assert sidechain_mask.shape[2] == self.num_atoms, "sidechain_mask last dim != num_atoms"

        mu_local = gaussian_rigid._local_mean          # [B, N, 3] (nm)
        sigma_local = torch.exp(gaussian_rigid._scaling_log)  # [B, N, 3] (nm)

        # Thickness floor (nm)
        if thickness_nm is None:
            thickness_nm = torch.full((B, N, 1), self.base_thickness_ang * self.ang_to_nm, device=s.device, dtype=s.dtype)
        if thickness_nm.shape[-1] == 1:
            thickness_nm = thickness_nm.expand(B, N, 3)
        sigma_eff = torch.maximum(sigma_local, thickness_nm)  # [B, N, 3]

        mu_exp = mu_local.unsqueeze(-2)        # [B, N, 1, 3]
        sigma_exp = sigma_eff.unsqueeze(-2)    # [B, N, 1, 3]

        u_raw = self.projection(s).view(B, N, self.num_atoms, 3)
        u = torch.tanh(u_raw) * (1.0 + 0.2 * torch.abs(u_raw))

        local_pred = mu_exp + u * sigma_exp  # [B, N, A, 3]
        local_pred = local_pred * sidechain_mask.unsqueeze(-1)

        rigid_expanded = gaussian_rigid.unsqueeze(-1)  # [B, N, 1]
        global_pred = rigid_expanded.apply(local_pred)  # [B, N, A, 3] (nm)

        return global_pred, local_pred, gaussian_rigid


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------
class HierarchicalGaussianFieldModel(nn.Module):
    """
    Cleaned version of the original HierarchicalGaussianFieldModel.

    High-level:
      Feature extraction -> GaussianRigid init -> IGA trunk ->
      HGF down/bottleneck/up/final -> heads (sequence + atoms)
    """

    def __init__(self, model_conf):
        super().__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa

        # Feature extractors
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)
        self.feature_graph = BackboneEncoderGNN(dim_nodes=self._ipa_conf.c_s)

        sidechain_conf = getattr(model_conf, "sidechain_atoms", {})
        self.sidechain_hidden = int(sidechain_conf.get("hidden", 256))
        self.sidechain_head = SideAtomsFeatureHead(
            A=sidechain_conf.get("A", 10),
            hidden=self.sidechain_hidden,
            num_classes=0,
            dropout=sidechain_conf.get("dropout", 0.1),
            conv_blocks=sidechain_conf.get("conv_blocks", 4),
            mlp_blocks=sidechain_conf.get("mlp_blocks", 4),
            fuse_blocks=sidechain_conf.get("fuse_blocks", 4),
            conv_groups=sidechain_conf.get("conv_groups", 1),
        )

        # Optional ESM
        self.use_esm = bool(getattr(model_conf, "use_esm", False))
        if self.use_esm:
            esm_model_name = getattr(model_conf, "esm_model", "esm2_650M")
            self.seq_encoder = FrozenEsmModel(model_key=esm_model_name, use_esm_attn_map=True)
            self.sequence_to_trunk = SequenceToTrunkNetwork(
                esm_single_dim=self.seq_encoder.single_dim,
                num_layers=self.seq_encoder.num_layers,
                d_single=self._ipa_conf.c_s,
                esm_attn_dim=self.seq_encoder.attn_head * self.seq_encoder.num_layers,
                d_pair=self._model_conf.edge_embed_size,
                position_bins=32,
                pairwise_state_dim=self._model_conf.edge_embed_size,
            )
        else:
            self.seq_encoder = None
            self.sequence_to_trunk = None

        # Fusion blocks
        # node concat: node_feature + sidechain_feat + backbone_gnn (+ esm_single)
        node_in = self._ipa_conf.c_s + self.sidechain_hidden + self._ipa_conf.c_s
        if self.use_esm:
            node_in += self._ipa_conf.c_s
        self.feature_fusion = nn.Sequential(
            nn.Linear(node_in, self._ipa_conf.c_s),
            nn.LayerNorm(self._ipa_conf.c_s),
            nn.SiLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )

        # edge concat: edge_feature + gnn_pair (+ esm_pair)
        edge_in = self._ipa_conf.c_z + self._ipa_conf.c_z
        if self.use_esm:
            edge_in += self._model_conf.edge_embed_size
        self.edge_feature_fusion = nn.Sequential(
            nn.Linear(edge_in, self._ipa_conf.c_z),
            nn.LayerNorm(self._ipa_conf.c_z),
            nn.SiLU(),
            nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z),
        )

        # Norms
        self.node_feature_ln = nn.LayerNorm(self._ipa_conf.c_s)
        self.sidechain_feature_ln = nn.LayerNorm(self.sidechain_hidden)
        self.graph_feature_ln = nn.LayerNorm(self._ipa_conf.c_s)
        self.edge_init_ln = nn.LayerNorm(self._model_conf.edge_embed_size)
        self.edge_graph_ln = nn.LayerNorm(self._ipa_conf.c_z)
        if self.use_esm:
            self.esm_feature_ln = nn.LayerNorm(self._ipa_conf.c_s)
            self.edge_esm_ln = nn.LayerNorm(self._model_conf.edge_embed_size)

        # IGA trunk blocks
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f"iga_{b}"] = InvariantGaussianAttention(
                c_s=self._ipa_conf.c_s,
                c_z=self._ipa_conf.c_z,
                c_hidden=self._ipa_conf.c_hidden,
                no_heads=self._ipa_conf.no_heads,
                no_qk_gaussians=self._ipa_conf.no_qk_points,
                no_v_points=self._ipa_conf.no_v_points,
                layer_idx=b,
                enable_vis=False,
            )
            self.trunk[f"iga_ln_{b}"] = nn.LayerNorm(self._ipa_conf.c_s)

            tfmr_layer = nn.TransformerEncoderLayer(
                d_model=self._ipa_conf.c_s,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=self._ipa_conf.c_s,
                batch_first=True,
                dropout=0.0,
                norm_first=False,
            )
            self.trunk[f"seq_tfmr_{b}"] = nn.TransformerEncoder(tfmr_layer, self._ipa_conf.seq_tfmr_num_layers)
            self.trunk[f"post_tfmr_{b}"] = Linear(self._ipa_conf.c_s, self._ipa_conf.c_s, init="final")

            # keep same transition utilities you used previously via ipa_pytorch
            from models import ipa_pytorch
            self.trunk[f"node_transition_{b}"] = ipa_pytorch.StructureModuleTransition(c=self._ipa_conf.c_s)
            if b < self._ipa_conf.num_blocks - 1:
                self.trunk[f"edge_transition_{b}"] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=self._model_conf.edge_embed_size,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

            self.trunk[f"gau_update_{b}"] = GaussianUpdateBlock(self._ipa_conf.c_s)

        # HGF hierarchy
        self.downsampler = HierarchicalDownsampleIGAModule(
            c_s=self._ipa_conf.c_s,
            iga_conf=self._ipa_conf,
            OffsetGaussianRigid_cls=OffsetGaussianRigid,
            num_downsample=2,
            ratio=6.0,
            k_max_cap=64,
            coarse_iga_layers=4,
        )
        self.bottleneck = BottleneckIGAModule(
            c_s=self._ipa_conf.c_s,
            iga_conf=self._ipa_conf,
            bottleneck_layers=getattr(model_conf, "bottleneck_layers", 6),
            layer_idx_base=3000,
            enable_vis=False,
        )
        self.upsampler = HierarchicalUpsampleIGAModule(
            c_s=self._ipa_conf.c_s,
            iga_conf=self._ipa_conf,
            OffsetGaussianRigid_cls=OffsetGaussianRigid,
            num_upsample=2,
            M_max=8,
            K_target=None,
            up_ratio=2.0,
            neighbor_R=2,
            coarse_iga_layers=4,
        )
        self.final_up = FinalCoarseToFineIGAModule(
            c_s=self._ipa_conf.c_s,
            iga_conf=self._ipa_conf,
            OffsetGaussianRigid_cls=OffsetGaussianRigid,
            num_refine_layers=4,
            neighbor_R=2,
            w_attach=1.0,
            w_entB=0.0,
            enable_occ_loss=False,
        )

        # Heads
        self.base_thickness_ang = float(getattr(model_conf, "base_thickness", 0.5))
        self.atom_head = SidechainAtomHead(
            c_in=self._ipa_conf.c_s,
            num_atoms=10,
            base_thickness_ang=self.base_thickness_ang,
        )
        self.logits_head = SequenceHead(self._ipa_conf.c_s, self._ipa_conf.c_s, num_classes=21)

    # ---------------------------
    # helpers
    # ---------------------------
    def _make_noise_t(self, input_feats, node_mask):
        if "r3_t" in input_feats:
            return input_feats["r3_t"]
        return torch.ones((node_mask.shape[0], 1), device=node_mask.device, dtype=node_mask.dtype)

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

        if sidechain_features is None:
            # FIX: if missing, inject zeros so fusion shape is stable.
            sidechain_features = torch.zeros((init_node_embed.shape[0], init_node_embed.shape[1], self.sidechain_hidden),
                                            device=init_node_embed.device, dtype=init_node_embed.dtype)
        sidechain_features_n = self.sidechain_feature_ln(sidechain_features)

        parts = [init_node_embed_n, sidechain_features_n, node_h_n]
        if self.use_esm and seq_emb_s is not None:
            parts.append(self.esm_feature_ln(seq_emb_s))
        node_embed = self.feature_fusion(torch.cat(parts, dim=-1)) * node_mask[..., None]

        # edge feature net
        trans_t = input_feats["trans_1"]
        init_edge_embed = self.edge_feature_net(node_embed, trans_t, edge_mask, diffuse_mask)
        init_edge_embed_n = self.edge_init_ln(init_edge_embed)
        edge_h_n = self.edge_graph_ln(edge_h)

        if self.use_esm and seq_emb_z is not None:
            seq_emb_z_n = self.edge_esm_ln(seq_emb_z)
            edge_embed = self.edge_feature_fusion(torch.cat([init_edge_embed_n, edge_h_n, seq_emb_z_n], dim=-1))
        else:
            edge_embed = self.edge_feature_fusion(torch.cat([init_edge_embed_n, edge_h_n], dim=-1))
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

        atoms14_local = input_feats["atoms14_local_t"]
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

    # ---------------------------
    # forward
    # ---------------------------
    def forward(self, input_feats, step: int, total_steps: int, sideonly: bool = False):
        node_mask = input_feats["res_mask"]                          # [B,N]
        edge_mask = node_mask[:, None] * node_mask[:, :, None]       # [B,N,N]
        diffuse_mask = input_feats["diffuse_mask"]
        res_index = input_feats["res_idx"]
        chain_idx = input_feats["chain_idx"]

        # Feature extraction
        node_embed, edge_embed, sidechain_atom_mask = self._extract_features(
            input_feats=input_feats,
            node_mask=node_mask,
            edge_mask=edge_mask,
            diffuse_mask=diffuse_mask,
            res_index=res_index,
            chain_idx=chain_idx,
        )

        # Base backbone rigid (Angstrom), then GaussianRigid init (nm internal)
        base_rigid = du.create_rigid(input_feats["rotmats_1"], input_feats["trans_1"])
        curr_rigids, thickness_nm = self._init_gaussian_rigids_nm(input_feats, base_rigid)

        # IGA trunk
        for b in range(self._ipa_conf.num_blocks):
            iga_out = self.trunk[f"iga_{b}"](s=node_embed, z=edge_embed, r=curr_rigids, mask=node_mask)
            iga_out = iga_out * node_mask[..., None]
            node_embed = self.trunk[f"iga_ln_{b}"](node_embed + iga_out)

            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool)
            )
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)

            node_embed = self.trunk[f"node_transition_{b}"](node_embed) * node_mask[..., None]

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed) * edge_mask[..., None]

            curr_rigids = self.trunk[f"gau_update_{b}"](
                node_embed, curr_rigids, mask=input_feats["update_mask"]
            )

        # HGF hierarchy
        levels_down, pool_reg = self.downsampler(
            s_f=node_embed,
            r_f=curr_rigids,
            mask_f=node_mask,
            step=step,
            total_steps=total_steps,
        )
        coarsest = levels_down[-1]
        sL, rL, mL = coarsest["s"], coarsest["r"], coarsest["mask"]
        sL, rL = self.bottleneck(sL, rL, mL)

        levels_up, up_reg = self.upsampler(
            s_l=sL,
            r_l=rL,
            mask_l=mL,
            step=step,
            total_steps=total_steps,
        )

        # FIX: use the last upsampled level as parent inputs
        last_up = levels_up[-1]
        s_last, r_last, mask_last = last_up["s"], last_up["r"], last_up["mask"]

        final_levels, final_reg = self.final_up(
            s_parent=s_last,
            r_parent=r_last,
            mask_parent=mask_last,
            node_mask=node_mask,
            res_idx=res_index,
        )
        s_res = final_levels[-1]["s"]
        r_res = final_levels[-1]["r"]

        # Heads MUST use residue-level outputs from HGF
        logits = self.logits_head(s_res)

        pred_global_nm, pred_local_nm, gaussian_rigid_nm = self.atom_head(
            s_res, r_res, sidechain_atom_mask, thickness_nm=thickness_nm
        )

        # Convert back to Angstrom for losses/visualization
        pred_local_ang = pred_local_nm * du.NM_TO_ANG_SCALE
        pred_global_ang = pred_global_nm * du.NM_TO_ANG_SCALE
        final_gaussian_ang = gaussian_rigid_nm.scale_translation(du.NM_TO_ANG_SCALE)

        reg_total = pool_reg + up_reg + final_reg

        return {
            "pred_atoms": pred_local_ang,           # [B,N,A,3] local (Å)
            "pred_atoms_global": pred_global_ang,   # [B,N,A,3] global (Å)
            "logits": logits,                       # [B,N,21]
            "final_gaussian": final_gaussian_ang,   # OffsetGaussianRigid (Å)
            "reg_total": reg_total,
            "reg_parts": {"pool_reg": pool_reg, "up_reg": up_reg, "final_reg": final_reg},
        }
