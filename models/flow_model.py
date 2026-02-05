
import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models import ipa_pytorch
from data import utils as du
from data.GaussianRigid import OffsetGaussianRigid
from models.IGA import InvariantGaussianAttention, GaussianUpdateBlock
from openfold.utils import rigid_utils as ru


class AminoAcidEllipsoidIGAHead(nn.Module):
    def __init__(self, ipa_conf, edge_embed_size, num_classes=21, num_blocks=1):
        super(AminoAcidEllipsoidIGAHead, self).__init__()
        self.ipa = ipa_conf
        self.num_blocks = num_blocks

        self.trunk = nn.ModuleDict()
        for b in range(num_blocks):
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

            tfmr_in = self.ipa.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self.ipa.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f"seq_tfmr_{b}"] = torch.nn.TransformerEncoder(
                tfmr_layer, self.ipa.seq_tfmr_num_layers)
            self.trunk[f"post_tfmr_{b}"] = ipa_pytorch.Linear(
                tfmr_in, self.ipa.c_s, init="final")
            self.trunk[f"node_transition_{b}"] = ipa_pytorch.StructureModuleTransition(
                c=self.ipa.c_s)
            if b < num_blocks - 1:
                self.trunk[f"edge_transition_{b}"] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self.ipa.c_s,
                    edge_embed_in=edge_embed_size,
                    edge_embed_out=edge_embed_size,
                )

        self.aa_logit = ipa_pytorch.Linear(self.ipa.c_s, num_classes)
        self.alpha_head = ipa_pytorch.Linear(self.ipa.c_s, 3)

    def forward(self, node_embed, edge_embed, rigids_nm, node_mask, edge_mask):
        for b in range(self.num_blocks):
            iga_out = self.trunk[f"iga_{b}"](
                s=node_embed, z=edge_embed, r=rigids_nm, mask=node_mask)
            iga_out = iga_out * node_mask[..., None]
            node_embed = self.trunk[f"iga_ln_{b}"](node_embed + iga_out)

            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)

            node_embed = self.trunk[f"node_transition_{b}"](node_embed) * node_mask[..., None]
            if b < self.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed) * edge_mask[..., None]

        aa_logits = self.aa_logit(node_embed)
        alpha = self.alpha_head(node_embed)

        return aa_logits, alpha


class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

    def forward(self, input_feats):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        res_index = input_feats['res_idx']
        so3_t = input_feats['so3_t']
        r3_t = input_feats['r3_t']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']

        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            so3_t,
            r3_t,
            node_mask,
            diffuse_mask,
            res_index
        )
        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']
        init_edge_embed = self.edge_feature_net(
            init_node_embed,
            trans_t,
            trans_sc,
            edge_mask,
            diffuse_mask,
        )

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, (node_mask * diffuse_mask)[..., None])
            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
        }


class FlowModelIGA(nn.Module):
    def __init__(self, model_conf):
        super(FlowModelIGA, self).__init__()
        self._model_conf = model_conf
        self.ipa = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

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

            tfmr_in = self.ipa.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self.ipa.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            )
            self.trunk[f"seq_tfmr_{b}"] = torch.nn.TransformerEncoder(
                tfmr_layer, self.ipa.seq_tfmr_num_layers)
            self.trunk[f"post_tfmr_{b}"] = ipa_pytorch.Linear(
                tfmr_in, self.ipa.c_s, init="final")
            self.trunk[f"node_transition_{b}"] = ipa_pytorch.StructureModuleTransition(
                c=self.ipa.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self.ipa.c_s, use_rot_updates=True)
            if b < self.ipa.num_blocks - 1:
                self.trunk[f"edge_transition_{b}"] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self.ipa.c_s,
                    edge_embed_in=self._model_conf.edge_embed_size,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

        # self.aa_ellipsoid_head = AminoAcidEllipsoidIGAHead(
        #     self.ipa,
        #     self._model_conf.edge_embed_size,
        #     num_classes=getattr(model_conf, "num_aa_classes", 21),
        #     num_blocks=getattr(model_conf, "aa_head_blocks", 1),
        # )

    def _make_rigids_nm(self, input_feats):
        if "rigids_t" in input_feats:
            rigids_ang = input_feats["rigids_t"]
        else:
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
        return rigids_ang.scale_translation(du.ANG_TO_NM_SCALE)

    def forward(self, input_feats):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        res_index = input_feats['res_idx']
        so3_t = input_feats['so3_t']
        r3_t = input_feats['r3_t']
        trans_t = input_feats['trans_t']

        init_node_embed = self.node_feature_net(
            so3_t,
            r3_t,
            node_mask,
            diffuse_mask,
            res_index
        )
        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']
        init_edge_embed = self.edge_feature_net(
            init_node_embed,
            trans_t,
            trans_sc,
            edge_mask,
            diffuse_mask,
        )

        rigids_nm = self._make_rigids_nm(input_feats)

        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self.ipa.num_blocks):
            iga_out = self.trunk[f"iga_{b}"](
                s=node_embed, z=edge_embed, r=rigids_nm, mask=node_mask)
            iga_out = iga_out * node_mask[..., None]
            node_embed = self.trunk[f"iga_ln_{b}"](node_embed + iga_out)

            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)

            node_embed = self.trunk[f"node_transition_{b}"](node_embed) * node_mask[..., None]
            if b < self.ipa.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed) * edge_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](node_embed * node_mask[..., None])
            # IPA-style backbone update, but applied to OffsetGaussianRigid so IGA can
            # keep using the (noisy) ellipsoid parameters in attention.
            rigids_nm = rigids_nm.compose_update_12D(
                rigid_update, update_mask=(node_mask * diffuse_mask)[..., None]
            )

        # aa_logits, ellipsoid_alpha = self.aa_ellipsoid_head(
        #     node_embed, edge_embed, rigids_nm, node_mask, edge_mask
        # )

        rigids_ang = rigids_nm.scale_translation(du.NM_TO_ANG_SCALE)
        pred_trans = rigids_ang.get_trans()
        pred_rotmats = rigids_ang.get_rots().get_rot_mats()
        return {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
            # 'aa_logits': aa_logits,
            # 'ellipsoid_alpha': ellipsoid_alpha,
            # 'ellipsoid_scaling': rigids_ang.scaling,
        }
