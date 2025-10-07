
import torch
from torch import nn
import torch.nn.functional as F
from data import all_atom
from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet,EdgeFeatureNet_backuo
from models import ipa_pytorch,so3_theta,rope3D
from models import GA_block
from openfold.utils import rigid_utils as ru
from data import utils as du
from models.resnet import Conv2DFeatureExtractor
from models.basic_vae import Encoder, Decoder
from models.hours._hourglass import HourglassProteinCompressionTransformer
from models import utils as mu
from models.features.backbone_gnn_feature import BackboneEncoderGNN
from openfold.np.residue_constants import restype_name_to_atom14_names
from models.shattetnion.shframeawareattention import SHTransformer
from models.shattetnion.ShDecoderSidechain import NodeFeatExtractorWithHeads,SHSidechainDecoder,DynamicKSidechainDecoder,SHTypeHybridHead,SHGeoResHead,assemble_atom14
from models.shattetnion.SHTemplateRefiner import SHTemplateRefiner
from models.shattetnion.SHTemplateRefinerHard import SHTemplateRefinerHard,build_residue_meta_dict_from_rc
from models.shattetnion.Frameaware import FrameAwareTransformerRot
from models.shattetnion.ShDecoderSidechain import  Feat2Atom11,SideAtomsFeatureHead

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
            #self.trunk[f'ga_{b}'] = GA_block.UnifiedTransformerBlock(d_model=self._ipa_conf.c_s,v_heads=self._ipa_conf.no_heads,use_geom_attn=True)

            # self.trunk[f'ipa_{b}'] =rope3D.InvariantPointAttention_3DROPE(self._ipa_conf)
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
        chain_idx=input_feats['chain_idx']
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

        # Concatenate the features along the channel axis (axis=1)

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
                node_mask, #chain_idx
            )
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


class VAE(nn.Module):

    def __init__(self, model_conf):
        super(VAE, self).__init__()
        self.Encoder=Encoder(model_conf)
        self.Decoder=Decoder(model_conf)
        self.PCT=HourglassProteinCompressionTransformer(**model_conf.Hourglass)

    def forward(self, input_feats,infer_only=False):
        if infer_only:
            return self.encode(x, mask, verbose, infer_only)

        else:

            x = self.Encoder(input_feats)

            # encode and obtain post-quantize embedding
            z_q, downsampled_mask, log_dict = self.PCT.encode(x, mask=input_feats['res_mask'], verbose=False, infer_only=False)

            # decode back to original
            x_ = self.PCT.decode(z_q, downsampled_mask, verbose=False)
            padded_feats,M=mu.pad_input_feats_like(x_,input_feats)

            x_recons=self.Decoder(x_,padded_feats,M)
            # calculate losses
            # recons_loss = masked_mse_loss(x_recons, x, mask)
            vq_loss = log_dict.get("vq_loss", 0.0)
            # loss = vq_loss + recons_loss
            # log_dict["recons_loss"] = recons_loss.item()
            # log_dict["loss"] = loss.item()

            return x_recons, vq_loss, log_dict, z_q, downsampled_mask


class SHDecoder(nn.Module):
    def __init__(self,model_conf):
        super(SHDecoder, self).__init__()
        '''
        forward for binder,CSB is True
        forward_fixed topo for base model, which use node update ,rcsb_cluster30_GNNfromIPAscratch_node_mixed_again
        forward_ss for motif, neigh update, rcsb_ipagnnneigh_cluster_motif_updateall_reeidx_homo_heto

        '''
        self.model_conf=model_conf
        SH_conf=self.model_conf.sh

        # Hyperparameters
        # self.k_neighbors = k_neighbors

        #self.SHTransformer = SHTransformer(C, L_max, R_bins, L_edge, n_layers, n_radial, dropout, hidden_scalar)

        from openfold.np import residue_constants as rc
        from openfold.np.residue_constants import rigid_group_atom_positions
        residue_meta_dict = build_residue_meta_dict_from_rc(rc, allow_backbone_torsion=True)

        self.SHTemplateRefiner = SHTemplateRefiner(**SH_conf,rigid_group_atom_positions=rigid_group_atom_positions,
                                                       residue_meta_dict=residue_meta_dict)
        # self.SHTypePredictor = SHTypeHybridHead(**SH_conf)
        base = SHSidechainDecoder(**SH_conf)
        self.dyn = DynamicKSidechainDecoder(base, include_backbone=True)


    def forward(self,noisy_batch, SH, node_mask,aatype,Rm, t,state='SHdeocder'
                ):

        """ Graph-conditioned sequence/str model for binder
            graph will be calculated evary encoder layers
        """
        if state=='only_type':

            # out=self.dyn(SH, Rm, t, aatype=aatype,node_mask=node_mask)
            #
            # coords = out['coords_global']  # [B,N,4,K,3]
            # mask = out['peaks_mask']  # [B,N,4,K]
            # score = out['scores']  # [B,N,4,K]
            # atom14_xyz, atom14_exists = assemble_atom14_with_CA(
            #     coords_global=coords,
            #     peaks_mask=mask,
            #     scores=score,  # 也可以用 out['peak_probs']
            #     aatype_probs= F.one_hot(aatype,20).float(),  # 若是 hard label: F.one_hot(aatype,20).float()
            #     restype_name_to_atom14_names=restype_name_to_atom14_names,
            #     tpos=t,
            # )
            # out.update({'atom14_xyz':atom14_xyz, 'atom14_exists':atom14_exists})
            out = self.SHTypePredictor(SH, node_mask)
            return out
        elif state=='SHdeocder':
            out=self.dyn(noisy_batch,SH, Rm, t, aatype=aatype,node_mask=node_mask)

            coords = out['coords_global']  # [B,N,4,K,3]
            mask = out['peaks_mask']  # [B,N,4,K]
            score = out['scores']  # [B,N,4,K]
            atom14_xyz, atom14_exists = assemble_atom14_with_CA(
                coords_global=coords,
                peaks_mask=mask,
                scores=score,  # 也可以用 out['peak_probs']
                aatype_probs= F.one_hot(aatype,20).float(),  # 若是 hard label: F.one_hot(aatype,20).float()
                restype_name_to_atom14_names=restype_name_to_atom14_names,
                tpos=t,
            )
            out.update({'atom14_xyz':atom14_xyz, 'atom14_exists':atom14_exists})
            out=self.SHTemplateRefiner(SH,Rm, t, F.one_hot(aatype,20).float(), node_mask=node_mask)



            return out
        elif state=='SHTemplateRefiner':

            out=self.SHTemplateRefiner(SH,Rm, t, F.one_hot(aatype,20).float(), node_mask=node_mask)



            return out
        else:
            B,N,D = node_h.shape
            t=R.get_trans()
            d = torch.cdist(t, t)  # [B, N, N]
            k = self.k_neighbors
            idx = torch.topk(-d, k + 1, dim=-1).indices[:, :, 1:]  # [B, N, k]
            src = idx.reshape(B, -1)  # 每个 batch 各自的 knn 源点索引
            dst = torch.arange(N).repeat_interleave(k)[None, :].repeat(B, 1)  # 目标索引
            node_mask = torch.ones(B, N, dtype=torch.bool)
            node_mask[:, -2:] = False  # last two are padding
            update_mask = torch.ones(B, N, dtype=torch.bool)
            update_mask[:, :2] = False  # first two are frozen (no update)

            model = SHTransformer(C=C, L_max=L_max, R_bins=R_bins, n_layers=2)
            out = model(SH, Rm, t, src, dst, node_mask=node_mask, update_mask=update_mask)


class SHframe_fbb(nn.Module):
    def __init__(self,model_conf):
        super(SHframe_fbb, self).__init__()
        '''
        forward for binder,CSB is True
        forward_fixed topo for base model, which use node update ,rcsb_cluster30_GNNfromIPAscratch_node_mixed_again
        forward_ss for motif, neigh update, rcsb_ipagnnneigh_cluster_motif_updateall_reeidx_homo_heto

        '''
        self.model_conf=model_conf
        FA_conf=self.model_conf.FA

        # Hyperparameters
        # self.k_neighbors = k_neighbors

        self.SH_embedding = SHFeatureHead(C=model_conf.sh.C, L_max=model_conf.sh.L_max, R_bins=model_conf.sh.R_bins,
                                          hidden=model_conf.sh.hidden_scalar, num_classes=-1, dropout=model_conf.sh.dropout)

        self.FrameTransformer = FrameAwareTransformerRot(**FA_conf)
        self.out=SHPredictionHead(model_conf.sh.hidden_scalar,C=model_conf.sh.C, L_max=model_conf.sh.L_max,
                                  R_bins=model_conf.sh.R_bins,
                                  n_blocks=2,
                                  hidden=model_conf.sh.hidden_scalar,
                                 )

        self.SHGeoResHead=SHGeoResHead(C=model_conf.sh.C, L_max=model_conf.sh.L_max,
                                  R_bins=model_conf.sh.R_bins,hidden=model_conf.sh.hidden_scalar,dropout=0.1)


    def forward(self, SH, Rm, t,node_mask
                ):

        """ Graph-conditioned sequence/str model for binder
            graph will be calculated evary encoder layers
        """

        _,SH_E=self.SH_embedding(SH,node_mask)
        out =  self.FrameTransformer(SH_E, Rm, t,  node_mask=node_mask)
        shout=self.out(out)
        aalogits,coords=self.SHGeoResHead(shout,node_mask)

        return shout, aalogits,coords


class SideAtomsFlowModel(nn.Module):
    """
    FlowModel that incorporates noisy sidechain atom features using SideAtomsFeatureHead
    """

    def __init__(self, model_conf):
        super(SideAtomsFlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        self.feature_graph = BackboneEncoderGNN(dim_nodes=self._ipa_conf.c_s)

        # Sidechain atom feature extractor
        sidechain_conf = getattr(model_conf, 'sidechain_atoms', {})
        self.sidechain_head = SideAtomsFeatureHead(
            A=sidechain_conf.get('A', 11),  # number of sidechain atoms
            hidden=sidechain_conf.get('hidden', 256),
            num_classes=0,  # we only want features, not classification
            dropout=sidechain_conf.get('dropout', 0.1),
            conv_blocks=sidechain_conf.get('conv_blocks', 4),
            mlp_blocks=sidechain_conf.get('mlp_blocks', 4),
            fuse_blocks=sidechain_conf.get('fuse_blocks', 4),
            conv_groups=sidechain_conf.get('conv_groups', 1)
        )

        self.atoms_head=Feat2Atom11()

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
            # self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
            #     self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

        # Feature fusion layer to combine node features with sidechain features
        sidechain_hidden = getattr(model_conf, 'sidechain_atoms', {}).get('hidden', 256)
        self.feature_fusion = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + sidechain_hidden+self._ipa_conf.c_s , self._ipa_conf.c_s),
            nn.LayerNorm(self._ipa_conf.c_s),
            nn.SiLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s)
        )

        # Pairwise fusion for sidechain self-conditioning: [curr_sc_feat, sc_feat] -> sc_feat
        self.sc_pair_fusion = nn.Sequential(
            nn.Linear(2 * sidechain_hidden, sidechain_hidden),
            nn.LayerNorm(sidechain_hidden),
            nn.SiLU(),
            nn.Linear(sidechain_hidden, sidechain_hidden)
        )

        self.edge_feature_fusion = nn.Sequential(
            nn.Linear(self._ipa_conf.c_z +self._ipa_conf.c_z , self._ipa_conf.c_z),
            nn.LayerNorm(self._ipa_conf.c_z),
            nn.SiLU(),
            nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z)
        )

        self.NodeFeatExtractorWithHeads=NodeFeatExtractorWithHeads()

    def forward(self, input_feats,t_set=None):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        res_index = input_feats['res_idx']
        chain_idx = input_feats['chain_idx']

        noise_t = input_feats['r3_t']

        if t_set is not  None:
            noise_t=torch.tensor([t_set],dtype=torch.float32,device=noise_t.device).unsqueeze(0)

        rotmats_t = input_feats['rotmats_1']
        trans_t = input_feats['trans_1']


        # Cancel trans-based self-conditioning; we will use sidechain-level SC instead

        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            noise_t,
            node_mask,
            diffuse_mask,
            res_index
        )



        # Extract sidechain atom features if available
        sidechain_features = None
        sidechain_features_sc = None
        if 'atoms14_local_t' in input_feats and 'atom14_gt_exists' in input_feats:
            # Extract sidechain atoms (indices 3-13, assuming backbone is 0-2)
            atoms14_local_t = input_feats['atoms14_local_t']  # [B,N,14,3]
            atom14_exists = input_feats['atom14_gt_exists']   # [B,N,14]
            
            # Get sidechain atoms only (indices 3:13)
            sidechain_atoms = atoms14_local_t[..., 3:14, :]  # [B,N,11,3]
            sidechain_atom_mask = atom14_exists[..., 3:14]   # [B,N,11]
            
            # Extract features using SideAtomsFeatureHead

            _, sidechain_features = self.sidechain_head(
                sidechain_atoms,
                atom_mask=sidechain_atom_mask,
                node_mask=node_mask
            )  # [B,N,sidechain_hidden]

            # Optional sidechain self-conditioning branch
            if 'atoms14_local_sc' in input_feats:
                atoms14_local_sc = input_feats['atoms14_local_sc']  # scaled coordinates
                sc_side_atoms = atoms14_local_sc[..., 3:14, :]
                sc_side_mask = sidechain_atom_mask
                _, sidechain_features_sc = self.sidechain_head(
                    sc_side_atoms,
                    atom_mask=sc_side_mask,
                    node_mask=node_mask
                )  # [B,N,sidechain_hidden]

        # feature_graph 的几何阈值按 Å 设计，这里临时乘回 8 再计算图特征
        coord_scale = 8.0
        atoms14_local_for_graph = atoms14_local_t * coord_scale
        node_h, edge_h, edge_idx, mask_i, mask_ij = self.feature_graph(
            atoms14_local_for_graph[..., :4, :], chain_idx
        )


        # Fuse sidechain features with node embeddings if available
        if sidechain_features is not None:
            # Fuse SC if provided
            if sidechain_features_sc is not None:
                sidechain_features = self.sc_pair_fusion(
                    torch.cat([sidechain_features, sidechain_features_sc], dim=-1)
                )
            # Concatenate and fuse features
            combined_features = torch.cat([init_node_embed, sidechain_features, node_h], dim=-1)
            init_node_embed = self.feature_fusion(combined_features)
            # Apply masking to fused features
            init_node_embed = init_node_embed * node_mask[..., None]

        # Keep trans_sc zeros (disable trans-based SC)

        init_edge_embed = self.edge_feature_net(
            init_node_embed,
            trans_t,

            edge_mask,
            diffuse_mask,
        )

        init_edge_embed=self.edge_feature_fusion(torch.cat([init_edge_embed, edge_h], dim=-1))

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](
                node_embed,
                edge_embed,
                curr_rigids,
                node_mask,
            )
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]

            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        side_atoms,logits= self.NodeFeatExtractorWithHeads(node_embed,node_mask)

        curr_rigids_ang = self.rigids_nm_to_ang(curr_rigids)

        if 'atoms14_local_t' in input_feats:
            backbone_local = input_feats['atoms14_local_t'][..., :3, :]
        else:
            backbone_local = torch.zeros(*side_atoms.shape[:2], 3, 3, device=side_atoms.device, dtype=side_atoms.dtype)

        # Assemble local in scaled domain, then convert to Å before applying rigid
        local_full = torch.cat([backbone_local, side_atoms], dim=-2)
        coord_scale = 8.0
        local_full_ang = local_full #* coord_scale
        global_full = curr_rigids_ang[..., None].apply(local_full_ang)

        return {
            'side_atoms': side_atoms,
            'side_atoms_local_full': local_full,
            'atoms_global_full': global_full,
            'rigids_global': curr_rigids_ang,
            'logits':logits

        }


class SideAtomsFlowModel_backup(nn.Module):
    """旧版 SideAtomsFlowModel：
    - 无侧链自条件 SC 分支
    - feature_graph 直接用 atoms14_local_t（不乘 8）
    - 输出组装时不将局部坐标乘回 8（保持原始行为）
    """
    def __init__(self, model_conf):
        super(SideAtomsFlowModel_backup, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet_backuo(model_conf.edge_features)
        self.feature_graph = BackboneEncoderGNN(dim_nodes=self._ipa_conf.c_s)

        sidechain_conf = getattr(model_conf, 'sidechain_atoms', {})
        self.sidechain_head = SideAtomsFeatureHead(
            A=sidechain_conf.get('A', 11),
            hidden=sidechain_conf.get('hidden', 256),
            num_classes=0,
            dropout=sidechain_conf.get('dropout', 0.1),
            conv_blocks=sidechain_conf.get('conv_blocks', 4),
            mlp_blocks=sidechain_conf.get('mlp_blocks', 4),
            fuse_blocks=sidechain_conf.get('fuse_blocks', 4),
            conv_groups=sidechain_conf.get('conv_groups', 1),
        )
        self.atoms_head = Feat2Atom11()

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
                norm_first=False,
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False
            )
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(c=self._ipa_conf.c_s)
            if b < self._ipa_conf.num_blocks - 1:
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

        sidechain_hidden = getattr(model_conf, 'sidechain_atoms', {}).get('hidden', 256)
        self.feature_fusion = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s + sidechain_hidden + self._ipa_conf.c_s, self._ipa_conf.c_s),
            nn.LayerNorm(self._ipa_conf.c_s),
            nn.SiLU(),
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s),
        )
        self.edge_feature_fusion = nn.Sequential(
            nn.Linear(self._ipa_conf.c_z + self._ipa_conf.c_z, self._ipa_conf.c_z),
            nn.LayerNorm(self._ipa_conf.c_z),
            nn.SiLU(),
            nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z),
        )
        self.NodeFeatExtractorWithHeads = NodeFeatExtractorWithHeads()

    def forward(self, input_feats, t_set=None):
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        res_index = input_feats['res_idx']
        chain_idx = input_feats['chain_idx']

        noise_t = input_feats['r3_t']
        if t_set is not None:
            noise_t = torch.tensor([t_set], dtype=torch.float32, device=noise_t.device).unsqueeze(0)

        rotmats_t = input_feats['rotmats_1']
        trans_t = input_feats['trans_1']
        input_feats['trans_sc'] = input_feats['trans_1']

        init_node_embed = self.node_feature_net(
            noise_t,
            node_mask,
            diffuse_mask,
            res_index,
        )

        sidechain_features = None
        if 'atoms14_local_t' in input_feats and 'atom14_gt_exists' in input_feats:
            atoms14_local_t = input_feats['atoms14_local_t']
            atom14_exists = input_feats['atom14_gt_exists']
            sidechain_atoms = atoms14_local_t[..., 3:14, :]
            sidechain_atom_mask = atom14_exists[..., 3:14]
            _, sidechain_features = self.sidechain_head(
                sidechain_atoms, atom_mask=sidechain_atom_mask, node_mask=node_mask
            )

        node_h, edge_h, edge_idx, mask_i, mask_ij = self.feature_graph(
            atoms14_local_t[..., :4, :], chain_idx
        )

        if sidechain_features is not None:
            combined_features = torch.cat([init_node_embed, sidechain_features, node_h], dim=-1)
            init_node_embed = self.feature_fusion(combined_features)
            init_node_embed = init_node_embed * node_mask[..., None]

        trans_sc = input_feats['trans_sc'] if 'trans_sc' in input_feats else torch.zeros_like(trans_t)
        init_edge_embed = self.edge_feature_net(
            init_node_embed, trans_t, trans_sc, edge_mask, diffuse_mask
        )
        init_edge_embed = self.edge_feature_fusion(torch.cat([init_edge_embed, edge_h], dim=-1))

        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)

        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](node_embed, edge_embed, curr_rigids, node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool)
            )
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        side_atoms, logits = self.NodeFeatExtractorWithHeads(node_embed, node_mask)
        curr_rigids_ang = self.rigids_nm_to_ang(curr_rigids)

        if 'atoms14_local_t' in input_feats:
            backbone_local = input_feats['atoms14_local_t'][..., :3, :]
        else:
            backbone_local = torch.zeros(*side_atoms.shape[:2], 3, 3, device=side_atoms.device, dtype=side_atoms.dtype)

        local_full = torch.cat([backbone_local, side_atoms], dim=-2)
        global_full = curr_rigids_ang[..., None].apply(local_full)

        return {
            'side_atoms': side_atoms,
            'side_atoms_local_full': local_full,
            'atoms_global_full': global_full,
            'rigids_global': curr_rigids_ang,
            'logits': logits,
        }
