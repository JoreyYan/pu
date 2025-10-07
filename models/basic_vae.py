from typing import Callable, Literal, Optional, Tuple, Union
from types import SimpleNamespace
import torch
from torch import nn
from data import all_atom
from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models import ipa_pytorch,so3_theta,rope3D
from data import utils as du
from chroma.layers.structure import protein_graph
from models.resnet import Conv2DFeatureExtractor

class BackboneEncoderGNN(nn.Module):
    """Graph Neural Network for processing protein structure into graph embeddings.

    Args:
        See documention of `structure.protein_graph.ProteinFeatureGraph`,
        and `graph.GraphNN` for more details.

        dim_nodes (int): Hidden dimension of node tensors.
        dim_edges (int): Hidden dimension of edge tensors.
        num_neighbors (int): Number of neighbors per nodes.
        node_features (tuple): List of node feature specifications. Features
            can be given as strings or as dictionaries.
        edge_features (tuple): List of edge feature specifications. Features
            can be given as strings or as dictionaries.
        num_layers (int): Number of layers.
        node_mlp_layers (int): Number of hidden layers for node update
            function.
        node_mlp_dim (int, optional): Dimension of hidden layers for node update
            function, defaults to match output dimension.
        edge_update (bool): Whether to include an edge update step.
        edge_mlp_layers (int): Number of hidden layers for edge update
            function.
        edge_mlp_dim (int, optional): Dimension of hidden layers for edge update
            function, defaults to match output dimension.
        skip_connect_input (bool): Whether to include skip connections between
            layers.
        mlp_activation (str): MLP nonlinearity function, `relu` or `softplus`
            accepted.
        dropout (float): Dropout fraction.
        graph_distance_atom_type (int): Atom type for computing residue-residue
            distances for graph construction. Negative values will specify
            centroid across atom types. Default is `-1` (centroid).
        graph_cutoff (float, optional): Cutoff distance for graph construction:
            mask any edges further than this cutoff. Default is `None`.
        graph_mask_interfaces (bool): Restrict connections only to within
            chains, excluding-between chain interactions. Default is `False`.
        graph_criterion (str): Method used for building graph from distances.
            Currently supported methods are `{knn, random_log, random_linear}`.
            Default is `knn`.
        graph_random_min_local (int): Minimum number of neighbors in GNN that
            come from local neighborhood, before random neighbors are chosen.
        checkpoint_gradients (bool): Switch to implement gradient checkpointing
            during training.

    Inputs:
        X (torch.Tensor): Backbone coordinates with shape
                `(num_batch, num_residues, num_atoms, 3)`.
        C (torch.LongTensor): Chain map with shape `(num_batch, num_residues)`.
        node_h_aux (torch.LongTensor, optional): Auxiliary node features with
            shape `(num_batch, num_residues, dim_nodes)`.
        edge_h_aux (torch.LongTensor, optional): Auxiliary edge features with
            shape `(num_batch, num_residues, num_neighbors, dim_edges)`.
        edge_idx (torch.LongTensor, optional): Input edge indices for neighbors
            with shape `(num_batch, num_residues, num_neighbors)`.
        mask_ij (torch.Tensor, optional): Input edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`.

    Outputs:
        node_h (torch.Tensor): Node features with shape
            `(num_batch, num_residues, dim_nodes)`.
        edge_h (torch.Tensor): Edge features with shape
            `(num_batch, num_residues, num_neighbors, dim_edges)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_residues, num_neighbors)`.
        mask_i (torch.Tensor): Node mask with shape `(num_batch, num_residues)`.
        mask_ij (torch.Tensor): Edge mask with shape
             `(num_batch, num_nodes, num_neighbors)`.
    """

    def __init__(
        self,
        dim_nodes: int = 256,
        dim_edges: int = 128,
        num_neighbors: int = 30,
        node_features: tuple = (("internal_coords", {"log_lengths": True}),),
        edge_features: tuple = (
            "distances_2mer",
            "orientations_2mer",
            "distances_chain",
        ),
        num_layers: int = 3,
        node_mlp_layers: int = 1,
        node_mlp_dim: Optional[int] = None,
        edge_update: bool = True,
        edge_mlp_layers: int = 1,
        edge_mlp_dim: Optional[int] = None,
        skip_connect_input: bool = False,
        mlp_activation: str = "softplus",
        dropout: float = 0.1,
        graph_distance_atom_type: int = -1,
        graph_cutoff: Optional[float] = None,
        graph_mask_interfaces: bool = False,
        graph_criterion: str = "knn",
        graph_random_min_local: int = 20,
        checkpoint_gradients: bool = False,
        **kwargs
    ) -> None:
        """Initialize BackboneEncoderGNN."""
        super(BackboneEncoderGNN, self).__init__()

        # Save configuration in kwargs
        self.kwargs = locals()
        self.kwargs.pop("self")
        for key in list(self.kwargs.keys()):
            if key.startswith("__") and key.endswith("__"):
                self.kwargs.pop(key)
        args = SimpleNamespace(**self.kwargs)

        # Important global options
        self.dim_nodes = dim_nodes
        self.dim_edges = dim_edges
        self.checkpoint_gradients = checkpoint_gradients

        graph_kwargs = {
            "distance_atom_type": args.graph_distance_atom_type,
            "cutoff": args.graph_cutoff,
            "mask_interfaces": args.graph_mask_interfaces,
            "criterion": args.graph_criterion,
            "random_min_local": args.graph_random_min_local,
        }

        self.feature_graph = protein_graph.ProteinFeatureGraph(
            dim_nodes=args.dim_nodes,
            dim_edges=args.dim_edges,
            num_neighbors=args.num_neighbors,
            graph_kwargs=graph_kwargs,
            node_features=args.node_features,
            edge_features=args.edge_features,
        )




    def forward(
        self,
        X: torch.Tensor,
        C: torch.LongTensor,
        node_h_aux: Optional[torch.Tensor] = None,
        edge_h_aux: Optional[torch.Tensor] = None,
        edge_idx: Optional[torch.Tensor] = None,
        mask_ij: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor
    ]:
        """Encode XC backbone structure into node and edge features."""
        num_batch, num_residues = C.shape

        # 生成所有可能的残基对的索引
        i_idx, j_idx = torch.meshgrid(torch.arange(num_residues), torch.arange(num_residues), indexing='ij')
        # 扩展 edge_idx 以匹配批次大小
        edge_idx = j_idx.unsqueeze(0).expand(num_batch, num_residues, num_residues).to(X.device)

        # Hack to enable checkpointing
        if self.checkpoint_gradients and (not X.requires_grad):
            X.requires_grad = True

        node_h, edge_h, edge_idx, mask_i, mask_ij = self.feature_graph( X, C, edge_idx, mask_ij)

        if node_h_aux is not None:
            node_h = node_h + mask_i.unsqueeze(-1) * node_h_aux
        if edge_h_aux is not None:
            edge_h = edge_h + mask_ij.unsqueeze(-1) * edge_h_aux

        return node_h, edge_h, edge_idx, mask_i, mask_ij
class Encoder(nn.Module):

    def __init__(self, model_conf):
        super(Encoder, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        dim_nodes = self._ipa_conf.c_s
        self.feature_graph = BackboneEncoderGNN(dim_nodes=dim_nodes)
        self.cnn = Conv2DFeatureExtractor(output_dim=model_conf.node_embed_size)
        # Fusion layer after CNN (Linear layer)
        self.fusion_fc = nn.Sequential(
            nn.Linear(model_conf.node_embed_size * 3, model_conf.node_embed_size),  # Concatenation of both features
            nn.ReLU(),
            nn.Linear(model_conf.node_embed_size, model_conf.node_embed_size)
        )

        # self.cnn=Conv2DFeatureExtractor(output_dim=model_conf.node_embed_size)
        # # Fusion layer after CNN (Linear layer)
        # self.fusion_fc = nn.Sequential(
        #     nn.Linear(model_conf.node_embed_size * 2, model_conf.node_embed_size),  # Concatenation of both features
        #     nn.ReLU(),
        #     nn.Linear(model_conf.node_embed_size, model_conf.node_embed_size)
        # )
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
        trans_t = input_feats['trans_1']
        rotmats_t = input_feats['rotmats_1']

        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            node_mask,
            diffuse_mask,
            res_index
        )
        node_h, edge_h, edge_idx, mask_i, mask_ij = self.feature_graph(input_feats['backbone'], chain_idx)

        atoms_f=self.cnn(input_feats['backbone'])
        # Concatenate the features along the channel axis (axis=1)
        fused_features = torch.cat((init_node_embed, atoms_f,node_h), dim=-1)
        init_node_embed = self.fusion_fc(fused_features)


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
        init_edge_embed=init_edge_embed+edge_h

        # Concatenate the features along the channel axis (axis=1)
        # fused_features = torch.cat((init_node_embed, atoms_f), dim=-1)
        # init_node_embed = self.fusion_fc(fused_features)


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

            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]


        return node_embed

class Decoder(nn.Module):

    def __init__(self, model_conf):
        super(Decoder, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        # self.cnn=Conv2DFeatureExtractor(output_dim=model_conf.node_embed_size)
        # # Fusion layer after CNN (Linear layer)
        # self.fusion_fc = nn.Sequential(
        #     nn.Linear(model_conf.node_embed_size * 2, model_conf.node_embed_size),  # Concatenation of both features
        #     nn.ReLU(),
        #     nn.Linear(model_conf.node_embed_size, model_conf.node_embed_size)
        # )
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

    def forward(self, init_node_embed,input_feats,M):

        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        res_index = input_feats['res_idx']
        chain_idx=input_feats['chain_idx']
        trans_t = input_feats['trans_0']
        rotmats_t = input_feats['rotmats_0']

        # Initialize node and edge embeddings

        # pred_bb_atoms = all_atom.to_atom37(trans_t, rotmats_t)[:, :, :3]
        # atoms_f=self.cnn(pred_bb_atoms)


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
        # fused_features = torch.cat((init_node_embed, atoms_f), dim=-1)
        # init_node_embed = self.fusion_fc(fused_features)


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

        if M==-1:
            return {
                'pred_trans': pred_trans,
                'pred_rotmats': pred_rotmats,
            }
        else:

            return {
                'pred_trans': pred_trans[:,:M,...],
                'pred_rotmats': pred_rotmats[:,:M,...],
            }