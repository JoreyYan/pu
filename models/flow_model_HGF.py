import torch
from torch import nn
import torch.nn.functional as F
from data import all_atom
from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet, EdgeFeatureNet_backuo
from models import ipa_pytorch  # ,so3_theta,rope3D
from models import GA_block
from openfold.utils import rigid_utils as ru
from data import utils as du
# from models.resnet import Conv2DFeatureExtractor
# from models.basic_vae import Encoder, Decoder
# from models.hours._hourglass import HourglassProteinCompressionTransformer
from models import utils as mu
from models.features.backbone_gnn_feature import BackboneEncoderGNN
from models.shattetnion.ShDecoderSidechain import Feat2Atom11, SideAtomsFeatureHead,SequenceHead
from data.GaussianRigid import OffsetGaussianRigid
from models.components.frozen_esm import FrozenEsmModel, ESM_REGISTRY
from models.components.sequence_adapters import SequenceToTrunkNetwork
from openfold.model.primitives import Linear, LayerNorm
from models.shattetnion.ShDecoderSidechain import Feat2Atom11, SideAtomsFeatureHead
from models.IGA import InvariantGaussianAttention,GaussianUpdateBlock,BottleneckIGAModule
from models.downblock import HierarchicalDownsampleIGAModule
from models.upsample_block import HierarchicalUpsampleIGAModule
from models.finnalup import FinalCoarseToFineIGAModule

# ============================================================================
# 4. 输出头 (Differentiable Atom Head)
# ============================================================================
class SidechainAtomHead(nn.Module):
    """
    可微原子头 (Nanometer Scale Version)：
    1. 预测局部原子坐标 (nm)。
    2. 在线计算高斯参数 (nm)。
    注意：这里的 base_thickness 必须适配 nm 尺度。
    """

    def __init__(self, c_in, num_atoms=11, base_thickness=None):
        """
        Args:
            c_in: input feature dimension
            num_atoms: number of sidechain atoms (default 11)
            base_thickness: base thickness for Gaussian scaling (Angstrom), must be provided
        """
        if base_thickness is None:
            raise ValueError("base_thickness must be provided from config")
        super().__init__()
        self.num_atoms = num_atoms
        self.projection = nn.Sequential(
            Linear(c_in, c_in),
            nn.LayerNorm(c_in),
            nn.SiLU(),
            Linear(c_in, num_atoms * 3)
        )

        # [修改点] 定义基础厚度 (Angstrom) 并转为内部单位
        # base_thickness Angstrom -> base_thickness * 0.1 Nanometer
        self.base_thickness_ang = base_thickness
        self.scale_factor = 0.1  # Ang -> Nm
        # 【关键修改】自定义初始化
        self._init_weights()

    def _init_weights(self):
        # 让最后一层 weight 很小，使得初始 u ~ 0，原子初始大致落在 μ 上
        nn.init.normal_(self.projection[-1].weight, std=1e-4)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, s, gaussian_rigid, sidechain_mask):
        """
        Args:
            s: [B, N, C]  节点语义特征 (node_embed)
            gaussian_rigid: OffsetGaussianRigid, [B, N] (nm unit)
                - 内部包含:
                    _local_mean: [B, N, 3]  (nm)
                    _scaling_log: [B, N, 3] (log σ, nm)
            sidechain_mask: [B, N, num_atoms]
                - 1 表示该原子存在且需要预测；0 表示不存在（如 Gly）或不参与 loss

        Returns:
            global_pred: [B, N, num_atoms, 3]  (nm, global frame)
            local_pred:  [B, N, num_atoms, 3]  (nm, local frame)
            gaussian_rigid: 直接返回传入的对象（不再在 head 中修改）
        """
        B, N, C = s.shape
        assert sidechain_mask.shape[0] == B and sidechain_mask.shape[1] == N, \
            "sidechain_mask batch/length mismatch"
        assert sidechain_mask.shape[2] == self.num_atoms, \
            "sidechain_mask last dim != num_atoms"

        # -------------------------------------------------------
        # 1. 从 gaussian_rigid 取出椭圆参数 (local μ, log σ)，单位 nm
        # -------------------------------------------------------
        # 这里约定：gaussian_rigid 已经是 nm 尺度（trunk 里 scale_translation(0.1) 之后）
        mu_local = gaussian_rigid._local_mean  # [B, N, 3]
        scale_log = gaussian_rigid._scaling_log  # [B, N, 3]
        sigma_local = torch.exp(scale_log)  # [B, N, 3], nm

        # 扩展到 [B, N, 1, 3]，方便和 num_atoms 广播
        mu_exp = mu_local.unsqueeze(-2)  # [B, N, 1, 3]
        sigma_exp = sigma_local.unsqueeze(-2)  # [B, N, 1, 3]

        # -------------------------------------------------------
        # 2. 预测“归一化偏移” u(s)，再用 (μ, σ) 解码坐标
        # -------------------------------------------------------
        # u: [B, N, num_atoms, 3]
        u_raw = self.projection(s).view(B, N, self.num_atoms, 3)
        # 可选：限制在 [-1, 1] 区间，避免一开始偏移过大 许 u 在需要的时候“突破 [-1,1]”去表达真实原子位置
        u = torch.tanh(u_raw) * (1 + 0.2 * torch.abs(u_raw))

        # 椭圆内解码：x_local = μ + u ⊙ σ
        local_pred = mu_exp + u*sigma_exp   # [B, N, num_atoms, 3]

        # 对不存在/不训练的原子位置，直接置 0，避免污染可视化/后续计算
        mask = sidechain_mask.unsqueeze(-1)  # [B, N, num_atoms, 1]
        local_pred = local_pred * mask

        # -------------------------------------------------------
        # 3. 由 GaussianRigid 做局部→全局变换 (nm)
        # -------------------------------------------------------
        # gaussian_rigid: [B, N]，扩展一个原子维度，用 apply 做刚体变换
        rigid_expanded = gaussian_rigid.unsqueeze(-1)  # [B, N, 1]
        global_pred = rigid_expanded.apply(local_pred)  # [B, N, num_atoms, 3]

        # 不在 head 里重新构造新的 gaussian_rigid，直接把传入的返回出去
        return global_pred, local_pred, gaussian_rigid


# ============================================================================
# 5. 主模型架构 (SideAtomsIGAModel)
# ============================================================================




class HierarchicalGaussianFieldModel(nn.Module):
    """
    IGA-Based Model preserving original feature extraction logic.
    """

    def __init__(self, model_conf):
        super(HierarchicalGaussianFieldModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa

        # Rigids utils
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)

        # ============================================================
        # 1. 保留你原有的特征提取层 (Do Not Touch)
        # ============================================================
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)
        self.feature_graph = BackboneEncoderGNN(dim_nodes=self._ipa_conf.c_s)

        sidechain_hidden = getattr(model_conf, 'sidechain_atoms', {}).get('hidden', 256)

        # ESM Logic
        self.use_esm = getattr(model_conf, 'use_esm', False)
        if self.use_esm:
            esm_model_name = getattr(model_conf, 'esm_model', 'esm2_650M')
            print(f'Initializing ESM model: {esm_model_name}')
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
            # Fusion with ESM
            self.feature_fusion = nn.Sequential(
                nn.Linear(self._ipa_conf.c_s + sidechain_hidden + self._ipa_conf.c_s + self._ipa_conf.c_s,
                          self._ipa_conf.c_s),
                nn.LayerNorm(self._ipa_conf.c_s), nn.SiLU(),
                nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s)
            )
            self.edge_feature_fusion = nn.Sequential(
                nn.Linear(self._ipa_conf.c_z + self._ipa_conf.c_z + self._model_conf.edge_embed_size,
                          self._ipa_conf.c_z),
                nn.LayerNorm(self._ipa_conf.c_z), nn.SiLU(),
                nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z)
            )
        else:
            self.seq_encoder = None
            self.sequence_to_trunk = None
            # Fusion without ESM
            self.feature_fusion = nn.Sequential(
                nn.Linear(self._ipa_conf.c_s + sidechain_hidden + self._ipa_conf.c_s,
                          self._ipa_conf.c_s),
                nn.LayerNorm(self._ipa_conf.c_s), nn.SiLU(),
                nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s)
            )
            self.edge_feature_fusion = nn.Sequential(
                nn.Linear(self._ipa_conf.c_z + self._ipa_conf.c_z,
                          self._ipa_conf.c_z),
                nn.LayerNorm(self._ipa_conf.c_z), nn.SiLU(),
                nn.Linear(self._ipa_conf.c_z, self._ipa_conf.c_z)
            )

        # LayerNorms
        self.node_feature_ln = nn.LayerNorm(self._ipa_conf.c_s)
        self.sidechain_feature_ln = nn.LayerNorm(sidechain_hidden)
        self.graph_feature_ln = nn.LayerNorm(self._ipa_conf.c_s)
        if self.use_esm:
            self.esm_feature_ln = nn.LayerNorm(self._ipa_conf.c_s)

        self.edge_init_ln = nn.LayerNorm(self._model_conf.edge_embed_size)
        self.edge_graph_ln = nn.LayerNorm(self._ipa_conf.c_z)
        if self.use_esm:
            self.edge_esm_ln = nn.LayerNorm(self._model_conf.edge_embed_size)

        # Sidechain Feature Extractors
        sidechain_conf = getattr(model_conf, 'sidechain_atoms', {})
        self.sidechain_head = SideAtomsFeatureHead(
            A=sidechain_conf.get('A', 10),
            hidden=sidechain_conf.get('hidden', 256),
            num_classes=0, dropout=sidechain_conf.get('dropout', 0.1),
            conv_blocks=sidechain_conf.get('conv_blocks', 4),
            mlp_blocks=sidechain_conf.get('mlp_blocks', 4),
            fuse_blocks=sidechain_conf.get('fuse_blocks', 4),
            conv_groups=sidechain_conf.get('conv_groups', 1)
        )

        # self.sc_pair_fusion = nn.Sequential(
        #     nn.Linear(2 * sidechain_hidden, sidechain_hidden),
        #     nn.LayerNorm(sidechain_hidden), nn.SiLU(),
        #     nn.Linear(sidechain_hidden, sidechain_hidden)
        # )

        # ============================================================
        # 2. IGA Trunk (Gaussian Attention Stack)
        # ============================================================
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            # [KEY] IPA -> IGA (No Pair branch used inside IGA, but we keep structure)
            self.trunk[f'iga_{b}'] = InvariantGaussianAttention(
                c_s=self._ipa_conf.c_s,
                c_z=self._ipa_conf.c_z,
                c_hidden=self._ipa_conf.c_hidden,
                no_heads=self._ipa_conf.no_heads,
                no_qk_gaussians=self._ipa_conf.no_qk_points,
                no_v_points=self._ipa_conf.no_v_points,
                layer_idx=b,
                enable_vis=True,  # 关闭可视化，避免DDP未使用参数错误
            )

            # self.trunk[f'iga_{b}'] = ipa_pytorch.InvariantPointAttention(
            #     self._ipa_conf,
            #
            # )

            self.trunk[f'iga_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)

            # Seq Transformer
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,  # or *2
                batch_first=True, dropout=0.0, norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers)

            self.trunk[f'post_tfmr_{b}'] = Linear(tfmr_in, self._ipa_conf.c_s, init="final")

            # Transition
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(c=self._ipa_conf.c_s)
            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )

            self.trunk[f'Gau_update_{b}'] = GaussianUpdateBlock(
                self._ipa_conf.c_s)

        self.downsampler = HierarchicalDownsampleIGAModule(
            c_s=self._ipa_conf.c_s,
            iga_conf=self._ipa_conf,
            OffsetGaussianRigid_cls=OffsetGaussianRigid,
            num_downsample=2,  # 你要的 K 次
            ratio=6.0,
            k_max_cap=64,
            coarse_iga_layers=4,  # 每次 down 后 IGA 4 次
        )

        self.upsampler = HierarchicalUpsampleIGAModule(
            c_s=self._ipa_conf.c_s,
            iga_conf=self._ipa_conf,
            OffsetGaussianRigid_cls=OffsetGaussianRigid,
            num_upsample=2,
            M_max=8,
            K_target=None,  # 不写死，自动按 up_ratio 生成期望预算
            up_ratio=2.0,
            neighbor_R=2,
            coarse_iga_layers=4,
        )

        self.bottleneck = BottleneckIGAModule(
            c_s=self._ipa_conf.c_s,
            iga_conf=self._ipa_conf,
            bottleneck_layers=getattr(model_conf, "bottleneck_layers", 6),
            layer_idx_base=3000,
            enable_vis=False,
        )

        self.final_up = FinalCoarseToFineIGAModule(
            c_s=self._ipa_conf.c_s,
            iga_conf=self._ipa_conf,
            OffsetGaussianRigid_cls=OffsetGaussianRigid,
            num_refine_layers=4,
            neighbor_R=2,  # R=1 最稳；R=2/4 纠错更强
            w_attach=1.0,
            w_entB=0.0,  # 后期你再退火加到 1e-3~1e-2
            enable_occ_loss=False,
        )

        # ============================================================
        # 3. Output Heads (Differentiable Atom & Sequence)
        # ============================================================
        # Differentiable Atom Head: Predicts Local Coords -> Gaussian On-the-fly
        # base_thickness must be set in model config (configs/model.yaml)
        self.base_thickness = model_conf.get('base_thickness', 0.5)#model_conf.base_thickness
        self.atom_head = SidechainAtomHead(self._ipa_conf.c_s, num_atoms=10, base_thickness=self.base_thickness)

        # Sequence Head
        self.logits_head = SequenceHead(self._ipa_conf.c_s, self._ipa_conf.c_s, num_classes=21)

    def forward(self, input_feats,step,total_steps, sideonly=False):
        """
        Forward logic merging original feature extraction with IGA trunk.
        """
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        res_index = input_feats['res_idx']
        chain_idx = input_feats['chain_idx']


        # Dummy t for NodeFeatureNet compatibility (since we do regression now)
        if 'r3_t' in input_feats:
            noise_t = input_feats['r3_t']
        else:
            noise_t = torch.ones((node_mask.shape[0], 1), device=node_mask.device)



        # ------------------------------------------------------------
        # A. Feature Extraction (Preserved from original)
        # ------------------------------------------------------------
        init_node_embed = self.node_feature_net(noise_t, node_mask, diffuse_mask, res_index)

        # Sidechain CNN Features
        sidechain_features = None
        if 'atoms14_local_t' in input_feats:
            # Masked input atoms (Target part is 0)
            atoms14_local_t = input_feats['atoms14_local_t']
            sidechain_atoms = atoms14_local_t[..., 4:14, :]
            sidechain_atom_mask = input_feats['sidechain_atom_mask']

            # Extract SC features
            _, sidechain_features = self.sidechain_head(
                sidechain_atoms, atom_mask=sidechain_atom_mask, node_mask=node_mask
            )



        # Graph Features (Backbone)
        node_h, edge_h, _, _, _ = self.feature_graph(input_feats['atoms14_local_t'][..., :4, :], chain_idx)

        # ESM Features
        if self.use_esm and self.seq_encoder is not None:
            # Note: For inverse folding, aatype should be masked before passing here!
            seq_emb_s, seq_emb_z = self.seq_encoder(input_feats['aatype'], chain_idx, attn_mask=node_mask)
            seq_emb_s, seq_emb_z = self.sequence_to_trunk(seq_emb_s, seq_emb_z, res_index, node_mask)

        # Fusion: Node Level
        init_node_embed_norm = self.node_feature_ln(init_node_embed)
        sidechain_features_norm = self.sidechain_feature_ln(sidechain_features)
        node_h_norm = self.graph_feature_ln(node_h)

        to_concat = [init_node_embed_norm, sidechain_features_norm, node_h_norm]
        if self.use_esm:
            to_concat.append(self.esm_feature_ln(seq_emb_s))

        init_node_embed = self.feature_fusion(torch.cat(to_concat, dim=-1))
        init_node_embed = init_node_embed * node_mask[..., None]

        # Fusion: Edge Level (Optional for IGA, but preserved for compatibility)
        # Note: Even if IGA doesn't use 'z', we keep this to not break legacy code structure
        # init_edge_embed = self.edge_feature_net(init_node_embed, trans_t, edge_mask, diffuse_mask)
        # ... (Edge fusion logic omitted for brevity, assuming standard) ...

        # ------------------------------------------------------------
        # B. Gaussian-IGA Trunk
        # ------------------------------------------------------------
        # Rigid (Fixed Backbone)
        # 1. 准备 Backbone Rigid (基础骨架)
        rotmats_t = input_feats['rotmats_1']
        trans_t = input_feats['trans_1']
        base_rigid = du.create_rigid(rotmats_t, trans_t)

        # 4. 构造动态厚度张量 (Dynamic Base Thickness)
        # update_mask: 1=Masked(待预测), 0=Context(已知)
        # [B, N]
        is_masked = input_feats['update_mask'].bool()
        # 策略:
        # - Masked: 2.5 Å (虚胖，为了搜索和梯度)
        # - Context: 0.5 Å (紧致，为了物理真实)
        # [B, N, 1]
        dynamic_thickness = torch.where(
            is_masked,
            torch.tensor(2.5, device=is_masked.device),
            torch.tensor(0.5, device=is_masked.device)
        ).unsqueeze(-1)





        # 2. 准备 Sidechain Atoms (用于初始化高斯形状)
        # 注意: 这里使用的是 input_feats['atoms14_local_t']
        # 对于被 Mask 的区域，这部分坐标是 0；对于 Context 区域，是真实坐标。
        if 'atoms14_local_t' in input_feats and sideonly:
            # [Old] 只用侧链原子

            # 0. 准备原子 (Local -> Global)
            # atoms14_in: [B, N, 14, 3]
            atoms14_local = input_feats['atoms14_local_t']
            sidechain_atoms_local = atoms14_local[..., 3:14, :]

            # 【关键步骤】将局部原子变换到全局坐标系
            # base_rigid: [B, N] -> [B, N, 1]
            sidechain_atoms_global = base_rigid.unsqueeze(-1).apply(sidechain_atoms_local)

            # 1. 获取 GT 存在掩码 (基础)
            # [B, N, 11]
            gt_sc_exists = input_feats['atom14_gt_exists'][..., 3:14].bool()

            # 2. 获取 预测任务掩码
            # update_mask: [B, N] -> [B, N, 1]
            is_masked_residue = input_feats['update_mask'][..., None].bool()

            # 3. 【核心修正】构建混合掩码
            # Context (未Mask) 区域: 使用 GT 掩码 (保持真实形状)
            # Masked (待预测) 区域: 强制为 False (全0，初始化为标准球)
            # 逻辑: exists AND (NOT masked)
            geom_mask = gt_sc_exists & (~is_masked_residue)



            # 5. 构建高斯 (传入 Tensor)
            # atoms14_in 在 masked 区域已经是 0，context 区域是 GT
            # 配合 geom_mask (之前讨论的: context=1, masked=0)

            curr_rigids = OffsetGaussianRigid.from_rigid_and_sidechain(
                base_rigid,
                sidechain_atoms_global,
                geom_mask,  # Masked=0 (触发回退到CA), Context=1 (使用真实原子)
                base_thickness=dynamic_thickness  # <--- 传入 Tensor！
            )




        if 'atoms14_local_t' in input_feats and not sideonly:
            # ... (前文: base_rigid, is_masked, dynamic_thickness 准备完毕) ...

            # ============================================================
            # 1. 准备全原子坐标 (Local -> Global)
            # ============================================================
            # atoms14_local_t: [B, N, 14, 3]
            # 包含: N, CA, C (真实值) + O, Sidechain (Masked区域为0, Context区域为真实值)
            atoms14_local = input_feats['atoms14_local_t']

            # 【关键】变换到全局坐标系 (OffsetGaussianRigid.from_... 需要全局坐标)
            # base_rigid: [B, N] -> expand -> [B, N, 14]
            all_atoms_global = base_rigid.unsqueeze(-1).apply(atoms14_local)

            # ============================================================
            # 2. 准备全原子掩码 (Geometric Mask)
            # ============================================================
            # 初始掩码: 物理上存在的原子 [B, N, 14]
            gt_exists = input_feats['atom14_gt_exists'].float()

            # 构造混合掩码:
            # - Context (is_masked=0): 使用 gt_exists (全原子)
            # - Masked  (is_masked=1): 只保留 N, CA, C (0-2), 屏蔽 O (3) 和 侧链 (4-13)

            # 扩展 mask 维度以匹配原子 [B, N, 1]
            is_masked_broad = is_masked.unsqueeze(-1)

            # 拆分 backbone (N, CA, C) 和 其他 (O, SC)
            mask_bb_core = gt_exists[..., :3]  # Indices 0, 1, 2 (总是保留)
            mask_others = gt_exists[..., 3:]  # Indices 3-13 (O + Sidechain)

            # 对 "其他" 部分应用屏蔽: 如果是 Masked 区域，则置 0
            mask_others_filtered = mask_others * (~is_masked_broad).float()

            # 拼接回全原子掩码 [B, N, 14]
            geom_mask_all = torch.cat([mask_bb_core, mask_others_filtered], dim=-1)

            # ============================================================
            # 3. 构建高斯 (使用全原子接口)
            # ============================================================
            # 调用新写的 from_rigid_and_all_atoms
            curr_rigids = OffsetGaussianRigid.from_rigid_and_all_atoms(
                base_rigid,
                all_atoms_global,  # 全局全原子坐标
                geom_mask_all,  # 修正后的全原子掩码
                base_thickness=dynamic_thickness  # 动态厚度 (Masked=2.5, Context=0.5)
            )

            # 此时:
            # Masked 区域: 基于 N, CA, C 统计 -> 生成沿主链方向的胖球 (Scale=2.5)
            # Context 区域: 基于 全原子 统计 -> 生成真实的残基形状 (Scale=0.5+std)

        # 【关键步骤】单位转换: Angstrom -> Nanometer
        # 神经网络喜欢小数值 (分布在 -1~1 或 0~10 之间)
        # Angstrom (坐标~100) 太大了，容易导致 Attention 数值不稳定
        curr_rigids = curr_rigids.scale_translation(0.1)
        node_embed = init_node_embed* node_mask[..., None]



        # edge
        init_edge_embed = self.edge_feature_net(
            init_node_embed,
            trans_t,

            edge_mask,
            diffuse_mask,
        )

        # Normalize edge features before concatenation to match scales
        init_edge_embed_norm = self.edge_init_ln(init_edge_embed)
        edge_h_norm = self.edge_graph_ln(edge_h)
        if self.use_esm:
            seq_emb_z_norm = self.edge_esm_ln(seq_emb_z)
            init_edge_embed = self.edge_feature_fusion(torch.cat([init_edge_embed_norm, edge_h_norm, seq_emb_z_norm], dim=-1))
        else:
            init_edge_embed = self.edge_feature_fusion(torch.cat([init_edge_embed_norm, edge_h_norm], dim=-1))
        edge_embed = init_edge_embed * edge_mask[..., None]


        for b in range(self._ipa_conf.num_blocks):
            # IGA Block (Replaces IPA)
            # Note: z is NOT passed to IGA (Lite version)
            iga_out = self.trunk[f'iga_{b}'](
                s=node_embed,
                z=edge_embed,
                r=curr_rigids,  # Pass Fixed Backbone Rigid
                mask=node_mask
            )
            iga_out *= node_mask[..., None]

            node_embed = self.trunk[f'iga_ln_{b}'](node_embed + iga_out)

            # Seq Transformer
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                node_embed,
                src_key_padding_mask=(1 - node_mask).to(torch.bool)
            )
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)

            # Transition
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]

            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

            curr_rigids = self.trunk[f'Gau_update_{b}'](
                node_embed,
                curr_rigids,
                mask=input_feats['update_mask']  # <--- 传入 mask
            )


        #down

        levels_down, pool_reg = self.downsampler(
            s_f=node_embed,
            r_f=curr_rigids,
            mask_f=node_mask,
            step=step,  # 训练时传 global_step
            total_steps=total_steps,  # 训练时传 total_steps
        )

        # --- take coarsest level ---
        coarsest = levels_down[-1]
        sL = coarsest["s"]
        rL = coarsest["r"]
        mL = coarsest["mask"]

        # --- bottleneck IGA refine (must-have) ---
        sL, rL = self.bottleneck(sL, rL, mL)

        levels_up, up_reg = self.upsampler(
            s_l=sL,
            r_l=rL,
            mask_l=mL,
            step=step,
            total_steps=total_steps,
        )

        final_levels, final_reg = self.final_up(
            s_parent=s_last, r_parent=r_last, mask_parent=mask_last,
            node_mask=node_mask,
            res_idx=input_feats["res_idx"],
        )
        # residue-level output:
        s_res = final_levels[-1]["s"]
        r_res = final_levels[-1]["r"]

        # ------------------------------------------------------------
        # C. Output Heads (Coordinates & Logits)
        # ------------------------------------------------------------
        # 1. Logits
        logits = self.logits_head(node_embed)

        # 2. Coordinates & Gaussian
        # Atom mask is needed for correct Gaussian calculation
        atom_mask = input_feats.get('sidechain_atom_mask')
        if atom_mask is None: atom_mask = torch.ones((*node_embed.shape[:2], 10), device=node_embed.device)

        # Atom Head: Predicts Local Coords -> Calculates Gaussian -> Returns Objects
        pred_global_nm, pred_local_nm, gaussian_rigid = self.atom_head(node_embed, curr_rigids, atom_mask)

        # 【关键步骤】转回 Angstrom 用于 Loss 和 输出
        # 坐标 * 10
        pred_local = pred_local_nm * 10.0
        pred_global = pred_global_nm * 10.0

        # 高斯对象也要转回来 (用于可视化或下一轮)
        final_gaussian_ang = gaussian_rigid.scale_translation(10.0)

        return {
            'pred_atoms': pred_local,  # For MSE Loss (Local Frame)
            'pred_atoms_global': pred_global,  # For Visualization
            'logits': logits,  # For CE Loss
            'final_gaussian': final_gaussian_ang  # For Analysis/Visualization
        }