import abc
import numpy as np
import pandas as pd
import logging
import tree
import torch
import random
from data.all_atom import prot_to_torsion_angles,torsion_angles_to_frames,frames_to_atom14_pos
from torch.utils.data import Dataset
from data import utils as du
from openfold.data import data_transforms
from data.GaussianRigid import OffsetGaussianRigid
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from chroma.layers.structure.backbone import FrameBuilder
#from data.sh_density import sh_density_from_atom14_with_masks#,sh_density_from_atom14_with_masks_e3nn
from analysis.show_element_name_coords import gather_positions_by_name_for_element,plot_element_positions_by_name
from openfold.np.residue_constants import rigid_group_atom_positions,residue_atoms
from analysis.show_tensor_cdf import plot_tensor_distribution
# from analysis.show_atoms_distance import analyze_your_data
# from openfold.data import all_atom
from openfold.utils.rigid_utils import Rigid, Rotation
# from data.openfold_frame_builder import OpenFoldFrameBuilder



from data.ele_atoms import build_elem_slot_maps,atom14_to_elem_slots,regroup_elem_to_atom14_fast

def show_atoms(aatype, atom14_positions,atom14_element_idx,rigid_group_atom_positions):
    # 选一个残基类型 & 元素
    aa = "ARG"  # 你要看的氨基酸
    elem = "N"  # 你要看的元素


    name2pts = gather_positions_by_name_for_element(
        target_aa_name=aa,
        target_elem=elem,
        aatype=aatype.unsqueeze(0),  # [B,N]
        atom14_positions=atom14_positions.unsqueeze(0),  # [B,N,14,3]
        atom14_element_idx=atom14_element_idx.unsqueeze(0),  # [B,N,14]
        rigid_group_atom_positions=rigid_group_atom_positions
    )

    plot_element_positions_by_name(name2pts, title=f"{aa} — {elem} atoms grouped by name")


def _rog_filter(df, quantile):
    y_quant = pd.pivot_table(
        df,
        values='radius_gyration', 
        index='modeled_seq_len',
        aggfunc=lambda x: np.quantile(x, quantile)
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    max_len = df.modeled_seq_len.max()
    pred_poly_features = poly.fit_transform(np.arange(max_len)[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1

    row_rog_cutoffs = df.modeled_seq_len.map(lambda x: pred_y[x-1])
    return df[df.radius_gyration < row_rog_cutoffs]


def _length_filter(data_csv, min_res, max_res):
    return data_csv[
        (data_csv.modeled_seq_len >= min_res)
        & (data_csv.modeled_seq_len <= max_res)
    ]


def _plddt_percent_filter(data_csv, min_plddt_percent):
    return data_csv[data_csv.num_confident_plddt > min_plddt_percent]


def _max_coil_filter(data_csv, max_coil_percent):
    return data_csv[data_csv.coil_percent <= max_coil_percent]


def _process_csv_row(processed_file_path, map, base_thickness,frame_builder):
    processed_feats = du.read_pkl(processed_file_path)
    processed_feats = du.parse_chain_feats(processed_feats)

    # Only take modeled residues.
    modeled_idx = processed_feats['modeled_idx']
    min_idx = np.min(modeled_idx)
    max_idx = np.max(modeled_idx)
    del processed_feats['modeled_idx']
    processed_feats = tree.map_structure(
        lambda x: x[min_idx:(max_idx+1)], processed_feats)

    # Run through OpenFold data transforms.
    chain_feats = {
        'aatype': torch.tensor(processed_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(processed_feats['atom_positions']).float(),
        'all_atom_mask': torch.tensor(processed_feats['atom_mask']).float(),
        # 'atom14_gt_exists': torch.tensor(processed_feats['atom14_gt_exists']).int(),
        # 'atom14_gt_positions': torch.tensor(processed_feats['atom14_gt_positions']),
        # 'atom14_alt_gt_positions': torch.tensor(processed_feats['atom14_alt_gt_positions']),
        # 'atom14_alt_gt_exists': torch.tensor(processed_feats['atom14_alt_gt_exists']).int(),
        # 'atom14_atom_is_ambiguous': torch.tensor(processed_feats['atom14_atom_is_ambiguous']).int(),
        'atom14_element_idx': torch.tensor(processed_feats['atom14_element_idx']).int()
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

    ##chain
    # Re-number residue indices for each chain such that it starts from 1.
    # Randomize chain indices.
    chain_idx = processed_feats['chain_index']
    res_idx = processed_feats['residue_index']
    new_res_idx = np.zeros_like(res_idx)
    new_chain_idx = np.zeros_like(res_idx)
    all_chain_idx = np.unique(chain_idx).tolist()
    shuffled_chain_idx = np.array(
        random.sample(all_chain_idx, len(all_chain_idx))) - np.min(all_chain_idx) + 1
    for i,chain_id in enumerate(all_chain_idx):
        chain_mask = (chain_idx == chain_id).astype(int)
        chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
        new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

        # Shuffle chain_index
        replacement_chain_id = shuffled_chain_idx[i]
        new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask

    backbone = torch.tensor(processed_feats['atom_positions'][:, [0, 1, 2, 4], :]).float()
    rotation_1,trans_1,q=frame_builder.inverse(backbone.unsqueeze(0),torch.tensor(new_res_idx).unsqueeze(0))
    rotation_1=rotation_1.squeeze(0)
    trans_1 = trans_1.squeeze(0)

    # bx=frame_builder(rotation_1,trans_1,torch.tensor(new_res_idx).unsqueeze(0))


    if torch.isnan(trans_1).any() or torch.isnan(rotation_1).any():
        raise ValueError(f'Found NaNs in {processed_file_path}')



    #torsion_angles, torsion_alt_angles,torsion_mask=prot_to_torsion_angles(chain_feats['aatype'],chain_feats['all_atom_positions'],chain_feats['all_atom_mask'])

    # ANGEL TO FRAMES
    # identity=rigid_utils.Rigid.identity((1,torsion_angles.shape[0]))
    # rigid=torsion_angles_to_frames(identity,torsion_angles.unsqueeze(0),chain_feats['aatype'].unsqueeze(0))
    # atoms14s=frames_to_atom14_pos(rigid,chain_feats['aatype'].unsqueeze(0))


    rigids_1 = Rigid(Rotation(rotation_1),trans_1)






    res_mask = torch.tensor(processed_feats['bb_mask']).int()

    dynamic_thickness = torch.where(
        ~res_mask.bool(),
        torch.tensor(2.5, device=res_mask.device),
        torch.tensor(0.0, device=res_mask.device)
    ).unsqueeze(-1)



    rigids_1 = OffsetGaussianRigid.from_rigid_and_all_atoms(
        rigids_1,
        chain_feats['atom14_gt_positions'],
        chain_feats['atom14_gt_exists'],
        base_thickness=dynamic_thickness
    )
    scaling_log_1 = rigids_1._scaling_log
    local_mean_1 = rigids_1._local_mean



    # R=rotmats_1
    # det_R = torch.det(R)
    # print(f"行列式: {det_R}")
    # print(f"应该接近1，实际偏差: {abs(det_R - 1)}")
    #
    # # 检查正交性
    # orthogonality_error = torch.norm(R @ R.transpose(-1, -2) - torch.eye(3, device=R.device))
    # print(f"正交性误差: {orthogonality_error}")



    aligned_frame=rigids_1[..., None].invert_apply(chain_feats['atom14_gt_positions'])*chain_feats['atom14_gt_exists'][...,None]





    # rotmats_1, trans_1,_ = frams.inverse(backbone.unsqueeze(0), torch.tensor(chain_idx).unsqueeze(0))





    # =================================================================

    return {
        'scaling_log_1':scaling_log_1.detach(),
        'local_mean_1':local_mean_1.detach(),
        'aatype': chain_feats['aatype']*res_mask,
        'rotmats_1': rotation_1.float(),
        'trans_1': trans_1.float(),
        'res_mask': res_mask,
        'chain_idx': new_chain_idx,
        'res_idx': new_res_idx,
        'backbone':  backbone.float(),
        # 'normalize_density': density.squeeze(0).float(),
        # 'density_mask': density_mask.squeeze(0).float(),
        # 'torsion_angles':chain_feats['torsion_angles_sin_cos'].float(),
        # 'torsion_alt_angles':chain_feats['alt_torsion_angles_sin_cos'].float(),
        # 'torsion_mask':chain_feats['torsion_angles_mask'].float(),
        'atom14_gt_exists': chain_feats['atom14_gt_exists'].float(),
        'atom14_alt_gt_exists': chain_feats['atom14_alt_gt_exists'].float(),
        'atom14_atom_is_ambiguous': chain_feats['atom14_atom_is_ambiguous'].float(),
        # 'atom14_element_idx': torch.tensor(map['elem14'][chain_feats['aatype']]),
        'atom14_gt_positions': chain_feats['atom14_gt_positions'].float(),
        'atom14_alt_gt_positions': chain_feats['atom14_alt_gt_positions'].float(),
        # 'atoms14_local':aligned_frame.float(),
        # 'rigids_1':rigids_1.to_tensor_7().float(),
        # 'sc_ca_t': torch.zeros_like(trans_1).float(),

    }


def _add_plddt_mask(feats, plddt_threshold):
    feats['plddt_mask'] = torch.tensor(
        feats['res_plddt'] > plddt_threshold).int()


def _read_clusters(cluster_path):
    pdb_to_cluster = {}
    with open(cluster_path, "r") as f:
        for i,line in enumerate(f):
            for chain in line.split(' '):
                pdb = chain.split('_')[0]
                pdb_to_cluster[pdb.upper()] = i
    return pdb_to_cluster


class BaseDataset(Dataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,

            task,
            is_predict=False,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._is_predict = is_predict
        self._dataset_cfg = dataset_cfg
        self.task = task
        self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path)
        if  self.dataset_cfg.do_not_filter:
            metadata_csv = self.raw_csv
        else:
            metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values(
            'modeled_seq_len', ascending=False)
        self._create_split(metadata_csv)
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

        self.map = build_elem_slot_maps()
        self.base_thickness =  dataset_cfg.get('base_thickness', 0.5)



    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg
    
    def __len__(self):
        return len(self.csv)

    @abc.abstractmethod
    def _filter_metadata(self, raw_csv: pd.DataFrame) -> pd.DataFrame:
        pass

    def _create_split(self, data_csv):
        # Training or validation specific logic.
        if self._is_predict:

            if self._dataset_cfg.max_eval_length is not None:
                data_csv = data_csv[
                    data_csv.modeled_seq_len <= self._dataset_cfg.max_eval_length
                ]
            data_csv = data_csv.sort_values('modeled_seq_len', ascending=False)
            self.csv = data_csv
            self._log.info(
                f'Validation: {len(self.csv)} examples with lengths {sorted(self.csv.modeled_seq_len.unique())}')
            self.csv['index'] = list(range(len(self.csv)))

        # Training or validation specific logic.
        else:

            if self.is_training:
                self.csv = data_csv
                self._log.info(
                    f'Training: {len(self.csv)} examples')
            else:
                if self._dataset_cfg.max_eval_length is None:
                    eval_lengths = data_csv.modeled_seq_len
                else:
                    eval_lengths = data_csv.modeled_seq_len[
                        data_csv.modeled_seq_len <= self._dataset_cfg.max_eval_length
                        ]
                all_lengths = np.sort(eval_lengths.unique())
                length_indices = (len(all_lengths) - 1) * np.linspace(
                    0.0, 1.0, self.dataset_cfg.num_eval_lengths)
                length_indices = length_indices.astype(int)
                eval_lengths = all_lengths[length_indices]
                eval_csv = data_csv[data_csv.modeled_seq_len.isin(eval_lengths)]

                # Fix a random seed to get the same split each time.
                eval_csv = eval_csv.groupby('modeled_seq_len').sample(
                    self.dataset_cfg.samples_per_eval_length,
                    replace=True,
                    random_state=123
                )
                eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
                self.csv = eval_csv
                self._log.info(
                    f'Validation: {len(self.csv)} examples with lengths {eval_lengths}')
            self.csv['index'] = list(range(len(self.csv)))

    def process_csv_row(self, csv_row):
        path = csv_row['processed_path']
        seq_len = csv_row['modeled_seq_len']
        # Large protein files are slow to read. Cache them.
        use_cache = seq_len > self._dataset_cfg.cache_num_res
        if use_cache and path in self._cache:
            return self._cache[path]
        processed_row = _process_csv_row(path, self.map, self.base_thickness,self.FrameBuilder)
        if use_cache:
            self._cache[path] = processed_row
        return processed_row
    
    def _sample_scaffold_mask(self, batch, rng):
        trans_1 = batch['trans_1']
        num_res = trans_1.shape[0]
        min_motif_size = int(self._dataset_cfg.min_motif_percent * num_res)
        max_motif_size = int(self._dataset_cfg.max_motif_percent * num_res)

        # Sample the total number of residues that will be used as the motif.
        total_motif_size = self._rng.integers(
            low=min_motif_size,
            high=max_motif_size
        )

        # Sample motifs at different locations.
        num_motifs = rng.integers(low=1, high=total_motif_size)

        # Attempt to sample
        attempt = 0
        while attempt < 100:
            # Sample lengths of each motif.
            motif_lengths = np.sort(
                rng.integers(
                    low=1,
                    high=max_motif_size,
                    size=(num_motifs,)
                )
            )

            # Truncate motifs to not go over the motif length.
            cumulative_lengths = np.cumsum(motif_lengths)
            motif_lengths = motif_lengths[cumulative_lengths < total_motif_size]
            if len(motif_lengths) == 0:
                attempt += 1
            else:
                break
        if len(motif_lengths) == 0:
            motif_lengths = [total_motif_size]

        # Sample start location of each motif.
        seed_residues = rng.integers(
            low=0,
            high=num_res-1,
            size=(len(motif_lengths),)
        )

        # Construct the motif mask.
        motif_mask = torch.zeros(num_res)
        for motif_seed, motif_len in zip(seed_residues, motif_lengths):
            motif_mask[motif_seed:min(motif_seed+motif_len, num_res)] = 1.0
        scaffold_mask = 1 - motif_mask
        return scaffold_mask * batch['res_mask']
    
    def setup_inpainting(self, feats, rng):
        diffuse_mask = self._sample_scaffold_mask(feats, rng)
        if 'plddt_mask' in feats:
            diffuse_mask = diffuse_mask * feats['plddt_mask']
        if torch.sum(diffuse_mask) < 1:
            # Should only happen rarely.
            diffuse_mask = torch.ones_like(diffuse_mask)
        feats['diffuse_mask'] = diffuse_mask

    def __getitem__(self, row_idx):
        # Process data example.
        csv_row = self.csv.iloc[row_idx]
        feats = self.process_csv_row(csv_row)

        if self._dataset_cfg.add_plddt_mask:
            _add_plddt_mask(feats, self._dataset_cfg.min_plddt_threshold)
        else:
            feats['plddt_mask'] = torch.ones_like(feats['res_mask'])

        if self.task == 'hallucination':
            feats['diffuse_mask'] = torch.ones_like(feats['res_mask']).bool()
        elif self.task == 'inpainting':
            if self._dataset_cfg.inpainting_percent < random.random():
                feats['diffuse_mask'] = torch.ones_like(feats['res_mask'])    
            else:
                rng = self._rng if self.is_training else np.random.default_rng(seed=123)
                self.setup_inpainting(feats, rng)
                # Center based on motif locations
                motif_mask = 1 - feats['diffuse_mask']
                trans_1 = feats['trans_1']
                motif_1 = trans_1 * motif_mask[:, None]
                motif_com = torch.sum(motif_1, dim=0) / (torch.sum(motif_mask) + 1)
                trans_1 -= motif_com[None, :]
                feats['trans_1'] = trans_1
        else:
            raise ValueError(f'Unknown task {self.task}')
        feats['diffuse_mask'] = feats['diffuse_mask'].int()

        # Storing the csv index is helpful for debugging.
        feats['csv_idx'] = torch.ones(1, dtype=torch.long) * row_idx

        # Attach human-readable source name (kept as Python str; collate_fn will return list[str])
        try:
            if 'pdb_name' in csv_row:
                feats['source_name'] = str(csv_row['pdb_name'])
            elif 'name' in csv_row:
                feats['source_name'] = str(csv_row['name'])
        except Exception:
            pass
        return feats


class ScopeDataset(BaseDataset):

    def _filter_metadata(self, raw_csv):
        filter_cfg = self.dataset_cfg.filter
        data_csv = _length_filter(
            raw_csv,
            filter_cfg.min_num_res,
            filter_cfg.max_num_res
        )
        data_csv['oligomeric_detail'] = 'monomeric'
        return data_csv


class PdbDataset(BaseDataset):

    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
            is_predict=False
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self._is_predict = is_predict
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        self.FrameBuilder = FrameBuilder()

        # Process clusters
        self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path)
        if not is_predict:
            metadata_csv = self._filter_metadata(self.raw_csv)
            metadata_csv = metadata_csv.sort_values(
                'modeled_seq_len', ascending=False)
            self._pdb_to_cluster = _read_clusters(self._dataset_cfg.cluster_path)
            self._max_cluster = max(self._pdb_to_cluster.values())
        else:
            metadata_csv = self.raw_csv
            metadata_csv = metadata_csv.sort_values(
                'modeled_seq_len', ascending=False)



        self._missing_pdbs = 0

        self.map = build_elem_slot_maps()
        self.base_thickness = dataset_cfg.get('base_thickness', 0.5)

        def cluster_lookup(pdb):
            pdb = pdb.upper()
            if pdb not in self._pdb_to_cluster:
                self._pdb_to_cluster[pdb] = self._max_cluster + 1
                self._max_cluster += 1
                self._missing_pdbs += 1
            return self._pdb_to_cluster[pdb]
        metadata_csv['cluster'] = metadata_csv['pdb_name'].map(cluster_lookup)
        self._create_split(metadata_csv)
        self._all_clusters = dict(
            enumerate(self.csv['cluster'].unique().tolist()))
        self._num_clusters = len(self._all_clusters)

    def _filter_metadata(self, raw_csv):
        """Filter metadata."""
        filter_cfg = self.dataset_cfg.filter
        data_csv = raw_csv[
            raw_csv.oligomeric_detail.isin(filter_cfg.oligomeric)]
        data_csv = data_csv[
            data_csv.num_chains.isin(filter_cfg.num_chains)]
        data_csv = _length_filter(
            data_csv, filter_cfg.min_num_res, filter_cfg.max_num_res)
        data_csv = _max_coil_filter(data_csv, filter_cfg.max_coil_percent)
        data_csv = _rog_filter(data_csv, filter_cfg.rog_quantile)
        return data_csv
