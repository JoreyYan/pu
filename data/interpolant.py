from collections import defaultdict
import torch
from data import so3_utils
from data import utils as du
from scipy.spatial.transform import Rotation
from data import all_atom
import copy
from torch import autograd
from motif_scaffolding import twisting
import openfold.utils.rigid_utils as ru
import numpy as np
from models.loss import type_ce_loss,type_top1_acc,compute_CE_perplexity,huber,pairwise_distance_loss,backbone_mse_loss
import torch.nn.functional as F

def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _shs_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None, None, None, None] + trans_1 * (1 - diffuse_mask[..., None, None, None, None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )



def bert15_simple_mask(node_mask: torch.Tensor, mask_prob: float = 0.15, g=None):
    """
    Args:
        node_mask: [B, N] bool，True=有效结点，False=padding
        mask_prob: 被选中mask的比例
        g: torch.Generator (可选) 用来固定随机数

    Returns:
        mask:        [B, N] bool，True=被mask掉（可以乘0）
        update_mask: [B, N] bool，True=需要更新预测的目标位置
    """
    device = node_mask.device
    rand = torch.rand(node_mask.shape, generator=g, device=device)
    mask = (rand < mask_prob) & node_mask  # 只在有效节点里抽样
    update_mask = mask.clone()
    return mask, update_mask

def logit_normal_sample(n=1, m=0.0, s=1.0, device=None):
    """
    from simplefold
    """
    # Logit-Normal Sampling from https://arxiv.org/pdf/2403.03206.pdf
    if device is not None:
        u = torch.randn(n, device=device) * s + m
    else:
        u = torch.randn(n) * s + m
    t = 1 / (1 + torch.exp(-u))
    return t

def right_pad_dims_to(x, t):
    """Pad dimensions of t to match x"""
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.reshape(*t.shape, *((1,) * padding_dims))

class BasePath_ssq:
    """base class for flow matching path"""

    def __init__(self):
        return

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return None, None

    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return None, None

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        return d_alpha_t / alpha_t

    def compute_mu_t(self, t, x0, x1):
        """Compute the mean of time-dependent density p_t"""
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        """Sample xt from time-dependent density p_t; rng is required"""
        xt = self.compute_mu_t(t, x0, x1)
        return xt

    def compute_ut(self, t, x0, x1):
        """Compute the vector field corresponding to p_t"""
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def interpolant(self, t, x0, x1):
        t = right_pad_dims_to(x0, t)
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1)
        return t, xt, ut

    def compute_drift(self, x, t):
        """We always output sde according to score parametrization; """
        t = right_pad_dims_to(x, t)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift_mean = alpha_ratio * x
        drift_var = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
        return -drift_mean, drift_var

    def compute_score_from_velocity(self, v_t, y_t, t):
        t = right_pad_dims_to(y_t, t)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = y_t
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * v_t - mean) / var
        return score

    def compute_velocity_from_score(self, s_t, y_t, t):
        t = right_pad_dims_to(y_t, t)
        drift_mean, drift_var = self.compute_drift(y_t, t)
        velocity = -drift_mean + drift_var * s_t
        return velocity

class LinearPath_ssq(BasePath_ssq):
    """
    Linear flow process:
    x0: noise, x1: data
    In inference, we sample data from 0 -> 1
    """

    def __init__(self):
        super().__init__()

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return t, 1

    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return 1 - t, -1

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        return 1 / t

class Interpolant:

    def __init__(self, cfg,task,noise_scheme):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None
        self.task = task
        self.noise_scheme=noise_scheme
        self._path = LinearPath_ssq()

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    def sample_t_ssq(self, num_batch):
        """使用 logit-normal 采样，对齐 ml-simplefold 的时间分布"""
        t_size = num_batch
        t = 0.98 * logit_normal_sample(t_size, m=0.8, s=1.7, device=self._device) + 0.02 * torch.rand(t_size, device=self._device)
        t = t * (1 - 2 * self._cfg.min_t) + self._cfg.min_t
        return t

    def _corrupt_trans_ssq(self, trans_1, t, res_mask, diffuse_mask):
        """使用线性桥生成平移噪声和目标向量场"""
        # 生成噪声 x0
        trans_0 = _centered_gaussian(*res_mask.shape, self._device) * du.NM_TO_ANG_SCALE
        
        # 使用线性桥插值得到 x_t 和向量场 u_t
        _, trans_t, trans_v = self._path.interpolant(t[..., None], trans_0, trans_1)
        
        # 应用 diffuse mask
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        trans_v = _trans_diffuse_mask(trans_v, torch.zeros_like(trans_v), diffuse_mask)
        
        # 应用 res_mask
        trans_t = trans_t * res_mask[..., None]
        trans_v = trans_v * res_mask[..., None]
        trans_0 = trans_0 * res_mask[..., None]
        
        return trans_t, trans_0, trans_v

    def _corrupt_trans(self, trans_1, t, res_mask, diffuse_mask):
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * res_mask[..., None]

    def _corrupt_shs(self, sh_1, t, res_mask, diffuse_mask):
        sh_gaus_0=torch.randn_like(sh_1,device=self._device)
       # sh_gaus_0 = _centered_gaussian(*res_mask.shape, self._device)
        sh_0 = sh_gaus_0#.view((*res_mask.shape,4, -1))
        sh_t = (1 - t[...,None, None, None, None]) * sh_0 + t[..., None, None, None, None] * sh_1#.view((*res_mask.shape,4, -1))
        sh_t = _shs_diffuse_mask(sh_t,  sh_1, diffuse_mask)  #.view((*res_mask.shape,4, -1))

        return sh_t * res_mask[..., None, None, None, None]
    
    def _corrupt_rotmats(self, rotmats_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch*num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum(
            "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)

    def _corrupt_rotmats_ssq(self, rotmats_1, t, res_mask, diffuse_mask):
        """使用线性桥生成旋转噪声和目标向量场"""
        num_batch, num_res = res_mask.shape
        
        # 生成噪声旋转矩阵
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        
        # 使用线性插值得到旋转矩阵
        # 对于旋转矩阵，我们使用 Slerp 或 geodesic 插值
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_0, rotmats_1)
        
        # 计算旋转向量场（轴角表示）
        rot_v = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
        
        # 应用 res_mask
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        rotmats_0 = (
            rotmats_0 * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        rot_v = rot_v * res_mask[..., None]
        
        # 应用 diffuse mask
        rotmats_t = _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)
        rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_1, diffuse_mask)
        rot_v = rot_v * diffuse_mask[..., None]
        
        return rotmats_t, rotmats_0, rot_v

    def corrupt_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        # [B, N]
        res_mask = batch['res_mask']
        diffuse_mask = batch['diffuse_mask']
        num_batch, _ = diffuse_mask.shape

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]

        # Apply corruptions
        if self._trans_cfg.corrupt:
            trans_t = self._corrupt_trans(
                trans_1, t, res_mask, diffuse_mask)
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError('NaN in trans_t during corruption')
        noisy_batch['trans_t'] = trans_t


        if self._rots_cfg.corrupt:
            rotmats_t = self._corrupt_rotmats(
                rotmats_1, t, res_mask, diffuse_mask)
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError('NaN in rotmats_t during corruption')

        noisy_batch['rotmats_t'] =rotmats_t

        if self.task=='shdiffusion':
            sh_t=self._corrupt_shs( batch['normalize_density'], t, res_mask, diffuse_mask)*batch['density_mask']
            noisy_batch['fixed_mask']=res_mask
        noisy_batch['sh_t'] = sh_t
        noisy_batch['t']=t[:,0]
        noisy_batch['rigids_t']=du.create_rigid(rotmats_t,trans_t).to_tensor_7()

        # t = torch.tensor([self._cfg.min_t] * trans_1.shape[0], device=self._device).unsqueeze(-1)
        # noisy_batch['trans_0']=ru.identity_trans((trans_1.shape[0], trans_1.shape[1]), dtype=trans_1.dtype,device=self._device)
        # noisy_batch['rotmats_0'] = ru.identity_rot_mats((rotmats_1.shape[0], rotmats_1.shape[1]), dtype=rotmats_1.dtype,device=self._device)
        return noisy_batch

    def corrupt_batch_ssq(self, batch):
        """使用线性桥的加噪函数，对齐 ml-simplefold 的训练目标"""
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        # [B, N]
        res_mask = batch['res_mask']
        diffuse_mask = batch['diffuse_mask']
        num_batch, _ = diffuse_mask.shape

        # [B, 1] - 使用 logit-normal 采样
        t = self.sample_t_ssq(num_batch)[:, None]

        # Apply corruptions with linear bridge
        if self._trans_cfg.corrupt:
            trans_t, trans_0, trans_v = self._corrupt_trans_ssq(
                trans_1, t, res_mask, diffuse_mask)
        else:
            trans_t = trans_1
            trans_0 = torch.zeros_like(trans_1)
            trans_v = torch.zeros_like(trans_1)
        
        if torch.any(torch.isnan(trans_t)):
            raise ValueError('NaN in trans_t during corruption')
        
        noisy_batch['trans_t'] = trans_t
        noisy_batch['trans_0'] = trans_0
        noisy_batch['trans_v'] = trans_v

        if self._rots_cfg.corrupt:
            rotmats_t, rotmats_0, rot_v = self._corrupt_rotmats_ssq(
                rotmats_1, t, res_mask, diffuse_mask)
        else:
            rotmats_t = rotmats_1
            rotmats_0 = torch.eye(3, device=self._device).unsqueeze(0).unsqueeze(0).expand(*rotmats_1.shape)
            rot_v = torch.zeros_like(trans_1)
        
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError('NaN in rotmats_t during corruption')

        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['rotmats_0'] = rotmats_0
        noisy_batch['rot_v'] = rot_v

        # SH 处理（如果需要）
        if self.task == 'shdiffusion':
            sh_t = self._corrupt_shs(batch['normalize_density'], t, res_mask, diffuse_mask) * batch['density_mask']
            noisy_batch['fixed_mask'] = res_mask
            noisy_batch['sh_t'] = sh_t
        
        # 时间信息
        noisy_batch['t'] = t[:, 0]
        noisy_batch['r3_t'] = torch.full(res_mask.shape, t[:, 0], device=self._device, dtype=torch.float32)
        noisy_batch['so3_t'] = torch.full(res_mask.shape, t[:, 0], device=self._device, dtype=torch.float32)
        
        # 刚体信息
        noisy_batch['rigids_t'] = du.create_rigid(rotmats_t, trans_t).to_tensor_7()

        return noisy_batch
    def test_fbb_corrupt_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        # [B, N]
        res_mask = batch['res_mask']
        diffuse_mask = batch['diffuse_mask']
        num_batch, _ = diffuse_mask.shape

        # [B, 1]
        t = self.sample_t(num_batch)[:, None]*0



        sh_t = self._corrupt_shs(batch['normalize_density'], t, res_mask, diffuse_mask) * batch['density_mask']*0

        noisy_batch['SH_masked'] = sh_t
        noisy_batch['t'] = t[:, 0]
        noisy_batch['update_mask']=batch['res_mask']

        return noisy_batch
    def fbb_corrupt_batch(self, batch,prob=None):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = noisy_batch['trans_1']  # Angstrom
        node_mask=noisy_batch['res_mask']

        if prob is not None:
            mask_prob=prob
        else:
            mask_prob=np.random.uniform(0.15, 1.0)
        mask, update_mask = bert15_simple_mask(node_mask, mask_prob=mask_prob)
        noisy_batch['update_mask'] = update_mask

        # 基于实验配置的FBB噪声方案

        if self.noise_scheme == 'side_atoms':
            # 使用局部坐标中的标准正态随机坐标，对侧链原子做 (1-t)*noise0 + t*clean 的线性扰动
            B = node_mask.shape[0]
            t = self.sample_t(B)[:, None]  # [B,1]
            so3_t = t
            r3_t = t

            noisy_batch['so3_t']=so3_t
            noisy_batch['r3_t'] = r3_t

            noisy_batch['t'] = t[:, 0]
            if ('atoms14_local' in noisy_batch) and ('atom14_gt_exists' in noisy_batch):
                atoms14_local = noisy_batch['atoms14_local']         # [B,N,14,3]
                coord_scale = 8.0
                atom14_exists = noisy_batch['atom14_gt_exists'].bool() # [B,N,14]
                # 侧链掩码：索引 0..2 视为主链，3..13 为侧链
                sidechain_mask = torch.zeros_like(atom14_exists, dtype=torch.bool)
                sidechain_mask[..., 3:] = True
                # 仅在需要更新的位置扰动
                update_mask_exp = update_mask[..., None].expand_as(sidechain_mask).bool()
                effective_mask = sidechain_mask.bool() & atom14_exists.bool() & update_mask_exp
                # 标准正态噪声（局部坐标，缩放空间）
                noise0 = torch.randn_like(atoms14_local)
                noise0 = noise0 * effective_mask[..., None]
                # 线性插值到 t
                t_expand = t[..., None, None]  # [B,1,1]
                atoms14_local_scaled = atoms14_local #/ coord_scale
                interp = (1.0 - t_expand) * noise0*coord_scale + t_expand * atoms14_local_scaled
                atoms14_local_t = torch.where(
                    effective_mask.bool()[..., None], interp, atoms14_local_scaled
                )
                noisy_batch['atoms14_local_t'] = atoms14_local_t
                noisy_batch['sidechain_atom_mask'] = sidechain_mask
            else:
                pass
        elif self.noise_scheme in ('torision', 'torsion'):
            # 预留：扭转角方案后续实现
            pass
        elif self.noise_scheme == 'sh':
            # 将 SH 按照 update_mask 做 BERT15 风格的 mask
            SH_masked = batch['normalize_density'] * (1-mask)[..., None, None, None, None]
            batch['SH_masked'] = SH_masked

        return noisy_batch

    def fbb_corrupt_batch_backup(self, batch, prob=None):
        """旧版（未缩放、直接线性桥）侧链扰动逻辑。"""
        noisy_batch = copy.deepcopy(batch)

        # [B, N]
        node_mask = noisy_batch['res_mask']

        # 采样 mask 比例
        mask_prob = prob if prob is not None else np.random.uniform(0.15, 1.0)
        mask, update_mask = bert15_simple_mask(node_mask, mask_prob=mask_prob)
        noisy_batch['update_mask'] = update_mask

        if self.noise_scheme == 'side_atoms':
            B = node_mask.shape[0]
            t = self.sample_t(B)[:, None]  # [B,1]
            noisy_batch['so3_t'] = t
            noisy_batch['r3_t'] = t
            noisy_batch['t'] = t[:, 0]

            if ('atoms14_local' in noisy_batch) and ('atom14_gt_exists' in noisy_batch):
                atoms14_local = noisy_batch['atoms14_local']         # [B,N,14,3]
                #atom14_exists = noisy_batch['atom14_gt_exists'].bool()  # [B,N,14]

                # 侧链掩码 3..13
                sidechain_mask = torch.zeros_like(noisy_batch['atom14_gt_exists'].bool(), dtype=torch.bool)
                sidechain_mask[..., 3:] = True

                # 仅在有效+更新位置扰动
                update_mask_exp = update_mask[..., None].expand_as(sidechain_mask).bool()
                effective_mask = sidechain_mask  & update_mask_exp

                # 标准正态噪声
                noise0 = torch.randn_like(atoms14_local)
                noise0 = noise0 * effective_mask[..., None]

                # 线性桥插值到 t
                t_expand = t[..., None, None]  # [B,1,1]
                interp = (1.0 - t_expand) * noise0*8 + t_expand * atoms14_local
                atoms14_local_t = torch.where(effective_mask[..., None], interp, atoms14_local)

                noisy_batch['atoms14_local_t'] = atoms14_local_t
                noisy_batch['sidechain_atom_mask'] = sidechain_mask

        elif self.noise_scheme in ('torision', 'torsion'):
            pass
        elif self.noise_scheme == 'sh':
            SH_masked = batch['normalize_density'] * (1-mask)[..., None, None, None, None]
            batch['SH_masked'] = SH_masked

        return noisy_batch

    def fbb_prepare_batch(self, batch, update_mask: torch.Tensor | None = None):
        """Prepare inference batch for FBB without any time-dependent noise."""
        clean_batch = copy.deepcopy(batch)

        if update_mask is None:
            update_mask = clean_batch['res_mask'].bool()

        clean_batch['update_mask'] = update_mask.to(clean_batch['res_mask'].dtype)
        # Start from noise (t=min_t) for inference sampling
        min_t = self._cfg.min_t  # Should be close to 0
        clean_batch['t'] = torch.full(
            (clean_batch['res_mask'].shape[0],),
            min_t,
            device=clean_batch['res_mask'].device,
            dtype=torch.float32,
        )
        # Time grids for coordinates should be float32; do NOT inherit mask dtype
        clean_batch['r3_t'] = torch.full_like(
            clean_batch['res_mask'], min_t, dtype=torch.float32
        )
        clean_batch['so3_t'] = torch.full_like(
            clean_batch['res_mask'], min_t, dtype=torch.float32
        )

        if ('atoms14_local' in clean_batch) and ('atom14_gt_exists' in clean_batch):
            coord_scale = 8.0
            atoms14_local = clean_batch['atoms14_local']
            atom14_exists = clean_batch['atom14_gt_exists'].bool()
            exists_mask = atom14_exists[..., 3:].float()  # Only for sidechain atoms [B,N,11]

            # Initialize sidechain atoms with pure noise, keep backbone clean
            noise = torch.randn_like(atoms14_local)
            noise[..., :3, :] = 0.0  # Keep backbone atoms (0,1,2) clean
            
            # For inference, start with pure noise for sidechain atoms
            atoms14_local_scaled = atoms14_local
            atoms14_local_t = atoms14_local_scaled.clone()
            atoms14_local_t[..., 3:, :] = noise[..., 3:, :]*coord_scale * exists_mask[..., None]
            
            clean_batch['atoms14_local_t'] = atoms14_local_t
            clean_batch['sidechain_atom_mask'] = atom14_exists[..., 3:]

        return clean_batch


    def fbb_batch(self, batch,designchain=1):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom
        node_mask=batch['res_mask']

        # if prob is not None:
        #     mask_prob=prob
        # else:
        #     mask_prob=np.random.uniform(0.15, 1.0)
        mask=batch['chain_idx']-designchain
        update_mask=mask.clone()
        SH_masked = batch['normalize_density'] * (~mask)[..., None, None, None, None]  # 把mask位置变成0
        noisy_batch['SH_masked'] = SH_masked
        noisy_batch['update_mask'] = update_mask
        del noisy_batch['normalize_density']


        return noisy_batch
    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    def _trans_vector_field(self, t, trans_1, trans_t):
        return (trans_1 - trans_t) / (1 - t)

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        assert d_t > 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        return so3_utils.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)


    def pf_ode_step(self, dt, t,  trans_1, trans_t,eps=1e-5):
        """
        Stable PF-ODE step for R^3 with t: 1 -> 0 (so typically dt < 0).
        Matches your stable implementation's behavior but keeps the PF-ODE form.

        x_{t+dt} = x_t + [ f(x_t,t) - 0.5 * g^2(t) * score_hat * temp ] * dt
        where:
          f(x,t) = + x/t       (since dt < 0; equals -x/t * |dt| ),
          g^2(t) = (2 - 2*t)/t (your schedule; damped near t≈1),
          score_hat = (trans_t - t*trans_1) / (1 - t)^2   # NOTE: flipped sign vs. before
          temp = λ0 / (t^2 + (1 - t^2) * λ0)
        """
        # 1) clamp t for stability
        t = t.clamp(min=eps, max=1 - eps)

        # 2) your temperature reweighting
        lambda_0 = getattr(self._cfg, "temp", 1.0) or 1.0
        temp = lambda_0 / (t ** 2 + (1.0 - t ** 2) * lambda_0)

        # 3) drift f and schedule g^2 consistent with your stable code
        f = trans_t / t  # with dt<0 this equals (-x/t)*|dt|
        g2 = (2.0 - 2.0 * t) / t  # damped near t≈1, ~2/t near t≈0

        # 4) use score_hat = - previous score  ==> aligns with PF-ODE "minus" sign
        score_hat = (trans_t - t * trans_1) / (1.0 - t) ** 2

        # 5) PF-ODE drift (minus sign), but with your schedule/temperature
        drift = f - 0.5 * g2 * score_hat * temp

        # 6) Euler step
        x_next = trans_t + drift * dt

        # optional: numeric guards (clip huge steps, squash NaN/Inf)
        # x_next = torch.nan_to_num(x_next, nan=0.0, posinf=1e6, neginf=-1e6)

        return x_next,drift

    def _trans_euler_step_ssq(self, d_t, t, trans_t, pred_trans_v):
        """线性桥的欧拉步，使用预测的向量场"""
        # 使用预测的向量场进行欧拉步
        return trans_t + pred_trans_v * d_t

    def _rots_euler_step_ssq(self, d_t, t, rotmats_t, pred_rot_v):
        """线性桥的旋转欧拉步"""
        # 对于旋转，我们需要将轴角向量场转换为旋转矩阵更新
        # 这里使用简单的近似：小角度旋转
        angle = torch.norm(pred_rot_v, dim=-1, keepdim=True)
        axis = pred_rot_v / (angle + 1e-8)
        
        # 创建旋转增量
        delta_angle = angle * d_t
        cos_angle = torch.cos(delta_angle)
        sin_angle = torch.sin(delta_angle)
        
        # 轴角到旋转矩阵
        I = torch.eye(3, device=rotmats_t.device, dtype=rotmats_t.dtype)
        K = torch.zeros_like(I).unsqueeze(0).unsqueeze(0).expand_as(rotmats_t)
        K[..., 0, 1] = -axis[..., 2]
        K[..., 0, 2] = axis[..., 1]
        K[..., 1, 0] = axis[..., 2]
        K[..., 1, 2] = -axis[..., 0]
        K[..., 2, 0] = -axis[..., 1]
        K[..., 2, 1] = axis[..., 0]
        
        delta_rot = I.unsqueeze(0).unsqueeze(0) + sin_angle[..., None] * K + (1 - cos_angle)[..., None] * torch.bmm(K.view(-1, 3, 3), K.view(-1, 3, 3)).view_as(K)
        
        return torch.bmm(rotmats_t.view(-1, 3, 3), delta_rot.view(-1, 3, 3)).view_as(rotmats_t)

    def heun_step_R3(self, dt, t, trans_1, trans_t,  eps=1e-5):
        """
        Heun（改进欧拉）：
          1) 预测: x' = x + drift(x,t)*dt
          2) 校正: drift' = drift(x', t+dt)
          3) 合成: x_next = x + 0.5*(drift + drift')*dt
        """
        # predictor
        _,drift1 = self.pf_ode_step(dt,t, trans_1, trans_t, eps=eps)
        x_pred = trans_t + drift1 * dt

        # corrector
        t2 = (t + dt).clamp(min=eps, max=1 - eps)  # 反向时间
        _,drift2 = self.pf_ode_step(dt,t2, trans_1, x_pred, eps=eps)

        x_next = trans_t + 0.5 * (drift1 + drift2) * dt

        return x_next


    def loss(self,batch,loss_mask,side_atoms):
        # === 侧链坐标损失 ===
        coord_scale = 8.0
        side_gt_local = batch['atoms14_local'][..., 3:, :]  # [B,N,11,3]
        # 保证掩码是 bool
        exists11_bool = batch['atom14_gt_exists'][..., 3:].bool()  # [B,N,11]
        loss_mask_bool = loss_mask.bool()  # [B,N]

        # 逻辑与
        atom_level_mask = exists11_bool & loss_mask_bool[..., None]  # [B,N,11]

        # 转回 float 参与 loss
        atom_level_mask = atom_level_mask.to(side_gt_local.dtype)

        # SNR-aware scaling (counteract high-noise dominance):
        # scale ~ 1 / (1 - t_clip). Larger weights for late timesteps.
        t_clip = 0.9
        eps = 1e-6
        # 将预测与 GT 统一放回 Å 量纲后再做 SNR 权重与损失
        side_gt_ang = side_gt_local #* coord_scale
        side_pred_ang = side_atoms #* coord_scale

        if 'r3_t' in batch:
            # r3_t shape [B,N]; make [B,N,1,1] for broadcasting over [B,N,11,3]
            r3_t = batch['r3_t'].to(side_gt_local.dtype)
            r3_norm_scale = 1.0 - torch.clamp(r3_t, max=t_clip)
            r3_norm_scale = torch.clamp(r3_norm_scale, min=eps)[..., None, None]
            snr_scale = ( 1.0) / r3_norm_scale
            side_gt_scaled = side_gt_ang * snr_scale
            side_pred_scaled = side_pred_ang * snr_scale
        else:
            side_gt_scaled = side_gt_ang
            side_pred_scaled = side_pred_ang

        local_mse_loss = backbone_mse_loss(
            side_gt_scaled,
            side_pred_scaled,
            atom_level_mask,
            bb_atom_scale=1
        ).mean()

        local_pair_loss = pairwise_distance_loss(
            side_gt_scaled,
            side_pred_scaled.clone(),
            atom_level_mask,
            use_huber=False
        ).mean()

        local_huber_loss = huber(
            side_pred_scaled,
            side_gt_scaled,
            atom_level_mask
        ).mean()

        atomsloss = local_mse_loss + local_huber_loss + local_pair_loss
        # Debug: print loss components
        try:
            print(
                'atomsloss=', float(atomsloss.mean().detach().cpu()),
                'mse=', float(local_mse_loss.mean().detach().cpu()),
                'huber=', float(local_huber_loss.mean().detach().cpu()),
                'pair=', float(local_pair_loss.mean().detach().cpu())
            )
        except Exception:
            pass


    def sample(
            self,
            num_batch,
            num_res,
            model,
            num_timesteps=None,
            trans_potential=None,
            trans_0=None,
            rotmats_0=None,
            trans_1=None,
            rotmats_1=None,
            diffuse_mask=None,
            chain_idx=None,
            res_idx=None,
            verbose=False,
            backbone=None,
        ):
        res_mask = torch.ones(num_batch, num_res, device=self._device)

        # Set-up initial prior samples
        if trans_0 is None:
            trans_0 = _centered_gaussian(
                num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        if rotmats_0 is None:
            rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        if res_idx is None:
            res_idx = torch.arange(
                num_res,
                device=self._device,
                dtype=torch.float32)[None].repeat(num_batch, 1)
        batch = {
            'res_mask': res_mask,
            'diffuse_mask': res_mask,
            'res_idx': res_idx ,
            'chain_idx': chain_idx,
            'backbone':backbone
        }

        motif_scaffolding = False
        if diffuse_mask is not None and trans_1 is not None and rotmats_1 is not None:
            motif_scaffolding = True
            motif_mask = ~diffuse_mask.bool().squeeze(0)
        else:
            motif_mask = None
        if motif_scaffolding and not self._cfg.twisting.use: # amortisation
            diffuse_mask = diffuse_mask.expand(num_batch, -1) # shape = (B, num_residue)
            batch['diffuse_mask'] = diffuse_mask
            rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_1, diffuse_mask)
            trans_0 = _trans_diffuse_mask(trans_0, trans_1, diffuse_mask)
            if torch.isnan(trans_0).any():
                raise ValueError('NaN detected in trans_0')

        logs_traj = defaultdict(list)
        if motif_scaffolding and self._cfg.twisting.use: # sampling / guidance
            assert trans_1.shape[0] == 1 # assume only one motif
            motif_locations = torch.nonzero(motif_mask).squeeze().tolist()
            true_motif_locations, motif_segments_length = twisting.find_ranges_and_lengths(motif_locations)

            # Marginalise both rotation and motif location
            assert len(motif_mask.shape) == 1
            trans_motif = trans_1[:, motif_mask]  # [1, motif_res, 3]
            R_motif = rotmats_1[:, motif_mask]  # [1, motif_res, 3, 3]
            num_res = trans_1.shape[-2]
            with torch.inference_mode(False):
                motif_locations = true_motif_locations if self._cfg.twisting.motif_loc else None
                F, motif_locations = twisting.motif_offsets_and_rots_vec_F(num_res, motif_segments_length, motif_locations=motif_locations, num_rots=self._cfg.twisting.num_rots, align=self._cfg.twisting.align, scale=self._cfg.twisting.scale_rots, trans_motif=trans_motif, R_motif=R_motif, max_offsets=self._cfg.twisting.max_offsets, device=self._device, dtype=torch.float64, return_rots=False)

        if motif_mask is not None and len(motif_mask.shape) == 1:
            motif_mask = motif_mask[None].expand((num_batch, -1))

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for i, t_2 in enumerate(ts[1:]):
            if verbose: # and i % 1 == 0:
                print(f'{i=}, t={t_1.item():.2f}')
                print(torch.cuda.mem_get_info(trans_0.device), torch.cuda.memory_allocated(trans_0.device))
            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            if self._trans_cfg.corrupt:
                batch['trans_t'] = trans_t_1
            else:
                if trans_1 is None:
                    raise ValueError('Must provide trans_1 if not corrupting.')
                batch['trans_t'] = trans_1
            if self._rots_cfg.corrupt:
                batch['rotmats_t'] = rotmats_t_1
            else:
                if rotmats_1 is None:
                    raise ValueError('Must provide rotmats_1 if not corrupting.')
                batch['rotmats_t'] = rotmats_1
            batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['so3_t'] = batch['t']
            batch['r3_t'] = batch['t']
            d_t = t_2 - t_1

            use_twisting = motif_scaffolding and self._cfg.twisting.use and t_1 >= self._cfg.twisting.t_min

            if use_twisting: # Reconstruction guidance
                with torch.inference_mode(False):
                    batch, Log_delta_R, delta_x = twisting.perturbations_for_grad(batch)
                    model_out = model(batch)
                    t = batch['r3_t'] #TODO: different time for SO3?
                    trans_t_1, rotmats_t_1, logs_traj = self.guidance(trans_t_1, rotmats_t_1, model_out, motif_mask, R_motif, trans_motif, Log_delta_R, delta_x, t, d_t, logs_traj)

            else:
                with torch.no_grad():
                    model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            if self._cfg.self_condition:
                if motif_scaffolding:
                    batch['trans_sc'] = (
                        pred_trans_1 * diffuse_mask[..., None]
                        + trans_1 * (1 - diffuse_mask[..., None])
                    )
                else:
                    batch['trans_sc'] = pred_trans_1

            # Take reverse step
            
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            if trans_potential is not None:
                with torch.inference_mode(False):
                    grad_pred_trans_1 = pred_trans_1.clone().detach().requires_grad_(True)
                    pred_trans_potential = autograd.grad(outputs=trans_potential(grad_pred_trans_1), inputs=grad_pred_trans_1)[0]
                if self._trans_cfg.potential_t_scaling:
                    trans_t_2 -= t_1 / (1 - t_1) * pred_trans_potential * d_t
                else:
                    trans_t_2 -= pred_trans_potential * d_t
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)
            if motif_scaffolding and not self._cfg.twisting.use:
                trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_1, diffuse_mask)
                rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_1, diffuse_mask)

            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        if self._trans_cfg.corrupt:
            batch['trans_t'] = trans_t_1
        else:
            if trans_1 is None:
                raise ValueError('Must provide trans_1 if not corrupting.')
            batch['trans_t'] = trans_1
        if self._rots_cfg.corrupt:
            batch['rotmats_t'] = rotmats_t_1
        else:
            if rotmats_1 is None:
                raise ValueError('Must provide rotmats_1 if not corrupting.')
            batch['rotmats_t'] = rotmats_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        return atom37_traj, clean_atom37_traj, clean_traj

    @torch.no_grad()
    def sample_ssq(
            self,
            num_batch,
            num_res,
            model,
            num_timesteps=None,
            trans_potential=None,
            trans_0=None,
            rotmats_0=None,
            trans_1=None,
            rotmats_1=None,
            diffuse_mask=None,
            chain_idx=None,
            res_idx=None,
            verbose=False,
            backbone=None,
        ):
        """使用线性桥的采样函数，对齐 ml-simplefold 的采样流程"""
        res_mask = torch.ones(num_batch, num_res, device=self._device)

        # Set-up initial prior samples
        if trans_0 is None:
            trans_0 = _centered_gaussian(
                num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
        if rotmats_0 is None:
            rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        if res_idx is None:
            res_idx = torch.arange(
                num_res,
                device=self._device,
                dtype=torch.float32)[None].repeat(num_batch, 1)
        
        # 构建batch
        batch = {
            'res_mask': res_mask,
            'diffuse_mask': res_mask,
            'res_idx': res_idx,
            'chain_idx': chain_idx,
            'backbone': backbone
        }

        # 处理motif scaffolding
        motif_scaffolding = False
        if diffuse_mask is not None and trans_1 is not None and rotmats_1 is not None:
            motif_scaffolding = True
            motif_mask = ~diffuse_mask.bool().squeeze(0)
        else:
            motif_mask = None
            
        if motif_scaffolding and not self._cfg.twisting.use:
            diffuse_mask = diffuse_mask.expand(num_batch, -1)
            batch['diffuse_mask'] = diffuse_mask
            rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_1, diffuse_mask)
            trans_0 = _trans_diffuse_mask(trans_0, trans_1, diffuse_mask)

        if motif_mask is not None and len(motif_mask.shape) == 1:
            motif_mask = motif_mask[None].expand((num_batch, -1))

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        
        for i, t_2 in enumerate(ts[1:]):
            if verbose:
                print(f'{i=}, t={t_1.item():.2f}')
                
            # Run model
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            
            # 构建模型输入
            if self._trans_cfg.corrupt:
                batch['trans_t'] = trans_t_1
            else:
                if trans_1 is None:
                    raise ValueError('Must provide trans_1 if not corrupting.')
                batch['trans_t'] = trans_1
                
            if self._rots_cfg.corrupt:
                batch['rotmats_t'] = rotmats_t_1
            else:
                if rotmats_1 is None:
                    raise ValueError('Must provide rotmats_1 if not corrupting.')
                batch['rotmats_t'] = rotmats_1
                
            batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['so3_t'] = batch['t']
            batch['r3_t'] = batch['t']
            d_t = t_2 - t_1

            # 模型前向
            with torch.no_grad():
                model_out = model(batch)

            # 处理模型输出 - 获取预测的向量场
            if 'trans_vectorfield' in model_out:
                pred_trans_v = model_out['trans_vectorfield']
            else:
                # 如果没有直接输出向量场，从预测的干净坐标计算
                pred_trans_1 = model_out.get('pred_trans', trans_1)
                pred_trans_v = (pred_trans_1 - trans_t_1) / (1 - t_1 + 1e-8)
                
            if 'rot_vectorfield' in model_out:
                pred_rot_v = model_out['rot_vectorfield']
            else:
                # 如果没有直接输出旋转向量场，从预测的干净旋转计算
                pred_rotmats_1 = model_out.get('pred_rotmats', rotmats_1)
                pred_rot_v = so3_utils.calc_rot_vf(rotmats_t_1, pred_rotmats_1)

            # 记录预测的干净坐标
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )

            # 使用线性桥的欧拉步
            if self._trans_cfg.corrupt:
                trans_t_2 = self._trans_euler_step_ssq(d_t, t_1, trans_t_1, pred_trans_v)
            else:
                trans_t_2 = trans_t_1
                
            if self._rots_cfg.corrupt:
                rotmats_t_2 = self._rots_euler_step_ssq(d_t, t_1, rotmats_t_1, pred_rot_v)
            else:
                rotmats_t_2 = rotmats_t_1

            # 应用motif scaffolding
            if motif_scaffolding and not self._cfg.twisting.use:
                trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_1, diffuse_mask)
                rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_1, diffuse_mask)

            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # Final step - 在最后一个时间步运行模型
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        
        if self._trans_cfg.corrupt:
            batch['trans_t'] = trans_t_1
        else:
            if trans_1 is None:
                raise ValueError('Must provide trans_1 if not corrupting.')
            batch['trans_t'] = trans_1
            
        if self._rots_cfg.corrupt:
            batch['rotmats_t'] = rotmats_t_1
        else:
            if rotmats_1 is None:
                raise ValueError('Must provide rotmats_1 if not corrupting.')
            batch['rotmats_t'] = rotmats_1
            
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        
        with torch.no_grad():
            model_out = model(batch)
            
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)
        return atom37_traj, clean_atom37_traj, clean_traj

    @torch.no_grad()
    def fbb_sample_iterative(
        self,
        batch: dict,
        model,
        num_timesteps: int | None = None,
    ):
        """Iterative sidechain sampling in local frame (diffusion-style ODE).

        Fixed across steps: res_mask, diffuse_mask, res_idx, chain_idx, trans_1, rotmats_1.
        Evolving: atoms14_local_t (only sidechain indices 3: are updated each step).
        """
        device = batch['res_mask'].device
        B, N = batch['res_mask'].shape

        res_mask = batch['res_mask']
        diffuse_mask = batch.get('diffuse_mask', torch.ones_like(res_mask))
        res_idx = batch['res_idx']
        chain_idx = batch['chain_idx']
        rotmats_1 = batch['rotmats_1']
        trans_1 = batch['trans_1']

        atoms14_local_gt = batch['atoms14_local']  # [B,N,14,3]


        side_exists = batch.get('atom14_gt_exists', torch.ones_like(atoms14_local_gt[..., 0]))[..., 3:]  # [B,N,11]

        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps, device=device)

        # init xt (sidechain local) at t = min_t with Gaussian noise


        # Prepare base features using the dedicated function
        input_feats_base = copy.deepcopy(batch)
        backbone_local=input_feats_base['atoms14_local_t'][..., :3, :]
        xt=input_feats_base['atoms14_local_t'][..., 3:, :]
        input_feats_base['atoms14_local_sc']=input_feats_base['atoms14_local_t']

        logs = []

        for i in range(len(ts) - 1):
            t1 = ts[i]
            t2 = ts[i + 1]
            dt = (t2 - t1).clamp_min(1e-6)

            # Update atoms14_local_t for current step


            atoms14_local_t = torch.cat([backbone_local, xt], dim=-2)

            
            # Use base features and only update time-dependent and coordinate fields
            input_feats = input_feats_base.copy()
            input_feats.update({
                't': torch.full((res_mask.shape[0],), float(t1), device=device, dtype=torch.float32),
                'r3_t': torch.full(res_mask.shape, float(t1), device=device, dtype=torch.float32),
                'so3_t': torch.full(res_mask.shape, float(t1), device=device, dtype=torch.float32),
                'atoms14_local_t': atoms14_local_t,
            })

            out = model(input_feats)
            x1_pred = out['side_atoms']  # [B,N,11,3] predicted clean sidechains (local)
            input_feats_base['atoms14_local_sc']=torch.cat([backbone_local, x1_pred], dim=-2)


            recovery = type_top1_acc(out['logits'], batch['aatype'], node_mask=res_mask)
            _, perplexity = compute_CE_perplexity(out['logits'], batch['aatype'], mask=res_mask)
            #print(f'recovery: {recovery}, perplexity: {perplexity}')


            # Use the standard translational Euler step helper for sidechain coords
            # Here, treat x1_pred as the clean target ("trans_1") and xt as current ("trans_t")

            xt2 = self._trans_euler_step(float(dt), float(t1), x1_pred, xt)

          #  xt2 = self.heun_step_R3( dt, t1, x1_pred, xt)

            #xt2true = (1.0 - t2) * x0 + t2 * xgt
            xt = xt2

           # self.loss(batch,res_mask,x1_pred)


            # ===== 诊断：一步 (t1 -> t2) 的误差分解与量化 =====
            # with torch.no_grad():
            #     eps = 1e-8
            #
            #     # 可选掩码（侧链存在 & 残基有效 & 参与扩散）
            #     side_mask = side_exists  # [B,N,11]
            #     if 'diffuse_mask' in batch:
            #         side_mask = side_mask * batch['diffuse_mask'][..., None]
            #     side_mask = side_mask * res_mask[..., None]  # [B,N,11]
            #     side_mask_f = side_mask.float()[..., None]  # [B,N,11,1]
            #
            #     # 核心量
            #     delta = xt2 - xt2true  # 你的一步结果 与 参考桥结果 的差
            #     step_true = xt2true - xt  # 桥上的“正确一步”增量
            #     step_pred = xt2 - xt  # 你的一步“预测增量”（欧拉或当前规则）
            #
            #     # 向量范数（每个原子）
            #     norm_delta = torch.linalg.norm(delta, dim=-1)  # [B,N,11]
            #     norm_true_step = torch.linalg.norm(step_true, dim=-1)  # [B,N,11]
            #     norm_pred_step = torch.linalg.norm(step_pred, dim=-1)  # [B,N,11]
            #
            #     # 相对误差：||xt2 - xt2true|| / ||xt2true - xt||
            #     rel_err = norm_delta / (norm_true_step + eps)  # [B,N,11]
            #
            #     # 方向相似度：cos(theta) between predicted step and true step
            #     dot_pred_true = (step_pred * step_true).sum(dim=-1)  # [B,N,11]
            #     cos_theta = dot_pred_true / (norm_pred_step * norm_true_step + eps)
            #
            #     # 误差的“沿桥/离桥”分解：以 step_true 的方向作为桥切向
            #     # 先取单位向量 u_true = step_true / ||step_true||
            #     u_true = step_true / (norm_true_step[..., None] + eps)
            #     # 平行分量投影系数（标量）与向量
            #     delta_parallel_coeff = (delta * u_true).sum(dim=-1, keepdim=True)  # [B,N,11,1]
            #     delta_parallel_vec = delta_parallel_coeff * u_true  # [B,N,11,3]
            #     # 垂直分量向量
            #     delta_perp_vec = delta - delta_parallel_vec  # [B,N,11,3]
            #
            #     # 对应的范数
            #     norm_delta_parallel = torch.linalg.norm(delta_parallel_vec, dim=-1)  # [B,N,11]
            #     norm_delta_perp = torch.linalg.norm(delta_perp_vec, dim=-1)  # [B,N,11]
            #
            #     # 掩码平均（只统计有效侧链原子）
            #     def masked_mean(x):
            #         num = (side_mask_f > 0.5).float().sum()
            #         return (x[..., None] * side_mask_f).sum() / (num + eps)
            #
            #     # 汇总指标（标量）
            #     rms_delta = masked_mean(norm_delta)  # 一步总误差均值 (Å)
            #     rms_true_step = masked_mean(norm_true_step)  # 桥上正确一步的均值 (Å)
            #     rms_rel_err = masked_mean(rel_err)  # 相对误差均值
            #     rms_delta_parallel = masked_mean(norm_delta_parallel)  # 沿桥方向的误差均值 (Å)
            #     rms_delta_perp = masked_mean(norm_delta_perp)  # 离桥（法向）误差均值 (Å)
            #     mean_cos_theta = masked_mean(cos_theta)  # 方向余弦均值（越接近 1 越好）
            #
            #     # 也给几个分位数帮助判断尾部
            #     def masked_percentile(x, q):
            #         # 返回全局分位数（忽略 batch 结构）
            #         x_flat = x[side_mask > 0.5].reshape(-1)
            #         if x_flat.numel() == 0:
            #             return torch.tensor(float('nan'), device=x.device)
            #         k = max(1, int((q / 100.0) * x_flat.numel()) - 1)
            #         vals, _ = torch.sort(x_flat)
            #         return vals[k]
            #
            #     p50_rel = masked_percentile(rel_err, 50)
            #     p90_rel = masked_percentile(rel_err, 90)
            #
            #     # 打印
            #     print(f"[STEP DIAG] t1={float(t1):.4f} -> t2={float(t2):.4f}")
            #     print(f"  |Δ| mean (Å):        {float(rms_delta):.4f}")
            #     print(f"  |Δ_true| mean (Å):   {float(rms_true_step):.4f}")
            #     print(
            #         f"  relErr mean:         {float(rms_rel_err):.4f}  (p50={float(p50_rel):.4f}, p90={float(p90_rel):.4f})")
            #     print(f"  ||Δ_parallel|| mean: {float(rms_delta_parallel):.4f}")
            #     print(f"  ||Δ_perp|| mean:     {float(rms_delta_perp):.4f}")
            #     print(f"  cos(pred,true) mean: {float(mean_cos_theta):.4f}")



        # Final step aligned with structure sample(): run one more model call at t_final
        t_final = ts[-1]
        atoms14_local_t = torch.cat([backbone_local, xt], dim=-2)
        input_feats_final = input_feats_base.copy()
        input_feats_final.update({
            't': torch.full((res_mask.shape[0],), float(t_final), device=device, dtype=torch.float32),
            'r3_t': torch.full(res_mask.shape, float(t_final), device=device, dtype=torch.float32),
            'so3_t': torch.full(res_mask.shape, float(t_final), device=device, dtype=torch.float32),
            'atoms14_local_t': atoms14_local_t,
        })

        with torch.no_grad():
            out_final = model(input_feats_final)
        side_atoms_final = out_final['side_atoms']  # use model's clean prediction at t_final
        final_logits = out_final.get('logits', None)

        # Compose final local14 with predicted clean sidechains
        atoms14_local_final = torch.cat([backbone_local, side_atoms_final], dim=-2)
        if side_exists is not None:
            atoms14_local_final[..., 3:, :] = atoms14_local_final[..., 3:, :] * side_exists[..., None]

        # Build global 14 using fixed frames
        rigid = du.create_rigid(rotmats_1, trans_1)
        atoms14_global_final = rigid[..., None].apply(atoms14_local_final)

        return {
            'atoms14_local_final': atoms14_local_final,
            'atoms14_global_final': atoms14_global_final,
            'logits_final': final_logits,
        }

    @torch.no_grad()
    def fbb_sample_iterative_ssq(
        self,
        batch: dict,
        model,
        num_timesteps: int | None = None,
    ):
        """使用线性桥的迭代侧链采样函数"""
        device = batch['res_mask'].device
        B, N = batch['res_mask'].shape

        res_mask = batch['res_mask']
        diffuse_mask = batch.get('diffuse_mask', torch.ones_like(res_mask))
        res_idx = batch['res_idx']
        chain_idx = batch['chain_idx']
        rotmats_1 = batch['rotmats_1']
        trans_1 = batch['trans_1']

        atoms14_local_gt = batch['atoms14_local']  # [B,N,14,3]
        side_exists = batch.get('atom14_gt_exists', torch.ones_like(atoms14_local_gt[..., 0]))[..., 3:]  # [B,N,11]

        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps, device=device)

        # Prepare base features
        input_feats_base = copy.deepcopy(batch)
        backbone_local = input_feats_base['atoms14_local_t'][..., :3, :]
        xt = input_feats_base['atoms14_local_t'][..., 3:, :]
        input_feats_base['atoms14_local_sc'] = input_feats_base['atoms14_local_t']

        logs = []

        for i in range(len(ts) - 1):
            t1 = ts[i]
            t2 = ts[i + 1]
            dt = (t2 - t1).clamp_min(1e-6)

            # Update atoms14_local_t for current step
            atoms14_local_t = torch.cat([backbone_local, xt], dim=-2)

            # Use base features and only update time-dependent and coordinate fields
            input_feats = input_feats_base.copy()
            input_feats.update({
                't': torch.full((res_mask.shape[0],), float(t1), device=device, dtype=torch.float32),
                'r3_t': torch.full(res_mask.shape, float(t1), device=device, dtype=torch.float32),
                'so3_t': torch.full(res_mask.shape, float(t1), device=device, dtype=torch.float32),
                'atoms14_local_t': atoms14_local_t,
            })

            out = model(input_feats)
            x1_pred = out['side_atoms']  # [B,N,11,3] predicted clean sidechains (local)
            
            # 计算预测的向量场
            pred_side_v = (x1_pred - xt) / (1 - t1 + 1e-8)
            
            # 使用线性桥的欧拉步
            xt2 = self._trans_euler_step_ssq(float(dt), float(t1), xt, pred_side_v)
            
            # 应用mask
            xt2 = xt2 * side_exists[..., None]
            xt = xt2

            # 更新input_feats_base
            input_feats_base['atoms14_local_sc'] = torch.cat([backbone_local, x1_pred], dim=-2)

            # 记录日志
            logs.append({
                't': float(t1),
                'side_atoms_pred': x1_pred.detach().cpu(),
                'side_atoms_current': xt.detach().cpu(),
            })

        # Final step
        atoms14_local_final = torch.cat([backbone_local, xt], dim=-2)
        input_feats_final = input_feats_base.copy()
        input_feats_final.update({
            't': torch.full((res_mask.shape[0],), float(ts[-1]), device=device, dtype=torch.float32),
            'r3_t': torch.full(res_mask.shape, float(ts[-1]), device=device, dtype=torch.float32),
            'so3_t': torch.full(res_mask.shape, float(ts[-1]), device=device, dtype=torch.float32),
            'atoms14_local_t': atoms14_local_final,
        })

        final_out = model(input_feats_final)
        final_side_atoms = final_out['side_atoms']
        final_logits = final_out['logits']
        
        atoms14_local_final = torch.cat([backbone_local, final_side_atoms], dim=-2)

        # Convert to global coordinates
        rigid = du.create_rigid(rotmats_1, trans_1)
        atoms14_global_final = rigid[..., None].apply(atoms14_local_final)

        return {
            'atoms14_local_final': atoms14_local_final,
            'atoms14_global_final': atoms14_global_final,
            'logits_final': final_logits,
            'logs': logs,
        }

    @torch.no_grad()
    def fbb_sample_single(
        self,
        batch: dict,
        model,
        t_eval: float = 0,
    ):
        """Single-step FBB sampling (no iteration).

        - Initializes sidechains once (noise at min_t via fbb_prepare_batch)
        - Sets time condition to t_eval (default 1.0)
        - Runs model a single time to predict clean sidechains
        - Composes final local/global 14-atom coordinates
        """

        t_eval=1.0

        device = batch['res_mask'].device

        res_mask = batch['res_mask']
        rotmats_1 = batch['rotmats_1']
        trans_1 = batch['trans_1']
        atoms14_local_gt = batch['atoms14_local']  # [B,N,14,3]
        backbone_local = atoms14_local_gt[..., :3, :]
        side_exists = batch.get('atom14_gt_exists', torch.ones_like(atoms14_local_gt[..., 0]))[..., 3:]

        # Prepare base features (provides atoms14_local_t at min_t noise)
        base_batch = copy.deepcopy(batch)
        base_feats = self.fbb_prepare_batch(base_batch)

        # Build model inputs at t_eval
        atoms14_local_t = base_feats['atoms14_local_t']
        input_feats = base_feats.copy()
        input_feats.update({
            't': torch.full((res_mask.shape[0],), float(t_eval), device=device, dtype=torch.float32),
            'r3_t': torch.full(res_mask.shape, float(t_eval), device=device, dtype=torch.float32),
            'so3_t': torch.full(res_mask.shape, float(t_eval), device=device, dtype=torch.float32),
            'atoms14_local_t': atoms14_local_t,
        })

        out = model(input_feats)
        side_atoms = out['side_atoms']  # [B,N,11,3]
        final_logits = out.get('logits', None)

        # Compose final local 14 and apply existence mask
        atoms14_local_final = torch.cat([backbone_local, side_atoms], dim=-2)
        if side_exists is not None:
            atoms14_local_final[..., 3:, :] = atoms14_local_final[..., 3:, :] * side_exists[..., None]

        # Global coords via fixed frames
        rigid = du.create_rigid(rotmats_1, trans_1)
        atoms14_global_final = rigid[..., None].apply(atoms14_local_final)

        return {
            'atoms14_local_final': atoms14_local_final,
            'atoms14_global_final': atoms14_global_final,
            'logits_final': final_logits,
        }

    @torch.no_grad()
    def fbb_sample_single_ssq(
        self,
        batch: dict,
        model,
        t_eval: float = 1.0,
    ):
        """使用线性桥的单步FBB采样函数"""
        device = batch['res_mask'].device

        res_mask = batch['res_mask']
        rotmats_1 = batch['rotmats_1']
        trans_1 = batch['trans_1']
        atoms14_local_gt = batch['atoms14_local']  # [B,N,14,3]
        backbone_local = atoms14_local_gt[..., :3, :]
        side_exists = batch.get('atom14_gt_exists', torch.ones_like(atoms14_local_gt[..., 0]))[..., 3:]

        # Prepare base features (provides atoms14_local_t at min_t noise)
        base_batch = copy.deepcopy(batch)
        base_feats = self.fbb_prepare_batch(base_batch)

        # Build model inputs at t_eval
        atoms14_local_t = base_feats['atoms14_local_t']
        input_feats = base_feats.copy()
        input_feats.update({
            't': torch.full((res_mask.shape[0],), float(t_eval), device=device, dtype=torch.float32),
            'r3_t': torch.full(res_mask.shape, float(t_eval), device=device, dtype=torch.float32),
            'so3_t': torch.full(res_mask.shape, float(t_eval), device=device, dtype=torch.float32),
            'atoms14_local_t': atoms14_local_t,
        })

        out = model(input_feats)
        
        # 获取预测的侧链原子和向量场
        if 'side_vectorfield' in out:
            pred_side_v = out['side_vectorfield']
        else:
            # 如果没有直接输出向量场，从预测的干净坐标计算
            side_atoms = out['side_atoms']  # [B,N,11,3]
            current_side = atoms14_local_t[..., 3:, :]  # 当前的侧链坐标
            pred_side_v = (side_atoms - current_side) / (1 - t_eval + 1e-8)
        
        side_atoms = out['side_atoms']  # [B,N,11,3]
        final_logits = out.get('logits', None)

        # 使用线性桥重建最终坐标
        # x_clean = x_t + (1-t) * v_t
        atoms14_local_clean = atoms14_local_t.clone()
        atoms14_local_clean[..., 3:, :] = current_side + (1 - t_eval) * pred_side_v

        # 应用存在mask
        if side_exists is not None:
            atoms14_local_clean[..., 3:, :] = atoms14_local_clean[..., 3:, :] * side_exists[..., None]

        # Global coords via fixed frames
        rigid = du.create_rigid(rotmats_1, trans_1)
        atoms14_global_final = rigid[..., None].apply(atoms14_local_clean)

        return {
            'atoms14_local_final': atoms14_local_clean,
            'atoms14_global_final': atoms14_global_final,
            'logits_final': final_logits,
            'pred_vectorfield': pred_side_v,
        }

    @torch.no_grad()
    def fbb_sample_consistent(
        self,
        batch: dict,
        model,
        num_timesteps: int | None = None,
    ):
        """DDIM/consistency-style sampling with per-step reprojection to the training bridge.

        x_t is updated by: x_t <- (1 - t) * noise0 + t * x1_pred, where noise0 is fixed per run.
        Only sidechain atoms (indices 3:) are evolved; backbone (0..2) is kept from input.
        """
        device = batch['res_mask'].device
        res_mask = batch['res_mask']
        rotmats_1 = batch['rotmats_1']
        trans_1 = batch['trans_1']

        atoms14_local_gt = batch['atoms14_local']  # [B,N,14,3]
        backbone_local = atoms14_local_gt[..., :3, :]
        side_exists = batch.get('atom14_gt_exists', torch.ones_like(atoms14_local_gt[..., 0]))[..., 3:]

        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps, device=device)

        # Fixed noise reference for the whole trajectory
        noise0 = torch.randn_like(atoms14_local_gt[..., 3:, :])  # [B,N,11,3]

        # Initialize at t0
        t0 = ts[0]
        xt = (1.0 - t0) * noise0 + t0 * atoms14_local_gt[..., 3:, :]

        # Base features from helper; will override time/coords each step
        base_batch = copy.deepcopy(batch)
        base_batch['atoms14_local_t'] = torch.cat([backbone_local, xt], dim=-2)
        input_feats_base = self.fbb_prepare_batch(base_batch)

        final_logits = None
        for i in range(len(ts) - 1):
            t1 = ts[i]
            t2 = ts[i + 1]

            atoms14_local_t = torch.cat([backbone_local, xt], dim=-2)
            input_feats = input_feats_base.copy()
            input_feats.update({
                't': torch.full((res_mask.shape[0],), float(t1), device=device, dtype=torch.float32),
                'r3_t': torch.full(res_mask.shape, float(t1), device=device, dtype=torch.float32),
                'so3_t': torch.full(res_mask.shape, float(t1), device=device, dtype=torch.float32),
                'atoms14_local_t': atoms14_local_t,
            })

            out = model(input_feats)
            x1_pred = out['side_atoms']  # [B,N,11,3]
            final_logits = out.get('logits', final_logits)

            # Reproject to training bridge at t2
            xt = (1.0 - t2) * noise0 + t2 * x1_pred

        atoms14_local_final = torch.cat([backbone_local, xt], dim=-2)
        if side_exists is not None:
            atoms14_local_final[..., 3:, :] = atoms14_local_final[..., 3:, :] * side_exists[..., None]

        rigid = du.create_rigid(rotmats_1, trans_1)
        atoms14_global_final = rigid[..., None].apply(atoms14_local_final)

        return {
            'atoms14_local_final': atoms14_local_final,
            'atoms14_global_final': atoms14_global_final,
            'logits_final': final_logits,
        }

    @torch.no_grad()
    def fbb_sample_consistent_ssq(
        self,
        batch: dict,
        model,
        num_timesteps: int | None = None,
    ):
        """使用线性桥的一致性采样函数 (DDIM-style)"""
        device = batch['res_mask'].device
        res_mask = batch['res_mask']
        rotmats_1 = batch['rotmats_1']
        trans_1 = batch['trans_1']

        atoms14_local_gt = batch['atoms14_local']  # [B,N,14,3]
        backbone_local = atoms14_local_gt[..., :3, :]
        side_exists = batch.get('atom14_gt_exists', torch.ones_like(atoms14_local_gt[..., 0]))[..., 3:]

        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps, device=device)

        # 固定噪声参考，用于整个轨迹
        noise0 = torch.randn_like(atoms14_local_gt[..., 3:, :])  # [B,N,11,3]

        # 在t0初始化
        t0 = ts[0]
        xt = (1.0 - t0) * noise0 + t0 * atoms14_local_gt[..., 3:, :]

        # 基础特征
        base_batch = copy.deepcopy(batch)
        base_batch['atoms14_local_t'] = torch.cat([backbone_local, xt], dim=-2)
        input_feats_base = self.fbb_prepare_batch(base_batch)

        final_logits = None
        for i in range(len(ts) - 1):
            t1 = ts[i]
            t2 = ts[i + 1]

            atoms14_local_t = torch.cat([backbone_local, xt], dim=-2)
            input_feats = input_feats_base.copy()
            input_feats.update({
                't': torch.full((res_mask.shape[0],), float(t1), device=device, dtype=torch.float32),
                'r3_t': torch.full(res_mask.shape, float(t1), device=device, dtype=torch.float32),
                'so3_t': torch.full(res_mask.shape, float(t1), device=device, dtype=torch.float32),
                'atoms14_local_t': atoms14_local_t,
            })

            out = model(input_feats)
            x1_pred = out['side_atoms']  # [B,N,11,3]
            final_logits = out.get('logits', None)

            # 使用线性桥的一致性更新
            # x_t2 = (1 - t2) * x0 + t2 * x1_pred
            xt = (1.0 - t2) * noise0 + t2 * x1_pred
            
            # 应用存在mask
            xt = xt * side_exists[..., None]

        # 最终步
        atoms14_local_final = torch.cat([backbone_local, xt], dim=-2)

        # 转换为全局坐标
        rigid = du.create_rigid(rotmats_1, trans_1)
        atoms14_global_final = rigid[..., None].apply(atoms14_local_final)

        return {
            'atoms14_local_final': atoms14_local_final,
            'atoms14_global_final': atoms14_global_final,
            'logits_final': final_logits,
        }

    # ===== Stable iterative sampling helpers (alpha-cap, Heun, displacement cap) =====
    def _alpha_raw(self, dt, t):
        """Compute raw alpha = dt / (1 - t) in float domain."""
        return float(dt) / max(1e-6, 1.0 - float(t))

    def _apply_alpha_cap(self, dt, t, alpha_max):
        """Cap alpha to alpha_max by shrinking effective dt (trust-region style)."""
        if alpha_max is None:
            return float(dt)
        alpha = self._alpha_raw(dt, t)
        if alpha <= alpha_max:
            return float(dt)
        # dt_eff = alpha_max * (1 - t)
        return float(alpha_max) * max(1e-6, 1.0 - float(t))

    @torch.no_grad()
    def _heun_step_R3(
        self,
        t1,
        t_next,
        xt,
        x1_pred_t1,
        model,
        input_feats_base,
        backbone_local,
        res_mask,
    ):
        """Heun predictor-corrector for sidechain R3.
        Vector field v(t,x) = (x1_pred - x) / (1 - t)
        xt2 = xt + 0.5 * (v1 + v2) * dt
        """
        device = xt.device
        # use effective next time (after alpha-cap)
        dt = float(t_next) - float(t1)
        dt = max(1e-6, dt)
        # v1 at (t1, xt)
        v1 = (x1_pred_t1 - xt) / max(1e-6, (1.0 - float(t1)))
        xt_pred = xt + v1 * float(dt)

        # second eval at (t_next, xt_pred)
        atoms14_local_t2 = torch.cat([backbone_local, xt_pred], dim=-2)
        input_feats_t2 = input_feats_base.copy()
        t_next_tensor_b = torch.full((res_mask.shape[0],), float(t_next), device=device, dtype=torch.float32)
        t_next_tensor_bn = torch.full(res_mask.shape, float(t_next), device=device, dtype=torch.float32)
        input_feats_t2.update({
            't':     t_next_tensor_b,
            'r3_t':  t_next_tensor_bn,
            'so3_t': t_next_tensor_bn,
            'atoms14_local_t': atoms14_local_t2,
        })
        out_t2 = model(input_feats_t2)
        x1_pred_t2 = out_t2['side_atoms']
        v2 = (x1_pred_t2 - xt_pred) / max(1e-6, (1.0 - float(t_next)))

        xt_heun = xt + 0.5 * (v1 + v2) * dt
        return xt_heun, out_t2

    def _displacement_cap(self, xt_old, xt_new, ang_cap=0.8):
        """Per-atom displacement cap (in Angstrom) to avoid overshoot."""
        disp = xt_new - xt_old
        # L2 norm over 3D coord
        norm = torch.linalg.norm(disp, dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(ang_cap / norm, max=1.0)
        return xt_old + disp * scale

    @torch.no_grad()
    def fbb_sample_iterative_stable(
        self,
        batch: dict,
        model,
        num_timesteps: int | None = None,
    ):
        """Iterative sampling with alpha-cap, optional Heun corrector, displacement cap, and early bridge mixing.

        This variant keeps the original API but stabilizes multi-step updates to mitigate exposure bias.
        """
        device = batch['res_mask'].device

        res_mask = batch['res_mask']
        res_idx = batch['res_idx']
        chain_idx = batch['chain_idx']
        rotmats_1 = batch['rotmats_1']
        trans_1 = batch['trans_1']
        atoms14_local_gt = batch['atoms14_local']  # [B,N,14,3]
        backbone_local = atoms14_local_gt[..., :3, :]
        side_exists = batch.get('atom14_gt_exists', torch.ones_like(atoms14_local_gt[..., 0]))[..., 3:].bool()

        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps, device=device)

        # init xt at t = min_t with Gaussian noise; keep anchors
        xt = torch.randn_like(atoms14_local_gt[..., 3:, :])
        xt = xt * side_exists[..., None]
        x0 = xt.detach().clone()  # fixed noise anchor
        xgt = atoms14_local_gt[..., 3:, :]

        # Base features for static fields
        base_batch = copy.deepcopy(batch)
        base_batch['atoms14_local_t'] = torch.cat([backbone_local, xt], dim=-2)
        input_feats_base = self.fbb_prepare_batch(base_batch)

        # Hyperparameters with defaults from sampling cfg
        alpha_max      = getattr(self._sample_cfg, 'alpha_max', 0.5)
        heun_steps     = getattr(self._sample_cfg, 'heun_steps', 10)
        bridge_gamma_K = getattr(self._sample_cfg, 'bridge_gamma_steps', 2)
        bridge_gamma0  = getattr(self._sample_cfg, 'bridge_gamma0', 0.7)
        disp_cap_ang   = getattr(self._sample_cfg, 'disp_cap_ang', 0.8)

        final_logits = None
        # use current time that progresses by dt_eff, instead of jumping to grid
        i = 0
        t1 = ts[0]
        while (i < len(ts) - 1) and (float(t1) < 1.0 - 1e-6):
            t_grid_next = ts[i + 1]
            dt_raw = (t_grid_next - t1).clamp_min(1e-6)
            # alpha-cap shrink dt to control along-bridge overshoot
            dt_eff = self._apply_alpha_cap(dt_raw, t1, alpha_max)
            t_next = float(t1) + float(dt_eff)

            # Assemble inputs at t1
            atoms14_local_t = torch.cat([backbone_local, xt], dim=-2)
            input_feats = input_feats_base.copy()
            input_feats.update({
                't':     torch.full((res_mask.shape[0],), float(t1), device=device, dtype=torch.float32),
                'r3_t':  torch.full(res_mask.shape, float(t1), device=device, dtype=torch.float32),
                'so3_t': torch.full(res_mask.shape, float(t1), device=device, dtype=torch.float32),
                'atoms14_local_t': atoms14_local_t,
            })

            out = model(input_feats)
            x1_pred = out['side_atoms']
            final_logits = out.get('logits', final_logits)

            # Optional diagnostics
            try:
                recovery = type_top1_acc(out['logits'], base_batch['aatype'], node_mask=res_mask)
                _, perplexity = compute_CE_perplexity(out['logits'], base_batch['aatype'], mask=res_mask)
                print(f'recovery: {float(recovery):.3f}, perplexity: {float(perplexity):.3f}')
            except Exception:
                pass

            # Atom-level error (sidechain) with current prediction
            try:
                self.loss(input_feats_base, res_mask, x1_pred)
            except Exception:
                pass

            # Predict next xt
            if i < heun_steps:
                xt2, out_t2 = self._heun_step_R3(
                    t1=t1,
                    t_next=t_next,
                    xt=xt,
                    x1_pred_t1=x1_pred,
                    model=model,
                    input_feats_base=input_feats_base,
                    backbone_local=backbone_local,
                    res_mask=res_mask,
                )
            else:
                v1 = (x1_pred - xt) / max(1e-6, (1.0 - float(t1)))
                xt2 = xt + v1 * float(dt_eff)

            # Displacement cap per atom (Angstrom)
            if disp_cap_ang is not None:
                xt2 = self._displacement_cap(xt, xt2, ang_cap=disp_cap_ang)

            # Early steps: mix back toward the predicted bridge to suppress off-bridge drift
            if i < bridge_gamma_K:
                xt2_bridge = (1.0 - float(t_next)) * x0 + float(t_next) * x1_pred
                gamma_i = bridge_gamma0 * (1.0 - (i / max(1, bridge_gamma_K)))
                xt2 = (1.0 - gamma_i) * xt2 + gamma_i * xt2_bridge

            xt = xt2
            # advance time using effective dt
            t1 = torch.as_tensor(float(t_next), device=device, dtype=torch.float32)
            i += 1

        # Compose final outputs
        atoms14_local_final = torch.cat([backbone_local, xt], dim=-2)
        if side_exists is not None:
            atoms14_local_final[..., 3:, :] = atoms14_local_final[..., 3:, :] * side_exists[..., None]

        rigid = du.create_rigid(rotmats_1, trans_1)
        atoms14_global_final = rigid[..., None].apply(atoms14_local_final)

        return {
            'atoms14_local_final': atoms14_local_final,
            'atoms14_global_final': atoms14_global_final,
            'logits_final': final_logits,
        }

    @torch.no_grad()
    def fbb_sample_diag_baseline_t2(
        self,
        batch: dict,
        model,
        t2: float = 0.5,
    ):
        """Diagnostic single-forward at t2 using bridge-distributed xt2_true.

        Builds xt2_true = (1 - t2) * noise0 + t2 * x1_gt (masked by existence),
        sets atoms14_local_t accordingly, and runs a single forward at t2.
        Returns final logits and predicted sidechains, plus the input xt2_true.
        """
        device = batch['res_mask'].device

        res_mask = batch['res_mask']
        rotmats_1 = batch['rotmats_1']
        trans_1 = batch['trans_1']
        atoms14_local_gt = batch['atoms14_local']  # [B,N,14,3]
        backbone_local = atoms14_local_gt[..., :3, :]
        side_exists = batch.get('atom14_gt_exists', torch.ones_like(atoms14_local_gt[..., 0]))[..., 3:].bool()

        # Build bridge-distributed xt2_true
        noise0 = torch.randn_like(atoms14_local_gt[..., 3:, :])
        noise0 = noise0 * side_exists[..., None]
        x1_gt = atoms14_local_gt[..., 3:, :]
        t2f = float(t2)
        xt2_true = (1.0 - t2f) * noise0 + t2f * x1_gt


        t_eval=float(1)
        # Base features and model inputs at t2
        base_feats = self.fbb_prepare_batch(copy.deepcopy(batch))
        atoms14_local_t = torch.cat([backbone_local, xt2_true], dim=-2)
        input_feats = base_feats.copy()
        input_feats.update({
            't': torch.full((res_mask.shape[0],), t_eval, device=device, dtype=torch.float32),
            'r3_t': torch.full(res_mask.shape, t_eval, device=device, dtype=torch.float32),
            'so3_t': torch.full(res_mask.shape, t_eval, device=device, dtype=torch.float32),
            'atoms14_local_t': atoms14_local_t,
        })

        out = model(input_feats)
        side_atoms = out['side_atoms']
        final_logits = out.get('logits', None)

        atoms14_local_pred = torch.cat([backbone_local, side_atoms], dim=-2)
        rigid = du.create_rigid(rotmats_1, trans_1)
        atoms14_global_pred = rigid[..., None].apply(atoms14_local_pred)

        return {
            'logits_final': final_logits,
            'atoms14_local_input': atoms14_local_t,
            'atoms14_local_final': atoms14_local_pred,
            'atoms14_global_final': atoms14_global_pred,
        }

    @torch.no_grad()
    def fbb_sample_diag_step_t1_to_t2(
        self,
        batch: dict,
        model,
        t1: float | None = None,
        t2: float = 0.5,
    ):
        """Diagnostic two-forward path: forward at t1, Euler update x to t2, forward at t2.

        Steps:
        - Build xt1 = (1 - t1) * noise0 + t1 * x1_gt (masked)
        - Forward once at t1 to get x1_pred
        - Euler update to xt2_est
        - Forward once at t2 with xt2_est
        Returns logits at t2 and predicted sidechains, plus the intermediate states.
        """
        device = batch['res_mask'].device

        if t1 is None:
            t1 = float(self._cfg.min_t)
        t1f = float(t1)
        t2f = float(t2)
        assert 0.0 <= t1f < t2f <= 1.0

        res_mask = batch['res_mask']
        rotmats_1 = batch['rotmats_1']
        trans_1 = batch['trans_1']
        atoms14_local_gt = batch['atoms14_local']  # [B,N,14,3]
        backbone_local = atoms14_local_gt[..., :3, :]
        side_exists = batch.get('atom14_gt_exists', torch.ones_like(atoms14_local_gt[..., 0]))[..., 3:].bool()

        # Build xt1 on the training bridge at t1
        noise0 = torch.randn_like(atoms14_local_gt[..., 3:, :])
        noise0 = noise0 * side_exists[..., None]
        x1_gt = atoms14_local_gt[..., 3:, :]
        xt1 = (1.0 - t1f) * noise0 + t1f * x1_gt

        # Forward at t1
        base_feats = self.fbb_prepare_batch(copy.deepcopy(batch))
        atoms14_local_t1 = torch.cat([backbone_local, xt1], dim=-2)
        input_t1 = base_feats.copy()
        input_t1.update({
            't': torch.full((res_mask.shape[0],), t1f, device=device, dtype=torch.float32),
            'r3_t': torch.full(res_mask.shape, t1f, device=device, dtype=torch.float32),
            'so3_t': torch.full(res_mask.shape, t1f, device=device, dtype=torch.float32),
            'atoms14_local_t': atoms14_local_t1,
        })
        out_t1 = model(input_t1)
        x1_pred = out_t1['side_atoms']  # [B,N,11,3]

        # Euler update xt to t2
        dt = max(t2f - t1f, 1e-6)
        xt2_est = self._trans_euler_step(dt, t1f, x1_pred, xt1)

        # Forward at t2
        atoms14_local_t2 = torch.cat([backbone_local, xt2_est], dim=-2)
        input_t2 = base_feats.copy()
        input_t2.update({
            't': torch.full((res_mask.shape[0],), t2f, device=device, dtype=torch.float32),
            'r3_t': torch.full(res_mask.shape, t2f, device=device, dtype=torch.float32),
            'so3_t': torch.full(res_mask.shape, t2f, device=device, dtype=torch.float32),
            'atoms14_local_t': atoms14_local_t2,
        })
        out_t2 = model(input_t2)
        side_atoms_t2 = out_t2['side_atoms']
        final_logits_t2 = out_t2.get('logits', None)

        atoms14_local_pred = torch.cat([backbone_local, side_atoms_t2], dim=-2)
        rigid = du.create_rigid(rotmats_1, trans_1)
        atoms14_global_pred = rigid[..., None].apply(atoms14_local_pred)

        return {
            'logits_final': final_logits_t2,
            'atoms14_local_input_t1': atoms14_local_t1,
            'atoms14_local_input_t2': atoms14_local_t2,
            'atoms14_local_final': atoms14_local_pred,
            'atoms14_global_final': atoms14_global_pred,
        }

    def guidance(self, trans_t, rotmats_t, model_out, motif_mask, R_motif, trans_motif, Log_delta_R, delta_x, t, d_t, logs_traj):
        # Select motif
        motif_mask = motif_mask.clone()
        trans_pred = model_out['pred_trans'][:, motif_mask]  # [B, motif_res, 3]
        R_pred = model_out['pred_rotmats'][:, motif_mask]  # [B, motif_res, 3, 3]

        # Proposal for marginalising motif rotation
        F = twisting.motif_rots_vec_F(trans_motif, R_motif, self._cfg.twisting.num_rots, align=self._cfg.twisting.align, scale=self._cfg.twisting.scale_rots, device=self._device, dtype=torch.float32)

        # Estimate p(motif|predicted_motif)
        grad_Log_delta_R, grad_x_log_p_motif, logs = twisting.grad_log_lik_approx(R_pred, trans_pred, R_motif, trans_motif, Log_delta_R, delta_x, None, None, None, F, twist_potential_rot=self._cfg.twisting.potential_rot, twist_potential_trans=self._cfg.twisting.potential_trans)

        with torch.no_grad():
            # Choose scaling
            t_trans = t
            t_so3 = t
            if self._cfg.twisting.scale_w_t == 'ot':
                var_trans = ((1 - t_trans) / t_trans)[:, None]
                var_rot = ((1 - t_so3) / t_so3)[:, None, None]
            elif self._cfg.twisting.scale_w_t == 'linear':
                var_trans = (1 - t)[:, None]
                var_rot = (1 - t_so3)[:, None, None]
            elif self._cfg.twisting.scale_w_t == 'constant':
                num_batch = trans_pred.shape[0]
                var_trans = torch.ones((num_batch, 1, 1)).to(R_pred.device)
                var_rot = torch.ones((num_batch, 1, 1, 1)).to(R_pred.device)
            var_trans = var_trans + self._cfg.twisting.obs_noise ** 2
            var_rot = var_rot + self._cfg.twisting.obs_noise ** 2

            trans_scale_t = self._cfg.twisting.scale / var_trans
            rot_scale_t = self._cfg.twisting.scale / var_rot

            # Compute update
            trans_t, rotmats_t = twisting.step(trans_t, rotmats_t, grad_x_log_p_motif, grad_Log_delta_R, d_t, trans_scale_t, rot_scale_t, self._cfg.twisting.update_trans, self._cfg.twisting.update_rot)

        # delete unsused arrays to prevent from any memory leak
        del grad_Log_delta_R
        del grad_x_log_p_motif
        del Log_delta_R
        del delta_x
        for key, value in model_out.items():
            model_out[key] = value.detach().requires_grad_(False)

        return trans_t, rotmats_t, logs_traj

