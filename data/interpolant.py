from collections import defaultdict
import torch
import tqdm

from data import so3_utils
from data import utils as du
from scipy.spatial.transform import Rotation
from data import all_atom
import copy
from torch import autograd
# from motif_scaffolding import twisting
import openfold.utils.rigid_utils as ru
import numpy as np
from models.loss import type_ce_loss, type_top1_acc, compute_CE_perplexity, huber, pairwise_distance_loss, \
    backbone_mse_loss
import torch.nn.functional as F
# from data.sh_density import sh_density_from_atom14_with_masks
# from data.sh_density import sh_density_from_atom14_with_masks_clean

def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)


def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch * num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)


def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])


def _shs_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None, None, None, None] + trans_1 * (
                1 - diffuse_mask[..., None, None, None, None])


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
    node_mask = node_mask.bool()
    rand = torch.rand(node_mask.shape, generator=g, device=device)
    mask = (rand < mask_prob) & node_mask  # 只在有效节点里抽样
    update_mask = mask.clone()
    return mask*1.0, update_mask


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
        var = sigma_t ** 2 - reverse_alpha_ratio * d_sigma_t * sigma_t
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

    def __init__(self, cfg, task, noise_scheme):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None
        self.task = task
        self.noise_scheme = noise_scheme
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
        return t * (1 - 2 * self._cfg.min_t) + self._cfg.min_t

    def sample_t_ssq(self, num_batch):
        """使用 logit-normal 采样，对齐 ml-simplefold 的时间分布"""
        t_size = num_batch
        t = 0.98 * logit_normal_sample(t_size, m=0.8, s=1.7, device=self._device) + 0.02 * torch.rand(t_size,
                                                                                                      device=self._device)
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
        sh_gaus_0 = torch.randn_like(sh_1, device=self._device)
        # sh_gaus_0 = _centered_gaussian(*res_mask.shape, self._device)
        sh_0 = sh_gaus_0  # .view((*res_mask.shape,4, -1))
        sh_t = (1 - t[..., None, None, None, None]) * sh_0 + t[
            ..., None, None, None, None] * sh_1  # .view((*res_mask.shape,4, -1))
        sh_t = _shs_diffuse_mask(sh_t, sh_1, diffuse_mask)  # .view((*res_mask.shape,4, -1))

        return sh_t * res_mask[..., None, None, None, None]

    def _corrupt_rotmats(self, rotmats_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch * num_res
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

        noisy_batch['rotmats_t'] = rotmats_t


        noisy_batch['fixed_mask'] = res_mask

        noisy_batch['t'] = t
        noisy_batch['rigids_t'] = du.create_rigid(rotmats_t, trans_t).to_tensor_7()

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
        t = self.sample_t(num_batch)[:, None] * 0

        sh_t = self._corrupt_shs(batch['normalize_density'], t, res_mask, diffuse_mask) * batch['density_mask'] * 0

        noisy_batch['SH_masked'] = sh_t
        noisy_batch['t'] = t[:, 0]
        noisy_batch['update_mask'] = batch['res_mask']

        return noisy_batch

    def fbb_corrupt_batch(self, batch, prob=None):
        noisy_batch = copy.deepcopy(batch)

        node_mask = noisy_batch['res_mask']
        device = node_mask.device

        mask_prob = prob if prob is not None else np.random.uniform(0.5, 1.0)
        mask, update_mask = bert15_simple_mask(node_mask, mask_prob=mask_prob)
        # ensure boolean dtype for downstream torch.where conditions
        mask = mask.bool()

        # check geo hgf
        update_mask = ~update_mask.bool()




        noisy_batch['update_mask'] = update_mask

        # Mask aatype: use 22 as MASK token for masked positions
        mask_token = 22
        noisy_batch['nativeaatype']=noisy_batch['aatype']
        noisy_batch['aatype'] = torch.where(
            update_mask,
            torch.full_like(batch['aatype'], mask_token),
            batch['aatype']
        )

        # Randomly offset residue indices per-chain to avoid leaking absolute positions.
        # Offset is sampled per example to keep intra-chain ordering consistent.
        if 'res_idx' in noisy_batch:
            res_idx = noisy_batch['res_idx']
            # Sample offset in [-K, K] where K ~ Uniform(0, offset_max)
            offset_max = getattr(self._cfg, 'res_idx_offset_max', 50)
            offsets = torch.randint(-offset_max, offset_max + 1, (res_idx.shape[0], 1),
                                    device=res_idx.device)
            noisy_batch['res_idx'] = res_idx + offsets

        if self.noise_scheme == 'side_atoms':
            batch_size = node_mask.shape[0]
            # t = self.sample_t_ssq(batch_size)[:, None]  # [B,1]
            t = self.sample_t(batch_size)[:, None]  # [B,1]
            t_eps = getattr(self._cfg, 'min_t', 1e-3)
            t = torch.clamp(t, min=t_eps, max=1 - t_eps)

            noisy_batch['t'] = t[:, 0]
            t_broadcast = t[:, 0][:, None].expand_as(node_mask)
            noisy_batch['r3_t'] = t_broadcast.to(dtype=torch.float32)
            noisy_batch['so3_t'] = t_broadcast.to(dtype=torch.float32)

            if ('atoms14_local' in noisy_batch) and ('atom14_gt_exists' in noisy_batch):
                atoms14_local = noisy_batch['atoms14_local']
                atom14_exists = noisy_batch['atom14_gt_exists'].bool()

                sidechain_exists = atom14_exists[..., 3:]
                effective_mask = update_mask[..., None].expand_as(sidechain_exists)
                # effective_mask = (sidechain_exists & update_mask_exp).bool()

                coord_scale = getattr(self._cfg, 'coord_scale', 1.0)
                noise_sc = torch.randn_like(atoms14_local[..., 3:, :]) * coord_scale
                noise_sc = noise_sc * effective_mask[..., None]

                t_expand = t[..., None, None]
                clean_sc = atoms14_local[..., 3:, :]
                y_sc = (1.0 - t_expand) * noise_sc + t_expand * clean_sc
                v_sc = clean_sc - noise_sc

                y_sc = torch.where(effective_mask[..., None].bool(), y_sc, clean_sc)
                v_sc = torch.where(effective_mask[..., None].bool(), v_sc, torch.zeros_like(v_sc))

                y_full =  torch.zeros_like(atoms14_local)
                v_full = torch.zeros_like(atoms14_local)
                y_full[..., 3:, :] = y_sc
                v_full[..., 3:, :] = v_sc

                noisy_batch['atoms14_local_t'] = y_full
                # noisy_batch['y_t'] = y_full
                noisy_batch['v_t'] = v_full
                noisy_batch['sidechain_atom_mask'] = effective_mask
            else:
                raise KeyError('atoms14_local or atom14_gt_exists missing for FBB side atom noise scheme')

        elif self.noise_scheme in ('torision', 'torsion'):
            pass
        elif self.noise_scheme == 'tokenatoms':
            # Token-style masking: set sidechain atoms to zero for masked positions
            if ('atoms14_local' in noisy_batch) and ('atom14_gt_exists' in noisy_batch):
                atoms14_local = noisy_batch['atoms14_local']
                atom14_exists = noisy_batch['atom14_gt_exists'].bool()

                # Create masked version: keep backbone (0:3), zero out sidechains where update_mask=True
                atoms14_masked = atoms14_local.clone()

                # Zero out sidechain atoms (indices 3:) for positions that need prediction
                update_mask_exp = update_mask[..., None, None].expand_as(atoms14_local[..., 3:, :])
                atoms14_masked[..., 3:, :] = torch.where(
                    update_mask_exp,
                    torch.zeros_like(atoms14_local[..., 3:, :]),
                    atoms14_local[..., 3:, :]
                )

                # Store the masked input and target
                noisy_batch['atoms14_local_t'] = atoms14_masked
                noisy_batch['atoms14_local_clean'] = atoms14_local  # Keep clean version as target

                # For compatibility with training loop, set v_t to the clean sidechains
                v_full = torch.zeros_like(atoms14_local)
                v_full[..., 3:, :] = atoms14_local[..., 3:, :]  # Target is the clean sidechains
                noisy_batch['v_t'] = v_full

                # Set sidechain mask
                sidechain_exists = atom14_exists[..., 3:]
                effective_mask = update_mask[..., None].expand_as(sidechain_exists)
                noisy_batch['sidechain_atom_mask'] = effective_mask

                # Set t to 1.0 for all samples (deterministic masking, no noise)
                batch_size = node_mask.shape[0]
                t = torch.ones(batch_size, device=device)
                noisy_batch['t'] = t
                t_broadcast = t[:, None].expand_as(node_mask)
                noisy_batch['r3_t'] = t_broadcast.to(dtype=torch.float32)
                noisy_batch['so3_t'] = t_broadcast.to(dtype=torch.float32)

        elif self.noise_scheme == 'Gaussianatoms':
            # 目标: Inverse Folding (Sequence Design) - 回归模式

            if ('atoms14_local' in noisy_batch) and ('atom14_gt_exists' in noisy_batch):
                atoms14_local = noisy_batch['atoms14_local']
                atom14_exists = noisy_batch['atom14_gt_exists'].bool()

                # -------------------------------------------------------
                # 1. 构造网络输入 (Input: atoms14_local_t)
                # -------------------------------------------------------
                atoms14_masked = atoms14_local.clone()

                # 扩展 update_mask: [B, N] -> [B, N, 14, 3] (只针对 sidechain 3:)
                update_mask_exp = update_mask[..., None, None].expand_as(atoms14_local[..., 4:, :])

                # 【核心修改点】:
                # 1. 目标区域 (update_mask=True): 侧链坐标全部置 0 (代表一个标准球的中心)
                # 2. 环境区域 (update_mask=False): 侧链坐标保留，作为物理上下文 (Context)
                atoms14_masked[..., 4:, :] = torch.where(
                    update_mask_exp,
                    # 目标：输入给 Gaussian Head 的是 0 坐标 (代表待生成的 Standard Sphere)
                    torch.zeros_like(atoms14_local[..., 4:, :]),
                    atoms14_local[...,4:, :]  # 保留真实的侧链坐标作为环境
                )

                # Store the masked input: 这是模型将要接收的原子坐标
                noisy_batch['atoms14_local_t'] = atoms14_masked

                # -------------------------------------------------------
                # 2. 构造网络目标 (Target: v_t)
                # -------------------------------------------------------

                # Target V_t 在回归模式下就是 Clean Target
                v_full = torch.zeros_like(atoms14_local)

                # Target Sidechain (只对被 Mask 的区域提供监督信号)
                sidechain_exists = atom14_exists[..., 4:]
                effective_mask = update_mask[..., None].expand_as(sidechain_exists)

                # 目标 V_t = 真实的 Clean Sidechain (用于计算 Gaussian 参数的 GT)
                v_target_sc = atoms14_local[..., 4:, :] * effective_mask.bool().unsqueeze(-1)
                v_full[..., 4:, :] = v_target_sc

                # Store the final target
                noisy_batch['v_t'] = v_full

                # 最终的 Loss Mask (只在被 Mask 的侧链原子上计算 Loss)
                noisy_batch['sidechain_atom_mask'] = effective_mask.bool()

                # 【关键】移除所有 t 相关的计算，模型工作在 t=1 (确定性)
                batch_size = node_mask.shape[0]
                noisy_batch['t'] = torch.ones(batch_size, device=device)  # t=1 只是占位符，不参与计算
                # 移除 r3_t 和 so3_t


            else:
                raise KeyError('atoms14_local or atom14_gt_exists missing for tokenatoms noise scheme')

        elif self.noise_scheme == 'sh':
            keep_mask = (~mask).to(dtype=batch['normalize_density'].dtype, device=batch['normalize_density'].device)
            SH_masked = batch['normalize_density'] * keep_mask[..., None, None, None, None]
            batch['SH_masked'] = SH_masked

        return noisy_batch

    def allatoms_corrupt_batch(self, batch, prob=None):
        """
        Corrupt both backbone and sidechain atoms simultaneously:
        - Backbone (N, CA, C): SE(3) flow matching (same as corrupt_batch)
        - Sidechain (atoms14[3:14]): R3 flow matching (same as fbb_corrupt_batch side_atoms)

        Args:
            batch: dict with required keys:
                - 'trans_1': [B, N, 3] clean translations
                - 'rotmats_1': [B, N, 3, 3] clean rotations
                - 'atoms14_local': [B, N, 14, 3] clean local coords (backbone + sidechain)
                - 'atom14_gt_exists': [B, N, 14] atom existence mask
                - 'res_mask': [B, N] residue mask
                - 'diffuse_mask': [B, N] diffusion mask
            prob: Optional mask probability for update_mask

        Returns:
            noisy_batch: dict with corrupted backbone and sidechain atoms
        """
        noisy_batch = copy.deepcopy(batch)

        # Get masks
        res_mask = batch['res_mask']
        # diffuse_mask = batch['diffuse_mask']
        num_batch, num_res = res_mask.shape
        device = res_mask.device

        # Sample update mask (similar to fbb_corrupt_batch)
        mask_prob = prob if prob is not None else np.random.uniform(0.15, 1.0)
        mask, update_mask = bert15_simple_mask(res_mask, mask_prob=mask_prob)


        noisy_batch['update_mask'] = update_mask

        # Randomly offset residue indices (same as fbb_corrupt_batch)
        if 'res_idx' in noisy_batch:
            res_idx = noisy_batch['res_idx']
            offset_max = getattr(self._cfg, 'res_idx_offset_max', 50)
            offsets = torch.randint(-offset_max, offset_max + 1, (res_idx.shape[0], 1),
                                    device=res_idx.device)
            noisy_batch['res_idx'] = res_idx + offsets

        # Sample time step
        t = self.sample_t(num_batch)[:, None]  # [B, 1]
        t_eps = getattr(self._cfg, 'min_t', 1e-3)
        t = torch.clamp(t, min=t_eps, max=1 - t_eps)
        noisy_batch['t'] = t

        # Broadcast time for different uses
        t_broadcast = t[:, 0][:, None].expand_as(res_mask)
        noisy_batch['r3_t'] = t_broadcast.to(dtype=torch.float32)
        noisy_batch['so3_t'] = t_broadcast.to(dtype=torch.float32)

        # ========== Part 1: Corrupt Backbone (SE(3) flow matching) ==========
        trans_1 = batch['trans_1']  # [B, N, 3]
        rotmats_1 = batch['rotmats_1']  # [B, N, 3, 3]

        # Corrupt translation (using update_mask, same as sidechain)
        # Construct trans_0, trans_t, trans_v explicitly (like sidechain)
        if self._trans_cfg.corrupt:
            # Generate noise
            trans_nm_0 = _centered_gaussian(*res_mask.shape, device)
            trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE

            # Linear interpolation: trans_t = (1-t) * trans_0 + t * trans_1
            trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1

            # Velocity field: v = trans_1 - trans_0 (direction from noise to clean)
            trans_v = trans_1 - trans_0

            # Apply diffuse mask
            trans_t = _trans_diffuse_mask(trans_t, trans_1, mask)
            trans_v = _trans_diffuse_mask(trans_v, torch.zeros_like(trans_v), mask)
            trans_0 = _trans_diffuse_mask(trans_0, trans_1, mask)

            # Apply res_mask
            trans_t = trans_t * res_mask[..., None]
            trans_v = trans_v * res_mask[..., None]
            trans_0 = trans_0 * res_mask[..., None]
        else:
            trans_t = trans_1
            trans_0 = trans_1
            trans_v = torch.zeros_like(trans_1)

        if torch.any(torch.isnan(trans_t)):
            raise ValueError('NaN in trans_t during allatoms corruption')

        noisy_batch['trans_t'] = trans_t
        noisy_batch['trans_0'] = trans_0
        noisy_batch['trans_v'] = trans_v

        # Corrupt rotation (using update_mask, same as sidechain)
        # Construct rotmats_0, rotmats_t, rot_v explicitly
        if self._rots_cfg.corrupt:
            # Generate noise rotations
            noisy_rotmats = self.igso3.sample(
                torch.tensor([1.5]),
                num_batch * num_res
            ).to(device)
            noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
            rotmats_0 = torch.einsum(
                "...ij,...jk->...ik", rotmats_1, noisy_rotmats)

            # Geodesic interpolation for SO(3)
            rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)

            # Compute rotation velocity field (axis-angle representation)
            rot_v = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)

            # Apply res_mask
            identity = torch.eye(3, device=device)
            rotmats_t = (
                rotmats_t * res_mask[..., None, None]
                + identity[None, None] * (1 - res_mask[..., None, None])
            )
            rotmats_0 = (
                rotmats_0 * res_mask[..., None, None]
                + identity[None, None] * (1 - res_mask[..., None, None])
            )
            rot_v = rot_v * res_mask[..., None]

            # Apply diffuse mask
            rotmats_t = _rots_diffuse_mask(rotmats_t, rotmats_1, mask)
            rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_1, mask)
            rot_v = rot_v * mask[..., None]
        else:
            rotmats_t = rotmats_1
            rotmats_0 = rotmats_1
            rot_v = torch.zeros(num_batch, num_res, 3, device=device)

        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError('NaN in rotmats_t during allatoms corruption')

        noisy_batch['rotmats_t'] = rotmats_t
        noisy_batch['rotmats_0'] = rotmats_0
        noisy_batch['rot_v'] = rot_v

        # Store rigids
        noisy_batch['rotmats_1'] = rotmats_1
        noisy_batch['trans_1'] = trans_1
        noisy_batch['rigids_t'] = du.create_rigid(rotmats_t, trans_t).to_tensor_7()

        # ========== Part 2: Corrupt Sidechain (R3 flow matching) ==========
        if ('atoms14_local' in batch) and ('atom14_gt_exists' in batch):
            atoms14_local = batch['atoms14_local']  # [B, N, 14, 3]
            atom14_exists = batch['atom14_gt_exists'].bool()  # [B, N, 14]

            # Extract sidechain atoms (indices 3:14)
            sidechain_exists = atom14_exists[..., 3:]  # [B, N, 11]
            update_mask_exp = update_mask[..., None].expand_as(sidechain_exists)
            effective_mask = (sidechain_exists & update_mask_exp).bool()

            # Generate noise for sidechain atoms
            coord_scale = getattr(self._cfg, 'coord_scale', 1.0)
            noise_sc = torch.randn_like(atoms14_local[..., 3:, :]) * coord_scale
            noise_sc = noise_sc * effective_mask[..., None]

            # Linear interpolation: y_t = (1-t) * noise + t * clean
            t_expand = t[..., None, None]  # [B, 1, 1, 1]
            clean_sc = atoms14_local[..., 3:, :]  # [B, N, 11, 3]
            y_sc = (1.0 - t_expand) * noise_sc + t_expand * clean_sc
            v_sc = clean_sc - noise_sc  # velocity field

            # Apply effective mask
            y_sc = torch.where(effective_mask[..., None].bool(), y_sc, clean_sc)
            v_sc = torch.where(effective_mask[..., None].bool(), v_sc, torch.zeros_like(v_sc))

            # Assemble full atoms14: [backbone(clean), sidechain(noisy)]
            # Note: backbone atoms (0:3) remain clean in local frame
            y_full = torch.zeros_like(atoms14_local)
            v_full = torch.zeros_like(atoms14_local)
            y_full[..., 3:, :] = y_sc
            v_full[..., 3:, :] = v_sc

            noisy_batch['atoms14_local_t'] = y_full
            # noisy_batch['y_t'] = y_full
            noisy_batch['v_t'] = v_full
            noisy_batch['sidechain_atom_mask'] = effective_mask
        else:
            raise KeyError('atoms14_local or atom14_gt_exists missing for allatoms noise scheme')

        noisy_batch['diffuse_mask'] = update_mask
        noisy_batch['fixed_mask'] = res_mask

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
                atoms14_local = noisy_batch['atoms14_local']  # [B,N,14,3]
                # atom14_exists = noisy_batch['atom14_gt_exists'].bool()  # [B,N,14]

                # 侧链掩码 3..13
                sidechain_mask = torch.zeros_like(noisy_batch['atom14_gt_exists'].bool(), dtype=torch.bool)
                sidechain_mask[..., 3:] = True

                # 仅在有效+更新位置扰动
                update_mask_exp = update_mask[..., None].expand_as(sidechain_mask).bool()
                effective_mask = sidechain_mask & update_mask_exp

                # 标准正态噪声
                noise0 = torch.randn_like(atoms14_local)
                noise0 = noise0 * effective_mask[..., None]

                # 线性桥插值到 t
                t_expand = t[..., None, None]  # [B,1,1]
                interp = (1.0 - t_expand) * noise0 * 8 + t_expand * atoms14_local
                atoms14_local_t = torch.where(effective_mask[..., None], interp, atoms14_local)

                noisy_batch['atoms14_local_t'] = atoms14_local_t
                noisy_batch['sidechain_atom_mask'] = sidechain_mask

        elif self.noise_scheme in ('torision', 'torsion'):
            pass
        elif self.noise_scheme == 'sh':
            SH_masked = batch['normalize_density'] * (1 - mask)[..., None, None, None, None]
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
            coord_scale = getattr(self._cfg, 'coord_scale', 1.0)
            atoms14_local = clean_batch['atoms14_local']
            atom14_exists = clean_batch['atom14_gt_exists'].bool()
            exists_mask = atom14_exists[..., 3:].float()  # Only for sidechain atoms [B,N,11]

            # Initialize sidechain atoms with pure noise, keep backbone clean
            noise = torch.randn_like(atoms14_local)
            noise[..., :3, :] = 0.0  # Keep backbone atoms (0,1,2) clean

            # For inference, start with pure noise for sidechain atoms
            atoms14_local_scaled = atoms14_local
            atoms14_local_t = atoms14_local_scaled.clone()
            atoms14_local_t[..., 3:, :] = noise[..., 3:, :] * coord_scale * exists_mask[..., None]

            clean_batch['atoms14_local_t'] = atoms14_local_t
            clean_batch['sidechain_atom_mask'] = atom14_exists[..., 3:]

        return clean_batch

    def fbb_batch(self, batch, designchain=1):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom
        node_mask = batch['res_mask']

        # if prob is not None:
        #     mask_prob=prob
        # else:
        #     mask_prob=np.random.uniform(0.15, 1.0)
        mask = batch['chain_idx'] - designchain
        update_mask = mask.clone()
        SH_masked = batch['normalize_density'] * (~mask)[..., None, None, None, None]  # 把mask位置变成0
        noisy_batch['SH_masked'] = SH_masked
        noisy_batch['update_mask'] = update_mask
        del noisy_batch['normalize_density']

        return noisy_batch

    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t * self._rots_cfg.exp_rate)
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

    def pf_ode_step(self, dt, t, trans_1, trans_t, eps=1e-5):
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

        return x_next, drift

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

        delta_rot = I.unsqueeze(0).unsqueeze(0) + sin_angle[..., None] * K + (1 - cos_angle)[..., None] * torch.bmm(
            K.view(-1, 3, 3), K.view(-1, 3, 3)).view_as(K)

        return torch.bmm(rotmats_t.view(-1, 3, 3), delta_rot.view(-1, 3, 3)).view_as(rotmats_t)

    def heun_step_R3(self, dt, t, trans_1, trans_t, eps=1e-5):
        """
        Heun（改进欧拉）：
          1) 预测: x' = x + drift(x,t)*dt
          2) 校正: drift' = drift(x', t+dt)
          3) 合成: x_next = x + 0.5*(drift + drift')*dt
        """
        # predictor
        _, drift1 = self.pf_ode_step(dt, t, trans_1, trans_t, eps=eps)
        x_pred = trans_t + drift1 * dt

        # corrector
        t2 = (t + dt).clamp(min=eps, max=1 - eps)  # 反向时间
        _, drift2 = self.pf_ode_step(dt, t2, trans_1, x_pred, eps=eps)

        x_next = trans_t + 0.5 * (drift1 + drift2) * dt

        return x_next

    def loss(self, batch, loss_mask, side_atoms):
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
        side_gt_ang = side_gt_local
        side_pred_ang = side_atoms

        if 'r3_t' in batch:
            r3_t = batch['r3_t'].to(side_gt_local.dtype)
            t_num = torch.clamp(r3_t, min=eps)
            t_den = 1.0 - torch.clamp(r3_t, max=t_clip)
            t_den = torch.clamp(t_den, min=eps)
            snr_scale = (t_num / t_den)[..., None, None]
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
            'res_idx': res_idx,
            'chain_idx': chain_idx,
            'backbone': backbone
        }

        motif_scaffolding = False
        if diffuse_mask is not None and trans_1 is not None and rotmats_1 is not None:
            motif_scaffolding = True
            motif_mask = ~diffuse_mask.bool().squeeze(0)
        else:
            motif_mask = None
        if motif_scaffolding and not self._cfg.twisting.use:  # amortisation
            diffuse_mask = diffuse_mask.expand(num_batch, -1)  # shape = (B, num_residue)
            batch['diffuse_mask'] = diffuse_mask
            rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_1, diffuse_mask)
            trans_0 = _trans_diffuse_mask(trans_0, trans_1, diffuse_mask)
            if torch.isnan(trans_0).any():
                raise ValueError('NaN detected in trans_0')

        logs_traj = defaultdict(list)
        if motif_scaffolding and self._cfg.twisting.use:  # sampling / guidance
            assert trans_1.shape[0] == 1  # assume only one motif
            motif_locations = torch.nonzero(motif_mask).squeeze().tolist()
            true_motif_locations, motif_segments_length = twisting.find_ranges_and_lengths(motif_locations)

            # Marginalise both rotation and motif location
            assert len(motif_mask.shape) == 1
            trans_motif = trans_1[:, motif_mask]  # [1, motif_res, 3]
            R_motif = rotmats_1[:, motif_mask]  # [1, motif_res, 3, 3]
            num_res = trans_1.shape[-2]
            with torch.inference_mode(False):
                motif_locations = true_motif_locations if self._cfg.twisting.motif_loc else None
                F, motif_locations = twisting.motif_offsets_and_rots_vec_F(num_res, motif_segments_length,
                                                                           motif_locations=motif_locations,
                                                                           num_rots=self._cfg.twisting.num_rots,
                                                                           align=self._cfg.twisting.align,
                                                                           scale=self._cfg.twisting.scale_rots,
                                                                           trans_motif=trans_motif, R_motif=R_motif,
                                                                           max_offsets=self._cfg.twisting.max_offsets,
                                                                           device=self._device, dtype=torch.float64,
                                                                           return_rots=False)

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
            if verbose:  # and i % 1 == 0:
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

            if use_twisting:  # Reconstruction guidance
                with torch.inference_mode(False):
                    batch, Log_delta_R, delta_x = twisting.perturbations_for_grad(batch)
                    model_out = model(batch)
                    t = batch['r3_t']  # TODO: different time for SO3?
                    trans_t_1, rotmats_t_1, logs_traj = self.guidance(trans_t_1, rotmats_t_1, model_out, motif_mask,
                                                                      R_motif, trans_motif, Log_delta_R, delta_x, t,
                                                                      d_t, logs_traj)

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
                    pred_trans_potential = \
                    autograd.grad(outputs=trans_potential(grad_pred_trans_1), inputs=grad_pred_trans_1)[0]
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

    def diffusion_coefficient(self, t, eps=0.01, w_cutoff=0.99):
        """
        SimpleFold-style diffusion coefficient for SDE sampling.
        w = (1 - t) / (t + eps)
        When t >= w_cutoff, w = 0 (no noise near the end)
        """
        t_scalar = float(t)
        w = (1.0 - t_scalar) / (t_scalar + eps)
        if t_scalar >= w_cutoff:
            w = 0.0
        return w

    def compute_score_from_velocity(self, v_t, y_t, t, eps=1e-6):
        """
        Convert velocity to score for linear interpolant.

        Training: x_t = (1-t)*noise + t*x_data, where noise ~ N(0, coord_scale^2)
                  v = x_data - noise

        To compute score = ∇ log p(x_t):
          noise = x_t - t*v  (derived from x_t and velocity)
          score = -noise / ((1-t)^2 * coord_scale^2)
                = -(x_t - t*v) / ((1-t)^2 * coord_scale^2)
                = (t*v - x_t) / ((1-t)^2 * coord_scale^2)
        """
        coord_scale = getattr(self._cfg, 'coord_scale', 1.0)

        t_expand = t if isinstance(t, torch.Tensor) else torch.tensor(t, device=y_t.device, dtype=y_t.dtype)
        t_expand = t_expand.view(-1, 1, 1, 1)  # [B, 1, 1, 1]

        # Clamp t to avoid division by zero
        t_clamped = torch.clamp(t_expand, min=eps, max=1-eps)

        # Estimate noise from current state and velocity
        noise_est = y_t - t_clamped * v_t

        # Score = -noise / variance
        variance = (1.0 - t_clamped) ** 2 * (coord_scale ** 2)
        score = -noise_est / (variance + eps)

        return score

    @torch.no_grad()
    def fbb_sample_iterative_sde(
            self,
            batch: dict,
            model,
            num_timesteps: int | None = None,
            tau: float = 0.3,
            w_cutoff: float = 0.99,
    ):
        """SDE sampling with Euler-Maruyama integrator (simplified).

        Args:
            batch: input batch
            model: the model
            num_timesteps: number of integration steps
            tau: temperature/noise scale (default 0.3)
            w_cutoff: cutoff for diffusion coefficient (default 0.99)

        This implements the Euler-Maruyama step:
            mean_y = x + v * dt  (ODE step)
            x_new = mean_y + sqrt(2 * D * tau * dt) * noise  (add stochastic term)
        where D is a time-dependent diffusion coefficient: D = (1-t)/(t+eps)
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

        # SimpleFold uses num_timesteps + 1 points
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps + 1, device=device)

        # Prepare base features
        input_feats_base = copy.deepcopy(batch)
        backbone_local = input_feats_base['atoms14_local_t'][..., :3, :]
        xt = input_feats_base['atoms14_local_t'][..., 3:, :]  # [B, N, 11, 3]
        input_feats_base['atoms14_local_sc'] = torch.zeros_like(input_feats_base['atoms14_local_t'])

        logs = []

        for i in range(len(ts) - 1):
            t1 = float(ts[i])
            t2 = float(ts[i + 1])
            dt = t2 - t1

            atoms14_local_t = torch.cat([backbone_local, xt], dim=-2)

            input_feats = input_feats_base.copy()
            input_feats.update({
                't': torch.full((res_mask.shape[0],), t1, device=device, dtype=torch.float32),
                'r3_t': torch.full(res_mask.shape, t1, device=device, dtype=torch.float32),
                'so3_t': torch.full(res_mask.shape, t1, device=device, dtype=torch.float32),
                'atoms14_local_t': atoms14_local_t,
            })

            # SH+FBB: 计算SH密度（和fbb_sample_iterative一致）
            # with torch.no_grad():
                # normalize_density, *_ = sh_density_from_atom14_with_masks_clean(
                #     input_feats['atoms14_local_t'],
                #     batch['atom14_element_idx'],
                #     batch['atom14_gt_exists'],
                #     L_max=8,
                #     R_bins=24,
                # )
                # normalize_density = normalize_density / torch.sqrt(torch.tensor(4 * torch.pi))
            # input_feats['normalize_density'] = normalize_density

            with torch.no_grad():
                out = model(input_feats)
            v_pred = out['speed_vectors'].detach()  # velocity prediction (和fbb_sample_iterative一致)

            # 记录velocity统计（和fbb_sample_iterative一致）
            v_norm = v_pred.norm(dim=-1)  # [B, N, 11]
            v_norm_masked = (v_norm * side_exists.float()).sum() / side_exists.sum()

            step_log = {
                'step': i,
                't': t1,
                'v_norm_mean': v_norm_masked.item(),
                'xt_norm_mean': xt.norm(dim=-1).mean().item(),
            }
            logs.append(step_log)

            # 每隔几步打印一次（可选）
            if i % max(1, (len(ts) - 1) // 5) == 0 or i == len(ts) - 2:
                print(f"  [SDE] Step {i:3d} t={t1:.3f}: v_norm={v_norm_masked.item():.3f}Å  xt_norm={xt.norm(dim=-1).mean().item():.3f}Å")

            # Compute score from velocity (non-centered, consistent with training)
            score = self.compute_score_from_velocity(v_pred, xt, t1)

            # Diffusion coefficient
            diff_coeff = self.diffusion_coefficient(t1, w_cutoff=w_cutoff)

            # Drift term: velocity + diffusion * score (Langevin dynamics)
            drift = v_pred + diff_coeff * score

            # Mean update (deterministic part)
            mean_y = xt + drift * dt

            # Stochastic term (Euler-Maruyama)
            if diff_coeff > 0 and tau > 0:
                noise = torch.randn_like(xt)
                noise = noise * side_exists[..., None]  # mask noise
                stochastic_term = torch.sqrt(torch.tensor(2.0 * dt * diff_coeff * tau, device=device)) * noise
                xt_new = mean_y + stochastic_term
            else:
                xt_new = mean_y

            # Apply existence mask
            xt = xt_new * side_exists[..., None]

            # Self-conditioning (disabled)
            clean_pred = xt + (1.0 - t2) * v_pred
            input_feats_base['atoms14_local_sc'] = torch.cat([backbone_local, clean_pred], dim=-2) * 0

            # 显式清理中间变量，避免大步数时的内存累积
            del atoms14_local_t, input_feats,  out, v_pred, score, drift, mean_y
            if diff_coeff > 0 and tau > 0:
                del noise, stochastic_term
            del xt_new, clean_pred
            if i % 50 == 0:  # 每50步清理一次GPU缓存
                torch.cuda.empty_cache()

        # Final step - run model at t_final
        t_final = float(ts[-1])

        atoms14_local_t = torch.cat([backbone_local, xt], dim=-2)
        input_feats_final = input_feats_base.copy()
        input_feats_final.update({
            't': torch.full((res_mask.shape[0],), t_final, device=device, dtype=torch.float32),
            'r3_t': torch.full(res_mask.shape, t_final, device=device, dtype=torch.float32),
            'so3_t': torch.full(res_mask.shape, t_final, device=device, dtype=torch.float32),
            'atoms14_local_t': atoms14_local_t,
        })

        # 添加SH密度计算（和fbb_sample_iterative一致）
        # normalize_density_final, *_ = sh_density_from_atom14_with_masks_clean(
        #     input_feats_final['atoms14_local_t'],
        #     batch['atom14_element_idx'],
        #     batch['atom14_gt_exists'],
        #     L_max=8,
        #     R_bins=24,
        # )
        # normalize_density_final = normalize_density_final / torch.sqrt(torch.tensor(4 * torch.pi))
        # input_feats_final['normalize_density'] = normalize_density_final

        with torch.no_grad():
            out_final = model(input_feats_final)
        v_final = out_final['speed_vectors']  # 修改为speed_vectors（和fbb_sample_iterative一致）
        final_logits = out_final.get('logits', None)

        # Final clean prediction
        clean_final = xt + (1.0 - t_final) * v_final

        atoms14_local_final = torch.cat([backbone_local, clean_final], dim=-2)
        if side_exists is not None:
            atoms14_local_final[..., 3:, :] = atoms14_local_final[..., 3:, :] * side_exists[..., None]

        # Build global 14 using fixed frames
        rigid = du.create_rigid(rotmats_1, trans_1)
        atoms14_global_final = rigid[..., None].apply(atoms14_local_final)

        # Diagnostics (same as ODE version)
        diagnostics = {}
        if 'atoms14_local' in batch:
            gt_atoms14_local = batch['atoms14_local']
            gt_sidechain = gt_atoms14_local[..., 3:, :]
            pred_sidechain = atoms14_local_final[..., 3:, :]

            if side_exists is not None:
                mask = side_exists[..., None].float()
                diff = (pred_sidechain - gt_sidechain) ** 2
                mse_per_atom = (diff * mask).sum(dim=-1)
                rmsd_per_atom = torch.sqrt(mse_per_atom + 1e-8)

                num_atoms = mask.sum()
                if num_atoms > 0:
                    mean_rmsd = (rmsd_per_atom * mask.squeeze(-1)).sum() / num_atoms
                    diagnostics['sidechain_rmsd'] = mean_rmsd.item()

            if final_logits is not None and 'aatype' in batch:
                atoms14_local_gt_input = torch.cat([backbone_local, gt_sidechain], dim=-2)
                input_feats_gt = input_feats_base.copy()
                input_feats_gt.update({
                    't': torch.full((res_mask.shape[0],), t_final, device=device, dtype=torch.float32),
                    'r3_t': torch.full(res_mask.shape, t_final, device=device, dtype=torch.float32),
                    'so3_t': torch.full(res_mask.shape, t_final, device=device, dtype=torch.float32),
                    'atoms14_local_t': atoms14_local_gt_input,
                })

                # # 添加GT coords的SH密度计算（和fbb_sample_iterative一致）
                # normalize_density_gt, *_ = sh_density_from_atom14_with_masks_clean(
                #     input_feats_gt['atoms14_local_t'],
                #     batch['atom14_element_idx'],
                #     batch['atom14_gt_exists'],
                #     L_max=8,
                #     R_bins=24,
                # )
                # normalize_density_gt = normalize_density_gt / torch.sqrt(torch.tensor(4 * torch.pi))
                # input_feats_gt['normalize_density'] = normalize_density_gt

                # with torch.no_grad():
                #     out_gt = model(input_feats_gt)
                # logits_with_gt = out_gt.get('logits', None)
                #
                # if logits_with_gt is not None:
                #     aatype = batch['aatype']
                #     mask = batch.get('res_mask', torch.ones_like(aatype))
                #
                #     ce_pred = torch.nn.functional.cross_entropy(
                #         final_logits.reshape(-1, final_logits.shape[-1]),
                #         aatype.reshape(-1),
                #         reduction='none'
                #     )
                #     ce_pred = ce_pred.reshape(aatype.shape)
                #     valid_count_pred = mask.sum()
                #     if valid_count_pred > 0:
                #         mean_ce_pred = (ce_pred * mask).sum() / valid_count_pred
                #         ppl_pred = torch.exp(mean_ce_pred)
                #         diagnostics['perplexity_with_pred_coords'] = ppl_pred.item()
                #
                #     ce_gt = torch.nn.functional.cross_entropy(
                #         logits_with_gt.reshape(-1, logits_with_gt.shape[-1]),
                #         aatype.reshape(-1),
                #         reduction='none'
                #     )
                #     ce_gt = ce_gt.reshape(aatype.shape)
                #     valid_count_gt = mask.sum()
                #     if valid_count_gt > 0:
                #         mean_ce_gt = (ce_gt * mask).sum() / valid_count_gt
                #         ppl_gt = torch.exp(mean_ce_gt)
                #         diagnostics['perplexity_with_gt_coords'] = ppl_gt.item()
                #
                #     pred_aa_with_pred = final_logits.argmax(dim=-1)
                #     pred_aa_with_gt = logits_with_gt.argmax(dim=-1)
                #
                #     correct_pred = ((pred_aa_with_pred == aatype).float() * mask).sum()
                #     correct_gt = ((pred_aa_with_gt == aatype).float() * mask).sum()
                #
                #     if valid_count_pred > 0:
                #         diagnostics['recovery_with_pred_coords'] = (correct_pred / valid_count_pred).item()
                #     if valid_count_gt > 0:
                #         diagnostics['recovery_with_gt_coords'] = (correct_gt / valid_count_gt).item()

        # 添加velocity统计到diagnostics
        if logs:
            diagnostics['velocity_logs'] = logs
            # 计算平均velocity norm
            avg_v_norm = sum(log['v_norm_mean'] for log in logs) / len(logs)
            diagnostics['avg_velocity_norm'] = avg_v_norm

        return {
            'atoms14_local_final': atoms14_local_final,
            'atoms14_global_final': atoms14_global_final,
            'logits_final': final_logits,
            'diagnostics': diagnostics,
        }

    @torch.no_grad()
    def sh_sample_iterative(
            self,
            batch: dict,
            model,
            sh_params: dict,
            num_timesteps: int | None = None,
            sh_noise: torch.Tensor | None = None,
            return_traj: bool = False,
    ):
        """
        Deterministic SH density sampling with linear-bridge ODE integration.
        We integrate d(sh)/dt = (sh_clean - sh_t) / (1 - t) using Euler steps.
        """
        required_keys = ('normalize_density', 'atom14_element_idx', 'atom14_gt_exists')
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Batch is missing required key '{key}' for SH sampling.")

        device = batch['res_mask'].device
        base_sh = batch['normalize_density']
        density_mask = batch.get('density_mask')
        if density_mask is not None:
            density_mask = density_mask.to(base_sh.dtype)
            while density_mask.dim() < base_sh.dim():
                density_mask = density_mask.unsqueeze(-1)
            density_mask = density_mask.expand_as(base_sh)

        if density_mask is not None:
            base_sh_valid = base_sh * density_mask
        else:
            base_sh_valid = base_sh

        if sh_noise is None:
            sh_noise = torch.randn_like(base_sh_valid)
        if density_mask is not None:
            sh_noise = sh_noise * density_mask

        if num_timesteps is None:
            num_timesteps = max(2, self._sample_cfg.num_timesteps)

        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps, device=device)
        sh_t = sh_noise.clone()
        logits_final = None
        atoms14_local_final = None
        clean_est = None
        traj = [] if return_traj else None

        def _set_time_fields(target_batch, t_value):
            target_batch['t'] = torch.full(
                (batch['res_mask'].shape[0],),
                float(t_value),
                device=device,
                dtype=torch.float32,
            )
            target_batch['r3_t'] = torch.full_like(batch['res_mask'], float(t_value), device=device, dtype=torch.float32)
            target_batch['so3_t'] = torch.full_like(batch['res_mask'], float(t_value), device=device, dtype=torch.float32)

        for idx in range(len(ts) - 1):
            t_curr = float(ts[idx])
            t_next = float(ts[idx + 1])
            sh_input = sh_t if density_mask is None else sh_t * density_mask
            model_batch = copy.deepcopy(batch)
            model_batch['normalize_density'] = sh_input
            _set_time_fields(model_batch, t_curr)

            logits_out, atoms14_local = model(model_batch)
            logits_final = logits_out
            atoms14_local_final = atoms14_local

            clean_est, *_ = sh_density_from_atom14_with_masks(
                atoms14_local_final,
                batch['atom14_element_idx'],
                batch['atom14_gt_exists'],
                L_max=sh_params['L_max'],
                R_bins=sh_params['R_bins'],
                sigma_r=sh_params.get('sigma_r', 0.25),
            )
            clean_est = clean_est / torch.sqrt(torch.tensor(4 * torch.pi))
            if density_mask is not None:
                clean_est = clean_est * density_mask

            if traj is not None:
                traj.append({
                    't': t_curr,
                    'sh_norm': float(sh_input.norm().detach().cpu()),
                    'clean_norm': float(clean_est.norm().detach().cpu()),
                })

            dt = t_next - t_curr
            denom = max(1e-4, 1.0 - t_curr)
            sh_t = sh_t + dt * (clean_est - sh_t) / denom

        # Final evaluation at t=1.0
        final_batch = copy.deepcopy(batch)
        sh_input = sh_t if density_mask is None else sh_t * density_mask
        final_batch['normalize_density'] = sh_input
        _set_time_fields(final_batch, 1.0)

        logits_final, atoms14_local_final = model(final_batch)
        clean_est, *_ = sh_density_from_atom14_with_masks(
            atoms14_local_final,
            batch['atom14_element_idx'],
            batch['atom14_gt_exists'],
            L_max=sh_params['L_max'],
            R_bins=sh_params['R_bins'],
            sigma_r=sh_params.get('sigma_r', 0.25),
        )
        if density_mask is not None:
            clean_est = clean_est * density_mask

        if traj is not None:
            traj.append({
                't': 1.0,
                'sh_norm': float(sh_input.norm().detach().cpu()),
                'clean_norm': float(clean_est.norm().detach().cpu()),
            })

        if 'atoms14_local' in batch:
            atoms14_local_final = atoms14_local_final.clone()
            atoms14_local_final[..., :3, :] = batch['atoms14_local'][..., :3, :]

        result = {
            'atoms14_local_final': atoms14_local_final,
            'logits_final': logits_final,
            'sh_noise': sh_noise,
            'sh_pred_final': clean_est,
        }
        if traj is not None:
            result['trajectory'] = traj
        return result

    @torch.no_grad()
    def fbb_sample_iterative(
            self,
            batch: dict,
            model,
            num_timesteps: int | None = None,
    ):
        """One-step sidechain prediction using IGA regression model.

        IGA 模型直接预测最终坐标，不需要多步 ODE 积分。
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

        side_exists = batch.get('atom14_gt_exists', torch.ones_like(atoms14_local_gt[..., 0]))[..., 4:]  # [B,N,10] (IGA预测10个侧链原子)

        # ====================================================================
        # 一步直接推理：IGA 回归模式
        # ====================================================================
        # 准备输入特征
        input_feats = copy.deepcopy(batch)
        backbone_local = input_feats['atoms14_local_t'][..., :4, :]
        xt_side = input_feats['atoms14_local_t'][..., 4:, :]  # Masked sidechain (noisy input)

        # 组装完整的 14 原子输入
        atoms14_local_t = torch.cat([backbone_local, xt_side], dim=-2)

        # 设置 t=1.0 (最终状态)
        input_feats.update({
            't': torch.ones((res_mask.shape[0],), device=device, dtype=torch.float32),
            'r3_t': torch.ones(res_mask.shape, device=device, dtype=torch.float32),
            'so3_t': torch.ones(res_mask.shape, device=device, dtype=torch.float32),
            'atoms14_local_t': atoms14_local_t,
        })

        # 一步推理
        out = model(input_feats)

        # 获取预测的侧链坐标 (IGA 直接输出坐标，不是速度)
        pred_sidechain = out['pred_atoms']  # [B, N, 10, 3]
        final_logits = out.get('logits', None)

        # 组装完整的 14 原子 (backbone GT + predicted sidechain)
        atoms14_local_final = torch.cat([backbone_local, pred_sidechain], dim=-2)

        # 应用存在掩码
        if side_exists is not None:
            atoms14_local_final[..., 4:, :] = atoms14_local_final[..., 4:, :] * side_exists[..., None]

        # Build global 14 using fixed frames
        rigid = du.create_rigid(rotmats_1, trans_1)
        atoms14_global_final = rigid[..., None].apply(atoms14_local_final)

        # ===== 诊断：计算 RMSD =====
        diagnostics = {}
        if 'atoms14_local' in batch:
            gt_atoms14_local = batch['atoms14_local']  # [B,N,14,3]
            gt_sidechain = gt_atoms14_local[..., 4:, :]  # [B,N,10,3] (IGA 预测 10 个侧链原子)
            pred_sidechain = atoms14_local_final[..., 4:, :]  # [B,N,10,3]

            # 计算侧链RMSD
            if side_exists is not None:
                mask = side_exists[..., None].float()  # [B,N,10,1]
                diff = (pred_sidechain - gt_sidechain) ** 2  # [B,N,10,3]
                mse_per_atom = (diff * mask).sum(dim=-1)  # [B,N,10]
                rmsd_per_atom = torch.sqrt(mse_per_atom + 1e-8)  # [B,N,10]

                # 平均RMSD（只计算存在的原子）
                num_atoms = mask.sum()
                if num_atoms > 0:
                    mean_rmsd = (rmsd_per_atom * mask.squeeze(-1)).sum() / num_atoms
                    diagnostics['sidechain_rmsd'] = mean_rmsd.item()

                    # 每个残基的平均RMSD
                    rmsd_per_res = (rmsd_per_atom * mask.squeeze(-1)).sum(dim=-1) / (mask.squeeze(-1).sum(dim=-1) + 1e-8)  # [B,N]
                    diagnostics['rmsd_per_res'] = rmsd_per_res  # 保存用于详细分析

        return {
            'atoms14_local_final': atoms14_local_final,
            'atoms14_global_final': atoms14_global_final,
            'logits_final': final_logits,
            'diagnostics': diagnostics,
            'out':out
        }

    @torch.no_grad()
    def fbb_sample_iterativeR3(
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
        # 修复：num_timesteps是步数，需要num_timesteps+1个时间点 (包括起点和终点)
        # 例如：1步需要[t0, t1]两个点，10步需要[t0, t1, ..., t10]共11个点
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps , device=device)

        # init xt (sidechain local) at t = min_t with Gaussian noise

        # Prepare base features using the dedicated function
        input_feats_base = copy.deepcopy(batch)
        backbone_local = input_feats_base['atoms14_local_t'][..., :3, :]
        xt = input_feats_base['atoms14_local_t'][..., 3:, :]
        input_feats_base['atoms14_local_sc'] = torch.zeros_like(input_feats_base['atoms14_local_t'])

        logs = []

        for i in range(len(ts) - 1):
            t1 = float(ts[i])
            t2 = float(ts[i + 1])

            atoms14_local_t = torch.cat([backbone_local, xt], dim=-2)

            input_feats = input_feats_base.copy()
            input_feats.update({
                't': torch.full((res_mask.shape[0],), t1, device=device, dtype=torch.float32),
                'r3_t': torch.full(res_mask.shape, t1, device=device, dtype=torch.float32),
                'so3_t': torch.full(res_mask.shape, t1, device=device, dtype=torch.float32),
                'atoms14_local_t': atoms14_local_t,
            })

            #fbb


            out = model(input_feats)
            v_pred = out['speed_vectors']

            # 诊断：记录velocity统计
            v_norm = v_pred.norm(dim=-1)  # [B, N, 11]
            v_norm_masked = (v_norm * side_exists.float()).sum() / side_exists.sum()

            step_log = {
                'step': i,
                't': t1,
                'v_norm_mean': v_norm_masked.item(),
                'xt_norm_mean': xt.norm(dim=-1).mean().item(),
            }
            logs.append(step_log)

            # 每隔几步打印一次（可选）
            if i % max(1, (len(ts) - 1) // 5) == 0 or i == len(ts) - 2:
                print(f"  Step {i:3d} t={t1:.3f}: v_norm={v_norm_masked.item():.3f}Å  xt_norm={xt.norm(dim=-1).mean().item():.3f}Å")

            # Standard Euler ODE step for v = x1 - x0
            # x_t = (1-t)*x0 + t*x1, so dx/dt = x1 - x0 = v
            dt = t2 - t1
            xt = xt + dt * v_pred
            xt = xt * side_exists[..., None]  # mask out non-existing atoms

            # For self-conditioning (currently disabled with *0)
            clean_pred = xt + (1.0 - t2) * v_pred  # predict x1 at new t
            input_feats_base['atoms14_local_sc'] = torch.cat([backbone_local, clean_pred], dim=-2)*0



        # Final step aligned with structure sample(): run one more model call at t_final
        t_final = float(ts[-1])
        atoms14_local_t = torch.cat([backbone_local, xt], dim=-2)
        input_feats_final = input_feats_base.copy()
        input_feats_final.update({
            't': torch.full((res_mask.shape[0],), t_final, device=device, dtype=torch.float32),
            'r3_t': torch.full(res_mask.shape, t_final, device=device, dtype=torch.float32),
            'so3_t': torch.full(res_mask.shape, t_final, device=device, dtype=torch.float32),
            'atoms14_local_t': atoms14_local_t,
        })

        with torch.no_grad():
            out_final = model(input_feats_final)
        v_final = out_final['speed_vectors']
        final_logits = out_final.get('logits', None)
        clean_final = xt + (1.0 - t_final) * v_final

        atoms14_local_final = torch.cat([backbone_local, clean_final], dim=-2)
        if side_exists is not None:
            atoms14_local_final[..., 3:, :] = atoms14_local_final[..., 3:, :] * side_exists[..., None]

        # Build global 14 using fixed frames
        rigid = du.create_rigid(rotmats_1, trans_1)
        atoms14_global_final = rigid[..., None].apply(atoms14_local_final)

        # ===== 诊断：比较多步推理 vs GT的坐标误差和logits质量 =====
        diagnostics = {}
        if 'atoms14_local' in batch:
            gt_atoms14_local = batch['atoms14_local']  # [B,N,14,3]
            gt_sidechain = gt_atoms14_local[..., 3:, :]  # [B,N,11,3]
            pred_sidechain = atoms14_local_final[..., 3:, :]  # [B,N,11,3]

            # 计算侧链RMSD
            if side_exists is not None:
                mask = side_exists[..., None].float()  # [B,N,11,1]
                diff = (pred_sidechain - gt_sidechain) ** 2  # [B,N,11,3]
                mse_per_atom = (diff * mask).sum(dim=-1)  # [B,N,11]
                rmsd_per_atom = torch.sqrt(mse_per_atom + 1e-8)  # [B,N,11]

                # 平均RMSD（只计算存在的原子）
                num_atoms = mask.sum()
                if num_atoms > 0:
                    mean_rmsd = (rmsd_per_atom * mask.squeeze(-1)).sum() / num_atoms
                    diagnostics['sidechain_rmsd'] = mean_rmsd.item()

                    # 每个残基的平均RMSD
                    rmsd_per_res = (rmsd_per_atom * mask.squeeze(-1)).sum(dim=-1) / (mask.squeeze(-1).sum(dim=-1) + 1e-8)  # [B,N]
                    diagnostics['rmsd_per_res'] = rmsd_per_res  # 保存用于详细分析

            # 计算用GT坐标时的logits（对比实验）
            if final_logits is not None and 'aatype' in batch:
                # 用GT坐标重新跑一次模型，看logits会有多好
                atoms14_local_gt_input = torch.cat([backbone_local, gt_sidechain], dim=-2)
                input_feats_gt = input_feats_base.copy()
                input_feats_gt.update({
                    't': torch.full((res_mask.shape[0],), t_final, device=device, dtype=torch.float32),
                    'r3_t': torch.full(res_mask.shape, t_final, device=device, dtype=torch.float32),
                    'so3_t': torch.full(res_mask.shape, t_final, device=device, dtype=torch.float32),
                    'atoms14_local_t': atoms14_local_gt_input,
                })

                with torch.no_grad():
                    out_gt = model(input_feats_gt)
                logits_with_gt = out_gt.get('logits', None)

                if logits_with_gt is not None:
                    # 计算用GT坐标的perplexity
                    aatype = batch['aatype']
                    mask = batch.get('res_mask', torch.ones_like(aatype))

                    # Perplexity with predicted coords
                    ce_pred = torch.nn.functional.cross_entropy(
                        final_logits.reshape(-1, final_logits.shape[-1]),
                        aatype.reshape(-1),
                        reduction='none'
                    )
                    ce_pred = ce_pred.reshape(aatype.shape)
                    valid_count_pred = mask.sum()
                    if valid_count_pred > 0:
                        mean_ce_pred = (ce_pred * mask).sum() / valid_count_pred
                        ppl_pred = torch.exp(mean_ce_pred)
                        diagnostics['perplexity_with_pred_coords'] = ppl_pred.item()

                    # Perplexity with GT coords
                    ce_gt = torch.nn.functional.cross_entropy(
                        logits_with_gt.reshape(-1, logits_with_gt.shape[-1]),
                        aatype.reshape(-1),
                        reduction='none'
                    )
                    ce_gt = ce_gt.reshape(aatype.shape)
                    valid_count_gt = mask.sum()
                    if valid_count_gt > 0:
                        mean_ce_gt = (ce_gt * mask).sum() / valid_count_gt
                        ppl_gt = torch.exp(mean_ce_gt)
                        diagnostics['perplexity_with_gt_coords'] = ppl_gt.item()

                    # Recovery对比
                    pred_aa_with_pred = final_logits.argmax(dim=-1)
                    pred_aa_with_gt = logits_with_gt.argmax(dim=-1)

                    correct_pred = ((pred_aa_with_pred == aatype).float() * mask).sum()
                    correct_gt = ((pred_aa_with_gt == aatype).float() * mask).sum()

                    if valid_count_pred > 0:
                        diagnostics['recovery_with_pred_coords'] = (correct_pred / valid_count_pred).item()
                    if valid_count_gt > 0:
                        diagnostics['recovery_with_gt_coords'] = (correct_gt / valid_count_gt).item()

        # 添加velocity统计到diagnostics
        if logs:
            diagnostics['velocity_logs'] = logs
            # 计算平均velocity norm
            avg_v_norm = sum(log['v_norm_mean'] for log in logs) / len(logs)
            diagnostics['avg_velocity_norm'] = avg_v_norm

        return {
            'atoms14_local_final': atoms14_local_final,
            'atoms14_global_final': atoms14_global_final,
            'logits_final': final_logits,
            'diagnostics': diagnostics,
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
            't': t_next_tensor_b,
            'r3_t': t_next_tensor_bn,
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
        alpha_max = getattr(self._sample_cfg, 'alpha_max', 0.5)
        heun_steps = getattr(self._sample_cfg, 'heun_steps', 10)
        bridge_gamma_K = getattr(self._sample_cfg, 'bridge_gamma_steps', 2)
        bridge_gamma0 = getattr(self._sample_cfg, 'bridge_gamma0', 0.7)
        disp_cap_ang = getattr(self._sample_cfg, 'disp_cap_ang', 0.8)

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
                't': torch.full((res_mask.shape[0],), float(t1), device=device, dtype=torch.float32),
                'r3_t': torch.full(res_mask.shape, float(t1), device=device, dtype=torch.float32),
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

        t_eval = float(1)
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

    def guidance(self, trans_t, rotmats_t, model_out, motif_mask, R_motif, trans_motif, Log_delta_R, delta_x, t, d_t,
                 logs_traj):
        # Select motif
        motif_mask = motif_mask.clone()
        trans_pred = model_out['pred_trans'][:, motif_mask]  # [B, motif_res, 3]
        R_pred = model_out['pred_rotmats'][:, motif_mask]  # [B, motif_res, 3, 3]

        # Proposal for marginalising motif rotation
        F = twisting.motif_rots_vec_F(trans_motif, R_motif, self._cfg.twisting.num_rots, align=self._cfg.twisting.align,
                                      scale=self._cfg.twisting.scale_rots, device=self._device, dtype=torch.float32)

        # Estimate p(motif|predicted_motif)
        grad_Log_delta_R, grad_x_log_p_motif, logs = twisting.grad_log_lik_approx(R_pred, trans_pred, R_motif,
                                                                                  trans_motif, Log_delta_R, delta_x,
                                                                                  None, None, None, F,
                                                                                  twist_potential_rot=self._cfg.twisting.potential_rot,
                                                                                  twist_potential_trans=self._cfg.twisting.potential_trans)

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
            trans_t, rotmats_t = twisting.step(trans_t, rotmats_t, grad_x_log_p_motif, grad_Log_delta_R, d_t,
                                               trans_scale_t, rot_scale_t, self._cfg.twisting.update_trans,
                                               self._cfg.twisting.update_rot)

        # delete unsused arrays to prevent from any memory leak
        del grad_Log_delta_R
        del grad_x_log_p_motif
        del Log_delta_R
        del delta_x
        for key, value in model_out.items():
            model_out[key] = value.detach().requires_grad_(False)

        return trans_t, rotmats_t, logs_traj

    @torch.no_grad()
    def batch_test_ode_vs_sde(
            self,
            dataset,
            model,
            num_samples=50,
            num_timesteps=10,
            tau=0.3,
            w_cutoff=0.99,
    ):
        """
        批量测试 ODE vs SDE 采样的差异

        Args:
            dataset: 数据集对象
            model: 模型
            num_samples: 测试样本数
            num_timesteps: 采样步数
            tau: SDE温度参数
            w_cutoff: SDE扩散系数截断

        Returns:
            dict: 包含 ODE 和 SDE 结果的字典
        """
        from tqdm import tqdm
        import numpy as np

        def compute_sidechain_rmsd(pred_atoms, gt_atoms, exists_mask):
            """计算侧链RMSD"""
            diff = (pred_atoms - gt_atoms) ** 2  # [N, 11, 3]
            diff = diff.sum(dim=-1)  # [N, 11]
            rmsd_per_atom = torch.sqrt(diff + 1e-8)

            mask = exists_mask.float()
            num_atoms = mask.sum()
            if num_atoms > 0:
                mean_rmsd = (rmsd_per_atom * mask).sum() / num_atoms
            else:
                mean_rmsd = torch.tensor(0.0)

            return mean_rmsd.item()

        results_ode = []
        results_sde = []

        # 限制样本数量
        num_samples = min(num_samples, len(dataset))

        print(f"\n{'='*80}")
        print(f"批量测试 ODE vs SDE 采样 ({num_timesteps} 步)")
        print(f"{'='*80}")

        for sample_idx in tqdm(range(num_samples), desc="处理样本"):
            # 获取样本
            batch = dataset[sample_idx]

            # 转为batch格式并移到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].unsqueeze(0).to(self._device)

            # GT坐标
            gt_atoms14_local = batch['atoms14_local']  # [1, N, 14, 3]
            gt_sidechain = gt_atoms14_local[..., 3:, :]  # [1, N, 11, 3]
            side_exists = batch['atom14_gt_exists'][..., 3:]  # [1, N, 11]

            # 测试 ODE 采样
            prepared_ode = self.fbb_prepare_batch(batch)
            sample_out_ode = self.fbb_sample_iterative(
                prepared_ode,
                model,
                num_timesteps=num_timesteps
            )
            pred_sidechain_ode = sample_out_ode['atoms14_local_final'][..., 3:, :]
            rmsd_ode = compute_sidechain_rmsd(
                pred_sidechain_ode[0],
                gt_sidechain[0],
                side_exists[0]
            )
            diagnostics_ode = sample_out_ode.get('diagnostics', {})
            results_ode.append({
                'rmsd': rmsd_ode,
                'ppl_pred': diagnostics_ode.get('perplexity_with_pred_coords'),
                'recovery_pred': diagnostics_ode.get('recovery_with_pred_coords'),
            })

            # 测试 SDE 采样
            prepared_sde = self.fbb_prepare_batch(batch)
            sample_out_sde = self.fbb_sample_iterative_sde(
                prepared_sde,
                model,
                num_timesteps=num_timesteps,
                tau=tau,
                w_cutoff=w_cutoff,
            )
            pred_sidechain_sde = sample_out_sde['atoms14_local_final'][..., 3:, :]
            rmsd_sde = compute_sidechain_rmsd(
                pred_sidechain_sde[0],
                gt_sidechain[0],
                side_exists[0]
            )
            diagnostics_sde = sample_out_sde.get('diagnostics', {})
            results_sde.append({
                'rmsd': rmsd_sde,
                'ppl_pred': diagnostics_sde.get('perplexity_with_pred_coords'),
                'recovery_pred': diagnostics_sde.get('recovery_with_pred_coords'),
            })

        # 统计结果
        print(f"\n{'='*80}")
        print("测试结果汇总")
        print(f"{'='*80}")

        # ODE结果
        rmsd_ode_list = [r['rmsd'] for r in results_ode]
        ppl_ode_list = [r['ppl_pred'] for r in results_ode if r['ppl_pred'] is not None]
        recovery_ode_list = [r['recovery_pred'] for r in results_ode if r['recovery_pred'] is not None]

        print(f"\n[ODE 采样 - {num_timesteps}步]")
        print(f"  平均 RMSD: {np.mean(rmsd_ode_list):.4f} ± {np.std(rmsd_ode_list):.4f} Å")
        print(f"  中位 RMSD: {np.median(rmsd_ode_list):.4f} Å")
        if ppl_ode_list:
            print(f"  平均 PPL: {np.mean(ppl_ode_list):.3f}")
            print(f"  平均 Recovery: {np.mean(recovery_ode_list):.3f}")

        # SDE结果
        rmsd_sde_list = [r['rmsd'] for r in results_sde]
        ppl_sde_list = [r['ppl_pred'] for r in results_sde if r['ppl_pred'] is not None]
        recovery_sde_list = [r['recovery_pred'] for r in results_sde if r['recovery_pred'] is not None]

        print(f"\n[SDE 采样 (SimpleFold风格) - {num_timesteps}步]")
        print(f"  平均 RMSD: {np.mean(rmsd_sde_list):.4f} ± {np.std(rmsd_sde_list):.4f} Å")
        print(f"  中位 RMSD: {np.median(rmsd_sde_list):.4f} Å")
        if ppl_sde_list:
            print(f"  平均 PPL: {np.mean(ppl_sde_list):.3f}")
            print(f"  平均 Recovery: {np.mean(recovery_sde_list):.3f}")

        # 对比
        print(f"\n{'='*80}")
        print("对比分析")
        print(f"{'='*80}")

        mean_rmsd_ode = np.mean(rmsd_ode_list)
        mean_rmsd_sde = np.mean(rmsd_sde_list)
        diff_rmsd = mean_rmsd_sde - mean_rmsd_ode
        pct_change = (diff_rmsd / mean_rmsd_ode) * 100

        print(f"\nRMSD 差异:")
        print(f"  ODE: {mean_rmsd_ode:.4f} Å")
        print(f"  SDE: {mean_rmsd_sde:.4f} Å")
        print(f"  差值: {diff_rmsd:+.4f} Å ({pct_change:+.1f}%)")

        if diff_rmsd < -0.01:
            print(f"\n✓ SDE 采样的坐标质量更好！")
        elif abs(diff_rmsd) < 0.01:
            print(f"\n≈ 两种方法的坐标质量相近")
        else:
            print(f"\n✓ ODE 采样的坐标质量更好！")

        # 统计显著性
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(rmsd_ode_list, rmsd_sde_list)
        print(f"\n配对t检验: t={t_stat:.3f}, p={p_value:.4f}")
        if p_value < 0.05:
            print(f"  差异具有统计显著性 (p < 0.05)")
        else:
            print(f"  差异不具有统计显著性 (p >= 0.05)")

        return {
            'ode': results_ode,
            'sde': results_sde,
            'summary': {
                'ode_mean_rmsd': mean_rmsd_ode,
                'sde_mean_rmsd': mean_rmsd_sde,
                'diff_rmsd': diff_rmsd,
                'pct_change': pct_change,
                't_stat': t_stat,
                'p_value': p_value,
            }
        }
