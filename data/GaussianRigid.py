
import Bio.PDB
import math


# ==========================================
# 1. 核心类: GaussianRigid
# ==========================================
import torch
import numpy as np
from openfold.utils.rigid_utils import Rigid,Rotation


class OffsetGaussianRigid(Rigid):
    """
    OffsetGaussianRigid: 偏心高斯刚体 (卫星模型)。

    它是一个刚体 (Rigid, 代表主链 Frame)，但携带了一个挂载的高斯分布 (侧链)。

    Attributes:
        _rots (Rotation): 主链旋转 (决定坐标系方向)
        _trans (Tensor): 主链位置 (通常是 CA)
        _local_quat (Tensor): 侧链局部旋转 (q_local, [..., 4])
        _scaling (Tensor): 侧链形状 (Log-space)
        _local_mean (Tensor): 侧链质心在局部坐标系下的位置 (Offset)
    """

    def __init__(self, rots, trans,  scaling_log, local_mean,  chol=None,normalize_quats=True): #local_quat,
        """
        Args:
            rots: Backbone Rotation
            trans: Backbone Translation (CA)
            local_quat: Sidechain local rotation [..., 4]
            scaling_log: Sidechain scaling [..., 3]
            local_mean: Sidechain local offset [..., 3]
            normalize_quats: 是否对 local_quat 进行归一化 (推荐 True)
        """
        super().__init__(rots, trans)

        # # 1. 强制归一化 Local Quaternion (参考 Rotation 类)
        # if normalize_quats and local_quat is not None:
        #     norm = torch.linalg.norm(local_quat, dim=-1, keepdim=True)
        #     local_quat = local_quat / (norm + 1e-6)
        #
        # self._local_quat = local_quat
        self._scaling_log = scaling_log
        self._local_mean = local_mean
        self._chol = chol  # [...,3,3] or None

    @property
    def scaling(self):
        return torch.exp(self._scaling_log)

    def get_gaussian_mean(self):
        """
        【关键】获取高斯椭球在全局坐标系下的真实中心。
        Formula: mu_global = CA + R * local_offset
        使用 Rigid 自带的 apply 方法将局部偏移转为全局。
        """
        # Rigid.apply(pts) = R * pts + T
        # 这里 pts 就是 _local_mean
        return self.apply(self._local_mean)

    def get_covariance(self):
        """
        获取高斯椭球的协方差矩阵 (全局坐标系)。
        Formula: Sigma = R * S^2 * R^T
        """
        # [B, N, 3, 3]
        # # 组合旋转: R_total = R_bb * R_local
        # local_rot_obj = Rotation(quats=self._local_quat, normalize_quats=False)  # 已经归一化过了
        # total_rot_obj = self.get_rots().compose_r(local_rot_obj)
        # R = total_rot_obj.get_rot_mats()

        R = self.get_rots().get_rot_mats()

        if self._chol is None:
            # [B, N, 3]
            s = self.scaling

            # 1. 防止 s 过于微小导致下溢 (虽然 exp 保证正，但可能极小)
            # 这一步保证了 S^2 是“足够正”的
            s = torch.clamp(s, min=1e-6)

            # 构造对角方差矩阵 S^2
            S_squared = torch.diag_embed(s * s)
        else:
            L = torch.tril(self._chol)
            S_squared = L @ L.transpose(-1, -2)

        # 旋转协方差
        # Sigma = R * S^2 * R^T
        Sigma = R @ S_squared @ R.transpose(-1, -2)

        # 2. 由于 float32 误差，先显式对称化一下（很便宜）
        Sigma = (Sigma + Sigma.transpose(-1, -2)).mul_(0.5)  # inplace保持dtype

        # 3. 【关键】为了数值稳定性，加上微小的对角噪声 (Jitter)
        # 这就像给气球打一点点气，防止它瘪成二维纸片导致矩阵不可逆
        # Cholesky 对此非常敏感
        eps = 1e-6
        eye = torch.eye(3, device=Sigma.device, dtype=Sigma.dtype)
        Sigma = Sigma + eye * eps

        return Sigma


    def get_covariance_with_delta(self, delta_local_scale_log, min_s: float = 1e-6):
        """
        计算加上微扰后的全局协方差矩阵。
        支持传入 delta_scale，用于 Attention 探测或 Loss 计算，而不改变内部状态。
        """


        R = self.get_rots().get_rot_mats()

        # [B, N, 3]
        s = torch.exp(self._scaling_log + delta_local_scale_log)

        # 1. 防止 s 过于微小导致下溢 (虽然 exp 保证正，但可能极小)
        # 这一步保证了 S^2 是“足够正”的
        s = torch.clamp(s, min=float(min_s))

        # 构造对角方差矩阵 S^2
        S_squared = torch.diag_embed(s * s)

        # 旋转协方差
        # Sigma = R * S^2 * R^T
        Sigma = R @ S_squared @ R.transpose(-1, -2)

        # 2. 由于 float32 误差，先显式对称化一下（很便宜）
        Sigma = (Sigma + Sigma.transpose(-1, -2)).mul_(0.5)  # inplace保持dtype

        # 3. 【关键】为了数值稳定性，加上微小的对角噪声 (Jitter)
        # 这就像给气球打一点点气，防止它瘪成二维纸片导致矩阵不可逆
        # Cholesky 对此非常敏感
        eps = 1e-6
        eye = torch.eye(3, device=Sigma.device, dtype=Sigma.dtype)
        Sigma = Sigma + eye * eps

        return Sigma

    def unsqueeze(self, dim):
        """
        重写 unsqueeze，确保返回的是 OffsetGaussianRigid 而不是基类 Rigid。
        同时处理 scaling 和 local_mean 的维度扩展。
        """
        # 1. 调用父类获取变换后的 旋转和平移
        # 这会返回一个 Rigid 对象
        rigid_parent = super().unsqueeze(dim)

        # 2. 手动处理子类属性
        # 注意 Rigid.unsqueeze 的维度逻辑:
        # 它的内部数据 (如 _trans) 比 Rigid 的 shape 多一维 (坐标维)
        # 所以如果 dim < 0，需要再减 1 才能对齐到 batch dimensions
        # 参考 rigid_utils.py 源码: trans = self._trans.unsqueeze(dim if dim >= 0 else dim - 1)
        target_dim = dim if dim >= 0 else dim - 1

        new_scaling_log = self._scaling_log.unsqueeze(target_dim)
        new_local_mean = self._local_mean.unsqueeze(target_dim)

        # 3. 如果你有 local_quat (完全体版本)，也要处理
        # new_local_quat = self._local_quat.unsqueeze(target_dim) if hasattr(self, '_local_quat') else None

        # 4. 返回正确的子类对象
        return OffsetGaussianRigid(
            rigid_parent.get_rots(),
            rigid_parent.get_trans(),
            new_scaling_log,
            new_local_mean
            # local_quat=new_local_quat (如果用的是完全体)
        )
    def compose_update(self, update_vec):
        """
        【集成更新逻辑】
        将神经网络预测的 update_vec 应用到当前对象上，返回更新后的新对象。

        Args:
            update_vec: [..., 15] 维度的更新向量
                - 0:6   : Backbone Update (3 trans + 3 rot_vec)
                - 6:9   : Local Mean Update (3)
                - 9:12  : Scaling Update (3)
                - 12:15 : Local Quat Update (3, vector part) (remove)
        Returns:
            New OffsetGaussianRigid
        """
        # 1. 解包 Update
        # bb_update_vec = update_vec[..., :6]
        local_mean_delta = update_vec[..., :3]
        scaling_delta = update_vec[..., 3:]
       # local_quat_delta_vec = update_vec[..., 12:15]

        # 2. 更新 Backbone (利用 Rigid 自带的 compose_q_update_vec)
        # 这会自动处理主链的旋转累积和平移累积
        # new_rigid_bb = super().compose_q_update_vec(bb_update_vec)

        # 3. 更新 Local Mean (位置累加)
        new_local_mean = self._local_mean + local_mean_delta

        # 4. 更新 Scaling (对数累加 = 乘法)
        new_scaling_log = self._scaling_log + scaling_delta

        # # 5. 更新 Local Quaternion (利用 Rotation 类的方法)
        # # [你提议的方法]
        # # 先把当前的 tensor 包装成 Rotation 对象 (不需要再次 normalize，因为存储时已归一化)
        # current_local_rot_obj = Rotation(quats=self._local_quat, normalize_quats=False)
        #
        # # 调用 compose_q_update_vec 更新
        # # 它内部会自动处理 quat_multiply_by_vec + add
        # new_local_rot_obj = current_local_rot_obj.compose_q_update_vec(
        #     local_quat_delta_vec,
        #     normalize_quats=True  # 更新后顺便归一化，保证数值稳定
        # )
        #
        # # 取出更新后的 tensor
        # new_local_quat = new_local_rot_obj.get_quats()

        # 返回新对象
        return OffsetGaussianRigid(
            self.get_rots(),
            self.get_trans(),
            new_scaling_log,
            new_local_mean,

            # normalize_quats=False  # 前面已经 normalize 过了，这里 False 即可
        )

    def compose_update_12D(self, update_vec: torch.Tensor, update_mask: torch.Tensor = None,eps=1e-6):
        """
        update_vec:
          - 6D : rigid (qvec3 + t3)
          - 12D: rigid (0:6) + local_mean(6:9) + log_scale(9:12)
        """
        D = update_vec.shape[-1]
        if D not in (6, 12):
            raise ValueError(f"Expect 6 or 12 dims, got {D}")

        # openfold expects update_mask broadcastable to [..., 4] for quaternion updates,
        # so we standardize to shape [..., 1] here.
        m = update_mask
        if m is not None:
            if m.dim() == 2:
                m = m[..., None]
            elif m.dim() == 3 and m.shape[-1] == 1:
                pass
            else:
                raise ValueError(f"update_mask must be [B,N] or [B,N,1], got {tuple(m.shape)}")

        # (1) 全局 R,t
        bb_update = update_vec[..., :6]
        new_rigid = super().compose_q_update_vec(bb_update, update_mask=m)

        # Default: keep current gaussian params unless explicitly updated.
        local_mean_new = self._local_mean
        new_scaling_log = self._scaling_log

        if D == 12:
            d_alpha = update_vec[..., 6:9]
            d_log = update_vec[..., 9:12]

            if m is not None:
                d_alpha = d_alpha * m
                d_log = d_log * m


            # log-scale：加法更新 + clamp（非常重要）
            LOG_MIN, LOG_MAX = -6.0, 3.0

            scaling = torch.exp(self._scaling_log).clamp_min(eps)
            alpha = self._local_mean / scaling

            alpha_new = alpha + d_alpha
            new_scaling_log = (self._scaling_log + d_log).clamp(LOG_MIN, LOG_MAX)
            local_mean_new = alpha_new * torch.exp(new_scaling_log)

        return OffsetGaussianRigid(
            new_rigid._rots,
            new_rigid._trans,
            new_scaling_log,
            local_mean_new,
        )

    @staticmethod
    def from_atoms(n, ca, c, sidechain_atoms, sidechain_mask, base_thickness):
        """
        构建逻辑：
        1. Rigid: 由主链 N-CA-C 决定。
        2. Gaussian Mean: 由侧链原子质心决定。
        3. Local Offset: 计算 质心 相对于 Frame 的局部坐标。
        4. Scaling: 计算 侧链原子 相对于 质心 的分布。
        """
        # 1. 构建主链 Frame (Rigid)
        rigid_backbone = Rigid.from_3_points(p_neg_x_axis=n, origin=ca, p_xy_plane=c)
        rots_backbone = rigid_backbone.get_rots()

        # 2. 计算侧链的真实全局质心 (Global Centroid)
        mask = sidechain_mask.unsqueeze(-1)
        atom_count = mask.sum(dim=-2)
        sc_sum = (sidechain_atoms * mask).sum(dim=-2)

        # 全局质心 mu_global
        sc_centroid_global = torch.where(
            atom_count > 0,
            sc_sum / (atom_count + 1e-6),
            ca  # Glycine 回退到 CA
        )

        # 3. 计算 Local Offset (核心修改!)
        # 我们需要知道这个质心在主链 Frame 里在哪里
        # Local = Rigid.invert_apply(Global)
        # rigid_backbone 需要 unsqueeze 吗？这里是一对一，不需要
        local_mean = rigid_backbone.invert_apply(sc_centroid_global)

        # 4. 计算 Scaling (相对于质心的分布)
        # 4.1 先构建一个以“侧链质心”为原点，但旋转跟随主链的临时 Rigid
        # 用于把原子投影到去中心化的局部系
        rigid_sc_centered = Rigid(rots_backbone, sc_centroid_global)

        # 4.2 投影原子 -> 局部去中心化坐标
        local_atoms_centered = rigid_sc_centered.unsqueeze(-1).invert_apply(sidechain_atoms)
        local_atoms_masked = local_atoms_centered * mask

        # 4.3 计算标准差
        variance = (local_atoms_masked ** 2).sum(dim=-2) / (atom_count + 1e-6)
        std_dev = torch.sqrt(variance)

        scaling_linear = std_dev + base_thickness
        scaling_log = torch.log(scaling_linear + 1e-6)

        # 返回 OffsetGaussianRigid
        # 注意：这里传入的是 rots, ca (主链), scaling, local_mean (偏移)
        return OffsetGaussianRigid(
            rots_backbone,
            ca,
            scaling_log,
            local_mean
        )

    def scale_translation(self, factor: float):
        """
        对 OffsetGaussianRigid 进行全局缩放 (例如 Angstrom -> Nanometer)。

        需要同时处理:
        1. 主链平移 (_trans) -> 乘 factor
        2. 侧链偏移 (_local_mean) -> 乘 factor
        3. 侧链形状 (_scaling_log) -> 加 log(factor)
        """
        # 1. 缩放主链位置 (Linear)
        new_trans = self._trans * factor

        # 2. 缩放局部偏移 (Linear)
        # 物理意义：局部坐标系里的距离也要变小
        new_local_mean = self._local_mean * factor

        # 3. 缩放形状 (Log Space)
        # S_new = S_old * factor
        # log(S_new) = log(S_old) + log(factor)
        factor_log = math.log(factor)
        new_scaling_log = self._scaling_log + factor_log

        return OffsetGaussianRigid(
            self._rots,  # 旋转是不受尺度影响的 (无量纲)
            new_trans,
            new_scaling_log,
            new_local_mean
        )

    @classmethod
    def from_rigid_and_sidechain(
        cls,
        rigid_backbone: Rigid,          # 例如 AF2 / SimpleFold 的 backbone frame: [*, 3,3] + [*,3]
        sidechain_atoms: torch.Tensor,  # [..., N_sc, 3]
        sidechain_mask: torch.Tensor,   # [..., N_sc]
        base_thickness: torch.Tensor,
    ):
        """
        已有 backbone 刚体时，根据侧链构造 OffsetGaussianRigid。
        等价于 from_atoms 但不需要 N/CA/C，直接用 backbone frame。
        """
        rots_backbone = rigid_backbone.get_rots()
        trans_backbone = rigid_backbone.get_trans()  # 通常是 CA

        # 下面逻辑和 update_from_sidechain 基本相同
        mask = sidechain_mask.unsqueeze(-1)
        atom_count = mask.sum(dim=-2)
        sc_sum = (sidechain_atoms * mask).sum(dim=-2)

        sc_centroid_global = torch.where(
            atom_count > 0,
            sc_sum / (atom_count + 1e-6),
            trans_backbone,
        )

        local_mean = rigid_backbone.invert_apply(sc_centroid_global)
        # print(local_mean[...,3,:])

        rigid_sc_centered = Rigid(rots_backbone, sc_centroid_global)
        local_atoms_centered = rigid_sc_centered.unsqueeze(-1).invert_apply(sidechain_atoms)
        local_atoms_masked = local_atoms_centered * mask

        atom_count_safe = atom_count + 1e-6
        variance = (local_atoms_masked ** 2).sum(dim=-2) / atom_count_safe
        std_dev = torch.sqrt(variance)

        scaling_linear = std_dev + base_thickness
        scaling_log = torch.log(scaling_linear + 1e-6)

        return OffsetGaussianRigid(
            rots=rots_backbone,
            trans=trans_backbone,
            scaling_log=scaling_log,
            local_mean=local_mean,
        )

    @staticmethod
    def from_backbone_atoms( n, ca, c, o, mask, base_thickness=0.4):
        """
        仅基于骨架原子 (N, CA, C, O) 生成高斯。
        用于调试 Frame 的方向和主链的局部几何分布。
        """
        # 1. 构建主链 Frame (Rigid)
        # 这一步决定了椭圆的主轴方向 (R)
        rigid_backbone = Rigid.from_3_points(p_neg_x_axis=n, origin=ca, p_xy_plane=c)
        rots_backbone = rigid_backbone.get_rots()

        # 2. 整理骨架原子 [B, N, 4, 3]
        bb_atoms = torch.stack([n, ca, c, o], dim=2)
        # 骨架 mask通常是全1，但为了严谨支持 padding
        # mask shape: [B, N] -> [B, N, 4]
        if mask.dim() == 2:
            mask_exp = mask.unsqueeze(-1).expand_as(bb_atoms[..., 0])
        else:
            mask_exp = mask

        # 3. 计算质心 (Centroid)
        # [B, N, 4, 1]
        m = mask_exp.unsqueeze(-1)
        count = m.sum(dim=-2).clamp(min=1.0)
        centroid = (bb_atoms * m).sum(dim=-2) / count

        # 4. 计算 Offset
        # 质心相对于 CA 的位置
        local_mean = rigid_backbone.invert_apply(centroid)

        # 5. 计算 Scaling (标准差)
        # 5.1 以质心为原点，但保持主链旋转
        rigid_centered = Rigid(rots_backbone, centroid)
        # 5.2 投影到局部
        # [B, N, 4, 3]
        local_atoms_centered = rigid_centered.unsqueeze(-1).invert_apply(bb_atoms)

        # 5.3 统计方差
        # 沿着主链的 x, y, z 轴的分布情况
        variance = ((local_atoms_centered ** 2) * m).sum(dim=-2) / count
        std_dev = torch.sqrt(variance + 1e-8)

        # 加上一点基础厚度，否则像肽平面这种扁平结构会导致协方差奇异
        scaling_log = torch.log(std_dev + base_thickness + 1e-6)

        return OffsetGaussianRigid(
            rots_backbone,
            ca,
            scaling_log,
            local_mean
        )
    @staticmethod
    def from_all_atoms(n, ca, c, o, sidechain_atoms, sidechain_mask, base_thickness=0.5):
        """
        从全原子 (Backbone + Sidechain) 构建高斯。
        不使用 PCA，不使用加权，完全基于统计。

        逻辑:
        1. R (Rotation): 严格跟随主链 (N-CA-C) Frame。
        2. μ (Centroid): 该氨基酸所有原子 (N,CA,C,O + Sidechain) 的几何中心。
        3. S (Scaling): 该氨基酸所有原子相对于质心的标准差 (轴对齐到主链 Frame)。
        4. Offset: 质心相对于 CA 的局部位移。
        """
        # 1. 构建主链 Frame (提供 R 和 CA锚点)
        rigid_backbone = Rigid.from_3_points(p_neg_x_axis=n, origin=ca, p_xy_plane=c)
        rots_backbone = rigid_backbone.get_rots()

        # 2. 整理全原子数据
        # Backbone Atoms: [B, N, 4, 3]
        bb_atoms = torch.stack([n, ca, c, o], dim=2)
        # Backbone Mask: [B, N, 4] (全1，因为骨架总是存在的)
        bb_mask = torch.ones(bb_atoms.shape[:-1], device=bb_atoms.device, dtype=sidechain_mask.dtype)

        # Sidechain Atoms: [B, N, M, 3]
        # 拼接得到 All Atoms
        all_atoms = torch.cat([bb_atoms, sidechain_atoms], dim=2)  # [B, N, K, 3]
        all_mask = torch.cat([bb_mask, sidechain_mask], dim=2)  # [B, N, K]

        # 3. 计算全原子质心 (Global Centroid)
        # mask: [B, N, K, 1]
        mask_exp = all_mask.unsqueeze(-1)
        # 原子总数 (分母)
        atom_count = mask_exp.sum(dim=-2).clamp(min=1.0)

        # 坐标求和
        all_sum = (all_atoms * mask_exp).sum(dim=-2)
        # 质心 μ
        centroid_global = all_sum / atom_count

        # 4. 计算 Local Offset (核心!)
        # 我们需要知道这个全原子质心，在主链局部坐标系里的位置
        # Local_Mean = R_bb^T * (Centroid - CA)
        local_mean = rigid_backbone.invert_apply(centroid_global)

        # 5. 计算 Scaling (标准差)
        # 5.1 构造一个临时的 "中心化刚体"
        # 原点在质心，但旋转方向依然是主链方向
        rigid_centered = Rigid(rots_backbone, centroid_global)
        rigid_centered_exp = rigid_centered.unsqueeze(-1)  # 广播到原子维度

        # 5.2 将所有原子投影到局部去中心化坐标系
        local_atoms_centered = rigid_centered_exp.invert_apply(all_atoms)
        local_atoms_masked = local_atoms_centered * mask_exp

        # 5.3 计算方差: Sum(x^2) / N
        # 这里的方差是沿着主链 Frame 的 X, Y, Z 轴统计的
        variance = (local_atoms_masked ** 2).sum(dim=-2) / atom_count
        std_dev = torch.sqrt(variance + 1e-8)

        # 加上基础厚度
        scaling_log = torch.log(std_dev + base_thickness + 1e-6)

        # 6. 返回 OffsetGaussianRigid
        return OffsetGaussianRigid(
            rots_backbone,  # R (Follow Backbone)
            ca,  # Trans (Anchor Point)
            scaling_log,  # S (Whole Residue Shape)
            local_mean  # Offset (Centroid relative to CA)
        )


    @classmethod
    def from_rigid_and_all_atoms(
            cls,
            rigid_backbone: Rigid,  # 已有的主链 Frame
            all_atoms: torch.Tensor,  # [..., N_all, 3] 全原子坐标 (Backbone + Sidechain)
            all_atom_mask: torch.Tensor,  # [..., N_all] 全原子 Mask
            base_thickness: torch.Tensor,
    ):
        """
        已有 backbone 刚体时，根据全原子 (Backbone + Sidechain) 构造 OffsetGaussianRigid。
        替代 from_all_atoms，省去了从 N/CA/C 重建 Frame 的步骤。

        逻辑:
        1. R: 直接复用输入的 rigid_backbone。
        2. μ (Centroid): 计算输入 all_atoms 的几何中心。
        3. Offset: 计算 μ 相对于 rigid_backbone 的局部位移。
        4. S (Scaling): 计算 all_atoms 相对于 μ 的标准差 (轴对齐到 rigid_backbone)。
        """
        # 1. 获取主链旋转和位置 (锚点)
        rots_backbone = rigid_backbone.get_rots()
        trans_backbone = rigid_backbone.get_trans()  # 通常是 CA

        # 2. 计算全原子质心 (Global Centroid)
        # mask: [..., K, 1]
        mask_exp = all_atom_mask.unsqueeze(-1)
        # 原子总数 (分母)
        atom_count = mask_exp.sum(dim=-2).clamp(min=1.0)

        # 坐标求和
        all_sum = (all_atoms * mask_exp).sum(dim=-2)

        # 质心 μ
        # 注意: 如果某个残基完全没有原子 (atom_count=0), 下面的除法靠 clamp 保护
        # 但物理上我们应该回退到 CA (trans_backbone)
        has_atoms = (all_atom_mask.sum(dim=-1) > 0.5)  # [..., ]

        centroid_global = torch.where(
            has_atoms.unsqueeze(-1),
            all_sum / atom_count,
            trans_backbone  # Fallback to CA
        )

        # 3. 计算 Local Offset
        # 我们需要知道这个全原子质心，在主链局部坐标系里的位置
        # Local_Mean = R_bb^T * (Centroid - CA)
        local_mean = rigid_backbone.invert_apply(centroid_global)

        # 4. 计算 Scaling (标准差)
        # 4.1 构造一个临时的 "中心化刚体"
        # 原点在质心，但旋转方向依然是主链方向
        rigid_centered = Rigid(rots_backbone, centroid_global)
        # 广播到原子维度: [..., 1]
        rigid_centered_exp = rigid_centered.unsqueeze(-1)

        # 4.2 将所有原子投影到局部去中心化坐标系
        # 这一步把原子转到了以质心为原点，且方向与主链对齐的坐标系中
        local_atoms_centered = rigid_centered_exp.invert_apply(all_atoms)
        local_atoms_masked = local_atoms_centered * mask_exp

        # 4.3 计算方差: Sum(x^2) / N
        # 这里的方差是沿着主链 Frame 的 X, Y, Z 轴统计的
        variance = (local_atoms_masked ** 2).sum(dim=-2) / atom_count
        std_dev = torch.sqrt(variance + 1e-8)

        # =================================================================
        # [DEBUG] 统计 Local Mean 与 Std Dev 的关系 (Per Axis)
        # =================================================================
        # if torch.rand(1).item() < 1:  # 1% 概率打印
        #     try:
        #         # local_mean: [..., N, 3]
        #         # std_dev:    [..., N, 3]
        #
        #         valid_mask = has_atoms.float()
        #         denom = valid_mask.sum().clamp_min(1.0)
        #
        #         # 1. 整体模长统计 (复用之前的)
        #         mean_norm = torch.norm(local_mean, dim=-1)
        #         std_norm = torch.norm(std_dev, dim=-1)
        #         avg_mean = (mean_norm * valid_mask).sum() / denom
        #         avg_std = (std_norm * valid_mask).sum() / denom
        #
        #         # 2. 分轴统计 (X, Y, Z)
        #         # abs_mean: [..., N, 3]
        #         abs_mean = local_mean.abs()
        #         abs_std = std_dev
        #
        #         # Ratio per axis: [..., N, 3]
        #         ratio_axis = abs_mean / (abs_std + 1e-6)
        #
        #         # Mask out invalid residues
        #         ratio_axis = ratio_axis * valid_mask.unsqueeze(-1)
        #
        #         # Max ratio per axis
        #         # 先在 N 维度取 max，再在 Batch 维度取 max
        #         # 注意：如果 batch size > 1，需要展平前几维
        #         flat_ratio = ratio_axis.view(-1, 3)
        #         max_ratio_xyz = flat_ratio.max(dim=0)[0] # [3]
        #
        #         # Ratio > 2.0 per axis
        #         gt2_xyz = (ratio_axis > 2.0).float().sum(dim=-2).sum(dim=0) / denom # [3]
        #         gt3_xyz = (ratio_axis > 3.0).float().sum(dim=-2).sum(dim=0) / denom # [3]
        #
        #         # 转换为 Python float 列表以便打印
        #         max_r = max_ratio_xyz.tolist()
        #         gt2 = gt2_xyz.tolist()
        #         gt3 = gt3_xyz.tolist()
        #
        #         print(f"\n[GaussianRigid DEBUG - Per Axis]")
        #         print(f"  Avg Norm |M|: {avg_mean:.4f} A, |S|: {avg_std:.4f} A")
        #         print(f"  Max Ratio (X, Y, Z): [{max_r[0]:.2f}, {max_r[1]:.2f}, {max_r[2]:.2f}]")
        #         print(f"  Ratio > 2.0 (X, Y, Z): [{gt2[0]*100:.2f}%, {gt2[1]*100:.2f}%, {gt2[2]*100:.2f}%]")
        #         print(f"  Ratio > 3.0 (X, Y, Z): [{gt3[0]*100:.2f}%, {gt3[1]*100:.2f}%, {gt3[2]*100:.2f}%]")
        #         print("=================================================================\n")
        #     except Exception as e:
        #         print(f"[GaussianRigid DEBUG Error] {e}")
        # # =================================================================

        # 加上基础厚度
        scaling_log = torch.log(std_dev + base_thickness + 1e-6)

        # 5. 返回 OffsetGaussianRigid 对象
        return cls(
            rots=rots_backbone,
            trans=trans_backbone,
            scaling_log=scaling_log,
            local_mean=local_mean,
            # normalize_quats=True # 如果你的 __init__ 有这个参数
        )
# ==========================================
# 导出 PDB + ANISOU (通用)
# ==========================================


def save_gaussian_as_pdb(
    gaussian_rigid,
    filename: str,
    res_names=None,
    mask: torch.Tensor | None = None,
    center_mode: str = "gaussian_mean",
):
    if center_mode == "gaussian_mean":
        xyz = gaussian_rigid.get_gaussian_mean()
    elif center_mode == "trans":
        xyz = gaussian_rigid.get_trans()
    else:
        raise ValueError(center_mode)

    covs = gaussian_rigid.get_covariance()

    xyz = xyz.detach().cpu()
    covs = covs.detach().cpu()

    if mask is not None:
        mask_f = (mask.detach().cpu().reshape(-1) > 0.5)
    else:
        mask_f = torch.ones(xyz.numel() // 3, dtype=torch.bool)

    xyz = xyz.reshape(-1, 3)[mask_f].numpy()
    covs = covs.reshape(-1, 3, 3)[mask_f].numpy()

    N = xyz.shape[0]
    if res_names is None or len(res_names) != N:
        res_names = ["GLY"] * N

    chain_id = "A"
    alt_loc = " "
    i_code = " "
    occupancy = 1.00
    b_iso = 1.00
    element = "C"   # 你用 CA 原子只是占位，元素随便但要放到列 77-78
    charge = "  "

    with open(filename, "w") as f:
        f.write("HEADER    GAUSSIAN_PROTEIN\n")
        # 可选：写个 CRYST1，避免某些软件按晶胞坐标脑补
        # f.write("CRYST1   1.000   1.000   1.000  90.00  90.00  90.00 P 1           1\n")

        for i in range(N):
            serial = i + 1
            name = "CA"
            res = res_names[i]
            res_seq = i + 1  # 不要用 serial 当 residue id；用自己的序号即可

            x, y, z = xyz[i]
            c = covs[i]

            # ANISOU 需要 1e-4 Å^2 的整数；用 round，别用 int 截断
            u11 = int(round(c[0, 0] * 1e4))
            u22 = int(round(c[1, 1] * 1e4))
            u33 = int(round(c[2, 2] * 1e4))
            u12 = int(round(c[0, 1] * 1e4))
            u13 = int(round(c[0, 2] * 1e4))
            u23 = int(round(c[1, 2] * 1e4))

            # ——严格 PDB 列宽——
            # ATOM: columns follow PDB v3 fixed width
            f.write(
                f"ATOM  {serial:5d} {name:>4s}{alt_loc:1s}{res:>3s} {chain_id:1s}"
                f"{res_seq:4d}{i_code:1s}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{occupancy:6.2f}{b_iso:6.2f}          "
                f"{element:>2s}{charge:2s}\n"
            )

            # ANISOU: U’s are 6 integers, each 7 columns
            f.write(
                f"ANISOU{serial:5d} {name:>4s}{alt_loc:1s}{res:>3s} {chain_id:1s}"
                f"{res_seq:4d}{i_code:1s} "
                f"{u11:7d}{u22:7d}{u33:7d}{u12:7d}{u13:7d}{u23:7d}          "
                f"{element:>2s}{charge:2s}\n"
            )

        f.write("END\n")

    print(f"Saved: {filename} (N={N}, center_mode={center_mode})")




# ==========================================
# 主程序
# ==========================================
def process_pdb_and_compare(pdb_path):
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    n, ca, c, o = [], [], [], []
    sc_coords, sc_masks = [], []
    res_names = []

    MAX_SC = 14

    for res in structure.get_residues():
        if Bio.PDB.is_aa(res, standard=True):
            if not (res.has_id('N') and res.has_id('CA') and res.has_id('C') and res.has_id('O')): continue

            n.append(res['N'].get_coord())
            ca.append(res['CA'].get_coord())
            c.append(res['C'].get_coord())
            o.append(res['O'].get_coord())  # 需要 O 原子
            res_names.append(res.get_resname())

            atoms = []
            for a in res:
                if a.name not in ['N', 'CA', 'C', 'O', 'OXT'] and a.element != 'H':
                    atoms.append(a.get_coord())

            tmp_c = np.zeros((MAX_SC, 3));
            tmp_m = np.zeros((MAX_SC))
            L = min(len(atoms), MAX_SC)
            if L > 0:
                tmp_c[:L] = np.array(atoms)[:L]
                tmp_m[:L] = 1.0
            sc_coords.append(tmp_c);
            sc_masks.append(tmp_m)

    # To Tensor
    def t(x):
        return torch.tensor(np.array(x), dtype=torch.float32).unsqueeze(0)

    n_t, ca_t, c_t, o_t = t(n), t(ca), t(c), t(o)
    sc_t, mask_t = t(sc_coords), t(sc_masks)

    # --- 1. 生成侧链主导高斯 (SC Only) ---
    gr_sc = OffsetGaussianRigid.from_atoms(n_t, ca_t, c_t, sc_t, mask_t, base_thickness=0.4)
    save_gaussian_as_pdb(gr_sc, f"{pdb_path[:-4]}_SC_only.pdb", res_names)

    N = n_t.shape[1]
    bb_mask_t = torch.ones((1, N), dtype=torch.float32)
    gr_bb = OffsetGaussianRigid.from_backbone_atoms(n_t, ca_t, c_t, o_t, bb_mask_t, base_thickness=0.4)
    save_name = f"{pdb_path[:-4]}_Backbone.pdb"
    save_gaussian_as_pdb(gr_bb, save_name, res_names)
    # # --- 2. 生成全原子高斯 (All Atoms) ---
    # gr_all = OffsetGaussianRigid.from_atom14_weighted(n_t, ca_t, c_t, o_t, sc_t, mask_t, base_thickness=0.8)
    # save_gaussian_as_pdb(gr_all, f"{pdb_path[:-4]}_ALL_atoms.pdb", res_names)


import numpy as np
import torch
from plyfile import PlyData, PlyElement  # 需要 pip install plyfile


def save_parents_as_ply(parents, filename="parents_data.ply"):
    """
    将 ParentParams 保存为标准 3DGS PLY 格式。
    保存内容: x, y, z, n_x, n_y, n_z (作为旋转), scale_0~2, rot_0~3 (四元数)
    """
    # 1. 提取数据并转为 CPU numpy
    # parents.s 是 [B, K, 3], parents.R 是 [B, K, 3, 3]
    # 我们这里展平 Batch 和 K 维度
    mu = parents.mu.detach().cpu().numpy().reshape(-1, 3)
    s = parents.s.detach().cpu().numpy().reshape(-1, 3)
    R = parents.R.detach().cpu().numpy().reshape(-1, 3, 3)
    mask = parents.mask_parent.detach().cpu().numpy().reshape(-1)

    # 过滤掉 mask 为 0 的无效点
    valid_idx = mask > 0.5
    xyz = mu[valid_idx]
    scales = s[valid_idx]
    rots = R[valid_idx]

    # 2. 将旋转矩阵转为四元数 (xyzw) 用于保存
    # 这里手动简易转换，或者使用 scipy.spatial.transform.Rotation
    from scipy.spatial.transform import Rotation
    # 注意：Rotation 库要求行列式为+1，V3代码里我们已经修正了手性，所以这里是安全的
    qs = Rotation.from_matrix(rots).as_quat()  # [N, 4] -> (x, y, z, w)

    # 3. 构建结构化数组
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),  # 仅仅为了兼容某些viewer，可填0
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),  # 颜色 (RGB)
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),  # Quaternion
        ('opacity', 'f4')
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)

    # 填充数据
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = 0
    elements['ny'] = 0
    elements['nz'] = 0

    # 颜色设为粉色 (RGB: 1, 0.5, 0.5) -> SH 转换有点麻烦，这里简化存 RGB
    elements['f_dc_0'] = 1.0
    elements['f_dc_1'] = 0.5
    elements['f_dc_2'] = 0.5

    # 3DGS 标准通常存 log(scale)，但为了通用性，这里存原始 scale
    # 如果你是给标准 3DGS 查看器用，建议存 np.log(scales + 1e-6)
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]

    # Rotation (w, x, y, z) vs (x, y, z, w) 取决于查看器，PLY通常存 xyzw
    elements['rot_0'] = qs[:, 3]  # w
    elements['rot_1'] = qs[:, 0]  # x
    elements['rot_2'] = qs[:, 1]  # y
    elements['rot_3'] = qs[:, 2]  # z

    elements['opacity'] = 1.0

    # 4. 写入文件
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(filename)
    print(f"Saved PLY data to {filename}")


def save_parents_as_pdb_manual_cov(parents, filename):
    """
    直接利用 mu, R, s 计算协方差并保存 PDB。
    完全绕过 OffsetGaussianRigid 类，避免黑盒导致的轴序错乱。
    """
    # 1. 提取数据
    mu = parents.mu.detach().cpu().numpy().reshape(-1, 3)  # [Total_K, 3]
    s = parents.s.detach().cpu().numpy().reshape(-1, 3)  # [Total_K, 3]
    R = parents.R.detach().cpu().numpy().reshape(-1, 3, 3)  # [Total_K, 3, 3]
    mask = parents.mask_parent.detach().cpu().numpy().reshape(-1)

    # 过滤无效点
    valid_idx = mask > 0.5
    mu = mu[valid_idx]
    s = s[valid_idx]
    R = R[valid_idx]

    N = len(mu)

    # 2. 【核心】手动计算协方差矩阵 Sigma = R * S^2 * R.T
    # 这一步是纯数学，绝对不会有 "长短轴反了" 的问题
    # s 也就是 radii
    covs = np.zeros((N, 3, 3))
    for i in range(N):
        # 构建对角阵 S^2
        S2 = np.diag(s[i] ** 2)
        # R @ S^2 @ R.T
        covs[i] = R[i] @ S2 @ R[i].T

    # 3. 写入 PDB (带数值保护)
    with open(filename, "w") as f:
        f.write("HEADER    MANUAL_COV_PARENTS\n")
        for i in range(N):
            serial = i + 1
            x, y, z = mu[i]

            # 写入 ATOM 行
            f.write(
                f"ATOM  {serial:5d}  CA  GLY A{serial:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  1.00           C  \n"
            )

            # 写入 ANISOU 行
            c = covs[i]
            u_vals = [c[0, 0], c[1, 1], c[2, 2], c[0, 1], c[0, 2], c[1, 2]]

            # 数值保护 (PDB 限制)
            LIMIT = 9999999
            u_int = []
            for v in u_vals:
                val = int(v * 10000)  # PDB 单位
                val = max(min(val, LIMIT), -LIMIT)
                u_int.append(val)

            f.write(
                f"ANISOU{serial:5d}  CA  GLY A{serial:4d} "
                f"{u_int[0]:7d}{u_int[1]:7d}{u_int[2]:7d}{u_int[3]:7d}{u_int[4]:7d}{u_int[5]:7d}       C  \n"
            )
        f.write("END\n")
    print(f"Saved PDB data to {filename}")





def save_parents_as_pdb_explicit(parents, filename, limit=9999999):
    """
    [推荐] 显式计算协方差并保存为 PDB。
    完全绕过 OffsetGaussianRigid 和 Rotation 类，彻底解决长短轴反转问题。
    """
    # 1. 提取数据 (CPU numpy)
    mu = parents.mu.detach().cpu().numpy().reshape(-1, 3)  # [Total_N, 3]
    s = parents.s.detach().cpu().numpy().reshape(-1, 3)  # [Total_N, 3]
    R = parents.R.detach().cpu().numpy().reshape(-1, 3, 3)  # [Total_N, 3, 3]
    mask = parents.mask_parent.detach().cpu().numpy().reshape(-1)

    # 过滤无效点
    valid_idx = mask > 0.5
    mu = mu[valid_idx]
    s = s[valid_idx]
    R = R[valid_idx]

    N = len(mu)
    print(f"Saving {N} ellipsoids to {filename}...")

    # 2. 【核心】手动计算协方差矩阵 Sigma = R * S^2 * R^T
    # 这一步是纯数学，绝对不会出错
    covs = np.zeros((N, 3, 3))
    for i in range(N):
        # 构建对角阵 S^2 (方差)
        S2 = np.diag(s[i] ** 2)
        # 矩阵乘法: (3,3) = (3,3) @ (3,3) @ (3,3)
        # 这就是协方差的定义，没有任何歧义
        covs[i] = R[i] @ S2 @ R[i].T

    # 3. 写入 PDB
    with open(filename, "w") as f:
        f.write("HEADER    EXPLICIT_COV_PARENTS\n")
        for i in range(N):
            serial = i + 1
            x, y, z = mu[i]

            # 写入原子坐标 (CA)
            f.write(
                f"ATOM  {serial:5d}  CA  GLY A{serial:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  1.00           C  \n"
            )

            # 写入各向异性 (ANISOU)
            c = covs[i]
            # 取出 PDB 需要的 6 个分量 (上三角)
            # U11, U22, U33, U12, U13, U23
            u_vals = [c[0, 0], c[1, 1], c[2, 2], c[0, 1], c[0, 2], c[1, 2]]

            # 数值转换与保护
            u_int = []
            for v in u_vals:
                val = int(v * 10000)  # PDB 单位转换
                # 强制截断，防止数据爆炸导致格式错乱
                val = max(min(val, limit), -limit)
                u_int.append(val)

            f.write(
                f"ANISOU{serial:5d}  CA  GLY A{serial:4d} "
                f"{u_int[0]:7d}{u_int[1]:7d}{u_int[2]:7d}{u_int[3]:7d}{u_int[4]:7d}{u_int[5]:7d}       C  \n"
            )
        f.write("END\n")
    print(f"[Success] PDB saved: {filename}")



if __name__ == "__main__":
    import sys, os

    # 使用你之前的 pdb
    pdb = "1fna.pdb"
    if os.path.exists(pdb):
        process_pdb_and_compare(pdb)
    else:
        print("PDB not found")
