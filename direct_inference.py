"""
Direct one-step inference for FBB sidechain prediction.
No diffusion, no iterative sampling - just one forward pass.
"""

import os
import torch
import numpy as np
from pathlib import Path
from openfold.np import residue_constants
from openfold.utils import rigid_utils as ru
from data import utils as du


def direct_fbb_inference(
    model,
    interpolant,
    batch: dict[str, torch.Tensor],
    output_dir: str,
    t_inference: float = 1.0,
) -> dict[str, torch.Tensor]:
    """
    直接一步生成侧链结构（无扩散过程）

    Args:
        model: 训练好的模型
        interpolant: Interpolant 对象（用于数据准备）
        batch: 输入数据 batch
        output_dir: 输出目录
        t_inference: 推理时刻（默认 1.0，表示直接预测干净结构）

    Returns:
        包含预测结果的字典
    """
    device = batch['res_mask'].device
    B, N = batch['res_mask'].shape

    # ============================================
    # 1. 准备输入数据（固定backbone，mask掉sidechain）
    # ============================================
    with torch.no_grad():
        # 使用 interpolant 的 fbb_corrupt_batch 准备数据
        prepared_batch = interpolant.fbb_corrupt_batch(batch)

        # 设置推理时刻 t
        prepared_batch['t'] = torch.full((B,), t_inference, device=device, dtype=torch.float32)
        prepared_batch['r3_t'] = torch.full((B, N), t_inference, device=device, dtype=torch.float32)
        prepared_batch['so3_t'] = torch.full((B, N), t_inference, device=device, dtype=torch.float32)

        # ============================================
        # 2. 模型前向传播（一步生成）
        # ============================================
        model.eval()
        out = model(prepared_batch)

        # 提取输出
        logits = out['logits']  # [B, N, 21] 氨基酸类型预测
        atoms14_local = out['atoms14_local']  # [B, N, 14, 3] 局部坐标

        # ============================================
        # 3. 后处理：用GT backbone替换predicted backbone
        # ============================================
        if 'atoms14_local' in batch:
            atoms14_local = atoms14_local.clone()
            # 保留GT backbone (前3个原子: N, CA, C)
            atoms14_local[..., :3, :] = batch['atoms14_local'][..., :3, :]

        # ============================================
        # 4. 转换到全局坐标系
        # ============================================
        rigid = ru.Rigid.from_tensor_7(batch['rigids_1'])
        atoms14_global = rigid[..., None].apply(atoms14_local)

        # ============================================
        # 5. 预测氨基酸序列
        # ============================================
        aa_pred = logits.argmax(dim=-1)  # [B, N]
        aa_true = batch.get('aatype', aa_pred)
        res_mask = batch['res_mask']

        # ============================================
        # 6. 保存结果
        # ============================================
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取样本ID
        sample_ids = batch.get('csv_idx')
        if sample_ids is None:
            sample_ids = torch.arange(B)
        sample_ids = sample_ids.squeeze().tolist()
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids

        # 氨基酸索引到名称映射
        idx_to_aa = {residue_constants.restype_order[aa]: aa for aa in residue_constants.restypes}

        predictions = []
        for b, sample_id in enumerate(sample_ids):
            sid = int(sample_id) if not isinstance(sample_id, int) else sample_id
            tag = f"{sid:06d}"

            # 创建样本目录
            sample_dir = output_dir / f'sample_{tag}'
            sample_dir.mkdir(exist_ok=True)

            # 提取该样本的有效残基
            mask = res_mask[b].bool()
            n_res = mask.sum().item()

            # 序列
            aa_true_seq = ''.join([idx_to_aa.get(int(x), 'X') for x in aa_true[b][mask].tolist()])
            aa_pred_seq = ''.join([idx_to_aa.get(int(x), 'X') for x in aa_pred[b][mask].tolist()])

            # 坐标
            pred_coords = atoms14_global[b][mask].cpu().numpy()  # [N_res, 14, 3]
            atom14_exists = batch.get('atom14_gt_exists', torch.ones_like(atoms14_global[..., 0]))[b][mask].cpu().numpy()

            # ============================================
            # 7. 保存为 PDB 文件
            # ============================================
            pdb_path = sample_dir / 'pred.pdb'
            _save_pdb(
                pdb_path,
                pred_coords,
                aa_pred_seq,
                atom14_exists,
                chain_id='A'
            )

            # 保存序列信息
            fasta_path = sample_dir / 'sequence.fasta'
            with open(fasta_path, 'w') as f:
                f.write(f'>sample_{tag}_predicted\n{aa_pred_seq}\n')
                f.write(f'>sample_{tag}_ground_truth\n{aa_true_seq}\n')

            # 保存诊断信息
            diag_path = sample_dir / 'diagnostics.txt'
            with open(diag_path, 'w') as f:
                f.write(f"Sample ID: {sid}\n")
                f.write(f"Number of residues: {n_res}\n")
                f.write(f"Predicted sequence: {aa_pred_seq}\n")
                f.write(f"Ground truth sequence: {aa_true_seq}\n")
                f.write(f"Sequence identity: {sum([a == b for a, b in zip(aa_pred_seq, aa_true_seq)]) / n_res * 100:.2f}%\n")

            predictions.append({
                'sample_id': sid,
                'pred_seq': aa_pred_seq,
                'true_seq': aa_true_seq,
                'pdb_path': str(pdb_path),
            })

            print(f"[DirectInference] Saved sample {tag} to {sample_dir}")

        return {
            'logits': logits,
            'atom14_local': atoms14_local,
            'atom14_global': atoms14_global,
            'aa_pred': aa_pred,
            'predictions': predictions,
        }


def _save_pdb(
    pdb_path: Path,
    coords: np.ndarray,  # [N, 14, 3]
    sequence: str,  # length N
    atom_mask: np.ndarray,  # [N, 14]
    chain_id: str = 'A',
):
    """保存预测的结构为 PDB 文件"""

    atom_types = residue_constants.atom_types  # ['N', 'CA', 'C', 'O', ...]

    with open(pdb_path, 'w') as f:
        atom_index = 1
        for res_idx, (aa, res_coords, res_mask) in enumerate(zip(sequence, coords, atom_mask), start=1):
            resname = residue_constants.restype_1to3.get(aa, 'UNK')

            for atom_idx, (atom_name, xyz, exists) in enumerate(zip(atom_types, res_coords, res_mask)):
                if exists < 0.5:  # 原子不存在
                    continue

                x, y, z = xyz
                # PDB format: ATOM line
                line = f"ATOM  {atom_index:5d}  {atom_name:<3s} {resname:>3s} {chain_id}{res_idx:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}\n"
                f.write(line)
                atom_index += 1

        f.write("END\n")


# ============================================
# 便捷的推理脚本
# ============================================
def run_direct_inference_from_checkpoint(
    ckpt_path: str,
    data_csv: str,
    output_dir: str,
    device: str = 'cuda',
    batch_size: int = 1,
    num_samples: int = None,
):
    """
    从 checkpoint 加载模型并运行直接推理

    Args:
        ckpt_path: 模型 checkpoint 路径
        data_csv: 数据 CSV 路径
        output_dir: 输出目录
        device: 运行设备
        batch_size: batch size
        num_samples: 推理的样本数量（None = 全部）
    """
    import pytorch_lightning as pl
    from models.flow_module import SE3FlowModule
    from data.datasets import get_dataloader

    # 加载模型
    print(f"Loading checkpoint from {ckpt_path}")
    module = SE3FlowModule.load_from_checkpoint(ckpt_path, map_location=device)
    module.eval()
    module = module.to(device)

    model = module.model
    interpolant = module.interpolant

    # 准备数据
    print(f"Loading data from {data_csv}")
    # 这里需要根据你的数据加载方式调整
    # 示例：使用 module 的配置创建 dataloader

    # TODO: 根据实际情况实现数据加载
    # dataloader = get_dataloader(...)

    print("Running direct inference (no diffusion)...")
    # for batch_idx, batch in enumerate(dataloader):
    #     if num_samples and batch_idx >= num_samples:
    #         break
    #
    #     batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    #
    #     result = direct_fbb_inference(
    #         model=model,
    #         interpolant=interpolant,
    #         batch=batch,
    #         output_dir=output_dir,
    #         t_inference=1.0,
    #     )
    #
    #     print(f"Batch {batch_idx} done")

    print(f"All samples saved to {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Direct one-step FBB inference')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data_csv', type=str, required=True, help='Data CSV path')
    parser.add_argument('--output_dir', type=str, default='./direct_inference_output', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to process')

    args = parser.parse_args()

    run_direct_inference_from_checkpoint(
        ckpt_path=args.ckpt,
        data_csv=args.data_csv,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )
