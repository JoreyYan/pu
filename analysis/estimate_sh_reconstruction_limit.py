import argparse
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets import PdbDataset
from data.ele_atoms import build_elem_slot_maps
from data.sh_density import sh_density_from_atom14_with_masks


def load_sample(cfg, dataset_cfg, sample_idx: int):
    task = cfg.data.task

    dataset = PdbDataset(
        dataset_cfg=dataset_cfg,
        is_training=False,
        task=task,
        is_predict=True,
    )
    sample = dataset[sample_idx]

    required_keys = [
        "atoms14_local",
        "atom14_gt_exists",
        "atom14_element_idx",
        "normalize_density",
        "density_mask",
    ]
    for key in required_keys:
        if key not in sample:
            raise KeyError(f"Sample missing key '{key}'. Available keys: {list(sample.keys())}")

    return sample


def compute_density(coords, elem_idx, atom_mask, L_max, R_bins, sigma_r):
    density, *_ = sh_density_from_atom14_with_masks(
        coords.unsqueeze(0).unsqueeze(0),
        elem_idx.unsqueeze(0).unsqueeze(0),
        atom_mask.unsqueeze(0).unsqueeze(0),
        L_max=L_max,
        R_bins=R_bins,
        sigma_r=sigma_r,
    )
    density = density.squeeze(0).squeeze(0) / math.sqrt(4 * math.pi)
    return density


def optimize_residue(gt_coords, elem_idx, atom_mask, target_density, density_mask, *,
                     L_max, R_bins, sigma_r, device,
                     noise_std=1.0, adam_steps=300, adam_lr=0.01, lbfgs_steps=50):
    atom_mask = atom_mask.to(device)
    atom_mask_bool = atom_mask.bool()
    num_atoms = atom_mask_bool.sum().item()
    if num_atoms == 0:
        return None

    gt_coords = gt_coords.to(device)
    elem_idx = elem_idx.to(device)

    target_density = target_density.to(device)
    density_mask = density_mask.to(device)

    init_coords = gt_coords + noise_std * torch.randn_like(gt_coords)
    param = torch.nn.Parameter(init_coords)

    def density_loss():
        pred_density = compute_density(param, elem_idx, atom_mask, L_max, R_bins, sigma_r)
        diff = pred_density - target_density
        if density_mask is not None and density_mask.numel() == diff.numel():
            weight = density_mask
        else:
            weight = torch.ones_like(diff)
        loss = (diff ** 2 * weight).mean()
        return loss

    if adam_steps > 0:
        optimizer = torch.optim.Adam([param], lr=adam_lr)
        for _ in range(adam_steps):
            optimizer.zero_grad()
            loss = density_loss()
            loss.backward()
            optimizer.step()

    if lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS([param], lr=1.0, max_iter=lbfgs_steps, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            loss = density_loss()
            loss.backward()
            return loss

        lbfgs.step(closure)

    optimized = param.detach()
    mask = atom_mask_bool.unsqueeze(-1)
    mse = ((optimized - gt_coords) ** 2 * mask).sum() / mask.sum()
    rmsd = torch.sqrt(mse).item()
    return {
        "rmsd": rmsd,
        "optimized": optimized.detach().cpu(),
        "gt": gt_coords.detach().cpu(),
        "mask": atom_mask_bool.detach().cpu(),
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate SH reconstruction limit via numerical optimization.")
    parser.add_argument("--config", default="configs/Train_SH.yaml", help="Config file to load dataset/model settings.")
    parser.add_argument("--sample_idx", type=int, default=0, help="Starting dataset sample index.")
    parser.add_argument("--sample_idxs", type=str, default=None, help="Comma-separated list of sample indices to evaluate.")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to evaluate if sample_idxs is not provided.")
    parser.add_argument("--num_residues", type=int, default=20, help="Number of residues to reconstruct (ordered by mask).")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Gaussian noise std added to initial coordinates (Å).")
    parser.add_argument("--adam_steps", type=int, default=1000)
    parser.add_argument("--adam_lr", type=float, default=0.001)
    parser.add_argument("--lbfgs_steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    model_root = OmegaConf.load("configs/SH.yaml")
    OmegaConf.resolve(model_root)
    cfg = OmegaConf.merge(model_root, cfg)

    datasets_root = OmegaConf.load("configs/datasets.yaml")
    OmegaConf.resolve(datasets_root)
    dataset_name = cfg.data.dataset
    dataset_cfg_node = datasets_root[f"{dataset_name}_dataset"]
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(dataset_cfg_node, resolve=True))

    if args.sample_idxs:
        sample_indices = [int(x.strip()) for x in args.sample_idxs.split(",") if x.strip()]
    else:
        sample_indices = list(range(args.sample_idx, args.sample_idx + args.num_samples))

    maps = build_elem_slot_maps()

    overall_rmsd = []
    atom_sqerr = torch.zeros(14)
    atom_counts = torch.zeros(14)

    L_max = cfg.model.sh.L_max
    R_bins = cfg.model.sh.R_bins
    sigma_r = 0.25  # matches dataset preprocessing

    device = torch.device(args.device)

    total_residues = 0

    for sample_idx in sample_indices:
        sample = load_sample(cfg, dataset_cfg, sample_idx)
        atoms14 = sample["atoms14_local"].float()
        atom_exists = sample["atom14_gt_exists"].float()
        atom_elem_idx = sample["atom14_element_idx"]
        if atom_elem_idx.numel() == 0:
            elem_idx_tensor = maps["elem14"][sample["aatype"].long()].clone()
        else:
            elem_idx_tensor = atom_elem_idx.long()
        density = sample["normalize_density"].float()
        density_mask = sample["density_mask"].float()
        if density_mask.ndim == 5:
            density_mask = density_mask.expand(density.shape[0], density.shape[1], density.shape[2], density.shape[3], density.shape[4]).clone()

        residue_indices = torch.nonzero(sample["res_mask"], as_tuple=False).squeeze(-1).tolist()

        print(f"\n=== Sample {sample_idx} ===")
        for res_idx in residue_indices[:args.num_residues]:
            result = optimize_residue(
                gt_coords=atoms14[res_idx],
                elem_idx=elem_idx_tensor[res_idx],
                atom_mask=atom_exists[res_idx],
                target_density=density[res_idx],
                density_mask=density_mask[res_idx],
                L_max=L_max,
                R_bins=R_bins,
                sigma_r=sigma_r,
                device=device,
                noise_std=args.noise_std,
                adam_steps=args.adam_steps,
                adam_lr=args.adam_lr,
                lbfgs_steps=args.lbfgs_steps,
            )
            if result is not None:
                total_residues += 1
                overall_rmsd.append(result["rmsd"])
                print(f"Residue {res_idx:3d}: RMSD = {result['rmsd']:.4f} Å")
                diff = (result["optimized"] - result["gt"]).pow(2).sum(dim=-1)
                mask = result["mask"]
                atom_sqerr += (diff * mask).to(atom_sqerr.dtype)
                atom_counts += mask.to(atom_counts.dtype)
            else:
                print(f"Residue {res_idx:3d}: skipped (no atoms).")

    if overall_rmsd:
        mean_rmsd = sum(overall_rmsd) / len(overall_rmsd)
        print(f"\nEvaluated {len(overall_rmsd)} residues across {len(sample_indices)} samples.")
        print(f"Mean RMSD: {mean_rmsd:.4f} Å")
        print(f"Max  RMSD: {max(overall_rmsd):.4f} Å")
        print(f"Min  RMSD: {min(overall_rmsd):.4f} Å")
        valid = atom_counts > 0
        per_atom_rmsd = torch.zeros_like(atom_counts)
        per_atom_rmsd[valid] = torch.sqrt(atom_sqerr[valid] / atom_counts[valid])
        print("\nPer-atom RMSD (atom index 0-13):")
        for i in range(14):
            if atom_counts[i] > 0:
                print(f"  atom{i:2d}: {per_atom_rmsd[i].item():.4f} Å (n={int(atom_counts[i].item())})")
    else:
        print("No residues evaluated.")


if __name__ == "__main__":
    main()
