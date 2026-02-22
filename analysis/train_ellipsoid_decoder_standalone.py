"""Phase A2: Standalone two-head MLP decoder — 6D → aatype + atom14.

Usage:
    python analysis/train_ellipsoid_decoder_standalone.py [--epochs 50] [--lr 3e-4]

Architecture:
    Input: scaling_log[3] + local_mean[3] = 6D per residue
      ↓ Shared stem: LayerNorm → Linear → GELU → 256D
      ↓ 2× MLPResBlock(256, 1024)
      ├─→ aa_head: Linear → 20 logits (CrossEntropy)
      └─→ atom14_head: Linear(256 + aatype_embed_64) → 42D → [14, 3]
          Training: GT aatype (teacher forcing)
          Eval: predicted aatype

Evaluates:
    - aatype top-1 accuracy, perplexity
    - atom14 RMSD per residue type
    - GT aatype vs predicted aatype atom14 RMSD comparison
"""

import sys, os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import utils as du
from data.GaussianRigid import OffsetGaussianRigid
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.np import residue_constants
from openfold.data import data_transforms


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPResBlock(nn.Module):
    def __init__(self, d_in, d_hidden, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_in)
        self.ln = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = F.gelu(self.fc1(x))
        y = self.dropout(y)
        y = self.fc2(y)
        return self.ln(x + y)


class TwoHeadEllipsoidDecoder(nn.Module):
    """Standalone 6D → (aatype, atom14) decoder for hypothesis validation."""

    def __init__(
        self,
        d_model: int = 256,
        d_hidden: int = 1024,
        num_blocks: int = 2,
        aatype_embed_dim: int = 64,
        num_aa_types: int = 21,
        dropout: float = 0.1,
        out_range: float = 16.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.out_range = out_range
        self.num_aa_types = num_aa_types

        # Shared stem: 6D → d_model
        self.stem = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, d_model),
            nn.GELU(),
        )

        # Shared blocks
        self.blocks = nn.ModuleList([
            MLPResBlock(d_model, d_hidden, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # AA head: d_model → 20 logits
        self.aa_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 20),
        )

        # Atom14 head: d_model + aatype_embed → 14*3
        self.aa_embed = nn.Embedding(num_aa_types, aatype_embed_dim)
        self.atom14_proj = nn.Sequential(
            nn.LayerNorm(d_model + aatype_embed_dim),
            nn.Linear(d_model + aatype_embed_dim, d_model),
            nn.GELU(),
        )
        self.atom14_blocks = nn.ModuleList([
            MLPResBlock(d_model, d_hidden, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.atom14_head = nn.Linear(d_model, 14 * 3)
        nn.init.zeros_(self.atom14_head.weight)
        nn.init.zeros_(self.atom14_head.bias)

        # atom14 mask per type
        atom14_mask = torch.tensor(
            residue_constants.restype_atom14_mask, dtype=torch.float32
        )
        if atom14_mask.shape[0] == 20:
            atom14_mask = torch.cat([
                atom14_mask,
                torch.zeros(1, 14, dtype=torch.float32)
            ], dim=0)
        self.register_buffer('atom14_mask_per_type', atom14_mask)

    def forward(self, scaling_log, local_mean, aatype=None):
        """
        Args:
            scaling_log: [B, N, 3]
            local_mean:  [B, N, 3]
            aatype:      [B, N] int, if None uses predicted aatype

        Returns:
            aa_logits:    [B, N, 20]
            atom14_local: [B, N, 14, 3]
        """
        B, N = scaling_log.shape[:2]
        feat = torch.cat([scaling_log, local_mean], dim=-1)  # [B, N, 6]

        x = self.stem(feat)
        for blk in self.blocks:
            x = blk(x)  # [B, N, d_model]

        # AA prediction
        aa_logits = self.aa_head(x)  # [B, N, 20]

        # Atom14 prediction
        if aatype is not None:
            aa_idx = aatype.clamp(0, self.num_aa_types - 1).long()
        else:
            aa_idx = aa_logits.argmax(dim=-1).clamp(0, self.num_aa_types - 1)
        aa_emb = self.aa_embed(aa_idx)  # [B, N, D]

        atom_feat = torch.cat([x, aa_emb], dim=-1)
        atom_feat = self.atom14_proj(atom_feat)
        for blk in self.atom14_blocks:
            atom_feat = blk(atom_feat)
        out = self.atom14_head(atom_feat)
        out = torch.tanh(out) * self.out_range
        atom14_local = out.view(B, N, 14, 3)

        # Mask invalid atoms
        type_mask = self.atom14_mask_per_type[aa_idx]
        atom14_local = atom14_local * type_mask.unsqueeze(-1)

        return aa_logits, atom14_local


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EllipsoidDataset(Dataset):
    """Loads per-protein features for standalone decoder training."""

    def __init__(self, csv_path, max_samples=5000, max_res=400, min_res=60):
        import pandas as pd
        import tree

        df = pd.read_csv(csv_path)
        if 'modeled_seq_len' in df.columns:
            df = df[(df['modeled_seq_len'] >= min_res) & (df['modeled_seq_len'] <= max_res)]
        if 'oligomeric_detail' in df.columns:
            df = df[df['oligomeric_detail'] == 'monomeric']

        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)

        self.samples = []
        skipped = 0

        for _, row in df.iterrows():
            processed_path = row.get('processed_path')
            if processed_path is None or not os.path.exists(processed_path):
                skipped += 1
                continue

            try:
                processed_feats = du.read_pkl(processed_path)
                processed_feats = du.parse_chain_feats(processed_feats)

                modeled_idx = processed_feats['modeled_idx']
                min_idx_val = np.min(modeled_idx)
                max_idx_val = np.max(modeled_idx)
                del processed_feats['modeled_idx']
                processed_feats = tree.map_structure(
                    lambda x: x[min_idx_val:(max_idx_val + 1)], processed_feats)

                aatype = torch.tensor(processed_feats['aatype']).long()
                all_atom_positions = torch.tensor(processed_feats['atom_positions']).float()
                all_atom_mask = torch.tensor(processed_feats['atom_mask']).float()
                bb_mask = torch.tensor(processed_feats['bb_mask']).int()

                chain_feats = {
                    'aatype': aatype,
                    'all_atom_positions': all_atom_positions,
                    'all_atom_mask': all_atom_mask,
                    'atom14_element_idx': torch.tensor(processed_feats['atom14_element_idx']).int(),
                }
                chain_feats = data_transforms.atom37_to_frames(chain_feats)
                chain_feats = data_transforms.make_atom14_masks(chain_feats)
                chain_feats = data_transforms.make_atom14_positions(chain_feats)

                rigids_1 = Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
                rotmats_1 = rigids_1.get_rots().get_rot_mats()
                trans_1 = rigids_1.get_trans()
                rigids_1 = Rigid(Rotation(rotmats_1), trans_1)

                dynamic_thickness = torch.where(
                    ~bb_mask.bool(),
                    torch.tensor(2.5),
                    torch.tensor(0.0),
                ).unsqueeze(-1)

                offset_rigid = OffsetGaussianRigid.from_rigid_and_all_atoms(
                    rigids_1,
                    chain_feats['atom14_gt_positions'],
                    chain_feats['atom14_gt_exists'],
                    base_thickness=dynamic_thickness,
                )
                scaling_log_1 = offset_rigid._scaling_log
                local_mean_1 = offset_rigid._local_mean

                # Compute atom14 in local frame
                atom14_local = rigids_1[..., None].invert_apply(
                    chain_feats['atom14_gt_positions']
                )

                self.samples.append({
                    'scaling_log_1': scaling_log_1.detach(),
                    'local_mean_1': local_mean_1.detach(),
                    'aatype': aatype,
                    'atom14_local': atom14_local.float(),
                    'atom14_gt_exists': chain_feats['atom14_gt_exists'].float(),
                    'res_mask': bb_mask.float(),
                })

            except Exception:
                skipped += 1
                continue

        print(f"EllipsoidDataset: loaded {len(self.samples)} proteins, skipped {skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad to max length in batch."""
    max_len = max(b['aatype'].shape[0] for b in batch)
    out = {}
    for key in batch[0]:
        tensors = []
        for b in batch:
            t = b[key]
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                pad_shape = [pad_len] + list(t.shape[1:])
                t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0)
            tensors.append(t)
        out[key] = torch.stack(tensors)
    return out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Phase A2: Standalone two-head MLP decoder")
    print("=" * 60)

    # Dataset
    print("\nLoading dataset...")
    dataset = EllipsoidDataset(args.csv_path, max_samples=args.max_samples)

    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)

    # Model
    model = TwoHeadEllipsoidDecoder(
        d_model=256, d_hidden=1024, num_blocks=2,
        aatype_embed_dim=64, dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train: {n_train}, Val: {n_val}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")

    best_val_acc = 0
    history = {'train_loss': [], 'val_acc': [], 'val_atom14_rmsd': []}

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0
        epoch_aa_correct = 0
        epoch_total = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            mask = batch['res_mask']
            valid_aa = (batch['aatype'] < 20) & mask.bool()

            # Forward with teacher forcing
            aa_logits, atom14_pred = model(
                batch['scaling_log_1'], batch['local_mean_1'],
                aatype=batch['aatype'],
            )

            # AA loss
            aa_loss_per = F.cross_entropy(
                aa_logits[valid_aa],
                batch['aatype'][valid_aa],
                reduction='mean',
            )

            # Atom14 loss (MSE on existing atoms)
            atom14_mask = batch['atom14_gt_exists'] * mask.unsqueeze(-1)
            sq_err = ((atom14_pred - batch['atom14_local']) ** 2).sum(dim=-1)
            atom14_loss = (sq_err * atom14_mask).sum() / (atom14_mask.sum() * 3 + 1e-8)

            loss = aa_loss_per + atom14_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * valid_aa.sum().item()
            epoch_aa_correct += (aa_logits[valid_aa].argmax(-1) == batch['aatype'][valid_aa]).sum().item()
            epoch_total += valid_aa.sum().item()

        scheduler.step()
        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_aa_correct / max(epoch_total, 1)

        # --- Validate ---
        model.eval()
        val_aa_correct = 0
        val_total = 0
        val_atom14_sq_err = 0
        val_atom14_count = 0
        # Also track with predicted aatype
        val_atom14_sq_err_pred = 0
        val_atom14_count_pred = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                mask = batch['res_mask']
                valid_aa = (batch['aatype'] < 20) & mask.bool()

                # With GT aatype
                aa_logits, atom14_gt_aa = model(
                    batch['scaling_log_1'], batch['local_mean_1'],
                    aatype=batch['aatype'],
                )

                # With predicted aatype
                _, atom14_pred_aa = model(
                    batch['scaling_log_1'], batch['local_mean_1'],
                    aatype=None,
                )

                val_aa_correct += (aa_logits[valid_aa].argmax(-1) == batch['aatype'][valid_aa]).sum().item()
                val_total += valid_aa.sum().item()

                atom14_mask = batch['atom14_gt_exists'] * mask.unsqueeze(-1)
                sq_err_gt = ((atom14_gt_aa - batch['atom14_local']) ** 2).sum(dim=-1)
                val_atom14_sq_err += (sq_err_gt * atom14_mask).sum().item()
                val_atom14_count += atom14_mask.sum().item()

                sq_err_pred = ((atom14_pred_aa - batch['atom14_local']) ** 2).sum(dim=-1)
                val_atom14_sq_err_pred += (sq_err_pred * atom14_mask).sum().item()
                val_atom14_count_pred += atom14_mask.sum().item()

        val_acc = val_aa_correct / max(val_total, 1)
        val_rmsd_gt = np.sqrt(val_atom14_sq_err / max(val_atom14_count, 1))
        val_rmsd_pred = np.sqrt(val_atom14_sq_err_pred / max(val_atom14_count_pred, 1))

        # Perplexity estimate
        val_perplexity = np.exp(min(train_loss, 10))

        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_atom14_rmsd'].append(val_rmsd_gt)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.1%} | "
              f"Val Acc: {val_acc:.1%} | "
              f"Atom14 RMSD (GT aa): {val_rmsd_gt:.3f}A | "
              f"Atom14 RMSD (pred aa): {val_rmsd_pred:.3f}A")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best_standalone_decoder.pt')

    # --- Final evaluation with per-type RMSD ---
    print("\n" + "=" * 60)
    print("Final evaluation on validation set")
    print("=" * 60)

    model.load_state_dict(torch.load(output_dir / 'best_standalone_decoder.pt', weights_only=True))
    model.eval()

    from models.ellipsoid_decoder import sidechain_rmsd_per_type

    all_aa_logits = []
    all_aa_true = []
    all_masks = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            mask = batch['res_mask']

            aa_logits, atom14_gt_aa = model(
                batch['scaling_log_1'], batch['local_mean_1'],
                aatype=batch['aatype'],
            )
            _, atom14_pred_aa = model(
                batch['scaling_log_1'], batch['local_mean_1'],
                aatype=None,
            )

            # Per-type RMSD with GT aatype
            rmsd_gt = sidechain_rmsd_per_type(
                atom14_gt_aa, batch['atom14_local'],
                batch['atom14_gt_exists'], batch['aatype'], mask,
            )
            rmsd_pred = sidechain_rmsd_per_type(
                atom14_pred_aa, batch['atom14_local'],
                batch['atom14_gt_exists'], batch['aatype'], mask,
            )

            valid_aa = (batch['aatype'] < 20) & mask.bool()
            all_aa_logits.append(aa_logits[valid_aa].cpu())
            all_aa_true.append(batch['aatype'][valid_aa].cpu())

    all_aa_logits = torch.cat(all_aa_logits)
    all_aa_true = torch.cat(all_aa_true)
    final_acc = (all_aa_logits.argmax(-1) == all_aa_true).float().mean().item()

    ce = F.cross_entropy(all_aa_logits, all_aa_true).item()
    final_perplexity = np.exp(min(ce, 10))

    print(f"\nFinal aatype accuracy: {final_acc:.1%}")
    print(f"Final aatype perplexity: {final_perplexity:.2f}")
    print(f"Best val accuracy: {best_val_acc:.1%}")

    if rmsd_gt:
        print("\nSidechain RMSD per type (GT aatype):")
        for aa, rmsd in sorted(rmsd_gt.items()):
            pred_rmsd = rmsd_pred.get(aa, float('nan'))
            print(f"  {aa}: GT={rmsd:.3f}A  Pred={pred_rmsd:.3f}A")

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()

    ax2.plot(history['val_acc'], label='Val AA Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'standalone_decoder_training.png', dpi=150)
    plt.close()

    # Update results file
    results_path = output_dir / 'ellipsoid_aatype_results.md'
    with open(results_path, 'a') as f:
        f.write(f"\n\n# Standalone Two-Head MLP Decoder Results\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| aatype top-1 accuracy | {final_acc:.1%} |\n")
        f.write(f"| aatype perplexity | {final_perplexity:.2f} |\n")
        f.write(f"| atom14 RMSD (GT aatype) | {val_rmsd_gt:.3f} A |\n")
        f.write(f"| atom14 RMSD (pred aatype) | {val_rmsd_pred:.3f} A |\n")
        f.write(f"\nEpochs: {args.epochs}, LR: {args.lr}\n")
        f.write(f"Decision: {'PROCEED to Phase B' if final_acc > 0.40 else 'RE-EVALUATE hypothesis'}\n")

    print(f"\nResults appended to {results_path}")
    print(f"\nPhase A2 complete. {'PROCEED to Phase B' if final_acc > 0.40 else 'RE-EVALUATE hypothesis'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='metadata/pdb_metadata_with_dates_local.csv')
    parser.add_argument('--max_samples', type=int, default=3000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
