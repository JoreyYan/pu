"""Phase A1: Linear baseline — can 6D ellipsoid params predict amino acid type?

Usage:
    python analysis/validate_ellipsoid_aatype.py

Loads GT data from the dataset, extracts per-residue (scaling_log_1[3], local_mean_1[3]) = 6D,
and trains a LogisticRegression(multi_class='multinomial') to classify aatype (20 classes).

Outputs:
    - Top-1 accuracy
    - Per-class precision / recall
    - Confusion matrix plot  →  analysis/ellipsoid_aatype_confusion.png
"""

import sys, os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import utils as du
from data.GaussianRigid import OffsetGaussianRigid
from openfold.utils.rigid_utils import Rigid, Rotation
from openfold.np import residue_constants


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset_features(
    csv_path: str,
    processed_dir: str | None = None,
    max_samples: int = 5000,
):
    """Load scaling_log_1, local_mean_1, aatype from processed PDB files.

    Returns:
        X: np.ndarray [M, 6]   (scaling_log[3] + local_mean[3])
        y: np.ndarray [M]      (aatype, 0-19)
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if 'modeled_seq_len' in df.columns:
        df = df[(df['modeled_seq_len'] >= 60) & (df['modeled_seq_len'] <= 400)]
    if 'oligomeric_detail' in df.columns:
        df = df[df['oligomeric_detail'] == 'monomeric']

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    all_X = []
    all_y = []
    skipped = 0

    for idx, row in df.iterrows():
        processed_path = row.get('processed_path')
        if processed_path is None or not os.path.exists(processed_path):
            skipped += 1
            continue

        try:
            processed_feats = du.read_pkl(processed_path)
            processed_feats = du.parse_chain_feats(processed_feats)

            modeled_idx = processed_feats['modeled_idx']
            min_idx = np.min(modeled_idx)
            max_idx = np.max(modeled_idx)
            del processed_feats['modeled_idx']
            import tree
            processed_feats = tree.map_structure(
                lambda x: x[min_idx:(max_idx + 1)], processed_feats)

            aatype = torch.tensor(processed_feats['aatype']).long()
            all_atom_positions = torch.tensor(processed_feats['atom_positions']).float()
            all_atom_mask = torch.tensor(processed_feats['atom_mask']).float()
            bb_mask = torch.tensor(processed_feats['bb_mask']).int()

            # Build atom14
            from openfold.data import data_transforms
            chain_feats = {
                'aatype': aatype,
                'all_atom_positions': all_atom_positions,
                'all_atom_mask': all_atom_mask,
                'atom14_element_idx': torch.tensor(processed_feats['atom14_element_idx']).int(),
            }
            chain_feats = data_transforms.atom37_to_frames(chain_feats)
            chain_feats = data_transforms.make_atom14_masks(chain_feats)
            chain_feats = data_transforms.make_atom14_positions(chain_feats)

            # Build rigid and extract ellipsoid
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
            scaling_log_1 = offset_rigid._scaling_log  # [N, 3]
            local_mean_1 = offset_rigid._local_mean    # [N, 3]

            # Filter valid residues
            mask = bb_mask.bool()
            sl = scaling_log_1[mask].numpy()
            lm = local_mean_1[mask].numpy()
            aa = aatype[mask].numpy()

            # Remove unknown types (>=20)
            valid = aa < 20
            sl = sl[valid]
            lm = lm[valid]
            aa = aa[valid]

            if len(aa) == 0:
                skipped += 1
                continue

            X = np.concatenate([sl, lm], axis=-1)  # [M, 6]
            all_X.append(X)
            all_y.append(aa)

        except Exception as e:
            skipped += 1
            continue

    print(f"Loaded {len(all_X)} proteins, skipped {skipped}")
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"Total residues: {len(y)}")
    return X, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Paths — adjust as needed
    csv_path = 'metadata/pdb_metadata_with_dates_local.csv'
    output_dir = Path('analysis')
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Phase A1: Linear baseline — 6D ellipsoid → aatype (20 classes)")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    X, y = load_dataset_features(csv_path, max_samples=3000)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Random baseline
    random_acc = 1.0 / 20
    print(f"\nRandom baseline: {random_acc:.1%}")

    # Frequency baseline
    from collections import Counter
    counts = Counter(y_train)
    most_common_class = counts.most_common(1)[0][0]
    freq_acc = np.mean(y_test == most_common_class)
    print(f"Frequency baseline (always predict '{residue_constants.restypes[most_common_class]}'): {freq_acc:.1%}")

    # Logistic Regression
    print("\nTraining LogisticRegression (multinomial, max_iter=2000)...")
    clf = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=2000,
        C=1.0,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print(f"Top-1 accuracy: {accuracy:.1%}")
    print(f"{'=' * 60}")

    if accuracy > 0.50:
        print(">> LINEAR SEPARABILITY IS STRONG (>50%)")
    elif accuracy > 0.30:
        print(">> Hypothesis partially confirmed (>30%), MLP worthwhile")
    elif accuracy > 0.15:
        print(">> Better than random (>15%), but weak signal")
    else:
        print(">> Near random — hypothesis may not hold for linear model")

    # Per-class report
    target_names = [residue_constants.restypes[i] for i in range(20)]
    print("\nPer-class classification report:")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(20)))
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_xticks(range(20))
    ax.set_yticks(range(20))
    ax.set_xticklabels(target_names, rotation=45, ha='right')
    ax.set_yticklabels(target_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'6D Ellipsoid → aatype (Linear) — Acc: {accuracy:.1%}')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = output_dir / 'ellipsoid_aatype_confusion.png'
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to {cm_path}")

    # Also try scaling_log only (3D) and local_mean only (3D) as ablation
    print("\n--- Ablation: scaling_log only (3D) ---")
    clf_sl = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000, random_state=42)
    clf_sl.fit(X_train[:, :3], y_train)
    acc_sl = accuracy_score(y_test, clf_sl.predict(X_test[:, :3]))
    print(f"scaling_log only accuracy: {acc_sl:.1%}")

    print("\n--- Ablation: local_mean only (3D) ---")
    clf_lm = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000, random_state=42)
    clf_lm.fit(X_train[:, 3:], y_train)
    acc_lm = accuracy_score(y_test, clf_lm.predict(X_test[:, 3:]))
    print(f"local_mean only accuracy: {acc_lm:.1%}")

    # Summary
    summary = f"""# Ellipsoid → aatype Linear Baseline Results

| Feature set       | Accuracy |
|-------------------|----------|
| Random baseline   | {random_acc:.1%}   |
| Frequency baseline| {freq_acc:.1%}   |
| scaling_log (3D)  | {acc_sl:.1%}   |
| local_mean (3D)   | {acc_lm:.1%}   |
| Both (6D)         | {accuracy:.1%}   |

Total residues: {len(y)} (train: {len(y_train)}, test: {len(y_test)})
Proteins loaded: ~{len(y) // 200} (estimated)
"""
    summary_path = output_dir / 'ellipsoid_aatype_results.md'
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\nSummary written to {summary_path}")


if __name__ == '__main__':
    main()
