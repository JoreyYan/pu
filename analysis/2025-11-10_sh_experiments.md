# 2025-11-10 — SH vs R³ Diffusion Notes

## Context
- Goal: understand why SH-based FBB inference is underperforming vs prior R³ diffusion + MPNN pipelines, and verify that multi-step sampling (1/10/100/200 iterations) actually changes predictions.
- Datasets: CASP15 validation set (45 domains) with GT files under `/home/junyu/project/casp15/targets/casp15.targets.TS-domains.public_12.20.2022`.
- Tooling: Lightning inference via `experiments/inference.py`, evaluation via `esm/genie/evaluations/pipeline/evaluate_fbb.py` → produces `fbb_results/fbb_scores.csv` inside `outputs/eval_*`.

## Experiment Log
### 1. Single-step SH inference sanity check
- **Cmd**: `/home/junyu/mambaforge/bin/python experiments/inference.py` with the latest SH-FBB checkpoint, `num_timesteps=1`.
- **Outputs**: `outputs/shfbb_atoms_sh_atoms_real/val_seperated_Rm0_t0_step0_20251110_020456`.
- **Diag logs** show reasonable metrics (Sidechain RMSD ≈2.0 Å, perplexity ≈2.5, recovery 0.55–0.73), so the inference loop runs and records per-batch stats.
- **Issue**: saving diagnostics with Chinese text triggered `UnicodeEncodeError` because `_save_structures` writes ASCII by default. Needs either UTF-8 encoding or ASCII-only text.

### 2. ESMFold + FBB evaluation pipeline
- **Cmd template**:
  ```bash
  python /home/junyu/project/esm/genie/evaluations/pipeline/evaluate_fbb.py \
      --fbb_output_dir <SH output dir> \
      --output_dir <eval dir> \
      --native_dir /home/junyu/project/casp15/targets/casp15.targets.TS-domains.public_12.20.2022
  ```
- Initial failures were just wrong paths (missing escaping of trailing space or missing nested timestamp dir). Once correct, the pipeline ran end-to-end (sequence extraction → ESMFold → TM-score/pLDDT aggregation).
- One run (`eval_sh_atoms_real_100step`) crashed in `_process_results` because no rows were produced (bug: local `df` never set). Re-running with valid `fbb_results` avoided the crash.

### 3. Multi-step sampling sweep (1/10/100 steps)
- Inference outputs:
  - `outputs/eval_sh_atoms_real_1step`
  - `outputs/eval_sh_atoms_real_10step`
  - `outputs/eval_sh_atoms_real_100step`
  - `outputs/eval_sh_atoms_real_100stepR`
  - Latest rerun: `outputs/eval_sh_atoms_real100stepR_1110` (from `/home/junyu/project/pu/outputs/shfbb_atoms_sh_atoms_real100stepR_1110/val_seperated_Rm0_t0_step0_20251110_094725`)
- **Observation**: 10-step and 100-step CSVs are identical down to every sample, implying `fbb_sample_iterative` ignores `num_timesteps`. Even a 200-step trial produced the same coordinates. Root cause: the model currently outputs `side_atoms` once and that tensor is re-used as both speed vector and final coordinate, so iterative updates never occur.
- **Metrics (mean ± std over 45 CASP15 targets):**

| Run | TM | RMSD (Å) | pLDDT | Recovery | Perplexity |
| --- | --- | --- | --- | --- | --- |
| SH atoms 1-step (`eval_sh_atoms_real_1step`) | 0.587 ± 0.272 | 14.29 ± 14.59 | 65.34 ± 17.85 | 0.651 ± 0.058 | 2.653 ± 0.470 |
| SH atoms 10-step (`eval_sh_atoms_real_10step`) | 0.648 ± 0.261 | 11.26 ± 11.51 | 72.21 ± 17.20 | 0.835 ± 0.040 | 1.541 ± 0.166 |
| SH atoms 100-step (`eval_sh_atoms_real_100step` & `_100stepR`) | identical to 10-step (bug) |
| SH atoms 100-step rerun (`eval_sh_atoms_real100stepR_1110`) | 0.660 ± 0.264 | 10.92 ± 11.63 | 73.47 ± 18.23 | 0.907 ± 0.028 | 1.279 ± 0.113 |

### 4. Comparison with R³ diffusion baseline
- Reference file: `outputs/evaluation_129_complete/tm_rmsd_results.json`.
- **Key stats (41 CASP15 domains):**
  - `ODE_129_1step`: TM 0.567 ± 0.356, CA RMSD 12.26 ± 15.96 Å.
  - `ODE_129_10step`: TM 0.544 ± 0.343, CA RMSD 13.09 ± 16.65 Å.
  - `ODE_129_100step`: TM 0.545 ± 0.348, CA RMSD 12.64 ± 15.67 Å.
- Baseline recovery numbers are lower (not logged in JSON) but visual inspection in PyMOL shows better sidechain geometry than current SH outputs.

### 5. Miscellaneous observations
- Recovery ≥0.9 from evaluation is inconsistent with PyMOL inspection (e.g., T1104-D1). Likely due to string matching without proper structural alignment; pipeline needs a stricter recovery definition (aligned residue IDs, mask for missing residues).
- TMs vs RMSD show large variance: despite RMSD ~11 Å, TM can stay ~0.65 because a few domains align well while others fail catastrophically.
- SH loss training still unstable: enabling SH reconstruction loss causes NaNs around step 2–3 even with anomaly detection and gradient clipping. Need better safeguards inside `sh_density_from_atom14_with_masks_clean`.

## Conclusions (11-10)
1. **Sampling Bug** – Multi-step SH sampling currently degenerates to a single-step output, explaining why 10/100/200 steps produce identical PDBs. Fix requires separating predicted velocity from coordinates inside `interpolant.fbb_sample_iterative` and ensuring each step updates `side_atoms`.
2. **Evaluation confirms stagnation** – Despite higher recovery/pLDDT when more steps are *intended*, the improvements stem from identical structures being re-evaluated, not better generation.
3. **Unicode/logging** – Diagnostics writer must avoid non-ASCII characters or open files with UTF-8 to prevent Lightning crashes mid-prediction.
4. **Training instabilities** – SH density computation still yields NaNs even after adding `sh_density_from_atom14_with_masks_clean`. Need gradient monitoring or smaller loss weights.

## Next Steps
1. **Instrument sampler**: log `num_timesteps`, dump intermediate `side_atoms` norms, and confirm updates happen per step.
2. **Refactor interpolant**: have model return explicit velocity (`speed_vectors`) separate from decoded coordinates, then integrate over timesteps.
3. **Unicode-safe logging**: update `_save_structures` to encode in UTF-8 or strip non-ASCII text.
4. **SH loss debugging**: track gradients through `sh_density_from_atom14_with_masks_clean`, possibly clamp/normalize outputs and monitor largest partials.
5. **Evaluation tooling**: adapt CASP15 recovery calculation to aligned residues to reconcile with PyMOL inspections, and script a combined report comparing SH vs R³ vs MPNN baselines.

