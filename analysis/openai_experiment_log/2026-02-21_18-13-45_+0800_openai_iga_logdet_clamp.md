# IGA logdet clamp (prevent unbounded +bias from near-singular Sigma)

Time: 2026-02-21 18:13:45 +0800

## Context

- Baseline IPA:
  - W&B: `fej59g91`
  - Local: `wandb/run-20260127_202731-fej59g91`
- IGA instability: frequent spikes aligned with `debug/iga/global/log_det_min ~ -39` and
  very large `debug/iga/global/geo_scaled_absmax` + `wmax_sat_frac ~ 1`.

Goal: remove the "Sigma volume collapse => infinite reward" mechanism.

## Change

- Files touched:
  - `models/IGA.py`
  - `configs/Train_fm.yaml` (W&B name/notes)
  - `analysis/iga_logdet_clamp.md` (original note)
- What changed:
  - Compute `log_det_raw = log|Sigma|`
  - Use `log_det_used = clamp(log_det_raw, min=-20.0)` in geometry logits:
    - `geom = -0.5 * dist_sq - 0.5 * log_det_used`
  - Debug continues to log *raw* log_det so we can verify clamp engagement.

Commits:

- `0dc81b8` (code + note)
- `40382d9` (label run name/notes)

## Rationale / logic

IGA geometry uses a Gaussian log-likelihood style bias:

- `log p(delta|Sigma) = -0.5 * dist_sq - 0.5 * log|Sigma| + const`

When `|Sigma| -> 0` (near-singular covariance), `log|Sigma| -> -inf`,
so `-0.5*log|Sigma| -> +inf` which becomes an unbounded positive attention bias.

This can saturate softmax and destabilize training.

## How to run / compare

- W&B name contains: `_logdetClamp20`
- Compare against:
  - IPA baseline `fej59g91`
  - IGA pre-clamp `pz2e6c0n`

## Results (early)

Run: `ybw5vj1q` (logdetClamp20)
Local: `wandb/run-20260221_182250-ybw5vj1q`

Compared on the same step window `0..2724` (matching run length at time of analysis):

- IPA (`fej59g91`):
  - `train/se3_vf_loss` mean ~ 2.59
  - `train/trans_loss >= 5` count: 2
- IGA pre-clamp (`pz2e6c0n`):
  - `train/se3_vf_loss` mean ~ 4.77
  - `train/trans_loss >= 5` count: 200
- IGA clamp (`ybw5vj1q`):
  - `train/se3_vf_loss` mean ~ 4.60
  - `train/trans_loss >= 5` count: 161

Interpretation:

- Clamp helps *somewhat* (reduces extreme rewards, improves stability marginally),
  but does not close the gap to IPA.
- Attention is still highly saturated; trans loss still frequently hits clamp=5.

## Next steps

Clamp logdet is not sufficient because `dist_sq` can still explode when sigma is tiny
(`dist_sq ~ 1/sigma^2`).

Next minimal fix:

- enforce `sigma_min_nm` in geometry (stop sigma collapse)
- scale geometry aggregation by `1/sqrt(P)` (avoid magnitude growing with number of points)

