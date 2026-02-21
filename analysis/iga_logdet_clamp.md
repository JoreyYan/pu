# IGA logdet clamp experiment

Time: 2026-02-21 18:13:45 +0800

## What changed

We clamp the Gaussian log-determinant term used in IGA's geometry logits:

- Before:
  - `geom = -0.5 * dist_sq - 0.5 * log_det`
- After:
  - `log_det_used = clamp(log_det_raw, min=LOGDET_MIN)`
  - `geom = -0.5 * dist_sq - 0.5 * log_det_used`

Default `LOGDET_MIN = -20.0` (can be tuned).

## Why this change (math + symptoms)

IGA uses a Gaussian log-likelihood style geometry bias. In 3D:

- `log p(delta | Sigma) = -0.5 * delta^T Sigma^{-1} delta - 0.5 * log|Sigma| + const`

Here `log_det = log|Sigma|` acts like a *volume/normalization* term.

When `Sigma` becomes near-singular (one axis variance collapses), `|Sigma| -> 0`,
so `log|Sigma| -> -inf`, making `-0.5 * log|Sigma| -> +inf`.

In attention logits, this becomes a huge positive bias, which can:

- blow up `G_geo` (`debug/iga/global/geo_scaled_absmax` very large),
- saturate softmax (`debug/iga/global/wmax_sat_frac -> 1`),
- produce unstable/hard attention and loss spikes (observed in IGA noGnoise runs).

Clamping `log_det` prevents the pathological "Sigma volume collapse => infinite reward"
feedback loop, while keeping the Mahalanobis term (direction sensitivity) intact.

## What to check in W&B

Compare before/after:

- `debug/iga/global/log_det_min` should no longer stick around ~ -39.
- `debug/iga/global/geo_scaled_absmax` should drop by orders of magnitude.
- `debug/iga/global/wmax_sat_frac` should decrease (less one-hot attention).
- Training spikes in `train/rots_vf_loss` / `train/trans_loss` should reduce.

