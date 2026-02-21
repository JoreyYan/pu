# IGA sigma_min + 1/sqrt(P) point scaling experiment

Time: 2026-02-21 19:01:15 +0800

## Motivation

Even after clamping the log-determinant term, IGA attention can still saturate because
the Mahalanobis term can explode when per-axis sigma becomes too small:

- `dist_sq = delta^T Sigma^{-1} delta` grows like `1/sigma^2`
- this drives geometry logits to extreme values and makes softmax nearly one-hot
  (`debug/iga/global/wmax_sat_frac ~ 1`), producing unstable updates and frequent
  `trans_loss` clamping to 5.

Two minimal changes target this directly:

1) enforce a physically reasonable lower bound on sigma (in nm)
2) scale the sum over Gaussian points by `1/sqrt(P)` to keep geometry logits magnitude
   comparable when changing `no_qk_gaussians = P`

## Changes

### A) sigma_min (nm)

We clamp the per-axis sigma used by IGA to:

- `sigma_min_nm = 0.03` (0.3 Angstrom)

This is applied consistently in:

- the local sigma used to compose `mu = anchor_mean + u * sigma_local`
- the covariance construction used for `Sigma` in attention

### B) 1/sqrt(P) scaling

Geometry logits were previously aggregated as:

- `attn_bias_geo = overlap_scores.sum(dim=-1)`

Now:

- `attn_bias_geo = overlap_scores.sum(dim=-1) * (1 / sqrt(P))`

so the geometry term does not grow linearly with the number of Gaussian points.

## Expected W&B effects

Primary (loss-level):

- fewer `train/trans_loss` clamp events (`trans_loss >= 5`)
- fewer extreme spikes in `train/rots_vf_loss`
- lower `train/se3_vf_loss` median / p90 vs previous IGA runs

Secondary (debug/iga):

- `debug/iga/global/geo_scaled_absmax` should drop by orders of magnitude
- `debug/iga/global/wmax_sat_frac` should decrease (less softmax saturation)
- raw `debug/iga/global/log_det_min` should stop sticking around ~ -39 (near-singular volumes)

