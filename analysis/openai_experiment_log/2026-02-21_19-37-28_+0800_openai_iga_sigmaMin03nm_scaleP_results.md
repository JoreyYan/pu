# IGA sigma_min+scaleP (bunuc9ei) early results vs IPA

Time: 2026-02-21 19:37:28 +0800

## Context

- Baseline IPA run: `fej59g91`
- IGA logdetClamp only: `ybw5vj1q`
- New run (sigma_min_nm=0.03 + 1/sqrt(P) scaling): `bunuc9ei`

Comparison window: step `0..3087` (matched to `bunuc9ei` length at time of analysis).

## Loss comparison (same step window)

IPA (`fej59g91`):

- `train/trans_loss` mean 1.37, p50 1.07, last 1.16
- `train/rots_vf_loss` mean 1.10, p50 1.05, last 0.98
- `train/se3_vf_loss` mean 2.47, p50 2.16, last 2.14
- `trans_loss >= 5` count: 2

IGA logdetClamp only (`ybw5vj1q`):

- `train/trans_loss` mean 3.13, p50 3.07
- `train/rots_vf_loss` mean 1.47, p50 1.47
- `train/se3_vf_loss` mean 4.60, p50 4.59
- `trans_loss >= 5` count: 161

IGA sigma_min + scaleP (`bunuc9ei`):

- `train/trans_loss` mean 2.20, p50 1.72, last 0.66
- `train/rots_vf_loss` mean 1.31, p50 1.35, last 0.89
- `train/se3_vf_loss` mean 3.51, p50 3.10, last 1.55
- `trans_loss >= 5` count: 115

## Debug signal changes (bunuc9ei)

Compared to pre-fix runs, the geometry saturation is much lower:

- `debug/iga/global/geo_scaled_absmax` drops from ~1e7–1e8 to ~1e4 range
- `debug/iga/global/wmax_sat_frac` median ~0.62 (was ~0.9+)
- `debug/iga/global/log_det_min` improves to ~ -13 (was ~ -39)

## Interpretation

- The sigma_min + scaleP change materially stabilizes geometry and reduces saturation.
- Losses improve substantially vs logdetClamp-only IGA, but are still worse than IPA
  on the same step window.
- The tail of `bunuc9ei` looks promising (lower losses), so it may need longer
  training to see if it can catch up or surpass IPA.

## Next steps

- Let `bunuc9ei` run longer; re-compare on a later matched step window.
- If still below IPA, introduce a geometry weight warm-up schedule (or smaller
  initial `geo_scale`) to reduce early-stage dominance.

