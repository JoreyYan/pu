# IGA geoInitM4 (vuzldfte) vs IPA baseline

Time: 2026-02-21 20:14:10 +0800

## Context

- Baseline IPA run: `fej59g91`
- IGA run with sigma_min + scaleP + geo_scale init -4.0: `vuzldfte`

Comparison window: step `0..1289` (matched to `vuzldfte` length at time of analysis).

## Loss comparison (same step window)

IPA (`fej59g91`):

- `train/trans_loss` mean 2.01, p50 1.71, last 0.75
- `train/rots_vf_loss` mean 1.41, p50 1.39, last 0.95
- `train/se3_vf_loss` mean 3.42, p50 3.14, last 1.70
- `trans_loss >= 5` count: 2
- `rot>10` count: 1

IGA (`vuzldfte`, geoInitM4):

- `train/trans_loss` mean 1.73, p50 1.30, last 0.79
- `train/rots_vf_loss` mean 1.21, p50 1.23, last 0.91
- `train/se3_vf_loss` mean 2.94, p50 2.55, last 1.70
- `trans_loss >= 5` count: 21
- `rot>10` count: 0

## Interpretation

Within the same early-step window, IGA with `geo_scale` init -4.0 outperforms IPA
on mean/p50 losses for trans, rot, and se3. This is the first run in which IGA
beats IPA on the matched window.

Note: `trans_loss >= 5` count is still higher than IPA, so stability improvements
are real but not fully solved.

## Next steps

- Let `vuzldfte` run longer and re-compare at a later matched step window.
- If performance stays above IPA, proceed to re-enable 12D Gaussian updates and
  ellipsoid loss to move toward sidechain/ellipsoid tasks.

