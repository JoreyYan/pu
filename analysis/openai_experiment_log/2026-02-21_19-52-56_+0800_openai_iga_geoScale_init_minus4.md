# IGA geo_scale init = -4.0 (soft start without warm-up schedule)

Time: 2026-02-21 19:52:56 +0800

## Context

- Current IGA stability improved via:
  - logdet clamp (`logdet_min=-20`)
  - `sigma_min_nm=0.03`
  - `1/sqrt(P)` scaling
- Debug still shows geometry larger than scalar early in training.
  - Example: `geo_raw_min` much more negative than scalar logits.

Goal: prevent geometry from dominating at step 0 without adding a warm-up schedule.

## Change

- File: `models/IGA.py`
  - `geo_scale` initialization changed from `1.0` to `-4.0`.
  - This makes `softplus(geo_scale) ~ 0.018` at init.

- File: `configs/Train_fm.yaml`
  - W&B name/notes updated with `_geoInitM4`

## Rationale / logic

With `geo_scale=1.0`, the initial geometry multiplier `a=softplus(1.0)≈1.31`,
so geometry logits can be hundreds of times larger than scalar logits.

Setting init to `-4.0`:

- `softplus(-4)≈0.018`
- geometry starts near the scalar scale
- model can still learn larger values if useful (no upper bound)
- avoids adding a warm-up schedule hyperparameter

## How to run / compare

- W&B name: `..._geoInitM4`
- Compare against:
  - `bunuc9ei` (sigma_min + scaleP, geo_scale init 1.0)
  - IPA baseline `fej59g91`

Key metrics:

- `train/trans_loss`, `train/rots_vf_loss`, `train/se3_vf_loss`
- `trans_loss >= 5` count
- `debug/iga/global/geo_scaled_absmax`
- `debug/iga/global/wmax_sat_frac`

## Expected outcome

- Reduced early saturation (`wmax_sat_frac` lower at early steps)
- Fewer `trans_loss` clamp events
- More stable loss curves; possibly closer to IPA on matched step windows

