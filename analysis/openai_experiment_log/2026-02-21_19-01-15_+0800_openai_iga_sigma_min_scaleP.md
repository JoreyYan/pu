# IGA sigma_min_nm + 1/sqrt(P) scaling (reduce Mahalanobis blow-ups)

Time: 2026-02-21 19:01:15 +0800

## Context

After logdet clamp, IGA still showed:

- very large geometry logits (`debug/iga/global/geo_scaled_absmax` often 1e7..1e8)
- very high softmax saturation (`debug/iga/global/wmax_sat_frac` ~ 0.9+)
- frequent `train/trans_loss` clamping to 5

Hypothesis: the main remaining driver is the Mahalanobis term exploding when sigma
collapses (tiny axis lengths => `Sigma^{-1}` huge).

## Change

Files touched:

- `data/GaussianRigid.py`
- `models/IGA.py`
- `configs/Train_fm.yaml`
- `analysis/iga_sigma_min_and_point_scaling.md` (original note)

What changed:

1) Enforce `sigma_min_nm = 0.03` (0.3A) in IGA geometry:
   - applied in local sigma used to compose `mu = anchor_mean + u * sigma_local`
   - applied in covariance construction (`get_covariance_with_delta(..., min_s=sigma_min_nm)`)

2) Scale geometry aggregation by `1/sqrt(P)`:
   - `attn_bias_geo = sum_P(overlap_scores) * (1/sqrt(P))`

3) W&B run name/notes labeled:
   - `_sigmaMin03nm_scaleP`

Commit:

- `721e94a`

## Rationale / logic

- If sigma is allowed to approach ~0 (even if numerically positive), then:
  - `dist_sq ~ ||delta||^2 / sigma^2` becomes enormous
  - geometry logits dominate, softmax saturates, and training becomes unstable.

Enforcing a physical sigma floor bounds `dist_sq` and should:

- reduce `geo_scaled_absmax` by orders of magnitude
- reduce `wmax_sat_frac` (less one-hot attention)
- reduce `trans_loss` clamp frequency

Scaling by `1/sqrt(P)` prevents geometry magnitude from increasing linearly with
the number of Gaussian points per head.

## How to run / compare

Compare against:

- IPA baseline: `fej59g91`
- IGA pre-fixes: `pz2e6c0n`
- IGA logdet clamp only: `ybw5vj1q`

Key metrics:

- losses: `train/trans_loss`, `train/rots_vf_loss`, `train/se3_vf_loss`
- stability: count of `train/trans_loss >= 5`, spikes in `train/rots_vf_loss`
- debug: `debug/iga/global/geo_scaled_absmax`, `debug/iga/global/wmax_sat_frac`, raw `log_det_min`

## Results

TBD (run not yet evaluated at time of writing).

Expected:

- `debug/iga/global/log_det_min` raw should stop sticking at ~ -39 (sigma collapse removed)
- `debug/iga/global/geo_scaled_absmax` drops significantly
- `train/trans_loss >= 5` count drops significantly

## Next steps

If still saturated:

- add a mild clamp/temperature on `G_geo` (after stabilizing sigma)
- initialize geometry weight smaller (e.g., make `geo_scale` start near 0)
- consider `t`-dependent geometry gating to avoid high-noise collapse

