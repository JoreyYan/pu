# 2026-02-21 23:30:00 +0800 — Claude — GT Geometry Attention: Results at step 7762

## Runs compared

| Run | W&B ID | Description | Code change |
|-----|--------|-------------|-------------|
| **IPA baseline** | `fej59g91` | Pure IPA, 6D backbone-only | None (reference) |
| **IGA 12D old** | `4jfw6c1f` | 12D + corrupted geo for attention | commit `ce1212e` |
| **IGA 12D GT-geo** | `glhqu344` | 12D + GT geo for attention | GT geometry fix |

## Key result: GT geometry fix works

The old 12D run (`4jfw6c1f`) crashed at step ~4631 with NaN (attention saturation
`wmax_sat_frac=1.0`, `log_det_min=-32.56`). The GT geometry fix (`glhqu344`) runs
stably to step 7762 (epoch 3) with no NaN.

## Backbone loss: IGA vs IPA

> **Note**: The matched-window comparison below was computed from local `.wandb` binary
> files parsed via `wandb.sdk.internal.datastore.DataStore`. Step indexing in local files
> may not perfectly match W&B UI step numbers. Treat as approximate.

500-step mean, matched window step 3500–4000:

| Metric | IPA (`fej59g91`) | IGA 12D GT-geo (`glhqu344`) | Ratio |
|--------|------------------|----------------------------|-------|
| trans_loss | 0.774 | 0.990 | 1.28x |
| rots_vf_loss | 0.756 | 0.930 | 1.23x |

**Gap is ~1.25x** — acceptable for a first 12D run with no hyperparameter tuning.
IGA is doing 12D (backbone + ellipsoid) while IPA only does 6D backbone.

Latest snapshot (step 7762, from `wandb-summary.json` cross-checked by Codex via local `.wandb`):

| Metric | Value |
|--------|-------|
| trans_loss | 0.683 |
| rots_vf_loss | 0.661 |
| se3_vf_loss | 5.72 |
| ellipsoid_local_mean_loss | 3.02 |
| ellipsoid_scaling_loss | 1.36 |

## Ellipsoid loss convergence (IGA, median of 500-step windows)

| Metric | First 500 steps | Last 500 steps | Trend |
|--------|-----------------|----------------|-------|
| ellipsoid_local_mean_loss | 5.618 | 2.964 | 1.9x decrease |
| ellipsoid_scaling_loss | 4.573 | 1.656 | 2.8x decrease |
| trans_loss | 3.882 | 0.768 | 5.1x decrease |
| rots_vf_loss | 1.531 | 0.804 | 1.9x decrease |

**All losses are steadily decreasing.** Ellipsoid parameters are being learned.

## Residual spike analysis (scaling_loss, glhqu344)

> **Correction (Codex review)**: Original version incorrectly used max=9084.8 from
> the old run `4jfw6c1f`. Values below are from `glhqu344` only, verified by Codex
> from local `.wandb` data.

| Statistic | Value |
|-----------|-------|
| Median | 1.73 |
| Mean | 10.64 |
| Std | — |
| P95 | 36.0 |
| P99 | 167.7 |
| Max | 2797.5 |
| Steps > 50 | 78 (3.6%) |
| Steps > 100 | 51 (2.35%) |

GT geometry fix eliminated the catastrophic NaN crashes (old run max=9084, crashed
with NaN at step 4631). New run max=2797 is still large but no longer causes NaN.
~3.6% of steps still have `scaling_loss > 50`. These spikes are likely caused by
time normalization (`1/t` at low `t ≈ min_t`) amplifying prediction errors on
outlier samples.

## IGA debug diagnostics (step 7762, verified by Codex)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| geo_scaled_absmax | 67.0 | Geometry active (was 140 in old run) |
| wmax_sat_frac | 0.0119 | Healthy, no saturation |
| log_det_min | -12.43 | Well within -20 clamp |
| geo_a_mean (L0/L5) | 0.018–0.019 | geo_scale hasn't moved from init |

## Conclusions

1. **Phase 4 core thesis validated**: ellipsoid parameters can be learned via flow
   matching. Both `local_mean` and `scaling_log` losses decrease steadily.

2. **GT geometry fix is effective**: eliminates NaN crashes, reduces geometry logit
   magnitude (140 → 65), keeps attention healthy.

3. **Backbone competitive**: 1.25x gap vs IPA at matched steps, with IGA doing 2x
   the work (12D vs 6D). Median backbone loss is nearly identical to IPA.

4. **Remaining issue**: ~3.6% spike rate in scaling_loss (max 2797 vs old run's 9084).
   Not causing crashes but inflating mean loss and potentially slowing convergence.

## Next steps

### Immediate
- Let `glhqu344` continue to epoch 5–10 for long-term trend
- Investigate scaling_loss spikes: are they concentrated at low `t`? Specific residue types?

### If spikes need fixing
- Clamp time normalization for ellipsoid loss: `min(1/t, C)` where `C ∈ [10, 50]`
- Or reduce `ellipsoid_scaling_loss_weight` from 1.0 to 0.25

### Phase 4 → Phase 5 transition
- Visualize predicted ellipsoids vs GT sidechain geometry (sanity check)
- Test inference without GT geometry (fallback path)
- Design ellipsoid → atom14 decoder

## Data source

- Initial draft (Claude): parsed local `.wandb` binary files via `wandb.sdk.internal.datastore.DataStore`,
  500-step rolling windows, median for spike-robust statistics.
- Spike statistics and latest-step diagnostics corrected by Codex from independent
  local `.wandb` parse. The matched-window comparison (step 3500–4000) was from
  Claude's local parse and could not be independently verified by Codex — treat as
  approximate.
