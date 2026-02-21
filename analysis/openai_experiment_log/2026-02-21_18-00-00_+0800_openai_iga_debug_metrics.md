# IGA debug/iga/* logging (geometry + saturation)

Time: 2026-02-21 18:00:00 +0800

## Context

- Baseline: IPA backbone flow model
  - W&B run: `fej59g91`
  - Local dir: `wandb/run-20260127_202731-fej59g91`
- Regression: IGA backbone flow model has frequent spikes and worse loss.
  - Example local run: `wandb/run-20260221_173746-pz2e6c0n`
- Goal: make the failure mode observable (not guessing).

## Change

- Files touched:
  - `models/IGA.py`
  - `models/flow_model.py`
  - `models/flow_module.py`
  - `configs/Train_fm.yaml` (docs + `iga_debug_interval`)
- What changed:
  - `InvariantGaussianAttention` stores per-forward debug scalars in `self._last_debug`.
  - `FlowModelIGA` collects debug snapshots for layer 0 and last layer + global extremes.
  - `FlowModule.model_step()` logs these values to W&B under `debug/iga/*` (rank0 only).

## Rationale / logic

IGA failures looked like softmax saturation / occasional loss explosions.
We need to measure:

- magnitude of scalar logits vs geometry logits
- softmax saturation (how often attention becomes near one-hot)
- whether covariance log-det collapses to extreme values

These directly answer: "Is geometry dominating attention?" and "What triggers spikes?"

## How to run / compare

- W&B notes mention the new debug keys.
- Controls:
  - `experiment.training.iga_debug_interval` (default 10; set 1 for every-step layer snapshots).

Key metrics:

- `debug/iga/global/geo_scaled_absmax`
- `debug/iga/global/wmax_sat_frac`
- `debug/iga/global/log_det_min`
- `train/trans_loss`, `train/rots_vf_loss`, `train/se3_vf_loss`

## Results (observed example)

In run `pz2e6c0n` (IGA, pre-fixes), debug showed:

- geometry logits magnitude extremely large (up to ~1e8)
- `wmax_sat_frac` often near 1.0 (softmax near one-hot)
- `log_det_min` frequently around ~ -39 on spike steps

Spike alignment example (same step):

- `geo_scaled_absmax` very large
- `wmax_sat_frac = 1.0`
- `log_det_min ~ -39`
- `train/rots_vf_loss` spikes (e.g. > 100)

Interpretation:

- IGA geometry term can dominate logits and saturate softmax.
- This likely drives unstable updates and frequent loss clamping.

## Next steps

- Mitigate saturation at the source:
  - clamp log-det term used in geometry logits
  - enforce a physically meaningful sigma lower bound (nm)
  - scale geometry aggregation by `1/sqrt(P)`

Commits:

- `e86c1fd` (add debug logging plumbing)
- `9dd4feb` (document debug metrics + add `iga_debug_interval`)

