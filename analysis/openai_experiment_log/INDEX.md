# Index

This index lists change records written by OpenAI Codex / Claude Code for this repo.

- `2026-02-21_18-00-00_+0800_openai_iga_debug_metrics.md`
  - Add `debug/iga/*` logging (geometry/logits/saturation) to W&B.
  - Commits: `e86c1fd`, `9dd4feb`

- `2026-02-21_18-13-45_+0800_openai_iga_logdet_clamp.md`
  - Clamp `log_det` used in geometry logits to prevent unbounded positive bias.
  - Commits: `0dc81b8`, `40382d9`

- `2026-02-21_19-01-15_+0800_openai_iga_sigma_min_scaleP.md`
  - Enforce `sigma_min_nm=0.03` + scale geometry by `1/sqrt(P)` to avoid Mahalanobis blow-ups.
  - Commit: `721e94a`

- `2026-02-21_19-37-28_+0800_openai_iga_sigmaMin03nm_scaleP_results.md`
  - Early results for `bunuc9ei` vs IPA (matched step window 0..3087).

- `2026-02-21_19-52-56_+0800_openai_iga_geoScale_init_minus4.md`
  - Initialize `geo_scale` at -4.0 (softplus≈0.018) to soft-start geometry without a schedule.

- `2026-02-21_20-14-10_+0800_openai_iga_geoInitM4_results.md`
  - Early results for `vuzldfte` vs IPA (matched step window 0..1289).

- `2026-02-21_20-52-02_+0800_openai_iga_12d_enable.md`
  - Enable 12D GaussianUpdateBlock + ellipsoid losses + gaussian param corruption.

- `2026-02-21_22-00-00_+0800_claude_iga_gt_geo_attn.md`
  - **[Claude Code]** Decouple IGA attention geometry from corrupted ellipsoid params.
  - Use GT ellipsoid geometry for IGA attention; update block still denoises corrupted rigids.
  - Files: `models/flow_model.py`, `configs/Train_fm.yaml`
