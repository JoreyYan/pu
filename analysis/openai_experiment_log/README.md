# OpenAI (Codex) experiment log

This folder is meant to be readable by *any* other AI agent (or human) who
joins later. Each change we make to the IGA/IPA codepath should be recorded as a
separate markdown file with:

- what we changed (files + key code paths)
- why we changed it (logic + hypothesis)
- how to run / how to compare (W&B run name and/or local wandb run dir)
- what happened (results + interpretation)
- next actions (what to try next)

## Naming convention

Use filenames like:

`YYYY-MM-DD_HH-MM-SS_+TZ_openai_<short-topic>.md`

Example:

`2026-02-21_19-15-19_+0800_openai_iga_logdet_clamp.md`

## Minimal template

See: `analysis/openai_experiment_log/TEMPLATE.md`

