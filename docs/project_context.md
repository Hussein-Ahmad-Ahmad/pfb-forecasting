# Project context

This project evaluates parallel patch-encoder fusion as a PatchTST-family design
choice for long-horizon time-series forecasting.

The main architectural comparison is:

- `PatchTST`: single patch encoder.
- `PFB-Direct`: two parallel patch encoders with direct concatenation.
- `PFB-Projected`: two parallel patch encoders with a projection block after
  concatenation.

The broader model set, including DLinear, iTransformer, TiDE, TimeXer, and
recent patch/fusion baselines, provides context for how the PatchTST-family
comparison sits among other forecasting pipelines.

## Public naming

The public names are `PFB-Direct` and `PFB-Projected`. Both models use
Transformer encoders over time-series patch tokens and are trained directly on
forecasting data.

## Reproducibility boundary

The repository tracks code, experiment runners, compact tabular summaries, and
current figure assets. Large generated artifacts are not tracked in git:

- datasets;
- checkpoints;
- raw per-run result folders;
- training logs.

For machine-to-machine continuation, copy `data/`, `checkpoints/`, `results/`,
and `test_results/` directly.
