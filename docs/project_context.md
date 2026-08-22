# Project context

This project evaluates parallel patch-encoder fusion as a PatchTST-family design
choice for long-horizon time-series forecasting.

The main architectural comparison is:

- `PatchTST`: single patch encoder.
- `PFB-Direct`: two parallel patch encoders with direct concatenation.
- `PFB-Projected`: two parallel patch encoders with a projection block after
  concatenation.

The broader model set, including DLinear, iTransformer, TiDE, and TimeXer, is used
as forecasting context. The primary architectural focus is the behavior of
parallel patch-encoder fusion relative to PatchTST-family controls.

## Model naming

The public names used in the paper-facing material are `PFB-Direct` and
`PFB-Projected`. Both models use Transformer encoders over time-series patch
tokens.

## Current reproducibility boundary

The repository tracks code, experiment runners, small tabular summaries, and
paper figures. Large generated artifacts are not tracked in git:

- datasets;
- checkpoints;
- raw per-run result folders;
- training logs.

These artifacts should be distributed through a separate archive or data-release
record when needed.
