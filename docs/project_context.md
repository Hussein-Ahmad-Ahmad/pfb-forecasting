# Project context

This project evaluates parallel patch-encoder fusion as a PatchTST-family design
choice for long-horizon time-series forecasting.

The main architectural comparison is:

- `PatchTST`: single patch encoder.
- `PatchFusionBERT_v0`: two parallel patch encoders with direct concatenation.
- `PatchFusionBERT_v2`: two parallel patch encoders with a projection block after
  concatenation.

The broader model set, including DLinear, iTransformer, TiDE, and TimeXer, is used
as forecasting context. The primary architectural claim is not that the proposed
model is universally better than all forecasting backbones. The claim is that
parallel patch-encoder fusion can be useful in selected regimes and should be
evaluated against PatchTST-family controls.

## Naming note

The model name `PatchFusionBERT` is retained for continuity with earlier code and
checkpoints. It does not imply pretrained BERT weights, language-model training,
or a text-domain objective. The secondary stream is a Transformer encoder applied
to time-series patch tokens.

## Current reproducibility boundary

The repository tracks code, experiment runners, small tabular summaries, and
paper figures. Large generated artifacts are not tracked in git:

- datasets;
- checkpoints;
- raw per-run result folders;
- training logs.

These artifacts should be distributed through a separate archive or data-release
record when needed.
