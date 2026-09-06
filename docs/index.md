# PFB Forecasting

PFB Forecasting studies parallel patch-encoder fusion as a PatchTST-family
design choice for long-horizon time-series forecasting.

![PFB architecture](../figures/architecture.png)

## Model scope

| Model | Role |
|---|---|
| PatchTST | Single patch-encoder reference model. |
| PFB-Direct | Parallel encoders with direct concatenation. |
| PFB-Projected | Parallel encoders with a projection block after concatenation. |

External baselines provide forecasting context. The primary architectural
comparison remains PFB versus PatchTST-family controls.

## Main folders

- `models/`: model definitions and controls.
- `experiments/`: date-free experiment runners.
- `analysis/`: notes for analysis workflow locations.
- `figures/`: current figure assets.
- `results_summary/`: compact result summaries with source-GPU provenance.

Large raw outputs are kept outside git: `data/`, `checkpoints/`, `results/`,
and `test_results/`.
