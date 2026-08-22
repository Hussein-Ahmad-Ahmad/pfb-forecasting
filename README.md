# Parallel Patch-Encoder Fusion for Long-Horizon Forecasting

This repository contains the code and experiment utilities for evaluating
PatchFusionBERT, a PatchTST-family forecasting architecture that processes shared
patch tokens through two parallel Transformer encoder streams before prediction.

The project is based on
[Time-Series-Library](https://github.com/thuml/Time-Series-Library) and keeps the
standard long-horizon forecasting workflow used by that codebase.

## Scope

The central question is architectural:

> Does adding a parallel patch-encoder fusion path to a PatchTST-style backbone
> provide useful accuracy and capacity tradeoffs, and under which forecasting
> regimes?

The broader baselines provide external context. The main controlled comparison is
against PatchTST and PatchTST-family capacity controls. The work does not claim a
universally superior forecasting backbone.

Despite the historical model name, PatchFusionBERT does not use language-model
pretraining, masked-language modeling, or pretrained BERT weights. In this code,
the retained attribute name `bert_encoder` is kept only for checkpoint
compatibility; it refers to a secondary Transformer encoder over time-series
patch tokens.

## Models

The main models are:

- `PatchTST`: PatchTST baseline.
- `PatchFusionBERT_v0`: direct parallel fusion. The two encoder streams are
  concatenated before the prediction head.
- `PatchFusionBERT_v2`: projected parallel fusion. The concatenated streams pass
  through a Linear-GELU-Dropout-LayerNorm projection before prediction.
- `PatchTST_LargeHead`: PatchTST capacity-control variant.
- `PatchTST_SerialMatched`: single-path serial control for isolating parallel
  versus serial processing.
- `DLinear_Norm`: DLinear with the same reversible window standardization used by
  the patch models.

## Repository layout

```text
.
├── run.py                         # main Time-Series-Library entry point
├── models/                        # forecasting models and controls
├── exp/                           # experiment dispatch
├── data_provider/                 # dataset loaders
├── layers/                        # shared layers
├── utils/                         # metrics, training utilities, masking
├── scripts/                       # reproducible experiment runners
│   └── paper/                     # manuscript-oriented benchmark runners
├── analysis/                      # result parsing and statistical summaries
├── profiling/                     # parameter, FLOP, and runtime utilities
├── robustness/                    # missingness and perturbation analyses
└── results_analysis/              # small tabular summaries used for reporting
```

Large generated outputs are intentionally not tracked:

- `data/`
- `results/`
- `checkpoints/`
- `test_results/`

## Environment

Use Python 3.10 with PyTorch and the scientific Python stack:

```powershell
pip install -r requirements.txt
```

The local experiments used CUDA-enabled PyTorch. If several Python environments
exist on the same machine, pass the intended interpreter explicitly with
`--python`.

## Dataset placement

Place benchmark CSV files under:

```text
Time-Series-Library/data/
```

For the Electricity and Traffic benchmark extension, the expected files are:

```text
data/electricity.csv
data/traffic.csv
```

## Running the core benchmark extension

Electricity:

```powershell
python .\scripts\paper\run_core_dataset_benchmark.py --dataset electricity --skip-completed
```

Traffic:

```powershell
python .\scripts\paper\run_core_dataset_benchmark.py --dataset traffic --skip-completed
```

These commands run:

- models: `PatchTST`, `PatchFusionBERT_v0`, `PatchFusionBERT_v2`
- horizons: `96, 192, 336, 720`
- seeds: `2021, 2022, 2023, 2024, 2025`

Use `--max-runs N` to execute the campaign in batches. Use `--dry-run` to inspect
the generated commands without training.

## Result handling

Training writes raw outputs to `results/` and checkpoints to `checkpoints/`.
The benchmark runner also writes a plan, manifest, logs, and status CSV under
`results_analysis/paper_runs/<dataset>/`.

Before reporting a result, use the status CSV and the corresponding
`metrics.npy` files to verify:

- dataset name and file path;
- horizon;
- model name;
- seed;
- MSE and MAE extraction order;
- completed status for all planned seeds.

## Citation

If you use this project, cite the manuscript associated with this repository and
the original Time-Series-Library project.
