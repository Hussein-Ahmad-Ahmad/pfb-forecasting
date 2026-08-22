# PFB Forecasting

Parallel patch-encoder fusion for long-horizon time-series forecasting.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Built on Time-Series-Library](https://img.shields.io/badge/built%20on-Time--Series--Library-orange.svg)](https://github.com/thuml/Time-Series-Library)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

PFB Forecasting provides the code, model definitions, experiment runners, and
figure assets for studying parallel patch-encoder fusion in long-horizon
forecasting. The architecture extends a PatchTST-style workflow by sending the
same patch-token sequence through two Transformer encoder streams and combining
their representations before prediction.

The repository is designed for controlled architectural comparison: PatchTST is
the primary reference model, while PFB-Direct and PFB-Projected test two parallel
fusion strategies under the same data and training workflow.

## Architecture

![PFB architecture](paper/figures/architecture.png)

The design has three main stages:

1. Convert each input window into patch tokens.
2. Process the shared patch-token sequence through parallel encoder streams.
3. Fuse the stream outputs and map them to the forecasting horizon.

## Models

| Model | Description |
|---|---|
| `PatchTST` | Single patch-encoder baseline. |
| `PFB-Direct` | Parallel encoder streams with direct concatenation before the prediction head. |
| `PFB-Projected` | Parallel encoder streams with a projection block after concatenation. |
| `PatchTST_LargeHead` | PatchTST capacity-control variant. |
| `PatchTST_SerialMatched` | Single-path serial control for parallel-versus-serial comparison. |
| `DLinear_Norm` | DLinear with reversible window standardization. |

PFB is trained directly on time-series data. The model uses Transformer encoders
over time-series patch tokens.

## Repository layout

```text
.
|-- run.py                         # main training and evaluation entry point
|-- models/                        # forecasting models and architectural controls
|-- exp/                           # experiment dispatch
|-- data_provider/                 # dataset loaders
|-- layers/                        # shared neural-network layers
|-- utils/                         # metrics and training utilities
|-- scripts/
|   `-- paper/                     # benchmark runners
|-- analysis/                      # result parsing and statistical summaries
|-- profiling/                     # parameter, FLOP, and runtime utilities
|-- robustness/                    # robustness utilities
|-- paper/figures/                 # figure assets
`-- results_analysis/              # compact tabular summaries
```

Generated datasets, checkpoints, raw result folders, and test outputs are kept
outside git:

- `data/`
- `results/`
- `checkpoints/`
- `test_results/`

## Installation

Use Python 3.10 and install the required packages:

```powershell
pip install -r requirements.txt
```

If multiple Python environments are available, pass the intended interpreter to
the experiment runner with `--python`.

## Data

Place benchmark CSV files under:

```text
Time-Series-Library/data/
```

For the high-dimensional benchmark extension:

```text
data/electricity.csv
data/traffic.csv
```

## Running benchmark extensions

Electricity:

```powershell
python .\scripts\paper\run_core_dataset_benchmark.py --dataset electricity --skip-completed
```

Traffic:

```powershell
python .\scripts\paper\run_core_dataset_benchmark.py --dataset traffic --skip-completed
```

Default grid:

| Setting | Values |
|---|---|
| Models | `PatchTST`, `PFB-Direct`, `PFB-Projected` |
| Horizons | `96`, `192`, `336`, `720` |
| Seeds | `2021`, `2022`, `2023`, `2024`, `2025` |

Useful options:

```powershell
--max-runs 5     # run a smaller batch
--dry-run        # print commands without training
--python PATH    # select a specific Python environment
```

## Outputs

Training writes raw outputs to:

```text
results/
checkpoints/
```

The benchmark runner writes compact run metadata to:

```text
results_analysis/paper_runs/<dataset>/
```

Each dataset run directory contains:

- `core_5seed_plan.csv`
- `core_5seed_manifest.json`
- `core_5seed_status.csv`
- per-run logs

Before using a result table, check the status CSV and the matching `metrics.npy`
files for the dataset, horizon, model, seed, and metric order.

## Citation

If you use this repository, cite it as:

```bibtex
@software{pfb_forecasting_2026,
  title  = {PFB Forecasting: Parallel Patch-Encoder Fusion for Time-Series Forecasting},
  author = {Ahmad, Hussein and Mortazavi, Seyyed Kasra and Benarbia, Taha and Al Machot, Fadi and Kyamakya, Kyandoghere},
  year   = {2026},
  url    = {https://github.com/Hussein-Ahmad-Ahmad/pfb-forecasting}
}
```

Please also cite the associated manuscript and the original
[Time-Series-Library](https://github.com/thuml/Time-Series-Library) project.
