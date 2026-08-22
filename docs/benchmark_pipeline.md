# Benchmark pipeline

## Core command format

Experiments use `run.py` from Time-Series-Library:

```powershell
python .\run.py --task_name long_term_forecast --is_training 1 ...
```

For manuscript-oriented high-dimensional benchmark extension runs, use:

```powershell
python .\scripts\paper\run_core_dataset_benchmark.py --dataset electricity --skip-completed
python .\scripts\paper\run_core_dataset_benchmark.py --dataset traffic --skip-completed
```

The runner writes:

- a plan CSV;
- a manifest JSON;
- per-run stdout and stderr logs;
- a status CSV.

These files are written under:

```text
results_analysis/paper_runs/<dataset>/
```

## Default high-dimensional dataset grid

| Dataset | File | Channels | Batch size | Horizons | Seeds |
|---|---:|---:|---:|---:|---:|
| Electricity | `electricity.csv` | 321 | 8 | 96, 192, 336, 720 | 2021-2025 |
| Traffic | `traffic.csv` | 862 | 4 | 96, 192, 336, 720 | 2021-2025 |

Default models:

- `PatchTST`
- `PatchFusionBERT_v0`
- `PatchFusionBERT_v2`

## Resuming interrupted campaigns

Use:

```powershell
python .\scripts\paper\run_core_dataset_benchmark.py --dataset electricity --skip-completed
```

The runner checks for existing `metrics.npy` files under `results/` and skips
completed runs.

For smaller batches:

```powershell
python .\scripts\paper\run_core_dataset_benchmark.py --dataset electricity --skip-completed --max-runs 5
```

For command inspection only:

```powershell
python .\scripts\paper\run_core_dataset_benchmark.py --dataset electricity --dry-run --max-runs 1
```

## Reporting checks

Before reporting a benchmark row, verify:

1. `status` is `completed` for each planned seed.
2. `metrics_path` points to an existing `metrics.npy`.
3. The dataset, horizon, model, seed, and command arguments match the intended
   protocol.
4. MSE and MAE are extracted in the same order used by Time-Series-Library
   metrics files.
5. Partial rows are labelled as partial if not all planned seeds are complete.
