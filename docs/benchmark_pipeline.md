# Benchmark pipeline

Experiments use `run.py` as the common training and evaluation entry point:

```powershell
python .\run.py --task_name long_term_forecast --is_training 1 ...
```

Date-free runners are stored under `experiments/`.

## Core commands

High-dimensional core datasets:

```powershell
python .\experiments\core_benchmarks\run_expensive_core_5seed.py --skip-completed
```

Candidate datasets:

```powershell
python .\experiments\core_benchmarks\run_candidate_core_5seed.py --skip-completed
```

Recent baseline context:

```powershell
python .\experiments\recent_baselines\run_recent_baselines_5seed.py --execute --skip-completed
```

Targeted PatchTST-family sensitivity:

```powershell
python .\experiments\targeted_hpo\run_targeted_hpo.py --stage final --skip-completed
python .\experiments\secondary_sensitivity\run_secondary_sensitivity.py --stage final --skip-completed
```

Diagnostic analyses:

```powershell
python .\experiments\diagnostics\run_branch_corruption.py
python .\experiments\diagnostics\run_representation_diagnostics.py
python .\experiments\diagnostics\run_missingness_mechanisms.py
```

## Resuming interrupted campaigns

Use `--skip-completed` to reuse existing scalar metrics when the corresponding
`metrics.npy` file exists under `results/`. Use `--max-runs N` where supported
to run resumable batches.

## Result checks

Before using a row in an aggregate table, verify:

1. the planned model, dataset, horizon, and seed;
2. the command arguments in the manifest/status file;
3. the existence of the corresponding `metrics.npy`;
4. the metric order used by the loader;
5. the `experiment_run_on` field in `results_summary/` when mixing outputs
   from multiple machines.
