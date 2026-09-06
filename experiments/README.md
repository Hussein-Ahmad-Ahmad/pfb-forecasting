# Experiment runners

This folder contains date-free runners for the current PFB forecasting package.
Raw outputs are written to the usual local artifact folders and are not tracked
in git:

- `results/`
- `checkpoints/`
- `test_results/`

Use `--skip-completed` to continue interrupted campaigns when existing
`metrics.npy` files are present.

## Folders

| Folder | Purpose |
|---|---|
| `core_benchmarks/` | Candidate and high-dimensional core benchmark runners. |
| `targeted_hpo/` | Patch/stride/learning-rate sensitivity. |
| `secondary_sensitivity/` | Local sensitivity around selected configurations. |
| `preprocessing_controls/` | Window-normalization and uniform-preprocessing controls. |
| `long_horizon/` | Long-horizon and topology campaign runner. |
| `recent_baselines/` | Contemporary baseline adapters and queue runner. |
| `cross_variate/` | Cross-variate PFB diagnostic runner. |
| `diagnostics/` | Checkpoint-based representation, corruption, and missingness diagnostics. |
