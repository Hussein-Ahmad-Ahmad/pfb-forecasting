# PFB (PatchFusionBERT)

This folder is a publication-oriented package for the PatchFusionBERT long-term forecasting study. It keeps the runnable code, the 7 bundled CSV datasets used by the study, the models actually used in the paper benchmark, the follow-up control scripts, and the compact result files needed for review and reuse.

## Included Scope

- Task support: `long_term_forecast` only
- Bundled datasets: `ETTm1`, `ETTm2`, `ETTh1`, `ETTh2`, `Exchange`, `Weather`, `Illness`
- Paper models:
  - `PatchFusionBERT_v0`
  - `PatchFusionBERT_v2`
  - `PatchFusionBERT_BERTOnly`
  - `PatchTST`
  - `DLinear`
  - `TiDE`
  - `TimeXer`
  - `iTransformer`
- Recent control studies:
  - capacity-matched `PatchTST` controls
  - `PatchFusionBERT_v0` K-depth ablation
  - multi-seed validation and illness multi-seed stability

## Folder Layout

- `data/`: bundled CSV datasets
- `data_provider/`, `exp/`, `layers/`, `models/`, `utils/`: runnable code
- `scripts/`: experiment launchers for multiseed and control studies
- `analysis/`: optional result parsers and control-analysis utilities for regenerating paper tables
- `results/`: shipped benchmark summaries and result logs

## Validation Modes

This package supports two different validation scopes.

### 1. Core Pipeline Validation

This is the default health check for the publishable package. It verifies that the packaged code can:

- find the bundled datasets
- import the runnable forecasting models
- generate benchmark commands
- run a tiny baseline training+test pass
- run a tiny PatchFusionBERT training+test pass
- append outputs into `results/result_long_term_forecast.txt`

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test_pipeline.ps1
```

Optional CPU-only variant:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test_pipeline.ps1 -UseCpu
```

### 2. Extended Reproducibility Validation

This mode additionally runs the shipped analysis scripts that rebuild paper-side summaries from historical logs. These scripts are useful for reproducibility, but they are not required for the core forecasting pipeline to be considered healthy.

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test_pipeline.ps1 -IncludeAnalysis
```

## Publishable Package Policy

For publication and release, the recommended interpretation is:

- Core code and bundled data are required.
- Tiny end-to-end train/test smoke validation is required.
- `analysis/` should be included in the repository for reproducibility.
- `analysis/` should not be treated as part of the default runtime health gate.

In other words, the package is considered operational if the core pipeline smoke test passes, even if some legacy analysis scripts need older result-log assumptions or additional historical files.

## Installation

### Option 1: Pip Install (Recommended)

```bash
pip install -e .
```

### Option 2: Manual Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This package requires PyTorch 2.0+ and Python 3.8-3.11. The dependencies have been updated from the original research codebase for compatibility with modern environments (March 2026).

## Quick Start

1. Verify installation:

```bash
python -c "import torch; from models import PatchFusionBERT_v0, DLinear; print('✓ Installation successful')"
```

2. Run the main single-seed benchmark grid:

```bash
python scripts/run_main_benchmark.py --phase all
```

3. Run follow-up studies from the package root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\capmatch_controls.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\pfb_v0_kdepth_ablation.ps1
```

4. Run the default package smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\smoke_test_pipeline.ps1
```

5. Rebuild shipped summary tables from the result log:

```bash
python analysis/analyze_multiseed_results.py
python analysis/analyze_illness_multiseed_stability.py
python analysis/analyze_capmatch_controls.py
python analysis/analyze_pfb_kdepth_ablation.py
```

## Shipped Results

- `results/result_long_term_forecast.txt`: appended raw result log
- `results/results_23-01.csv`: compact benchmark table source
- `results/results_23-01.md`: benchmark summary and protocol notes
- `results/multiseed_statistics.csv`: multi-seed summary
- `results/analysis/`: capacity-match, illness-stability, and K-depth summaries

## Version Information

**Current Version:** 1.0.0 (March 2026)
- Updated dependencies for modern Python/PyTorch environments
- PyTorch 2.0+ compatible
- Python 3.8-3.11 supported

## Notes

- Run all scripts from this package root.
- New experiment outputs are written under `results/`, `checkpoints/`, and `test_results/` as they are generated.
- `PatchFusionBERT_BERTOnly` depends on `transformers`.
- Default smoke validation skips `analysis/`; use `-IncludeAnalysis` only when you want an extended reproducibility check.

## Installation Issues

If you encounter compatibility issues:
1. Ensure Python version is 3.8-3.11
2. Update pip: `pip install --upgrade pip`
3. Install in a clean virtual environment
4. Check GPU compatibility: PyTorch 2.0+ requires CUDA 11.7+ for GPU support
