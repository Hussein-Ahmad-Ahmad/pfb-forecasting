# Prospective controlled add-on studies

These runners are isolated from historical experiments. They do not alter model
files, previous results, manuscripts, figures, or the completed full-P1 queue.
No training starts without `--execute`. Default plans are:

| Study | Frozen default scope | Training runs |
|---|---|---:|
| Projected attribution | ETTh2, ETTm2, Weather; H=192; 11 configurations; seeds 2021–2025 | 165 |
| HPO selection | Six standard datasets; seven families; 12 distinct active candidates; seed 2020 | 504 |
| Locked HPO testing | One validation winner per dataset/family; seeds 2021–2025 | 210 |
| Paired normalization | ETTh2, ETTm2, Exchange, Weather; four models; ON/OFF; five seeds | 160 |

This is a prospective supplement, not an externally registered preregistration.
The three-dataset attribution study cannot establish a universal mechanism.

## Projected design

All dual-encoder controls use the original Projected **3+4 branch depths**, shared
patch embedding, L=336, P=16/S=8, d=128, and the same training protocol.

- `concat34`: unequal-depth concatenation with a 2d-wide head; NOT historical
  equal-depth PFB-Direct.
- `linear_d` / `gelu_d`: pure Linear(2d,d), without/with GELU, no fusion
  LayerNorm or dropout. Same forecasting head and trainable parameter count.
- `linear_d_reg` / `projected34`: no GELU/GELU with the original fusion
  Dropout+LayerNorm, same head and parameters. The second exactly reproduces
  original Projected in deterministic forward tests.
- `linear_d_lift` / `gelu_d_lift`: width-d bottleneck lifted to 2d using fixed
  `[z,z]/sqrt(2)`, allowing the SAME 2d forecasting-head interface as
  `linear_2d` / `gelu_2d`. Bottleneck and full-width conditions have different
  capacities; do not describe their difference as pure compression causality.
- `PatchTST_base`: contemporaneous base reference.
- `PatchTST_budget`: real width/depth/feed-forward search, no unused parameter
  padding, within a declared 1% tolerance for BOTH parameters and MACs of final
  Projected. Other ablations are profiled but are not all budget matched.

The verified match is width128, depth7, feed-forward528: 2,451,376 parameters vs
Projected's 2,456,000 (-0.1883%); dense/attention MAC difference -0.2711%.
MACs include executed Linear/Conv1d and both attention contractions, per complete
multivariate sample. They exclude normalization, nonlinearities, bias additions,
pooling, positional additions, and other elementwise operations. These are
operation counts, not measured hardware latency. The native PatchTST final
BatchNorm is retained; this comparator controls budgets, not every architecture
detail. `parameter_mac_profiles.csv` records every ablation's cost.

Twelve contrasts and their primary/secondary roles are frozen in `design.py` and
the plan. Identical seeds give identical starting shared encoders and, within
compatible head conditions, identical head weights. Different architectures can
consume dropout randomness differently; paired seeds do not imply identical
optimization trajectories.

## HPO and historical reuse

Twelve unique **effective** configurations are planned for each family. Ignored
arguments are removed before deduplication. DLinear varies learning rate and
moving-average length; TiDE varies active MLP capacity/depth; patch families
include P=64 and additional capacities. Existing candidates are retained where
applicable. This is an equal-distinct-candidate-count comparison, NOT equal
GPU-hours, equal search-space coverage, or proof of optimal tuning.

All selection uses validation only, seed2020, max100 epochs, patience10.
Test data are not constructed during selection or during training of final
models. Final tests follow a frozen validation winner and prespecified test
seeds. Locking refuses any missing candidate or previously touched test split.
Allocated RTX4090 wall hours, actual configurations, validation scores, best
epochs, software versions and final test results are saved.

Historical scalar metrics are NOT automatically imported: the new protocol
uses element-weighted validation MSE, a separate seeded data-shuffle generator,
and complete source/data fingerprints. The previous engine averaged batch MSEs.
`legacy_reuse_audit.csv` identifies candidate overlaps and explains why automatic
reuse is unsafe. This avoids presenting nonidentical runs as paired controls.
Exact verified completions of the new frozen plan are always reused on resume;
interrupted attempts remain recoverable. No historical experiment is deleted.

## Normalization intervention

Exactly ONE reversible per-window/per-channel standardization is ON or OFF,
including inversion of the forecast. Learned LayerNorm/BatchNorm stays active.
Original native forecast cores are replayed without a second hidden native
normalization. DLinear, PatchTST, Direct and Projected use identical architecture,
configuration, initialization, data splits, batch order and seed within ON/OFF
pairs. The training-split StandardScaler remains ON in both conditions; output
metrics remain in that training-standardized scale, not physical units.

## Run commands (PowerShell, from any current directory)

```powershell
$py = 'C:\Users\lab\AppData\Local\Programs\Python\Python312\python.exe'
$runner = 'D:\Hussein-Files\PFB\pfb-forecasting\scripts\paper\run_controlled_addons.py'

# Inspect the default plan and profiles; NO training.
& $py -u -X utf8 $runner projected --plan-only

# First real experiment. Repeating it safely resumes verified results.
& $py -u -X utf8 $runner projected --execute --skip-completed --python $py

# Later: validation selection, freeze winners, then independent testing.
& $py -u -X utf8 $runner hpo-select --execute --skip-completed --python $py
& $py -u -X utf8 $runner hpo-lock
& $py -u -X utf8 $runner hpo-test --execute --skip-completed --python $py

# Paired normalization control.
& $py -u -X utf8 $runner preprocessing --execute --skip-completed --python $py

# Numerical summaries only, after the corresponding complete grid.
& $py -u -X utf8 $runner analyze --study projected
& $py -u -X utf8 $runner analyze --study hpo
& $py -u -X utf8 $runner analyze --study preprocessing
```

Use `--max-runs 1` for an initial one-run execution, then repeat without that
limit. `--plan-only` also works for `hpo-select` and `preprocessing`.
An HPO test plan cannot be built before selection is completed and locked.
`--datasets` / `--seeds` can declare a smaller scope prospectively; use a new
`--out-dir` when changing any frozen scope. Default output is
`results_analysis/controlled_addons_v1/{projected,hpo,preprocessing}`.

## Safeguards and analysis

- RTX4090 required in both launcher and worker; no CPU/other-GPU fallback.
- Shares the original full-P1 queue's OS lock, preventing duplicate queues.
- Every actual result has `EXP_RUN_BY_RTX4090_GPU` provenance. CPU-only profiles
  and unit fixtures are not experimental results and are never relabelled.
- Source, Python/package/CUDA versions, input CSV, configuration, checkpoint, metrics, provenance and history
  checksums prevent accidental mixing. Changed source/data require a new plan.
- Validation and test reductions weight individual elements, not batches.
  MAPE/MSPE clamp absolute denominators at1e-8; MSE/MAE are primary endpoints.
- Streaming evaluation does not create huge prediction-array or PDF collections.
- Partial grids export coverage only, not inferential results. Complete grids
  export paired per-seed effects, dataset-mean t intervals (datasets are the
  independent units), within-dataset seed intervals, and Holm-adjusted p values
  over all declared contrasts. Confidence intervals are unadjusted/descriptive;
  there is no formal equivalence claim. Three attribution datasets imply df=2,
  not 14 or 44, and limited population inference.
- No combined claims pool historical GPUs, campaigns or incompatible settings.

CPU-only verification (synthetic tensors/temporary fixtures, not experiment runs):

```powershell
Set-Location 'D:\Hussein-Files\PFB\pfb-forecasting'
& $py -X utf8 -m unittest experiments.controlled_addons.test_controls -v
```
