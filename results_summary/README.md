# Result records

The root summary CSVs retain their historical RTX 2080 Ti cohort labels.
They are separate from the newer controlled studies below.

| Location | Recorded scope |
|---|---|
| `controlled_addons/` | H=192 prospective RTX 4090 controls: 504 validation selections, 210 locked tests, 165 Projected controls and 160 normalization runs. |
| `review_campaigns/` | 3,942 completed historical/review records and matching scalar metric arrays. Of these, 3,807 explicitly identify RTX 4090; 135 have unspecified hardware. |
| `review_campaigns/verified_rtx4090_results.csv` | Only the 3,807 explicitly watermarked rows; no inferred hardware labels. |
| `plot_inputs/` | Retained numerical inputs for the benchmark, serial/parallel and Weather plots. |
| `factorial_cell_summary.csv`, `factorial_interval_audit.csv` | RTX 2080 Ti factorial cells and all 42 contrasts. The descriptive band is +/-1% of each dataset/metric mean across eight cells. Containment is not an adjusted equivalence test. |
| `capacity_task_winners.csv` | Nine three-seed capacity tasks: Direct leads five, Projected four. |

Each prospective run preserves its protocol, seed, configuration, source, data
and software fingerprints. Candidate vectors and locked configurations are
included. `PROVENANCE.json` describes the cohorts; `ARTIFACT_SHA256.json`
identifies the published code, result and figure payload.

Verify the checksums, scalar arrays, hardware labels and paired effects without
training:

```powershell
python analysis/verify_published_results.py
```

`experiments/controlled_addons/reference_source/` preserves the experiment source
snapshot, including all 50 recorded implementation fingerprints. The current
public interfaces additionally retain their existing normalization adapters and
baseline dispatch. Changes in source/environment create new run identities.

The compact records contain scalar metrics and receipts. Datasets, model weights,
prediction arrays and upstream source dependencies are obtained separately.
Hashes naming such artifacts record the original execution; they do not indicate
that the omitted files are distributed here. Scalar member scores cannot
reconstruct an output ensemble, so the retained ensemble scores are supplied.
