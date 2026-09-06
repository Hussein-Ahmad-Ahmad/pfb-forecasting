# Result summaries

This folder contains compact CSV summaries for the current PFB forecasting
package. It does not contain raw predictions, full checkpoints, or per-run result
arrays.

Each CSV includes:

- `source_gpu`: GPU used for the recorded local result.
- `source_note`: short provenance label.

Current local summaries are marked:

```text
source_gpu = NVIDIA GeForce RTX 2080 Ti
source_note = local workstation result
```

When results from another GPU are added, keep them separate until the run
status, command arguments, and metric files are verified.
