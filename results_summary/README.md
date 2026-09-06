# Result summaries

This folder contains compact CSV summaries for the current PFB forecasting
package. It does not contain raw predictions, full checkpoints, or per-run result
arrays.

Each CSV includes `experiment_run_on`, which records the GPU used for the
reported experiment. Current summaries use:

```text
experiment_run_on = NVIDIA GeForce RTX 2080 Ti
```

When results from another GPU are added, keep them separate until the run
status, command arguments, and metric files are verified.
