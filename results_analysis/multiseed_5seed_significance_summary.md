# 5-Seed Significance Testing

Method: Wilcoxon signed-rank test on seed-aligned pairs. Lower is better.

**Multiple-comparison correction applied (added 2025-04-27):**
- Pooled file: 8 simultaneous tests → Holm-Bonferroni + Benjamini-Hochberg FDR correction applied.
- Per-config file: 72 simultaneous tests → same corrections applied.
- Corrected p-values in columns `p_holm` and `p_fdr_bh` in both CSVs.

## Multiple-Comparison Correction Summary (Pooled, 8 tests)

| Comparison | Metric | p_raw | p_holm | p_fdr_bh | Sig (Holm) | Sig (BH-FDR) |
|---|---|---|---|---|---|---|
| PFBv0 vs DLinear | MSE | 0.003907 | 0.027350 | 0.015629 | **Yes** | **Yes** |
| PFBv0 vs DLinear | MAE | 0.000048 | 0.000388 | 0.000388 | **Yes** | **Yes** |
| PFBv0 vs PatchTST | MAE | 0.011624 | 0.069745 | 0.030998 | No | **Yes** |
| PFBv0 vs PatchTST | MSE | 0.034875 | 0.174376 | 0.069751 | No | No |
| PFBv0 vs PFBv2 | MSE | 0.047536 | 0.190144 | 0.076058 | No | No |
| PFBv0 vs PFBv2 | MAE | 0.480683 | 0.961366 | 0.538281 | No | No |
| PFBv2 vs PatchTST | MSE | 0.538281 | 0.961366 | 0.538281 | No | No |
| PFBv2 vs PatchTST | MAE | 0.134285 | 0.402854 | 0.179046 | No | No |

**Key finding:** After Holm correction, only PFBv0 vs DLinear (MSE and MAE) remain significant at α=0.05. The PFBv0 vs PatchTST comparison (MSE p=0.035 raw) **loses significance** after any correction. Under BH-FDR, PFBv0 vs PatchTST MAE additionally survives (p_fdr=0.031).

**Per-config tests (72 tests):** N=5 per config gives minimum Wilcoxon resolution of p≥0.0625, meaning none reach uncorrected significance. All per-config results should be reported as directional trends only, not formal significance claims.

**Manuscript implication:** The strong claim is "PFBv0 significantly outperforms DLinear (corrected p<0.05)." The PatchTST comparison is borderline — significant raw but not after Holm; significant under BH-FDR for MAE only. Report uncorrected p alongside corrected p and note the boundary.

## Pooled Results

| Comparison | Metric | N | Mean Diff (A-B) | p-value | Significant | Winner | Rank-Biserial |
|------------|--------|---|-----------------|---------|-------------|--------|---------------|
| PatchFusionBERT_v0 vs DLinear | MAE | 45 | -0.089210 | 0.000048 | Yes | PatchFusionBERT_v0 | -0.6638 |
| PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MAE | 45 | -0.008409 | 0.480683 | No | PatchFusionBERT_v0 | -0.1227 |
| PatchFusionBERT_v0 vs PatchTST | MAE | 45 | -0.015382 | 0.011624 | Yes | PatchFusionBERT_v0 | -0.4280 |
| PatchFusionBERT_v2 vs PatchTST | MAE | 45 | -0.006973 | 0.134285 | No | PatchFusionBERT_v2 | -0.2580 |
| PatchFusionBERT_v0 vs DLinear | MSE | 45 | -0.260907 | 0.003907 | Yes | PatchFusionBERT_v0 | -0.4860 |
| PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MSE | 45 | -0.030651 | 0.047536 | Yes | PatchFusionBERT_v0 | -0.3391 |
| PatchFusionBERT_v0 vs PatchTST | MSE | 45 | -0.031626 | 0.034875 | Yes | PatchFusionBERT_v0 | -0.3604 |
| PatchFusionBERT_v2 vs PatchTST | MSE | 45 | -0.000975 | 0.538281 | No | PatchFusionBERT_v2 | -0.1072 |

## Per-Configuration Results

| Dataset | Horizon | Comparison | Metric | N | Mean Diff (A-B) | p-value | Significant | Winner |
|---------|---------|------------|--------|---|-----------------|---------|-------------|--------|
| ETTh1 | 192 | PatchFusionBERT_v0 vs DLinear | MAE | 5 | 0.011475 | 0.062500 | No | DLinear |
| ETTh1 | 192 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MAE | 5 | -0.001066 | 1.000000 | No | PatchFusionBERT_v0 |
| ETTh1 | 192 | PatchFusionBERT_v0 vs PatchTST | MAE | 5 | -0.006350 | 0.187500 | No | PatchFusionBERT_v0 |
| ETTh1 | 192 | PatchFusionBERT_v2 vs PatchTST | MAE | 5 | -0.005284 | 0.125000 | No | PatchFusionBERT_v2 |
| ETTh1 | 192 | PatchFusionBERT_v0 vs DLinear | MSE | 5 | 0.012205 | 0.125000 | No | DLinear |
| ETTh1 | 192 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MSE | 5 | -0.002982 | 0.812500 | No | PatchFusionBERT_v0 |
| ETTh1 | 192 | PatchFusionBERT_v0 vs PatchTST | MSE | 5 | -0.012566 | 0.125000 | No | PatchFusionBERT_v0 |
| ETTh1 | 192 | PatchFusionBERT_v2 vs PatchTST | MSE | 5 | -0.009584 | 0.062500 | No | PatchFusionBERT_v2 |
| ETTh2 | 192 | PatchFusionBERT_v0 vs DLinear | MAE | 5 | -0.027960 | 0.062500 | No | PatchFusionBERT_v0 |
| ETTh2 | 192 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MAE | 5 | 0.001687 | 0.625000 | No | PatchFusionBERT_v2 |
| ETTh2 | 192 | PatchFusionBERT_v0 vs PatchTST | MAE | 5 | -0.010451 | 0.062500 | No | PatchFusionBERT_v0 |
| ETTh2 | 192 | PatchFusionBERT_v2 vs PatchTST | MAE | 5 | -0.012138 | 0.062500 | No | PatchFusionBERT_v2 |
| ETTh2 | 192 | PatchFusionBERT_v0 vs DLinear | MSE | 5 | -0.020517 | 0.062500 | No | PatchFusionBERT_v0 |
| ETTh2 | 192 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MSE | 5 | 0.004101 | 0.812500 | No | PatchFusionBERT_v2 |
| ETTh2 | 192 | PatchFusionBERT_v0 vs PatchTST | MSE | 5 | -0.008370 | 0.125000 | No | PatchFusionBERT_v0 |
| ETTh2 | 192 | PatchFusionBERT_v2 vs PatchTST | MSE | 5 | -0.012471 | 0.187500 | No | PatchFusionBERT_v2 |
| ETTm1 | 192 | PatchFusionBERT_v0 vs DLinear | MAE | 5 | 0.008511 | 0.062500 | No | DLinear |
| ETTm1 | 192 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MAE | 5 | -0.000990 | 0.812500 | No | PatchFusionBERT_v0 |
| ETTm1 | 192 | PatchFusionBERT_v0 vs PatchTST | MAE | 5 | -0.000931 | 0.312500 | No | PatchFusionBERT_v0 |
| ETTm1 | 192 | PatchFusionBERT_v2 vs PatchTST | MAE | 5 | 0.000059 | 1.000000 | No | PatchTST |
| ETTm1 | 192 | PatchFusionBERT_v0 vs DLinear | MSE | 5 | 0.004373 | 0.125000 | No | DLinear |
| ETTm1 | 192 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MSE | 5 | -0.000810 | 0.812500 | No | PatchFusionBERT_v0 |
| ETTm1 | 192 | PatchFusionBERT_v0 vs PatchTST | MSE | 5 | 0.003117 | 0.312500 | No | PatchTST |
| ETTm1 | 192 | PatchFusionBERT_v2 vs PatchTST | MSE | 5 | 0.003927 | 0.437500 | No | PatchTST |
| ETTm2 | 192 | PatchFusionBERT_v0 vs DLinear | MAE | 5 | -0.012871 | 0.062500 | No | PatchFusionBERT_v0 |
| ETTm2 | 192 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MAE | 5 | -0.001097 | 0.437500 | No | PatchFusionBERT_v0 |
| ETTm2 | 192 | PatchFusionBERT_v0 vs PatchTST | MAE | 5 | -0.009476 | 0.062500 | No | PatchFusionBERT_v0 |
| ETTm2 | 192 | PatchFusionBERT_v2 vs PatchTST | MAE | 5 | -0.008379 | 0.062500 | No | PatchFusionBERT_v2 |
| ETTm2 | 192 | PatchFusionBERT_v0 vs DLinear | MSE | 5 | -0.002110 | 0.625000 | No | PatchFusionBERT_v0 |
| ETTm2 | 192 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MSE | 5 | -0.003642 | 0.062500 | No | PatchFusionBERT_v0 |
| ETTm2 | 192 | PatchFusionBERT_v0 vs PatchTST | MSE | 5 | -0.008314 | 0.125000 | No | PatchFusionBERT_v0 |
| ETTm2 | 192 | PatchFusionBERT_v2 vs PatchTST | MSE | 5 | -0.004672 | 0.187500 | No | PatchFusionBERT_v2 |
| Exchange | 192 | PatchFusionBERT_v0 vs DLinear | MAE | 5 | 0.023453 | 0.062500 | No | DLinear |
| Exchange | 192 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MAE | 5 | 0.017785 | 0.062500 | No | PatchFusionBERT_v2 |
| Exchange | 192 | PatchFusionBERT_v0 vs PatchTST | MAE | 5 | 0.021481 | 0.062500 | No | PatchTST |
| Exchange | 192 | PatchFusionBERT_v2 vs PatchTST | MAE | 5 | 0.003696 | 0.312500 | No | PatchTST |
| Exchange | 192 | PatchFusionBERT_v0 vs DLinear | MSE | 5 | 0.046497 | 0.062500 | No | DLinear |
| Exchange | 192 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MSE | 5 | 0.023266 | 0.062500 | No | PatchFusionBERT_v2 |
| Exchange | 192 | PatchFusionBERT_v0 vs PatchTST | MSE | 5 | 0.032292 | 0.062500 | No | PatchTST |
| Exchange | 192 | PatchFusionBERT_v2 vs PatchTST | MSE | 5 | 0.009026 | 0.125000 | No | PatchTST |
| Illness | 24 | PatchFusionBERT_v0 vs DLinear | MAE | 5 | -0.253297 | 0.062500 | No | PatchFusionBERT_v0 |
| Illness | 24 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MAE | 5 | -0.056249 | 0.125000 | No | PatchFusionBERT_v0 |
| Illness | 24 | PatchFusionBERT_v0 vs PatchTST | MAE | 5 | -0.049354 | 0.312500 | No | PatchFusionBERT_v0 |
| Illness | 24 | PatchFusionBERT_v2 vs PatchTST | MAE | 5 | 0.006895 | 1.000000 | No | PatchTST |
| Illness | 24 | PatchFusionBERT_v0 vs DLinear | MSE | 5 | -0.674020 | 0.125000 | No | PatchFusionBERT_v0 |
| Illness | 24 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MSE | 5 | -0.122710 | 0.062500 | No | PatchFusionBERT_v0 |
| Illness | 24 | PatchFusionBERT_v0 vs PatchTST | MSE | 5 | -0.081409 | 0.625000 | No | PatchFusionBERT_v0 |
| Illness | 24 | PatchFusionBERT_v2 vs PatchTST | MSE | 5 | 0.041300 | 0.812500 | No | PatchTST |
| Illness | 48 | PatchFusionBERT_v0 vs DLinear | MAE | 5 | -0.264669 | 0.062500 | No | PatchFusionBERT_v0 |
| Illness | 48 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MAE | 5 | -0.036979 | 0.062500 | No | PatchFusionBERT_v0 |
| Illness | 48 | PatchFusionBERT_v0 vs PatchTST | MAE | 5 | -0.063231 | 0.125000 | No | PatchFusionBERT_v0 |
| Illness | 48 | PatchFusionBERT_v2 vs PatchTST | MAE | 5 | -0.026252 | 0.312500 | No | PatchFusionBERT_v2 |
| Illness | 48 | PatchFusionBERT_v0 vs DLinear | MSE | 5 | -0.813711 | 0.062500 | No | PatchFusionBERT_v0 |
| Illness | 48 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MSE | 5 | -0.131439 | 0.062500 | No | PatchFusionBERT_v0 |
| Illness | 48 | PatchFusionBERT_v0 vs PatchTST | MSE | 5 | -0.155727 | 0.062500 | No | PatchFusionBERT_v0 |
| Illness | 48 | PatchFusionBERT_v2 vs PatchTST | MSE | 5 | -0.024288 | 0.625000 | No | PatchFusionBERT_v2 |
| Illness | 60 | PatchFusionBERT_v0 vs DLinear | MAE | 5 | -0.254716 | 0.062500 | No | PatchFusionBERT_v0 |
| Illness | 60 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MAE | 5 | -0.000799 | 1.000000 | No | PatchFusionBERT_v0 |
| Illness | 60 | PatchFusionBERT_v0 vs PatchTST | MAE | 5 | -0.019001 | 0.625000 | No | PatchFusionBERT_v0 |
| Illness | 60 | PatchFusionBERT_v2 vs PatchTST | MAE | 5 | -0.018201 | 0.812500 | No | PatchFusionBERT_v2 |
| Illness | 60 | PatchFusionBERT_v0 vs DLinear | MSE | 5 | -0.880542 | 0.062500 | No | PatchFusionBERT_v0 |
| Illness | 60 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MSE | 5 | -0.044017 | 0.187500 | No | PatchFusionBERT_v0 |
| Illness | 60 | PatchFusionBERT_v0 vs PatchTST | MSE | 5 | -0.052446 | 0.625000 | No | PatchFusionBERT_v0 |
| Illness | 60 | PatchFusionBERT_v2 vs PatchTST | MSE | 5 | -0.008429 | 1.000000 | No | PatchFusionBERT_v2 |
| Weather | 192 | PatchFusionBERT_v0 vs DLinear | MAE | 5 | -0.032812 | 0.062500 | No | PatchFusionBERT_v0 |
| Weather | 192 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MAE | 5 | 0.002031 | 0.062500 | No | PatchFusionBERT_v2 |
| Weather | 192 | PatchFusionBERT_v0 vs PatchTST | MAE | 5 | -0.001123 | 0.187500 | No | PatchFusionBERT_v0 |
| Weather | 192 | PatchFusionBERT_v2 vs PatchTST | MAE | 5 | -0.003154 | 0.062500 | No | PatchFusionBERT_v2 |
| Weather | 192 | PatchFusionBERT_v0 vs DLinear | MSE | 5 | -0.020341 | 0.062500 | No | PatchFusionBERT_v0 |
| Weather | 192 | PatchFusionBERT_v0 vs PatchFusionBERT_v2 | MSE | 5 | 0.002375 | 0.062500 | No | PatchFusionBERT_v2 |
| Weather | 192 | PatchFusionBERT_v0 vs PatchTST | MSE | 5 | -0.001214 | 0.312500 | No | PatchFusionBERT_v0 |
| Weather | 192 | PatchFusionBERT_v2 vs PatchTST | MSE | 5 | -0.003589 | 0.062500 | No | PatchFusionBERT_v2 |
