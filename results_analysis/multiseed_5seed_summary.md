---
# 🔬 Multi-Seed Experiment Results (Detailed)

Expected seeds: 2021, 2022, 2023, 2024, 2025

Coverage: 36/36 configurations complete (100.0%)

## Standard Datasets (H=192)

| Dataset | Model | Seeds | MSE | MAE | Status |
|---------|-------|-------|-----|-----|--------|
| ETTm1 | PatchFusionBERT_v0 | 5/5 | 0.3395+-0.0023 | 0.3743+-0.0018 | COMPLETE |
| ETTm1 | PatchFusionBERT_v2 | 5/5 | 0.3403+-0.0072 | 0.3753+-0.0056 | COMPLETE |
| ETTm1 | PatchTST | 5/5 | 0.3364+-0.0041 | 0.3753+-0.0020 | COMPLETE |
| ETTm1 | DLinear | 5/5 | 0.3351+-0.0020 | 0.3658+-0.0032 | COMPLETE |
| ETTm2 | PatchFusionBERT_v0 | 5/5 | 0.2287+-0.0042 | 0.2975+-0.0023 | COMPLETE |
| ETTm2 | PatchFusionBERT_v2 | 5/5 | 0.2323+-0.0031 | 0.2986+-0.0024 | COMPLETE |
| ETTm2 | PatchTST | 5/5 | 0.2370+-0.0055 | 0.3070+-0.0037 | COMPLETE |
| ETTm2 | DLinear | 5/5 | 0.2308+-0.0051 | 0.3104+-0.0065 | COMPLETE |
| ETTh1 | PatchFusionBERT_v0 | 5/5 | 0.4196+-0.0101 | 0.4297+-0.0065 | COMPLETE |
| ETTh1 | PatchFusionBERT_v2 | 5/5 | 0.4225+-0.0071 | 0.4308+-0.0066 | COMPLETE |
| ETTh1 | PatchTST | 5/5 | 0.4321+-0.0014 | 0.4360+-0.0047 | COMPLETE |
| ETTh1 | DLinear | 5/5 | 0.4074+-0.0023 | 0.4182+-0.0029 | COMPLETE |
| ETTh2 | PatchFusionBERT_v0 | 5/5 | 0.3740+-0.0083 | 0.3980+-0.0028 | COMPLETE |
| ETTh2 | PatchFusionBERT_v2 | 5/5 | 0.3699+-0.0065 | 0.3963+-0.0032 | COMPLETE |
| ETTh2 | PatchTST | 5/5 | 0.3824+-0.0071 | 0.4084+-0.0030 | COMPLETE |
| ETTh2 | DLinear | 5/5 | 0.3945+-0.0072 | 0.4259+-0.0052 | COMPLETE |
| Exchange | PatchFusionBERT_v0 | 5/5 | 0.2244+-0.0084 | 0.3395+-0.0087 | COMPLETE |
| Exchange | PatchFusionBERT_v2 | 5/5 | 0.2011+-0.0133 | 0.3217+-0.0109 | COMPLETE |
| Exchange | PatchTST | 5/5 | 0.1921+-0.0052 | 0.3180+-0.0050 | COMPLETE |
| Exchange | DLinear | 5/5 | 0.1779+-0.0056 | 0.3160+-0.0046 | COMPLETE |
| Weather | PatchFusionBERT_v0 | 5/5 | 0.1964+-0.0019 | 0.2429+-0.0022 | COMPLETE |
| Weather | PatchFusionBERT_v2 | 5/5 | 0.1941+-0.0014 | 0.2409+-0.0014 | COMPLETE |
| Weather | PatchTST | 5/5 | 0.1976+-0.0016 | 0.2440+-0.0017 | COMPLETE |
| Weather | DLinear | 5/5 | 0.2168+-0.0020 | 0.2757+-0.0042 | COMPLETE |

## Illness Dataset (H=24/48/60)

| Horizon | Model | Seeds | MSE | MAE | Status |
|---------|-------|-------|-----|-----|--------|
| 24 | PatchFusionBERT_v0 | 5/5 | 2.0422+-0.1302 | 0.9146+-0.0525 | COMPLETE |
| 24 | PatchFusionBERT_v2 | 5/5 | 2.1649+-0.1159 | 0.9709+-0.0469 | COMPLETE |
| 24 | PatchTST | 5/5 | 2.1236+-0.1389 | 0.9640+-0.0705 | COMPLETE |
| 24 | DLinear | 5/5 | 2.7162+-0.5102 | 1.1679+-0.1177 | COMPLETE |
| 48 | PatchFusionBERT_v0 | 5/5 | 1.9117+-0.0183 | 0.8969+-0.0066 | COMPLETE |
| 48 | PatchFusionBERT_v2 | 5/5 | 2.0432+-0.0488 | 0.9339+-0.0156 | COMPLETE |
| 48 | PatchTST | 5/5 | 2.0674+-0.1297 | 0.9601+-0.0569 | COMPLETE |
| 48 | DLinear | 5/5 | 2.7254+-0.5867 | 1.1616+-0.1125 | COMPLETE |
| 60 | PatchFusionBERT_v0 | 5/5 | 2.0191+-0.0552 | 0.9470+-0.0338 | COMPLETE |
| 60 | PatchFusionBERT_v2 | 5/5 | 2.0631+-0.0339 | 0.9478+-0.0118 | COMPLETE |
| 60 | PatchTST | 5/5 | 2.0715+-0.0994 | 0.9660+-0.0520 | COMPLETE |
| 60 | DLinear | 5/5 | 2.8996+-0.6421 | 1.2017+-0.1116 | COMPLETE |

## Coverage By Model

| Model | Complete Configs | Total Configs | Status |
|-------|------------------|---------------|--------|
| PatchFusionBERT_v0 | 9 | 9 | COMPLETE |
| PatchFusionBERT_v2 | 9 | 9 | COMPLETE |
| PatchTST | 9 | 9 | COMPLETE |
| DLinear | 9 | 9 | COMPLETE |
