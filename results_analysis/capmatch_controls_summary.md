# Capacity-Control Summary

Controls included:
- PatchTST base
- PatchTST width-match
- PatchTST depth-match
- PatchFusionBERT v0
- PatchFusionBERT v2

## ETTh2 H=96

- PatchFusionBERT v2: params=1,939,808, mse=0.2972, mae=0.3497, runs=3
- PatchFusionBERT v0: params=2,224,480, mse=0.3007, mae=0.3525, runs=3
- PatchTST width-match: params=2,205,120, mse=0.3155, mae=0.3677, runs=3
- PatchTST base: params=1,113,312, mse=0.3179, mae=0.3658, runs=3

- PFB v0 vs PatchTST base: delta_mse=+0.0173, delta_mae=+0.0133
- PFB v0 vs PatchTST width-match: delta_mse=+0.0149, delta_mae=+0.0152
- PFB v0 vs PatchFusionBERT v2: delta_mse=-0.0035, delta_mae=-0.0028

## ETTh2 H=192

- PatchFusionBERT v2: params=2,456,000, mse=0.3701, mae=0.3967, runs=3
- PatchFusionBERT v0: params=3,256,768, mse=0.3709, mae=0.3986, runs=3
- PatchTST depth-match: params=3,215,680, mse=0.3833, mae=0.4129, runs=3
- PatchTST base: params=1,629,504, mse=0.3848, mae=0.4079, runs=3
- PatchTST width-match: params=3,283,456, mse=0.3883, mae=0.4142, runs=3

- PFB v0 vs PatchTST base: delta_mse=+0.0138, delta_mae=+0.0093
- PFB v0 vs PatchTST width-match: delta_mse=+0.0173, delta_mae=+0.0156
- PFB v0 vs PatchTST depth-match: delta_mse=+0.0124, delta_mae=+0.0143
- PFB v0 vs PatchFusionBERT v2: delta_mse=-0.0008, delta_mae=-0.0019

## ETTh2 H=336

- PatchFusionBERT v0: params=4,805,200, mse=0.3868, mae=0.4171, runs=3
- PatchFusionBERT v2: params=3,230,288, mse=0.3945, mae=0.4196, runs=3
- PatchTST base: params=2,403,792, mse=0.4075, mae=0.4286, runs=3
- PatchTST width-match: params=4,808,112, mse=0.4238, mae=0.4433, runs=3

- PFB v0 vs PatchTST base: delta_mse=+0.0207, delta_mae=+0.0116
- PFB v0 vs PatchTST width-match: delta_mse=+0.0370, delta_mae=+0.0262
- PFB v0 vs PatchFusionBERT v2: delta_mse=+0.0077, delta_mae=+0.0025

## ETTm2 H=96

- PatchFusionBERT v0: params=2,224,480, mse=0.1707, mae=0.2572, runs=3
- PatchFusionBERT v2: params=1,939,808, mse=0.1717, mae=0.2576, runs=3
- PatchTST base: params=1,113,312, mse=0.1770, mae=0.2671, runs=3
- PatchTST width-match: params=2,205,120, mse=0.1811, mae=0.2701, runs=3

- PFB v0 vs PatchTST base: delta_mse=+0.0062, delta_mae=+0.0100
- PFB v0 vs PatchTST width-match: delta_mse=+0.0104, delta_mae=+0.0129
- PFB v0 vs PatchFusionBERT v2: delta_mse=+0.0010, delta_mae=+0.0005

## ETTm2 H=192

- PatchFusionBERT v0: params=3,256,768, mse=0.2287, mae=0.2974, runs=3
- PatchFusionBERT v2: params=2,456,000, mse=0.2318, mae=0.2977, runs=3
- PatchTST base: params=1,629,504, mse=0.2348, mae=0.3069, runs=3
- PatchTST width-match: params=3,283,456, mse=0.2374, mae=0.3115, runs=3
- PatchTST depth-match: params=3,215,680, mse=0.2413, mae=0.3089, runs=3

- PFB v0 vs PatchTST base: delta_mse=+0.0061, delta_mae=+0.0096
- PFB v0 vs PatchTST width-match: delta_mse=+0.0087, delta_mae=+0.0142
- PFB v0 vs PatchTST depth-match: delta_mse=+0.0127, delta_mae=+0.0116
- PFB v0 vs PatchFusionBERT v2: delta_mse=+0.0031, delta_mae=+0.0004

## ETTm2 H=336

- PatchFusionBERT v0: params=4,805,200, mse=0.2821, mae=0.3345, runs=3
- PatchTST width-match: params=4,808,112, mse=0.2898, mae=0.3445, runs=3
- PatchTST base: params=2,403,792, mse=0.2950, mae=0.3447, runs=3
- PatchFusionBERT v2: params=3,230,288, mse=0.2979, mae=0.3454, runs=3

- PFB v0 vs PatchTST base: delta_mse=+0.0129, delta_mae=+0.0102
- PFB v0 vs PatchTST width-match: delta_mse=+0.0077, delta_mae=+0.0100
- PFB v0 vs PatchFusionBERT v2: delta_mse=+0.0158, delta_mae=+0.0109

## Exchange H=192

- PatchTST base: params=1,629,504, mse=0.1904, mae=0.3164, runs=3
- PatchFusionBERT v2: params=2,456,000, mse=0.1958, mae=0.3178, runs=3
- PatchTST width-match: params=3,283,456, mse=0.2182, mae=0.3400, runs=3
- PatchFusionBERT v0: params=3,256,768, mse=0.2222, mae=0.3377, runs=3

- PFB v0 vs PatchTST base: delta_mse=-0.0319, delta_mae=-0.0214
- PFB v0 vs PatchTST width-match: delta_mse=-0.0040, delta_mae=+0.0023
- PFB v0 vs PatchFusionBERT v2: delta_mse=-0.0264, delta_mae=-0.0199

## Illness H=24

- PatchTST width-match: params=1,231,896, mse=1.8992, mae=0.8622, runs=3
- PatchTST depth-match: params=1,231,896, mse=1.8992, mae=0.8622, runs=3
- PatchFusionBERT v0: params=1,272,088, mse=2.0606, mae=0.9188, runs=3
- PatchTST base: params=637,080, mse=2.0759, mae=0.9474, runs=3
- PatchFusionBERT v2: params=1,463,576, mse=2.1998, mae=0.9818, runs=3

- PFB v0 vs PatchTST base: delta_mse=+0.0153, delta_mae=+0.0286
- PFB v0 vs PatchTST width-match: delta_mse=-0.1614, delta_mae=-0.0566
- PFB v0 vs PatchTST depth-match: delta_mse=-0.1614, delta_mae=-0.0566
- PFB v0 vs PatchFusionBERT v2: delta_mse=+0.1393, delta_mae=+0.0630

## Illness H=48

- PatchTST width-match: params=1,400,400, mse=1.8916, mae=0.8862, runs=3
- PatchFusionBERT v0: params=1,351,984, mse=1.9111, mae=0.8977, runs=3
- PatchTST depth-match: params=1,271,856, mse=2.0112, mae=0.9281, runs=3
- PatchFusionBERT v2: params=1,503,536, mse=2.0577, mae=0.9404, runs=3
- PatchTST base: params=677,040, mse=2.0784, mae=0.9662, runs=3

- PFB v0 vs PatchTST base: delta_mse=+0.1673, delta_mae=+0.0684
- PFB v0 vs PatchTST width-match: delta_mse=-0.0195, delta_mae=-0.0115
- PFB v0 vs PatchTST depth-match: delta_mse=+0.1001, delta_mae=+0.0303
- PFB v0 vs PatchFusionBERT v2: delta_mse=+0.1466, delta_mae=+0.0427

## Illness H=60

- PatchTST depth-match: params=1,490,108, mse=1.9460, mae=0.9134, runs=3
- PatchTST base: params=697,020, mse=2.0002, mae=0.9283, runs=3
- PatchFusionBERT v0: params=1,391,932, mse=2.0358, mae=0.9589, runs=3
- PatchFusionBERT v2: params=1,523,516, mse=2.0719, mae=0.9522, runs=3
- PatchTST width-match: params=1,421,628, mse=2.0972, mae=0.9648, runs=3

- PFB v0 vs PatchTST base: delta_mse=-0.0356, delta_mae=-0.0306
- PFB v0 vs PatchTST width-match: delta_mse=+0.0614, delta_mae=+0.0059
- PFB v0 vs PatchTST depth-match: delta_mse=-0.0899, delta_mae=-0.0456
- PFB v0 vs PatchFusionBERT v2: delta_mse=+0.0361, delta_mae=-0.0067

## Weather H=96

- PatchFusionBERT v0: params=2,224,480, mse=0.1492, mae=0.1981, runs=3
- PatchFusionBERT v2: params=1,939,808, mse=0.1496, mae=0.1985, runs=3
- PatchTST base: params=1,113,312, mse=0.1515, mae=0.2010, runs=3
- PatchTST width-match: params=2,205,120, mse=0.1546, mae=0.2029, runs=3

- PFB v0 vs PatchTST base: delta_mse=+0.0023, delta_mae=+0.0029
- PFB v0 vs PatchTST width-match: delta_mse=+0.0054, delta_mae=+0.0048
- PFB v0 vs PatchFusionBERT v2: delta_mse=+0.0004, delta_mae=+0.0004

## Weather H=192

- PatchFusionBERT v2: params=2,456,000, mse=0.1946, mae=0.2411, runs=3
- PatchFusionBERT v0: params=3,256,768, mse=0.1957, mae=0.2421, runs=3
- PatchTST depth-match: params=3,215,680, mse=0.1969, mae=0.2444, runs=3
- PatchTST base: params=1,629,504, mse=0.1977, mae=0.2437, runs=3
- PatchTST width-match: params=3,283,456, mse=0.2026, mae=0.2495, runs=3

- PFB v0 vs PatchTST base: delta_mse=+0.0021, delta_mae=+0.0016
- PFB v0 vs PatchTST width-match: delta_mse=+0.0070, delta_mae=+0.0073
- PFB v0 vs PatchTST depth-match: delta_mse=+0.0012, delta_mae=+0.0023
- PFB v0 vs PatchFusionBERT v2: delta_mse=-0.0010, delta_mae=-0.0010

## Weather H=336

- PatchFusionBERT v2: params=3,230,288, mse=0.2481, mae=0.2832, runs=3
- PatchFusionBERT v0: params=4,805,200, mse=0.2492, mae=0.2833, runs=3
- PatchTST base: params=2,403,792, mse=0.2539, mae=0.2852, runs=3
- PatchTST width-match: params=4,808,112, mse=0.2565, mae=0.2894, runs=3

- PFB v0 vs PatchTST base: delta_mse=+0.0047, delta_mae=+0.0018
- PFB v0 vs PatchTST width-match: delta_mse=+0.0073, delta_mae=+0.0060
- PFB v0 vs PatchFusionBERT v2: delta_mse=-0.0011, delta_mae=-0.0001

## Coverage

- PatchTST base total seed-runs aggregated: 39
- PatchTST width-match total seed-runs aggregated: 39
- PatchTST depth-match total seed-runs aggregated: 18
- PatchFusionBERT v0 total seed-runs aggregated: 39
- PatchFusionBERT v2 total seed-runs aggregated: 39
