# B2 Broader Robustness Matrix

Scope: Weather and ETTm2, H=96 and H=192, clean plus random/block missingness.

## ETTm2 (H=96)

### Random missingness

- Rate 0.1: best MSE = PatchFusionBERT_v2
  DLinear: mse=0.2432, mae=0.3366, DeltaMSE=+43.6%
  PatchTST: mse=0.2461, mae=0.3275, DeltaMSE=+41.3%
  PatchFusionBERT_v0: mse=0.2339, mae=0.3233, DeltaMSE=+39.3%
  PatchFusionBERT_v2: mse=0.2310, mae=0.3127, DeltaMSE=+36.2%

- Rate 0.2: best MSE = PatchFusionBERT_v2
  DLinear: mse=0.3652, mae=0.4240, DeltaMSE=+115.6%
  PatchTST: mse=0.3713, mae=0.4147, DeltaMSE=+113.1%
  PatchFusionBERT_v0: mse=0.3456, mae=0.4075, DeltaMSE=+105.8%
  PatchFusionBERT_v2: mse=0.3130, mae=0.3817, DeltaMSE=+84.6%

- Rate 0.3: best MSE = PatchFusionBERT_v2
  DLinear: mse=0.5385, mae=0.5251, DeltaMSE=+217.9%
  PatchTST: mse=0.5531, mae=0.5195, DeltaMSE=+217.5%
  PatchFusionBERT_v0: mse=0.5105, mae=0.5079, DeltaMSE=+204.0%
  PatchFusionBERT_v2: mse=0.4456, mae=0.4709, DeltaMSE=+162.8%

### Block missingness

- Rate 0.1: best MSE = PatchFusionBERT_v2
  DLinear: mse=0.2245, mae=0.3148, DeltaMSE=+32.5%
  PatchTST: mse=0.2347, mae=0.3240, DeltaMSE=+34.8%
  PatchFusionBERT_v0: mse=0.2207, mae=0.3105, DeltaMSE=+31.4%
  PatchFusionBERT_v2: mse=0.2207, mae=0.3103, DeltaMSE=+30.2%

- Rate 0.2: best MSE = PatchFusionBERT_v0
  DLinear: mse=0.2759, mae=0.3519, DeltaMSE=+62.8%
  PatchTST: mse=0.2740, mae=0.3539, DeltaMSE=+57.3%
  PatchFusionBERT_v0: mse=0.2667, mae=0.3453, DeltaMSE=+58.8%
  PatchFusionBERT_v2: mse=0.2667, mae=0.3444, DeltaMSE=+57.3%

- Rate 0.3: best MSE = PatchFusionBERT_v2
  DLinear: mse=0.3646, mae=0.4067, DeltaMSE=+115.2%
  PatchTST: mse=0.4037, mae=0.4320, DeltaMSE=+131.7%
  PatchFusionBERT_v0: mse=0.3555, mae=0.4046, DeltaMSE=+111.7%
  PatchFusionBERT_v2: mse=0.3529, mae=0.3996, DeltaMSE=+108.2%

## ETTm2 (H=192)

### Random missingness

- Rate 0.1: best MSE = PatchFusionBERT_v0
  DLinear: mse=0.3151, mae=0.3831, DeltaMSE=+33.7%
  PatchTST: mse=0.2911, mae=0.3585, DeltaMSE=+27.6%
  PatchFusionBERT_v0: mse=0.2718, mae=0.3422, DeltaMSE=+15.8%
  PatchFusionBERT_v2: mse=0.2932, mae=0.3525, DeltaMSE=+24.9%

- Rate 0.2: best MSE = PatchFusionBERT_v0
  DLinear: mse=0.4397, mae=0.4658, DeltaMSE=+86.5%
  PatchTST: mse=0.3951, mae=0.4256, DeltaMSE=+73.2%
  PatchFusionBERT_v0: mse=0.3519, mae=0.4072, DeltaMSE=+49.9%
  PatchFusionBERT_v2: mse=0.3762, mae=0.4164, DeltaMSE=+60.3%

- Rate 0.3: best MSE = PatchFusionBERT_v0
  DLinear: mse=0.6133, mae=0.5611, DeltaMSE=+160.2%
  PatchTST: mse=0.5480, mae=0.5080, DeltaMSE=+140.1%
  PatchFusionBERT_v0: mse=0.4828, mae=0.4919, DeltaMSE=+105.7%
  PatchFusionBERT_v2: mse=0.4965, mae=0.4943, DeltaMSE=+111.5%

### Block missingness

- Rate 0.1: best MSE = DLinear
  DLinear: mse=0.2916, mae=0.3587, DeltaMSE=+23.7%
  PatchTST: mse=0.3511, mae=0.4047, DeltaMSE=+53.8%
  PatchFusionBERT_v0: mse=0.2995, mae=0.3574, DeltaMSE=+27.6%
  PatchFusionBERT_v2: mse=0.3351, mae=0.3787, DeltaMSE=+42.8%

- Rate 0.2: best MSE = DLinear
  DLinear: mse=0.3439, mae=0.3934, DeltaMSE=+45.9%
  PatchTST: mse=0.4587, mae=0.4599, DeltaMSE=+101.0%
  PatchFusionBERT_v0: mse=0.3744, mae=0.4059, DeltaMSE=+59.5%
  PatchFusionBERT_v2: mse=0.4460, mae=0.4430, DeltaMSE=+90.0%

- Rate 0.3: best MSE = DLinear
  DLinear: mse=0.4352, mae=0.4469, DeltaMSE=+84.6%
  PatchTST: mse=0.5609, mae=0.5058, DeltaMSE=+145.8%
  PatchFusionBERT_v0: mse=0.5475, mae=0.4999, DeltaMSE=+133.3%
  PatchFusionBERT_v2: mse=0.6496, mae=0.5347, DeltaMSE=+176.8%

## Weather (H=96)

### Random missingness

- Rate 0.1: best MSE = PatchFusionBERT_v0
  DLinear: mse=0.1845, mae=0.2568, DeltaMSE=+5.9%
  PatchTST: mse=0.1698, mae=0.2321, DeltaMSE=+11.3%
  PatchFusionBERT_v0: mse=0.1661, mae=0.2280, DeltaMSE=+10.9%
  PatchFusionBERT_v2: mse=0.1665, mae=0.2296, DeltaMSE=+11.3%

- Rate 0.2: best MSE = PatchFusionBERT_v0
  DLinear: mse=0.2027, mae=0.2861, DeltaMSE=+16.3%
  PatchTST: mse=0.1901, mae=0.2636, DeltaMSE=+24.6%
  PatchFusionBERT_v0: mse=0.1845, mae=0.2597, DeltaMSE=+23.2%
  PatchFusionBERT_v2: mse=0.1864, mae=0.2613, DeltaMSE=+24.6%

- Rate 0.3: best MSE = PatchFusionBERT_v0
  DLinear: mse=0.2289, mae=0.3199, DeltaMSE=+31.3%
  PatchTST: mse=0.2182, mae=0.2996, DeltaMSE=+43.0%
  PatchFusionBERT_v0: mse=0.2115, mae=0.2967, DeltaMSE=+41.2%
  PatchFusionBERT_v2: mse=0.2142, mae=0.2976, DeltaMSE=+43.1%

### Block missingness

- Rate 0.1: best MSE = PatchFusionBERT_v0
  DLinear: mse=0.1838, mae=0.2493, DeltaMSE=+5.4%
  PatchTST: mse=0.1592, mae=0.2159, DeltaMSE=+4.3%
  PatchFusionBERT_v0: mse=0.1576, mae=0.2118, DeltaMSE=+5.2%
  PatchFusionBERT_v2: mse=0.1579, mae=0.2131, DeltaMSE=+5.5%

- Rate 0.2: best MSE = PatchFusionBERT_v0
  DLinear: mse=0.1931, mae=0.2630, DeltaMSE=+10.8%
  PatchTST: mse=0.1637, mae=0.2225, DeltaMSE=+7.3%
  PatchFusionBERT_v0: mse=0.1628, mae=0.2207, DeltaMSE=+8.7%
  PatchFusionBERT_v2: mse=0.1628, mae=0.2215, DeltaMSE=+8.8%

- Rate 0.3: best MSE = PatchFusionBERT_v0
  DLinear: mse=0.2116, mae=0.2869, DeltaMSE=+21.4%
  PatchTST: mse=0.1754, mae=0.2398, DeltaMSE=+14.9%
  PatchFusionBERT_v0: mse=0.1738, mae=0.2369, DeltaMSE=+16.1%
  PatchFusionBERT_v2: mse=0.1755, mae=0.2407, DeltaMSE=+17.3%

## Weather (H=192)

### Random missingness

- Rate 0.1: best MSE = PatchFusionBERT_v2
  DLinear: mse=0.2306, mae=0.3013, DeltaMSE=+5.7%
  PatchTST: mse=0.2073, mae=0.2672, DeltaMSE=+5.6%
  PatchFusionBERT_v0: mse=0.2026, mae=0.2563, DeltaMSE=+4.7%
  PatchFusionBERT_v2: mse=0.2012, mae=0.2565, DeltaMSE=+3.8%

- Rate 0.2: best MSE = PatchFusionBERT_v2
  DLinear: mse=0.2497, mae=0.3279, DeltaMSE=+14.5%
  PatchTST: mse=0.2220, mae=0.2931, DeltaMSE=+13.2%
  PatchFusionBERT_v0: mse=0.2152, mae=0.2796, DeltaMSE=+11.2%
  PatchFusionBERT_v2: mse=0.2148, mae=0.2815, DeltaMSE=+10.8%

- Rate 0.3: best MSE = PatchFusionBERT_v0
  DLinear: mse=0.2752, mae=0.3577, DeltaMSE=+26.2%
  PatchTST: mse=0.2441, mae=0.3229, DeltaMSE=+24.4%
  PatchFusionBERT_v0: mse=0.2364, mae=0.3096, DeltaMSE=+22.1%
  PatchFusionBERT_v2: mse=0.2378, mae=0.3135, DeltaMSE=+22.7%

### Block missingness

- Rate 0.1: best MSE = PatchFusionBERT_v2
  DLinear: mse=0.2278, mae=0.2922, DeltaMSE=+4.4%
  PatchTST: mse=0.2041, mae=0.2558, DeltaMSE=+4.0%
  PatchFusionBERT_v0: mse=0.2020, mae=0.2515, DeltaMSE=+4.3%
  PatchFusionBERT_v2: mse=0.1997, mae=0.2502, DeltaMSE=+3.1%

- Rate 0.2: best MSE = PatchFusionBERT_v2
  DLinear: mse=0.2374, mae=0.3046, DeltaMSE=+8.8%
  PatchTST: mse=0.2082, mae=0.2625, DeltaMSE=+6.1%
  PatchFusionBERT_v0: mse=0.2058, mae=0.2573, DeltaMSE=+6.3%
  PatchFusionBERT_v2: mse=0.2034, mae=0.2567, DeltaMSE=+5.0%

- Rate 0.3: best MSE = PatchFusionBERT_v2
  DLinear: mse=0.2560, mae=0.3266, DeltaMSE=+17.4%
  PatchTST: mse=0.2144, mae=0.2722, DeltaMSE=+9.3%
  PatchFusionBERT_v0: mse=0.2157, mae=0.2707, DeltaMSE=+11.4%
  PatchFusionBERT_v2: mse=0.2137, mae=0.2732, DeltaMSE=+10.3%
