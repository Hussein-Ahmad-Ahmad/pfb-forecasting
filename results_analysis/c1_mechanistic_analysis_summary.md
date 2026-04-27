# C1 Mechanistic Analysis

Goal: explain why PatchFusion-style models help more on ETTm2 and Weather than on Exchange using dataset characteristics rather than model-internal probes.

## 5-seed H=192 fusion advantage

- Weather: gain of +1.82% MSE vs best non-fusion baseline (PatchFusionBERT_v2 vs PatchTST); Highest channel count and relatively low redundancy; auxiliary sensors look complementary rather than duplicated, which favors fusion.
- ETTm2: gain of +0.91% MSE vs best non-fusion baseline (PatchFusionBERT_v0 vs DLinear); Strong short- and medium-range periodicity with moderate channel diversity; fusion has structured multivariate context to exploit.
- Exchange: drop of -13.06% MSE vs best non-fusion baseline (PatchFusionBERT_v2 vs DLinear); Shorter, highly collinear series with very strong persistence; most channels move together, so fusion adds less beyond strong simple baselines.

## Dataset characteristics

- ETTm2: rows=69680, features=6, sampling=15 min, lag1 autocorr=0.999, day autocorr=0.934, redundancy=0.366, diversity=0.511, top target-feature corr=0.495.
- Exchange: rows=7588, features=7, sampling=1440 min, lag1 autocorr=0.999, week autocorr=0.994, redundancy=0.474, diversity=0.412, top target-feature corr=0.901.
- Weather: rows=52696, features=20, sampling=10 min, lag1 autocorr=0.960, day autocorr=0.004, redundancy=0.323, diversity=0.554, top target-feature corr=0.140.

## Interpretation

- ETTm2 combines strong repeated temporal structure with non-trivial multivariate context, so fusion can improve over plain patching when the model exploits shared periodic patterns across channels.
- Weather has the richest sensor set and lower shared variance concentration than Exchange, so fusion appears most useful when many partially complementary channels must be combined.
- Exchange is the most redundant of the three focus datasets: channels are highly co-moving and the target is strongly explained by a few linear relationships, which reduces the marginal value of fusion relative to simpler baselines.

## Horizon check from broader aggregated results

- ETTm2: H=96: +0.52%, H=192: -1.26%, H=336: +7.07%
- Exchange: H=192: -0.61%, H=336: +7.87%
- Weather: H=192: +1.03%, H=336: -1.79%
