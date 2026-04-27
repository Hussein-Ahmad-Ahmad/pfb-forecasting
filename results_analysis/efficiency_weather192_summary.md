# Efficiency Benchmark Summary (Weather H=192)

Protocol:
- Training wall-clock: 3 repeated runs (seeds 2021, 2022, 2023)
- Inference latency: 3 repeated benchmark sessions with warmup and timed forward passes
- Same machine and GPU for all models

## DLinear

- Params: 129,408
- Training wall-clock median: 321.37 s
- Training wall-clock std: 33.94 s
- Training peak GPU memory median: 1966.00 MB
- Inference latency median: 0.33 ms
- Inference latency std between runs: 0.00 ms
- Inference throughput median: 23820.87 samples/s
- Inference peak GPU memory median: 9.99 MB

## PatchTST

- Params: 1,629,504
- Training wall-clock median: 1018.09 s
- Training wall-clock std: 96.27 s
- Training peak GPU memory median: 2812.00 MB
- Inference latency median: 3.25 ms
- Inference latency std between runs: 0.25 ms
- Inference throughput median: 2382.12 samples/s
- Inference peak GPU memory median: 62.05 MB

## PatchTST_cap

- Params: 3,283,456
- Training wall-clock median: 1377.87 s
- Training wall-clock std: 60.92 s
- Training peak GPU memory median: 3107.00 MB
- Inference latency median: 6.64 ms
- Inference latency std between runs: 0.03 ms
- Inference throughput median: 1197.95 samples/s
- Inference peak GPU memory median: 192.64 MB

## PatchFusionBERT v0

- Params: 3,256,768
- Training wall-clock median: 1374.73 s
- Training wall-clock std: 71.63 s
- Training peak GPU memory median: 3205.00 MB
- Inference latency median: 5.87 ms
- Inference latency std between runs: 0.01 ms
- Inference throughput median: 1361.56 samples/s
- Inference peak GPU memory median: 71.71 MB

## PatchFusionBERT v2

- Params: 2,456,000
- Training wall-clock median: 1460.76 s
- Training wall-clock std: 3.12 s
- Training peak GPU memory median: 1994.00 MB
- Inference latency median: 6.75 ms
- Inference latency std between runs: 0.01 ms
- Inference throughput median: 1183.40 samples/s
- Inference peak GPU memory median: 68.65 MB
