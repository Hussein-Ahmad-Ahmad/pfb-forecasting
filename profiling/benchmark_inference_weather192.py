from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from models import DLinear, PatchFusionBERT_v0, PatchFusionBERT_v2, PatchTST


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results_analysis"
RAW_CSV = OUT_DIR / "efficiency_weather192_inference_raw.csv"
SUMMARY_CSV = OUT_DIR / "efficiency_weather192_inference_summary.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
SEQ_LEN = 336
LABEL_LEN = 96
PRED_LEN = 192
ENC_IN = 21
WARMUP_RUNS = 10
MEASURE_RUNS = 50
REPEATS = 3


class Args:
    def __init__(self, model_name: str):
        self.task_name = "long_term_forecast"
        self.model = model_name
        self.seq_len = SEQ_LEN
        self.pred_len = PRED_LEN
        self.label_len = LABEL_LEN
        self.features = "M"
        self.freq = "h"
        self.enc_in = ENC_IN
        self.dec_in = ENC_IN
        self.c_out = ENC_IN
        self.d_model = 128
        self.n_heads = 8
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 512
        self.moving_avg = 25
        self.factor = 3
        self.dropout = 0.1
        self.activation = "gelu"
        self.embed = "timeF"
        self.distil = True
        self.expand = 2
        self.d_conv = 4
        self.top_k = 5
        self.num_kernels = 6
        self.channel_independence = 1
        self.decomp_method = "moving_avg"
        self.use_norm = 1
        self.down_sampling_layers = 0
        self.down_sampling_window = 1
        self.down_sampling_method = None
        self.seg_len = 96
        self.num_class = 1
        self.patch_len = 16
        self.stride = 8
        self.individual = False
        self.pfb_k = 0


MODEL_MAP = {
    "DLinear": DLinear,
    "PatchTST": PatchTST,
    "PatchTST_capacity": PatchTST,
    "PatchFusionBERT_v0": PatchFusionBERT_v0,
    "PatchFusionBERT_v2": PatchFusionBERT_v2,
}


def build_model(model_name: str) -> torch.nn.Module:
    module = MODEL_MAP[model_name]
    args = Args(model_name)
    if model_name == "PatchTST_capacity":
        args.d_model = 192
        args.e_layers = 5
    return module.Model(args).float().to(DEVICE)


def query_gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def benchmark_model(
    model_name: str,
    warmup_runs: int,
    measure_runs: int,
    repeats: int,
    batch_size: int,
) -> List[Dict[str, object]]:
    model = build_model(model_name)
    model.eval()

    x_enc = torch.randn(batch_size, SEQ_LEN, ENC_IN, device=DEVICE)
    x_mark_enc = torch.randn(batch_size, SEQ_LEN, 4, device=DEVICE)
    x_dec = torch.randn(batch_size, LABEL_LEN + PRED_LEN, ENC_IN, device=DEVICE)
    x_mark_dec = torch.randn(batch_size, LABEL_LEN + PRED_LEN, 4, device=DEVICE)

    params = sum(parameter.numel() for parameter in model.parameters())
    rows: List[Dict[str, object]] = []

    for repeat in range(1, repeats + 1):
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()

        latencies_ms = []
        with torch.no_grad():
            for _ in range(measure_runs):
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies_ms.append((end - start) * 1000.0)

        peak_gpu_memory_mb = 0.0
        if DEVICE == "cuda":
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

        latency_mean_ms = statistics.mean(latencies_ms)
        latency_median_ms = statistics.median(latencies_ms)
        latency_std_ms = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0

        rows.append(
            {
                "model": model_name,
                "repeat": repeat,
                "gpu_name": query_gpu_name(),
                "params": params,
                "latency_mean_ms": round(latency_mean_ms, 4),
                "latency_median_ms": round(latency_median_ms, 4),
                "latency_std_ms_within_repeat": round(latency_std_ms, 4),
                "throughput_samples_per_sec": round((batch_size * 1000.0) / latency_mean_ms, 4),
                "peak_gpu_memory_mb": round(peak_gpu_memory_mb, 2),
            }
        )

    del model, x_enc, x_mark_enc, x_dec, x_mark_dec
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for model, model_rows in sorted(grouped.items()):
        lat_medians = [float(row["latency_median_ms"]) for row in model_rows]
        throughputs = [float(row["throughput_samples_per_sec"]) for row in model_rows]
        peak_memories = [float(row["peak_gpu_memory_mb"]) for row in model_rows]
        summary_rows.append(
            {
                "model": model,
                "runs": len(model_rows),
                "gpu_name": model_rows[0]["gpu_name"],
                "params": model_rows[0]["params"],
                "latency_median_ms": round(statistics.median(lat_medians), 4),
                "latency_std_between_runs_ms": round(statistics.pstdev(lat_medians) if len(lat_medians) > 1 else 0.0, 4),
                "throughput_median_samples_per_sec": round(statistics.median(throughputs), 4),
                "peak_gpu_memory_median_mb": round(statistics.median(peak_memories), 2),
                "peak_gpu_memory_std_mb": round(statistics.pstdev(peak_memories) if len(peak_memories) > 1 else 0.0, 2),
            }
        )
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=list(MODEL_MAP.keys()))
    parser.add_argument("--warmup_runs", type=int, default=WARMUP_RUNS)
    parser.add_argument("--measure_runs", type=int, default=MEASURE_RUNS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows: List[Dict[str, object]] = []
    for model_name in args.models:
        if model_name not in MODEL_MAP:
            raise SystemExit(f"Unknown model: {model_name}")
        print(f"Benchmarking inference: {model_name}")
        all_rows.extend(
            benchmark_model(
                model_name,
                warmup_runs=args.warmup_runs,
                measure_runs=args.measure_runs,
                repeats=args.repeats,
                batch_size=args.batch_size,
            )
        )

    summary_rows = summarize(all_rows)
    write_csv(RAW_CSV, all_rows)
    write_csv(SUMMARY_CSV, summary_rows)

    print(f"Wrote: {RAW_CSV}")
    print(f"Wrote: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()