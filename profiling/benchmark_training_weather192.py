from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent
RUN_PY = ROOT / "run.py"
OUT_DIR = ROOT / "results_analysis"
LOG_DIR = OUT_DIR / "efficiency_weather192_training_logs"
OUT_CSV = OUT_DIR / "efficiency_weather192_training_raw.csv"

GPU_MEMORY_QUERY = [
    "nvidia-smi",
    "--query-gpu=memory.used",
    "--format=csv,noheader,nounits",
]
GPU_NAME_QUERY = [
    "nvidia-smi",
    "--query-gpu=name",
    "--format=csv,noheader",
]

MODELS = ["DLinear", "PatchTST", "PatchTST_capacity", "PatchFusionBERT_v0", "PatchFusionBERT_v2"]
SEEDS = [2021, 2022, 2023]

EPOCH_RE = re.compile(r"Epoch:\s*(\d+)\s*cost time:\s*([0-9.]+)")


def get_learning_rate(model: str) -> str:
    if model == "DLinear":
        return "0.01"
    return "0.0001"


def query_gpu_name() -> str:
    try:
        completed = subprocess.run(GPU_NAME_QUERY, capture_output=True, text=True, check=True)
        first = completed.stdout.strip().splitlines()[0].strip()
        return first or "unknown"
    except Exception:
        return "unknown"


def query_gpu_memory_mb() -> Optional[float]:
    try:
        completed = subprocess.run(GPU_MEMORY_QUERY, capture_output=True, text=True, check=True)
        first = completed.stdout.strip().splitlines()[0].strip()
        return float(first)
    except Exception:
        return None


def build_command(model: str, seed: int) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        str(RUN_PY),
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", f"EfficiencyWeather192_MS{seed}_{model}",
        "--model", model,
        "--data", "custom",
        "--root_path", "./data/",
        "--data_path", "weather.csv",
        "--features", "M",
        "--seq_len", "336",
        "--label_len", "96",
        "--pred_len", "192",
        "--enc_in", "21",
        "--dec_in", "21",
        "--c_out", "21",
        "--patch_len", "16",
        "--stride", "8",
        "--d_model", "192" if model == "PatchTST_capacity" else "128",
        "--n_heads", "8",
        "--e_layers", "5" if model == "PatchTST_capacity" else "3",
        "--d_layers", "1",
        "--d_ff", "512",
        "--factor", "3",
        "--dropout", "0.1",
        "--itr", "1",
        "--train_epochs", "100",
        "--patience", "10",
        "--batch_size", "8",
        "--learning_rate", get_learning_rate(model),
        "--num_workers", "0",
        "--seed", str(seed),
        "--des", "efficiency_weather192",
    ]
    return cmd


def parse_epochs(log_text: str) -> int:
    matches = list(EPOCH_RE.finditer(log_text))
    return len(matches)


def load_existing_rows() -> List[Dict[str, str]]:
    if not OUT_CSV.exists():
        return []

    with OUT_CSV.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_pairs(rows: Iterable[Dict[str, str]]) -> set[tuple[str, int]]:
    done = set()
    for row in rows:
        if row.get("status") == "ok":
            done.add((row["model"], int(row["seed"])))
    return done


def write_rows(rows: List[Dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "seed",
        "gpu_name",
        "training_seconds",
        "peak_gpu_memory_mb",
        "epochs_ran",
        "return_code",
        "status",
        "log_file",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_single(model: str, seed: int, poll_interval: float) -> Dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{model}_MS{seed}_Weather_192.log"
    cmd = build_command(model, seed)

    env = os.environ.copy()
    env["USE_TF"] = "0"
    env["TRANSFORMERS_NO_TF"] = "1"
    env["TRANSFORMERS_NO_FLAX"] = "1"
    env["TRANSFORMERS_NO_JAX"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    start = time.perf_counter()
    peak_gpu_memory_mb = 0.0
    gpu_seen = False
    output_lines: List[str] = []

    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        assert process.stdout is not None
        while True:
            line = process.stdout.readline()
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
                output_lines.append(line)
                memory_mb = query_gpu_memory_mb()
                if memory_mb is not None:
                    gpu_seen = True
                    peak_gpu_memory_mb = max(peak_gpu_memory_mb, memory_mb)
                continue

            if process.poll() is not None:
                break

            memory_mb = query_gpu_memory_mb()
            if memory_mb is not None:
                gpu_seen = True
                peak_gpu_memory_mb = max(peak_gpu_memory_mb, memory_mb)
            time.sleep(poll_interval)

        remainder = process.stdout.read()
        if remainder:
            sys.stdout.write(remainder)
            sys.stdout.flush()
            output_lines.append(remainder)
    finally:
        return_code = process.wait()

    training_seconds = time.perf_counter() - start
    log_text = "".join(output_lines)
    log_path.write_text(log_text, encoding="utf-8", errors="ignore")

    return {
        "model": model,
        "seed": seed,
        "gpu_name": query_gpu_name(),
        "training_seconds": round(training_seconds, 4),
        "peak_gpu_memory_mb": round(peak_gpu_memory_mb, 2) if gpu_seen else "",
        "epochs_ran": parse_epochs(log_text),
        "return_code": return_code,
        "status": "ok" if return_code == 0 else "failed",
        "log_file": str(log_path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--poll_interval", type=float, default=0.5)
    args = parser.parse_args()

    rows = load_existing_rows()
    done = completed_pairs(rows)
    output_rows: List[Dict[str, object]] = [dict(row) for row in rows]

    for model in args.models:
        for seed in args.seeds:
            if args.skip_completed and (model, seed) in done:
                print(f"Skipping completed run: {model} seed={seed}")
                continue

            print("=" * 80)
            print(f"Training benchmark: {model} | seed={seed}")
            print("=" * 80)
            row = run_single(model, seed, poll_interval=args.poll_interval)

            output_rows = [
                existing
                for existing in output_rows
                if not (existing["model"] == model and int(existing["seed"]) == seed)
            ]
            output_rows.append(row)
            write_rows(output_rows)

            if row["status"] != "ok":
                raise SystemExit(f"Training benchmark failed for {model} seed={seed}")


if __name__ == "__main__":
    main()