#!/usr/bin/env python
"""
Traffic core PatchTST-family benchmark runner.

Runs:
    3 models x 4 horizons x 5 seeds = 60 runs

Models:
    PatchTST
    PFB-Direct
    PFB-Projected

Dataset:
    Time-Series-Library/data/traffic.csv

This script writes plans, logs, and status files beside this runner and is safe
to resume with --skip-completed.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CAMPAIGN = "TrafficCoreFiveSeed"
DATASET = "Traffic"
DATA_PATH = "traffic.csv"
DESCRIPTION = "traffic_core"
CHANNELS = 862


@dataclass(frozen=True)
class RunSpec:
    model: str
    horizon: int
    seed: int
    batch_size: int

    @property
    def model_id(self) -> str:
        return f"{CAMPAIGN}_{self.model}_{DATASET}_H{self.horizon}_seed{self.seed}"

    @property
    def factor(self) -> int:
        return 1 if self.model == "PFB-Projected" else 3


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_strings(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def make_plan(models: Iterable[str], horizons: Iterable[int], seeds: Iterable[int], batch_size: int) -> list[RunSpec]:
    return [
        RunSpec(model=model, horizon=horizon, seed=seed, batch_size=batch_size)
        for horizon in horizons
        for model in models
        for seed in seeds
    ]


def python_has_torch_numpy(python_exe: str) -> bool:
    try:
        proc = subprocess.run(
            [python_exe, "-c", "import torch, numpy"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=20,
        )
        return proc.returncode == 0
    except Exception:
        return False


def resolve_python(workspace_root: Path, requested_python: str | None) -> str:
    candidates = []
    if requested_python:
        candidates.append(requested_python)
    candidates.extend(
        [
            str(workspace_root / ".venv" / "Scripts" / "python.exe"),
            sys.executable,
            r"C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe",
            "python.exe",
        ]
    )

    checked: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        checked.append(candidate)
        if python_has_torch_numpy(candidate):
            return candidate

    raise SystemExit(
        "No Python interpreter with torch and numpy was found.\n"
        "Checked:\n  - "
        + "\n  - ".join(checked)
        + "\nPass the correct environment with --python C:\\path\\to\\python.exe"
    )


def command_for(spec: RunSpec, python_exe: str) -> list[str]:
    return [
        python_exe,
        "-u",
        "run.py",
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model_id",
        spec.model_id,
        "--model",
        spec.model,
        "--data",
        "custom",
        "--root_path",
        "./data/",
        "--data_path",
        DATA_PATH,
        "--features",
        "M",
        "--freq",
        "h",
        "--seq_len",
        "336",
        "--label_len",
        "96",
        "--pred_len",
        str(spec.horizon),
        "--enc_in",
        str(CHANNELS),
        "--dec_in",
        str(CHANNELS),
        "--c_out",
        str(CHANNELS),
        "--patch_len",
        "16",
        "--stride",
        "8",
        "--d_model",
        "128",
        "--n_heads",
        "8",
        "--e_layers",
        "3",
        "--d_layers",
        "1",
        "--d_ff",
        "512",
        "--factor",
        str(spec.factor),
        "--dropout",
        "0.1",
        "--embed",
        "timeF",
        "--num_workers",
        "0",
        "--itr",
        "1",
        "--train_epochs",
        "100",
        "--patience",
        "10",
        "--learning_rate",
        "0.0001",
        "--batch_size",
        str(spec.batch_size),
        "--des",
        DESCRIPTION,
        "--seed",
        str(spec.seed),
    ]


def result_metrics_path(ts_root: Path, spec: RunSpec) -> Path | None:
    result_root = ts_root / "results"
    if not result_root.exists():
        return None

    # Prefer the standardized model identifier used by all newly launched runs.
    for candidate in result_root.glob(f"long_term_forecast_{spec.model_id}_*"):
        metrics = candidate / "metrics.npy"
        if metrics.exists():
            return metrics

    # Earlier campaign outputs can still be resumed without exposing their
    # previous labels in newly generated plans or status rows. The campaign used
    # fc3 for direct fusion and fc1 for projected fusion.
    if spec.model in {"PFB-Direct", "PFB-Projected"}:
        pattern = (
            f"long_term_forecast_{CAMPAIGN}_*_{DATASET}_H{spec.horizon}_"
            f"seed{spec.seed}_*"
        )
        expected_marker = "_fc3_" if spec.model == "PFB-Direct" else "_fc1_"
        for candidate in result_root.glob(pattern):
            if expected_marker not in candidate.name or "PatchTST" in candidate.name:
                continue
            metrics = candidate / "metrics.npy"
            if metrics.exists():
                return metrics
    return None


def write_plan_csv(path: Path, plan: list[RunSpec], python_exe: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "campaign",
                "dataset",
                "data_path",
                "horizon",
                "model",
                "seed",
                "batch_size",
                "model_id",
                "factor",
                "command",
            ],
        )
        writer.writeheader()
        for spec in plan:
            writer.writerow(
                {
                    "campaign": CAMPAIGN,
                    "dataset": DATASET,
                    "data_path": DATA_PATH,
                    "horizon": spec.horizon,
                    "model": spec.model,
                    "seed": spec.seed,
                    "batch_size": spec.batch_size,
                    "model_id": spec.model_id,
                    "factor": spec.factor,
                    "command": " ".join(command_for(spec, python_exe)),
                }
            )


def append_status(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "campaign",
                "dataset",
                "horizon",
                "model",
                "seed",
                "status",
                "returncode",
                "elapsed_seconds",
                "model_id",
                "metrics_path",
                "stdout_log",
                "stderr_log",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Traffic core 5-seed benchmark.")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means all.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-data-check", action="store_true")
    parser.add_argument("--python", default="")
    parser.add_argument("--models", default="PatchTST,PFB-Direct,PFB-Projected")
    parser.add_argument("--horizons", default="96,192,336,720")
    parser.add_argument("--seeds", default="2021,2022,2023,2024,2025")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    ts_root = script_dir.parents[1]
    workspace_root = ts_root
    if not (ts_root / "run.py").exists():
        raise SystemExit(f"Cannot find Time-Series-Library/run.py at: {ts_root}")
    data_file = ts_root / "data" / DATA_PATH
    if not args.no_data_check and not data_file.exists():
        raise SystemExit(f"Missing dataset file: {data_file}")

    python_exe = resolve_python(workspace_root, args.python or None)
    plan = make_plan(parse_strings(args.models), parse_ints(args.horizons), parse_ints(args.seeds), args.batch_size)

    logs_dir = script_dir / "logs_traffic"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plan_path = script_dir / "traffic_core_5seed_plan.csv"
    status_path = script_dir / "traffic_core_5seed_status.csv"
    write_plan_csv(plan_path, plan, python_exe)
    (script_dir / "traffic_core_5seed_manifest.json").write_text(
        json.dumps(
            {
                "campaign": CAMPAIGN,
                "dataset": DATASET,
                "data_path": str(data_file),
                "n_runs_planned": len(plan),
                "python_exe": python_exe,
                "batch_size": args.batch_size,
                "plan_csv": str(plan_path),
                "status_csv": str(status_path),
                "logs_dir": str(logs_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("TRAFFIC CORE FIVE-SEED CAMPAIGN")
    print("=" * 78)
    print(f"Plan written: {plan_path}")
    print(f"Runs planned: {len(plan)}")
    print(f"Skip completed: {args.skip_completed}")
    print(f"Max new runs: {'ALL' if args.max_runs == 0 else args.max_runs}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 78)

    executed = 0
    skipped = 0
    for idx, spec in enumerate(plan, start=1):
        existing = result_metrics_path(ts_root, spec)
        if args.skip_completed and existing is not None:
            skipped += 1
            print(f"[{idx}/{len(plan)}] SKIP {spec.model} {DATASET} H={spec.horizon} seed={spec.seed}")
            append_status(
                status_path,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "campaign": CAMPAIGN,
                    "dataset": DATASET,
                    "horizon": spec.horizon,
                    "model": spec.model,
                    "seed": spec.seed,
                    "status": "skipped_completed",
                    "returncode": 0,
                    "elapsed_seconds": 0,
                    "model_id": spec.model_id,
                    "metrics_path": str(existing),
                    "stdout_log": "",
                    "stderr_log": "",
                },
            )
            continue
        if args.max_runs and executed >= args.max_runs:
            print(f"Reached --max-runs={args.max_runs}. Stop.")
            break

        cmd = command_for(spec, python_exe)
        stdout_log = logs_dir / f"{spec.model_id}.stdout.log"
        stderr_log = logs_dir / f"{spec.model_id}.stderr.log"
        if args.dry_run:
            print(f"[{idx}/{len(plan)}] DRY {spec.model} {DATASET} H={spec.horizon} seed={spec.seed}")
            print(" ".join(cmd))
            executed += 1
            continue

        print(f"[{idx}/{len(plan)}] RUN {spec.model} {DATASET} H={spec.horizon} seed={spec.seed}")
        start = time.time()
        with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
            proc = subprocess.run(cmd, cwd=ts_root, stdout=out, stderr=err, text=True)
        elapsed = time.time() - start
        metrics = result_metrics_path(ts_root, spec)
        status = "completed" if proc.returncode == 0 and metrics is not None else "failed"
        append_status(
            status_path,
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "campaign": CAMPAIGN,
                "dataset": DATASET,
                "horizon": spec.horizon,
                "model": spec.model,
                "seed": spec.seed,
                "status": status,
                "returncode": proc.returncode,
                "elapsed_seconds": round(elapsed, 3),
                "model_id": spec.model_id,
                "metrics_path": "" if metrics is None else str(metrics),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
            },
        )
        executed += 1
        if status != "completed":
            print(f"FAILED: see {stderr_log}")
            return proc.returncode or 1

    print("=" * 78)
    if args.dry_run:
        print(f"Done. Dry-run commands emitted: {executed}; skipped completed: {skipped}")
    else:
        print(f"Done. New runs executed: {executed}; skipped completed: {skipped}")
    print(f"Status CSV: {status_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
