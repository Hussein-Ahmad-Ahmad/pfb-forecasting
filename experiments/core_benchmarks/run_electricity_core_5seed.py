#!/usr/bin/env python
"""
Electricity / ECL core PatchTST-family benchmark runner.

This script prepares and executes a resumable 5-seed Electricity campaign:
    3 models x 4 horizons x 5 seeds = 60 runs

Models:
    PatchTST
    PFB-Direct
    PFB-Projected

Dataset:
    Time-Series-Library/data/electricity.csv

The script writes plans, logs, and status files beside this runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CAMPAIGN = "ElectricityCoreFiveSeed"
DATASET = "Electricity"
DATA_PATH = "electricity.csv"
DESCRIPTION = "electricity_core"


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
        # PFB-Projected uses fc1 in the generated result directory, while
        # PatchTST and PFB-Direct use fc3.
        return 1 if self.model == "PFB-Projected" else 3


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> list[str]:
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
    candidates: list[str] = []
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

    seen: set[str] = set()
    unique_candidates: list[str] = []
    for candidate in candidates:
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    checked: list[str] = []
    for candidate in unique_candidates:
        checked.append(candidate)
        if python_has_torch_numpy(candidate):
            return candidate

    raise SystemExit(
        "No Python interpreter with both torch and numpy was found.\n"
        "Checked:\n  - "
        + "\n  - ".join(checked)
        + "\nInstall/activate the environment that contains torch, or pass it explicitly, e.g.:\n"
        "  python .\\experiments\\core_benchmarks\\run_electricity_core_5seed.py "
        "--python C:\\path\\to\\python.exe --skip-completed"
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
        "321",
        "--dec_in",
        "321",
        "--c_out",
        "321",
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
    path.parent.mkdir(parents=True, exist_ok=True)
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
    path.parent.mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser(description="Run Electricity core 5-seed benchmark.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip runs with existing metrics.npy.")
    parser.add_argument("--max-runs", type=int, default=0, help="Maximum number of new runs to execute. 0 means all.")
    parser.add_argument("--dry-run", action="store_true", help="Write the plan and print commands without training.")
    parser.add_argument("--no-data-check", action="store_true", help="Do not fail if data/electricity.csv is absent.")
    parser.add_argument("--python", default="", help="Python executable used to call Time-Series-Library/run.py.")
    parser.add_argument("--models", default="PatchTST,PFB-Direct,PFB-Projected")
    parser.add_argument("--horizons", default="96,192,336,720")
    parser.add_argument("--seeds", default="2021,2022,2023,2024,2025")
    parser.add_argument("--batch-size", type=int, default=8)
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

    models = parse_csv_strings(args.models)
    horizons = parse_csv_ints(args.horizons)
    seeds = parse_csv_ints(args.seeds)
    plan = make_plan(models=models, horizons=horizons, seeds=seeds, batch_size=args.batch_size)

    plan_path = script_dir / "electricity_core_5seed_plan.csv"
    status_path = script_dir / "electricity_core_5seed_status.csv"
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_plan_csv(plan_path, plan, python_exe)

    manifest = {
        "campaign": CAMPAIGN,
        "dataset": DATASET,
        "data_path": str(data_file),
        "n_runs_planned": len(plan),
        "models": models,
        "horizons": horizons,
        "seeds": seeds,
        "batch_size": args.batch_size,
        "skip_completed": args.skip_completed,
        "max_runs": args.max_runs,
        "dry_run": args.dry_run,
        "python_exe": python_exe,
        "plan_csv": str(plan_path),
        "status_csv": str(status_path),
        "logs_dir": str(logs_dir),
    }
    (script_dir / "electricity_core_5seed_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("=" * 78)
    print("ELECTRICITY CORE FIVE-SEED CAMPAIGN")
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
        existing_metrics = result_metrics_path(ts_root, spec)
        if args.skip_completed and existing_metrics is not None:
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
                    "metrics_path": str(existing_metrics),
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
        metrics_after = result_metrics_path(ts_root, spec)
        status = "completed" if proc.returncode == 0 and metrics_after is not None else "failed"
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
                "metrics_path": "" if metrics_after is None else str(metrics_after),
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
