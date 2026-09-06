#!/usr/bin/env python
"""
Resume-safe DLinear matched-window-normalization campaign.

Default grid:
    6 datasets x 3 horizons x 5 seeds x 1 model = 90 runs

Model:
    DLinear_Norm

The script writes a plan, manifest, logs, and status CSV under this folder. It
skips completed runs by checking for metrics.npy in Time-Series-Library/results.
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


CAMPAIGN = "DLinearWindowNormFiveSeed"
MODEL = "DLinear_Norm"
DESCRIPTION = "dlinear_window_norm"

DATASETS = {
    "ETTm1": {"data": "ETTm1", "data_path": "ETTm1.csv", "enc_in": 7, "batch_size": 16},
    "ETTm2": {"data": "ETTm2", "data_path": "ETTm2.csv", "enc_in": 7, "batch_size": 16},
    "ETTh1": {"data": "ETTh1", "data_path": "ETTh1.csv", "enc_in": 7, "batch_size": 16},
    "ETTh2": {"data": "ETTh2", "data_path": "ETTh2.csv", "enc_in": 7, "batch_size": 16},
    "Exchange": {"data": "custom", "data_path": "exchange_rate.csv", "enc_in": 8, "batch_size": 16},
    "Weather": {"data": "custom", "data_path": "weather.csv", "enc_in": 21, "batch_size": 8},
}


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    horizon: int
    seed: int

    @property
    def cfg(self) -> dict[str, object]:
        return DATASETS[self.dataset]

    @property
    def model_id(self) -> str:
        return f"{CAMPAIGN}_{MODEL}_{self.dataset}_H{self.horizon}_seed{self.seed}"


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def make_plan(datasets: Iterable[str], horizons: Iterable[int], seeds: Iterable[int]) -> list[RunSpec]:
    return [RunSpec(dataset=d, horizon=h, seed=s) for d in datasets for h in horizons for s in seeds]


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
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        if python_has_torch_numpy(candidate):
            return candidate
    raise SystemExit("No Python interpreter with torch and numpy was found. Pass --python explicitly.")


def command_for(spec: RunSpec, python_exe: str) -> list[str]:
    cfg = spec.cfg
    enc_in = str(cfg["enc_in"])
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
        MODEL,
        "--data",
        str(cfg["data"]),
        "--root_path",
        "./data/",
        "--data_path",
        str(cfg["data_path"]),
        "--features",
        "M",
        "--seq_len",
        "336",
        "--label_len",
        "96",
        "--pred_len",
        str(spec.horizon),
        "--enc_in",
        enc_in,
        "--dec_in",
        enc_in,
        "--c_out",
        enc_in,
        "--e_layers",
        "2",
        "--d_layers",
        "1",
        "--factor",
        "3",
        "--d_model",
        "512",
        "--d_ff",
        "2048",
        "--n_heads",
        "8",
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
        "0.01",
        "--batch_size",
        str(cfg["batch_size"]),
        "--des",
        DESCRIPTION,
        "--seed",
        str(spec.seed),
    ]


def result_metrics_path(ts_root: Path, spec: RunSpec) -> Path | None:
    result_root = ts_root / "results"
    if not result_root.exists():
        return None
    for candidate in result_root.glob(f"long_term_forecast_{spec.model_id}_*"):
        metrics = candidate / "metrics.npy"
        if metrics.exists():
            return metrics
    return None


def write_plan_csv(path: Path, plan: list[RunSpec], python_exe: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["campaign", "dataset", "horizon", "model", "seed", "model_id", "command"],
        )
        writer.writeheader()
        for spec in plan:
            writer.writerow(
                {
                    "campaign": CAMPAIGN,
                    "dataset": spec.dataset,
                    "horizon": spec.horizon,
                    "model": MODEL,
                    "seed": spec.seed,
                    "model_id": spec.model_id,
                    "command": json.dumps(command_for(spec, python_exe)),
                }
            )


def append_status(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = [
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
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DLinear_Norm 5-seed matched-preprocessing grid.")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means all remaining runs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default="")
    parser.add_argument("--datasets", default="ETTm1,ETTm2,ETTh1,ETTh2,Exchange,Weather")
    parser.add_argument("--horizons", default="96,192,336")
    parser.add_argument("--seeds", default="2021,2022,2023,2024,2025")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    ts_root = script_dir.parents[1]
    workspace_root = ts_root
    if not (ts_root / "run.py").exists():
        raise SystemExit(f"Cannot find Time-Series-Library/run.py at: {ts_root}")

    datasets = parse_csv_strings(args.datasets)
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}. Valid: {sorted(DATASETS)}")
    for d in datasets:
        data_path = ts_root / "data" / str(DATASETS[d]["data_path"])
        if not data_path.exists():
            raise SystemExit(f"Missing data file: {data_path}")

    python_exe = resolve_python(workspace_root, args.python or None)
    horizons = parse_csv_ints(args.horizons)
    seeds = parse_csv_ints(args.seeds)
    plan = make_plan(datasets=datasets, horizons=horizons, seeds=seeds)

    plan_path = script_dir / "dlinear_norm_5seed_plan.csv"
    status_path = script_dir / "dlinear_norm_5seed_status.csv"
    manifest_path = script_dir / "dlinear_norm_5seed_manifest.json"
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_plan_csv(plan_path, plan, python_exe)
    manifest_path.write_text(
        json.dumps(
            {
                "campaign": CAMPAIGN,
                "model": MODEL,
                "datasets": datasets,
                "horizons": horizons,
                "seeds": seeds,
                "n_runs_planned": len(plan),
                "skip_completed": args.skip_completed,
                "max_runs": args.max_runs,
                "dry_run": args.dry_run,
                "python_exe": python_exe,
                "plan_csv": str(plan_path),
                "status_csv": str(status_path),
                "logs_dir": str(logs_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("DLINEAR MATCHED-NORMALIZATION 5-SEED CAMPAIGN")
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
            print(f"[{idx}/{len(plan)}] SKIP {MODEL} {spec.dataset} H={spec.horizon} seed={spec.seed}")
            append_status(
                status_path,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "campaign": CAMPAIGN,
                    "dataset": spec.dataset,
                    "horizon": spec.horizon,
                    "model": MODEL,
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
            print(f"[{idx}/{len(plan)}] DRY {MODEL} {spec.dataset} H={spec.horizon} seed={spec.seed}")
            print(" ".join(cmd))
            executed += 1
            continue

        print(f"[{idx}/{len(plan)}] RUN {MODEL} {spec.dataset} H={spec.horizon} seed={spec.seed}")
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
                "dataset": spec.dataset,
                "horizon": spec.horizon,
                "model": MODEL,
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
    print(f"Done. New/dry-run commands: {executed}; skipped completed: {skipped}")
    print(f"Status CSV: {status_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
