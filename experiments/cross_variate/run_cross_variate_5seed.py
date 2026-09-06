#!/usr/bin/env python
"""Run the 10-cell PFB cross-variate diagnostic campaign.

The campaign trains PFB-CrossVariate on Weather and Exchange at H=192 with
seeds 2021--2025.  PatchTST, PFB-Direct, and PFB-Projected controls are not
retrained because matching five-seed H=192 results already exist.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


CAMPAIGN = "PFBCrossVariate"
DESCRIPTION = "pfb_cross_variate"
MODEL = "PFB-CrossVariate"
DATASETS = {
    "Weather": {
        "data": "custom",
        "data_path": "weather.csv",
        "enc_in": 21,
        "batch_size": 8,
        "freq": "h",
        "target": "OT",
    },
    "Exchange": {
        "data": "custom",
        "data_path": "exchange_rate.csv",
        "enc_in": 8,
        "batch_size": 16,
        "freq": "d",
        "target": "OT",
    },
}


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    seed: int

    @property
    def model_id(self) -> str:
        return f"{CAMPAIGN}_{MODEL}_{self.dataset}_H192_seed{self.seed}"


def parse_strings(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def python_has_dependencies(executable: str) -> bool:
    try:
        result = subprocess.run(
            [executable, "-c", "import torch, numpy, pandas"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def resolve_python(workspace: Path, requested: str) -> str:
    candidates = [
        requested,
        str(workspace / ".venv" / "Scripts" / "python.exe"),
        sys.executable,
        r"C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe",
        "python.exe",
    ]
    checked: list[str] = []
    for candidate in dict.fromkeys(item for item in candidates if item):
        checked.append(candidate)
        if python_has_dependencies(candidate):
            return candidate
    raise SystemExit("No Python with torch/numpy/pandas found: " + ", ".join(checked))


def validate_dataset(ts_root: Path, name: str) -> None:
    cfg = DATASETS[name]
    path = ts_root / "data" / str(cfg["data_path"])
    if not path.exists():
        raise SystemExit(f"Missing dataset: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if not header or header[0] != "date":
        raise SystemExit(f"{path}: expected 'date' as the first column")
    if len(header) != int(cfg["enc_in"]) + 1:
        raise SystemExit(
            f"{path}: expected {int(cfg['enc_in']) + 1} columns, found {len(header)}"
        )
    if str(cfg["target"]) not in header:
        raise SystemExit(f"{path}: target {cfg['target']!r} is missing")


def command_for(spec: RunSpec, python_exe: str) -> list[str]:
    cfg = DATASETS[spec.dataset]
    channels = str(cfg["enc_in"])
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
        "--target",
        str(cfg["target"]),
        "--freq",
        str(cfg["freq"]),
        "--seq_len",
        "336",
        "--label_len",
        "96",
        "--pred_len",
        "192",
        "--enc_in",
        channels,
        "--dec_in",
        channels,
        "--c_out",
        channels,
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
        "1",
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
        str(cfg["batch_size"]),
        "--des",
        DESCRIPTION,
        "--seed",
        str(spec.seed),
        "--defer_test_until_after_training",
    ]


def metrics_path(ts_root: Path, spec: RunSpec) -> Path | None:
    result_root = ts_root / "results"
    if not result_root.exists():
        return None
    for directory in result_root.glob(f"long_term_forecast_{spec.model_id}_*"):
        candidate = directory / "metrics.npy"
        if candidate.exists():
            return candidate
    return None


def write_plan(path: Path, plan: list[RunSpec], python_exe: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "horizon", "model", "seed", "model_id", "command"],
        )
        writer.writeheader()
        for spec in plan:
            writer.writerow(
                {
                    "dataset": spec.dataset,
                    "horizon": 192,
                    "model": MODEL,
                    "seed": spec.seed,
                    "model_id": spec.model_id,
                    "command": json.dumps(command_for(spec, python_exe)),
                }
            )


def append_status(path: Path, row: dict[str, object]) -> None:
    fields = [
        "timestamp",
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
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 10 cross-variate PFB cells.")
    parser.add_argument("--datasets", default="Weather,Exchange")
    parser.add_argument("--seeds", default="2021,2022,2023,2024,2025")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="0 runs all remaining cells")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default="")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    ts_root = script_dir.parents[1]
    workspace = ts_root
    if not (ts_root / "run.py").exists():
        raise SystemExit(f"Missing Time-Series-Library/run.py: {ts_root}")

    datasets = parse_strings(args.datasets)
    seeds = parse_ints(args.seeds)
    unknown = [name for name in datasets if name not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown datasets {unknown}; valid values: {list(DATASETS)}")
    if args.max_runs < 0:
        raise SystemExit("--max-runs must be zero or positive")
    for dataset in datasets:
        validate_dataset(ts_root, dataset)

    python_exe = resolve_python(workspace, args.python)
    plan = [RunSpec(dataset, seed) for dataset, seed in itertools.product(datasets, seeds)]
    logs_dir = script_dir / "logs" / "cross_variate"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plan_path = script_dir / "cross_variate_plan.csv"
    status_path = script_dir / "cross_variate_status.csv"
    manifest_path = script_dir / "cross_variate_manifest.json"
    write_plan(plan_path, plan, python_exe)
    manifest_path.write_text(
        json.dumps(
            {
                "campaign": CAMPAIGN,
                "purpose": "post-fusion cross-variate diagnostic",
                "model": MODEL,
                "datasets": datasets,
                "horizon": 192,
                "seeds": seeds,
                "runs_planned": len(plan),
                "architecture": "PFB-Projected plus one gated cross-variate attention block",
                "control_results_reused": ["PatchTST", "PFB-Direct", "PFB-Projected"],
                "python": python_exe,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("PFB CROSS-VARIATE 5-SEED DIAGNOSTIC")
    print("=" * 78)
    print(f"Runs planned: {len(plan)}")
    print(f"Skip completed: {args.skip_completed}")
    print(f"Max new runs: {'ALL' if args.max_runs == 0 else args.max_runs}")
    print(f"Dry run: {args.dry_run}")
    print(f"Plan: {plan_path}")
    print("=" * 78)

    executed = 0
    skipped = 0
    for index, spec in enumerate(plan, start=1):
        existing = metrics_path(ts_root, spec)
        if args.skip_completed and existing is not None:
            skipped += 1
            print(f"[{index}/{len(plan)}] SKIP {MODEL} {spec.dataset} seed={spec.seed}")
            continue
        if args.max_runs and executed >= args.max_runs:
            print(f"Reached --max-runs={args.max_runs}.")
            break
        command = command_for(spec, python_exe)
        if args.dry_run:
            print(f"[{index}/{len(plan)}] DRY {MODEL} {spec.dataset} seed={spec.seed}")
            print(subprocess.list2cmdline(command))
            executed += 1
            continue

        stdout_log = logs_dir / f"{spec.model_id}.stdout.log"
        stderr_log = logs_dir / f"{spec.model_id}.stderr.log"
        print(f"[{index}/{len(plan)}] RUN {MODEL} {spec.dataset} seed={spec.seed}")
        started = time.time()
        with stdout_log.open("w", encoding="utf-8") as stdout, stderr_log.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.run(
                command,
                cwd=ts_root,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
        elapsed = time.time() - started
        result = metrics_path(ts_root, spec)
        status = "completed" if process.returncode == 0 and result else "failed"
        append_status(
            status_path,
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": spec.dataset,
                "horizon": 192,
                "model": MODEL,
                "seed": spec.seed,
                "status": status,
                "returncode": process.returncode,
                "elapsed_seconds": round(elapsed, 3),
                "model_id": spec.model_id,
                "metrics_path": "" if result is None else str(result),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
            },
        )
        executed += 1
        if status == "failed":
            print(f"FAILED: see {stderr_log}")
            return process.returncode or 1

    print("=" * 78)
    print(f"Finished. New/dry runs: {executed}; skipped: {skipped}")
    print(f"Status: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
