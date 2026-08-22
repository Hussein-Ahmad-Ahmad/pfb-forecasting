#!/usr/bin/env python
"""Run the core PatchTST-family benchmark on high-dimensional datasets.

Supported datasets:
    electricity: data/electricity.csv, 321 channels, hourly
    traffic:     data/traffic.csv,     862 channels, hourly

Default grid:
    models   = PatchTST, PatchFusionBERT_v0, PatchFusionBERT_v2
    horizons = 96, 192, 336, 720
    seeds    = 2021, 2022, 2023, 2024, 2025

Outputs are written to results_analysis/paper_runs/<dataset>/.
The script is resumable with --skip-completed.
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


DATASETS = {
    "electricity": {
        "label": "Electricity",
        "data_path": "electricity.csv",
        "channels": 321,
        "batch_size": 8,
        "description": "electricity_core",
    },
    "traffic": {
        "label": "Traffic",
        "data_path": "traffic.csv",
        "channels": 862,
        "batch_size": 4,
        "description": "traffic_core",
    },
}


@dataclass(frozen=True)
class RunSpec:
    campaign: str
    dataset_label: str
    data_path: str
    channels: int
    description: str
    model: str
    horizon: int
    seed: int
    batch_size: int

    @property
    def model_id(self) -> str:
        return f"{self.campaign}_{self.model}_{self.dataset_label}_H{self.horizon}_seed{self.seed}"

    @property
    def factor(self) -> int:
        return 1 if self.model == "PatchFusionBERT_v2" else 3


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_strings(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def make_plan(
    campaign: str,
    dataset_cfg: dict[str, object],
    models: Iterable[str],
    horizons: Iterable[int],
    seeds: Iterable[int],
    batch_size: int,
) -> list[RunSpec]:
    return [
        RunSpec(
            campaign=campaign,
            dataset_label=str(dataset_cfg["label"]),
            data_path=str(dataset_cfg["data_path"]),
            channels=int(dataset_cfg["channels"]),
            description=str(dataset_cfg["description"]),
            model=model,
            horizon=horizon,
            seed=seed,
            batch_size=batch_size,
        )
        for horizon in horizons
        for model in models
        for seed in seeds
    ]


def python_has_required_packages(python_exe: str) -> bool:
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


def resolve_python(repo_root: Path, requested_python: str | None) -> str:
    candidates: list[str] = []
    if requested_python:
        candidates.append(requested_python)
    candidates.extend(
        [
            str(repo_root.parent / ".venv" / "Scripts" / "python.exe"),
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
        if python_has_required_packages(candidate):
            return candidate

    raise SystemExit(
        "No Python interpreter with torch and numpy was found.\n"
        "Checked:\n  - "
        + "\n  - ".join(checked)
        + "\nPass the intended environment with --python C:\\path\\to\\python.exe"
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
        spec.data_path,
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
        str(spec.channels),
        "--dec_in",
        str(spec.channels),
        "--c_out",
        str(spec.channels),
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
        spec.description,
        "--seed",
        str(spec.seed),
    ]


def result_metrics_path(repo_root: Path, spec: RunSpec) -> Path | None:
    result_root = repo_root / "results"
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
                    "campaign": spec.campaign,
                    "dataset": spec.dataset_label,
                    "data_path": spec.data_path,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="Maximum new runs to execute. 0 means all.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-data-check", action="store_true")
    parser.add_argument("--python", default="", help="Python executable used to call run.py.")
    parser.add_argument("--models", default="PatchTST,PatchFusionBERT_v0,PatchFusionBERT_v2")
    parser.add_argument("--horizons", default="96,192,336,720")
    parser.add_argument("--seeds", default="2021,2022,2023,2024,2025")
    parser.add_argument("--batch-size", type=int, default=0, help="Override default dataset batch size.")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    dataset_cfg = DATASETS[args.dataset]
    data_file = repo_root / "data" / str(dataset_cfg["data_path"])
    if not (repo_root / "run.py").exists():
        raise SystemExit(f"Cannot find run.py under {repo_root}")
    if not args.no_data_check and not data_file.exists():
        raise SystemExit(f"Missing dataset file: {data_file}")

    python_exe = resolve_python(repo_root, args.python or None)
    batch_size = args.batch_size if args.batch_size > 0 else int(dataset_cfg["batch_size"])
    campaign = f"{str(dataset_cfg['label'])}Core5Seed"
    plan = make_plan(
        campaign=campaign,
        dataset_cfg=dataset_cfg,
        models=parse_strings(args.models),
        horizons=parse_ints(args.horizons),
        seeds=parse_ints(args.seeds),
        batch_size=batch_size,
    )

    out_dir = repo_root / "results_analysis" / "paper_runs" / args.dataset
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "core_5seed_plan.csv"
    status_path = out_dir / "core_5seed_status.csv"
    manifest_path = out_dir / "core_5seed_manifest.json"
    write_plan_csv(plan_path, plan, python_exe)
    manifest_path.write_text(
        json.dumps(
            {
                "campaign": campaign,
                "dataset": dataset_cfg["label"],
                "data_path": str(data_file),
                "models": parse_strings(args.models),
                "horizons": parse_ints(args.horizons),
                "seeds": parse_ints(args.seeds),
                "batch_size": batch_size,
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
    print(f"{str(dataset_cfg['label']).upper()} CORE 5-SEED CAMPAIGN")
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
        existing_metrics = result_metrics_path(repo_root, spec)
        if args.skip_completed and existing_metrics is not None:
            skipped += 1
            print(f"[{idx}/{len(plan)}] SKIP {spec.model} {spec.dataset_label} H={spec.horizon} seed={spec.seed}")
            append_status(
                status_path,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "campaign": spec.campaign,
                    "dataset": spec.dataset_label,
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
            print(f"[{idx}/{len(plan)}] DRY {spec.model} {spec.dataset_label} H={spec.horizon} seed={spec.seed}")
            print(" ".join(cmd))
            executed += 1
            continue

        print(f"[{idx}/{len(plan)}] RUN {spec.model} {spec.dataset_label} H={spec.horizon} seed={spec.seed}")
        start = time.time()
        with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
            proc = subprocess.run(cmd, cwd=repo_root, stdout=out, stderr=err, text=True)
        elapsed = time.time() - start
        metrics_after = result_metrics_path(repo_root, spec)
        status = "completed" if proc.returncode == 0 and metrics_after is not None else "not_completed"
        append_status(
            status_path,
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "campaign": spec.campaign,
                "dataset": spec.dataset_label,
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
            print(f"Run did not complete. See {stderr_log}")
            return proc.returncode or 1

    print("=" * 78)
    label = "Dry-run commands emitted" if args.dry_run else "New runs executed"
    print(f"{label}: {executed}; skipped completed: {skipped}")
    print(f"Status CSV: {status_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
