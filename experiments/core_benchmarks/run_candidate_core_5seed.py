#!/usr/bin/env python
"""Resume-safe PatchTST-family benchmarks for additional multivariate datasets."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


CAMPAIGN = "CandidateCoreFiveSeed"
DESCRIPTION = "candidate_core"

DATASETS = {
    "Solar": {
        "data_path": "Solar.csv",
        "enc_in": 137,
        "rows": 52560,
        "freq": "10min",
        "target": "channel_99",
        "batch_size": 8,
        "origin": "non-graph energy",
    },
    "AQShunyi": {
        "data_path": "AQShunyi.csv",
        "enc_in": 11,
        "rows": 35064,
        "freq": "h",
        "target": "WSPM",
        "batch_size": 16,
        "origin": "non-graph air quality",
    },
    "METR-LA": {
        "data_path": "METR-LA.csv",
        "enc_in": 207,
        "rows": 34272,
        "freq": "5min",
        "target": "ch_774204",
        "batch_size": 8,
        "origin": "spatial traffic represented as a wide non-graph table",
    },
    "PEMS04": {
        "data_path": "PEMS04.csv",
        "enc_in": 307,
        "rows": 16992,
        "freq": "5min",
        "target": "channel_99",
        "batch_size": 8,
        "origin": "spatial traffic represented as a wide non-graph table",
    },
    "PEMS08": {
        "data_path": "PEMS08.csv",
        "enc_in": 170,
        "rows": 17856,
        "freq": "5min",
        "target": "channel_99",
        "batch_size": 8,
        "origin": "spatial traffic represented as a wide non-graph table",
    },
    "PEMS-BAY": {
        "data_path": "PEMS-BAY.csv",
        "enc_in": 325,
        "rows": 52116,
        "freq": "5min",
        "target": "channel_414695",
        "batch_size": 8,
        "origin": "spatial traffic represented as a wide non-graph table",
    },
    "CzeLan": {
        "data_path": "CzeLan.csv",
        "enc_in": 11,
        "rows": 19934,
        "freq": "30min",
        "target": "ws",
        "batch_size": 16,
        "origin": "non-graph ecology",
    },
    "Wind": {
        "data_path": "Wind.csv",
        "enc_in": 7,
        "rows": 48673,
        "freq": "15min",
        "target": "ture_w_speed",
        "batch_size": 16,
        "origin": "non-graph energy and meteorology",
    },
}

MODELS = ("PatchTST", "PFB-Direct", "PFB-Projected")


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    model: str
    horizon: int
    seed: int

    @property
    def config(self) -> dict[str, object]:
        return DATASETS[self.dataset]

    @property
    def model_id(self) -> str:
        return f"{CAMPAIGN}_{self.model}_{self.dataset}_H{self.horizon}_seed{self.seed}"

    @property
    def factor(self) -> int:
        return 1 if self.model == "PFB-Projected" else 3


def parse_strings(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def python_has_dependencies(python_exe: str) -> bool:
    try:
        proc = subprocess.run(
            [python_exe, "-c", "import torch, numpy, pandas"],
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
    checked: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        checked.append(candidate)
        if python_has_dependencies(candidate):
            return candidate
    raise SystemExit(
        "No Python interpreter with torch, numpy, and pandas was found. Checked:\n  - "
        + "\n  - ".join(checked)
    )


def validate_dataset(path: Path, dataset: str, full_scan: bool = False) -> dict[str, object]:
    cfg = DATASETS[dataset]
    if not path.exists():
        raise SystemExit(f"Missing dataset file: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SystemExit(f"Empty dataset file: {path}") from exc

        expected_columns = int(cfg["enc_in"]) + 1
        if len(header) != expected_columns:
            raise SystemExit(
                f"{dataset}: expected {expected_columns} columns, found {len(header)}"
            )
        if header[0] != "date":
            raise SystemExit(f"{dataset}: first column must be 'date', found {header[0]!r}")
        if str(cfg["target"]) not in header:
            raise SystemExit(f"{dataset}: target {cfg['target']!r} is absent")
        if len(header) != len(set(header)):
            raise SystemExit(f"{dataset}: duplicate column names detected")

        row_count = 0
        first_date = ""
        last_date = ""
        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            if not first_date:
                first_date = row[0] if row else ""
            last_date = row[0] if row else ""
            if len(row) != expected_columns:
                raise SystemExit(
                    f"{dataset}: row {line_number} has {len(row)} columns; "
                    f"expected {expected_columns}"
                )
            if not row[0]:
                raise SystemExit(f"{dataset}: empty date at row {line_number}")
            if full_scan:
                for column_number, value in enumerate(row[1:], start=2):
                    if value == "":
                        raise SystemExit(
                            f"{dataset}: empty value at row {line_number}, column {column_number}"
                        )
                    try:
                        number = float(value)
                    except ValueError as exc:
                        raise SystemExit(
                            f"{dataset}: nonnumeric value at row {line_number}, "
                            f"column {column_number}: {value!r}"
                        ) from exc
                    if not math.isfinite(number):
                        raise SystemExit(
                            f"{dataset}: nonfinite value at row {line_number}, column {column_number}"
                        )

    if row_count != int(cfg["rows"]):
        raise SystemExit(f"{dataset}: expected {cfg['rows']} rows, found {row_count}")
    return {
        "dataset": dataset,
        "rows": row_count,
        "variables": int(cfg["enc_in"]),
        "first_date": first_date,
        "last_date": last_date,
        "full_numeric_scan": full_scan,
    }


def make_plan(
    datasets: list[str], models: list[str], horizons: list[int], seeds: list[int]
) -> list[RunSpec]:
    return [
        RunSpec(dataset, model, horizon, seed)
        for dataset, horizon, model, seed in itertools.product(
            datasets, horizons, models, seeds
        )
    ]


def command_for(spec: RunSpec, python_exe: str, storage_root: Path) -> list[str]:
    cfg = spec.config
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
        spec.model,
        "--data",
        "custom",
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
        str(spec.horizon),
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
        str(cfg["batch_size"]),
        "--des",
        DESCRIPTION,
        "--seed",
        str(spec.seed),
        "--checkpoints",
        str(storage_root / "checkpoints"),
        "--artifacts_root",
        str(storage_root / "artifacts"),
        "--skip_prediction_arrays",
        "--defer_test_until_after_training",
    ]


def result_metrics_path(ts_root: Path, storage_root: Path, spec: RunSpec) -> Path | None:
    # Retain compatibility with results produced before external artifact storage
    # was introduced, then check the current storage root.
    roots = [ts_root / "results", storage_root / "artifacts" / "results"]
    for result_root in roots:
        if not result_root.exists():
            continue
        for candidate in result_root.glob(f"long_term_forecast_{spec.model_id}_*"):
            metrics = candidate / "metrics.npy"
            if metrics.exists():
                return metrics
    return None


def write_plan(path: Path, plan: list[RunSpec], python_exe: str, storage_root: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "campaign",
                "dataset",
                "data_path",
                "origin",
                "horizon",
                "model",
                "seed",
                "channels",
                "frequency",
                "batch_size",
                "model_id",
                "command",
            ],
        )
        writer.writeheader()
        for spec in plan:
            cfg = spec.config
            writer.writerow(
                {
                    "campaign": CAMPAIGN,
                    "dataset": spec.dataset,
                    "data_path": cfg["data_path"],
                    "origin": cfg["origin"],
                    "horizon": spec.horizon,
                    "model": spec.model,
                    "seed": spec.seed,
                    "channels": cfg["enc_in"],
                    "frequency": cfg["freq"],
                    "batch_size": cfg["batch_size"],
                    "model_id": spec.model_id,
                    "command": json.dumps(command_for(spec, python_exe, storage_root)),
                }
            )


def append_status(path: Path, row: dict[str, object]) -> None:
    fields = [
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
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run additional PatchTST-family multivariate benchmarks."
    )
    parser.add_argument("--datasets", default="Solar,AQShunyi")
    parser.add_argument("--models", default="PatchTST,PFB-Direct,PFB-Projected")
    parser.add_argument("--horizons", default="96,192,336")
    parser.add_argument("--seeds", default="2021,2022,2023,2024,2025")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="0 runs all remaining cells.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-data-only", action="store_true")
    parser.add_argument(
        "--full-data-scan",
        action="store_true",
        help="Check every numeric value in addition to headers, shapes, and row lengths.",
    )
    parser.add_argument("--python", default="")
    parser.add_argument(
        "--storage-root",
        default="",
        help="Checkpoint/artifact root. Defaults to LOCALAPPDATA/PFB-4CAST/candidate_benchmarks.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    ts_root = script_dir.parents[1]
    workspace_root = ts_root
    if not (ts_root / "run.py").exists():
        raise SystemExit(f"Cannot find Time-Series-Library/run.py at {ts_root}")

    datasets = parse_strings(args.datasets)
    models = parse_strings(args.models)
    horizons = parse_ints(args.horizons)
    seeds = parse_ints(args.seeds)

    unknown_datasets = [name for name in datasets if name not in DATASETS]
    if unknown_datasets:
        raise SystemExit(f"Unknown datasets: {unknown_datasets}. Valid: {sorted(DATASETS)}")
    unknown_models = [name for name in models if name not in MODELS]
    if unknown_models:
        raise SystemExit(f"Unknown models: {unknown_models}. Valid: {list(MODELS)}")
    if any(horizon <= 0 for horizon in horizons):
        raise SystemExit("Horizons must be positive integers.")
    if args.max_runs < 0:
        raise SystemExit("--max-runs must be zero or positive.")

    validation_rows = []
    for dataset in datasets:
        data_path = ts_root / "data" / str(DATASETS[dataset]["data_path"])
        validation_rows.append(
            validate_dataset(data_path, dataset, full_scan=args.full_data_scan)
        )

    print("Validated datasets:")
    for row in validation_rows:
        print(
            f"  {row['dataset']}: {row['rows']} timestamps, "
            f"{row['variables']} variables, {row['first_date']} -> {row['last_date']}"
        )
    if args.validate_data_only:
        return 0

    python_exe = resolve_python(workspace_root, args.python or None)
    local_appdata = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    storage_root = (
        Path(args.storage_root).expanduser().resolve()
        if args.storage_root
        else local_appdata / "PFB-4CAST" / "candidate_benchmarks"
    )
    (storage_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (storage_root / "artifacts").mkdir(parents=True, exist_ok=True)
    plan = make_plan(datasets, models, horizons, seeds)
    logs_dir = script_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plan_path = script_dir / "candidate_core_plan.csv"
    status_path = script_dir / "candidate_core_status.csv"
    manifest_path = script_dir / "candidate_core_manifest.json"
    write_plan(plan_path, plan, python_exe, storage_root)
    manifest_path.write_text(
        json.dumps(
            {
                "campaign": CAMPAIGN,
                "datasets": datasets,
                "dataset_configurations": {name: DATASETS[name] for name in datasets},
                "models": models,
                "horizons": horizons,
                "seeds": seeds,
                "runs_planned": len(plan),
                "skip_completed": args.skip_completed,
                "max_runs": args.max_runs,
                "dry_run": args.dry_run,
                "python_exe": python_exe,
                "storage_root": str(storage_root),
                "plan_csv": str(plan_path),
                "status_csv": str(status_path),
                "logs_dir": str(logs_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("ADDITIONAL MULTIVARIATE CORE BENCHMARKS")
    print("=" * 78)
    print(f"Plan written: {plan_path}")
    print(f"Runs planned: {len(plan)}")
    print(f"Skip completed: {args.skip_completed}")
    print(f"Max new runs: {'ALL' if args.max_runs == 0 else args.max_runs}")
    print(f"Dry run: {args.dry_run}")
    print(f"Storage root: {storage_root}")
    print("=" * 78)

    executed = 0
    skipped = 0
    for index, spec in enumerate(plan, start=1):
        existing = result_metrics_path(ts_root, storage_root, spec)
        if args.skip_completed and existing is not None:
            skipped += 1
            print(
                f"[{index}/{len(plan)}] SKIP {spec.model} {spec.dataset} "
                f"H={spec.horizon} seed={spec.seed}"
            )
            append_status(
                status_path,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "campaign": CAMPAIGN,
                    "dataset": spec.dataset,
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

        command = command_for(spec, python_exe, storage_root)
        stdout_log = logs_dir / f"{spec.model_id}.stdout.log"
        stderr_log = logs_dir / f"{spec.model_id}.stderr.log"
        if args.dry_run:
            print(
                f"[{index}/{len(plan)}] DRY {spec.model} {spec.dataset} "
                f"H={spec.horizon} seed={spec.seed}"
            )
            print(" ".join(command))
            executed += 1
            continue

        print(
            f"[{index}/{len(plan)}] RUN {spec.model} {spec.dataset} "
            f"H={spec.horizon} seed={spec.seed}"
        )
        start = time.time()
        with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open(
            "w", encoding="utf-8"
        ) as err:
            proc = subprocess.run(
                command, cwd=ts_root, stdout=out, stderr=err, text=True
            )
        elapsed = time.time() - start
        metrics = result_metrics_path(ts_root, storage_root, spec)
        status = "completed" if proc.returncode == 0 and metrics is not None else "failed"
        append_status(
            status_path,
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "campaign": CAMPAIGN,
                "dataset": spec.dataset,
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
    print(f"Done. New/dry runs: {executed}; skipped completed: {skipped}")
    print(f"Status CSV: {status_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
