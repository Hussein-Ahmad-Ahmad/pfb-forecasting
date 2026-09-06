#!/usr/bin/env python
"""Create, validate, and optionally execute the 60-cell baseline plan.

Official upstream implementations are used where available. When no public
official code exists, execution is allowed only through an explicitly labelled
paper-aligned reproduction with source and implementation hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path


MODELS = ("Gateformer", "EntroPE", "TimeSqueeze", "CT-PatchTST")
DATASETS = {
    "ETTh2": {"data_path": "ETTh2.csv", "channels": 7, "frequency": "h"},
    "ETTm2": {"data_path": "ETTm2.csv", "channels": 7, "frequency": "15min"},
    "Weather": {"data_path": "weather.csv", "channels": 21, "frequency": "h"},
}
SEEDS = (2021, 2022, 2023, 2024, 2025)
LOOKBACKS = {
    "Gateformer": 96,
    "EntroPE": 96,
    "TimeSqueeze": 96,
    "CT-PatchTST": 336,
}
SOURCE_STATUS = {
    "Gateformer": {
        "official_repository": "https://github.com/nyuolab/Gateformer",
        "availability": "official public implementation identified",
        "implementation": "official upstream implementation",
        "paper": "",
    },
    "EntroPE": {
        "official_repository": "https://github.com/Sachithx/EntroPE",
        "availability": "official public implementation identified",
        "implementation": "official upstream implementation",
        "paper": "",
    },
    "TimeSqueeze": {
        "official_repository": "",
        "availability": (
            "paper-aligned reproduction available; no public official "
            "implementation verified on 2026-08-29"
        ),
        "implementation": "independent paper-aligned reproduction",
        "paper": "https://arxiv.org/abs/2603.11352",
    },
    "CT-PatchTST": {
        "official_repository": "",
        "availability": (
            "paper-aligned reproduction available; no public official "
            "implementation verified on 2026-08-29"
        ),
        "implementation": "independent paper-aligned reproduction",
        "paper": "https://arxiv.org/abs/2501.08620",
    },
}
REQUIRED_PROTOCOL_VERSION = {
    "EntroPE": "entrope_lr003_five_seed_v1",
}


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
    for candidate in dict.fromkeys(value for value in candidates if value):
        checked.append(candidate)
        if python_has_dependencies(candidate):
            return str(Path(candidate).resolve()) if Path(candidate).exists() else candidate
    raise SystemExit(
        "No Python interpreter with torch, numpy, and pandas was found. Checked: "
        + ", ".join(checked)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_path(script_dir: Path, model: str) -> Path:
    names = {
        "Gateformer": "run_gateformer_adapter.py",
        "EntroPE": "run_entrope_adapter.py",
        "TimeSqueeze": "run_timesqueeze_adapter.py",
        "CT-PatchTST": "run_ct_patchtst_adapter.py",
    }
    return script_dir / names[model]


def result_path(script_dir: Path, model: str, dataset: str, seed: int) -> Path:
    return (
        script_dir
        / "results"
        / model
        / dataset
        / "H192"
        / f"seed{seed}"
        / "metrics.json"
    )


def metrics_are_complete(
    path: Path, model: str, dataset: str, seed: int
) -> bool:
    """A result is complete only when its identity and metrics are valid."""
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model") != model:
            return False
        if payload.get("dataset") != dataset:
            return False
        if int(payload.get("horizon")) != 192 or int(payload.get("seed")) != seed:
            return False
        mse = float(payload["mse"])
        mae = float(payload["mae"])
        if not (math.isfinite(mse) and math.isfinite(mae)):
            return False
        required_version = REQUIRED_PROTOCOL_VERSION.get(model)
        if required_version and payload.get("protocol_version") != required_version:
            return False
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def adapter_command(
    python_exe: str,
    adapter: Path,
    model: str,
    dataset: str,
    seed: int,
    data_root: Path,
    output: Path,
) -> list[str]:
    cfg = DATASETS[dataset]
    return [
        python_exe,
        str(adapter),
        "--model",
        model,
        "--dataset",
        dataset,
        "--data-root",
        str(data_root),
        "--data-path",
        str(cfg["data_path"]),
        "--seq-len",
        str(LOOKBACKS[model]),
        "--pred-len",
        "192",
        "--seed",
        str(seed),
        "--output",
        str(output),
    ]


def write_plan(
    path: Path,
    script_dir: Path,
    data_root: Path,
    python_exe: str,
    models: list[str],
    datasets: list[str],
    seeds: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, dataset, seed in itertools.product(models, datasets, seeds):
        adapter = adapter_path(script_dir, model)
        output = result_path(script_dir, model, dataset, seed)
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "horizon": 192,
                "lookback": LOOKBACKS[model],
                "seed": seed,
                "data_path": DATASETS[dataset]["data_path"],
                "channels": DATASETS[dataset]["channels"],
                "adapter": str(adapter),
                "adapter_present": adapter.exists(),
                "adapter_sha256": sha256(adapter) if adapter.exists() else "",
                "official_repository": SOURCE_STATUS[model]["official_repository"],
                "source_status": SOURCE_STATUS[model]["availability"],
                "implementation": SOURCE_STATUS[model]["implementation"],
                "paper": SOURCE_STATUS[model]["paper"],
                "metrics_output": str(output),
                "completed": metrics_are_complete(output, model, dataset, seed),
                "command": json.dumps(
                    adapter_command(
                        python_exe, adapter, model, dataset, seed, data_root, output
                    )
                ),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def validate_metrics(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"mse", "mae", "model", "dataset", "horizon", "seed"}
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"{path}: missing metrics fields {sorted(missing)}")
    mse = float(payload["mse"])
    mae = float(payload["mae"])
    if not (math.isfinite(mse) and math.isfinite(mae)):
        raise RuntimeError(f"{path}: MSE/MAE must be finite, received {mse}/{mae}")
    required_version = REQUIRED_PROTOCOL_VERSION.get(str(payload["model"]))
    if required_version and payload.get("protocol_version") != required_version:
        raise RuntimeError(
            f"{path}: expected protocol_version={required_version}, "
            f"received {payload.get('protocol_version')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan/run the 60 recent-baseline cells.")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--python",
        default="",
        help="Python executable with torch, numpy, and pandas; auto-detected by default",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    workspace = script_dir.parents[1]
    data_root = workspace / "data"
    python_exe = resolve_python(workspace, args.python)
    models = parse_strings(args.models)
    datasets = parse_strings(args.datasets)
    seeds = parse_ints(args.seeds)
    unknown_models = [value for value in models if value not in MODELS]
    unknown_datasets = [value for value in datasets if value not in DATASETS]
    if unknown_models or unknown_datasets:
        raise SystemExit(
            f"Unknown models={unknown_models} or datasets={unknown_datasets}"
        )
    if args.max_runs < 0:
        raise SystemExit("--max-runs must be zero or positive")
    for dataset in datasets:
        path = data_root / str(DATASETS[dataset]["data_path"])
        if not path.exists():
            raise SystemExit(f"Missing dataset: {path}")

    plan_path = script_dir / "recent_baselines_plan.csv"
    rows = write_plan(
        plan_path, script_dir, data_root, python_exe, models, datasets, seeds
    )
    missing_adapters = sorted({row["model"] for row in rows if not row["adapter_present"]})
    remaining = [row for row in rows if not (args.skip_completed and row["completed"])]

    print("=" * 78)
    print("RECENT BASELINE FIVE-SEED PLAN")
    print("=" * 78)
    print(f"Planned cells: {len(rows)}")
    print(f"Remaining cells: {len(remaining)}")
    print(f"Python: {python_exe}")
    print(f"Plan: {plan_path}")
    if missing_adapters:
        print("Unavailable execution adapters: " + ", ".join(missing_adapters))
    print("=" * 78)

    if not args.execute:
        print("Planning only. No model was trained.")
        return 0
    if missing_adapters:
        raise SystemExit(
            "Execution stopped before training because provenance-recorded adapters "
            "are missing."
        )

    executed = 0
    for index, row in enumerate(remaining, start=1):
        if args.max_runs and executed >= args.max_runs:
            break
        command = json.loads(str(row["command"]))
        output = Path(str(row["metrics_output"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[{index}/{len(remaining)}] {row['model']} {row['dataset']} "
            f"H=192 seed={row['seed']}"
        )
        result = subprocess.run(command, cwd=script_dir)
        if result.returncode != 0:
            return result.returncode
        if not output.exists():
            raise SystemExit(f"Adapter returned successfully but did not create {output}")
        validate_metrics(output)
        executed += 1
    print(f"Completed new runs: {executed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
