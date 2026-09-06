"""Five-seed cross-family comparison under common reversible window preprocessing."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

MODELS = ("DLinear", "PatchTST", "PFB-Direct", "PFB-Projected", "iTransformer", "TiDE", "TimeXer")
DATASETS = {
    "ETTh2": {"data": "ETTh2", "file": "ETTh2.csv", "freq": "h", "channels": 7, "batch": 16},
    "ETTm2": {"data": "ETTm2", "file": "ETTm2.csv", "freq": "t", "channels": 7, "batch": 16},
    "Weather": {"data": "custom", "file": "weather.csv", "freq": "h", "channels": 21, "batch": 8},
}


@dataclass(frozen=True)
class Cell:
    model: str
    dataset: str
    seed: int

    @property
    def model_id(self) -> str:
        return f"UniformPreprocessing_{self.model}_{self.dataset}_H192_seed{self.seed}"


def items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ints(value: str) -> list[int]:
    return [int(item) for item in items(value)]


def resolve_python(workspace: Path, requested: str) -> str:
    candidates = [requested] if requested else []
    candidates += [str(workspace / ".venv" / "Scripts" / "python.exe"), sys.executable, "python.exe"]
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "-c", "import torch,numpy,pandas"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20,
            )
            if result.returncode == 0:
                return candidate
        except (OSError, subprocess.SubprocessError):
            pass
    raise SystemExit("No Python interpreter with torch, numpy, and pandas was found")


def hyperparameters(model: str) -> dict[str, str]:
    if model == "DLinear":
        return {"d_model": "512", "d_ff": "2048", "e_layers": "2", "factor": "3", "lr": "0.01"}
    if model in {"iTransformer", "TiDE", "TimeXer"}:
        return {"d_model": "128", "d_ff": "512", "e_layers": "2", "factor": "3", "lr": "0.001"}
    return {
        "d_model": "128", "d_ff": "512", "e_layers": "3",
        "factor": "1" if model == "PFB-Projected" else "3", "lr": "0.0001",
    }


def command(cell: Cell, python_exe: str, ts_root: Path, storage: Path) -> list[str]:
    cfg = DATASETS[cell.dataset]
    hp = hyperparameters(cell.model)
    channels = str(cfg["channels"])
    return [
        python_exe, "-u", "run.py", "--task_name", "long_term_forecast", "--is_training", "1",
        "--model_id", cell.model_id, "--model", cell.model, "--data", str(cfg["data"]),
        "--root_path", str(ts_root / "data"), "--data_path", str(cfg["file"]),
        "--features", "M", "--target", "OT", "--freq", str(cfg["freq"]),
        "--seq_len", "336", "--label_len", "96", "--pred_len", "192",
        "--enc_in", channels, "--dec_in", channels, "--c_out", channels,
        "--patch_len", "16", "--stride", "8", "--d_model", hp["d_model"],
        "--n_heads", "8", "--e_layers", hp["e_layers"], "--d_layers", "1",
        "--d_ff", hp["d_ff"], "--factor", hp["factor"], "--dropout", "0.1",
        "--embed", "timeF", "--num_workers", "0", "--itr", "1",
        "--train_epochs", "100", "--patience", "10", "--learning_rate", hp["lr"],
        "--batch_size", str(cfg["batch"]), "--des", "uniform_preprocessing",
        "--seed", str(cell.seed), "--checkpoints", str(storage / "checkpoints"),
        "--artifacts_root", str(storage / "artifacts"), "--uniform_outer_normalization",
        "--disable_native_window_normalization", "--use_norm", "0",
        "--skip_prediction_arrays", "--defer_test_until_after_training",
    ]


def metric_path(storage: Path, cell: Cell) -> Path | None:
    root = storage / "artifacts" / "results"
    if not root.exists():
        return None
    for folder in root.glob(f"long_term_forecast_{cell.model_id}_*"):
        path = folder / "metrics.npy"
        if path.exists():
            try:
                import numpy as np
                values = np.load(path)
                if values.size >= 2 and all(math.isfinite(float(value)) for value in values[:2]):
                    return path
            except Exception:
                continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default="2021,2022,2023,2024,2025")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--python", default="")
    parser.add_argument("--storage-root", type=Path, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    ts_root = script_dir.parents[1]
    workspace = ts_root
    selected_models, selected_datasets, seeds = items(args.models), items(args.datasets), ints(args.seeds)
    invalid_models = set(selected_models) - set(MODELS)
    invalid_datasets = set(selected_datasets) - set(DATASETS)
    if invalid_models or invalid_datasets:
        raise SystemExit(f"Unknown models={sorted(invalid_models)} datasets={sorted(invalid_datasets)}")
    python_exe = resolve_python(workspace, args.python)
    storage = (args.storage_root or (script_dir / "storage")).resolve()
    (storage / "checkpoints").mkdir(parents=True, exist_ok=True)
    (storage / "artifacts").mkdir(parents=True, exist_ok=True)
    cells = [Cell(model, dataset, seed) for model, dataset, seed in itertools.product(selected_models, selected_datasets, seeds)]

    plan_path = script_dir / "uniform_preprocessing_plan.csv"
    with plan_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "dataset", "horizon", "seed", "model_id", "completed", "command"])
        writer.writeheader()
        for cell in cells:
            writer.writerow({
                "model": cell.model, "dataset": cell.dataset, "horizon": 192, "seed": cell.seed,
                "model_id": cell.model_id, "completed": metric_path(storage, cell) is not None,
                "command": json.dumps(command(cell, python_exe, ts_root, storage)),
            })

    remaining = [cell for cell in cells if not (args.skip_completed and metric_path(storage, cell))]
    print("=" * 78)
    print("COMMON WINDOW-PREPROCESSING FIVE-SEED CAMPAIGN")
    print("=" * 78)
    print(f"Planned cells: {len(cells)}")
    print(f"Remaining cells: {len(remaining)}")
    print(f"Storage: {storage}")
    print(f"Plan: {plan_path}")
    if not args.execute:
        print("Planning only. No model was trained.")
        return 0

    logs = script_dir / "logs" / "uniform_preprocessing"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = script_dir / "uniform_preprocessing_status.csv"
    executed = 0
    for index, cell in enumerate(remaining, 1):
        if args.max_runs and executed >= args.max_runs:
            break
        print(f"[{index}/{len(remaining)}] {cell.model} {cell.dataset} H=192 seed={cell.seed}")
        stdout = logs / f"{cell.model_id}.stdout.log"
        stderr = logs / f"{cell.model_id}.stderr.log"
        started = time.time()
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            result = subprocess.run(command(cell, python_exe, ts_root, storage), cwd=ts_root, stdout=out, stderr=err, text=True)
        metric = metric_path(storage, cell)
        status = "completed" if result.returncode == 0 and metric else "failed"
        exists = status_path.exists()
        with status_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["time", "model", "dataset", "seed", "status", "returncode", "seconds", "metrics", "stderr"])
            if not exists:
                writer.writeheader()
            writer.writerow({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": cell.model, "dataset": cell.dataset,
                             "seed": cell.seed, "status": status, "returncode": result.returncode,
                             "seconds": round(time.time() - started, 2), "metrics": metric or "", "stderr": stderr})
        executed += 1
        if status != "completed":
            print(f"FAILED: {stderr}")
            return result.returncode or 1
    print(f"Completed new runs: {executed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
