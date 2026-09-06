#!/usr/bin/env python
"""Run one paper-aligned TimeSqueeze in-distribution reproduction cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


CONFIGS = {
    "ETTh2": {"data": "ETTh2", "freq": "h", "enc_in": 7},
    "ETTm2": {"data": "ETTm2", "freq": "t", "enc_in": 7},
    "Weather": {"data": "custom", "freq": "h", "enc_in": 21},
}


def external_storage_root() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        raise SystemExit("LOCALAPPDATA is not defined; cannot select the checkpoint drive")
    root = Path(local_appdata) / "PFB-4CAST" / "recent_baselines"
    root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < 512 * 1024 * 1024:
        raise SystemExit(
            f"Checkpoint drive has only {free_bytes / 1024**2:.1f} MB free: {root}"
        )
    return root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one independent TimeSqueeze Section-6 reproduction cell."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=CONFIGS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.model != "TimeSqueeze":
        raise SystemExit(f"This adapter handles TimeSqueeze, not {args.model}")
    if args.seq_len != 96 or args.pred_len != 192:
        raise SystemExit("This focused reproduction is fixed at L=96 and H=192")
    data_file = args.data_root / args.data_path
    if not data_file.exists():
        raise SystemExit(f"Missing dataset: {data_file}")

    library = Path(__file__).resolve().parents[2]
    run_file = library / "run.py"
    model_file = library / "models" / "TimeSqueeze_Reproduction.py"
    for required in (run_file, model_file):
        if not required.exists():
            raise SystemExit(f"Missing required file: {required}")

    cfg = CONFIGS[args.dataset]
    model_id = f"Recent_TimeSqueezeReproduction_{args.dataset}_H192_seed{args.seed}"
    storage_root = external_storage_root()
    checkpoints = storage_root / "checkpoints" / "TimeSqueeze-Reproduction"
    artifacts = storage_root / "artifacts" / "TimeSqueeze-Reproduction"
    command = [
        sys.executable,
        "-u",
        "run.py",
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--root_path",
        str(args.data_root),
        "--data_path",
        args.data_path,
        "--model_id",
        model_id,
        "--model",
        "TimeSqueeze-Reproduction",
        "--data",
        str(cfg["data"]),
        "--features",
        "M",
        "--target",
        "OT",
        "--freq",
        str(cfg["freq"]),
        "--seq_len",
        str(args.seq_len),
        "--label_len",
        "48",
        "--pred_len",
        str(args.pred_len),
        "--enc_in",
        str(cfg["enc_in"]),
        "--dec_in",
        str(cfg["enc_in"]),
        "--c_out",
        str(cfg["enc_in"]),
        "--d_model",
        "64",
        "--n_heads",
        "4",
        "--e_layers",
        "3",
        "--d_ff",
        "256",
        "--dropout",
        "0.1",
        "--timesqueeze_tau",
        "0.3",
        "--timesqueeze_power_window",
        "8",
        "--timesqueeze_max_patch",
        "8",
        "--learning_rate",
        "0.01",
        "--batch_size",
        "128",
        "--train_epochs",
        "30",
        "--patience",
        "10",
        "--num_workers",
        "0",
        "--checkpoints",
        str(checkpoints),
        "--artifacts_root",
        str(artifacts),
        "--des",
        "aligned_reproduction",
        "--seed",
        str(args.seed),
        "--defer_test_until_after_training",
        "--skip_prediction_arrays",
    ]
    def locate_metrics() -> list[Path]:
        return [
            path / "metrics.npy"
            for path in (artifacts / "results").glob(
                f"long_term_forecast_{model_id}_TimeSqueeze-Reproduction_*"
            )
            if (path / "metrics.npy").exists()
        ]

    # A prior process can finish training but stop before the compact JSON is
    # emitted. Reuse its validated scalar metrics instead of retraining it.
    metric_files = locate_metrics()
    if not metric_files:
        result = subprocess.run(command, cwd=library)
        if result.returncode != 0:
            return result.returncode
        metric_files = locate_metrics()
    if len(metric_files) != 1:
        raise SystemExit(f"Expected one metrics file for {model_id}, found {metric_files}")
    values = np.load(metric_files[0])
    if values.size < 2:
        raise SystemExit(f"Malformed metrics: {metric_files[0]}")

    payload = {
        "model": "TimeSqueeze",
        "dataset": args.dataset,
        "horizon": args.pred_len,
        "lookback": args.seq_len,
        "seed": args.seed,
        "mae": float(values[0]),
        "mse": float(values[1]),
        "implementation_status": "independent paper-aligned reproduction",
        "variant": "in-distribution encoder-only forecasting experiment",
        "protocol": (
            "relative-deviation patching (tau=0.3, maximum patch=8) with a "
            "non-causal GlobalTransformer-style backbone"
        ),
        "paper": "https://arxiv.org/abs/2603.11352",
        "official_code_available_at_preparation": False,
        "model_sha256": sha256(model_file),
        "source_metrics": str(metric_files[0]),
        "checkpoint_root": str(checkpoints),
        "artifacts_root": str(artifacts),
        "command": command,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
