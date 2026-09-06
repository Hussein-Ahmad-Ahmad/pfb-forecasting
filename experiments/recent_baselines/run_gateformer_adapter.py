#!/usr/bin/env python
"""Uniform adapter for the official Gateformer implementation."""

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
    "ETTh2": {
        "data": "ETTh2",
        "freq": "h",
        "enc_in": 7,
        "batch_size": 8,
        "d_model": 64,
        "d_ff": 64,
        "n_heads": 4,
        "e_layers": 1,
        "learning_rate": 0.0005,
    },
    "ETTm2": {
        "data": "ETTm2",
        "freq": "t",
        "enc_in": 7,
        "batch_size": 8,
        "d_model": 64,
        "d_ff": 64,
        "n_heads": 4,
        "e_layers": 1,
        "learning_rate": 0.0001,
    },
    "Weather": {
        "data": "custom",
        "freq": "h",
        "enc_in": 21,
        "batch_size": 8,
        "d_model": 512,
        "d_ff": 1024,
        "n_heads": 8,
        "e_layers": 2,
        "learning_rate": 0.0001,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def external_checkpoint_root() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        raise SystemExit("LOCALAPPDATA is not defined; cannot select the checkpoint drive")
    root = Path(local_appdata) / "PFB-4CAST" / "recent_baselines" / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < 512 * 1024 * 1024:
        raise SystemExit(
            f"Checkpoint drive has only {free_bytes / 1024**2:.1f} MB free: {root}"
        )
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one official Gateformer cell.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=CONFIGS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.model != "Gateformer":
        raise SystemExit(f"This adapter handles Gateformer, not {args.model}")
    if args.pred_len != 192:
        raise SystemExit("The focused campaign is fixed at H=192")
    data_file = args.data_root / args.data_path
    if not data_file.exists():
        raise SystemExit(f"Missing dataset: {data_file}")

    package_dir = Path(__file__).resolve().parent
    source = package_dir / "official_sources" / "Gateformer"
    run_file = source / "run.py"
    if not run_file.exists():
        raise SystemExit(f"Missing official Gateformer source: {run_file}")
    cfg = CONFIGS[args.dataset]
    model_id = f"Recent_Gateformer_{args.dataset}_H192_seed{args.seed}"
    checkpoints = external_checkpoint_root() / "Gateformer"
    command = [
        sys.executable,
        "-u",
        "run.py",
        "--is_training",
        "1",
        "--root_path",
        str(args.data_root),
        "--data_path",
        args.data_path,
        "--model_id",
        model_id,
        "--model",
        "Gateformer",
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
        "--e_layers",
        str(cfg["e_layers"]),
        "--d_layers",
        "1",
        "--factor",
        "3",
        "--enc_in",
        str(cfg["enc_in"]),
        "--dec_in",
        str(cfg["enc_in"]),
        "--c_out",
        str(cfg["enc_in"]),
        "--batch_size",
        str(cfg["batch_size"]),
        "--d_model",
        str(cfg["d_model"]),
        "--d_ff",
        str(cfg["d_ff"]),
        "--learning_rate",
        str(cfg["learning_rate"]),
        "--n_heads",
        str(cfg["n_heads"]),
        "--train_epochs",
        "10",
        "--patience",
        "3",
        "--num_workers",
        "0",
        "--checkpoints",
        str(checkpoints),
        "--des",
        "recent_context",
        "--seed",
        str(args.seed),
    ]
    result = subprocess.run(command, cwd=source)
    if result.returncode != 0:
        return result.returncode

    matches = list((source / "results").glob(f"{model_id}_Gateformer_*"))
    metrics_files = [path / "metrics.npy" for path in matches if (path / "metrics.npy").exists()]
    if len(metrics_files) != 1:
        raise SystemExit(
            f"Expected one Gateformer metrics file for {model_id}, found {metrics_files}"
        )
    values = np.load(metrics_files[0])
    if values.size < 2:
        raise SystemExit(f"Malformed metrics: {metrics_files[0]}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    payload = {
        "model": "Gateformer",
        "dataset": args.dataset,
        "horizon": args.pred_len,
        "lookback": args.seq_len,
        "seed": args.seed,
        "mae": float(values[0]),
        "mse": float(values[1]),
        "protocol": "official dataset-specific H=192 architecture and optimization settings",
        "official_repository": "https://github.com/nyuolab/Gateformer",
        "upstream_commit": commit,
        "seed_control_patch": True,
        "patched_run_sha256": file_sha256(run_file),
        "source_metrics": str(metrics_files[0]),
        "command": command,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
