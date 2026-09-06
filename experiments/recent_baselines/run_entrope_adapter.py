#!/usr/bin/env python
"""Uniform adapter for the official EntroPE_v2 implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


CONFIGS = {
    "ETTh2": {"data": "ETTh2", "freq": "h", "enc_in": 7},
    "ETTm2": {"data": "ETTm2", "freq": "t", "enc_in": 7},
    "Weather": {"data": "custom", "freq": "h", "enc_in": 21},
}


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def parse_result(path: Path) -> tuple[float, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = float(value.strip())
    if "mse" not in values or "mae" not in values:
        raise RuntimeError(f"Malformed EntroPE result: {path}")
    if not (math.isfinite(values["mse"]) and math.isfinite(values["mae"])):
        raise RuntimeError(
            f"Non-finite EntroPE result in {path}: "
            f"MSE={values['mse']}, MAE={values['mae']}"
        )
    return values["mse"], values["mae"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one official EntroPE_v2 cell.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=CONFIGS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.model != "EntroPE":
        raise SystemExit(f"This adapter handles EntroPE, not {args.model}")
    if args.pred_len != 192:
        raise SystemExit("The focused campaign is fixed at H=192")
    data_file = args.data_root / args.data_path
    if not data_file.exists():
        raise SystemExit(f"Missing dataset: {data_file}")

    package_dir = Path(__file__).resolve().parent
    source = package_dir / "official_sources" / "EntroPE"
    train_file = source / "scripts" / "train_entrope_v2.py"
    if not train_file.exists():
        raise SystemExit(f"Missing official EntroPE source: {train_file}")
    cfg = CONFIGS[args.dataset]
    model_id = f"Recent_EntroPEStableLR003_{args.dataset}_H192_seed{args.seed}"
    checkpoints = external_checkpoint_root() / "EntroPE"
    command = [
        sys.executable,
        "-u",
        "scripts/train_entrope_v2.py",
        "--data",
        str(cfg["data"]),
        "--root_path",
        str(args.data_root),
        "--data_path",
        args.data_path,
        "--model_id_name",
        args.dataset,
        "--model_id",
        model_id,
        "--seq_len",
        str(args.seq_len),
        "--label_len",
        "48",
        "--pred_len",
        str(args.pred_len),
        "--enc_in",
        str(cfg["enc_in"]),
        "--freq",
        str(cfg["freq"]),
        "--d_model",
        "64",
        "--n_heads",
        "4",
        "--global_layers",
        "3",
        "--local_layers",
        "1",
        "--cmi_threshold",
        "0.125",
        "--k_max_estimate",
        "14",
        "--ape_pooler",
        "--fusion_head",
        "--fusion_token_dim",
        "16",
        "--lambda_count",
        "0.0",
        "--lambda_min",
        "0.0",
        "--lambda_div",
        "0.0",
        "--lambda_nll",
        "0.1",
        "--learning_rate",
        "0.003",
        "--dropout",
        "0.1",
        "--train_epochs",
        "30",
        "--patience",
        "10",
        "--batch_size",
        "128",
        "--num_workers",
        "0",
        "--checkpoints",
        str(checkpoints),
        "--random_seed",
        str(args.seed),
    ]
    result = subprocess.run(command, cwd=source)
    if result.returncode != 0:
        return result.returncode

    result_file = checkpoints / model_id / "result.txt"
    if not result_file.exists():
        raise SystemExit(f"EntroPE did not create {result_file}")
    mse, mae = parse_result(result_file)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    payload = {
        "model": "EntroPE",
        "dataset": args.dataset,
        "horizon": args.pred_len,
        "lookback": args.seq_len,
        "seed": args.seed,
        "mae": mae,
        "mse": mse,
        "protocol": (
            "official EntroPE_v2 architecture with LR=0.003 from the upstream "
            "APE search grid, applied consistently across all five seeds"
        ),
        "protocol_version": "entrope_lr003_five_seed_v1",
        "official_repository": "https://github.com/Sachithx/EntroPE",
        "upstream_commit": commit,
        "seed_control_patch": True,
        "patched_train_sha256": file_sha256(train_file),
        "source_metrics": str(result_file),
        "checkpoint_root": str(checkpoints),
        "command": command,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
