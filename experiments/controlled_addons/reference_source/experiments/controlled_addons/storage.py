"""Atomic records and fail-closed completion checks."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .design import canonical, digest, file_hash, validate_spec


def write_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Frozen file differs; use a new output directory: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["no_completed_rows"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})
    temporary.replace(path)


def validate_receipt(folder, spec):
    validate_spec(spec)
    folder = Path(folder)
    complete = folder / "complete.json"
    if not complete.exists():
        return None
    receipt = json.loads(complete.read_text(encoding="utf-8"))
    if receipt.get("spec") != spec or receipt.get("spec_sha256") != digest(spec):
        raise ValueError(f"Completion/spec mismatch: {folder}")
    for filename in ("metrics.npy", "checkpoint.pth", "run_provenance.json", "epoch_history.csv", "training_summary.json"):
        if filename not in receipt["sha256"] or file_hash(folder / filename) != receipt["sha256"][filename]:
            raise ValueError(f"Completed artifact changed: {folder / filename}")
    provenance = json.loads((folder / "run_provenance.json").read_text(encoding="utf-8"))
    if (provenance.get("run_by_gpu_watermark") != "EXP_RUN_BY_RTX4090_GPU"
            or not provenance.get("actual_cuda_used")
            or "RTX 4090" not in str(provenance.get("actual_cuda_device"))
            or provenance.get("split") != spec["split"]
            or provenance.get("model_id") != spec["run_id"]
            or provenance.get("seed") != spec["seed"]):
        raise ValueError(f"Hardware/split/identity mismatch: {folder}")
    metrics = np.load(folder / "metrics.npy", allow_pickle=False)
    if metrics.shape != (5,) or not np.isfinite(metrics).all() or np.any(metrics < 0):
        raise ValueError(f"Invalid metric array: {folder}")
    if not np.isclose(metrics[2]**2, metrics[1], rtol=1e-10, atol=1e-14):
        raise ValueError(f"RMSE/MSE inconsistent: {folder}")
    summary = json.loads((folder / "training_summary.json").read_text(encoding="utf-8"))
    if receipt.get("summary") != summary or summary.get("test_ever_loaded") != (spec["split"] == "test"):
        raise ValueError(f"Training-summary/split mismatch: {folder}")
    return dict(receipt, folder=str(folder.resolve()), mae=float(metrics[0]), mse=float(metrics[1]))


def completed(output, spec):
    found = []
    directory = Path(output) / "runs" / spec["run_id"]
    if directory.exists():
        for attempt in sorted(directory.glob("attempt_*")):
            result = validate_receipt(attempt, spec)
            if result:
                found.append(result)
    if len(found) > 1:
        raise ValueError(f"Ambiguous duplicate completed results: {directory}")
    return found[0] if found else None
