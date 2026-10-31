#!/usr/bin/env python
"""Resume the original full P1 grids, without changing the frozen manuscript.

Six datasets, H={96,192,336}, five seeds: 180 ensemble-member runs,
1080 lookback runs, and 810 temporal-prefix runs. The latter retain the
original custom 70/10/20 split and are not fixed-test rolling-origin trials.
The separately completed H=720 targeted lookback runs are preserved.
"""

from __future__ import annotations

import argparse
import csv
import errno
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import run_review_gap_campaigns as gap


DATASETS = list(gap.DATASETS)
HORIZONS = [96, 192, 336]
SEEDS = list(gap.DEFAULT_SEEDS_5)
LOOKBACKS = [96, 192, 336, 720]
PREFIXES = [0.60, 0.80, 1.00]
OUT = gap.OUT_ROOT / "full_p1"


class QueueAlreadyRunning(RuntimeError):
    """Expected duplicate-launch condition, not a training failure."""


def build_plans():
    return {
        "ensemble-members": gap.ensemble_members(DATASETS, HORIZONS, SEEDS),
        "lookback": gap.lookback(DATASETS, HORIZONS, SEEDS, LOOKBACKS),
        "rolling-origin": gap.rolling_origin(DATASETS, HORIZONS, SEEDS, PREFIXES, prepare_files=False),
    }


def inventory(plans):
    return {
        name: {
            "planned": len(runs),
            "completed": sum(gap.metrics_path(run, strict=True) is not None for run in runs),
        }
        for name, runs in plans.items()
    }


@contextmanager
def queue_lock(path):
    """OS lock released automatically on exit, including abrupt termination."""
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK} and getattr(exc, "winerror", None) != 33:
                raise
            raise QueueAlreadyRunning("Another full P1 queue is already running. Do not launch a duplicate.") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def export_metrics(path, plans):
    """Export only this full grid; retain lookback/prefix tags and GPU labels."""
    fields = ["campaign", "dataset", "model", "horizon", "seed", "tag", "seq_len",
              "label_len", "model_id", "mae", "mse", "rmse", "mape", "mspe",
              "metrics_path", "run_by_gpu_watermark"]
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for runs in plans.values():
            for run in runs:
                metric = gap.metrics_path(run, strict=True)
                if metric is None:
                    continue
                values = np.load(metric, allow_pickle=False)
                writer.writerow({
                    **{field: getattr(run, field) for field in fields[:9]},
                    **dict(zip(fields[9:14], values.tolist())),
                    "metrics_path": metric.relative_to(gap.ROOT).as_posix(),
                    "run_by_gpu_watermark": gap.WATERMARK,
                })
    temporary.replace(path)


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def print_inventory(plans):
    counts = inventory(plans)
    for name, count in counts.items():
        count["remaining"] = count["planned"] - count["completed"]
        print(f"{name}: {count['planned']} planned; {count['completed']} verified completed; {count['remaining']} remaining.", flush=True)
    return counts


def run_locked_queue(args):
    """Only inspect results and train after acquiring exclusive ownership."""
    plans = build_plans()
    counts = print_inventory(plans)

    import torch
    if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This queue requires the RTX4090; it will not fall back to CPU or another GPU.")
    for dataset in DATASETS:
        source = gap.ROOT / "data" / gap.DATASETS[dataset]["file"]
        if not source.is_file():
            raise FileNotFoundError(source)
    if shutil.disk_usage(gap.ROOT).free < args.min_free_gb * 1024 ** 3:
        raise RuntimeError("Insufficient free disk space to start.")

    state = {
        "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running", "scope": "original_full_p1_grid",
        "datasets": DATASETS, "horizons": HORIZONS, "seeds": SEEDS,
        "lookbacks": LOOKBACKS, "prefixes": PREFIXES,
        "counts_at_start": counts, "run_by_gpu_watermark": gap.WATERMARK,
        "note": "Exploratory extension after targeted results; no manuscript files are modified. Temporal-prefix sensitivity, not fixed-test rolling origin.",
    }
    save_state(OUT / "queue.json", state)
    try:
        for name, runs in plans.items():
            state["active_campaign"] = name
            save_state(OUT / "queue.json", state)
            print(f"\nStarting {name} (sequential training, skip verified RTX4090 results).", flush=True)
            if name == "rolling-origin":
                gap.prepare_rolling_files(DATASETS, PREFIXES)
            campaign_args = argparse.Namespace(
                campaign=name, datasets=",".join(DATASETS), horizons=",".join(map(str, HORIZONS)),
                seeds=",".join(map(str, SEEDS)), lookbacks=",".join(map(str, LOOKBACKS)),
                origins=",".join(map(str, PREFIXES)), python=sys.executable,
                skip_completed=True, dry_run=False, max_runs=0,
                require_rtx4090=True, min_free_gb=args.min_free_gb,
            )
            returncode = gap.execute(runs, sys.executable, OUT / name, campaign_args)
            export_metrics(OUT / "full_grid_metrics.csv", plans)
            if returncode:
                state["status"] = "failed"
                state["returncode"] = returncode
                save_state(OUT / "queue.json", state)
                print("Queue stopped on failure. Fix the logged error and rerun this same command to resume.", flush=True)
                return returncode
        final_counts = inventory(plans)
        if any(count["completed"] != count["planned"] for count in final_counts.values()):
            raise RuntimeError("Final completion audit failed.")
        state.update(status="completed", completed_at=time.strftime("%Y-%m-%d %H:%M:%S"), final_counts=final_counts)
        save_state(OUT / "queue.json", state)
        print("Full P1 queue completed; scoped numerical results: " + str(OUT / "full_grid_metrics.csv"), flush=True)
        return 0
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        save_state(OUT / "queue.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true", help="Read-only inventory; do not prepare prefixes or start training.")
    parser.add_argument("--min-free-gb", type=float, default=50, help="Stop before a run if free disk space is below this many GiB.")
    args = parser.parse_args()
    if args.min_free_gb < 0:
        parser.error("--min-free-gb must be nonnegative")
    if args.plan_only:
        print_inventory(build_plans())
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        with queue_lock(OUT / "queue.lock"):
            return run_locked_queue(args)
    except QueueAlreadyRunning:
        details = ""
        try:
            state = json.loads((OUT / "queue.json").read_text(encoding="utf-8"))
            details = f" (PID {state.get('pid', 'unknown')}, campaign {state.get('active_campaign', 'unknown')})"
        except (OSError, ValueError, AttributeError):
            pass
        print(f"Already running{details}. No second queue was started; the existing run continues.", flush=True)
        print(f"Monitor the existing log: {OUT / 'queue.stdout.log'}", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
