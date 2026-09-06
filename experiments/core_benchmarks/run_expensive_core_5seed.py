#!/usr/bin/env python
"""Sequential, resume-safe queue for Solar, Electricity, and Traffic.

Each dataset uses PatchTST, PFB-Direct, and PFB-Projected at horizons
96/192/336 with seeds 2021--2025: 45 cells per dataset, 135 total.
The established dataset-specific runners remain responsible for plans, logs,
status rows, data validation, and completed-result detection.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASETS = ("Solar", "Electricity", "Traffic")
MODELS = "PatchTST,PFB-Direct,PFB-Projected"
HORIZONS = "96,192,336"
SEEDS = "2021,2022,2023,2024,2025"


def parse_datasets(text: str) -> list[str]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    unknown = [value for value in values if value not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown datasets {unknown}; valid values: {list(DATASETS)}")
    return values


def command_for(workspace: Path, dataset: str, max_runs: int, dry_run: bool) -> list[str]:
    common = [
        "--models",
        MODELS,
        "--horizons",
        HORIZONS,
        "--seeds",
        SEEDS,
        "--skip-completed",
    ]
    if max_runs:
        common.extend(["--max-runs", str(max_runs)])
    if dry_run:
        common.append("--dry-run")

    if dataset == "Solar":
        script = workspace / "experiments" / "core_benchmarks" / "run_candidate_core_5seed.py"
        return [sys.executable, str(script), "--datasets", "Solar", *common]
    if dataset == "Electricity":
        script = workspace / "experiments" / "core_benchmarks" / "run_electricity_core_5seed.py"
        return [sys.executable, str(script), *common]
    script = workspace / "experiments" / "core_benchmarks" / "run_traffic_core_5seed.py"
    return [sys.executable, str(script), *common]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the expensive Solar/Electricity/Traffic five-seed queue."
    )
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument(
        "--max-runs-per-dataset",
        type=int,
        default=0,
        help="0 runs every remaining cell; otherwise limits each selected dataset",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_runs_per_dataset < 0:
        raise SystemExit("--max-runs-per-dataset must be zero or positive")

    script_dir = Path(__file__).resolve().parent
    workspace = script_dir.parents[1]
    datasets = parse_datasets(args.datasets)
    print("=" * 78)
    print("EXPENSIVE CORE FIVE-SEED QUEUE")
    print("=" * 78)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Models: {MODELS}")
    print(f"Horizons: {HORIZONS}")
    print(f"Seeds: {SEEDS}")
    print(f"Full selected plan: {45 * len(datasets)} cells")
    print(
        "New-run limit per dataset: "
        + ("ALL" if args.max_runs_per_dataset == 0 else str(args.max_runs_per_dataset))
    )
    print(f"Dry run: {args.dry_run}")
    print("=" * 78)

    for index, dataset in enumerate(datasets, start=1):
        command = command_for(
            workspace, dataset, args.max_runs_per_dataset, args.dry_run
        )
        print(f"QUEUE [{index}/{len(datasets)}]: {dataset}")
        result = subprocess.run(command, cwd=workspace)
        if result.returncode != 0:
            print(f"Queue stopped at {dataset}; completed outputs remain resumable.")
            return result.returncode
    print("Queue complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
