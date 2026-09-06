#!/usr/bin/env python
"""
Resume-safe runners for long-horizon and topology-control experiment grids.

Prepared campaigns:
  serial_parallel
      PatchTST_SerialMatched vs PFB-Direct on 6 datasets x H={96,192,336} x 5 seeds.

  h720_external
      External/context baselines at H=720. Defaults: DLinear, iTransformer, TiDE, TimeXer.

  equal_budget_hpo
      Equal-count candidate grid for PatchTST, PFB-Direct, PFB-Projected, and a selected
      external baseline. This script launches candidates; select final configurations
      from validation logs after completion.

It writes plans, manifests, logs, and status files under this folder and skips
completed runs by detecting metrics.npy in Time-Series-Library/results.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DATASETS = {
    "ETTm1": {"data": "ETTm1", "data_path": "ETTm1.csv", "enc_in": 7, "batch_size": 16, "freq": "t"},
    "ETTm2": {"data": "ETTm2", "data_path": "ETTm2.csv", "enc_in": 7, "batch_size": 16, "freq": "t"},
    "ETTh1": {"data": "ETTh1", "data_path": "ETTh1.csv", "enc_in": 7, "batch_size": 16, "freq": "h"},
    "ETTh2": {"data": "ETTh2", "data_path": "ETTh2.csv", "enc_in": 7, "batch_size": 16, "freq": "h"},
    "Exchange": {"data": "custom", "data_path": "exchange_rate.csv", "enc_in": 8, "batch_size": 16, "freq": "d"},
    "Weather": {"data": "custom", "data_path": "weather.csv", "enc_in": 21, "batch_size": 8, "freq": "h"},
}

PATCH_MODELS = {"PatchTST", "PFB-Direct", "PFB-Projected", "PatchTST_SerialMatched"}
PFB_PROJECTED = "PFB-Projected"
DLINEAR_MODELS = {"DLinear", "DLinear_Norm"}
MODERN_MODELS = {"iTransformer", "TiDE", "TimeXer"}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    model: str
    dataset: str
    horizon: int
    seed: int
    params: dict[str, str]

    @property
    def campaign(self) -> str:
        return self.candidate_id.split("_")[0]

    @property
    def model_id(self) -> str:
        return f"{self.candidate_id}_{self.model}_{self.dataset}_H{self.horizon}_seed{self.seed}"


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_strings(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def label_len_for(horizon: int, model: str) -> int:
    if model == PFB_PROJECTED:
        return 48
    return 96


def python_has_torch_numpy(python_exe: str) -> bool:
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


def resolve_python(workspace_root: Path, requested_python: str | None) -> str:
    candidates = []
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
    seen: set[str] = set()
    checked: list[str] = []
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        checked.append(candidate)
        if python_has_torch_numpy(candidate):
            return candidate
    raise SystemExit("No Python interpreter with torch and numpy found. Checked:\n  - " + "\n  - ".join(checked))


def base_params_for(model: str) -> dict[str, str]:
    if model in DLINEAR_MODELS:
        return {
            "d_model": "512",
            "d_ff": "2048",
            "e_layers": "2",
            "d_layers": "1",
            "n_heads": "8",
            "factor": "3",
            "learning_rate": "0.01",
            "dropout": "0.1",
        }
    if model in MODERN_MODELS:
        return {
            "d_model": "128",
            "d_ff": "512",
            "e_layers": "2",
            "d_layers": "1",
            "n_heads": "8",
            "factor": "3",
            "learning_rate": "0.001",
            "dropout": "0.1",
        }
    return {
        "d_model": "128",
        "d_ff": "512",
        "e_layers": "3",
        "d_layers": "1",
        "n_heads": "8",
        "factor": "1" if model == PFB_PROJECTED else "3",
        "learning_rate": "0.0001",
        "dropout": "0.1",
        "patch_len": "16",
        "stride": "8",
    }


def command_for(spec: Candidate, python_exe: str) -> list[str]:
    cfg = DATASETS[spec.dataset]
    enc_in = str(cfg["enc_in"])
    params = base_params_for(spec.model)
    params.update(spec.params)

    cmd = [
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
        str(cfg["data"]),
        "--root_path",
        "./data/",
        "--data_path",
        str(cfg["data_path"]),
        "--features",
        "M",
        "--freq",
        str(cfg["freq"]),
        "--seq_len",
        "336",
        "--label_len",
        str(label_len_for(spec.horizon, spec.model)),
        "--pred_len",
        str(spec.horizon),
        "--enc_in",
        enc_in,
        "--dec_in",
        enc_in,
        "--c_out",
        enc_in,
        "--d_model",
        params["d_model"],
        "--d_ff",
        params["d_ff"],
        "--e_layers",
        params["e_layers"],
        "--d_layers",
        params["d_layers"],
        "--n_heads",
        params["n_heads"],
        "--factor",
        params["factor"],
        "--dropout",
        params["dropout"],
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
        params["learning_rate"],
        "--batch_size",
        str(cfg["batch_size"]),
        "--des",
        "campaign_grid",
        "--seed",
        str(spec.seed),
    ]
    if spec.model in PATCH_MODELS:
        cmd += ["--patch_len", params["patch_len"], "--stride", params["stride"]]
    return cmd


def result_metrics_path(ts_root: Path, spec: Candidate) -> Path | None:
    result_root = ts_root / "results"
    if not result_root.exists():
        return None
    for candidate in result_root.glob(f"long_term_forecast_{spec.model_id}_*"):
        metrics = candidate / "metrics.npy"
        if metrics.exists():
            return metrics
    return None


def serial_parallel_plan(datasets: list[str], horizons: list[int], seeds: list[int], models: list[str]) -> list[Candidate]:
    allowed = ["PatchTST_SerialMatched", "PFB-Direct"]
    models = [m for m in allowed if m in models]
    plan = []
    for dataset, horizon, model, seed in itertools.product(datasets, horizons, models, seeds):
        plan.append(Candidate("SerialParallel", model, dataset, horizon, seed, {}))
    return plan


def h720_external_plan(datasets: list[str], seeds: list[int], models: list[str]) -> list[Candidate]:
    allowed = ["DLinear", "iTransformer", "TiDE", "TimeXer"]
    models = [m for m in allowed if m in models]
    return [
        Candidate("H720External", model, dataset, 720, seed, {})
        for dataset, model, seed in itertools.product(datasets, models, seeds)
    ]


def equal_budget_hpo_plan(
    datasets: list[str],
    horizons: list[int],
    seeds: list[int],
    models: list[str],
    max_candidates_per_cell: int,
) -> list[Candidate]:
    # Same candidate count per model/dataset/horizon. The grid is intentionally
    # compact; increase --max-candidates-per-cell if compute allows.
    patch_grid = []
    for p, s in [("8", "4"), ("16", "8"), ("32", "16"), ("64", "32")]:
        for lr in ["0.0001", "0.0003"]:
            patch_grid.append({"patch_len": p, "stride": s, "learning_rate": lr, "dropout": "0.1", "d_model": "128"})
    for p, s in [("8", "4"), ("16", "8"), ("32", "16"), ("64", "32")]:
        for lr in ["0.0001", "0.0003"]:
            patch_grid.append({"patch_len": p, "stride": s, "learning_rate": lr, "dropout": "0.2", "d_model": "128"})
    modern_grid = [
        {"learning_rate": lr, "dropout": dr, "d_model": dm, "d_ff": ff}
        for lr in ["0.0003", "0.001"]
        for dr in ["0.1", "0.2"]
        for dm, ff in [("128", "512"), ("256", "1024")]
    ]
    dlinear_grid = [
        {"learning_rate": lr, "dropout": dr}
        for lr in ["0.001", "0.005", "0.01", "0.02"]
        for dr in ["0.0", "0.1"]
    ]
    plan: list[Candidate] = []
    for dataset, horizon, model, seed in itertools.product(datasets, horizons, models, seeds):
        grid = patch_grid if model in PATCH_MODELS else dlinear_grid if model in DLINEAR_MODELS else modern_grid
        selected = grid[:max_candidates_per_cell]
        for idx, params in enumerate(selected, start=1):
            cid = f"EqualBudgetHPO_C{idx:02d}"
            plan.append(Candidate(cid, model, dataset, horizon, seed, params))
    return plan


def write_plan_csv(path: Path, plan: list[Candidate], python_exe: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["campaign", "candidate_id", "dataset", "horizon", "model", "seed", "model_id", "params_json", "command"],
        )
        writer.writeheader()
        for spec in plan:
            writer.writerow(
                {
                    "campaign": spec.candidate_id.split("_")[0],
                    "candidate_id": spec.candidate_id,
                    "dataset": spec.dataset,
                    "horizon": spec.horizon,
                    "model": spec.model,
                    "seed": spec.seed,
                    "model_id": spec.model_id,
                    "params_json": json.dumps(spec.params, sort_keys=True),
                    "command": json.dumps(command_for(spec, python_exe)),
                }
            )


def append_status(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = [
        "timestamp",
        "candidate_id",
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
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_plan(args: argparse.Namespace) -> list[Candidate]:
    datasets = parse_strings(args.datasets)
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}. Valid: {sorted(DATASETS)}")
    horizons = parse_ints(args.horizons)
    seeds = parse_ints(args.seeds)
    models = parse_strings(args.models)
    if args.campaign == "serial_parallel":
        return serial_parallel_plan(datasets, horizons, seeds, models)
    if args.campaign == "h720_external":
        return h720_external_plan(datasets, seeds, models)
    if args.campaign == "equal_budget_hpo":
        return equal_budget_hpo_plan(datasets, horizons, seeds, models, args.max_candidates_per_cell)
    raise SystemExit(f"Unsupported campaign: {args.campaign}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run long-horizon and topology-control campaigns.")
    parser.add_argument("--campaign", required=True, choices=["serial_parallel", "h720_external", "equal_budget_hpo"])
    parser.add_argument("--datasets", default="ETTm1,ETTm2,ETTh1,ETTh2,Exchange,Weather")
    parser.add_argument("--horizons", default="96,192,336")
    parser.add_argument("--seeds", default="2021,2022,2023,2024,2025")
    parser.add_argument("--models", default="")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means all remaining runs.")
    parser.add_argument("--max-candidates-per-cell", type=int, default=8)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default="")
    args = parser.parse_args()

    if not args.models:
        if args.campaign == "serial_parallel":
            args.models = "PatchTST_SerialMatched,PFB-Direct"
        elif args.campaign == "h720_external":
            args.models = "DLinear,iTransformer,TiDE,TimeXer"
        else:
            args.models = "PatchTST,PFB-Direct,PFB-Projected,TiDE"

    script_dir = Path(__file__).resolve().parent
    ts_root = script_dir.parents[1]
    workspace_root = ts_root
    if not (ts_root / "run.py").exists():
        raise SystemExit(f"Cannot find Time-Series-Library/run.py at {ts_root}")

    for dataset in parse_strings(args.datasets):
        data_path = ts_root / "data" / str(DATASETS[dataset]["data_path"])
        if not data_path.exists():
            raise SystemExit(f"Missing data file: {data_path}")

    python_exe = resolve_python(workspace_root, args.python or None)
    plan = build_plan(args)
    out_dir = script_dir / args.campaign
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / f"{args.campaign}_plan.csv"
    status_path = out_dir / f"{args.campaign}_status.csv"
    manifest_path = out_dir / f"{args.campaign}_manifest.json"
    write_plan_csv(plan_path, plan, python_exe)
    manifest_path.write_text(
        json.dumps(
            {
                "campaign": args.campaign,
                "datasets": parse_strings(args.datasets),
                "horizons": parse_ints(args.horizons),
                "seeds": parse_ints(args.seeds),
                "models": parse_strings(args.models),
                "n_runs_planned": len(plan),
                "max_candidates_per_cell": args.max_candidates_per_cell,
                "skip_completed": args.skip_completed,
                "max_runs": args.max_runs,
                "dry_run": args.dry_run,
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
    print(f"EXPERIMENT CAMPAIGN: {args.campaign}")
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
        existing_metrics = result_metrics_path(ts_root, spec)
        if args.skip_completed and existing_metrics is not None:
            skipped += 1
            print(f"[{idx}/{len(plan)}] SKIP {spec.model} {spec.dataset} H={spec.horizon} seed={spec.seed}")
            append_status(
                status_path,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "candidate_id": spec.candidate_id,
                    "dataset": spec.dataset,
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
            print(f"[{idx}/{len(plan)}] DRY {spec.model} {spec.dataset} H={spec.horizon} seed={spec.seed}")
            print(" ".join(cmd))
            executed += 1
            continue

        print(f"[{idx}/{len(plan)}] RUN {spec.model} {spec.dataset} H={spec.horizon} seed={spec.seed}")
        start = time.time()
        with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
            proc = subprocess.run(cmd, cwd=ts_root, stdout=out, stderr=err, text=True)
        elapsed = time.time() - start
        metrics_after = result_metrics_path(ts_root, spec)
        status = "completed" if proc.returncode == 0 and metrics_after is not None else "failed"
        append_status(
            status_path,
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "candidate_id": spec.candidate_id,
                "dataset": spec.dataset,
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
            print(f"FAILED: see {stderr_log}")
            return proc.returncode or 1

    print("=" * 78)
    print(f"Done. New/dry-run commands: {executed}; skipped completed: {skipped}")
    print(f"Status CSV: {status_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
