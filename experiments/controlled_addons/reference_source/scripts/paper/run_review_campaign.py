#!/usr/bin/env python
"""Reproducible runners for the IEEE-review repair experiments.

The runner deliberately separates validation-only HPO selection from locked
five-seed test evaluation.  It also prepares the common outer-preprocessing
grid.  Every invocation writes plans, manifests, per-run logs, and a status
CSV beneath ``results_analysis/review_campaigns``.

Examples
--------
Create or run HPO selection trials (no test metrics are read)::

    python scripts/paper/run_review_campaign.py hpo-select --dry-run

After all selection trials have completed, lock one configuration per
model/dataset and run independent test seeds::

    python scripts/paper/run_review_campaign.py hpo-lock
    python scripts/paper/run_review_campaign.py hpo-test --skip-completed

Run the matched outer-preprocessing grid::

    python scripts/paper/run_review_campaign.py common-preproc --skip-completed
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "results_analysis" / "review_campaigns"
PRIMARY_DATASETS = {
    "ETTh1": {"data": "ETTh1", "file": "ETTh1.csv", "freq": "h", "channels": 7, "batch": 32},
    "ETTh2": {"data": "ETTh2", "file": "ETTh2.csv", "freq": "h", "channels": 7, "batch": 32},
    "ETTm1": {"data": "ETTm1", "file": "ETTm1.csv", "freq": "t", "channels": 7, "batch": 32},
    "ETTm2": {"data": "ETTm2", "file": "ETTm2.csv", "freq": "t", "channels": 7, "batch": 32},
    "Exchange": {"data": "custom", "file": "exchange_rate.csv", "freq": "h", "channels": 8, "batch": 32},
    "Weather": {"data": "custom", "file": "weather.csv", "freq": "h", "channels": 21, "batch": 16},
}
MODELS = ("PatchTST", "PFB-Direct", "PFB-Projected", "DLinear", "iTransformer", "TiDE", "TimeXer")
EVALUATION_SEEDS = (2021, 2022, 2023, 2024, 2025)
SELECTION_SEED = 2020


@dataclass(frozen=True)
class Run:
    campaign: str
    stage: str
    dataset: str
    model: str
    horizon: int
    seed: int
    tag: str
    params: tuple[tuple[str, str], ...]
    selection_only: bool = False
    report_split: str = "test"
    outer_window_norm: bool = False

    @property
    def model_id(self) -> str:
        return f"{self.campaign}_{self.stage}_{self.dataset}_{self.model}_H{self.horizon}_S{self.seed}_{self.tag}"


def parse_csv(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def assert_known(values: Iterable[str], allowed: Iterable[str], label: str) -> None:
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise SystemExit(f"Unknown {label}: {', '.join(unknown)}")


def resolve_python(requested: str) -> str:
    if requested:
        return requested
    return sys.executable


def grid_candidate(model: str, trial: int) -> dict[str, str]:
    """Deterministic, equal-count, validation-only candidate generator.

    The grid is deliberately compact; ``--trials`` is the same for every
    family.  Model-specific parameters are only varied where supported.
    """
    common = [
        {"learning_rate": "0.0001", "dropout": "0.05", "d_model": "64", "n_heads": "4", "e_layers": "2", "d_ff": "256", "patch_len": "8", "stride": "4"},
        {"learning_rate": "0.0001", "dropout": "0.10", "d_model": "128", "n_heads": "8", "e_layers": "3", "d_ff": "512", "patch_len": "16", "stride": "8"},
        {"learning_rate": "0.0003", "dropout": "0.10", "d_model": "128", "n_heads": "8", "e_layers": "2", "d_ff": "256", "patch_len": "32", "stride": "16"},
        {"learning_rate": "0.0003", "dropout": "0.20", "d_model": "64", "n_heads": "4", "e_layers": "3", "d_ff": "512", "patch_len": "16", "stride": "8"},
        {"learning_rate": "0.001", "dropout": "0.05", "d_model": "128", "n_heads": "8", "e_layers": "3", "d_ff": "256", "patch_len": "8", "stride": "4"},
        {"learning_rate": "0.001", "dropout": "0.20", "d_model": "64", "n_heads": "4", "e_layers": "2", "d_ff": "512", "patch_len": "32", "stride": "16"},
    ]
    candidate = dict(common[trial % len(common)])
    if model == "DLinear":
        candidate = {"learning_rate": ("0.001", "0.003", "0.01")[trial % 3], "individual": "0"}
    elif model == "TiDE":
        candidate.update({"d_model": "128", "e_layers": "2", "d_ff": "512"})
    return candidate


def command_for(run: Run, python_exe: str) -> list[str]:
    cfg = PRIMARY_DATASETS[run.dataset]
    cmd = [
        python_exe, "-u", "run.py",
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--model_id", run.model_id, "--model", run.model,
        "--data", str(cfg["data"]), "--root_path", "./data/", "--data_path", str(cfg["file"]),
        "--features", "M", "--freq", str(cfg["freq"]),
        "--seq_len", "336", "--label_len", "96", "--pred_len", str(run.horizon),
        "--enc_in", str(cfg["channels"]), "--dec_in", str(cfg["channels"]), "--c_out", str(cfg["channels"]),
        "--batch_size", str(cfg["batch"]), "--train_epochs", "100", "--patience", "10",
        "--num_workers", "0", "--itr", "1", "--seed", str(run.seed),
        "--des", run.campaign, "--report_split", run.report_split,
    ]
    for key, value in run.params:
        if key == "individual":
            if value == "1":
                cmd.append("--individual")
        else:
            cmd.extend([f"--{key}", value])
    if run.selection_only:
        cmd.append("--selection_only")
    if run.outer_window_norm:
        cmd.append("--outer_window_norm")
    return cmd


def metrics_path(run: Run) -> Path | None:
    root = ROOT / ("results" if run.report_split == "test" else "val_results")
    matches = sorted(root.glob(f"long_term_forecast_{run.model_id}_*/metrics.npy")) if root.exists() else []
    return matches[0] if matches else None


def make_hpo_selection(datasets: list[str], models: list[str], trials: int) -> list[Run]:
    return [
        Run("equal_budget_hpo", "selection", dataset, model, 192, SELECTION_SEED, f"T{trial:02d}",
            tuple(sorted(grid_candidate(model, trial).items())), selection_only=True, report_split="val")
        for dataset in datasets for model in models for trial in range(trials)
    ]


def make_preproc(datasets: list[str], models: list[str], horizons: list[int], seeds: list[int]) -> list[Run]:
    base = {"learning_rate": "0.0001", "dropout": "0.1", "d_model": "128", "n_heads": "8", "e_layers": "3", "d_ff": "512", "patch_len": "16", "stride": "8"}
    return [
        Run("common_outer_preproc", "test", dataset, model, horizon, seed, "fixed", tuple(sorted(base.items())), outer_window_norm=True)
        for dataset in datasets for model in models for horizon in horizons for seed in seeds
    ]


def campaign_dir(name: str) -> Path:
    path = OUT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(exist_ok=True)
    return path


def write_plan(path: Path, runs: list[Run], python_exe: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["campaign", "stage", "dataset", "model", "horizon", "seed", "tag", "selection_only", "report_split", "outer_window_norm", "params", "model_id", "command"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = asdict(run)
            row["params"] = json.dumps(dict(run.params), sort_keys=True)
            row["model_id"] = run.model_id
            row["command"] = " ".join(command_for(run, python_exe))
            writer.writerow(row)


def append_status(path: Path, run: Run, status: str, elapsed: float, returncode: int, metric: Path | None, stdout: Path, stderr: Path) -> None:
    fields = ["timestamp", "model_id", "dataset", "model", "horizon", "seed", "tag", "status", "returncode", "elapsed_seconds", "metrics_path", "stdout_log", "stderr_log"]
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "model_id": run.model_id, "dataset": run.dataset, "model": run.model, "horizon": run.horizon, "seed": run.seed, "tag": run.tag, "status": status, "returncode": returncode, "elapsed_seconds": round(elapsed, 3), "metrics_path": "" if metric is None else str(metric), "stdout_log": str(stdout), "stderr_log": str(stderr)})


def execute(runs: list[Run], python_exe: str, output: Path, args: argparse.Namespace) -> int:
    write_plan(output / "plan.csv", runs, python_exe)
    (output / "manifest.json").write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "python": python_exe, "runs": len(runs), "arguments": vars(args)}, indent=2), encoding="utf-8")
    status_path = output / "status.csv"
    executed = 0
    for index, run in enumerate(runs, start=1):
        existing = metrics_path(run)
        if args.skip_completed and existing:
            print(f"[{index}/{len(runs)}] SKIP {run.model_id}")
            append_status(status_path, run, "skipped_completed", 0, 0, existing, Path(), Path())
            continue
        if args.max_runs and executed >= args.max_runs:
            break
        command = command_for(run, python_exe)
        print(f"[{index}/{len(runs)}] {'DRY' if args.dry_run else 'RUN'} {run.model_id}")
        if args.dry_run:
            print(" ".join(command))
            executed += 1
            continue
        stdout = output / "logs" / f"{run.model_id}.stdout.log"
        stderr = output / "logs" / f"{run.model_id}.stderr.log"
        started = time.time()
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            proc = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, text=True)
        metric = metrics_path(run)
        state = "completed" if proc.returncode == 0 and metric else "not_completed"
        append_status(status_path, run, state, time.time() - started, proc.returncode, metric, stdout, stderr)
        executed += 1
        if state != "completed":
            print(f"FAILED: inspect {stderr}")
            return proc.returncode or 1
    print(f"Planned {len(runs)} runs; executed {executed}. Artifacts: {output}")
    return 0


def lock_hpo(selection_runs: list[Run]) -> int:
    grouped: dict[tuple[str, str], list[tuple[float, Run, Path]]] = {}
    missing = []
    for run in selection_runs:
        metric = metrics_path(run)
        if metric is None:
            missing.append(run.model_id)
            continue
        grouped.setdefault((run.dataset, run.model), []).append((float(np.load(metric)[1]), run, metric))
    if missing:
        print("Cannot lock HPO: selection metrics are missing for {} trial(s).".format(len(missing)))
        return 1
    locked = {}
    for key, candidates in sorted(grouped.items()):
        mse, run, metric = min(candidates, key=lambda item: item[0])
        locked[f"{key[0]}::{key[1]}"] = {"validation_mse": mse, "source_metrics": str(metric), "params": dict(run.params), "selection_seed": run.seed}
    output = campaign_dir("equal_budget_hpo") / "locked_configs.json"
    output.write_text(json.dumps(locked, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Locked {len(locked)} configurations: {output}")
    return 0


def make_hpo_test(datasets: list[str], models: list[str], seeds: list[int]) -> list[Run]:
    path = campaign_dir("equal_budget_hpo") / "locked_configs.json"
    if not path.exists():
        raise SystemExit("Run hpo-lock after all validation trials complete.")
    locked = json.loads(path.read_text(encoding="utf-8"))
    runs = []
    for dataset in datasets:
        for model in models:
            item = locked.get(f"{dataset}::{model}")
            if item is None:
                raise SystemExit(f"No locked configuration for {dataset}/{model}")
            for seed in seeds:
                runs.append(Run("equal_budget_hpo", "locked_test", dataset, model, 192, seed, "locked", tuple(sorted(item["params"].items()))))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", choices=("hpo-select", "hpo-lock", "hpo-test", "common-preproc"))
    parser.add_argument("--datasets", default=",".join(PRIMARY_DATASETS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--horizons", default="96,192,336")
    parser.add_argument("--seeds", default=",".join(map(str, EVALUATION_SEEDS)))
    parser.add_argument("--trials", type=int, default=6, help="Equal number of validation trials per model/dataset.")
    parser.add_argument("--python", default="", help="Python interpreter that has torch and project dependencies.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    args = parser.parse_args()
    datasets, models = parse_csv(args.datasets), parse_csv(args.models)
    assert_known(datasets, PRIMARY_DATASETS, "datasets")
    assert_known(models, MODELS, "models")
    if args.trials < 1:
        raise SystemExit("--trials must be positive")
    python_exe = resolve_python(args.python)
    selection = make_hpo_selection(datasets, models, args.trials)
    if args.campaign == "hpo-lock":
        return lock_hpo(selection)
    if args.campaign == "hpo-select":
        return execute(selection, python_exe, campaign_dir("equal_budget_hpo"), args)
    if args.campaign == "hpo-test":
        return execute(make_hpo_test(datasets, models, parse_csv(args.seeds, int)), python_exe, campaign_dir("equal_budget_hpo"), args)
    return execute(make_preproc(datasets, models, parse_csv(args.horizons, int), parse_csv(args.seeds, int)), python_exe, campaign_dir("common_outer_preproc"), args)


if __name__ == "__main__":
    raise SystemExit(main())
