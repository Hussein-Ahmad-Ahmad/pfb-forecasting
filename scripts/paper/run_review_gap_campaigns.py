#!/usr/bin/env python
"""Prepare and run the remaining reviewer-gap experiment campaigns.

This runner is intentionally additive. It does not modify model code or active
campaign state, so plans can be generated while another campaign is running.

Campaigns
---------
projected-match
    PFB-Projected controls with size-matched PatchTST/PFB-Direct variants.

ten-seed
    Ten paired seeds for PFB-Direct vs PatchTST and serial-vs-parallel.

lookback
    Lookback-length robustness for the main PatchTST-family contrast.

rolling-origin
    Approximate rolling-origin robustness using deterministic time-prefix
    dataset copies under data/review_rolling_origin.

ensemble-members
    Train the two PatchTST members needed by the output-level ensemble analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "results_analysis" / "review_gap_campaigns"
WATERMARK = "EXP_RUN_BY_RTX4090_GPU"

DATASETS = {
    "ETTh1": {"data": "ETTh1", "file": "ETTh1.csv", "freq": "h", "channels": 7, "batch": 32},
    "ETTh2": {"data": "ETTh2", "file": "ETTh2.csv", "freq": "h", "channels": 7, "batch": 32},
    "ETTm1": {"data": "ETTm1", "file": "ETTm1.csv", "freq": "t", "channels": 7, "batch": 32},
    "ETTm2": {"data": "ETTm2", "file": "ETTm2.csv", "freq": "t", "channels": 7, "batch": 32},
    "Exchange": {"data": "custom", "file": "exchange_rate.csv", "freq": "d", "channels": 8, "batch": 32},
    "Weather": {"data": "custom", "file": "weather.csv", "freq": "h", "channels": 21, "batch": 16},
}

PATCH_MODELS = {"PatchTST", "PFB-Direct", "PFB-Projected", "PatchTST_SerialMatched"}
DEFAULT_SEEDS_5 = (2021, 2022, 2023, 2024, 2025)
DEFAULT_SEEDS_10 = tuple(range(2021, 2031))


@dataclass(frozen=True)
class Run:
    campaign: str
    dataset: str
    model: str
    horizon: int
    seed: int
    tag: str
    seq_len: int = 336
    label_len: int = 96
    params: tuple[tuple[str, str], ...] = ()
    data_override: str = ""
    file_override: str = ""

    @property
    def model_id(self) -> str:
        return f"{self.campaign}_{self.dataset}_{self.model}_H{self.horizon}_S{self.seed}_{self.tag}"


def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def resolve_python(requested: str) -> str:
    return requested or sys.executable


def assert_known(values: list[str]) -> None:
    unknown = sorted(set(values) - set(DATASETS))
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}. Valid: {sorted(DATASETS)}")


def base_params(model: str) -> dict[str, str]:
    if model == "DLinear":
        return {"learning_rate": "0.001", "d_model": "512", "n_heads": "8", "e_layers": "2", "d_ff": "2048"}
    if model in {"iTransformer", "TiDE", "TimeXer"}:
        return {"learning_rate": "0.001", "dropout": "0.1", "d_model": "128", "n_heads": "8", "e_layers": "2", "d_ff": "512"}
    return {
        "learning_rate": "0.0001",
        "dropout": "0.1",
        "d_model": "128",
        "n_heads": "8",
        "e_layers": "3",
        "d_ff": "512",
        "patch_len": "16",
        "stride": "8",
    }


def command_for(run: Run, python_exe: str) -> list[str]:
    cfg = dict(DATASETS[run.dataset])
    if run.data_override:
        cfg["data"] = run.data_override
    if run.file_override:
        cfg["file"] = run.file_override
    params = base_params(run.model)
    params.update(dict(run.params))
    cmd = [
        python_exe, "-u", "run.py",
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--model_id", run.model_id, "--model", run.model,
        "--data", str(cfg["data"]), "--root_path", "./data/", "--data_path", str(cfg["file"]),
        "--features", "M", "--freq", str(cfg["freq"]),
        "--seq_len", str(run.seq_len), "--label_len", str(run.label_len), "--pred_len", str(run.horizon),
        "--enc_in", str(cfg["channels"]), "--dec_in", str(cfg["channels"]), "--c_out", str(cfg["channels"]),
        "--batch_size", str(cfg["batch"]), "--train_epochs", "100", "--patience", "10",
        "--num_workers", "0", "--itr", "1", "--seed", str(run.seed), "--des", run.campaign,
        "--learning_rate", params["learning_rate"], "--dropout", params.get("dropout", "0.1"),
        "--d_model", params["d_model"], "--n_heads", params["n_heads"], "--e_layers", params["e_layers"],
        "--d_ff", params["d_ff"],
    ]
    if run.model in PATCH_MODELS:
        cmd += ["--patch_len", params["patch_len"], "--stride", params["stride"]]
    return cmd


def metrics_path(run: Run, strict: bool = False) -> Path | None:
    result_root = ROOT / "results"
    if not result_root.exists():
        return None
    matches = sorted(result_root.glob(f"long_term_forecast_{run.model_id}_*/metrics.npy"))
    if not matches:
        return None
    if strict:
        if len(matches) != 1:
            raise ValueError(f"Ambiguous results for {run.model_id}: {matches}")
        metric = matches[0]
        values = np.load(metric, allow_pickle=False)
        if values.shape != (5,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid metrics, preserved for inspection: {metric}")
        provenance = json.loads(metric.with_name("run_provenance.json").read_text(encoding="utf-8"))
        expected = {
            "model_id": run.model_id, "model": run.model, "horizon": run.horizon,
            "seed": run.seed, "seq_len": run.seq_len, "label_len": run.label_len,
            "split": "test", "run_by_gpu_watermark": WATERMARK,
        }
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Result provenance mismatch, preserved for inspection: {metric}")
        if not provenance.get("actual_cuda_available") or "RTX 4090" not in str(provenance.get("actual_cuda_device")):
            raise ValueError(f"Result is not verified RTX4090 output: {metric}")
        if provenance.get("actual_cuda_used") is False:
            raise ValueError(f"Result was not executed on CUDA: {metric}")
        if run.campaign == "patchtst_output_ensemble":
            pred = np.load(metric.with_name("pred.npy"), mmap_mode="r", allow_pickle=False)
            true = np.load(metric.with_name("true.npy"), mmap_mode="r", allow_pickle=False)
            expected_tail = (run.horizon, DATASETS[run.dataset]["channels"])
            if pred.ndim != 3 or pred.shape != true.shape or pred.shape[0] == 0 or pred.shape[1:] != expected_tail:
                raise ValueError(f"Invalid ensemble prediction shapes: {metric.parent}")
    return matches[0]


def projected_match(datasets: list[str], horizons: list[int], seeds: list[int]) -> list[Run]:
    configs = {
        "PatchTST": {"d_model": "128", "e_layers": "3", "d_ff": "512"},
        "PFB-Direct": {"d_model": "128", "e_layers": "3", "d_ff": "512"},
        "PFB-Projected": {"d_model": "128", "e_layers": "3", "d_ff": "512"},
        "PatchTST_SerialMatched": {"d_model": "128", "e_layers": "3", "d_ff": "512"},
    }
    return [
        Run("projected_param_match", dataset, model, horizon, seed, "matched", params=tuple(sorted(params.items())))
        for dataset in datasets for horizon in horizons for seed in seeds for model, params in configs.items()
    ]


def ten_seed(datasets: list[str], horizons: list[int], seeds: list[int]) -> list[Run]:
    runs = []
    for dataset in datasets:
        for seed in seeds:
            runs.append(Run("ten_seed_direct_patch", dataset, "PatchTST", 192, seed, "paired"))
            runs.append(Run("ten_seed_direct_patch", dataset, "PFB-Direct", 192, seed, "paired"))
        for horizon in horizons:
            for seed in seeds:
                runs.append(Run("ten_seed_serial_parallel", dataset, "PatchTST_SerialMatched", horizon, seed, "paired"))
                runs.append(Run("ten_seed_serial_parallel", dataset, "PFB-Direct", horizon, seed, "paired"))
    return runs


def lookback(datasets: list[str], horizons: list[int], seeds: list[int], lookbacks: list[int]) -> list[Run]:
    runs = []
    for dataset in datasets:
        for horizon in horizons:
            for seed in seeds:
                for seq_len in lookbacks:
                    label_len = min(96, max(24, seq_len // 2))
                    for model in ("PatchTST", "PFB-Direct", "PFB-Projected"):
                        runs.append(Run("lookback_robustness", dataset, model, horizon, seed, f"L{seq_len}", seq_len, label_len))
    return runs


def prepare_rolling_files(datasets: list[str], fractions: list[float]) -> dict[tuple[str, str], str]:
    target_dir = ROOT / "data" / "review_rolling_origin"
    target_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for dataset in datasets:
        source = ROOT / "data" / DATASETS[dataset]["file"]
        if not source.exists():
            raise SystemExit(f"Missing source data: {source}")
        df = pd.read_csv(source)
        for fraction in fractions:
            tag = f"R{int(round(fraction * 100)):03d}"
            n_rows = max(1000, int(len(df) * fraction))
            out_name = f"review_rolling_origin/{dataset}_{tag}.csv"
            out_path = ROOT / "data" / out_name
            if not out_path.exists():
                df.iloc[:n_rows].to_csv(out_path, index=False)
            mapping[(dataset, tag)] = out_name
    return mapping


def rolling_origin(datasets: list[str], horizons: list[int], seeds: list[int], fractions: list[float], prepare_files: bool = True) -> list[Run]:
    files = prepare_rolling_files(datasets, fractions) if prepare_files else {
        (dataset, f"R{int(round(fraction * 100)):03d}"):
        f"review_rolling_origin/{dataset}_R{int(round(fraction * 100)):03d}.csv"
        for dataset in datasets for fraction in fractions
    }
    runs = []
    for dataset in datasets:
        for fraction in fractions:
            tag = f"R{int(round(fraction * 100)):03d}"
            for horizon in horizons:
                for seed in seeds:
                    for model in ("PatchTST", "PFB-Direct", "PFB-Projected"):
                        runs.append(
                            Run(
                                "rolling_origin_prefix",
                                dataset,
                                model,
                                horizon,
                                seed,
                                tag,
                                data_override="custom",
                                file_override=files[(dataset, tag)],
                            )
                        )
    return runs


def ensemble_members(datasets: list[str], horizons: list[int], seeds: list[int]) -> list[Run]:
    runs = []
    for dataset in datasets:
        for horizon in horizons:
            for seed in seeds:
                runs.append(Run("patchtst_output_ensemble", dataset, "PatchTST", horizon, seed, "member_a"))
                runs.append(Run("patchtst_output_ensemble", dataset, "PatchTST", horizon, seed + 1000, "member_b"))
    return runs


def build_plan(args: argparse.Namespace) -> list[Run]:
    datasets = parse_csv(args.datasets)
    horizons = parse_csv(args.horizons, int)
    seeds = parse_csv(args.seeds, int)
    assert_known(datasets)
    if args.campaign == "projected-match":
        return projected_match(datasets, horizons, seeds)
    if args.campaign == "ten-seed":
        return ten_seed(datasets, horizons, seeds)
    if args.campaign == "lookback":
        return lookback(datasets, horizons, seeds, parse_csv(args.lookbacks, int))
    if args.campaign == "rolling-origin":
        return rolling_origin(datasets, horizons, seeds, parse_csv(args.origins, float))
    if args.campaign == "ensemble-members":
        return ensemble_members(datasets, horizons, seeds)
    raise SystemExit(f"Unknown campaign: {args.campaign}")


def write_plan(path: Path, runs: list[Run], python_exe: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(asdict(runs[0]).keys()) + ["model_id", "command"] if runs else ["model_id", "command"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = asdict(run)
            row["model_id"] = run.model_id
            row["command"] = json.dumps(command_for(run, python_exe))
            writer.writerow(row)


def append_status(path: Path, row: dict[str, object]) -> None:
    fields = [
        "timestamp", "campaign", "dataset", "model", "horizon", "seed", "tag",
        "status", "returncode", "elapsed_seconds", "model_id", "metrics_path",
        "stdout_log", "stderr_log", "run_by_gpu_watermark",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def execute(runs: list[Run], python_exe: str, output: Path, args: argparse.Namespace) -> int:
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    write_plan(output / "plan.csv", runs, python_exe)
    (output / "manifest.json").write_text(
        json.dumps({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "arguments": vars(args), "runs": len(runs), "run_by_gpu_watermark": WATERMARK}, indent=2),
        encoding="utf-8",
    )
    status_path = output / "status.csv"
    executed = 0
    for index, run in enumerate(runs, start=1):
        strict = getattr(args, "require_rtx4090", False)
        existing = metrics_path(run, strict=strict)
        if args.skip_completed and existing is not None:
            print(f"[{index}/{len(runs)}] SKIP {run.model_id}")
            append_status(status_path, {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "campaign": run.campaign,
                "dataset": run.dataset, "model": run.model, "horizon": run.horizon, "seed": run.seed,
                "tag": run.tag, "status": "skipped_completed", "returncode": 0, "elapsed_seconds": 0,
                "model_id": run.model_id, "metrics_path": str(existing), "stdout_log": "", "stderr_log": "",
                "run_by_gpu_watermark": WATERMARK,
            })
            continue
        if args.max_runs and executed >= args.max_runs:
            break
        cmd = command_for(run, python_exe)
        print(f"[{index}/{len(runs)}] {'DRY' if args.dry_run else 'RUN'} {run.model_id}")
        if args.dry_run:
            print(" ".join(cmd))
            executed += 1
            continue
        if strict:
            import torch
            if not torch.cuda.is_available() or "RTX 4090" not in torch.cuda.get_device_name(0):
                raise RuntimeError("RTX4090 CUDA execution is required; CPU fallback is not allowed.")
        minimum_free_gb = getattr(args, "min_free_gb", 0)
        if shutil.disk_usage(ROOT).free < minimum_free_gb * 1024 ** 3:
            raise RuntimeError(f"Stopping before training: less than {minimum_free_gb} GiB free on the result drive.")
        stdout = logs / f"{run.model_id}.stdout.log"
        stderr = logs / f"{run.model_id}.stderr.log"
        started = time.time()
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=err, text=True)
        metric = metrics_path(run, strict=strict) if proc.returncode == 0 else None
        status = "completed" if proc.returncode == 0 and metric else "failed"
        append_status(status_path, {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "campaign": run.campaign,
            "dataset": run.dataset, "model": run.model, "horizon": run.horizon, "seed": run.seed,
            "tag": run.tag, "status": status, "returncode": proc.returncode,
            "elapsed_seconds": round(time.time() - started, 3), "model_id": run.model_id,
            "metrics_path": "" if metric is None else str(metric), "stdout_log": str(stdout),
            "stderr_log": str(stderr), "run_by_gpu_watermark": WATERMARK,
        })
        executed += 1
        if status != "completed":
            print(f"FAILED: inspect {stderr}")
            return proc.returncode or 1
    print(f"Planned {len(runs)} runs; executed {executed}. Artifacts: {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", choices=("projected-match", "ten-seed", "lookback", "rolling-origin", "ensemble-members"))
    parser.add_argument("--datasets", default="ETTh1,ETTh2,ETTm1,ETTm2,Exchange,Weather")
    parser.add_argument("--horizons", default="96,192,336")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS_5)))
    parser.add_argument("--lookbacks", default="96,192,336,720")
    parser.add_argument("--origins", default="0.60,0.80,1.00")
    parser.add_argument("--python", default="")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    args = parser.parse_args()
    if args.campaign == "ten-seed" and args.seeds == ",".join(map(str, DEFAULT_SEEDS_5)):
        args.seeds = ",".join(map(str, DEFAULT_SEEDS_10))
    runs = build_plan(args)
    return execute(runs, resolve_python(args.python), OUT_ROOT / args.campaign, args)


if __name__ == "__main__":
    raise SystemExit(main())
