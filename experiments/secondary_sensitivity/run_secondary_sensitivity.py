#!/usr/bin/env python
"""Secondary heads/d_ff/dropout sensitivity after targeted P/S/LR selection.

Selection: 3 models x 3 datasets x 6 single-factor alternatives = 54 runs.
Selection uses seed 2020 and validation MSE only. The test split is disabled.

Final: at most one selected single-factor candidate per model/dataset is tested
with seeds 2021--2025. Selection requires at least a 1% validation-MSE
improvement over that cell's locked P*/S*/LR* baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


CAMPAIGN = "SecondarySensitivity"
MODELS = ("PatchTST", "PFB-Direct", "PFB-Projected")
DATASETS = {
    "ETTh2": {"data": "ETTh2", "data_path": "ETTh2.csv", "enc_in": 7, "batch_size": 16, "freq": "h"},
    "ETTm2": {"data": "ETTm2", "data_path": "ETTm2.csv", "enc_in": 7, "batch_size": 16, "freq": "t"},
    "Weather": {"data": "custom", "data_path": "weather.csv", "enc_in": 21, "batch_size": 8, "freq": "h"},
}
HORIZON = 192
SELECTION_SEED = 2020
EVALUATION_SEEDS = (2021, 2022, 2023, 2024, 2025)
MIN_IMPROVEMENT_PERCENT = 1.0

SCRIPT_DIR = Path(__file__).resolve().parent
TS_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = TS_ROOT
TARGETED_DIR = WORKSPACE_ROOT / "experiments" / "targeted_hpo"
LOCKED_HPO_PATH = TARGETED_DIR / "selections" / "locked_configurations.csv"
VALIDATION_DIR = SCRIPT_DIR / "validation"
LOGS_DIR = SCRIPT_DIR / "logs"
SELECTION_DIR = SCRIPT_DIR / "selections"
TABLES_DIR = SCRIPT_DIR / "tables"
LOCKED_SECONDARY_PATH = SELECTION_DIR / "locked_secondary_configurations.csv"


@dataclass(frozen=True)
class RunSpec:
    stage: str
    model: str
    dataset: str
    seed: int
    factor: str
    value: str
    params: dict

    @property
    def model_tag(self) -> str:
        return self.model.replace("-", "")

    @property
    def run_id(self) -> str:
        value_tag = self.value.replace(".", "p").replace("-", "m")
        return (
            f"{CAMPAIGN}_{self.stage}_{self.model_tag}_{self.dataset}_H{HORIZON}_"
            f"{self.factor}_{value_tag}_seed{self.seed}"
        )

    @property
    def validation_path(self) -> Path:
        return VALIDATION_DIR / f"{self.run_id}.json"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def locked_hpo_rows() -> list[dict]:
    rows = read_csv(LOCKED_HPO_PATH)
    expected = {(dataset, model) for dataset in DATASETS for model in MODELS}
    actual = {(row["dataset"], row["model"]) for row in rows}
    if actual != expected:
        raise SystemExit(
            f"Locked HPO grid mismatch. Missing={sorted(expected - actual)}; extra={sorted(actual - expected)}"
        )
    return rows


def baseline_params(row: dict) -> dict:
    return {
        "seq_len": 336,
        "label_len": 96,
        "patch_len": int(row["patch_len"]),
        "stride": int(row["stride"]),
        "learning_rate": float(row["learning_rate"]),
        "d_model": 128,
        "d_ff": 512,
        "n_heads": 8,
        "dropout": 0.1,
        "batch_size": int(DATASETS[row["dataset"]]["batch_size"]),
        "optimizer": "Adam",
        "weight_decay": 0.0,
    }


def alternatives() -> list[tuple[str, str, dict]]:
    return [
        ("n_heads", "4", {"n_heads": 4}),
        ("n_heads", "16", {"n_heads": 16}),
        ("d_ff", "256", {"d_ff": 256}),
        ("d_ff", "1024", {"d_ff": 1024}),
        ("dropout", "0.0", {"dropout": 0.0}),
        ("dropout", "0.2", {"dropout": 0.2}),
    ]


def selection_plan() -> list[RunSpec]:
    plan = []
    for row in locked_hpo_rows():
        base = baseline_params(row)
        for factor, value, change in alternatives():
            params = dict(base)
            params.update(change)
            plan.append(
                RunSpec("selection", row["model"], row["dataset"], SELECTION_SEED, factor, value, params)
            )
    return plan


def choose_secondary(require_complete: bool = True) -> list[dict]:
    plan = selection_plan()
    selections = []
    for row in locked_hpo_rows():
        dataset, model = row["dataset"], row["model"]
        baseline_path = Path(row["validation_path"])
        if not baseline_path.exists():
            raise SystemExit(f"Missing tuned-baseline validation record: {baseline_path}")
        baseline_record = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_mse = float(baseline_record["best_validation_mse"])
        candidates = []
        for spec in [s for s in plan if s.dataset == dataset and s.model == model]:
            if not spec.validation_path.exists():
                if require_complete:
                    raise SystemExit(f"Selection stage is missing required validation record: {spec.validation_path}")
                continue
            record = json.loads(spec.validation_path.read_text(encoding="utf-8"))
            candidates.append((float(record["best_validation_mse"]), spec, record))
        if not candidates:
            continue
        best_mse, best_spec, best_record = min(candidates, key=lambda item: (item[0], item[1].factor, item[1].value))
        improvement = 100.0 * (baseline_mse - best_mse) / baseline_mse
        selected = improvement >= MIN_IMPROVEMENT_PERCENT
        selections.append(
            {
                "dataset": dataset,
                "model": model,
                "status": "secondary_selected" if selected else "tuned_baseline_retained",
                "threshold_percent": MIN_IMPROVEMENT_PERCENT,
                "baseline_validation_mse": baseline_mse,
                "best_secondary_validation_mse": best_mse,
                "relative_improvement_percent": improvement,
                "selected_factor": best_spec.factor if selected else "",
                "selected_value": best_spec.value if selected else "",
                "selected_params_json": json.dumps(best_spec.params, sort_keys=True) if selected else "",
                "best_validation_epoch": int(best_record["best_validation_epoch"]),
                "source_run_id": best_spec.run_id,
                "validation_path": str(best_spec.validation_path),
                "tuned_patch_len": int(row["patch_len"]),
                "tuned_stride": int(row["stride"]),
                "tuned_learning_rate": float(row["learning_rate"]),
            }
        )
    return selections


def persist_selection_if_complete(plan: list[RunSpec]) -> None:
    if all(spec.validation_path.exists() for spec in plan):
        write_rows(LOCKED_SECONDARY_PATH, choose_secondary(require_complete=True))


def final_plan() -> list[RunSpec]:
    if not LOCKED_SECONDARY_PATH.exists():
        raise SystemExit("Secondary configurations are not locked. Complete --stage selection first.")
    plan = []
    for row in read_csv(LOCKED_SECONDARY_PATH):
        if row["status"] != "secondary_selected":
            continue
        params = json.loads(row["selected_params_json"])
        for seed in EVALUATION_SEEDS:
            plan.append(
                RunSpec(
                    "final", row["model"], row["dataset"], seed,
                    row["selected_factor"], row["selected_value"], params,
                )
            )
    return plan


def python_has_dependencies(python_exe: str) -> bool:
    try:
        proc = subprocess.run(
            [python_exe, "-c", "import torch, numpy"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            text=True,
        )
        return proc.returncode == 0
    except Exception:
        return False


def resolve_python(requested: str | None) -> str:
    candidates = []
    if requested:
        candidates.append(requested)
    candidates.extend([
        r"C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe",
        sys.executable,
        str(WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"),
        "python.exe",
    ])
    checked, seen = [], set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        checked.append(candidate)
        if python_has_dependencies(candidate):
            return candidate
    raise SystemExit("No Python interpreter with torch and numpy found. Checked:\n  - " + "\n  - ".join(checked))


def command_for(spec: RunSpec, python_exe: str) -> list[str]:
    cfg, p = DATASETS[spec.dataset], spec.params
    command = [
        python_exe, "-u", "run.py",
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--model_id", spec.run_id, "--model", spec.model,
        "--data", str(cfg["data"]), "--root_path", "./data/", "--data_path", str(cfg["data_path"]),
        "--features", "M", "--freq", str(cfg["freq"]),
        "--seq_len", str(p["seq_len"]), "--label_len", str(p["label_len"]), "--pred_len", str(HORIZON),
        "--enc_in", str(cfg["enc_in"]), "--dec_in", str(cfg["enc_in"]), "--c_out", str(cfg["enc_in"]),
        "--patch_len", str(p["patch_len"]), "--stride", str(p["stride"]),
        "--d_model", str(p["d_model"]), "--d_ff", str(p["d_ff"]),
        "--e_layers", "3", "--d_layers", "1", "--n_heads", str(p["n_heads"]), "--factor", "3",
        "--dropout", str(p["dropout"]), "--embed", "timeF", "--num_workers", "0", "--itr", "1",
        "--train_epochs", "100", "--patience", "10", "--learning_rate", str(p["learning_rate"]),
        "--optimizer", str(p.get("optimizer", "Adam")), "--weight_decay", str(p.get("weight_decay", 0.0)),
        "--batch_size", str(p["batch_size"]), "--des", "secondary_sensitivity", "--seed", str(spec.seed),
    ]
    if spec.stage == "selection":
        command += ["--validation_only", "--validation_output", str(spec.validation_path)]
    else:
        command += ["--defer_test_until_after_training"]
    return command


def result_metrics_path(spec: RunSpec) -> Path | None:
    result_root = TS_ROOT / "results"
    if not result_root.exists():
        return None
    for result_dir in result_root.glob(f"long_term_forecast_{spec.run_id}_*"):
        metrics = result_dir / "metrics.npy"
        if metrics.exists():
            return metrics
    return None


def targeted_baseline_metrics_path(row: dict, seed: int) -> Path | None:
    model_tag = row["model"].replace("-", "")
    lr_tag = f"{float(row['learning_rate']):.8g}".replace(".", "p").replace("-", "m")
    run_id = (
        f"TargetedPatchLRSensitivity_final_{model_tag}_{row['dataset']}_H{HORIZON}_"
        f"P{int(row['patch_len'])}_S{int(row['stride'])}_LR{lr_tag}_seed{seed}"
    )
    for result_dir in (TS_ROOT / "results").glob(f"long_term_forecast_{run_id}_*"):
        metrics = result_dir / "metrics.npy"
        if metrics.exists():
            return metrics
    return None


def write_plan(stage: str, plan: list[RunSpec], python_exe: str) -> Path:
    path = SCRIPT_DIR / f"secondary_sensitivity_{stage}_plan.csv"
    rows = [{
        "stage": stage, "model": s.model, "dataset": s.dataset, "horizon": HORIZON,
        "seed": s.seed, "factor": s.factor, "value": s.value,
        "params_json": json.dumps(s.params, sort_keys=True), "run_id": s.run_id,
        "command_json": json.dumps(command_for(s, python_exe)),
    } for s in plan]
    if rows:
        write_rows(path, rows)
    else:
        path.write_text("stage,model,dataset,horizon,seed,factor,value,params_json,run_id,command_json\n", encoding="utf-8")
    return path


def append_status(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_combined_summary() -> None:
    selections = read_csv(LOCKED_SECONDARY_PATH)
    locked = {(row["dataset"], row["model"]): row for row in locked_hpo_rows()}
    secondary = {(s.dataset, s.model, s.seed): s for s in final_plan()}
    rows = []
    for selection in selections:
        cell = (selection["dataset"], selection["model"])
        for seed in EVALUATION_SEEDS:
            baseline_path = targeted_baseline_metrics_path(locked[cell], seed)
            if baseline_path is None:
                return
            import numpy as np
            metrics = np.load(baseline_path)
            rows.append({"dataset": cell[0], "model": cell[1], "configuration": "tuned_baseline",
                         "seed": seed, "mse": float(metrics[1]), "mae": float(metrics[0]),
                         "metrics_path": str(baseline_path)})
            if selection["status"] == "secondary_selected":
                spec = secondary.get((cell[0], cell[1], seed))
                path = result_metrics_path(spec) if spec else None
                if path is None:
                    return
                metrics = np.load(path)
                rows.append({"dataset": cell[0], "model": cell[1], "configuration": "selected_secondary",
                             "seed": seed, "mse": float(metrics[1]), "mae": float(metrics[0]),
                             "metrics_path": str(path)})
    write_rows(TABLES_DIR / "combined_tuning_sensitivity_seed_metrics.csv", rows)
    summary = []
    for dataset in DATASETS:
        for model in MODELS:
            for config in ("tuned_baseline", "selected_secondary"):
                cell_rows = [r for r in rows if r["dataset"] == dataset and r["model"] == model and r["configuration"] == config]
                if not cell_rows:
                    continue
                import numpy as np
                summary.append({"dataset": dataset, "model": model, "configuration": config,
                                "n_seeds": len(cell_rows),
                                "mse_mean": float(np.mean([r["mse"] for r in cell_rows])),
                                "mse_std": float(np.std([r["mse"] for r in cell_rows], ddof=1)),
                                "mae_mean": float(np.mean([r["mae"] for r in cell_rows])),
                                "mae_std": float(np.std([r["mae"] for r in cell_rows], ddof=1))})
    write_rows(TABLES_DIR / "combined_tuning_sensitivity_summary.csv", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description="Secondary hyperparameter sensitivity runner")
    parser.add_argument("--stage", required=True, choices=("selection", "final"))
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="maximum new runs; 0 means all remaining")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default="")
    args = parser.parse_args()

    if not (TS_ROOT / "run.py").exists():
        raise SystemExit(f"Missing training entry point: {TS_ROOT / 'run.py'}")
    for cfg in DATASETS.values():
        data_path = TS_ROOT / "data" / str(cfg["data_path"])
        if not data_path.exists():
            raise SystemExit(f"Missing dataset: {data_path}")
    python_exe = resolve_python(args.python or None)
    plan = selection_plan() if args.stage == "selection" else final_plan()
    for directory in (VALIDATION_DIR, LOGS_DIR, SELECTION_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    plan_path = write_plan(args.stage, plan, python_exe)
    status_path = SCRIPT_DIR / f"secondary_sensitivity_{args.stage}_status.csv"
    (SCRIPT_DIR / f"secondary_sensitivity_{args.stage}_manifest.json").write_text(
        json.dumps({"campaign": CAMPAIGN, "stage": args.stage, "planned_runs": len(plan),
                    "selection_seed": SELECTION_SEED, "evaluation_seeds": EVALUATION_SEEDS,
                    "minimum_improvement_percent": MIN_IMPROVEMENT_PERCENT,
                    "test_evaluated": args.stage == "final", "python_executable": python_exe,
                    "plan_path": str(plan_path), "status_path": str(status_path)}, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"SECONDARY HYPERPARAMETER SENSITIVITY: {args.stage}")
    print("=" * 78)
    print(f"Plan: {plan_path}")
    print(f"Runs planned: {len(plan)}")
    print(f"Test evaluation: {'once after training' if args.stage == 'final' else 'disabled'}")
    print("=" * 78)
    executed = skipped = 0
    for index, spec in enumerate(plan, start=1):
        completed = spec.validation_path if spec.stage == "selection" else result_metrics_path(spec)
        if args.skip_completed and completed is not None and Path(completed).exists():
            skipped += 1
            print(f"[{index}/{len(plan)}] SKIP {spec.model} {spec.dataset} {spec.factor}={spec.value}")
            continue
        if args.max_runs and executed >= args.max_runs:
            print(f"Reached --max-runs={args.max_runs}.")
            break
        command = command_for(spec, python_exe)
        if args.dry_run:
            print(f"[{index}/{len(plan)}] DRY {spec.run_id}")
            print(subprocess.list2cmdline(command))
            executed += 1
            continue
        stdout_path, stderr_path = LOGS_DIR / f"{spec.run_id}.stdout.log", LOGS_DIR / f"{spec.run_id}.stderr.log"
        print(f"[{index}/{len(plan)}] RUN {spec.model} {spec.dataset} {spec.factor}={spec.value}")
        started = time.time()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(command, cwd=TS_ROOT, stdout=stdout, stderr=stderr, text=True)
        elapsed = time.time() - started
        output = spec.validation_path if spec.stage == "selection" else result_metrics_path(spec)
        status = "completed" if proc.returncode == 0 and output is not None and Path(output).exists() else "failed"
        append_status(status_path, {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": spec.stage,
                      "dataset": spec.dataset, "model": spec.model, "horizon": HORIZON, "seed": spec.seed,
                      "factor": spec.factor, "value": spec.value, "status": status, "returncode": proc.returncode,
                      "elapsed_seconds": round(elapsed, 3), "run_id": spec.run_id,
                      "output_path": "" if output is None else str(output),
                      "stdout_log": str(stdout_path), "stderr_log": str(stderr_path)})
        executed += 1
        if status != "completed":
            print(f"FAILED: see {stderr_path}")
            return proc.returncode or 1
    if args.stage == "selection":
        persist_selection_if_complete(plan)
    else:
        write_combined_summary()
    print("=" * 78)
    print(f"Done. New/dry runs: {executed}; skipped: {skipped}")
    print(f"Status: {status_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
