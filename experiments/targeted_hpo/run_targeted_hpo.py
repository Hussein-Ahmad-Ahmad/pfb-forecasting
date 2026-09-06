#!/usr/bin/env python
"""Run the targeted PatchTST-family patch/LR sensitivity study.

Protocol
--------
Selection uses seed 2020 and validation MSE only.

Stage 1:
    3 models x 3 datasets x 3 patch/stride pairs = 27 runs
    (P, S) in {(8, 4), (16, 8), (32, 16)}, LR = 1e-4

Stage 2:
    3 models x 3 datasets x 3 additional learning rates = 27 runs
    selected (P, S), LR in {5e-5, 1.5e-4, 2e-4}

Final:
    3 models x 3 datasets x 5 evaluation seeds = 45 runs
    selected configuration locked before test evaluation

The three stages total 99 runs. Selection stages do not load or evaluate the
test split. Plans, validation records, logs, selections, and final summaries
are stored beside this script.
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


CAMPAIGN = "TargetedPatchLRSensitivity"
MODELS = ("PatchTST", "PFB-Direct", "PFB-Projected")
DATASETS = {
    "ETTh2": {"data": "ETTh2", "data_path": "ETTh2.csv", "enc_in": 7, "batch_size": 16, "freq": "h"},
    "ETTm2": {"data": "ETTm2", "data_path": "ETTm2.csv", "enc_in": 7, "batch_size": 16, "freq": "t"},
    "Weather": {"data": "custom", "data_path": "weather.csv", "enc_in": 21, "batch_size": 8, "freq": "h"},
}
HORIZON = 192
TUNING_SEED = 2020
EVALUATION_SEEDS = (2021, 2022, 2023, 2024, 2025)
BASE_LR = 1e-4
PATCH_GRID = ((8, 4), (16, 8), (32, 16))
ADDITIONAL_LRS = (5e-5, 1.5e-4, 2e-4)

SCRIPT_DIR = Path(__file__).resolve().parent
TS_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = TS_ROOT
VALIDATION_DIR = SCRIPT_DIR / "validation"
LOGS_DIR = SCRIPT_DIR / "logs"
SELECTION_DIR = SCRIPT_DIR / "selections"
TABLES_DIR = SCRIPT_DIR / "tables"


@dataclass(frozen=True)
class RunSpec:
    stage: str
    model: str
    dataset: str
    seed: int
    patch_len: int
    stride: int
    learning_rate: float

    @property
    def lr_tag(self) -> str:
        return f"{self.learning_rate:.8g}".replace(".", "p").replace("-", "m")

    @property
    def model_tag(self) -> str:
        return self.model.replace("-", "")

    @property
    def run_id(self) -> str:
        return (
            f"{CAMPAIGN}_{self.stage}_{self.model_tag}_{self.dataset}_H{HORIZON}_"
            f"P{self.patch_len}_S{self.stride}_LR{self.lr_tag}_seed{self.seed}"
        )

    @property
    def validation_path(self) -> Path:
        return VALIDATION_DIR / f"{self.run_id}.json"


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
    candidates.extend(
        [
            # Prefer the local environment used for the forecasting runs.
            r"C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe",
            sys.executable,
            str(WORKSPACE_ROOT / ".venv" / "Scripts" / "python.exe"),
            "python.exe",
        ]
    )
    checked = []
    seen = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        checked.append(candidate)
        if python_has_dependencies(candidate):
            return candidate
    raise SystemExit("No Python interpreter with torch and numpy found. Checked:\n  - " + "\n  - ".join(checked))


def stage1_plan() -> list[RunSpec]:
    return [
        RunSpec("stage1", model, dataset, TUNING_SEED, patch_len, stride, BASE_LR)
        for dataset in DATASETS
        for model in MODELS
        for patch_len, stride in PATCH_GRID
    ]


def read_validation(spec: RunSpec) -> dict:
    if not spec.validation_path.exists():
        raise SystemExit(f"Missing validation record: {spec.validation_path}")
    record = json.loads(spec.validation_path.read_text(encoding="utf-8"))
    value = record.get("best_validation_mse")
    if value is None:
        raise SystemExit(f"Validation MSE is absent from: {spec.validation_path}")
    return record


def choose_stage1(require_complete: bool = True) -> list[dict]:
    selected = []
    for dataset in DATASETS:
        for model in MODELS:
            candidates = [s for s in stage1_plan() if s.dataset == dataset and s.model == model]
            available = []
            for spec in candidates:
                if not spec.validation_path.exists():
                    if require_complete:
                        raise SystemExit(
                            "Stage 1 is missing required validation records. Run stage1 with --skip-completed before stage2.\n"
                            f"Missing: {spec.validation_path}"
                        )
                    continue
                record = read_validation(spec)
                available.append((float(record["best_validation_mse"]), spec, record))
            if not available:
                continue
            _, spec, record = min(
                available,
                key=lambda item: (item[0], abs(item[1].patch_len - 16), item[1].patch_len),
            )
            selected.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "patch_len": spec.patch_len,
                    "stride": spec.stride,
                    "learning_rate": spec.learning_rate,
                    "best_validation_mse": float(record["best_validation_mse"]),
                    "best_validation_epoch": int(record["best_validation_epoch"]),
                    "source_stage": "stage1",
                    "source_run_id": spec.run_id,
                    "validation_path": str(spec.validation_path),
                }
            )
    return selected


def stage2_plan() -> list[RunSpec]:
    selected = choose_stage1(require_complete=True)
    return [
        RunSpec(
            "stage2",
            row["model"],
            row["dataset"],
            TUNING_SEED,
            int(row["patch_len"]),
            int(row["stride"]),
            learning_rate,
        )
        for row in selected
        for learning_rate in ADDITIONAL_LRS
    ]


def choose_final(require_complete: bool = True) -> list[dict]:
    stage1_selected = choose_stage1(require_complete=require_complete)
    stage1_by_cell = {(row["dataset"], row["model"]): row for row in stage1_selected}
    stage2_specs = stage2_plan() if require_complete or len(stage1_selected) == 9 else []
    selected = []
    for dataset in DATASETS:
        for model in MODELS:
            cell = (dataset, model)
            if cell not in stage1_by_cell:
                continue
            base = stage1_by_cell[cell]
            candidates = [
                (
                    float(base["best_validation_mse"]),
                    float(base["learning_rate"]),
                    int(base["best_validation_epoch"]),
                    "stage1",
                    base["source_run_id"],
                    base["validation_path"],
                )
            ]
            for spec in [s for s in stage2_specs if s.dataset == dataset and s.model == model]:
                if not spec.validation_path.exists():
                    if require_complete:
                        raise SystemExit(
                            "Stage 2 is missing required validation records. Run stage2 with --skip-completed before final.\n"
                            f"Missing: {spec.validation_path}"
                        )
                    continue
                record = read_validation(spec)
                candidates.append(
                    (
                        float(record["best_validation_mse"]),
                        spec.learning_rate,
                        int(record["best_validation_epoch"]),
                        "stage2",
                        spec.run_id,
                        str(spec.validation_path),
                    )
                )
            if require_complete and len(candidates) != 4:
                raise SystemExit(f"Expected four LR candidates for {dataset}/{model}; found {len(candidates)}")
            best = min(candidates, key=lambda item: (item[0], abs(item[1] - BASE_LR), item[1]))
            selected.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "patch_len": int(base["patch_len"]),
                    "stride": int(base["stride"]),
                    "learning_rate": best[1],
                    "best_validation_mse": best[0],
                    "best_validation_epoch": best[2],
                    "source_stage": best[3],
                    "source_run_id": best[4],
                    "validation_path": best[5],
                }
            )
    return selected


def final_plan() -> list[RunSpec]:
    selected = choose_final(require_complete=True)
    return [
        RunSpec(
            "final",
            row["model"],
            row["dataset"],
            seed,
            int(row["patch_len"]),
            int(row["stride"]),
            float(row["learning_rate"]),
        )
        for row in selected
        for seed in EVALUATION_SEEDS
    ]


def command_for(spec: RunSpec, python_exe: str) -> list[str]:
    cfg = DATASETS[spec.dataset]
    command = [
        python_exe,
        "-u",
        "run.py",
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model_id",
        spec.run_id,
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
        "96",
        "--pred_len",
        str(HORIZON),
        "--enc_in",
        str(cfg["enc_in"]),
        "--dec_in",
        str(cfg["enc_in"]),
        "--c_out",
        str(cfg["enc_in"]),
        "--patch_len",
        str(spec.patch_len),
        "--stride",
        str(spec.stride),
        "--d_model",
        "128",
        "--d_ff",
        "512",
        "--e_layers",
        "3",
        "--d_layers",
        "1",
        "--n_heads",
        "8",
        "--factor",
        "3",
        "--dropout",
        "0.1",
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
        f"{spec.learning_rate:.8g}",
        "--batch_size",
        str(cfg["batch_size"]),
        "--des",
        "targeted_patch_lr",
        "--seed",
        str(spec.seed),
    ]
    if spec.stage in {"stage1", "stage2"}:
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


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plan(stage: str, plan: list[RunSpec], python_exe: str) -> Path:
    rows = []
    for spec in plan:
        rows.append(
            {
                "stage": stage,
                "model": spec.model,
                "dataset": spec.dataset,
                "horizon": HORIZON,
                "seed": spec.seed,
                "patch_len": spec.patch_len,
                "stride": spec.stride,
                "learning_rate": f"{spec.learning_rate:.8g}",
                "run_id": spec.run_id,
                "selection_metric": "validation_mse" if stage != "final" else "locked_configuration",
                "command_json": json.dumps(command_for(spec, python_exe)),
            }
        )
    path = SCRIPT_DIR / f"targeted_hpo_{stage}_plan.csv"
    write_rows(path, rows)
    return path


def append_status(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def persist_selections() -> None:
    stage1_specs = stage1_plan()
    if not all(spec.validation_path.exists() for spec in stage1_specs):
        return
    stage1 = choose_stage1(require_complete=True)
    write_rows(SELECTION_DIR / "stage1_patch_selection.csv", stage1)

    stage2_specs = stage2_plan()
    if not all(spec.validation_path.exists() for spec in stage2_specs):
        return
    final = choose_final(require_complete=True)
    write_rows(SELECTION_DIR / "locked_configurations.csv", final)


def summarize_final(plan: list[RunSpec]) -> None:
    metric_rows = []
    for spec in plan:
        path = result_metrics_path(spec)
        if path is None:
            return
        import numpy as np

        values = np.load(path)
        metric_rows.append(
            {
                "dataset": spec.dataset,
                "horizon": HORIZON,
                "model": spec.model,
                "seed": spec.seed,
                "patch_len": spec.patch_len,
                "stride": spec.stride,
                "learning_rate": f"{spec.learning_rate:.8g}",
                "mae": float(values[0]),
                "mse": float(values[1]),
                "metrics_path": str(path),
            }
        )
    write_rows(TABLES_DIR / "targeted_hpo_final_seed_metrics.csv", metric_rows)

    summary = []
    for dataset in DATASETS:
        for model in MODELS:
            cell = [r for r in metric_rows if r["dataset"] == dataset and r["model"] == model]
            if not cell:
                continue
            import numpy as np

            summary.append(
                {
                    "dataset": dataset,
                    "horizon": HORIZON,
                    "model": model,
                    "n_seeds": len(cell),
                    "patch_len": cell[0]["patch_len"],
                    "stride": cell[0]["stride"],
                    "learning_rate": cell[0]["learning_rate"],
                    "mse_mean": float(np.mean([r["mse"] for r in cell])),
                    "mse_std": float(np.std([r["mse"] for r in cell], ddof=1)),
                    "mae_mean": float(np.mean([r["mae"] for r in cell])),
                    "mae_std": float(np.std([r["mae"] for r in cell], ddof=1)),
                }
            )
    write_rows(TABLES_DIR / "targeted_hpo_final_summary.csv", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted PatchTST-family patch/LR sensitivity runner")
    parser.add_argument("--stage", required=True, choices=("stage1", "stage2", "final"))
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
    if args.stage == "stage1":
        plan = stage1_plan()
    elif args.stage == "stage2":
        plan = stage2_plan()
    else:
        secondary_lock = (
            WORKSPACE_ROOT / "experiments" / "secondary_sensitivity"
            / "selections" / "locked_secondary_configurations.csv"
        )
        if not secondary_lock.exists():
            raise SystemExit(
                "Run and lock secondary sensitivity selection before final test evaluation:\n"
                "  python .\\experiments\\secondary_sensitivity\\run_secondary_sensitivity.py "
                "--stage selection --skip-completed"
            )
        plan = final_plan()

    for directory in (VALIDATION_DIR, LOGS_DIR, SELECTION_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    plan_path = write_plan(args.stage, plan, python_exe)
    status_path = SCRIPT_DIR / f"targeted_hpo_{args.stage}_status.csv"
    manifest_path = SCRIPT_DIR / f"targeted_hpo_{args.stage}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "campaign": CAMPAIGN,
                "stage": args.stage,
                "selection_metric": "validation_mse",
                "test_evaluated": args.stage == "final",
                "tuning_seed": TUNING_SEED,
                "evaluation_seeds": EVALUATION_SEEDS,
                "models": MODELS,
                "datasets": list(DATASETS),
                "horizon": HORIZON,
                "planned_runs": len(plan),
                "python_executable": python_exe,
                "plan_path": str(plan_path),
                "status_path": str(status_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print(f"TARGETED PATCH/LR SENSITIVITY: {args.stage}")
    print("=" * 78)
    print(f"Plan: {plan_path}")
    print(f"Runs planned in stage: {len(plan)}")
    print(f"Python: {python_exe}")
    print(f"Test evaluation: {'enabled' if args.stage == 'final' else 'disabled'}")
    print("=" * 78)

    executed = 0
    skipped = 0
    for index, spec in enumerate(plan, start=1):
        completed_path = spec.validation_path if spec.stage != "final" else result_metrics_path(spec)
        if args.skip_completed and completed_path is not None and Path(completed_path).exists():
            skipped += 1
            print(f"[{index}/{len(plan)}] SKIP {spec.model} {spec.dataset} seed={spec.seed}")
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

        stdout_path = LOGS_DIR / f"{spec.run_id}.stdout.log"
        stderr_path = LOGS_DIR / f"{spec.run_id}.stderr.log"
        print(f"[{index}/{len(plan)}] RUN {spec.model} {spec.dataset} seed={spec.seed}")
        started = time.time()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(command, cwd=TS_ROOT, stdout=stdout, stderr=stderr, text=True)
        elapsed = time.time() - started
        output_path = spec.validation_path if spec.stage != "final" else result_metrics_path(spec)
        status = "completed" if proc.returncode == 0 and output_path is not None and Path(output_path).exists() else "failed"
        append_status(
            status_path,
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "stage": spec.stage,
                "dataset": spec.dataset,
                "model": spec.model,
                "horizon": HORIZON,
                "seed": spec.seed,
                "patch_len": spec.patch_len,
                "stride": spec.stride,
                "learning_rate": f"{spec.learning_rate:.8g}",
                "status": status,
                "returncode": proc.returncode,
                "elapsed_seconds": round(elapsed, 3),
                "run_id": spec.run_id,
                "output_path": "" if output_path is None else str(output_path),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            },
        )
        executed += 1
        if status != "completed":
            print(f"FAILED: see {stderr_path}")
            return proc.returncode or 1

    persist_selections()
    if args.stage == "final":
        summarize_final(plan)
    print("=" * 78)
    print(f"Done. New/dry runs: {executed}; skipped: {skipped}")
    print(f"Status: {status_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
