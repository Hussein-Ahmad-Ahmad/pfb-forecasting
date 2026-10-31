"""Plan first; execute only with --execute. Never edits a manuscript."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import design
from .design import ROOT, DATASETS, FAMILIES, SEEDS, digest, file_hash
from .storage import completed, write_csv, write_json


def split_items(value, allowed=None, cast=str):
    values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)) or (allowed is not None and not set(values) <= set(allowed)):
        raise ValueError("Empty, duplicate, or unknown selection")
    return values


def freeze_plan(output, action, datasets, seeds, tolerance=0.01, families=FAMILIES):
    import torch
    torch.set_num_threads(1)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    filename = output / "plan.json"
    # Resume uses the frozen plan, never a newly generated/mutated candidate set.
    if filename.exists():
        plan = json.loads(filename.read_text(encoding="utf-8"))
        if plan["scope"] != dict(datasets=datasets, seeds=seeds, families=list(families), tolerance=tolerance):
            raise ValueError("Requested scope differs from frozen plan; use a new output directory")
        if plan["action"] != action or plan["output"] != str(output):
            raise ValueError("Frozen action/output mismatch")
        from .worker import verify_environment
        for ds in datasets:
            verify_environment(plan, next(s for s in plan["runs"] if s["dataset"] == ds))
        return plan
    env = design.environment(datasets, seeds)
    metadata = {}
    if action == "projected":
        runs, metadata = design.projected_plan(env, datasets, seeds, tolerance)
    elif action == "preprocessing":
        runs = design.preprocessing_plan(env, datasets, seeds)
        metadata = dict(intervention="Exactly one reversible window normalization ON/OFF; learned normalizations unchanged")
    elif action == "hpo-select":
        runs = [s for s in design.hpo_plan(env, datasets) if s["family"] in families]
        metadata = dict(unique_candidates_per_family=12, selection_seed=2020,
                        note="Equal distinct-configuration count, NOT equal GPU-hours or guaranteed optimal tuning")
    else:
        raise ValueError(action)
    plan = dict(schema=1, action=action, output=str(output), environment=env,
                scope=dict(datasets=datasets, seeds=seeds, families=list(families), tolerance=tolerance),
                metadata=metadata, runs=runs)
    if len({s["run_id"] for s in runs}) != len(runs):
        raise ValueError("Duplicate planned run identity")
    write_json(filename, plan, immutable=True)
    write_csv(output / "plan.csv", runs)
    if metadata.get("profiles"):
        write_csv(output / "parameter_mac_profiles.csv", metadata["profiles"])
    if action == "hpo-select":
        audit_legacy_candidates(output, runs)
    return plan


def audit_legacy_candidates(output, runs):
    """Do not silently reuse historical trials lacking the new protocol identity.

    Historical validation averaged batch MSEs, whereas the prospective study
    uses element-weighted validation MSE and a separate data-shuffle generator.
    Without historical checkpoints and a full protocol audit, scalar metrics
    are not interchangeable. The old experiments remain valid and untouched.
    """
    rows = []
    from scripts.paper.run_review_campaign import grid_candidate
    for spec in runs:
        legacy = [i for i in range(6) if design.effective_params(spec["family"], grid_candidate(spec["family"], i)) == spec["params"]]
        rows.append(dict(dataset=spec["dataset"], family=spec["family"], candidate=spec["variant"],
                         legacy_candidate_ids=legacy, automatically_reused=False,
                         reason="New protocol/validation reduction fingerprint; reuse only exact verified completions of this prospective plan"))
    write_csv(Path(output) / "legacy_reuse_audit.csv", rows)


def lock_hpo(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if plan["action"] != "hpo-select":
        raise ValueError("Not an HPO selection plan")
    from .worker import verify_environment
    groups, verified_datasets = {}, set()
    for spec in plan["runs"]:
        design.validate_spec(spec)
        if spec["dataset"] not in verified_datasets:
            verify_environment(plan, spec)
            verified_datasets.add(spec["dataset"])
        result = completed(output, spec)
        if result is None:
            raise ValueError(f"Cannot lock incomplete validation search: {spec['run_id']}")
        if spec["split"] != "val" or spec["seed"] != 2020 or result["summary"]["test_ever_loaded"]:
            raise ValueError("HPO selection used an invalid split/seed or touched test data")
        groups.setdefault((spec["dataset"], spec["family"]), []).append((result["mse"], spec["run_id"], spec, result))
    locked = []
    for (dataset, family), entries in sorted(groups.items()):
        if len(entries) != 12 or len({design.canonical(e[2]["params"]) for e in entries}) != 12:
            raise ValueError("Lock requires 12 unique completed effective candidates per cell")
        mse, _, winner, result = min(entries, key=lambda e: (e[0], e[1]))
        locked.append(dict(dataset=dataset, family=family, params=winner["params"],
                           validation_mse=mse, selection_run_id=winner["run_id"],
                           selection_metric_sha256=result["sha256"]["metrics.npy"],
                           allocated_device_wall_hours=sum(e[3]["summary"]["allocated_device_wall_hours"] for e in entries)))
    value = dict(selection_plan_sha256=file_hash(output / "plan.json"), winners=locked)
    write_json(output / "locked_configs.json", value, immutable=True)
    print(f"Locked {len(locked)} configurations using validation ONLY.", flush=True)
    return value


def test_hpo_plan(output, seeds):
    output = Path(output)
    selection = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    if seeds != selection["scope"]["seeds"]:
        raise ValueError("Locked test seeds must equal the seeds prospectively frozen in the selection plan")
    locked_path = output / "locked_configs.json"
    # Re-audit all selection artifacts before every locked-test resume.
    locked = lock_hpo(output)
    env = selection["environment"]
    runs = [design.run_spec(env, "hpo_test", item["dataset"], item["family"], item["family"],
                           seed, item["params"], "test") for item in locked["winners"] for seed in seeds]
    plan = dict(schema=1, action="hpo-test", output=str(output.resolve()), environment=env,
                scope=dict(selection["scope"], seeds=seeds),
                metadata=dict(locked_sha256=file_hash(locked_path)), runs=runs)
    write_json(output / "test_plan.json", plan, immutable=True)
    write_csv(output / "test_plan.csv", runs)
    return plan


def execute(plan, plan_path, args):
    from scripts.paper.run_full_p1_queue import queue_lock, QueueAlreadyRunning
    from .worker import require_gpu
    output = Path(plan["output"])
    # Shared with the original full-P1 runner: do not overlap those training queues.
    lockfile = ROOT / "results_analysis/review_gap_campaigns/full_p1/queue.lock"
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        with queue_lock(lockfile):
            require_gpu()
            state = dict(pid=os.getpid(), status="running", action=plan["action"],
                         started=time.strftime("%Y-%m-%d %H:%M:%S"), planned=len(plan["runs"]))
            executed = 0
            try:
                for index, spec in enumerate(plan["runs"], 1):
                    if completed(output, spec):
                        print(f"[{index}/{len(plan['runs'])}] SKIP {spec['run_id']}", flush=True)
                        continue
                    if args.max_runs and executed >= args.max_runs:
                        state["status"] = "paused_at_run_limit"
                        break
                    if shutil.disk_usage(output).free < args.min_free_gb*1024**3:
                        raise RuntimeError("Free disk space below safety threshold")
                    run_dir = output / "runs" / spec["run_id"]
                    run_dir.mkdir(parents=True, exist_ok=True)
                    number = max((int(p.name.split("_")[1]) for p in run_dir.glob("attempt_*") if p.name.split("_")[1].isdigit()), default=0)+1
                    attempt = run_dir / f"attempt_{number:03d}"
                    attempt.mkdir()  # Preserve all interrupted attempts; never overwrite them.
                    logs = output / "logs"
                    logs.mkdir(exist_ok=True)
                    prefix = f"{spec['run_id']}_attempt{number:03d}"
                    stdout, stderr = logs / (prefix+".stdout.log"), logs / (prefix+".stderr.log")
                    state.update(current_run=spec["run_id"], stdout_log=str(stdout), stderr_log=str(stderr))
                    write_json(output / "queue.json", state)
                    print(f"[{index}/{len(plan['runs'])}] RUN {spec['run_id']}", flush=True)
                    command = [args.python, "-u", "-X", "utf8", "-m", "experiments.controlled_addons.worker",
                               "--plan", str(plan_path), "--run-id", spec["run_id"], "--attempt", str(attempt)]
                    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
                        process = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err)
                    executed += 1
                    if process.returncode or completed(output, spec) is None:
                        raise RuntimeError(f"Worker failed (exit {process.returncode}). Inspect {stderr}")
                    write_json(output / "queue.json", dict(state, executed_this_invocation=executed))
                else:
                    state["status"] = "completed"
                state.update(ended=time.strftime("%Y-%m-%d %H:%M:%S"), executed_this_invocation=executed)
                write_json(output / "queue.json", state)
            except BaseException as exc:
                write_json(output / "queue.json", dict(state, status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc)))
                raise
    except QueueAlreadyRunning:
        print("A full-P1/add-on queue already holds the lock; no duplicate was started.", flush=True)
        return 0
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("projected", "preprocessing", "hpo-select", "hpo-lock", "hpo-test", "analyze"))
    parser.add_argument("--datasets", help="Comma-separated; defaults to the declared study scope")
    parser.add_argument("--models", default=",".join(FAMILIES), help="HPO families only")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--out-dir", type=Path, default=design.OUTPUT)
    parser.add_argument("--python", default=sys.executable)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true", help="Explicitly authorize sequential RTX4090 training")
    modes.add_argument("--plan-only", action="store_true", help="Default: save the plan/profiles, do not train")
    parser.add_argument("--skip-completed", action="store_true", help="Verified completions are ALWAYS skipped safely")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--min-free-gb", type=float, default=50)
    parser.add_argument("--match-tolerance", type=float, default=0.01)
    parser.add_argument("--study", choices=("projected", "preprocessing", "hpo"), default="projected", help="Analysis study")
    parser.add_argument("--allow-partial", action="store_true", help="Coverage/metrics only, no inferential claims on incomplete grids")
    args = parser.parse_args(argv)
    if args.max_runs < 0 or args.min_free_gb <= 0 or not 0 < args.match_tolerance <= 0.05:
        raise ValueError("Invalid resource/run-limit/matching options")
    seeds = split_items(args.seeds, cast=int)
    if 2020 in seeds or min(seeds) < 1:
        raise ValueError("Test seeds must be positive and exclude selection seed 2020")
    if args.action == "analyze":
        from .analyze import analyze
        return analyze(args.out_dir / args.study, args.allow_partial)
    study = "hpo" if args.action.startswith("hpo") else args.action
    output = args.out_dir.resolve() / study
    if args.action == "hpo-lock":
        lock_hpo(output)
        return 0
    if args.action == "hpo-test":
        plan = test_hpo_plan(output, seeds)
        plan_path = output / "test_plan.json"
    else:
        default_ds = ("ETTh2", "ETTm2", "Weather") if study == "projected" else ("ETTh2", "ETTm2", "Exchange", "Weather") if study == "preprocessing" else tuple(DATASETS)
        datasets = split_items(args.datasets or ",".join(default_ds), DATASETS)
        families = split_items(args.models, FAMILIES)
        plan = freeze_plan(output, args.action, datasets, seeds, args.match_tolerance, families)
        plan_path = output / "plan.json"
    verified = sum(completed(output, spec) is not None for spec in plan["runs"])
    print(f"{args.action}: {len(plan['runs'])} planned; {verified} verified complete; {len(plan['runs'])-verified} remaining.", flush=True)
    if plan["metadata"].get("match"):
        match = plan["metadata"]["match"]
        print(f"Budget comparator: {match['params']}; params {match['parameter_difference_pct']:+.4f}%, MACs {match['mac_difference_pct']:+.4f}%", flush=True)
    print(f"Frozen plan: {plan_path}", flush=True)
    if not args.execute:
        print("PLAN ONLY. No training launched. Add --execute when ready.", flush=True)
        return 0
    return execute(plan, plan_path, args)
