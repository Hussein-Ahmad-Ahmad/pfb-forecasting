"""Paired effects with dataset-level intervals; never treats seeds as datasets."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .design import CONTRASTS, digest
from .storage import completed, write_csv, write_json


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(values)-rank)*values[index]))
        adjusted[index] = previous
    return adjusted


def interval(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Inference requires >=2 finite independent units")
    mean = float(values.mean())
    se = float(values.std(ddof=1)/math.sqrt(len(values)))
    margin = float(stats.t.ppf(0.975, len(values)-1)*se)
    p = float(2*stats.t.sf(abs(mean/se), len(values)-1)) if se else (1.0 if mean == 0 else 0.0)
    return dict(effect_pct=mean, ci95_low_pct=mean-margin, ci95_high_pct=mean+margin, p_unadjusted=p)


def paired(frame, lhs, rhs, selector="variant"):
    keys = ["dataset", "seed"]
    a = frame[frame[selector].eq(lhs)]
    b = frame[frame[selector].eq(rhs)]
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        raise ValueError("Duplicate seed/dataset pairing")
    if set(map(tuple, a[keys].to_numpy())) != set(map(tuple, b[keys].to_numpy())):
        raise ValueError("Unbalanced paired contrast; no silent inner-join deletion")
    joined = a.merge(b, on=keys, suffixes=("_lhs", "_rhs"), validate="one_to_one")
    for field in ("protocol_sha256", "software_sha256", "source_sha256", "data_sha256"):
        if not joined[field+"_lhs"].eq(joined[field+"_rhs"]).all():
            raise ValueError("Cross-protocol/source/data pairing is forbidden")
    if joined.empty or (joined.mse_rhs <= 0).any():
        raise ValueError("Empty contrast/nonpositive reference MSE")
    result = joined[keys].copy()
    result["lhs"], result["rhs"] = lhs, rhs
    result["mse_lhs"], result["mse_rhs"] = joined.mse_lhs, joined.mse_rhs
    result["delta_mse"] = joined.mse_lhs - joined.mse_rhs
    result["effect_pct"] = 100*result.delta_mse / result.mse_rhs
    result["delta_mae"] = joined.mae_lhs - joined.mae_rhs
    return result


def summarize_pairs(pairs, meaning, priority):
    dataset_means = pairs.groupby("dataset", sort=True).effect_pct.mean()
    summary = dict(lhs=pairs.lhs.iloc[0], rhs=pairs.rhs.iloc[0], interpretation=meaning,
                   priority=priority, n_pairs=len(pairs), n_datasets=len(dataset_means),
                   **interval(dataset_means.to_numpy()))
    return summary


def analyze(output, allow_partial=False):
    output = Path(output)
    analysis = output / "analysis"
    write_json(analysis / "status.json", dict(status="verifying", inferential_results_generated=False))
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    plans = [plan]
    if plan["action"] == "hpo-select" and (output / "test_plan.json").exists():
        plans.append(json.loads((output / "test_plan.json").read_text(encoding="utf-8")))
    coverage, records = [], []
    for item in plans:
        for spec in item["runs"]:
            result = completed(output, spec)
            coverage.append(dict(run_id=spec["run_id"], study=spec["study"], dataset=spec["dataset"],
                                 family=spec["family"], split=spec["split"], completed=result is not None))
            if result:
                records.append(dict(**spec, mae=result["mae"], mse=result["mse"],
                                    parameters=result["summary"]["parameters"],
                                    allocated_device_wall_hours=result["summary"]["allocated_device_wall_hours"],
                                    metrics_sha256=result["sha256"]["metrics.npy"],
                                    run_by_gpu_watermark="EXP_RUN_BY_RTX4090_GPU"))
    write_csv(analysis / "coverage.csv", coverage)
    write_csv(analysis / "verified_metrics.csv", records)
    incomplete = any(not r["completed"] for r in coverage)
    if plan["action"] == "hpo-select" and len(plans) < 2:
        incomplete = True
    if incomplete:
        # Do not leave previously generated inference looking current if input
        # coverage subsequently becomes incomplete. Raw inputs are untouched.
        for filename in ("paired_effects.csv", "dataset_level_effects.csv", "within_dataset_seed_effects.csv",
                         "hpo_compute_budget.csv", "locked_test_leaderboard.csv"):
            if (analysis / filename).exists():
                write_csv(analysis / filename, [])
        write_json(analysis / "status.json", dict(status="incomplete", planned=len(coverage),
                   verified=len(records), inferential_results_generated=False,
                   note="For HPO, both selection and all locked test seeds are required"))
        if not allow_partial:
            raise ValueError("Incomplete grid: coverage exported, no inference. Use --allow-partial for coverage only.")
        print(f"Coverage only: {len(records)}/{len(coverage)} verified; no inferential conclusions.")
        return 0
    frame = pd.DataFrame(records)
    effects, summaries, task_summaries = [], [], []
    contrasts = CONTRASTS
    if plan["action"] == "preprocessing":
        frames = [(family, group, (("norm_on", "norm_off", "One window normalization ON vs OFF", "primary"),))
                  for family, group in frame.groupby("family", sort=True)]
    elif plan["action"] == "hpo-select":
        tests = frame[frame.split.eq("test")]
        frames = [("locked_hpo", tests, (("PFB-Direct", "PatchTST", "Validation-selected complete configurations", "primary"),
                                         ("PFB-Projected", "PatchTST", "Validation-selected complete configurations", "secondary")))]
        # Distinct active candidates and occupied wall hours are both reported.
        selection = frame[frame.split.eq("val")]
        compute = selection.groupby(["dataset", "family"], as_index=False).agg(
            distinct_candidates=("variant", "nunique"), executions=("run_id", "count"),
            allocated_device_wall_hours=("allocated_device_wall_hours", "sum"))
        write_csv(analysis / "hpo_compute_budget.csv", compute.to_dict("records"))
        table = tests.groupby(["dataset", "family"], as_index=False).agg(
            mse_mean=("mse", "mean"), mse_sd=("mse", "std"), mae_mean=("mae", "mean"), n_seeds=("seed", "nunique"))
        table["mse_rank"] = table.groupby("dataset").mse_mean.rank(method="min")
        write_csv(analysis / "locked_test_leaderboard.csv", table.to_dict("records"))
    else:
        frames = [("mechanism", frame, contrasts)]
    for label, group, contrasts in frames:
        for lhs, rhs, meaning, priority in contrasts:
            pairs = paired(group, lhs, rhs)
            pairs["analysis_group"] = label
            effects.extend(pairs.to_dict("records"))
            if pairs.dataset.nunique() < 2:
                # Per-dataset tests can still be exported, but not a fabricated
                # population interval from five seeds of one dataset.
                raise ValueError("Dataset-level inference requires at least two dataset identities")
            summaries.append(dict(analysis_group=label, **summarize_pairs(pairs, meaning, priority)))
            for dataset, subset in pairs.groupby("dataset", sort=True):
                task_summaries.append(dict(analysis_group=label, dataset=dataset, lhs=lhs, rhs=rhs,
                                           n_seeds=len(subset), **interval(subset.effect_pct.to_numpy())))
    # Holm covers ALL declared tests, not just whichever look favorable.
    for rows in (summaries, task_summaries):
        for row, adjusted in zip(rows, holm([r["p_unadjusted"] for r in rows])):
            row["p_holm_all_declared_contrasts"] = float(adjusted)
            row["interval_note"] = "Unadjusted 95% t interval; not an equivalence test"
    write_csv(analysis / "paired_effects.csv", effects)
    write_csv(analysis / "dataset_level_effects.csv", summaries)
    write_csv(analysis / "within_dataset_seed_effects.csv", task_summaries)
    write_json(analysis / "status.json", dict(status="complete", verified=len(records),
               inferential_results_generated=True, plan_sha256=digest(plan),
               note="Small dataset counts limit generalization; no predictive intervals or causal universality claims"))
    print(f"Verified complete grid: {len(records)} runs. Paired effects: {analysis}")
    return 0
