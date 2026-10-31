#!/usr/bin/env python
"""Summarize controlled replication campaign results.

The script reads Time-Series-Library ``metrics.npy`` files and produces compact
CSV summaries for:

- paired model contrasts;
- dataset-level summaries and nested random-effects estimates;
- PatchTST output-level ensembles from two trained members.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import optimize, stats


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_analysis" / "review_gap_campaigns" / "analysis"
WATERMARK = "EXP_RUN_BY_RTX4090_GPU"


def parse_model_id(folder: Path) -> dict[str, str] | None:
    name = folder.name
    if not name.startswith("long_term_forecast_"):
        return None
    short = name[len("long_term_forecast_") :]
    parts = short.split("_")
    if len(parts) < 6:
        return None
    try:
        h_idx = next(i for i, part in enumerate(parts) if part.startswith("H") and part[1:].isdigit())
        s_idx = next(
            i
            for i, part in enumerate(parts)
            if (part.startswith("S") and part[1:].isdigit())
            or (part.startswith("seed") and part[4:].isdigit())
        )
    except StopIteration:
        return None
    seed = parts[s_idx][1:] if parts[s_idx].startswith("S") else parts[s_idx][4:]
    tag = parts[s_idx + 1] if len(parts) > s_idx + 1 else ""
    if tag == "member" and len(parts) > s_idx + 2 and parts[s_idx + 2] in {"a", "b"}:
        tag = f"member_{parts[s_idx + 2]}"

    if parts[0] in {"ElectricityCoreFiveSeed", "TrafficCoreFiveSeed"} and h_idx >= 3:
        model = parts[1]
        dataset_idx = 2
    elif h_idx >= 2 and parts[h_idx - 2] == "PatchTST" and parts[h_idx - 1] == "SerialMatched":
        model = "PatchTST_SerialMatched"
        dataset_idx = h_idx - 3
    else:
        model = parts[h_idx - 1]
        dataset_idx = h_idx - 2
    if dataset_idx < 1:
        return None
    campaign = "_".join(parts[:dataset_idx])
    if campaign.startswith("ElectricityCoreFiveSeed"):
        campaign = "electricity_core_5seed"
    elif campaign.startswith("TrafficCoreFiveSeed"):
        campaign = "traffic_core_5seed"
    return {
        "campaign": campaign,
        "dataset": parts[dataset_idx],
        "model": model,
        "horizon": parts[h_idx][1:],
        "seed": seed,
        "tag": tag,
    }


def metric_rows() -> list[dict[str, object]]:
    rows = []
    for metrics in (ROOT / "results").glob("long_term_forecast_*/metrics.npy"):
        info = parse_model_id(metrics.parent)
        if info is None:
            continue
        values = np.load(metrics)
        provenance = metrics.parent / "run_provenance.json"
        watermark = ""
        if provenance.exists():
            try:
                watermark = json.loads(provenance.read_text(encoding="utf-8")).get("run_by_gpu_watermark", "")
            except json.JSONDecodeError:
                watermark = ""
        rows.append({
            **info,
            "mae": float(values[0]),
            "mse": float(values[1]),
            "rmse": float(values[2]),
            "mape": float(values[3]),
            "mspe": float(values[4]),
            "metrics_path": str(metrics),
            "run_by_gpu_watermark": watermark,
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_contrasts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {(r["campaign"], r["dataset"], r["horizon"], r["seed"], r["tag"], r["model"]): r for r in rows}
    pairs = [
        ("common_outer_preproc_test", "PFB-Direct", "PatchTST"),
        ("common_outer_preproc_test", "PFB-Projected", "PatchTST"),
        ("equal_budget_hpo_locked_test", "PFB-Direct", "PatchTST"),
        ("equal_budget_hpo_locked_test", "PFB-Projected", "PatchTST"),
        ("ten_seed_direct_patch", "PFB-Direct", "PatchTST"),
        ("ten_seed_serial_parallel", "PFB-Direct", "PatchTST_SerialMatched"),
        ("projected_param_match", "PFB-Projected", "PatchTST"),
        ("projected_param_match", "PFB-Projected", "PFB-Direct"),
        ("lookback_robustness", "PFB-Direct", "PatchTST"),
        ("rolling_origin_prefix", "PFB-Direct", "PatchTST"),
    ]
    out = []
    for campaign, lhs, rhs in pairs:
        lhs_rows = [r for r in rows if r["campaign"] == campaign and r["model"] == lhs]
        for left in lhs_rows:
            right = by_key.get((campaign, left["dataset"], left["horizon"], left["seed"], left["tag"], rhs))
            if right is None:
                continue
            out.append({
                "campaign": campaign,
                "dataset": left["dataset"],
                "horizon": left["horizon"],
                "seed": left["seed"],
                "tag": left["tag"],
                "lhs_model": lhs,
                "rhs_model": rhs,
                "lhs_mse": left["mse"],
                "rhs_mse": right["mse"],
                "delta_mse_lhs_minus_rhs": left["mse"] - right["mse"],
                "relative_mse_change": (left["mse"] - right["mse"]) / right["mse"] if right["mse"] else "",
                "run_by_gpu_watermark": left.get("run_by_gpu_watermark") or right.get("run_by_gpu_watermark"),
            })
    return out


def hierarchical_effects(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Fit paired relative effects with dataset and nested horizon intercepts.

    The model is ``effect = intercept + dataset + dataset:horizon + residual``,
    with independent Gaussian random intercepts.  Variance components are
    estimated by restricted maximum likelihood.  When a campaign contains one
    horizon per dataset, the nested term is omitted because it is not
    separately identifiable.  Confidence intervals use dataset-level degrees
    of freedom, reflecting the number of independent dataset identities rather
    than the number of seeds.
    """
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in pairs:
        grouped[(str(row["campaign"]), str(row["lhs_model"]), str(row["rhs_model"]))].append(row)

    output = []
    for (campaign, lhs, rhs), cell in sorted(grouped.items()):
        y = np.asarray([100.0 * float(row["relative_mse_change"]) for row in cell], dtype=float)
        datasets = [str(row["dataset"]) for row in cell]
        dataset_horizons = [f'{row["dataset"]}::{row["horizon"]}' for row in cell]
        dataset_levels = sorted(set(datasets))
        nested_levels = sorted(set(dataset_horizons))
        n = len(y)
        z_dataset = np.zeros((n, len(dataset_levels)), dtype=float)
        for i, value in enumerate(datasets):
            z_dataset[i, dataset_levels.index(value)] = 1.0
        use_nested = len(nested_levels) > len(dataset_levels)
        components = [z_dataset]
        if use_nested:
            z_nested = np.zeros((n, len(nested_levels)), dtype=float)
            for i, value in enumerate(dataset_horizons):
                z_nested[i, nested_levels.index(value)] = 1.0
            components.append(z_nested)

        x = np.ones((n, 1), dtype=float)
        identity = np.eye(n, dtype=float)

        def fit_at(log_variances: np.ndarray) -> tuple[float, float, float]:
            variances = np.exp(log_variances)
            covariance = variances[-1] * identity
            for variance, design in zip(variances[:-1], components):
                covariance = covariance + variance * (design @ design.T)
            sign_v, logdet_v = np.linalg.slogdet(covariance)
            if sign_v <= 0:
                return np.inf, np.nan, np.nan
            solved_x = np.linalg.solve(covariance, x)
            solved_y = np.linalg.solve(covariance, y)
            xt_vinv_x = float((x.T @ solved_x)[0, 0])
            beta = float((x.T @ solved_y)[0] / xt_vinv_x)
            residual = y - beta
            quadratic = float(residual.T @ np.linalg.solve(covariance, residual))
            reml = 0.5 * (
                logdet_v + np.log(xt_vinv_x) + quadratic + (n - 1) * np.log(2.0 * np.pi)
            )
            return reml, beta, float(np.sqrt(1.0 / xt_vinv_x))

        initial_variance = max(float(np.var(y, ddof=1)), 1e-6)
        initial = np.log(np.full(len(components) + 1, initial_variance / (len(components) + 1)))
        result = optimize.minimize(
            lambda values: fit_at(values)[0],
            initial,
            method="L-BFGS-B",
            bounds=[(-18.0, 12.0)] * len(initial),
        )
        _, estimate, standard_error = fit_at(result.x)
        degrees_freedom = max(len(dataset_levels) - 1, 1)
        critical = float(stats.t.ppf(0.975, degrees_freedom))
        variances = np.exp(result.x)
        output.append({
            "campaign": campaign,
            "lhs_model": lhs,
            "rhs_model": rhs,
            "n_pairs": n,
            "n_datasets": len(dataset_levels),
            "n_dataset_horizon_cells": len(nested_levels),
            "mean_relative_mse_change_pct": estimate,
            "se_pct": standard_error,
            "ci95_low_pct": estimate - critical * standard_error,
            "ci95_high_pct": estimate + critical * standard_error,
            "dataset_variance_pct2": float(variances[0]),
            "nested_horizon_variance_pct2": float(variances[1]) if use_nested else "",
            "residual_variance_pct2": float(variances[-1]),
            "reml_converged": bool(result.success),
            "run_by_gpu_watermark": WATERMARK,
        })
    return output


def dataset_level(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["campaign"], row["dataset"], row["horizon"], row["model"])].append(row)
    summary = []
    for (campaign, dataset, horizon, model), cell in sorted(grouped.items()):
        mses = np.array([float(r["mse"]) for r in cell], dtype=float)
        summary.append({
            "campaign": campaign,
            "dataset": dataset,
            "horizon": horizon,
            "model": model,
            "n": len(mses),
            "mean_mse": float(np.mean(mses)),
            "std_mse": float(np.std(mses, ddof=1)) if len(mses) > 1 else 0.0,
            "se_mse": float(np.std(mses, ddof=1) / np.sqrt(len(mses))) if len(mses) > 1 else 0.0,
            "run_by_gpu_watermark": WATERMARK if any(r.get("run_by_gpu_watermark") == WATERMARK for r in cell) else "",
        })
    return summary


def ensemble_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_cell = defaultdict(dict)
    for row in rows:
        if row["campaign"] != "patchtst_output_ensemble":
            continue
        pair_seed = int(row["seed"])
        if row["tag"] == "member_b":
            pair_seed -= 1000
        key = (row["dataset"], row["horizon"], str(pair_seed))
        by_cell[key][row["tag"]] = row
    out = []
    for (dataset, horizon, seed), cell in sorted(by_cell.items()):
        a = cell.get("member_a")
        b = cell.get("member_b")
        if not a or not b:
            continue
        pred_a = np.load(Path(a["metrics_path"]).with_name("pred.npy"))
        pred_b = np.load(Path(b["metrics_path"]).with_name("pred.npy"))
        true = np.load(Path(a["metrics_path"]).with_name("true.npy"))
        pred = (pred_a + pred_b) / 2.0
        diff = pred - true
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))
        out.append({
            "campaign": "patchtst_output_ensemble",
            "dataset": dataset,
            "horizon": horizon,
            "seed": seed,
            "ensemble_mse": mse,
            "ensemble_mae": mae,
            "member_a_mse": a["mse"],
            "member_b_mse": b["mse"],
            "run_by_gpu_watermark": WATERMARK,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    rows = metric_rows()
    write_csv(out_dir / "all_metrics.csv", rows)
    contrasts = paired_contrasts(rows)
    write_csv(out_dir / "paired_contrasts.csv", contrasts)
    write_csv(out_dir / "hierarchical_effects.csv", hierarchical_effects(contrasts))
    write_csv(out_dir / "dataset_level_summary.csv", dataset_level(rows))
    write_csv(out_dir / "patchtst_output_ensemble.csv", ensemble_rows(rows))
    print(f"Wrote review-gap summaries to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
