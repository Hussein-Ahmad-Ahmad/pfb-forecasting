#!/usr/bin/env python
"""Generate manuscript figures for the controlled RTX4090 replication studies."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "results_analysis" / "review_gap_campaigns" / "analysis"
DEFAULT_OUT = ROOT / "results_analysis" / "review_gap_campaigns" / "figures"
WATERMARK = "EXP_RUN_BY_RTX4090_GPU"


def common_preprocessing_figure(summary: pd.DataFrame, output: Path, publication: bool = False) -> None:
    campaign = summary[summary["campaign"] == "common_outer_preproc_test"].copy()
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather"]
    horizons = [96, 192, 336]
    patch = campaign[campaign["model"] == "PatchTST"].set_index(["dataset", "horizon"])["mean_mse"]

    effects: dict[str, np.ndarray] = {}
    for model in ["PFB-Direct", "PFB-Projected"]:
        values = campaign[campaign["model"] == model].set_index(["dataset", "horizon"])["mean_mse"]
        matrix = np.zeros((len(datasets), len(horizons)), dtype=float)
        for row, dataset in enumerate(datasets):
            for column, horizon in enumerate(horizons):
                matrix[row, column] = 100.0 * (
                    values.loc[(dataset, horizon)] - patch.loc[(dataset, horizon)]
                ) / patch.loc[(dataset, horizon)]
        effects[model] = matrix

    limit = max(6.0, max(float(np.abs(matrix).max()) for matrix in effects.values()))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.1), constrained_layout=True)
    for axis, (model, matrix) in zip(axes, effects.items()):
        image = axis.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
        axis.set_title(f"{model} vs. PatchTST", fontsize=10, weight="bold")
        axis.set_xticks(range(len(horizons)), [str(horizon) for horizon in horizons])
        axis.set_yticks(range(len(datasets)), datasets)
        axis.set_xlabel("Forecast horizon")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                color = "white" if abs(value) > 0.55 * limit else "black"
                axis.text(column, row, f"{value:+.1f}%", ha="center", va="center", fontsize=8, color=color)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.83, pad=0.02)
    colorbar.set_label("Relative MSE change (negative favors PFB)")
    if not publication:
        fig.text(0.5, -0.015, WATERMARK, ha="center", va="top", fontsize=7, color="0.45")
    fig.savefig(output / "common_preprocessing_full_rtx4090.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def hierarchical_figure(effects: pd.DataFrame, output: Path, publication: bool = False) -> None:
    requested = [
        ("common_outer_preproc_test", "PFB-Direct", "Outer-preprocessing: Direct vs PatchTST"),
        ("common_outer_preproc_test", "PFB-Projected", "Outer-preprocessing: Projected vs PatchTST"),
        ("equal_budget_hpo_locked_test", "PFB-Direct", "Finite-candidate selection: Direct vs PatchTST"),
        ("equal_budget_hpo_locked_test", "PFB-Projected", "Finite-candidate selection: Projected vs PatchTST"),
        ("ten_seed_direct_patch", "PFB-Direct", "Ten-seed: Direct vs PatchTST"),
        ("ten_seed_serial_parallel", "PFB-Direct", "Ten-seed: Direct vs matched serial"),
        ("projected_param_match", "PFB-Projected", "Fixed configuration: Projected vs PatchTST"),
        ("lookback_robustness", "PFB-Direct", "Lookback (pooled): Direct vs PatchTST"),
        ("rolling_origin_prefix", "PFB-Direct", "Temporal prefixes (pooled): Direct vs PatchTST"),
    ]
    rows = []
    for campaign, model, label in requested:
        candidates = effects[(effects["campaign"] == campaign) & (effects["lhs_model"] == model)]
        if campaign == "projected_param_match":
            candidates = candidates[candidates["rhs_model"] == "PatchTST"]
        if candidates.empty:
            continue
        row = candidates.iloc[0].copy()
        row["label"] = label
        rows.append(row)
    frame = pd.DataFrame(rows)
    y = np.arange(len(frame))[::-1]
    mean = frame["mean_relative_mse_change_pct"].to_numpy(dtype=float)
    lower = frame["ci95_low_pct"].to_numpy(dtype=float)
    upper = frame["ci95_high_pct"].to_numpy(dtype=float)

    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    axis.axvspan(-1.0, 1.0, color="0.92", label="1% practical-effect band")
    axis.axvline(0.0, color="0.25", linewidth=1.0)
    axis.errorbar(
        mean,
        y,
        xerr=np.vstack([mean - lower, upper - mean]),
        fmt="o",
        color="#135D8D",
        ecolor="#135D8D",
        capsize=3,
        linewidth=1.2,
        markersize=5,
    )
    axis.set_yticks(y, frame["label"].tolist(), fontsize=8.5)
    axis.set_xlabel("Hierarchical mean relative MSE change (%)\nnegative favors the PFB configuration")
    axis.grid(axis="x", color="0.88", linewidth=0.7)
    axis.legend(loc="upper right", bbox_to_anchor=(1.0, 1.06), fontsize=8, frameon=False)
    if not publication:
        fig.text(0.995, 0.005, WATERMARK, ha="right", va="bottom", fontsize=7, color="0.45")
    fig.savefig(output / "hierarchical_effects_rtx4090.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def additional_dataset_figure(output: Path) -> None:
    """Keep prespecified variants separate; never select the best after testing."""
    frame = pd.read_csv(ROOT / "results_summary" / "additional_patch_family_summary.csv")
    reference = frame[frame.model == "PatchTST"].set_index(["dataset", "horizon"]).mse_mean
    tasks = [(ds, h) for ds in ["AQShunyi", "CzeLan", "Wind"] for h in [96, 192, 336]]
    x = np.arange(len(tasks))
    fig, axis = plt.subplots(figsize=(8.2, 3.5), constrained_layout=True)
    for model, offset, color in [("PFB-Direct", -.18, "#135D8D"), ("PFB-Projected", .18, "#DE8B32")]:
        series = frame[frame.model == model].set_index(["dataset", "horizon"]).mse_mean
        values = [100 * (reference.loc[key] - series.loc[key]) / reference.loc[key] for key in tasks]
        axis.bar(x + offset, values, .36, label=model, color=color)
        print(f"{model}: lower mean MSE in {sum(v > 0 for v in values)}/{len(values)} additional tasks")
    axis.axhline(0, color="0.25", linewidth=.8)
    axis.set_xticks(x, [f"{ds}\n{h}" for ds, h in tasks], fontsize=8)
    axis.set_ylabel("Relative MSE reduction vs. PatchTST (%)")
    axis.legend(frameon=False, fontsize=9)
    axis.grid(axis="y", alpha=.2)
    fig.savefig(output / "additional_dataset_gains_separate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--publication", action="store_true", help="Omit visible hardware tags; raw CSV/JSON provenance is unchanged")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.analysis_dir / "dataset_level_summary.csv")
    effects = pd.read_csv(args.analysis_dir / "hierarchical_effects.csv")
    common_preprocessing_figure(summary, args.out_dir, args.publication)
    hierarchical_figure(effects, args.out_dir, args.publication)
    additional_dataset_figure(args.out_dir)
    print(f"Wrote controlled-replication figures to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
