"""
Error-by-Forecast-Step Analysis
=================================
Compute and visualize how per-step MSE evolves across the forecast horizon.

pred.npy / true.npy shape: (n_samples, pred_len, n_channels)
per-step MSE at step t = mean over samples and channels of (pred[:, t, :] - true[:, t, :])^2

Datasets covered: Weather H=192 and ETTm2 H=192 (multi-seed runs).

Usage:
    python analyze_error_by_step.py [--plot]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR = Path(__file__).resolve().parent / "results_analysis"

# ---------------------------------------------------------------------------
# Folder discovery patterns per (dataset_tag, horizon)
# Key = (dataset_tag, horizon)  value = list of (model_name, regex_pattern)
# ---------------------------------------------------------------------------
# We match against folder names inside RESULTS_DIR.
DISCOVERY_SPECS: dict[tuple[str, int], list[tuple[str, str]]] = {
    ("Weather", 192): [
        ("DLinear",           r"DLinear_MS\d+_Weather_192_DLinear_.*multiseed_h192"),
        ("PatchTST",          r"PatchTST_MS\d+_Weather_192_PatchTST_.*multiseed_h192"),
        ("PFB-Direct", r"PFB-Direct_MS\d+_Weather_192_PFB-Direct_.*multiseed_h192"),
    ],
    ("ETTm2", 192): [
        ("DLinear",           r"DLinear_MS\d+_ETTm2_192_DLinear_.*multiseed_h192"),
        ("PatchTST",          r"PatchTST_MS\d+_ETTm2_192_PatchTST_.*multiseed_h192"),
        ("PFB-Direct", r"PFB-Direct_MS\d+_ETTm2_192_PFB-Direct_.*multiseed_h192"),
    ],
}


def discover_folders(pattern: str) -> list[Path]:
    """Return all folders in RESULTS_DIR whose name matches *pattern* (regex)."""
    compiled = re.compile(pattern)
    return sorted(f for f in RESULTS_DIR.iterdir() if f.is_dir() and compiled.search(f.name))


def per_step_mse(folder: Path) -> np.ndarray | None:
    """Load pred/true npy files and return per-step MSE array (shape: pred_len,)."""
    pred_path = folder / "pred.npy"
    true_path = folder / "true.npy"
    if not pred_path.exists() or not true_path.exists():
        return None
    pred = np.load(str(pred_path))   # (n_samples, pred_len, n_channels)
    true = np.load(str(true_path))
    sq_err = (pred - true) ** 2       # (n_samples, pred_len, n_channels)
    return sq_err.mean(axis=(0, 2))   # (pred_len,)


def compute_step_curves(
    dataset_tag: str, horizon: int
) -> pd.DataFrame | None:
    specs = DISCOVERY_SPECS.get((dataset_tag, horizon))
    if specs is None:
        print(f"No spec for {dataset_tag} H={horizon}")
        return None

    records: list[dict] = []
    for model_name, pattern in specs:
        folders = discover_folders(pattern)
        if not folders:
            print(f"  [{dataset_tag} H={horizon}] No folders found for {model_name} (pattern: {pattern})")
            continue
        per_seed_curves: list[np.ndarray] = []
        for folder in folders:
            curve = per_step_mse(folder)
            if curve is not None:
                per_seed_curves.append(curve)
        if not per_seed_curves:
            print(f"  [{dataset_tag} H={horizon}] No valid pred/true files for {model_name}")
            continue
        mean_curve = np.mean(per_seed_curves, axis=0)  # (pred_len,)
        for step_idx, mse_val in enumerate(mean_curve):
            records.append(
                {
                    "dataset": dataset_tag,
                    "horizon": horizon,
                    "model": model_name,
                    "step": step_idx + 1,
                    "per_step_mse": float(mse_val),
                    "n_seeds": len(per_seed_curves),
                }
            )
        print(f"  [{dataset_tag} H={horizon}] {model_name}: {len(per_seed_curves)} seeds, "
              f"step-1 MSE={mean_curve[0]:.4f}, step-{len(mean_curve)} MSE={mean_curve[-1]:.4f}")

    return pd.DataFrame(records) if records else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib plots")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_dfs: list[pd.DataFrame] = []
    for (dataset_tag, horizon) in DISCOVERY_SPECS:
        print(f"\nProcessing {dataset_tag} H={horizon} ...")
        df = compute_step_curves(dataset_tag, horizon)
        if df is not None and not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("No results found. Check that pred.npy/true.npy exist in results/ directories.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    out_csv = OUT_DIR / "error_by_step.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping plots")
            return

        MODEL_COLORS = {
            "DLinear": "gray",
            "PatchTST": "steelblue",
            "PFB-Direct": "tomato",
            "PFB-Projected": "darkorange",
        }
        MODEL_LABELS = {
            "DLinear": "DLinear",
            "PatchTST": "PatchTST",
            "PFB-Direct": "PFB-Direct",
            "PFB-Projected": "PFB-Projected",
        }

        for (dataset_tag, horizon), grp in combined.groupby(["dataset", "horizon"]):
            fig, ax = plt.subplots(figsize=(8, 4))
            for model, mgrp in grp.groupby("model"):
                mgrp_sorted = mgrp.sort_values("step")
                color = MODEL_COLORS.get(model, "black")
                label = MODEL_LABELS.get(model, model)
                ax.plot(mgrp_sorted["step"], mgrp_sorted["per_step_mse"],
                        label=label, color=color, linewidth=1.5)
            ax.set_xlabel("Forecast Step")
            ax.set_ylabel("Per-Step MSE")
            ax.set_title(f"{dataset_tag} H={horizon}: Error by Forecast Step")
            ax.legend(framealpha=0.5)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plot_path = OUT_DIR / f"error_by_step_{dataset_tag}_{horizon}.pdf"
            fig.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
