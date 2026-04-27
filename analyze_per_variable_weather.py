"""
Per-Variable Error Analysis — Weather H=192
=============================================
For each of the 21 Weather channels, compute what fraction of the total MSE
each variable contributes, and how much PatchFusionBERT_v0 gains/loses per
variable versus PatchTST (the backbone).

Per-variable MSE at variable c:
    mse_c = mean((pred[:, :, c] - true[:, :, c])^2)  over all samples and steps

Relative gain at variable c (positive = PFBv0 wins, negative = PFBv0 loses):
    gain_c = (patchtst_mse_c - pfbv0_mse_c) / patchtst_mse_c  * 100

Output: results_analysis/per_variable_weather_error.csv
Optionally: results_analysis/per_variable_weather_error.pdf  (with --plot)

Usage:
    python analyze_per_variable_weather.py [--plot]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR = Path(__file__).resolve().parent / "results_analysis"

# 21 Weather feature names (order matches enc_in columns, excluding 'date' col)
WEATHER_FEATURE_NAMES = [
    "p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)",
    "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)",
    "H2OC (mmol/mol)", "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)",
    "wd (deg)", "rain (mm)", "raining (s)", "SWDR (W/m2)",
    "PAR (umol/m2/s)", "max. PAR (umol/m2/s)", "Tlog (degC)", "OT",
]

MODELS = {
    "DLinear": r"DLinear_MS\d+_Weather_192_DLinear_.*multiseed_h192",
    "PatchTST": r"PatchTST_MS\d+_Weather_192_PatchTST_.*multiseed_h192",
    "PatchFusionBERT_v0": r"PFB_v0_MS\d+_Weather_192_PatchFusionBERT_v0_.*multiseed_h192",
}


def discover_folders(pattern: str) -> list[Path]:
    compiled = re.compile(pattern)
    return sorted(f for f in RESULTS_DIR.iterdir() if f.is_dir() and compiled.search(f.name))


def per_variable_mse(folder: Path) -> np.ndarray | None:
    """Return per-variable MSE array of shape (n_channels,)."""
    pred_path = folder / "pred.npy"
    true_path = folder / "true.npy"
    if not pred_path.exists() or not true_path.exists():
        return None
    pred = np.load(str(pred_path))   # (n_samples, pred_len, n_channels)
    true = np.load(str(true_path))
    return ((pred - true) ** 2).mean(axis=(0, 1))  # (n_channels,)


def compute_model_curves() -> dict[str, np.ndarray]:
    """Return {model_name: mean_per_variable_mse (21,)} averaged over seeds."""
    curves: dict[str, np.ndarray] = {}
    for model_name, pattern in MODELS.items():
        folders = discover_folders(pattern)
        if not folders:
            print(f"  No folders found for {model_name}")
            continue
        seed_curves: list[np.ndarray] = []
        for folder in folders:
            c = per_variable_mse(folder)
            if c is not None:
                seed_curves.append(c)
        if seed_curves:
            mean_curve = np.mean(seed_curves, axis=0)
            curves[model_name] = mean_curve
            print(f"  {model_name}: {len(seed_curves)} seeds, "
                  f"overall MSE={mean_curve.mean():.4f}, "
                  f"max variable MSE={mean_curve.max():.4f} ({WEATHER_FEATURE_NAMES[mean_curve.argmax()]})")
    return curves


def build_dataframe(curves: dict[str, np.ndarray]) -> pd.DataFrame:
    n_vars = len(WEATHER_FEATURE_NAMES)
    rows = []
    for var_idx in range(n_vars):
        row: dict = {"variable_idx": var_idx, "variable": WEATHER_FEATURE_NAMES[var_idx]}
        for model_name, curve in curves.items():
            row[f"mse_{model_name}"] = float(curve[var_idx])
        # relative gain: PFBv0 vs PatchTST backbone
        if "PatchFusionBERT_v0" in curves and "PatchTST" in curves:
            ptst_mse = curves["PatchTST"][var_idx]
            pfb_mse = curves["PatchFusionBERT_v0"][var_idx]
            if ptst_mse > 0:
                row["pfbv0_vs_patchtst_gain_pct"] = (ptst_mse - pfb_mse) / ptst_mse * 100
            else:
                row["pfbv0_vs_patchtst_gain_pct"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing per-variable MSE for Weather H=192 ...")
    curves = compute_model_curves()

    if not curves:
        print("No results found.")
        return

    df = build_dataframe(curves)
    out_csv = OUT_DIR / "per_variable_weather_error.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Print summary table
    if "pfbv0_vs_patchtst_gain_pct" in df.columns:
        df_sorted = df.sort_values("pfbv0_vs_patchtst_gain_pct", ascending=False)
        print("\nPer-variable % gain of PFBv0 over PatchTST (positive = PFBv0 wins):")
        print(f"{'Variable':<22} {'PatchTST MSE':>13} {'PFBv0 MSE':>10} {'Gain %':>8}")
        print("-" * 58)
        for _, row in df_sorted.iterrows():
            ptst = row.get("mse_PatchTST", float("nan"))
            pfb = row.get("mse_PatchFusionBERT_v0", float("nan"))
            gain = row["pfbv0_vs_patchtst_gain_pct"]
            mark = "+" if gain > 0 else ""
            print(f"{row['variable']:<22} {ptst:>13.5f} {pfb:>10.5f} {mark}{gain:>7.2f}%")

        n_win = (df["pfbv0_vs_patchtst_gain_pct"] > 0).sum()
        n_lose = (df["pfbv0_vs_patchtst_gain_pct"] < 0).sum()
        print(f"\nPFBv0 improves on {n_win}/21 variables, hurts {n_lose}/21 variables")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: absolute MSE per variable
        ax = axes[0]
        x = np.arange(len(WEATHER_FEATURE_NAMES))
        colors = {"DLinear": "gray", "PatchTST": "steelblue", "PatchFusionBERT_v0": "tomato"}
        width = 0.25
        offsets = {"DLinear": -width, "PatchTST": 0, "PatchFusionBERT_v0": width}
        for model_name, curve in curves.items():
            ax.bar(x + offsets.get(model_name, 0), curve,
                   width=width, label=model_name,
                   color=colors.get(model_name, "black"), alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(WEATHER_FEATURE_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Per-Variable MSE")
        ax.set_title("Absolute MSE per Weather Variable")
        ax.legend(fontsize=8)

        # Right: relative gain PFBv0 vs PatchTST
        ax2 = axes[1]
        if "pfbv0_vs_patchtst_gain_pct" in df.columns:
            df_sorted2 = df.sort_values("pfbv0_vs_patchtst_gain_pct", ascending=True)
            colors_bar = ["tomato" if g > 0 else "steelblue"
                          for g in df_sorted2["pfbv0_vs_patchtst_gain_pct"]]
            ax2.barh(df_sorted2["variable"], df_sorted2["pfbv0_vs_patchtst_gain_pct"],
                     color=colors_bar, alpha=0.8)
            ax2.axvline(0, color="black", linewidth=0.8)
            ax2.set_xlabel("% Gain (positive = PFBv0 better)")
            ax2.set_title("Per-Variable Gain: PFBv0 vs PatchTST")
            ax2.tick_params(axis="y", labelsize=7)

        fig.tight_layout()
        plot_path = OUT_DIR / "per_variable_weather_error.pdf"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
