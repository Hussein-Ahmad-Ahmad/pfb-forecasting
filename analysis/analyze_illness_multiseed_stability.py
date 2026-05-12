"""Experiment 1 — Illness Multi-Seed Stability.

Reads CapMatch Illness results from result_long_term_forecast.txt (and ../result_long_term_forecast.txt)
then computes mean/std across seeds (2021/2022/2023) for each model+horizon.

Outputs:
  - results_analysis/illness_multiseed_stability.csv
  - results_analysis/illness_multiseed_stability.tex

Expected:
  - model_id contains CapMatch_MS{seed}_Illness_{H}
  - models: PatchFusionBERT_v0, PatchFusionBERT_v2, PatchTST_base, PatchTST_capacity
  - horizons: 24, 48, 60

Run:
  python analyze_illness_multiseed_stability.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RESULT_FILES = [
    Path("result_long_term_forecast.txt"),
    Path("..") / "result_long_term_forecast.txt",
]

MODELS = [
    "PatchFusionBERT_v0",
    "PatchFusionBERT_v2",
    "PatchTST_base",
    "PatchTST_capacity",
]

HORIZONS = [24, 48, 60]
SEEDS = [2021, 2022, 2023]

OUT_DIR = Path("results_analysis")
OUT_CSV = OUT_DIR / "illness_multiseed_stability.csv"
OUT_TEX = OUT_DIR / "illness_multiseed_stability.tex"


def _parse_seed(exp_name: str) -> int | None:
    m = re.search(r"MS(\d{4})", exp_name)
    return int(m.group(1)) if m else None


def _parse_model(exp_name: str) -> str | None:
    for m in MODELS:
        if f"_{m}_" in exp_name:
            return m
    return None


def _parse_horizon(exp_name: str) -> int | None:
    # We key off model_id naming we used: CapMatch_MS{seed}_Illness_{H}
    m = re.search(r"CapMatch_MS\d{4}_Illness_(\d+)", exp_name)
    return int(m.group(1)) if m else None


def _iter_capmatch_lines(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line.startswith("long_term_forecast_"):
            continue
        if "CapMatch_" not in line:
            continue
        if "_Illness_" not in line:
            continue

        # metrics can be on same line or next line
        metrics_text = line
        if "mse:" not in metrics_text or "mae:" not in metrics_text:
            maybe_next = lines[i + 1].strip() if i + 1 < len(lines) else ""
            metrics_text = f"{metrics_text} {maybe_next}".strip()

        mse_m = re.search(r"mse:([\d.]+)", metrics_text)
        mae_m = re.search(r"mae:([\d.]+)", metrics_text)
        if not (mse_m and mae_m):
            continue

        yield line, float(mse_m.group(1)), float(mae_m.group(1))


def to_latex(df: pd.DataFrame) -> str:
    # Columns: Horizon, Model, MSE_mean, MSE_std, MAE_mean, MAE_std
    df2 = df.copy()
    for c in ["MSE_mean", "MSE_std", "MAE_mean", "MAE_std"]:
        df2[c] = df2[c].map(lambda x: f"{x:.4f}")

    lines: list[str] = []
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("Horizon & Model & MSE (mean) & MSE (std) & MAE (mean) & MAE (std) \\")
    lines.append("\\midrule")
    for _, r in df2.iterrows():
        lines.append(
            f"{int(r['Horizon'])} & {r['Model']} & {r['MSE_mean']} & {r['MSE_std']} & {r['MAE_mean']} & {r['MAE_std']} \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    existing = [p for p in RESULT_FILES if p.exists()]
    if not existing:
        raise FileNotFoundError("No result_long_term_forecast.txt found (Time-Series-Library/ or repo root)")

    rows = []
    for p in existing:
        for exp_name, mse, mae in _iter_capmatch_lines(p):
            seed = _parse_seed(exp_name)
            model = _parse_model(exp_name)
            horizon = _parse_horizon(exp_name)
            if seed is None or model is None or horizon is None:
                continue
            if seed not in SEEDS or horizon not in HORIZONS:
                continue
            rows.append(
                {
                    "Seed": seed,
                    "Horizon": horizon,
                    "Model": model,
                    "MSE": mse,
                    "MAE": mae,
                    "Source": str(p),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No CapMatch Illness entries found.")

    # If duplicates exist across files, dedupe by (Seed,Horizon,Model,MSE,MAE)
    df = df.drop_duplicates(subset=["Seed", "Horizon", "Model", "MSE", "MAE"]).copy()

    # Ensure we only keep one value per (seed,horizon,model) if duplicated.
    df = df.sort_values(["Seed", "Horizon", "Model"]).groupby(["Seed", "Horizon", "Model"], as_index=False).head(1)

    agg = (
        df.groupby(["Horizon", "Model"], as_index=False)
        .agg(
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            Runs=("MSE", "count"),
        )
        .sort_values(["Horizon", "Model"])
        .reset_index(drop=True)
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT_CSV, index=False)
    OUT_TEX.write_text(to_latex(agg), encoding="utf-8")

    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_TEX}")
    print("\nPreview:")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
