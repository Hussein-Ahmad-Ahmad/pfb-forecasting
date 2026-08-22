from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results_analysis"
MULTISEED_CSV = OUT_DIR / "multiseed_5seed_summary.csv"
COMPLETE_RESULTS_CSV = OUT_DIR / "complete_results_all_models.csv"

OUT_CHARACTERISTICS_CSV = OUT_DIR / "c1_dataset_characteristics.csv"
OUT_GAINS_CSV = OUT_DIR / "c1_fusion_gain_summary.csv"
OUT_MD = OUT_DIR / "c1_mechanistic_analysis_summary.md"
OUT_TEX = OUT_DIR / "c1_mechanistic_analysis_summary.tex"

FOCUS_DATASETS = {
    "ETTm2": ROOT / "data" / "ETTm2.csv",
    "Weather": ROOT / "data" / "weather.csv",
    "Exchange": ROOT / "data" / "exchange_rate.csv",
}
FUSION_MODELS = ["PFB-Direct", "PFB-Projected"]
BASELINE_MODELS = ["PatchTST", "DLinear"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze C1 mechanistic dataset characteristics and fusion gains"
    )
    parser.add_argument("--multiseed_csv", type=Path, default=MULTISEED_CSV)
    parser.add_argument("--complete_results_csv", type=Path, default=COMPLETE_RESULTS_CSV)
    parser.add_argument("--characteristics_csv", type=Path, default=OUT_CHARACTERISTICS_CSV)
    parser.add_argument("--gains_csv", type=Path, default=OUT_GAINS_CSV)
    parser.add_argument("--summary_md", type=Path, default=OUT_MD)
    parser.add_argument("--summary_tex", type=Path, default=OUT_TEX)
    return parser.parse_args()


def _candidate_lags(freq_minutes: float) -> list[tuple[str, int]]:
    if freq_minutes < 60:
        return [
            ("day", int(round(24 * 60 / freq_minutes))),
            ("week", int(round(7 * 24 * 60 / freq_minutes))),
        ]
    return [("week", 7), ("month", 30)]


def _safe_autocorr(series: pd.Series, lag: int) -> float:
    if lag <= 0 or len(series) <= lag:
        return float("nan")
    return float(series.autocorr(lag=lag))


def _first_pc_share(frame: pd.DataFrame) -> float:
    standardized = (frame - frame.mean()) / (frame.std(ddof=0) + 1e-12)
    matrix = standardized.to_numpy(dtype=float)
    _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    variances = singular_values ** 2
    return float(variances[0] / variances.sum())


def characterize_dataset(dataset: str, csv_path: Path) -> dict[str, float | int | str]:
    df = pd.read_csv(csv_path)
    numeric = df.select_dtypes(include=[np.number]).copy()
    if "OT" not in numeric.columns:
        raise ValueError(f"Dataset {dataset} is missing OT column: {csv_path}")

    target = numeric["OT"]
    features = numeric.drop(columns=["OT"])
    dt = pd.to_datetime(df["date"], errors="coerce")
    if dt.notna().sum() < 2:
        raise ValueError(f"Dataset {dataset} does not have a usable date column")

    freq_minutes = float((dt.iloc[1] - dt.iloc[0]).total_seconds() / 60.0)
    lag1_autocorr = _safe_autocorr(target, 1)
    seasonal_values = {
        label: _safe_autocorr(target, lag)
        for label, lag in _candidate_lags(freq_minutes)
    }
    primary_label, primary_autocorr = max(
        seasonal_values.items(), key=lambda item: abs(item[1])
    )

    corr_matrix = features.corr().abs()
    mask = ~np.eye(len(features.columns), dtype=bool)
    mean_abs_inter_feature_corr = float(corr_matrix.where(mask).stack().mean())
    target_corrs = features.corrwith(target).abs().sort_values(ascending=False)
    first_pc_share = _first_pc_share(features)

    return {
        "dataset": dataset,
        "rows": int(len(df)),
        "feature_count": int(len(features.columns)),
        "sampling_minutes": freq_minutes,
        "target_std": float(target.std()),
        "target_diff_std_ratio": float(target.diff().std() / (target.std() + 1e-12)),
        "lag1_autocorr": lag1_autocorr,
        "primary_season_label": primary_label,
        "primary_season_autocorr": float(primary_autocorr),
        "mean_abs_inter_feature_corr": mean_abs_inter_feature_corr,
        "mean_abs_target_feature_corr": float(target_corrs.mean()),
        "top_target_feature_corr": float(target_corrs.iloc[0]),
        "first_pc_variance_share": first_pc_share,
        "channel_diversity_score": float(1.0 - first_pc_share),
    }


def _best_rows(
    frame: pd.DataFrame,
    model_column: str,
    metric_column: str,
    fusion_models: Iterable[str],
    baseline_models: Iterable[str],
) -> tuple[pd.Series, pd.Series]:
    best_fusion = frame[frame[model_column].isin(list(fusion_models))].sort_values(metric_column).iloc[0]
    best_baseline = frame[frame[model_column].isin(list(baseline_models))].sort_values(metric_column).iloc[0]
    return best_fusion, best_baseline


def build_gain_summary(multiseed: pd.DataFrame, complete_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for dataset in FOCUS_DATASETS:
        subset = multiseed[(multiseed["dataset"] == dataset) & (multiseed["horizon"] == 192)]
        if subset.empty:
            continue
        best_fusion, best_baseline = _best_rows(
            subset,
            model_column="model",
            metric_column="mse_mean",
            fusion_models=FUSION_MODELS,
            baseline_models=BASELINE_MODELS,
        )
        rows.append(
            {
                "dataset": dataset,
                "scope": "5seed_h192",
                "horizon": 192,
                "best_fusion_model": best_fusion["model"],
                "best_fusion_mse": float(best_fusion["mse_mean"]),
                "best_baseline_model": best_baseline["model"],
                "best_baseline_mse": float(best_baseline["mse_mean"]),
                "fusion_gain_pct": float(
                    (best_baseline["mse_mean"] - best_fusion["mse_mean"])
                    / best_baseline["mse_mean"]
                    * 100.0
                ),
            }
        )

    subset = complete_results[
        complete_results["Dataset"].isin(list(FOCUS_DATASETS))
        & complete_results["Model"].isin(FUSION_MODELS + BASELINE_MODELS)
    ]
    for (dataset, horizon), group in subset.groupby(["Dataset", "Horizon"], sort=True):
        if len(group[group["Model"].isin(FUSION_MODELS)]) == 0 or len(group[group["Model"].isin(BASELINE_MODELS)]) == 0:
            continue
        best_fusion, best_baseline = _best_rows(
            group,
            model_column="Model",
            metric_column="MSE_mean",
            fusion_models=FUSION_MODELS,
            baseline_models=BASELINE_MODELS,
        )
        rows.append(
            {
                "dataset": dataset,
                "scope": "single_or_mixed_all_horizons",
                "horizon": int(horizon),
                "best_fusion_model": best_fusion["Model"],
                "best_fusion_mse": float(best_fusion["MSE_mean"]),
                "best_baseline_model": best_baseline["Model"],
                "best_baseline_mse": float(best_baseline["MSE_mean"]),
                "fusion_gain_pct": float(
                    (best_baseline["MSE_mean"] - best_fusion["MSE_mean"])
                    / best_baseline["MSE_mean"]
                    * 100.0
                ),
            }
        )

    gains = pd.DataFrame(rows).sort_values(["scope", "dataset", "horizon"]).reset_index(drop=True)
    return gains


def _focus_interpretation(row: pd.Series) -> str:
    dataset = row["dataset"]
    if dataset == "ETTm2":
        return (
            "Strong short- and medium-range periodicity with moderate channel diversity; "
            "fusion has structured multivariate context to exploit."
        )
    if dataset == "Weather":
        return (
            "Highest channel count and relatively low redundancy; auxiliary sensors look complementary "
            "rather than duplicated, which favors fusion."
        )
    return (
        "Shorter, highly collinear series with very strong persistence; most channels move together, "
        "so fusion adds less beyond strong simple baselines."
    )


def build_markdown(characteristics: pd.DataFrame, gains: pd.DataFrame) -> str:
    lines = ["# C1 Mechanistic Analysis", ""]
    lines.append(
        "Goal: explain why PatchFusion-style models help more on ETTm2 and Weather than on Exchange using dataset characteristics rather than model-internal probes."
    )
    lines.append("")

    focus = gains[gains["scope"] == "5seed_h192"].copy()
    merged = focus.merge(characteristics, on="dataset", how="left")
    merged = merged.sort_values("fusion_gain_pct", ascending=False)

    lines.append("## 5-seed H=192 fusion advantage")
    lines.append("")
    for _, row in merged.iterrows():
        direction = "gain" if row["fusion_gain_pct"] >= 0 else "drop"
        lines.append(
            "- {dataset}: {direction} of {gain:+.2f}% MSE vs best non-fusion baseline ({fusion_model} vs {baseline_model}); {interp}".format(
                dataset=row["dataset"],
                direction=direction,
                gain=row["fusion_gain_pct"],
                fusion_model=row["best_fusion_model"],
                baseline_model=row["best_baseline_model"],
                interp=_focus_interpretation(row),
            )
        )
    lines.append("")

    lines.append("## Dataset characteristics")
    lines.append("")
    for _, row in characteristics.sort_values("dataset").iterrows():
        lines.append(
            "- {dataset}: rows={rows}, features={features}, sampling={freq:.0f} min, lag1 autocorr={lag1:.3f}, {season_label} autocorr={season:.3f}, redundancy={redundancy:.3f}, diversity={diversity:.3f}, top target-feature corr={top_corr:.3f}.".format(
                dataset=row["dataset"],
                rows=int(row["rows"]),
                features=int(row["feature_count"]),
                freq=row["sampling_minutes"],
                lag1=row["lag1_autocorr"],
                season_label=row["primary_season_label"],
                season=row["primary_season_autocorr"],
                redundancy=row["mean_abs_inter_feature_corr"],
                diversity=row["channel_diversity_score"],
                top_corr=row["top_target_feature_corr"],
            )
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- ETTm2 combines strong repeated temporal structure with non-trivial multivariate context, so fusion can improve over plain patching when the model exploits shared periodic patterns across channels."
    )
    lines.append(
        "- Weather has the richest sensor set and lower shared variance concentration than Exchange, so fusion appears most useful when many partially complementary channels must be combined."
    )
    lines.append(
        "- Exchange is the most redundant of the three focus datasets: channels are highly co-moving and the target is strongly explained by a few linear relationships, which reduces the marginal value of fusion relative to simpler baselines."
    )
    lines.append("")

    extended = gains[gains["scope"] == "single_or_mixed_all_horizons"].copy()
    if not extended.empty:
        lines.append("## Horizon check from broader aggregated results")
        lines.append("")
        for dataset in ["ETTm2", "Exchange", "Weather"]:
            subset = extended[extended["dataset"] == dataset].sort_values("horizon")
            if subset.empty:
                continue
            parts = [f"H={int(row.horizon)}: {row.fusion_gain_pct:+.2f}%" for row in subset.itertuples()]
            lines.append(f"- {dataset}: " + ", ".join(parts))
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_latex(gains: pd.DataFrame, characteristics: pd.DataFrame) -> str:
    merged = gains[gains["scope"] == "5seed_h192"].merge(characteristics, on="dataset", how="left")
    lines = []
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\toprule")
    lines.append(
        "Dataset & Fusion Gain (\\%) & Features & Lag-1 AC & Redundancy & Diversity & Top Corr \\\\" 
    )
    lines.append("\\midrule")
    for _, row in merged.sort_values("dataset").iterrows():
        lines.append(
            f"{row['dataset']} & {row['fusion_gain_pct']:+.2f} & {int(row['feature_count'])} & {row['lag1_autocorr']:.3f} & {row['mean_abs_inter_feature_corr']:.3f} & {row['channel_diversity_score']:.3f} & {row['top_target_feature_corr']:.3f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.multiseed_csv.exists():
        raise FileNotFoundError(f"Missing multiseed summary: {args.multiseed_csv}")
    if not args.complete_results_csv.exists():
        raise FileNotFoundError(f"Missing complete results summary: {args.complete_results_csv}")

    characteristics = pd.DataFrame(
        [characterize_dataset(dataset, path) for dataset, path in FOCUS_DATASETS.items()]
    )
    multiseed = pd.read_csv(args.multiseed_csv)
    complete_results = pd.read_csv(args.complete_results_csv)
    gains = build_gain_summary(multiseed, complete_results)

    args.characteristics_csv.parent.mkdir(parents=True, exist_ok=True)
    characteristics.to_csv(args.characteristics_csv, index=False)
    gains.to_csv(args.gains_csv, index=False)
    args.summary_md.write_text(build_markdown(characteristics, gains), encoding="utf-8")
    args.summary_tex.write_text(build_latex(gains, characteristics), encoding="utf-8")

    print(f"Wrote: {args.characteristics_csv}")
    print(f"Wrote: {args.gains_csv}")
    print(f"Wrote: {args.summary_md}")
    print(f"Wrote: {args.summary_tex}")
    print("\n5-seed H192 focus:")
    print(gains[gains["scope"] == "5seed_h192"].to_string(index=False))


if __name__ == "__main__":
    main()