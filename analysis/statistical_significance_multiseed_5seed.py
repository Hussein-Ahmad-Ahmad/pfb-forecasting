from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

import build_multiseed_table as multiseed


SEEDS = [2021, 2022, 2023, 2024, 2025]
COMPARISONS = [
    ("PatchFusionBERT_v0", "PatchTST"),
    ("PatchFusionBERT_v0", "PatchFusionBERT_v2"),
    ("PatchFusionBERT_v0", "DLinear"),
    ("PatchFusionBERT_v2", "PatchTST"),
]
METRICS = ["mse", "mae"]


def flatten_results() -> pd.DataFrame:
    parsed = multiseed.parse_multiseed_results(Path("result_long_term_forecast.txt"), SEEDS)
    rows = []
    for (model, dataset, horizon), seed_map in parsed.items():
        for seed, metrics in sorted(seed_map.items()):
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "horizon": horizon,
                    "seed": seed,
                    "mse": metrics["mse"],
                    "mae": metrics["mae"],
                }
            )
    return pd.DataFrame(rows)


def rank_biserial_from_diffs(differences: pd.Series) -> float:
    non_zero = differences[differences != 0]
    if non_zero.empty:
        return 0.0
    abs_ranks = rankdata(non_zero.abs())
    positive_rank_sum = abs_ranks[non_zero > 0].sum()
    negative_rank_sum = abs_ranks[non_zero < 0].sum()
    total = positive_rank_sum + negative_rank_sum
    if total == 0:
        return 0.0
    return float((positive_rank_sum - negative_rank_sum) / total)


def wilcoxon_summary(a: pd.Series, b: pd.Series) -> dict[str, float | int | str]:
    differences = a - b
    non_zero = differences[differences != 0]
    if non_zero.empty:
        return {
            "n_pairs": int(len(differences)),
            "n_nonzero": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "mean_diff": float(differences.mean()),
            "median_diff": float(differences.median()),
            "rank_biserial": 0.0,
            "winner": "Tie",
            "significant": "No",
        }

    stat = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="auto")
    mean_diff = float(differences.mean())
    median_diff = float(differences.median())
    rank_biserial = rank_biserial_from_diffs(differences)

    if math.isclose(mean_diff, 0.0, abs_tol=1e-12):
        winner = "Tie"
    else:
        winner = "Model_A" if mean_diff < 0 else "Model_B"

    return {
        "n_pairs": int(len(differences)),
        "n_nonzero": int(len(non_zero)),
        "statistic": float(stat.statistic),
        "p_value": float(stat.pvalue),
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "rank_biserial": rank_biserial,
        "winner": winner,
        "significant": "Yes" if stat.pvalue < 0.05 else "No",
    }


def build_pooled_results(frame: pd.DataFrame) -> pd.DataFrame:
    pooled_rows = []
    for model_a, model_b in COMPARISONS:
        subset_a = frame[frame["model"] == model_a]
        subset_b = frame[frame["model"] == model_b]
        merged = subset_a.merge(
            subset_b,
            on=["dataset", "horizon", "seed"],
            suffixes=("_a", "_b"),
        )

        for metric in METRICS:
            summary = wilcoxon_summary(merged[f"{metric}_a"], merged[f"{metric}_b"])
            pooled_rows.append(
                {
                    "comparison": f"{model_a} vs {model_b}",
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric.upper(),
                    **summary,
                    "winner": model_a if summary["winner"] == "Model_A" else model_b if summary["winner"] == "Model_B" else "Tie",
                }
            )

    result_df = pd.DataFrame(pooled_rows)

    # Apply Benjamini-Hochberg FDR correction across all pooled tests
    if len(result_df) > 0:
        raw_pvals = result_df["p_value"].values
        reject, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
        result_df["p_value_fdr"] = np.round(pvals_corrected, 8)
        result_df["significant_fdr"] = ["Yes" if r else "No" for r in reject]

    return result_df


def build_config_results(frame: pd.DataFrame) -> pd.DataFrame:
    config_rows = []
    configs = (
        frame[["dataset", "horizon"]]
        .drop_duplicates()
        .sort_values(["dataset", "horizon"])
        .itertuples(index=False)
    )

    for dataset, horizon in configs:
        config_frame = frame[(frame["dataset"] == dataset) & (frame["horizon"] == horizon)]
        for model_a, model_b in COMPARISONS:
            subset_a = config_frame[config_frame["model"] == model_a]
            subset_b = config_frame[config_frame["model"] == model_b]
            merged = subset_a.merge(subset_b, on=["dataset", "horizon", "seed"], suffixes=("_a", "_b"))
            for metric in METRICS:
                summary = wilcoxon_summary(merged[f"{metric}_a"], merged[f"{metric}_b"])
                config_rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "comparison": f"{model_a} vs {model_b}",
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric": metric.upper(),
                        **summary,
                        "winner": model_a if summary["winner"] == "Model_A" else model_b if summary["winner"] == "Model_B" else "Tie",
                    }
                )

    return pd.DataFrame(config_rows)


def build_markdown(pooled: pd.DataFrame, by_config: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# 5-Seed Significance Testing\n\n")
    lines.append("Method: Wilcoxon signed-rank test on seed-aligned pairs. Lower is better.\n\n")
    lines.append("## Pooled Results\n\n")
    lines.append("| Comparison | Metric | N | Mean Diff (A-B) | p-value | p-FDR | Significant | Sig-FDR | Winner | Rank-Biserial |\n")
    lines.append("|------------|--------|---|-----------------|---------|-------|-------------|---------|--------|---------------|\n")
    has_fdr = "p_value_fdr" in pooled.columns
    for row in pooled.sort_values(["metric", "comparison"]).itertuples(index=False):
        p_fdr = f"{row.p_value_fdr:.6f}" if has_fdr else "n/a"
        sig_fdr = row.significant_fdr if has_fdr else "n/a"
        lines.append(
            f"| {row.comparison} | {row.metric} | {row.n_pairs} | {row.mean_diff:.6f} | {row.p_value:.6f} | {p_fdr} | {row.significant} | {sig_fdr} | {row.winner} | {row.rank_biserial:.4f} |\n"
        )

    lines.append("\n## Per-Configuration Results\n\n")
    lines.append("| Dataset | Horizon | Comparison | Metric | N | Mean Diff (A-B) | p-value | Significant | Winner |\n")
    lines.append("|---------|---------|------------|--------|---|-----------------|---------|-------------|--------|\n")
    ordered = by_config.sort_values(["dataset", "horizon", "metric", "comparison"])
    for row in ordered.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.horizon} | {row.comparison} | {row.metric} | {row.n_pairs} | {row.mean_diff:.6f} | {row.p_value:.6f} | {row.significant} | {row.winner} |\n"
        )

    return "".join(lines)


def main() -> None:
    output_dir = Path("results_analysis")
    output_dir.mkdir(exist_ok=True)

    frame = flatten_results()
    pooled = build_pooled_results(frame)
    by_config = build_config_results(frame)

    pooled_path = output_dir / "multiseed_5seed_significance_pooled.csv"
    config_path = output_dir / "multiseed_5seed_significance_by_config.csv"
    markdown_path = output_dir / "multiseed_5seed_significance_summary.md"

    pooled.to_csv(pooled_path, index=False)
    by_config.to_csv(config_path, index=False)
    markdown_path.write_text(build_markdown(pooled, by_config), encoding="utf-8")

    print(f"Wrote {pooled_path}")
    print(f"Wrote {config_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()