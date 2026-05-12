from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results_analysis"
CHARACTERISTICS_CSV = OUT_DIR / "c1_dataset_characteristics.csv"
GAINS_CSV = OUT_DIR / "c1_fusion_gain_summary.csv"
FIGURE_DIR = OUT_DIR / "publication_figures"
PNG_PATH = FIGURE_DIR / "fig7_mechanistic_analysis.png"
PDF_PATH = FIGURE_DIR / "fig7_mechanistic_analysis.pdf"

COLORS = {
    "ETTm2": "#1f77b4",
    "Weather": "#2a9d8f",
    "Exchange": "#e76f51",
}


def main() -> None:
    characteristics = pd.read_csv(CHARACTERISTICS_CSV)
    gains = pd.read_csv(GAINS_CSV)
    focus = gains[gains["scope"] == "5seed_h192"].merge(characteristics, on="dataset", how="left")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "figure.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "savefig.dpi": 300,
            "figure.dpi": 300,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    for _, row in focus.iterrows():
        color = COLORS[row["dataset"]]
        size = 90 + row["feature_count"] * 18
        axes[0].scatter(
            row["channel_diversity_score"],
            row["fusion_gain_pct"],
            s=size,
            color=color,
            edgecolor="black",
            linewidth=0.9,
            alpha=0.9,
        )
        axes[0].annotate(
            row["dataset"],
            (row["channel_diversity_score"], row["fusion_gain_pct"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

        axes[1].scatter(
            row["top_target_feature_corr"],
            row["fusion_gain_pct"],
            s=size,
            color=color,
            edgecolor="black",
            linewidth=0.9,
            alpha=0.9,
        )
        axes[1].annotate(
            row["dataset"],
            (row["top_target_feature_corr"], row["fusion_gain_pct"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Channel Diversity Score")
    axes[0].set_ylabel("Fusion Gain vs Best Non-Fusion Baseline (%)")
    axes[0].set_title("(a) Diversity vs Fusion Gain", loc="left", fontweight="bold")
    axes[0].grid(True, alpha=0.25, linestyle="--")

    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Top Absolute Target-Feature Correlation")
    axes[1].set_ylabel("Fusion Gain vs Best Non-Fusion Baseline (%)")
    axes[1].set_title("(b) Single-Feature Dominance vs Fusion Gain", loc="left", fontweight="bold")
    axes[1].grid(True, alpha=0.25, linestyle="--")

    fig.suptitle("C1 Mechanistic Analysis: Fusion Helps More When Channels Are More Complementary", fontweight="bold")
    fig.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {PNG_PATH}")
    print(f"Wrote: {PDF_PATH}")


if __name__ == "__main__":
    main()