"""Build a 5-seed summary for the multiseed campaign.

This parser deliberately focuses on the professor-priority campaign scope:
- Standard datasets at H=192
- Illness at H=24/48/60
- Models: PatchFusionBERT_v0, PatchFusionBERT_v2, PatchTST, DLinear

It handles the legacy PatchFusionBERT_v0 / PatchTST / DLinear naming scheme and
the later PatchFusionBERT_v2 naming scheme without mutating the manuscript by
default. Outputs are written to results_analysis/.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


MODELS = ["PatchFusionBERT_v0", "PatchFusionBERT_v2", "PatchTST", "DLinear"]
STANDARD_DATASETS = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "Exchange", "Weather"]
ILLNESS_DATASET = "Illness"
ILLNESS_HORIZONS = [24, 48, 60]
STANDARD_HORIZON = 192
DEFAULT_SEEDS = [2021, 2022, 2023, 2024, 2025]
MANUSCRIPT_HEADER = "# 🔬 Multi-Seed Experiment Results (Detailed)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize multiseed campaign coverage and metrics.")
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("result_long_term_forecast.txt"),
        help="Path to the raw results log.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Expected seed set for coverage calculations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_analysis"),
        help="Directory for markdown/csv outputs.",
    )
    parser.add_argument(
        "--update-markdown",
        action="store_true",
        help="Replace the multiseed section in results_23-01.md.",
    )
    parser.add_argument(
        "--markdown-file",
        type=Path,
        default=Path("results_23-01.md"),
        help="Manuscript file to update when --update-markdown is set.",
    )
    return parser.parse_args()


def normalize_model(name_line: str) -> str | None:
    if "PatchFusionBERT_v0" in name_line or "_PFB_v0_" in name_line:
        return "PatchFusionBERT_v0"
    if "PatchFusionBERT_v2" in name_line:
        return "PatchFusionBERT_v2"
    if "_PatchTST_" in name_line:
        return "PatchTST"
    if "_DLinear_" in name_line:
        return "DLinear"
    return None


def extract_seed(name_line: str) -> int | None:
    match = re.search(r"_MS(20\d{2})_", name_line)
    if match:
        return int(match.group(1))
    return None


def extract_dataset(name_line: str) -> str | None:
    for dataset in STANDARD_DATASETS + [ILLNESS_DATASET]:
        if f"_{dataset}_" in name_line:
            return dataset
    return None


def extract_horizon(name_line: str) -> int | None:
    match = re.search(r"_pl(\d+)_", name_line)
    if match:
        return int(match.group(1))
    return None


def in_scope(name_line: str, model: str | None, dataset: str | None, horizon: int | None) -> bool:
    if not name_line.startswith("long_term_forecast_"):
        return False
    if "CapMatch" in name_line or "capmatch" in name_line:
        return False
    if model not in MODELS or dataset is None or horizon is None:
        return False
    if dataset in STANDARD_DATASETS:
        return horizon == STANDARD_HORIZON
    if dataset == ILLNESS_DATASET:
        return horizon in ILLNESS_HORIZONS
    return False


def parse_multiseed_results(results_file: Path, expected_seeds: list[int]) -> dict[tuple[str, str, int], dict[int, dict[str, float]]]:
    results: dict[tuple[str, str, int], dict[int, dict[str, float]]] = defaultdict(dict)
    with results_file.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    index = 0
    while index < len(lines):
        name_line = lines[index].strip()
        if not name_line:
            index += 1
            continue

        metrics_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
        index += 2

        seed = extract_seed(name_line)
        if seed not in expected_seeds:
            continue

        model = normalize_model(name_line)
        dataset = extract_dataset(name_line)
        horizon = extract_horizon(name_line)
        if not in_scope(name_line, model, dataset, horizon):
            continue

        mse_match = re.search(r"mse:([\d.]+)", metrics_line)
        mae_match = re.search(r"mae:([\d.]+)", metrics_line)
        if not mse_match or not mae_match:
            continue

        key = (model, dataset, horizon)
        results[key][seed] = {
            "mse": float(mse_match.group(1)),
            "mae": float(mae_match.group(1)),
        }

    return results


def mean_and_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def build_rows(results: dict[tuple[str, str, int], dict[int, dict[str, float]]], expected_seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in STANDARD_DATASETS + [ILLNESS_DATASET]:
        horizons = [STANDARD_HORIZON] if dataset in STANDARD_DATASETS else ILLNESS_HORIZONS
        for horizon in horizons:
            for model in MODELS:
                seed_map = results.get((model, dataset, horizon), {})
                present_seeds = sorted(seed_map)
                mse_values = [seed_map[seed]["mse"] for seed in present_seeds]
                mae_values = [seed_map[seed]["mae"] for seed in present_seeds]
                mse_mean, mse_std = mean_and_std(mse_values)
                mae_mean, mae_std = mean_and_std(mae_values)
                rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "horizon": horizon,
                        "seed_count": len(present_seeds),
                        "expected_seed_count": len(expected_seeds),
                        "present_seeds": ",".join(str(seed) for seed in present_seeds),
                        "mse_mean": mse_mean,
                        "mse_std": mse_std,
                        "mae_mean": mae_mean,
                        "mae_std": mae_std,
                    }
                )
    return rows


def coverage_status(seed_count: int, expected_seed_count: int) -> str:
    if seed_count == expected_seed_count:
        return "COMPLETE"
    if seed_count == 0:
        return "MISSING"
    return f"PARTIAL ({seed_count}/{expected_seed_count})"


def format_metric(mean_value: float | None, std_value: float | None) -> str:
    if mean_value is None or std_value is None:
        return "-"
    return f"{mean_value:.4f}+-{std_value:.4f}"


def build_markdown(rows: list[dict[str, object]], expected_seeds: list[int]) -> str:
    total_complete = sum(1 for row in rows if row["seed_count"] == row["expected_seed_count"])
    md: list[str] = []
    md.append("---\n")
    md.append(f"{MANUSCRIPT_HEADER}\n\n")
    md.append(f"Expected seeds: {', '.join(str(seed) for seed in expected_seeds)}\n\n")
    md.append(
        f"Coverage: {total_complete}/{len(rows)} configurations complete "
        f"({100.0 * total_complete / len(rows):.1f}%)\n\n"
    )

    md.append("## Standard Datasets (H=192)\n\n")
    md.append("| Dataset | Model | Seeds | MSE | MAE | Status |\n")
    md.append("|---------|-------|-------|-----|-----|--------|\n")
    for dataset in STANDARD_DATASETS:
        for model in MODELS:
            row = next(item for item in rows if item["dataset"] == dataset and item["horizon"] == STANDARD_HORIZON and item["model"] == model)
            md.append(
                f"| {dataset} | {model} | {row['seed_count']}/{row['expected_seed_count']} | "
                f"{format_metric(row['mse_mean'], row['mse_std'])} | "
                f"{format_metric(row['mae_mean'], row['mae_std'])} | "
                f"{coverage_status(row['seed_count'], row['expected_seed_count'])} |\n"
            )

    md.append("\n## Illness Dataset (H=24/48/60)\n\n")
    md.append("| Horizon | Model | Seeds | MSE | MAE | Status |\n")
    md.append("|---------|-------|-------|-----|-----|--------|\n")
    for horizon in ILLNESS_HORIZONS:
        for model in MODELS:
            row = next(item for item in rows if item["dataset"] == ILLNESS_DATASET and item["horizon"] == horizon and item["model"] == model)
            md.append(
                f"| {horizon} | {model} | {row['seed_count']}/{row['expected_seed_count']} | "
                f"{format_metric(row['mse_mean'], row['mse_std'])} | "
                f"{format_metric(row['mae_mean'], row['mae_std'])} | "
                f"{coverage_status(row['seed_count'], row['expected_seed_count'])} |\n"
            )

    md.append("\n## Coverage By Model\n\n")
    md.append("| Model | Complete Configs | Total Configs | Status |\n")
    md.append("|-------|------------------|---------------|--------|\n")
    configs_per_model = len(STANDARD_DATASETS) + len(ILLNESS_HORIZONS)
    for model in MODELS:
        complete = sum(1 for row in rows if row["model"] == model and row["seed_count"] == row["expected_seed_count"])
        status = "COMPLETE" if complete == configs_per_model else f"{complete}/{configs_per_model}"
        md.append(f"| {model} | {complete} | {configs_per_model} | {status} |\n")

    return "".join(md)


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "dataset",
                "horizon",
                "seed_count",
                "expected_seed_count",
                "present_seeds",
                "mse_mean",
                "mse_std",
                "mae_mean",
                "mae_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def update_markdown(markdown_file: Path, new_section: str) -> None:
    existing = markdown_file.read_text(encoding="utf-8")
    if MANUSCRIPT_HEADER in existing:
        pattern = re.compile(r"---\n# 🔬 Multi-Seed Experiment Results \(Detailed\).*?(?=\n---\n# |\Z)", re.DOTALL)
        updated = pattern.sub(new_section.strip(), existing)
    else:
        updated = existing + "\n\n" + new_section
    markdown_file.write_text(updated, encoding="utf-8")


def main() -> None:
    args = parse_args()
    results = parse_multiseed_results(args.results_file, args.seeds)
    rows = build_rows(results, args.seeds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "multiseed_5seed_summary.csv"
    markdown_path = args.output_dir / "multiseed_5seed_summary.md"

    write_csv(rows, csv_path)
    markdown_text = build_markdown(rows, args.seeds)
    markdown_path.write_text(markdown_text, encoding="utf-8")

    if args.update_markdown:
        update_markdown(args.markdown_file, markdown_text)

    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")
    if args.update_markdown:
        print(f"Updated {args.markdown_file}")


if __name__ == "__main__":
    main()
