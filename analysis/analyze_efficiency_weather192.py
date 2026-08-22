from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results_analysis"
TRAIN_RAW = OUT_DIR / "efficiency_weather192_training_raw.csv"
INFER_SUMMARY = OUT_DIR / "efficiency_weather192_inference_summary.csv"
OUT_CSV = OUT_DIR / "efficiency_weather192_summary.csv"
OUT_TEX = OUT_DIR / "efficiency_weather192_summary.tex"
OUT_MD = OUT_DIR / "efficiency_weather192_summary.md"


DISPLAY_NAMES = {
    "DLinear": "DLinear",
    "PatchTST": "PatchTST",
    "PatchTST_capacity": "PatchTST_cap",
    "PFB-Direct": "PFB-Direct",
    "PFB-Projected": "PFB-Projected",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_training(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault(row["model"], []).append(row)

    summary: Dict[str, Dict[str, object]] = {}
    for model, model_rows in grouped.items():
        seconds = [float(row["training_seconds"]) for row in model_rows]
        epochs = [float(row["epochs_ran"]) for row in model_rows]
        memory = [float(row["peak_gpu_memory_mb"]) for row in model_rows if row.get("peak_gpu_memory_mb")]
        summary[model] = {
            "training_runs": len(model_rows),
            "gpu_name": model_rows[0]["gpu_name"],
            "train_seconds_median": round(statistics.median(seconds), 4),
            "train_seconds_std": round(statistics.pstdev(seconds) if len(seconds) > 1 else 0.0, 4),
            "epochs_median": round(statistics.median(epochs), 2),
            "train_peak_gpu_memory_median_mb": round(statistics.median(memory), 2) if memory else "",
            "train_peak_gpu_memory_std_mb": round(statistics.pstdev(memory), 2) if len(memory) > 1 else 0.0,
        }
    return summary


def summarize_inference(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for row in rows:
        summary[row["model"]] = {
            "inference_runs": int(row["runs"]),
            "gpu_name": row["gpu_name"],
            "params": int(float(row["params"])),
            "latency_median_ms": float(row["latency_median_ms"]),
            "latency_std_between_runs_ms": float(row["latency_std_between_runs_ms"]),
            "throughput_median_samples_per_sec": float(row["throughput_median_samples_per_sec"]),
            "infer_peak_gpu_memory_median_mb": float(row["peak_gpu_memory_median_mb"]),
            "infer_peak_gpu_memory_std_mb": float(row["peak_gpu_memory_std_mb"]),
        }
    return summary


def build_rows(train_summary: Dict[str, Dict[str, object]], infer_summary: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    models = ["DLinear", "PatchTST", "PatchTST_capacity", "PFB-Direct", "PFB-Projected"]
    rows: List[Dict[str, object]] = []
    for model in models:
        if model not in train_summary or model not in infer_summary:
            continue
        rows.append(
            {
                "model": model,
                "display_model": DISPLAY_NAMES.get(model, model),
                "gpu_name": train_summary[model]["gpu_name"] or infer_summary[model]["gpu_name"],
                "params": infer_summary[model]["params"],
                "training_runs": train_summary[model]["training_runs"],
                "train_seconds_median": train_summary[model]["train_seconds_median"],
                "train_seconds_std": train_summary[model]["train_seconds_std"],
                "epochs_median": train_summary[model]["epochs_median"],
                "train_peak_gpu_memory_median_mb": train_summary[model]["train_peak_gpu_memory_median_mb"],
                "latency_median_ms": infer_summary[model]["latency_median_ms"],
                "latency_std_between_runs_ms": infer_summary[model]["latency_std_between_runs_ms"],
                "throughput_median_samples_per_sec": infer_summary[model]["throughput_median_samples_per_sec"],
                "infer_peak_gpu_memory_median_mb": infer_summary[model]["infer_peak_gpu_memory_median_mb"],
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Efficiency Benchmark Summary (Weather H=192)",
        "",
        "Protocol:",
        "- Training wall-clock: 3 repeated runs (seeds 2021, 2022, 2023)",
        "- Inference latency: 3 repeated benchmark sessions with warmup and timed forward passes",
        "- Same machine and GPU for all models",
        "",
    ]

    for row in rows:
        lines.append(f"## {row['display_model']}")
        lines.append("")
        lines.append(f"- Params: {int(row['params']):,}")
        lines.append(f"- Training wall-clock median: {float(row['train_seconds_median']):.2f} s")
        lines.append(f"- Training wall-clock std: {float(row['train_seconds_std']):.2f} s")
        lines.append(f"- Training peak GPU memory median: {float(row['train_peak_gpu_memory_median_mb']):.2f} MB")
        lines.append(f"- Inference latency median: {float(row['latency_median_ms']):.2f} ms")
        lines.append(f"- Inference latency std between runs: {float(row['latency_std_between_runs_ms']):.2f} ms")
        lines.append(f"- Inference throughput median: {float(row['throughput_median_samples_per_sec']):.2f} samples/s")
        lines.append(f"- Inference peak GPU memory median: {float(row['infer_peak_gpu_memory_median_mb']):.2f} MB")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(path: Path, rows: List[Dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Params & Train Sec & Train Mem(MB) & Infer Lat(ms) & Infer Mem(MB) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['display_model']} & {int(row['params']):,} & "
            f"{float(row['train_seconds_median']):.2f} $\\pm$ {float(row['train_seconds_std']):.2f} & "
            f"{float(row['train_peak_gpu_memory_median_mb']):.2f} & "
            f"{float(row['latency_median_ms']):.2f} $\\pm$ {float(row['latency_std_between_runs_ms']):.2f} & "
            f"{float(row['infer_peak_gpu_memory_median_mb']):.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    train_rows = read_csv(TRAIN_RAW)
    infer_rows = read_csv(INFER_SUMMARY)
    rows = build_rows(summarize_training(train_rows), summarize_inference(infer_rows))
    if not rows:
        raise SystemExit("No efficiency summary rows could be built.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_CSV, rows)
    write_markdown(OUT_MD, rows)
    write_latex(OUT_TEX, rows)

    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_MD}")
    print(f"Wrote: {OUT_TEX}")


if __name__ == "__main__":
    main()