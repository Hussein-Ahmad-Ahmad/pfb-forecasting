import csv
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "results_analysis" / "wallclock_logs"
OUT_CSV = ROOT / "results_analysis" / "wallclock_weather192.csv"
OUT_TEX = ROOT / "results_analysis" / "wallclock_weather192.tex"

EPOCH_RE = re.compile(r"Epoch:\s*(\d+)\s*cost time:\s*([0-9.]+)")
META_RE = re.compile(r"^(MODEL|SEED|GPU)=(.*)$")


def parse_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    meta = {}
    for line in text.splitlines():
        m = META_RE.match(line.strip())
        if m:
            meta[m.group(1)] = m.group(2).strip()

    epochs = [(int(m.group(1)), float(m.group(2))) for m in EPOCH_RE.finditer(text)]
    if not epochs:
        return None

    epoch_times = [t for _, t in epochs]
    return {
        "model": meta.get("MODEL", path.stem.split("_MS")[0]),
        "seed": int(meta.get("SEED", re.search(r"MS(\d+)", path.stem).group(1))),
        "gpu": meta.get("GPU", "unknown"),
        "epochs_ran": len(epoch_times),
        "seconds_per_epoch_mean": statistics.mean(epoch_times),
        "seconds_per_epoch_std_within_run": statistics.pstdev(epoch_times) if len(epoch_times) > 1 else 0.0,
        "total_training_seconds": sum(epoch_times),
    }


def main():
    rows = []
    for path in sorted(LOG_DIR.glob("*.log")):
        parsed = parse_log(path)
        if parsed:
            rows.append(parsed)

    if not rows:
        raise SystemExit("No parsed wall-clock logs found.")

    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)

    summary = []
    for model, model_rows in sorted(grouped.items()):
        means = [r["seconds_per_epoch_mean"] for r in model_rows]
        total_seconds = [r["total_training_seconds"] for r in model_rows]
        epochs = [r["epochs_ran"] for r in model_rows]
        gpu = model_rows[0]["gpu"]
        summary.append({
            "model": model,
            "runs": len(model_rows),
            "epochs_ran_mean": round(statistics.mean(epochs), 2),
            "seconds_per_epoch_mean": round(statistics.mean(means), 4),
            "seconds_per_epoch_std": round(statistics.stdev(means), 4) if len(means) > 1 else 0.0,
            "total_training_seconds_mean": round(statistics.mean(total_seconds), 2),
            "gpu": gpu,
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "runs",
                "epochs_ran_mean",
                "seconds_per_epoch_mean",
                "seconds_per_epoch_std",
                "total_training_seconds_mean",
                "gpu",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Runs & Avg Epochs & Sec/Epoch & Total Sec \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            f"{row['model']} & {row['runs']} & {row['epochs_ran_mean']} & "
            f"{row['seconds_per_epoch_mean']:.4f} $\\pm$ {row['seconds_per_epoch_std']:.4f} & "
            f"{row['total_training_seconds_mean']:.2f} \\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved CSV to {OUT_CSV}")
    print(f"Saved LaTeX to {OUT_TEX}")
    if summary:
        print(f"GPU: {summary[0]['gpu']}")
        for row in summary:
            print(
                f"{row['model']}: {row['seconds_per_epoch_mean']:.4f} ± {row['seconds_per_epoch_std']:.4f} s/epoch "
                f"over {row['runs']} runs"
            )


if __name__ == "__main__":
    main()