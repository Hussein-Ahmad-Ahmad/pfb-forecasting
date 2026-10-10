"""K-depth refinement ablation analyzer for PFB-Direct and PFB-Projected.

Parses result_long_term_forecast.txt (and ../result_long_term_forecast.txt) for runs launched
by the PFB K-depth ablation scripts.

Expected model_id format:
    KDepth_MS2021_<DatasetTag>_<Horizon>_K<k>
where DatasetTag is one of ETTh2, ETTm2, Weather, Illness.

Outputs:
  - results_analysis/pfb_direct_kdepth_ablation.csv
  - results_analysis/pfb_projected_kdepth_ablation.csv

Run:
  python analyze_pfb_kdepth_ablation.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RESULT_FILES = [
    Path("result_long_term_forecast.txt"),
    Path("..") / "result_long_term_forecast.txt",
]

OUT_DIR = Path("results_analysis")
OUT_CSV = OUT_DIR / "pfb_direct_kdepth_ablation.csv"
OUT_CSV_PROJECTED = OUT_DIR / "pfb_projected_kdepth_ablation.csv"
OUT_TEX = OUT_DIR / "pfb_direct_kdepth_ablation.tex"
OUT_TEX_PROJECTED = OUT_DIR / "pfb_projected_kdepth_ablation.tex"


def _iter_lines(path: Path, tag: str = "KDepth_MS"):
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if not s.startswith("long_term_forecast_"):
            continue
        if tag not in s:
            continue
        # metrics can be on same line or next line
        metrics_text = s
        if "mse:" not in metrics_text or "mae:" not in metrics_text:
            maybe_next = lines[i + 1].strip() if i + 1 < len(lines) else ""
            metrics_text = f"{metrics_text} {maybe_next}".strip()
        mse_m = re.search(r"mse:([\d.]+)", metrics_text)
        mae_m = re.search(r"mae:([\d.]+)", metrics_text)
        if not (mse_m and mae_m):
            continue
        yield i, s, float(mse_m.group(1)), float(mae_m.group(1))


def _parse_fields(exp: str, tag: str = "KDepth_MS"):
    prefix = "long_term_forecast_"
    rest = exp[len(prefix):]
    m = re.search(rf"{re.escape(tag)}(\d{{4}})_(ETTh2|ETTm2|Weather|Illness)_(\d+)_K(\d+)", rest)
    if not m:
        return None
    seed = int(m.group(1))
    dataset = m.group(2)
    horizon = int(m.group(3))
    k = int(m.group(4))
    return seed, dataset, horizon, k


def to_latex(df: pd.DataFrame) -> str:
    df2 = df.copy()
    for c in ["MSE", "MAE"]:
        df2[c] = df2[c].map(lambda x: f"{x:.4f}")

    lines: list[str] = []
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Dataset & Horizon & K & MSE / MAE \\")
    lines.append("\\midrule")
    for _, r in df2.iterrows():
        lines.append(f"{r['Dataset']} & {int(r['Horizon'])} & {int(r['K'])} & {r['MSE']} / {r['MAE']} \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def _collect_rows(tag: str) -> list[dict]:
    existing = [p for p in RESULT_FILES if p.exists()]
    if not existing:
        return []
    rows = []
    for p in existing:
        for line_no, exp, mse, mae in _iter_lines(p, tag=tag):
            parsed = _parse_fields(exp, tag=tag)
            if parsed is None:
                continue
            seed, dataset, horizon, k = parsed
            rows.append(
                {
                    "Seed": seed,
                    "Dataset": dataset,
                    "Horizon": horizon,
                    "K": k,
                    "MSE": mse,
                    "MAE": mae,
                    "LineNo": line_no,
                    "Source": str(p),
                }
            )
    return rows


def main() -> None:
    existing = [p for p in RESULT_FILES if p.exists()]
    if not existing:
        raise FileNotFoundError("No result_long_term_forecast.txt found")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for tag, out_csv, out_tex, label in [
        ("KDepth_MS",   OUT_CSV,           OUT_TEX,           "PFB-Direct"),
        ("KDepthProjected_MS", OUT_CSV_PROJECTED, OUT_TEX_PROJECTED, "PFB-Projected"),
    ]:
        rows = _collect_rows(tag)
        if not rows:
            print(f"No {label} K-depth runs found — skipping.")
            continue
        df = pd.DataFrame(rows)
        df = df.sort_values(["Source", "LineNo"]).copy()
        df = df.groupby(["Seed", "Dataset", "Horizon", "K"], as_index=False).tail(1)
        df = df.sort_values(["Dataset", "Horizon", "K"]).reset_index(drop=True)
        df.to_csv(out_csv, index=False)
        out_tex.write_text(to_latex(df), encoding="utf-8")
        print(f"[{label}] Wrote: {out_csv}")
        print(f"[{label}] Wrote: {out_tex}")
        print(f"\n[{label}] Preview:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
