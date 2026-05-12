"""Build the requested control comparison table for capacity-matched PatchTST.

Reads: result_long_term_forecast.txt (appended by Exp_Long_Term_Forecast.test)
Filters: experiments whose model_id contains 'CapMatch_' or 'CapMatchDepth_'
Outputs:
  - results_analysis/capmatch_controls_results.csv
  - results_analysis/capmatch_controls_table.tex
    - results_analysis/capmatch_controls_summary.md

Table columns:
  Model, Params, MSE, MAE

Aggregation:
- Mean over seeds for each (Model, Dataset, Horizon)

Expected model names:
  PatchTST_base
  PatchTST_capacity
    PatchTST_depth
  PatchFusionBERT_v0
  PatchFusionBERT_v2

Run:
  python analyze_capmatch_controls.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


RESULT_FILES = [
    Path("result_long_term_forecast.txt"),
    Path("..") / "result_long_term_forecast.txt",
]
OUT_DIR = Path("results_analysis")
OUT_CSV = OUT_DIR / "capmatch_controls_results.csv"
OUT_TEX = OUT_DIR / "capmatch_controls_table.tex"
OUT_MD = OUT_DIR / "capmatch_controls_summary.md"


MODELS = [
    "PatchTST_base",
    "PatchTST_capacity",
    "PatchTST_depth",
    "PatchFusionBERT_v0",
    "PatchFusionBERT_v2",
]


@dataclass(frozen=True)
class ParsedExperiment:
    exp_name: str
    model: str
    dataset: str
    horizon: int
    d_model: int
    n_heads: int
    e_layers: int
    d_ff: int
    seq_len: int
    pred_len: int
    label_len: int
    patch_len: int
    stride: int
    seed: Optional[int]
    mse: float
    mae: float


class _Args:
    def __init__(
        self,
        *,
        task_name: str,
        seq_len: int,
        pred_len: int,
        label_len: int,
        enc_in: int,
        dec_in: int,
        c_out: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_layers: int,
        d_ff: int,
        factor: int,
        dropout: float,
        activation: str,
        embed: str,
        distil: bool,
        patch_len: int,
        stride: int,
    ):
        self.task_name = task_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len

        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out

        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff

        self.factor = factor
        self.dropout = dropout
        self.activation = activation
        self.embed = embed
        self.distil = distil

        self.patch_len = patch_len
        self.stride = stride


def _count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _infer_dataset(exp_name: str) -> Optional[str]:
    for ds in ["ETTh2", "ETTm2", "Weather", "Illness", "ETTh1", "ETTm1", "Exchange", "Traffic", "Electricity"]:
        if ds in exp_name:
            return ds
    # fall back: if `--data ETTh2` etc appears in the setting
    for ds in ["ETTh2", "ETTm2", "ETTh1", "ETTm1"]:
        if f"_{ds}_" in exp_name:
            return ds
    return None


def _parse_seed(exp_name: str) -> Optional[int]:
    m = re.search(r"MS(\d{4})", exp_name)
    if m:
        return int(m.group(1))
    m = re.search(r"seed(\d{1,6})", exp_name)
    if m:
        return int(m.group(1))
    return None


def _parse_setting(exp_name: str) -> Optional[dict]:
    # run.py setting format includes these tokens:
    # ..._sl{seq}_ll{label}_pl{pred}_dm{d_model}_nh{n_heads}_el{e_layers}_dl{d_layers}_df{d_ff}_...
    patterns = {
        "seq_len": r"_sl(\d+)_",
        "label_len": r"_ll(\d+)_",
        "pred_len": r"_pl(\d+)_",
        "d_model": r"_dm(\d+)_",
        "n_heads": r"_nh(\d+)_",
        "e_layers": r"_el(\d+)_",
        "d_layers": r"_dl(\d+)_",
        "d_ff": r"_df(\d+)_",
    }

    out = {}
    for k, pat in patterns.items():
        m = re.search(pat, exp_name)
        if not m:
            return None
        out[k] = int(m.group(1))

    # Patch parameters may be present multiple ways; default to paper defaults.
    patch_len = 16
    stride = 8
    m = re.search(r"--patch_len\s+(\d+)", exp_name)
    if m:
        patch_len = int(m.group(1))

    out["patch_len"] = patch_len
    out["stride"] = stride
    return out


def _parse_model(exp_name: str) -> Optional[str]:
    # Prefer exact matches to avoid PatchTST matching PatchTST_capacity.
    for m in MODELS:
        if f"_{m}_" in exp_name or f"_{m}" in exp_name:
            return m
    # Fall back to substring checks
    if "PatchFusionBERT_v0" in exp_name:
        return "PatchFusionBERT_v0"
    if "PatchFusionBERT_v2" in exp_name:
        return "PatchFusionBERT_v2"
    if "PatchTST_capacity" in exp_name:
        return "PatchTST_capacity"
    if "PatchTST_depth" in exp_name:
        return "PatchTST_depth"
    if "PatchTST_base" in exp_name:
        return "PatchTST_base"
    return None


def parse_result_file(path: Path) -> List[ParsedExperiment]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: List[ParsedExperiment] = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line.startswith("long_term_forecast_"):
            continue
        if "CapMatch_" not in line and "CapMatchDepth_" not in line:
            continue

        # Logging format varies across runs:
        # - sometimes mse/mae are written on the *next* line
        # - sometimes they are appended on the *same* line
        metrics_text = line
        if "mse:" not in metrics_text or "mae:" not in metrics_text:
            maybe_next = lines[i + 1].strip() if i + 1 < len(lines) else ""
            metrics_text = f"{metrics_text} {maybe_next}".strip()

        mse_m = re.search(r"mse:([\d.]+)", metrics_text)
        mae_m = re.search(r"mae:([\d.]+)", metrics_text)
        if not (mse_m and mae_m):
            continue

        model = _parse_model(line)
        dataset = _infer_dataset(line)
        setting = _parse_setting(line)
        if model is None or dataset is None or setting is None:
            continue

        pred_len = setting["pred_len"]

        out.append(
            ParsedExperiment(
                exp_name=line,
                model=model,
                dataset=dataset,
                horizon=pred_len,
                d_model=setting["d_model"],
                n_heads=setting["n_heads"],
                e_layers=setting["e_layers"],
                d_ff=setting["d_ff"],
                seq_len=setting["seq_len"],
                pred_len=pred_len,
                label_len=setting["label_len"],
                patch_len=setting["patch_len"],
                stride=setting["stride"],
                seed=_parse_seed(line),
                mse=float(mse_m.group(1)),
                mae=float(mae_m.group(1)),
            )
        )

    return out


def compute_param_counts(exps: List[ParsedExperiment]) -> Dict[Tuple[str, int, int, int, int, int], int]:
    """Compute params per unique (model, seq_len, pred_len, d_model, e_layers, d_ff).

    n_heads is included because it changes attention projections.
    """
    from models import PatchTST, PatchFusionBERT_v0, PatchFusionBERT_v2

    model_map = {
        "PatchTST_base": PatchTST,
        "PatchTST_capacity": PatchTST,
        "PatchTST_depth": PatchTST,
        "PatchFusionBERT_v0": PatchFusionBERT_v0,
        "PatchFusionBERT_v2": PatchFusionBERT_v2,
    }

    cache: Dict[Tuple[str, int, int, int, int, int, int], int] = {}

    for e in exps:
        key = (e.model, e.seq_len, e.pred_len, e.d_model, e.n_heads, e.e_layers, e.d_ff)
        if key in cache:
            continue

        # enc_in does not affect params for these heads (shared across vars), but keep consistent.
        args = _Args(
            task_name="long_term_forecast",
            seq_len=e.seq_len,
            pred_len=e.pred_len,
            label_len=e.label_len,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=e.d_model,
            n_heads=e.n_heads,
            e_layers=e.e_layers,
            d_layers=1,
            d_ff=e.d_ff,
            factor=3,
            dropout=0.1,
            activation="gelu",
            embed="timeF",
            distil=True,
            patch_len=e.patch_len,
            stride=e.stride,
        )

        model_cls = model_map[e.model]
        m = model_cls.Model(args).float()
        cache[key] = _count_params(m)

    # Convert to a simpler mapping used downstream
    simplified: Dict[Tuple[str, int, int, int, int, int], int] = {}
    for (model, seq_len, pred_len, d_model, n_heads, e_layers, d_ff), params in cache.items():
        simplified[(model, pred_len, d_model, n_heads, e_layers, d_ff)] = params
    return simplified


def to_latex_table(df: pd.DataFrame) -> str:
    # Minimal LaTeX tabular; keep it self-contained.
    cols = ["Dataset", "Horizon", "Model", "Params", "MSE", "MAE"]
    df2 = df[cols].copy()
    df2["Params"] = df2["Params"].map(lambda x: f"{int(x):,}")
    df2["MSE"] = df2["MSE"].map(lambda x: f"{x:.4f}")
    df2["MAE"] = df2["MAE"].map(lambda x: f"{x:.4f}")

    lines = []
    lines.append("\\begin{tabular}{lllrcc}")
    lines.append("\\toprule")
    lines.append("Dataset & Horizon & Model & Params & MSE & MAE \\\\")
    lines.append("\\midrule")
    for _, r in df2.iterrows():
        lines.append(f"{r['Dataset']} & {r['Horizon']} & {r['Model']} & {r['Params']} & {r['MSE']} & {r['MAE']} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def to_markdown_summary(df: pd.DataFrame) -> str:
    display_names = {
        "PatchTST_base": "PatchTST base",
        "PatchTST_capacity": "PatchTST width-match",
        "PatchTST_depth": "PatchTST depth-match",
        "PatchFusionBERT_v0": "PatchFusionBERT v0",
        "PatchFusionBERT_v2": "PatchFusionBERT v2",
    }

    lines = [
        "# Capacity-Control Summary",
        "",
        "Controls included:",
        "- PatchTST base",
        "- PatchTST width-match",
        "- PatchTST depth-match",
        "- PatchFusionBERT v0",
        "- PatchFusionBERT v2",
        "",
    ]

    for (dataset, horizon), subset in df.groupby(["Dataset", "Horizon"], sort=True):
        lines.append(f"## {dataset} H={horizon}")
        lines.append("")

        ordered = subset.sort_values("MSE").reset_index(drop=True)
        for _, row in ordered.iterrows():
            lines.append(
                "- {model}: params={params:,}, mse={mse:.4f}, mae={mae:.4f}, runs={runs}".format(
                    model=display_names.get(row["Model"], row["Model"]),
                    params=int(row["Params"]),
                    mse=row["MSE"],
                    mae=row["MAE"],
                    runs=int(row["Runs"]),
                )
            )

        pfb_v0 = subset[subset["Model"] == "PatchFusionBERT_v0"]
        lines.append("")
        if not pfb_v0.empty:
            reference = pfb_v0.iloc[0]
            for control in ["PatchTST_base", "PatchTST_capacity", "PatchTST_depth", "PatchFusionBERT_v2"]:
                control_row = subset[subset["Model"] == control]
                if control_row.empty:
                    continue
                compare = control_row.iloc[0]
                lines.append(
                    "- PFB v0 vs {model}: delta_mse={mse:+.4f}, delta_mae={mae:+.4f}".format(
                        model=display_names.get(control, control),
                        mse=compare["MSE"] - reference["MSE"],
                        mae=compare["MAE"] - reference["MAE"],
                    )
                )
        lines.append("")

    coverage = df.groupby("Model")["Runs"].sum().to_dict()
    lines.append("## Coverage")
    lines.append("")
    for model in MODELS:
        if model in coverage:
            lines.append(f"- {display_names.get(model, model)} total seed-runs aggregated: {int(coverage[model])}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    existing = [p for p in RESULT_FILES if p.exists()]
    if not existing:
        tried = ", ".join(str(p) for p in RESULT_FILES)
        raise FileNotFoundError(f"Missing result files. Tried: {tried}. Run experiments first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exps: List[ParsedExperiment] = []
    for p in existing:
        exps.extend(parse_result_file(p))

    # Dedupe in case the same run was logged in multiple places.
    deduped: List[ParsedExperiment] = []
    seen = set()
    for e in exps:
        k = (
            e.model,
            e.dataset,
            e.horizon,
            e.seq_len,
            e.label_len,
            e.pred_len,
            e.d_model,
            e.n_heads,
            e.e_layers,
            e.d_ff,
            e.seed,
            e.mse,
            e.mae,
        )
        if k in seen:
            continue
        seen.add(k)
        deduped.append(e)

    exps = deduped
    if not exps:
        raise RuntimeError("No CapMatch experiments found in result_long_term_forecast.txt")

    params_map = compute_param_counts(exps)

    rows = []
    for e in exps:
        params_key = (e.model, e.pred_len, e.d_model, e.n_heads, e.e_layers, e.d_ff)
        params = params_map.get(params_key)
        if params is None:
            continue
        rows.append(
            {
                "Model": e.model,
                "Dataset": e.dataset,
                "Horizon": e.horizon,
                "Seed": e.seed,
                "Params": params,
                "MSE": e.mse,
                "MAE": e.mae,
                "d_model": e.d_model,
                "n_heads": e.n_heads,
                "e_layers": e.e_layers,
                "d_ff": e.d_ff,
            }
        )

    df = pd.DataFrame(rows)

    # Mean over seeds per setting
    agg = (
        df.groupby(["Dataset", "Horizon", "Model", "Params"], as_index=False)
        .agg(MSE=("MSE", "mean"), MAE=("MAE", "mean"), Runs=("MSE", "count"))
        .sort_values(["Dataset", "Horizon", "Model"])
        .reset_index(drop=True)
    )

    agg.to_csv(OUT_CSV, index=False)

    tex = to_latex_table(agg)
    OUT_TEX.write_text(tex, encoding="utf-8")
    OUT_MD.write_text(to_markdown_summary(agg), encoding="utf-8")

    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_TEX}")
    print(f"Wrote: {OUT_MD}")
    print("\nPreview:")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
