"""Regenerate the benchmark heatmap at publication scale, without training.

Only displayed manuscript means are used. No seed dispersion is inferred from
rounded table entries. Regeneration does not constitute a new experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather"]
HORIZONS = [96, 192, 336]
MODELS = ["DLinear", "PatchTST", "iTransformer", "TiDE", "TimeXer", "PFB-Direct", "PFB-Projected"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--native-tex", type=Path)
    source_group.add_argument("--displayed-means", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    if args.native_tex:
        text = args.native_tex.read_text(encoding="utf-8")
        block = text.split(r"\label{tab:main_benchmark}", 1)[1].split(r"\end{table*}", 1)[0]
        rows, dataset = [], None
        for line in block.splitlines():
            match = re.search(r"\\multirow\{7\}\{\*\}\{([^}]+)\}", line)
            if match:
                dataset = match[1]
            if line.startswith("&"):
                fields = line.rstrip("\\").split("&")
                for horizon, value in zip(HORIZONS, fields[2:]):
                    mse, mae = map(float, re.findall(r"\d+\.\d+", value))
                    rows.append(dict(dataset=dataset, horizon=horizon, model=fields[1].strip(), mse=mse, mae=mae))
        displayed = pd.DataFrame(rows)
    else:
        displayed = pd.read_csv(args.displayed_means)
    keys = ["dataset", "horizon", "model"]
    if len(displayed) != 126 or displayed.duplicated(keys).any():
        raise ValueError("Expected the unique 6 x 3 x 7 principal grid")
    index = pd.MultiIndex.from_product([DATASETS, HORIZONS], names=["dataset", "horizon"])
    matrix = displayed.pivot(index=["dataset", "horizon"], columns="model", values="mse").reindex(index=index, columns=MODELS)
    if matrix.isna().any().any() or not (matrix >= 0).all().all():
        raise ValueError("Incomplete or invalid benchmark grid")
    rank = matrix.rank(axis=1, method="min", ascending=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2,
                         "axes.titlesize": 8, "axes.labelsize": 7.2,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, axis = plt.subplots(figsize=(3.45, 3.4), layout="constrained")
        im = axis.imshow(rank.to_numpy(), cmap="viridis", vmin=1, vmax=7, aspect="auto")
        axis.set_yticks(range(18), [f"{d} / {h}" for d, h in matrix.index], fontsize=6.1)
        axis.set_xticks(range(7), ["DLinear", "PatchTST", "iTransformer", "TiDE", "TimeXer", "PFB-Direct", "PFB-Projected"],
                        rotation=65, ha="right", fontsize=6.1)
        axis.tick_params(length=0, pad=2)
        for i in range(18):
            for j in range(7):
                axis.text(j, i, f"{matrix.iloc[i, j]:.4f}", ha="center", va="center", fontsize=5.8,
                          color="white" if rank.iloc[i, j] <= 3 else "black")
        for y in (2.5, 5.5, 8.5, 11.5, 14.5):
            axis.axhline(y, color="white", linewidth=0.6)
        for x in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
            axis.axvline(x, color="white", linewidth=0.2, alpha=0.5)
        cb = fig.colorbar(im, ax=axis, pad=0.015, shrink=0.85, ticks=range(1, 8), fraction=0.035)
        cb.set_label("Within-task rank (1 = lowest MSE)", fontsize=6)
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        fig.savefig(args.out_dir / "main_heatmap.png", dpi=400)
        plt.close(fig)

    glyphs = [str(w.message) for w in caught if "Glyph" in str(w.message) and "missing" in str(w.message)]
    if glyphs:
        raise ValueError(glyphs)
    displayed.to_csv(args.audit_dir / "readable_benchmark_displayed_means.csv", index=False)
    report = {"heatmap_cells": 126, "uncertainty": "No dispersion or confidence intervals inferred from rounded means",
              "displayed_means_from_manuscript": True, "heatmap_rank_method": "min, ascending, within task",
              "source_sha256": {str(args.native_tex or args.displayed_means): sha(args.native_tex or args.displayed_means)},
              "output_sha256": {"main_heatmap.png": sha(args.out_dir / "main_heatmap.png")},
              "training_launched": False, "experiment_hardware_provenance_changed": False, "missing_glyphs": glyphs}
    (args.audit_dir / "readable_benchmark_figure_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
