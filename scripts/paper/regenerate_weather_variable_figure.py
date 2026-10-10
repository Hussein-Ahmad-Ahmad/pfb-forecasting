"""Replot the archived Weather H=192 diagnostic with typeset units.

No training is performed and no experiment results or GPU provenance are changed.
The retained gain column is plotted without rounding or recomputing its values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    0: r"p (mbar)",
    1: r"T ($^\circ$C)",
    2: r"T$_{\mathrm{pot}}$ (K)",
    3: r"T$_{\mathrm{dew}}$ ($^\circ$C)",
    4: r"rh (%)",
    5: r"VP$_{\mathrm{max}}$ (mbar)",
    6: r"VP$_{\mathrm{act}}$ (mbar)",
    7: r"VP$_{\mathrm{def}}$ (mbar)",
    8: r"sh (g kg$^{-1}$)",
    9: r"H$_2$OC (mmol mol$^{-1}$)",
    10: r"$\rho$ (g m$^{-3}$)",
    11: r"wv (m s$^{-1}$)",
    12: r"max. wv (m s$^{-1}$)",
    13: r"wd ($^\circ$)",
    14: r"rain (mm)",
    15: r"raining (s)",
    16: r"SWDR (W m$^{-2}$)",
    17: r"PAR ($\mu$mol m$^{-2}$ s$^{-1}$)",
    18: r"max. PAR ($\mu$mol m$^{-2}$ s$^{-1}$)",
    19: r"T$_{\mathrm{log}}$ ($^\circ$C)",
    20: "OT",
}


def generate(source: Path, output: Path) -> dict:
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 21 or {int(row["variable_idx"]) for row in rows} != set(LABELS):
        raise ValueError("Expected exactly the 21 indexed Weather channels")
    for row in rows:
        patch = float(row["mse_PatchTST"])
        direct = float(row["mse_PFB-Direct"])
        gain = float(row["pfb_direct_vs_patchtst_gain_pct"])
        if patch <= 0 or direct < 0 or abs(gain - 100 * (patch - direct) / patch) > 1e-5:
            raise ValueError(f"Inconsistent archived metric for channel {row['variable_idx']}")
    rows.sort(key=lambda row: float(row["pfb_direct_vs_patchtst_gain_pct"]), reverse=True)
    gains = [float(row["pfb_direct_vs_patchtst_gain_pct"]) for row in rows]
    labels = [LABELS[int(row["variable_idx"])] for row in rows]
    if any("\ufffd" in label or "**" in label or "degC" in label for label in labels):
        raise AssertionError("Invalid display typography")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10.5,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "mathtext.fontset": "dejavusans",
    })
    fig, axis = plt.subplots(figsize=(9.2, 6.8))
    axis.barh(range(len(rows)), gains, color=["#16818b" if gain >= 0 else "#e53935" for gain in gains], height=0.8)
    axis.set_yticks(range(len(rows)), labels)
    axis.invert_yaxis()
    axis.axvline(0, color="#263238", linewidth=1)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="0.88", linewidth=0.7)
    axis.set_xlabel("Relative MSE reduction of PFB-Direct vs. PatchTST (%)", labelpad=8)
    axis.set_title(r"Weather, $H=192$: channel-wise MSE reduction", pad=10)
    fig.subplots_adjust(left=0.30, right=0.985, top=0.925, bottom=0.105)
    output.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    glyph_warnings = [str(item.message) for item in caught if "Glyph" in str(item.message) and "missing" in str(item.message)]
    if glyph_warnings:
        raise AssertionError(glyph_warnings)
    return {
        "source_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "output": output.name, "channels": len(rows),
        "gain_values_plotted_without_modification": True,
        "formula_consistency_tolerance_percentage_points": 1e-5,
        "positive_channels": sum(gain > 0 for gain in gains),
        "negative_channels": sum(gain < 0 for gain in gains),
        "missing_glyph_warnings": glyph_warnings,
        "training_launched": False, "gpu_provenance_unchanged": True,
        "display_order": [int(row["variable_idx"]) for row in rows],
        "display_labels": labels,
        "gain_values": gains,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path)
    args = parser.parse_args()
    report = generate(args.source_csv, args.output)
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
