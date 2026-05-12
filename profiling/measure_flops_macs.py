"""
FLOPs / MACs Measurement Script
==================================
Measure forward-pass FLOPs (multiply-accumulate operations x2) for each model
using the `thop` library. Mirrors the model configuration in
benchmark_inference_weather192.py.

Usage:
    python measure_flops_macs.py [--batch_size 1]

Output:
    results_analysis/flops_macs.csv  (and printed table)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from thop import profile, clever_format

from models import DLinear, PatchFusionBERT_v0, PatchFusionBERT_v2, PatchTST

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results_analysis"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = 336
LABEL_LEN = 96
PRED_LEN = 192
ENC_IN = 21


class Args:
    def __init__(self, model_name: str):
        self.task_name = "long_term_forecast"
        self.model = model_name
        self.seq_len = SEQ_LEN
        self.pred_len = PRED_LEN
        self.label_len = LABEL_LEN
        self.features = "M"
        self.freq = "h"
        self.enc_in = ENC_IN
        self.dec_in = ENC_IN
        self.c_out = ENC_IN
        self.d_model = 128
        self.n_heads = 8
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 512
        self.moving_avg = 25
        self.factor = 3
        self.dropout = 0.1
        self.activation = "gelu"
        self.embed = "timeF"
        self.distil = True
        self.expand = 2
        self.d_conv = 4
        self.top_k = 5
        self.num_kernels = 6
        self.channel_independence = 1
        self.decomp_method = "moving_avg"
        self.use_norm = 1
        self.down_sampling_layers = 0
        self.down_sampling_window = 1
        self.down_sampling_method = None
        self.seg_len = 96
        self.num_class = 1
        self.patch_len = 16
        self.stride = 8
        self.individual = False
        self.pfb_k = 0


MODEL_SPECS: list[tuple[str, type, dict]] = [
    ("DLinear",             DLinear,           {}),
    ("PatchTST",            PatchTST,          {}),
    ("PatchTST_capacity",   PatchTST,          {"d_model": 192, "e_layers": 5, "d_ff": 512}),
    ("PatchFusionBERT_v0",  PatchFusionBERT_v0, {}),
    ("PatchFusionBERT_v2",  PatchFusionBERT_v2, {}),
]


def build_model(model_name: str, module: type, overrides: dict) -> torch.nn.Module:
    args = Args(model_name)
    for k, v in overrides.items():
        setattr(args, k, v)
    return module.Model(args).float().to(DEVICE)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_model(
    model_name: str, module: type, overrides: dict, batch_size: int
) -> dict:
    model = build_model(model_name, module, overrides)
    model.eval()

    # Inputs: (batch, seq_len, enc_in) for x_enc; zeros for time marks
    x_enc = torch.randn(batch_size, SEQ_LEN, ENC_IN, device=DEVICE)
    x_mark_enc = torch.zeros(batch_size, SEQ_LEN, 4, device=DEVICE)
    x_dec = torch.zeros(batch_size, LABEL_LEN + PRED_LEN, ENC_IN, device=DEVICE)
    x_mark_dec = torch.zeros(batch_size, LABEL_LEN + PRED_LEN, 4, device=DEVICE)

    with torch.no_grad():
        macs, params = profile(
            model,
            inputs=(x_enc, x_mark_enc, x_dec, x_mark_dec),
            verbose=False,
        )

    flops = macs * 2  # MACs → FLOPs (1 MAC = 1 multiply + 1 add)
    n_params = count_params(model)

    macs_fmt, params_fmt = clever_format([macs, params], "%.3f")
    flops_fmt, _ = clever_format([flops, params], "%.3f")

    return {
        "model": model_name,
        "batch_size": batch_size,
        "macs": int(macs),
        "flops": int(flops),
        "thop_params": int(params),
        "actual_params": n_params,
        "macs_fmt": macs_fmt,
        "flops_fmt": flops_fmt,
    }


def print_table(rows: list[dict]) -> None:
    hdr = f"{'Model':<25} {'Params':>12} {'MACs':>12} {'FLOPs':>12}"
    print("\n" + "=" * len(hdr))
    print("FLOPs / MACs Report — Weather H=192 configuration")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['model']:<25} {r['actual_params']:>12,} "
            f"{r['macs_fmt']:>12} {r['flops_fmt']:>12}"
        )
    print("=" * len(hdr))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for profiling (default 1 for per-sample MACs)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for model_name, module, overrides in MODEL_SPECS:
        print(f"Profiling {model_name} ...", end=" ", flush=True)
        try:
            row = measure_model(model_name, module, overrides, args.batch_size)
            rows.append(row)
            print(f"MACs={row['macs_fmt']} FLOPs={row['flops_fmt']}")
        except Exception as exc:
            print(f"FAILED: {exc}")

    print_table(rows)

    out_csv = OUT_DIR / "flops_macs.csv"
    fieldnames = ["model", "batch_size", "macs", "flops", "thop_params",
                  "actual_params", "macs_fmt", "flops_fmt"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
