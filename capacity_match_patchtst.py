"""Utilities to build a capacity-matched PatchTST control.

Goal:
- Find PatchTST (plain) hyperparameters (d_model/e_layers/d_ff/n_heads) that yield
  a parameter count close to PatchFusionBERT (v0/v2) under the same (seq_len, pred_len, enc_in).

This script does NOT run training. It only instantiates models and counts parameters.

Usage examples:
  python capacity_match_patchtst.py --dataset ETTh2 --seq_len 336 --pred_len 192 --enc_in 7
  python capacity_match_patchtst.py --dataset Weather --seq_len 336 --pred_len 192 --enc_in 21 --target_params 3500000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class ModelConfig:
    model: str
    d_model: int
    n_heads: int
    e_layers: int
    d_ff: int


class _Args:
    """Minimal configs object expected by the model implementations."""

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


def _build_model(model_name: str, args: _Args) -> torch.nn.Module:
    # Import locally so this script can be run from Time-Series-Library/.
    from models import PatchTST, PatchFusionBERT_v0, PatchFusionBERT_v2

    model_map = {
        "PatchTST": PatchTST,
        "PatchFusionBERT_v0": PatchFusionBERT_v0,
        "PatchFusionBERT_v2": PatchFusionBERT_v2,
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model '{model_name}'. Expected one of: {sorted(model_map.keys())}")

    return model_map[model_name].Model(args).float()


def _suggest_grid() -> Tuple[List[int], List[int], List[int], List[int]]:
    # Conservative search space: keep head count fixed (8) and only scale capacity.
    d_models = [128, 160, 192, 224, 256, 288, 320, 384]
    n_heads = [8]
    e_layers = [3, 4, 5, 6, 7, 8]
    d_ff = [512, 768, 1024, 1536, 2048]
    return d_models, n_heads, e_layers, d_ff


def _match_capacity(
    *,
    seq_len: int,
    pred_len: int,
    enc_in: int,
    target_params: int,
    patch_len: int,
    stride: int,
    top_k: int,
) -> List[Tuple[ModelConfig, int, float]]:
    d_models, n_heads_list, e_layers_list, d_ff_list = _suggest_grid()

    matches: List[Tuple[ModelConfig, int, float]] = []
    for d_model, n_heads, e_layers, d_ff in product(d_models, n_heads_list, e_layers_list, d_ff_list):
        if d_model % n_heads != 0:
            continue

        args = _Args(
            task_name="long_term_forecast",
            seq_len=seq_len,
            pred_len=pred_len,
            label_len=max(1, pred_len // 2),
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=enc_in,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=1,
            d_ff=d_ff,
            factor=3,
            dropout=0.1,
            activation="gelu",
            embed="timeF",
            distil=True,
            patch_len=patch_len,
            stride=stride,
        )

        try:
            model = _build_model("PatchTST", args)
        except Exception:
            # Some combinations might fail due to assumptions elsewhere.
            continue

        params = _count_params(model)
        rel_diff = abs(params - target_params) / float(target_params)
        matches.append((ModelConfig("PatchTST", d_model, n_heads, e_layers, d_ff), params, rel_diff))

    matches.sort(key=lambda x: x[2])
    return matches[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ETTh2")
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--pred_len", type=int, default=192)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--target_model", type=str, default="PatchFusionBERT_v0", choices=["PatchFusionBERT_v0", "PatchFusionBERT_v2"])
    parser.add_argument("--target_params", type=int, default=0, help="Override target params (absolute count). If 0, compute from --target_model.")

    args = parser.parse_args()

    base_args = _Args(
        task_name="long_term_forecast",
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        label_len=max(1, args.pred_len // 2),
        enc_in=args.enc_in,
        dec_in=args.enc_in,
        c_out=args.enc_in,
        d_model=128,
        n_heads=8,
        e_layers=3,
        d_layers=1,
        d_ff=512,
        factor=3,
        dropout=0.1,
        activation="gelu",
        embed="timeF",
        distil=True,
        patch_len=args.patch_len,
        stride=args.stride,
    )

    patchtst_base = _build_model("PatchTST", base_args)
    pfb = _build_model(args.target_model, base_args)

    patchtst_base_params = _count_params(patchtst_base)
    pfb_params = _count_params(pfb) if args.target_params <= 0 else int(args.target_params)

    print("=" * 90)
    print("Capacity match: PatchTST vs PatchFusionBERT")
    print("=" * 90)
    print(f"Dataset={args.dataset}  seq_len={args.seq_len}  pred_len={args.pred_len}  enc_in={args.enc_in}  patch_len={args.patch_len}  stride={args.stride}")
    print(f"PatchTST-base (d_model=128,e_layers=3,d_ff=512,n_heads=8): {patchtst_base_params:,} params")
    print(f"Target={args.target_model}: {pfb_params:,} params")
    print("" )

    matches = _match_capacity(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        enc_in=args.enc_in,
        target_params=pfb_params,
        patch_len=args.patch_len,
        stride=args.stride,
        top_k=args.top_k,
    )

    print(f"Top {len(matches)} PatchTST candidates closest to target params:")
    print("  rank | d_model | e_layers | d_ff  | n_heads | params    | rel_diff")
    for idx, (cfg, params, rel_diff) in enumerate(matches, 1):
        print(f"  {idx:>4d} | {cfg.d_model:>6d} | {cfg.e_layers:>7d} | {cfg.d_ff:>4d} | {cfg.n_heads:>7d} | {params:>8,} | {rel_diff*100:>7.2f}%")

    # Print a copy/paste friendly recommended choice if within 10%.
    if matches:
        best_cfg, best_params, best_rel = matches[0]
        within = best_rel <= 0.10
        print("" )
        print("Recommended config:" if within else "Closest config (not within 10%):")
        print(
            f"  --model PatchTST_capacity --d_model {best_cfg.d_model} --n_heads {best_cfg.n_heads} --e_layers {best_cfg.e_layers} --d_ff {best_cfg.d_ff}"
        )


if __name__ == "__main__":
    main()
