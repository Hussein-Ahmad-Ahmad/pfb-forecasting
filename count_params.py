"""
Compute exact parameter counts for all models used in the paper.
Configs match those used in the experiments.
"""
import sys
sys.path.insert(0, '.')

import torch
import argparse

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def make_args(**kwargs):
    defaults = dict(
        task_name='long_term_forecast',
        seq_len=336, label_len=48, pred_len=192,
        enc_in=7, dec_in=7, c_out=7,
        d_model=128, n_heads=8, e_layers=3, d_layers=1,
        d_ff=512, factor=1, dropout=0.1,
        embed='timeF', freq='h', activation='gelu',
        output_attention=False, distil=True,
        moving_avg=25,
        patch_len=16, stride=8,
        num_kernels=6, top_k=5,
        channel_independence=1,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)

results = []

# ── PatchFusionBERT_v0 ──────────────────────────────────────────────────────
from models.PatchFusionBERT_v0 import Model as PFBv0
for C in [7, 21]:
    a = make_args(enc_in=C, dec_in=C, c_out=C)
    m = PFBv0(a)
    results.append(('PFBv0', C, count_params(m)))

# ── PatchFusionBERT_v2 ──────────────────────────────────────────────────────
from models.PatchFusionBERT_v2 import Model as PFBv2
for C in [7, 21]:
    a = make_args(enc_in=C, dec_in=C, c_out=C)
    m = PFBv2(a)
    results.append(('PFBv2', C, count_params(m)))

# ── PatchTST base (d=128, d_ff=256, n_heads=16, 3 layers) ───────────────────
from models.PatchTST import Model as PatchTST
for C in [7, 21]:
    a = make_args(enc_in=C, dec_in=C, c_out=C, d_model=128, d_ff=256, n_heads=16, e_layers=3)
    m = PatchTST(a)
    results.append(('PatchTST_base', C, count_params(m)))

# ── PatchTST_cap (d=128, d_ff=512, n_heads=8, 3 layers) ─────────────────────
for C in [7, 21]:
    a = make_args(enc_in=C, dec_in=C, c_out=C, d_model=128, d_ff=512, n_heads=8, e_layers=3)
    m = PatchTST(a)
    results.append(('PatchTST_cap', C, count_params(m)))

# ── DLinear ──────────────────────────────────────────────────────────────────
from models.DLinear import Model as DLinear
for C in [7, 21]:
    a = make_args(enc_in=C, dec_in=C, c_out=C, d_model=512)
    m = DLinear(a)
    results.append(('DLinear', C, count_params(m)))

# ── Print ────────────────────────────────────────────────────────────────────
print(f"\n{'Model':<18} {'C':>4} {'Params':>12} {'Params (M)':>12}")
print('-' * 50)
for name, C, n in results:
    print(f"{name:<18} {C:>4} {n:>12,} {n/1e6:>12.3f}M")

print("\n=== Markdown table ===")
print("| Model | C | Parameters | Notes |")
print("|-------|---|-----------|-------|")
seen = set()
for name, C, n in results:
    key = (name, C)
    if key not in seen:
        seen.add(key)
        print(f"| {name} | {C} | {n:,} ({n/1e6:.2f}M) | |")
