"""Pinned-checkpoint RSA, branch orthogonality, and attention-diversity analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from diagnostic_common import decoder_input, load_model, load_pins, test_loader

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "tables" / "representation_diagnostics.csv"


def tensor_from_output(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def deterministic_sample(array: np.ndarray, limit: int, seed: int = 2021) -> np.ndarray:
    if len(array) <= limit:
        return array
    rng = np.random.default_rng(seed)
    return array[rng.choice(len(array), size=limit, replace=False)]


def rankdata(values: np.ndarray) -> np.ndarray:
    # Distances are floating point and ties are rare; stable ordering preserves
    # deterministic behavior if ties do occur.
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def rsa_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    indices = np.triu_indices(len(x), k=1)
    dx = (1.0 - x @ x.T)[indices]
    dy = (1.0 - y @ y.T)[indices]
    return float(np.corrcoef(rankdata(dx), rankdata(dy))[0, 1])


def orthogonality_metrics(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    denominator = np.sqrt(
        np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    )
    cross_covariance_alignment = float(np.linalg.norm(cross, ord="fro") / max(denominator, 1e-12))
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    yn = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    paired_abs_cosine = float(np.mean(np.abs(np.sum(xn * yn, axis=1))))
    return cross_covariance_alignment, paired_abs_cosine


def normalized_entropy(attention: np.ndarray) -> float:
    attention = attention / np.maximum(attention.sum(axis=-1, keepdims=True), 1e-12)
    entropy = -np.sum(attention * np.log(np.maximum(attention, 1e-12)), axis=-1)
    return float(np.mean(entropy / np.log(attention.shape[-1])))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
    q = q / np.maximum(q.sum(axis=-1, keepdims=True), 1e-12)
    midpoint = 0.5 * (p + q)
    kl_p = np.sum(p * (np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(midpoint, 1e-12))), axis=-1)
    kl_q = np.sum(q * (np.log(np.maximum(q, 1e-12)) - np.log(np.maximum(midpoint, 1e-12))), axis=-1)
    return float(np.mean(0.5 * (kl_p + kl_q)))


def within_branch_head_js(attention: np.ndarray) -> float:
    # Average over batch, retaining one map per head.
    maps = attention.mean(axis=0)
    values = [js_divergence(maps[i], maps[j]) for i in range(len(maps)) for j in range(i + 1, len(maps))]
    return float(np.mean(values)) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pins = load_pins(["PFB-Direct", "PFB-Projected"])
    print(f"Pinned PFB checkpoints: {len(pins)}")
    if args.dry_run:
        for pin in pins:
            print(f"{pin['public_model']} {pin['dataset']} SHA-256={pin['sha256']}")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    for pin in pins:
        model_args, loader = test_loader(pin, args.batch_size)
        model = load_model(pin, model_args, device)
        primary_features: list[torch.Tensor] = []
        secondary_features: list[torch.Tensor] = []
        primary_attention: list[torch.Tensor] = []
        secondary_attention: list[torch.Tensor] = []

        for layer in model.patch_encoder.attn_layers:
            layer.attention.inner_attention.output_attention = True
        for layer in model.secondary_encoder.layers:
            layer.attention.inner_attention.output_attention = True

        def feature_hook(destination):
            def hook(_module, _inputs, output):
                destination.append(tensor_from_output(output).detach().cpu())
            return hook

        def attention_hook(destination):
            def hook(_module, _inputs, output):
                if isinstance(output, (tuple, list)) and output[1] is not None:
                    destination.append(output[1].detach().cpu())
            return hook

        hooks = [
            model.patch_encoder.register_forward_hook(feature_hook(primary_features)),
            model.secondary_encoder.register_forward_hook(feature_hook(secondary_features)),
            model.patch_encoder.attn_layers[-1].register_forward_hook(attention_hook(primary_attention)),
            model.secondary_encoder.layers[-1].register_forward_hook(attention_hook(secondary_attention)),
        ]
        with torch.no_grad():
            for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                if args.max_batches and batch_index >= args.max_batches:
                    break
                y = batch_y.float().to(device)
                model(
                    batch_x.float().to(device), batch_x_mark.float().to(device),
                    decoder_input(y, model_args, device), batch_y_mark.float().to(device),
                )
        for hook in hooks:
            hook.remove()

        primary = torch.cat(primary_features).numpy().reshape(-1, model_args.d_model)
        secondary = torch.cat(secondary_features).numpy().reshape(-1, model_args.d_model)
        count = min(len(primary), len(secondary))
        paired = np.concatenate([primary[:count], secondary[:count]], axis=1)
        paired = deterministic_sample(paired, args.max_tokens)
        primary = paired[:, : model_args.d_model]
        secondary = paired[:, model_args.d_model :]

        pa = torch.cat(primary_attention).numpy()
        sa = torch.cat(secondary_attention).numpy()
        shared_batches = min(len(pa), len(sa))
        pa, sa = pa[:shared_batches], sa[:shared_batches]
        primary_mean = pa.mean(axis=1)
        secondary_mean = sa.mean(axis=1)
        covariance_alignment, abs_cosine = orthogonality_metrics(primary, secondary)
        row = {
            "Dataset": pin["dataset"], "Horizon": model_args.pred_len,
            "Model": pin["public_model"], "Representations": len(primary),
            "RSA_Spearman": rsa_spearman(primary, secondary),
            "Cross_Covariance_Alignment": covariance_alignment,
            "Paired_Absolute_Cosine": abs_cosine,
            "Primary_Attention_Entropy": normalized_entropy(pa),
            "Secondary_Attention_Entropy": normalized_entropy(sa),
            "Primary_Head_JS": within_branch_head_js(pa),
            "Secondary_Head_JS": within_branch_head_js(sa),
            "Branch_Attention_JS": js_divergence(primary_mean, secondary_mean),
            "Checkpoint_ID": pin["folder"], "Checkpoint_SHA256": pin["sha256"],
        }
        rows.append(row)
        print(
            f"{pin['public_model']} {pin['dataset']}: RSA={row['RSA_Spearman']:.4f}, "
            f"attention-JS={row['Branch_Attention_JS']:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
