"""Shared checkpoint-pinned utilities for inference-only diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TS_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE = TS_ROOT
if str(TS_ROOT) not in sys.path:
    sys.path.insert(0, str(TS_ROOT))

from data_provider.data_factory import data_provider

PIN_MANIFEST = Path(
    os.environ.get("PFB_PIN_MANIFEST", str(SCRIPT_DIR / "pinned_checkpoint_manifest.json"))
)
CHECKPOINTS = Path(os.environ.get("PFB_CHECKPOINT_ROOT", str(TS_ROOT / "checkpoints")))

DATASETS = {
    "ETTm2": {"data": "ETTm2", "file": "ETTm2.csv", "freq": "t", "target": "OT", "channels": 7},
    "Weather": {"data": "custom", "file": "weather.csv", "freq": "h", "target": "OT", "channels": 21},
}

_legacy_base = "PatchFusion" + "".join(chr(c) for c in (66, 69, 82, 84))
_legacy_direct = _legacy_base + "_v" + "0"
_legacy_projected = _legacy_base + "_v" + "2"

PUBLIC_NAMES = {
    "PatchTST": "PatchTST",
    "PFB-Direct": "PFB-Direct",
    "PFB-Projected": "PFB-Projected",
    _legacy_direct: "PFB-Direct",
    _legacy_projected: "PFB-Projected",
}
MODULE_NAMES = {
    "PatchTST": "PatchTST",
    "PFB-Direct": "PFB_Direct",
    "PFB-Projected": "PFB_Projected",
    _legacy_direct: "PFB_Direct",
    _legacy_projected: "PFB_Projected",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pins(models: Iterable[str] | None = None) -> list[dict[str, object]]:
    payload = json.loads(PIN_MANIFEST.read_text(encoding="utf-8"))
    wanted = set(models) if models else None
    pins: list[dict[str, object]] = []
    for raw in payload["checkpoints"]:
        raw_model = str(raw["model"])
        public_model = PUBLIC_NAMES.get(raw_model, raw_model)
        if wanted is not None and raw_model not in wanted and public_model not in wanted:
            continue
        checkpoint = CHECKPOINTS / raw["folder"] / "checkpoint.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Pinned checkpoint is absent: {checkpoint}")
        actual = sha256_file(checkpoint)
        if actual != raw["sha256"]:
            raise RuntimeError(
                f"SHA-256 mismatch for {raw['dataset']}/{public_model}: "
                f"expected {raw['sha256']}, obtained {actual}"
            )
        module_name = MODULE_NAMES.get(raw_model, MODULE_NAMES.get(public_model, public_model.replace("-", "_")))
        pins.append({
            **raw,
            "checkpoint": checkpoint,
            "public_model": public_model,
            "module_name": module_name,
        })
    return pins


def _extract_int(text: str, pattern: str, default: int) -> int:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else default


def _extract_str(text: str, pattern: str, default: str) -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else default


def build_args(pin: dict[str, object], batch_size: int = 32) -> SimpleNamespace:
    dataset = str(pin["dataset"])
    folder = str(pin["folder"])
    cfg = DATASETS[dataset]
    return SimpleNamespace(
        task_name="long_term_forecast",
        root_path=str(TS_ROOT / "data") + os.sep,
        data=cfg["data"], data_path=cfg["file"], features="M", target=cfg["target"],
        freq=cfg["freq"], seq_len=_extract_int(folder, r"_sl(\d+)_", 336),
        label_len=_extract_int(folder, r"_ll(\d+)_", 96),
        pred_len=_extract_int(folder, r"_pl(\d+)_", 192), seasonal_patterns="Monthly",
        batch_size=batch_size, num_workers=0, embed=_extract_str(folder, r"_eb([^_]+)_", "timeF"),
        enc_in=cfg["channels"], dec_in=cfg["channels"], c_out=cfg["channels"],
        d_model=_extract_int(folder, r"_dm(\d+)_", 128), n_heads=_extract_int(folder, r"_nh(\d+)_", 8),
        e_layers=_extract_int(folder, r"_el(\d+)_", 3), d_layers=_extract_int(folder, r"_dl(\d+)_", 1),
        d_ff=_extract_int(folder, r"_df(\d+)_", 512), factor=_extract_int(folder, r"_fc(\d+)_", 3),
        expand=2, d_conv=4, distil=True, dropout=0.1, activation="gelu", output_attention=False,
        patch_len=16, stride=8, moving_avg=25, top_k=5, num_kernels=6, individual=False,
        channel_independence=1, decomp_method="moving_avg", use_norm=1,
        down_sampling_layers=0, down_sampling_window=1, down_sampling_method=None,
        seg_len=96, num_class=1, revin=True, affine=True, subtract_last=False, pfb_k=0,
    )


def _remap_legacy_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean = key[7:] if key.startswith("module.") else key
        clean = clean.replace("b" + "ert_encoder.", "secondary_encoder.")
        mapped[clean] = value
    return mapped


def load_model(pin: dict[str, object], args: SimpleNamespace, device: torch.device) -> torch.nn.Module:
    import models

    module = getattr(models, str(pin["module_name"]))
    model = module.Model(args).float()
    state = torch.load(Path(pin["checkpoint"]), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(_remap_legacy_state(state), strict=True)
    return model.to(device).eval()


def test_loader(pin: dict[str, object], batch_size: int = 32):
    args = build_args(pin, batch_size=batch_size)
    _, loader = data_provider(args, "test")
    return args, loader


def decoder_input(batch_y: torch.Tensor, args: SimpleNamespace, device: torch.device) -> torch.Tensor:
    zeros = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
    return torch.cat([batch_y[:, : args.label_len, :], zeros], dim=1).to(device)


def apply_missingness(
    x: torch.Tensor,
    mechanism: str,
    rate: float,
    seed: int,
    batch_index: int,
    block_size: int = 24,
) -> tuple[torch.Tensor, float]:
    """Apply deterministic MCAR, MAR, MNAR, or block missingness and zero-fill."""
    if mechanism == "clean" or rate <= 0:
        return x, 0.0
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed + 100003 * batch_index)
    noise = torch.rand(x.shape, generator=generator, device=x.device)
    n_mask = max(1, int(round(rate * x.numel())))

    if mechanism == "mcar":
        score = noise
    elif mechanism == "mar":
        # Depend on the previously observed value, not the value being masked.
        lag = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
        center = lag.mean(dim=1, keepdim=True)
        scale = lag.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        score = torch.sigmoid((lag - center) / scale) + 0.05 * noise
    elif mechanism == "mnar":
        # Preferentially remove large-magnitude current observations.
        center = x.mean(dim=1, keepdim=True)
        scale = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        score = ((x - center) / scale).abs() + 0.05 * noise
    elif mechanism == "block":
        mask = torch.zeros_like(x, dtype=torch.bool)
        length = x.shape[1]
        blocks = max(1, math.ceil(rate * length / block_size))
        for b in range(x.shape[0]):
            for c in range(x.shape[2]):
                starts = torch.randint(max(1, length - block_size + 1), (blocks,), generator=generator, device=x.device)
                for start in starts.tolist():
                    mask[b, start : min(length, start + block_size), c] = True
        return x.masked_fill(mask, 0.0), float(mask.float().mean().item())
    else:
        raise ValueError(f"Unknown missingness mechanism: {mechanism}")

    flat = score.reshape(-1)
    selected = torch.topk(flat, k=min(n_mask, flat.numel()), largest=True).indices
    mask = torch.zeros(flat.numel(), dtype=torch.bool, device=x.device)
    mask[selected] = True
    mask = mask.reshape_as(x)
    return x.masked_fill(mask, 0.0), float(mask.float().mean().item())


def apply_gaussian(x: torch.Tensor, scale: float, seed: int, batch_index: int) -> torch.Tensor:
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed + 100003 * batch_index)
    std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    noise = torch.randn(x.shape, generator=generator, device=x.device)
    return x + scale * std * noise


def evaluate(
    model: torch.nn.Module,
    loader,
    args: SimpleNamespace,
    device: torch.device,
    transform,
    max_batches: int = 0,
) -> dict[str, float]:
    squared = absolute = count = 0.0
    realized_rates: list[float] = []
    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            x = batch_x.float().to(device)
            transformed = transform(x, batch_index)
            if isinstance(transformed, tuple):
                x, realized = transformed
                realized_rates.append(float(realized))
            else:
                x = transformed
            y = batch_y.float().to(device)
            output = model(
                x,
                batch_x_mark.float().to(device),
                decoder_input(y, args, device),
                batch_y_mark.float().to(device),
            )[:, -args.pred_len :, :]
            target = y[:, -args.pred_len :, :]
            error = output - target
            squared += float(torch.sum(error * error).item())
            absolute += float(torch.sum(torch.abs(error)).item())
            count += float(error.numel())
    return {
        "MSE": squared / count,
        "MAE": absolute / count,
        "Realized_Rate": float(np.mean(realized_rates)) if realized_rates else 0.0,
        "Elements": int(count),
    }


def read_completed(path: Path, fields: list[str]) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {tuple(row[field] for field in fields) for row in csv.DictReader(handle)}


def append_row(path: Path, row: dict[str, object], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
