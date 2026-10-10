"""C.C.2 — Gaussian noise + channel dropout robustness.

Applies two new inference-time corruptions to existing checkpoints:
  - gaussian: adds Gaussian noise scaled by per-feature std * noise_level
  - channel_dropout: zeros out a random fraction of input channels

Outputs results_analysis/c2_gaussian_channel_robustness.csv.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.missing_data import mask_block, mask_random


ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = ROOT / 'checkpoints'
OUT_DIR = ROOT / 'results_analysis'
RAW_CSV = OUT_DIR / 'c2_gaussian_channel_robustness.csv'

MODEL_MODULES = {
    'DLinear': 'DLinear',
    'PatchTST': 'PatchTST',
    'PatchFusionBERT_v0': 'PatchFusionBERT_v0',
    'PatchFusionBERT_v2': 'PatchFusionBERT_v2',
}

DATASET_CONFIGS = {
    'ETTm2': {
        'data': 'ETTm2',
        'data_path': 'ETTm2.csv',
        'features': 'M',
        'target': 'OT',
        'freq': 't',
        'enc_in': 7,
    },
    'Weather': {
        'data': 'custom',
        'data_path': 'weather.csv',
        'features': 'M',
        'target': 'OT',
        'freq': 'h',
        'enc_in': 21,
    },
}

DEFAULT_MODELS = ['DLinear', 'PatchTST', 'PatchFusionBERT_v0', 'PatchFusionBERT_v2']
DEFAULT_DATASETS = ['ETTm2', 'Weather']
DEFAULT_HORIZONS = [96, 192]
# Noise levels: fraction of per-feature std to add
DEFAULT_NOISE_LEVELS = [0.1, 0.3, 0.5]
# Channel dropout rates: fraction of channels to zero out
DEFAULT_DROPOUT_RATES = [0.1, 0.2, 0.3]

EXCLUDE_TOKENS = [
    'b1chain',
    'b1smoke',
    'patchonly',
    'refineonly',
    'fusiononly',
    'bertonly',
    'capacity',
    'capmatchdepth',
    'kdepth',
    'calibrate',
    'efficiency',
    'wallclock',
    'smoke',
    'debug',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='C.C.2 Gaussian + channel-dropout robustness')
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    parser.add_argument('--horizons', nargs='+', type=int, default=DEFAULT_HORIZONS)
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--noise_levels', nargs='+', type=float, default=DEFAULT_NOISE_LEVELS)
    parser.add_argument('--dropout_rates', nargs='+', type=float, default=DEFAULT_DROPOUT_RATES)
    parser.add_argument('--corruptions', nargs='+', default=['gaussian', 'channel_dropout'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=RAW_CSV)
    parser.add_argument('--skip_missing', action='store_true')
    return parser.parse_args()


def dataset_tag(dataset: str) -> str:
    return dataset.lower()


def parse_bool(value: str, default: bool) -> bool:
    if value == 'True':
        return True
    if value == 'False':
        return False
    return default


def extract_int(folder_name: str, pattern: str, default: int) -> int:
    match = re.search(pattern, folder_name)
    return int(match.group(1)) if match else default


def extract_str(folder_name: str, pattern: str, default: str) -> str:
    match = re.search(pattern, folder_name)
    return match.group(1) if match else default


def is_compatible_folder(folder_name: str, model_name: str, dataset: str, horizon: int) -> bool:
    lower = folder_name.lower()
    if f'_pl{horizon}_' not in folder_name:
        return False
    if dataset_tag(dataset) not in lower:
        return False
    if model_name == 'DLinear':
        return 'dlinear' in lower
    if model_name == 'PatchTST':
        return 'patchtst' in lower and 'patchtst_capacity' not in lower and 'patchtst_depth' not in lower
    if model_name == 'PatchFusionBERT_v0':
        return 'patchfusionbert_v0' in lower or 'pfb_v0' in lower
    if model_name == 'PatchFusionBERT_v2':
        return 'patchfusionbert_v2' in lower or 'pfb_v2' in lower
    return False


def folder_score(folder_name: str, horizon: int) -> int:
    lower = folder_name.lower()
    score = 0
    if f'h{horizon}' in lower:
        score += 40
    if 'ms2021' in lower or 'seed2021' in lower:
        score += 20
    if 'baseline_100' in lower or 'multiseed_h192' in lower:
        score += 10
    if 'exp_0' in lower:
        score += 5
    for token in EXCLUDE_TOKENS:
        if token in lower:
            score -= 20
    return score


def find_checkpoint_folder(model_name: str, dataset: str, horizon: int) -> Path:
    candidates: List[Path] = []
    for folder in CHECKPOINTS_DIR.iterdir():
        if not folder.is_dir():
            continue
        checkpoint = folder / 'checkpoint.pth'
        if not checkpoint.exists():
            continue
        if is_compatible_folder(folder.name, model_name, dataset, horizon):
            candidates.append(folder)

    if not candidates:
        raise FileNotFoundError(f'No checkpoint folder found for {model_name} / {dataset} / H={horizon}')

    candidates.sort(key=lambda folder: (folder_score(folder.name, horizon), len(folder.name)), reverse=True)
    return candidates[0]


def build_args(dataset: str, folder_name: str, batch_size: int, num_workers: int) -> SimpleNamespace:
    dataset_cfg = DATASET_CONFIGS[dataset]
    features = extract_str(folder_name, r'_ft([A-Z]+)_', dataset_cfg['features'])
    embed = extract_str(folder_name, r'_eb([^_]+)_', 'timeF')

    args = SimpleNamespace()
    args.task_name = 'long_term_forecast'
    args.root_path = './data/'
    args.data = dataset_cfg['data']
    args.data_path = dataset_cfg['data_path']
    args.features = features
    args.target = dataset_cfg['target']
    args.freq = dataset_cfg['freq']
    args.seq_len = extract_int(folder_name, r'_sl(\d+)_', 336)
    args.label_len = extract_int(folder_name, r'_ll(\d+)_', 48)
    args.pred_len = extract_int(folder_name, r'_pl(\d+)_', 96)
    args.seasonal_patterns = 'Monthly'
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.embed = embed
    args.enc_in = dataset_cfg['enc_in']
    args.dec_in = dataset_cfg['enc_in']
    args.c_out = dataset_cfg['enc_in']
    args.d_model = extract_int(folder_name, r'_dm(\d+)_', 512)
    args.n_heads = extract_int(folder_name, r'_nh(\d+)_', 8)
    args.e_layers = extract_int(folder_name, r'_el(\d+)_', 2)
    args.d_layers = extract_int(folder_name, r'_dl(\d+)_', 1)
    args.d_ff = extract_int(folder_name, r'_df(\d+)_', 2048)
    args.expand = extract_int(folder_name, r'_expand(\d+)_', 2)
    args.d_conv = extract_int(folder_name, r'_dc(\d+)_', 4)
    args.factor = extract_int(folder_name, r'_fc(\d+)_', 1)
    args.distil = parse_bool(extract_str(folder_name, r'_dt([^_]+)_', 'True'), True)
    args.dropout = 0.1
    args.activation = 'gelu'
    args.output_attention = False
    args.patch_len = 16
    args.stride = 8
    args.moving_avg = 25
    args.top_k = 5
    args.num_kernels = 6
    args.individual = False
    args.channel_independence = 1
    args.decomp_method = 'moving_avg'
    args.use_norm = 1
    args.down_sampling_layers = 0
    args.down_sampling_window = 1
    args.down_sampling_method = None
    args.seg_len = 96
    args.num_class = 1
    args.revin = True
    args.affine = True
    args.subtract_last = False
    args.pfb_k = 0
    return args


def load_model(model_name: str, checkpoint_path: Path, args: SimpleNamespace, device: torch.device) -> torch.nn.Module:
    module = __import__('models', fromlist=[MODEL_MODULES[model_name]])
    model_module = getattr(module, MODEL_MODULES[model_name])
    model = model_module.Model(args).float()
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def apply_gaussian_noise(batch_x: torch.Tensor, noise_level: float, seed: int, batch_index: int) -> torch.Tensor:
    """Add Gaussian noise scaled by per-feature std * noise_level."""
    rng = torch.Generator()
    rng.manual_seed(seed + batch_index)
    # Compute per-feature std across time dimension (dim=1)
    feat_std = batch_x.std(dim=1, keepdim=True)  # (B, 1, C)
    # Generate noise on CPU with seeded generator, then move to same device as batch_x
    noise = torch.randn(batch_x.shape, generator=rng).to(batch_x.device)
    return batch_x + noise_level * feat_std * noise


def apply_channel_dropout(batch_x: torch.Tensor, rate: float, seed: int, batch_index: int) -> torch.Tensor:
    """Zero out a random fraction of channels (entire time series for dropped channels)."""
    n_channels = batch_x.shape[2]
    n_drop = max(1, round(rate * n_channels))
    rng = np.random.default_rng(seed + batch_index)
    drop_idx = rng.choice(n_channels, size=n_drop, replace=False)
    corrupted = batch_x.clone()
    corrupted[:, :, drop_idx] = 0.0
    return corrupted


def apply_corruption(
    batch_x: torch.Tensor,
    corruption: str,
    rate: float,
    seed: int,
    batch_index: int,
    block_size: int = 24,
) -> torch.Tensor:
    if corruption == 'clean' or rate <= 0:
        return batch_x
    local_seed = seed + batch_index
    if corruption == 'gaussian':
        return apply_gaussian_noise(batch_x, rate, seed, batch_index)
    if corruption == 'channel_dropout':
        return apply_channel_dropout(batch_x, rate, seed, batch_index)
    if corruption == 'random':
        masked, _ = mask_random(batch_x, rate=rate, seed=local_seed)
        return masked
    if corruption == 'block':
        masked, _ = mask_block(batch_x, rate=rate, block_size=block_size, seed=local_seed)
        return masked
    raise ValueError(f'Unsupported corruption: {corruption}')


def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    pred_len: int,
    label_len: int,
    corruption: str,
    rate: float,
    seed: int,
) -> Dict[str, float]:
    preds: List = []
    trues: List = []

    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            batch_x_corrupted = apply_corruption(batch_x, corruption, rate, seed, batch_index)
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(batch_x_corrupted, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -pred_len:, :]
            target = batch_y[:, -pred_len:, :].to(device)

            preds.append(outputs.detach().cpu().numpy())
            trues.append(target.detach().cpu().numpy())

    mae, mse, rmse, mape, mspe = metric(
        np.concatenate(preds, axis=0),
        np.concatenate(trues, axis=0),
    )
    return {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'MSPE': float(mspe),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rows: List[Dict] = []

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        for horizon in args.horizons:
            for model_name in args.models:
                try:
                    checkpoint_folder = find_checkpoint_folder(model_name, dataset, horizon)
                except FileNotFoundError as exc:
                    if args.skip_missing:
                        print(f'  SKIP: {exc}')
                        continue
                    raise

                model_args = build_args(dataset, checkpoint_folder.name, args.batch_size, args.num_workers)
                test_data, test_loader = data_provider(model_args, flag='test')
                model = load_model(model_name, checkpoint_folder / 'checkpoint.pth', model_args, device)

                print(f'Loaded {model_name} on {dataset} H={horizon} from {checkpoint_folder.name}')
                print(f'  Test samples: {len(test_data)}')

                base_info = {
                    'Dataset': dataset,
                    'Horizon': horizon,
                    'Model': model_name,
                    'Checkpoint_Folder': checkpoint_folder.name,
                    'Seq_Len': model_args.seq_len,
                    'D_Model': model_args.d_model,
                    'E_Layers': model_args.e_layers,
                    'Batch_Size': args.batch_size,
                }

                # Clean baseline
                clean_metrics = evaluate_model(
                    model, test_loader, device, model_args.pred_len, model_args.label_len,
                    corruption='clean', rate=0.0, seed=args.seed,
                )
                rows.append({**base_info, 'Corruption': 'clean', 'Rate': 0.0, **clean_metrics})
                print(f"  clean -> MSE {clean_metrics['MSE']:.6f}, MAE {clean_metrics['MAE']:.6f}")

                # Gaussian noise at multiple levels
                if 'gaussian' in args.corruptions:
                    for level in args.noise_levels:
                        metrics = evaluate_model(
                            model, test_loader, device, model_args.pred_len, model_args.label_len,
                            corruption='gaussian', rate=level, seed=args.seed,
                        )
                        rows.append({**base_info, 'Corruption': 'gaussian', 'Rate': level, **metrics})
                        print(f"  gaussian {level:.2f} -> MSE {metrics['MSE']:.6f}, MAE {metrics['MAE']:.6f}")

                # Channel dropout at multiple rates
                if 'channel_dropout' in args.corruptions:
                    for rate in args.dropout_rates:
                        metrics = evaluate_model(
                            model, test_loader, device, model_args.pred_len, model_args.label_len,
                            corruption='channel_dropout', rate=rate, seed=args.seed,
                        )
                        rows.append({**base_info, 'Corruption': 'channel_dropout', 'Rate': rate, **metrics})
                        print(f"  channel_dropout {rate:.2f} -> MSE {metrics['MSE']:.6f}, MAE {metrics['MAE']:.6f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f'\nWrote: {args.output}')
    print(f'Total rows: {len(df)}')


if __name__ == '__main__':
    main()
