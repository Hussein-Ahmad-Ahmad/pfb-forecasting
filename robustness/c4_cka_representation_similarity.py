"""C.C.4 — CKA representation similarity before/after refinement.

Computes Centered Kernel Alignment (CKA) between:
  1. PatchTST patch-encoder output vs PFBv0 patch-encoder output  (same arch → expect high CKA)
  2. PFBv0 patch-encoder output vs PFBv0 BERT-encoder output      (before vs after refinement)
  3. PFBv0 patch-encoder output vs PFBv2 BERT-encoder output      (v0 vs v2 refinement)

Uses existing checkpoints from multiseed H=192 folders. No retraining.
Dataset: Weather H=192 (21 channels, rich structure — best for mechanistic analysis).

Output: results_analysis/c4_cka_similarity.csv
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_provider.data_factory import data_provider

ROOT = Path(__file__).resolve().parent
CHECKPOINTS_DIR = ROOT / 'checkpoints'
OUT_DIR = ROOT / 'results_analysis'
OUT_CSV = OUT_DIR / 'c4_cka_similarity.csv'

DATASET_CONFIGS = {
    'Weather': {
        'data': 'custom',
        'data_path': 'weather.csv',
        'features': 'M',
        'target': 'OT',
        'freq': 'h',
        'enc_in': 21,
    },
    'ETTm2': {
        'data': 'ETTm2',
        'data_path': 'ETTm2.csv',
        'features': 'M',
        'target': 'OT',
        'freq': 't',
        'enc_in': 7,
    },
}

EXCLUDE_TOKENS = [
    'b1chain', 'b1smoke', 'patchonly', 'refineonly', 'fusiononly', 'bertonly',
    'capacity', 'capmatchdepth', 'kdepth', 'calibrate', 'efficiency', 'wallclock',
    'smoke', 'debug',
]

MODEL_MODULES = {
    'PatchTST': 'PatchTST',
    'PatchFusionBERT_v0': 'PatchFusionBERT_v0',
    'PatchFusionBERT_v2': 'PatchFusionBERT_v2',
}


# ---------------------------------------------------------------------------
# Checkpoint helpers (shared with b2 framework)
# ---------------------------------------------------------------------------

def parse_bool(v: str, default: bool) -> bool:
    return True if v == 'True' else (False if v == 'False' else default)

def extract_int(s: str, pat: str, default: int) -> int:
    m = re.search(pat, s)
    return int(m.group(1)) if m else default

def extract_str(s: str, pat: str, default: str) -> str:
    m = re.search(pat, s)
    return m.group(1) if m else default

def dataset_tag(dataset: str) -> str:
    return dataset.lower()

def is_compatible_folder(name: str, model_name: str, dataset: str, horizon: int) -> bool:
    lower = name.lower()
    if f'_pl{horizon}_' not in name:
        return False
    if dataset_tag(dataset) not in lower:
        return False
    if model_name == 'PatchTST':
        return 'patchtst' in lower and 'patchtst_capacity' not in lower and 'patchtst_depth' not in lower
    if model_name == 'PatchFusionBERT_v0':
        return 'patchfusionbert_v0' in lower or 'pfb_v0' in lower
    if model_name == 'PatchFusionBERT_v2':
        return 'patchfusionbert_v2' in lower or 'pfb_v2' in lower
    return False

def folder_score(name: str, horizon: int) -> int:
    lower = name.lower()
    score = 0
    if f'h{horizon}' in lower: score += 40
    if 'ms2021' in lower or 'seed2021' in lower: score += 20
    if 'baseline_100' in lower or 'multiseed_h192' in lower: score += 10
    if 'exp_0' in lower: score += 5
    for t in EXCLUDE_TOKENS:
        if t in lower: score -= 20
    return score

def find_checkpoint_folder(model_name: str, dataset: str, horizon: int) -> Path:
    candidates = [
        f for f in CHECKPOINTS_DIR.iterdir()
        if f.is_dir() and (f / 'checkpoint.pth').exists()
        and is_compatible_folder(f.name, model_name, dataset, horizon)
    ]
    if not candidates:
        raise FileNotFoundError(f'No checkpoint: {model_name}/{dataset}/H={horizon}')
    candidates.sort(key=lambda f: (folder_score(f.name, horizon), len(f.name)), reverse=True)
    return candidates[0]

def build_args(dataset: str, folder_name: str, batch_size: int = 64) -> SimpleNamespace:
    cfg = DATASET_CONFIGS[dataset]
    args = SimpleNamespace()
    args.task_name = 'long_term_forecast'
    args.root_path = './data/'
    args.data = cfg['data']
    args.data_path = cfg['data_path']
    args.features = extract_str(folder_name, r'_ft([A-Z]+)_', cfg['features'])
    args.target = cfg['target']
    args.freq = cfg['freq']
    args.seq_len = extract_int(folder_name, r'_sl(\d+)_', 336)
    args.label_len = extract_int(folder_name, r'_ll(\d+)_', 48)
    args.pred_len = extract_int(folder_name, r'_pl(\d+)_', 192)
    args.seasonal_patterns = 'Monthly'
    args.batch_size = batch_size
    args.num_workers = 0
    args.embed = extract_str(folder_name, r'_eb([^_]+)_', 'timeF')
    args.enc_in = cfg['enc_in']
    args.dec_in = cfg['enc_in']
    args.c_out = cfg['enc_in']
    args.d_model = extract_int(folder_name, r'_dm(\d+)_', 512)
    args.n_heads = extract_int(folder_name, r'_nh(\d+)_', 8)
    args.e_layers = extract_int(folder_name, r'_el(\d+)_', 2)
    args.d_layers = extract_int(folder_name, r'_dl(\d+)_', 1)
    args.d_ff = extract_int(folder_name, r'_df(\d+)_', 2048)
    args.expand = 2; args.d_conv = 4
    args.factor = extract_int(folder_name, r'_fc(\d+)_', 1)
    args.distil = parse_bool(extract_str(folder_name, r'_dt([^_]+)_', 'True'), True)
    args.dropout = 0.1
    args.activation = 'gelu'
    args.output_attention = False
    args.patch_len = 16; args.stride = 8; args.moving_avg = 25
    args.top_k = 5; args.num_kernels = 6
    args.individual = False; args.channel_independence = 1
    args.decomp_method = 'moving_avg'; args.use_norm = 1
    args.down_sampling_layers = 0; args.down_sampling_window = 1
    args.down_sampling_method = None; args.seg_len = 96; args.num_class = 1
    args.revin = True; args.affine = True; args.subtract_last = False
    args.pfb_k = 0
    return args


# ---------------------------------------------------------------------------
# CKA implementation
# ---------------------------------------------------------------------------

def gram_rbf(X: np.ndarray, sigma_sq: float) -> np.ndarray:
    """RBF kernel Gram matrix for CKA."""
    sq_dists = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
    return np.exp(-sq_dists / (2 * sigma_sq))

def center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def hsic(K: np.ndarray, L: np.ndarray) -> float:
    n = K.shape[0]
    Kc = center_gram(K)
    Lc = center_gram(L)
    return float(np.sum(Kc * Lc) / (n - 1) ** 2)

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between matrices X (n, p) and Y (n, q)."""
    # Center columns
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    K = X @ X.T
    L = Y @ Y.T
    numerator = hsic(K, L)
    denom = np.sqrt(hsic(K, K) * hsic(L, L))
    return float(numerator / (denom + 1e-12))


# ---------------------------------------------------------------------------
# Feature extraction via forward hooks
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Attaches forward hooks and collects intermediate activations."""

    def __init__(self):
        self.features: Dict[str, List[torch.Tensor]] = {}
        self._hooks = []

    def register(self, module: nn.Module, name: str):
        def hook(mod, inp, out):
            # out may be a tuple (Encoder returns (out, attn))
            tensor = out[0] if isinstance(out, (tuple, list)) else out
            self.features.setdefault(name, []).append(tensor.detach().cpu())
        h = module.register_forward_hook(hook)
        self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def collected(self, name: str, max_samples: int = 4096) -> np.ndarray:
        """Concatenate collected tensors, flatten to (n_samples, features)."""
        tensors = self.features.get(name, [])
        if not tensors:
            raise KeyError(f'No features collected for "{name}"')
        arr = torch.cat(tensors, dim=0).float().numpy()  # (total, ...)
        arr = arr.reshape(arr.shape[0], -1)              # (total, flat)
        if arr.shape[0] > max_samples:
            rng = np.random.default_rng(2021)
            idx = rng.choice(arr.shape[0], size=max_samples, replace=False)
            arr = arr[idx]
        return arr


def collect_pfbv0_features(
    model: nn.Module,
    loader,
    device: torch.device,
    pred_len: int,
    label_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run PFBv0 forward pass and collect patch_enc_out and bert_enc_out."""
    extractor = FeatureExtractor()
    extractor.register(model.patch_encoder, 'patch_enc')
    extractor.register(model.bert_encoder, 'bert_enc')

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], 1).float().to(device)
            model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    extractor.remove()
    patch = extractor.collected('patch_enc')
    bert = extractor.collected('bert_enc')
    return patch, bert


def collect_patchtst_features(
    model: nn.Module,
    loader,
    device: torch.device,
    pred_len: int,
    label_len: int,
) -> np.ndarray:
    """Run PatchTST forward pass and collect encoder output."""
    extractor = FeatureExtractor()
    # PatchTST uses self.encoder — try to find it
    enc_module = getattr(model, 'encoder', None)
    if enc_module is None:
        raise AttributeError('PatchTST has no .encoder attribute')
    extractor.register(enc_module, 'patch_enc')

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], 1).float().to(device)
            model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    extractor.remove()
    return extractor.collected('patch_enc')


def load_model(model_name: str, ckpt_path: Path, args: SimpleNamespace, device: torch.device) -> nn.Module:
    module = __import__('models', fromlist=[MODEL_MODULES[model_name]])
    model_module = getattr(module, MODEL_MODULES[model_name])
    model = model_module.Model(args).float()
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for dataset, horizon in [('Weather', 192), ('ETTm2', 192)]:
        print(f'\n=== {dataset} H={horizon} ===')

        # Load PFBv0
        try:
            pfbv0_folder = find_checkpoint_folder('PatchFusionBERT_v0', dataset, horizon)
        except FileNotFoundError as e:
            print(f'  SKIP PFBv0: {e}')
            continue
        pfbv0_args = build_args(dataset, pfbv0_folder.name, batch_size=32)
        _, test_loader = data_provider(pfbv0_args, flag='test')
        pfbv0_model = load_model('PatchFusionBERT_v0', pfbv0_folder / 'checkpoint.pth', pfbv0_args, device)
        print(f'  PFBv0: {pfbv0_folder.name[:60]}...')
        pfbv0_patch, pfbv0_bert = collect_pfbv0_features(
            pfbv0_model, test_loader, device, pfbv0_args.pred_len, pfbv0_args.label_len)
        print(f'  PFBv0 patch features: {pfbv0_patch.shape}, bert features: {pfbv0_bert.shape}')

        # CKA: before vs after refinement (patch_enc vs bert_enc within PFBv0)
        cka_before_after = linear_cka(pfbv0_patch, pfbv0_bert)
        print(f'  CKA (PFBv0: patch_enc vs bert_enc, before vs after refinement): {cka_before_after:.4f}')
        rows.append({
            'Dataset': dataset, 'Horizon': horizon,
            'Comparison': 'PFBv0_before_vs_after_refinement',
            'Stream_A': 'PFBv0_patch_encoder_out',
            'Stream_B': 'PFBv0_bert_encoder_out',
            'Linear_CKA': cka_before_after,
            'n_samples': pfbv0_patch.shape[0],
        })

        # Load PatchTST
        try:
            ptst_folder = find_checkpoint_folder('PatchTST', dataset, horizon)
        except FileNotFoundError as e:
            print(f'  SKIP PatchTST: {e}')
            continue
        ptst_args = build_args(dataset, ptst_folder.name, batch_size=32)
        _, ptst_loader = data_provider(ptst_args, flag='test')
        ptst_model = load_model('PatchTST', ptst_folder / 'checkpoint.pth', ptst_args, device)
        print(f'  PatchTST: {ptst_folder.name[:60]}...')
        try:
            ptst_enc = collect_patchtst_features(
                ptst_model, ptst_loader, device, ptst_args.pred_len, ptst_args.label_len)
            print(f'  PatchTST encoder features: {ptst_enc.shape}')

            # CKA: PatchTST encoder vs PFBv0 patch encoder (same architecture stream — should be high)
            # Align sample count
            n = min(ptst_enc.shape[0], pfbv0_patch.shape[0])
            cka_cross_patch = linear_cka(ptst_enc[:n], pfbv0_patch[:n])
            print(f'  CKA (PatchTST enc vs PFBv0 patch_enc): {cka_cross_patch:.4f}')
            rows.append({
                'Dataset': dataset, 'Horizon': horizon,
                'Comparison': 'PatchTST_vs_PFBv0_patch_stream',
                'Stream_A': 'PatchTST_encoder_out',
                'Stream_B': 'PFBv0_patch_encoder_out',
                'Linear_CKA': cka_cross_patch,
                'n_samples': n,
            })

            # CKA: PatchTST encoder vs PFBv0 bert encoder (after refinement)
            cka_ptst_vs_bert = linear_cka(ptst_enc[:n], pfbv0_bert[:n])
            print(f'  CKA (PatchTST enc vs PFBv0 bert_enc): {cka_ptst_vs_bert:.4f}')
            rows.append({
                'Dataset': dataset, 'Horizon': horizon,
                'Comparison': 'PatchTST_vs_PFBv0_bert_stream',
                'Stream_A': 'PatchTST_encoder_out',
                'Stream_B': 'PFBv0_bert_encoder_out',
                'Linear_CKA': cka_ptst_vs_bert,
                'n_samples': n,
            })
        except (AttributeError, KeyError) as e:
            print(f'  Could not extract PatchTST features: {e}')

        # Load PFBv2
        try:
            pfbv2_folder = find_checkpoint_folder('PatchFusionBERT_v2', dataset, horizon)
            pfbv2_args = build_args(dataset, pfbv2_folder.name, batch_size=32)
            _, pfbv2_loader = data_provider(pfbv2_args, flag='test')
            pfbv2_model = load_model('PatchFusionBERT_v2', pfbv2_folder / 'checkpoint.pth', pfbv2_args, device)
            print(f'  PFBv2: {pfbv2_folder.name[:60]}...')
            pfbv2_patch, pfbv2_bert = collect_pfbv0_features(  # same interface
                pfbv2_model, pfbv2_loader, device, pfbv2_args.pred_len, pfbv2_args.label_len)
            print(f'  PFBv2 patch features: {pfbv2_patch.shape}, bert features: {pfbv2_bert.shape}')

            n = min(pfbv0_patch.shape[0], pfbv2_patch.shape[0])
            # v0 vs v2 patch stream comparison
            cka_v0v2_patch = linear_cka(pfbv0_patch[:n], pfbv2_patch[:n])
            print(f'  CKA (PFBv0 patch_enc vs PFBv2 patch_enc): {cka_v0v2_patch:.4f}')
            rows.append({
                'Dataset': dataset, 'Horizon': horizon,
                'Comparison': 'PFBv0_vs_PFBv2_patch_stream',
                'Stream_A': 'PFBv0_patch_encoder_out',
                'Stream_B': 'PFBv2_patch_encoder_out',
                'Linear_CKA': cka_v0v2_patch,
                'n_samples': n,
            })

            # v0 bert vs v2 bert (different refinement architectures)
            cka_v0v2_bert = linear_cka(pfbv0_bert[:n], pfbv2_bert[:n])
            print(f'  CKA (PFBv0 bert_enc vs PFBv2 bert_enc): {cka_v0v2_bert:.4f}')
            rows.append({
                'Dataset': dataset, 'Horizon': horizon,
                'Comparison': 'PFBv0_vs_PFBv2_bert_stream',
                'Stream_A': 'PFBv0_bert_encoder_out',
                'Stream_B': 'PFBv2_bert_encoder_out',
                'Linear_CKA': cka_v0v2_bert,
                'n_samples': n,
            })

            # PFBv2 before vs after refinement
            cka_v2_before_after = linear_cka(pfbv2_patch, pfbv2_bert)
            print(f'  CKA (PFBv2: patch_enc vs bert_enc): {cka_v2_before_after:.4f}')
            rows.append({
                'Dataset': dataset, 'Horizon': horizon,
                'Comparison': 'PFBv2_before_vs_after_refinement',
                'Stream_A': 'PFBv2_patch_encoder_out',
                'Stream_B': 'PFBv2_bert_encoder_out',
                'Linear_CKA': cka_v2_before_after,
                'n_samples': pfbv2_patch.shape[0],
            })

        except FileNotFoundError as e:
            print(f'  SKIP PFBv2: {e}')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nWrote: {OUT_CSV}')
    print(f'Total rows: {len(df)}')
    print('\nResults:')
    print(df[['Dataset', 'Horizon', 'Comparison', 'Linear_CKA']].to_string(index=False))


if __name__ == '__main__':
    main()
