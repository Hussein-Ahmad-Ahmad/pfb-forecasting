"""
OPTIONAL PHASE 5: Robustness to Missing Data
============================================
Test how models perform when input data has missing values (NaN/zeros).

This script:
1. Loads trained models from checkpoints
2. Tests them with artificially corrupted input data (different missing rates)
3. Compares performance degradation across models

Missing rates tested: 0%, 10%, 20%, 30%
Models: PatchFusionBERT_v0, DLinear, PatchTST (our best vs top baselines)
Dataset: ETTm1 with H=192 (representative benchmark)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_provider.data_factory import data_provider
from utils.metrics import metric


def apply_missing_mask(data, missing_rate, strategy='zero'):
    """
    Apply missing data mask to input tensor.
    
    Args:
        data: Input tensor (batch, seq_len, features)
        missing_rate: Fraction of values to mask (0.0 to 1.0)
        strategy: 'zero' (replace with 0), 'mean' (replace with mean), 'last' (forward fill)
    
    Returns:
        Masked data tensor
    """
    if missing_rate <= 0:
        return data
    
    # Create random mask
    mask = torch.rand_like(data) > missing_rate  # True = keep, False = mask
    
    if strategy == 'zero':
        # Replace masked values with zeros
        masked_data = data * mask.float()
    elif strategy == 'mean':
        # Replace masked values with channel mean
        mean_vals = data.mean(dim=1, keepdim=True)
        masked_data = torch.where(mask, data, mean_vals.expand_as(data))
    elif strategy == 'last':
        # Forward fill (use last valid value)
        masked_data = data.clone()
        for b in range(data.shape[0]):
            for f in range(data.shape[2]):
                last_valid = data[b, 0, f].item()
                for t in range(data.shape[1]):
                    if mask[b, t, f]:
                        last_valid = data[b, t, f].item()
                    else:
                        masked_data[b, t, f] = last_valid
    else:
        masked_data = data * mask.float()
    
    return masked_data


def load_model_for_inference(model_name, args, checkpoint_path):
    """Load a trained model from checkpoint."""
    from models import DLinear, PatchTST, PatchFusionBERT_v0, PatchFusionBERT_v2, iTransformer, TiDE, TimeXer
    
    model_dict = {
        'DLinear': DLinear,
        'PatchTST': PatchTST,
        'PatchFusionBERT_v0': PatchFusionBERT_v0,
        'PatchFusionBERT_v2': PatchFusionBERT_v2,
        'iTransformer': iTransformer,
        'TiDE': TiDE,
        'TimeXer': TimeXer,
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not found in model_dict")
    
    model = model_dict[model_name].Model(args).float()
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return model


def find_checkpoint(model_name, dataset, pred_len, checkpoints_dir='./checkpoints'):
    """Find the checkpoint file for a given model/dataset/horizon."""
    if not os.path.exists(checkpoints_dir):
        return None
    
    # Map model names to checkpoint naming patterns
    model_patterns = {
        'PatchFusionBERT_v0': ['PFB_v0_H', 'PatchFusionBERT_v0'],
        'PatchFusionBERT_v2': ['PFB_v2_H', 'PatchFusionBERT_v2'],
        'DLinear': ['DLinear_H', 'DLinear'],
        'PatchTST': ['PatchTST_H', 'PatchTST'],
        'iTransformer': ['iTransformer_H', 'iTransformer'],
        'TiDE': ['TiDE_H', 'TiDE'],
        'TimeXer': ['TimeXer_H', 'TimeXer'],
    }
    
    patterns = model_patterns.get(model_name, [model_name])
    
    for folder in os.listdir(checkpoints_dir):
        # Check if folder matches expected pattern
        folder_check = folder.lower()
        model_match = any(p.lower() in folder_check for p in patterns)
        dataset_match = dataset.lower() in folder_check
        horizon_match = f"_{pred_len}_" in folder or f"_pl{pred_len}_" in folder
        
        if model_match and dataset_match and horizon_match:
            # Prefer H{pred_len} naming (newer 100-epoch runs)
            if f"_H{pred_len}_" in folder or f"H{pred_len}_" in folder:
                ckpt_path = os.path.join(checkpoints_dir, folder, 'checkpoint.pth')
                if os.path.exists(ckpt_path):
                    return ckpt_path
    
    # Second pass: any matching folder
    for folder in os.listdir(checkpoints_dir):
        folder_check = folder.lower()
        model_match = any(p.lower() in folder_check for p in patterns)
        dataset_match = dataset.lower() in folder_check
        horizon_match = f"pl{pred_len}_" in folder
        
        if model_match and dataset_match and horizon_match:
            ckpt_path = os.path.join(checkpoints_dir, folder, 'checkpoint.pth')
            if os.path.exists(ckpt_path):
                return ckpt_path
    
    return None


def test_with_missing_data(model, test_loader, device, missing_rate, pred_len, label_len, strategy='zero'):
    """Test model with artificially missing data."""
    model.eval()
    preds = []
    trues = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            # Apply missing data mask to input
            batch_x_masked = apply_missing_mask(batch_x, missing_rate, strategy)
            
            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
            
            # Forward pass with masked input
            try:
                outputs = model(batch_x_masked, batch_x_mark, dec_inp, batch_y_mark)
            except Exception as e:
                # Some models don't need all inputs
                try:
                    outputs = model(batch_x_masked)
                except:
                    raise e
            
            # Get predictions
            f_dim = 0
            outputs = outputs[:, -pred_len:, f_dim:]
            batch_y = batch_y[:, -pred_len:, f_dim:].to(device)
            
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # Calculate metrics
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    
    return mse, mae


def run_robustness_experiments():
    """Run robustness experiments with missing data."""
    
    print("=" * 60)
    print("OPTIONAL PHASE 5: Robustness to Missing Data")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    models_to_test = ['PatchFusionBERT_v0', 'DLinear', 'PatchTST']
    missing_rates = [0.0, 0.1, 0.2, 0.3]
    dataset = 'ETTm1'
    pred_len = 192
    seq_len = 336
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Setup args for data loading
    class Args:
        def __init__(self):
            self.root_path = './data/'
            self.data_path = f'{dataset}.csv'
            self.data = dataset
            self.features = 'M'
            self.target = 'OT'
            self.seq_len = seq_len
            self.label_len = 96
            self.pred_len = pred_len
            self.seasonal_patterns = 'Monthly'
            self.freq = 'h'
            self.embed = 'timeF'
            self.batch_size = 32
            self.num_workers = 0
            # Model params
            self.enc_in = 7
            self.dec_in = 7
            self.c_out = 7
            self.d_model = 128
            self.n_heads = 8
            self.e_layers = 3
            self.d_layers = 2
            self.d_ff = 512
            self.factor = 1
            self.dropout = 0.1
            self.embed = 'timeF'
            self.activation = 'gelu'
            self.output_attention = False
            self.patch_len = 16
            self.stride = 8
            self.top_k = 5
            self.num_kernels = 6
            self.individual = False
            self.channel_independence = 1
            self.decomp_method = 'moving_avg'
            self.moving_avg = 25  # For DLinear
            self.use_norm = True
            self.down_sampling_layers = 0
            self.down_sampling_window = 1
            self.down_sampling_method = None
            self.seg_len = 48
            self.task_name = 'long_term_forecast'
            # Additional params for various models
            self.revin = True
            self.affine = True
            self.subtract_last = False
            self.head_dropout = 0.0
    
    args = Args()
    
    # Load test data
    print(f"Loading {dataset} test data...")
    test_data, test_loader = data_provider(args, flag='test')
    print(f"Test samples: {len(test_data)}")
    print()
    
    # Results storage
    results = []
    
    # Test each model
    for model_name in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing: {model_name}")
        print(f"{'='*50}")
        
        # Find checkpoint
        ckpt_path = find_checkpoint(model_name, dataset, pred_len)
        
        if ckpt_path is None:
            print(f"  WARNING: No checkpoint found for {model_name}")
            print(f"  Skipping this model...")
            continue
        
        try:
            # Load model
            model = load_model_for_inference(model_name, args, ckpt_path)
            model = model.to(device)
            model.eval()
            
            # Test with different missing rates
            for rate in missing_rates:
                print(f"\n  Missing rate: {rate*100:.0f}%")
                mse, mae = test_with_missing_data(
                    model, test_loader, device, rate, pred_len, args.label_len, strategy='zero'
                )
                print(f"    MSE: {mse:.6f}, MAE: {mae:.6f}")
                
                results.append({
                    'Model': model_name,
                    'Missing_Rate': rate,
                    'MSE': mse,
                    'MAE': mae,
                    'Dataset': dataset,
                    'Horizon': pred_len
                })
            
            # Calculate degradation
            base_mse = results[-4]['MSE']  # 0% missing
            for i in range(-3, 0):
                degradation = (results[i]['MSE'] - base_mse) / base_mse * 100
                print(f"    {results[i]['Missing_Rate']*100:.0f}% missing -> {degradation:+.1f}% MSE degradation")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Create results dataframe
    if results:
        df = pd.DataFrame(results)
        
        # Save results
        output_file = 'robustness_missing_data_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\n\nResults saved to: {output_file}")
        
        # Print summary table
        print("\n" + "=" * 60)
        print("ROBUSTNESS SUMMARY (MSE)")
        print("=" * 60)
        
        pivot = df.pivot_table(values='MSE', index='Model', columns='Missing_Rate')
        print(pivot.to_string())
        
        # Calculate degradation table
        print("\n" + "=" * 60)
        print("DEGRADATION (% increase from baseline)")
        print("=" * 60)
        
        for model in pivot.index:
            base = pivot.loc[model, 0.0]
            degrad = [(pivot.loc[model, r] - base) / base * 100 for r in [0.1, 0.2, 0.3]]
            print(f"{model:25s}: 10%→{degrad[0]:+.1f}%  20%→{degrad[1]:+.1f}%  30%→{degrad[2]:+.1f}%")
    
    print("\n" + "=" * 60)
    print(f"PHASE 5 COMPLETE! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == '__main__':
    run_robustness_experiments()
