"""
OPTIONAL PHASE 6: Patch Length Sensitivity Analysis
===================================================
Test how patch length affects model performance.

This script:
1. Trains PatchFusionBERT_v0 and PatchTST with different patch lengths
2. Analyzes sensitivity to this key hyperparameter
3. Validates our default choice of patch_len=16

Patch lengths tested: 8, 16, 32, 64
Datasets: ETTm1, ETTh1 (representative benchmarks)
Horizon: 192 (standard setting)
"""

import os
import sys
import subprocess
from datetime import datetime

def run_experiment(model, dataset, patch_len, horizon=192):
    """Run a single patch sensitivity experiment."""
    
    # Map dataset to data file
    data_files = {
        'ETTm1': 'ETTm1.csv',
        'ETTh1': 'ETTh1.csv',
        'ETTm2': 'ETTm2.csv',
        'ETTh2': 'ETTh2.csv',
    }
    
    # Determine stride (typically patch_len // 2)
    stride = max(patch_len // 2, 1)
    
    # Build command
    cmd = [
        'python', '-u', 'run.py',
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--root_path', './data/',
        '--data_path', data_files.get(dataset, f'{dataset}.csv'),
        '--model_id', f'PatchSens_P{patch_len}_{dataset}_H{horizon}',
        '--model', model,
        '--data', dataset,
        '--features', 'M',
        '--seq_len', '336',
        '--label_len', '96',
        '--pred_len', str(horizon),
        '--e_layers', '3',
        '--d_model', '128',
        '--d_ff', '512',
        '--n_heads', '8',
        '--dropout', '0.1',
        '--batch_size', '32',
        '--learning_rate', '0.0001',
        '--train_epochs', '100',
        '--patience', '10',
        '--patch_len', str(patch_len),
        '--stride', str(stride),
        '--des', f'PatchSensitivity_P{patch_len}'
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {model} on {dataset} H={horizon} with patch_len={patch_len}")
    print(f"{'='*60}")
    
    # Run experiment
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    return result.returncode == 0


def main():
    print("=" * 70)
    print("OPTIONAL PHASE 6: Patch Length Sensitivity Analysis")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    models = ['PatchFusionBERT_v0', 'PatchTST']
    datasets = ['ETTm1', 'ETTh1']
    patch_lengths = [8, 16, 32, 64]
    horizon = 192
    
    total_experiments = len(models) * len(datasets) * len(patch_lengths)
    print(f"Total experiments: {total_experiments}")
    print(f"Models: {', '.join(models)}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Patch lengths: {', '.join(map(str, patch_lengths))}")
    print(f"Horizon: {horizon}")
    print()
    
    # Track results
    completed = 0
    failed = 0
    
    # Run experiments
    for model in models:
        for dataset in datasets:
            for patch_len in patch_lengths:
                success = run_experiment(model, dataset, patch_len, horizon)
                if success:
                    completed += 1
                else:
                    failed += 1
                
                print(f"\nProgress: {completed + failed}/{total_experiments} ({completed} success, {failed} failed)")
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 6 COMPLETE!")
    print("=" * 70)
    print(f"Completed: {completed}/{total_experiments}")
    print(f"Failed: {failed}/{total_experiments}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Next steps:")
    print("1. Check result_long_term_forecast.txt for results")
    print("2. Analyze patch sensitivity patterns")
    print("3. Update results_23-01.md with findings")
    print("=" * 70)


if __name__ == '__main__':
    main()
