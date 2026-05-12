"""
Complete Missing Baseline Experiments
Train all baseline models on datasets where results are missing
"""

import subprocess
import time
import json
from pathlib import Path
import pandas as pd

print("="*80)
print("IDENTIFYING MISSING BASELINE EXPERIMENTS")
print("="*80)

# Load current results
comparison_file = Path('./analysis_results/tables/complete_comparison.csv')
df = pd.read_csv(comparison_file)

# Define what should be trained
all_datasets = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather', 'Illness']
baseline_models = ['PatchTST', 'DLinear', 'TiDE', 'TimeXer']

# Check what's missing
missing_experiments = []

for model in baseline_models:
    model_row = df[df['Model'] == model]
    if model_row.empty:
        # Model not in results at all - train on all datasets
        for dataset in all_datasets:
            missing_experiments.append({'model': model, 'dataset': dataset})
    else:
        # Check each dataset
        for dataset in all_datasets:
            mse_col = f'{dataset}_MSE'
            if mse_col in df.columns:
                value = model_row[mse_col].values[0]
                if pd.isna(value):
                    missing_experiments.append({'model': model, 'dataset': dataset})

print(f"\nFound {len(missing_experiments)} missing experiments:")
print("-" * 80)

for exp in missing_experiments:
    print(f"  - {exp['model']:12s} on {exp['dataset']}")

print("\n" + "="*80)
print("BASELINE TRAINING CONFIGURATION")
print("="*80)

# Dataset configurations
dataset_configs = {
    'ETTm1': {
        'data': 'ETTm1',
        'data_path': 'ETTm1.csv',
        'features': 'M',
        'seq_len': 336,
        'label_len': 48,
        'pred_len': 96,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'batch_size': 16
    },
    'ETTm2': {
        'data': 'ETTm2',
        'data_path': 'ETTm2.csv',
        'features': 'M',
        'seq_len': 336,
        'label_len': 48,
        'pred_len': 96,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'batch_size': 16
    },
    'ETTh1': {
        'data': 'ETTh1',
        'data_path': 'ETTh1.csv',
        'features': 'M',
        'seq_len': 336,
        'label_len': 48,
        'pred_len': 96,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'batch_size': 16
    },
    'ETTh2': {
        'data': 'ETTh2',
        'data_path': 'ETTh2.csv',
        'features': 'M',
        'seq_len': 336,
        'label_len': 48,
        'pred_len': 96,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'batch_size': 16
    },
    'Exchange': {
        'data': 'custom',
        'data_path': 'exchange_rate.csv',
        'features': 'M',
        'seq_len': 336,
        'label_len': 48,
        'pred_len': 96,
        'enc_in': 8,
        'dec_in': 8,
        'c_out': 8,
        'batch_size': 16
    },
    'Weather': {
        'data': 'custom',
        'data_path': 'weather.csv',
        'features': 'M',
        'seq_len': 336,
        'label_len': 48,
        'pred_len': 96,
        'enc_in': 21,
        'dec_in': 21,
        'c_out': 21,
        'batch_size': 8  # Reduced for 21 features
    },
    'Illness': {
        'data': 'custom',
        'data_path': 'national_illness.csv',
        'features': 'M',
        'seq_len': 104,
        'label_len': 18,
        'pred_len': 24,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'batch_size': 16
    }
}

# Model-specific parameters
model_params = {
    'PatchTST': {
        'd_model': 128,
        'n_heads': 8,
        'e_layers': 3,
        'd_layers': 1,
        'd_ff': 512,
        'dropout': 0.1
    },
    'DLinear': {
        'd_model': 512,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 2048,
        'dropout': 0.1
    },
    'TiDE': {
        'd_model': 128,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 512,
        'dropout': 0.1
    },
    'TimeXer': {
        'd_model': 128,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 512,
        'dropout': 0.1
    }
}

print(f"\nTotal experiments to run: {len(missing_experiments)}")
print(f"Estimated time: {len(missing_experiments) * 20:.0f} - {len(missing_experiments) * 30:.0f} minutes")
print(f"                (~{len(missing_experiments) * 25 / 60:.1f} hours)")

# User confirmation
print("\n" + "="*80)
response = input("Proceed with baseline training? [y/N]: ")
if response.lower() != 'y':
    print("Aborted.")
    exit()

# Track progress
progress_file = Path('./analysis_results/baseline_training_progress.json')
completed = []
failed = []
start_time = time.time()

# Training loop
for idx, exp in enumerate(missing_experiments, 1):
    model = exp['model']
    dataset = exp['dataset']
    config = dataset_configs[dataset]
    params = model_params[model]
    
    print("\n" + "="*80)
    print(f"EXPERIMENT {idx}/{len(missing_experiments)}")
    print(f"Model: {model}, Dataset: {dataset}")
    print("="*80)
    
    # Build command
    cmd = [
        'python', 'run.py',
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--root_path', './data/',
        '--data_path', config['data_path'],
        '--model_id', f"{dataset}_{model}_baseline",
        '--model', model,
        '--data', config['data'],
        '--features', config['features'],
        '--seq_len', str(config['seq_len']),
        '--label_len', str(config['label_len']),
        '--pred_len', str(config['pred_len']),
        '--enc_in', str(config['enc_in']),
        '--dec_in', str(config['dec_in']),
        '--c_out', str(config['c_out']),
        '--d_model', str(params['d_model']),
        '--n_heads', str(params['n_heads']),
        '--e_layers', str(params['e_layers']),
        '--d_layers', str(params['d_layers']),
        '--d_ff', str(params['d_ff']),
        '--dropout', str(params['dropout']),
        '--batch_size', str(config['batch_size']),
        '--learning_rate', '0.0001',
        '--train_epochs', '20',
        '--patience', '5',
        '--des', 'baseline',
        '--itr', '1',
        '--use_gpu', '1',
        '--gpu', '0'
    ]
    
    print(f"\nStarting: {model} on {dataset}")
    print(f"Command: {' '.join(cmd)}")
    
    exp_start = time.time()
    
    try:
        # Clear GPU cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        exp_time = (time.time() - exp_start) / 60
        
        if result.returncode == 0:
            completed.append({
                'model': model,
                'dataset': dataset,
                'time_minutes': exp_time,
                'status': 'success'
            })
            print(f"✓ Completed in {exp_time:.1f} minutes")
        else:
            failed.append({
                'model': model,
                'dataset': dataset,
                'error': result.stderr[-500:] if result.stderr else 'Unknown error'
            })
            print(f"✗ Failed: {result.stderr[-200:] if result.stderr else 'Unknown error'}")
    
    except subprocess.TimeoutExpired:
        exp_time = (time.time() - exp_start) / 60
        failed.append({
            'model': model,
            'dataset': dataset,
            'error': 'Timeout (>1 hour)'
        })
        print(f"✗ Failed: Timeout after {exp_time:.1f} minutes")
    
    except Exception as e:
        failed.append({
            'model': model,
            'dataset': dataset,
            'error': str(e)
        })
        print(f"✗ Failed: {str(e)}")
    
    # Save progress
    progress_data = {
        'completed': completed,
        'failed': failed,
        'total': len(missing_experiments),
        'remaining': len(missing_experiments) - len(completed) - len(failed)
    }
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    # Progress update
    elapsed_hours = (time.time() - start_time) / 3600
    remaining = len(missing_experiments) - idx
    avg_time_per_exp = elapsed_hours / idx
    estimated_remaining = remaining * avg_time_per_exp
    
    print(f"\n{'='*80}")
    print(f"PROGRESS: {idx}/{len(missing_experiments)} experiments")
    print(f"Successful: {len(completed)}, Failed: {len(failed)}")
    print(f"Time elapsed: {elapsed_hours:.2f} hours")
    print(f"Estimated remaining: {estimated_remaining:.2f} hours")
    print(f"{'='*80}")

# Final summary
total_time = (time.time() - start_time) / 3600

print("\n" + "="*80)
print("BASELINE TRAINING CAMPAIGN COMPLETE!")
print("="*80)
print(f"\nTotal time: {total_time:.2f} hours")
print(f"Successful: {len(completed)}/{len(missing_experiments)}")
print(f"Failed: {len(failed)}/{len(missing_experiments)}")

if failed:
    print("\nFailed experiments:")
    for f in failed:
        print(f"  - {f['model']} on {f['dataset']}: {f['error'][:100]}")

print(f"\nProgress saved to: {progress_file}")
print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Run: python generate_complete_comparison.py")
print("   → Update comparison tables with new baseline results")
print("2. Run: python generate_publication_figures.py")
print("   → Regenerate figures with complete data")
print("3. Run: python generate_word_report.py")
print("   → Update Word document with complete comparison")
print("="*80)
