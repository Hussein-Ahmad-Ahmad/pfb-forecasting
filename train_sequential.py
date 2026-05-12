"""
Sequential Model-by-Model Training
Train one complete model across all datasets, then move to next
"""

import subprocess
import time
import json
from pathlib import Path
import torch

# Define models and their datasets
training_plan = [
    {
        'model_name': 'PatchTST',
        'model': 'PatchTST',
        'datasets': ['ETTm1', 'ETTm2', 'ETTh1', 'Illness'],
        'd_model': 128, 'n_heads': 8, 'e_layers': 3, 'd_ff': 512
    },
    {
        'model_name': 'DLinear',
        'model': 'DLinear',
        'datasets': ['ETTm1', 'ETTh1', 'Exchange', 'Illness'],
        'd_model': 512, 'n_heads': 8, 'e_layers': 2, 'd_ff': 2048
    },
    {
        'model_name': 'TiDE',
        'model': 'TiDE',
        'datasets': ['ETTm1', 'ETTh2', 'Exchange', 'Illness'],
        'd_model': 128, 'n_heads': 8, 'e_layers': 2, 'd_ff': 512
    },
    {
        'model_name': 'TimeXer',
        'model': 'TimeXer',
        'datasets': ['ETTm2', 'ETTh1', 'Weather', 'Illness'],
        'd_model': 128, 'n_heads': 8, 'e_layers': 2, 'd_ff': 512
    },
    {
        'model_name': 'Informer',
        'model': 'Informer',
        'datasets': ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather', 'Illness'],
        'd_model': 128, 'n_heads': 8, 'e_layers': 2, 'd_ff': 512
    },
    {
        'model_name': 'Transformer',
        'model': 'Transformer',
        'datasets': ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather', 'Illness'],
        'd_model': 128, 'n_heads': 8, 'e_layers': 2, 'd_ff': 512
    },
    {
        'model_name': 'Autoformer',
        'model': 'Autoformer',
        'datasets': ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather', 'Illness'],
        'd_model': 128, 'n_heads': 8, 'e_layers': 2, 'd_ff': 512
    }
]

# Dataset configurations
dataset_configs = {
    'ETTm1': {'data': 'ETTm1', 'data_path': 'ETTm1.csv', 'seq_len': 336, 'label_len': 48, 'pred_len': 96, 'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'batch_size': 16},
    'ETTm2': {'data': 'ETTm2', 'data_path': 'ETTm2.csv', 'seq_len': 336, 'label_len': 48, 'pred_len': 96, 'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'batch_size': 16},
    'ETTh1': {'data': 'ETTh1', 'data_path': 'ETTh1.csv', 'seq_len': 336, 'label_len': 48, 'pred_len': 96, 'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'batch_size': 16},
    'ETTh2': {'data': 'ETTh2', 'data_path': 'ETTh2.csv', 'seq_len': 336, 'label_len': 48, 'pred_len': 96, 'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'batch_size': 16},
    'Exchange': {'data': 'custom', 'data_path': 'exchange_rate.csv', 'seq_len': 336, 'label_len': 48, 'pred_len': 96, 'enc_in': 8, 'dec_in': 8, 'c_out': 8, 'batch_size': 16},
    'Weather': {'data': 'custom', 'data_path': 'weather.csv', 'seq_len': 336, 'label_len': 48, 'pred_len': 96, 'enc_in': 21, 'dec_in': 21, 'c_out': 21, 'batch_size': 8},
    'Illness': {'data': 'custom', 'data_path': 'national_illness.csv', 'seq_len': 104, 'label_len': 18, 'pred_len': 24, 'enc_in': 7, 'dec_in': 7, 'c_out': 7, 'batch_size': 16}
}

print("="*80)
print("SEQUENTIAL MODEL-BY-MODEL TRAINING")
print("="*80)

total_experiments = sum(len(plan['datasets']) for plan in training_plan)
print(f"\nTotal experiments: {total_experiments}")
print("\nTraining plan:")
for idx, plan in enumerate(training_plan, 1):
    print(f"  {idx}. {plan['model_name']}: {len(plan['datasets'])} datasets - {', '.join(plan['datasets'])}")

print(f"\nEstimated time: {total_experiments * 20:.0f} - {total_experiments * 30:.0f} minutes")
print(f"                (~{total_experiments * 25 / 60:.1f} hours)")
print("\n" + "="*80)
print("STARTING AUTOMATICALLY IN 3 SECONDS...")
print("="*80)

import time
time.sleep(3)

# Progress tracking
progress_file = Path('./analysis_results/sequential_training_progress.json')
progress_file.parent.mkdir(exist_ok=True)

completed = []
failed = []
start_time = time.time()

# Train model by model
for model_idx, model_plan in enumerate(training_plan, 1):
    model_name = model_plan['model_name']
    model = model_plan['model']
    datasets = model_plan['datasets']
    
    print("\n" + "="*80)
    print(f"MODEL {model_idx}/{len(training_plan)}: {model_name}")
    print(f"Datasets: {', '.join(datasets)}")
    print("="*80)
    
    # Train on each dataset
    for dataset_idx, dataset in enumerate(datasets, 1):
        config = dataset_configs[dataset]
        
        print(f"\n--- Dataset {dataset_idx}/{len(datasets)}: {dataset} ---")
        
        # Clear GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
            '--features', 'M',
            '--seq_len', str(config['seq_len']),
            '--label_len', str(config['label_len']),
            '--pred_len', str(config['pred_len']),
            '--enc_in', str(config['enc_in']),
            '--dec_in', str(config['dec_in']),
            '--c_out', str(config['c_out']),
            '--d_model', str(model_plan['d_model']),
            '--n_heads', str(model_plan['n_heads']),
            '--e_layers', str(model_plan['e_layers']),
            '--d_layers', '1',
            '--d_ff', str(model_plan['d_ff']),
            '--dropout', '0.1',
            '--batch_size', str(config['batch_size']),
            '--learning_rate', '0.0001',
            '--train_epochs', '20',
            '--patience', '5',
            '--des', 'baseline',
            '--itr', '1',
            '--num_workers', '0',
            '--use_gpu', '1',
            '--gpu', '0'
        ]
        
        print(f"Training: {model} on {dataset}...")
        exp_start = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            exp_time = (time.time() - exp_start) / 60
            
            if result.returncode == 0:
                completed.append({'model': model, 'dataset': dataset, 'time_min': exp_time})
                print(f"✓ Success in {exp_time:.1f} min")
            else:
                failed.append({'model': model, 'dataset': dataset, 'error': 'Non-zero exit'})
                print(f"✗ Failed (exit code {result.returncode})")
                
        except subprocess.TimeoutExpired:
            failed.append({'model': model, 'dataset': dataset, 'error': 'Timeout'})
            print(f"✗ Timeout (>60 min)")
        except Exception as e:
            failed.append({'model': model, 'dataset': dataset, 'error': str(e)})
            print(f"✗ Error: {e}")
        
        # Save progress
        with open(progress_file, 'w') as f:
            json.dump({
                'completed': completed,
                'failed': failed,
                'current_model': model_name,
                'completed_count': len(completed),
                'failed_count': len(failed),
                'total': total_experiments
            }, f, indent=2)
        
        print(f"Progress: {len(completed)}/{total_experiments} completed, {len(failed)} failed")
    
    # Model completed
    elapsed = (time.time() - start_time) / 3600
    print(f"\n✓ {model_name} COMPLETE")
    print(f"Time elapsed: {elapsed:.2f} hours")
    print(f"Remaining models: {len(training_plan) - model_idx}")

# Final summary
total_time = (time.time() - start_time) / 3600
print("\n" + "="*80)
print("SEQUENTIAL TRAINING COMPLETE!")
print("="*80)
print(f"Total time: {total_time:.2f} hours")
print(f"Completed: {len(completed)}/{total_experiments}")
print(f"Failed: {len(failed)}")
if failed:
    print("\nFailed experiments:")
    for f in failed:
        print(f"  - {f['model']} on {f['dataset']}: {f['error']}")
