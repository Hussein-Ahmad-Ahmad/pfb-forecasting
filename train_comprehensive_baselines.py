"""
Comprehensive Baseline Training Campaign
Train all specified model-dataset combinations for complete comparison
"""

import subprocess
import time
import json
from pathlib import Path
import torch

print("="*80)
print("COMPREHENSIVE BASELINE TRAINING CAMPAIGN")
print("="*80)

# Define all training experiments
experiments = [
    # PatchTST
    {'model': 'PatchTST', 'dataset': 'ETTm1'},
    {'model': 'PatchTST', 'dataset': 'ETTm2'},
    {'model': 'PatchTST', 'dataset': 'Weather'},
    
    # PFB_v2
    {'model': 'PatchFusionBERT_v2', 'dataset': 'ETTm2'},
    {'model': 'PatchFusionBERT_v2', 'dataset': 'Weather'},
    
    # DLinear
    {'model': 'DLinear', 'dataset': 'Exchange'},
    {'model': 'DLinear', 'dataset': 'Illness'},
    {'model': 'DLinear', 'dataset': 'ETTh2'},
    {'model': 'DLinear', 'dataset': 'ETTh1'},
    
    # TiDE
    {'model': 'TiDE', 'dataset': 'ETTm1'},
    {'model': 'TiDE', 'dataset': 'Exchange'},
    {'model': 'TiDE', 'dataset': 'Illness'},
    {'model': 'TiDE', 'dataset': 'ETTh2'},
    
    # TimeXer
    {'model': 'TimeXer', 'dataset': 'ETTm2'},
    {'model': 'TimeXer', 'dataset': 'Weather'},
    {'model': 'TimeXer', 'dataset': 'ETTh1'},
    
    # BERTOnly
    {'model': 'PatchFusionBERT_BERTOnly', 'dataset': 'Weather'},
    {'model': 'PatchFusionBERT_BERTOnly', 'dataset': 'ETTh2'},
    {'model': 'PatchFusionBERT_BERTOnly', 'dataset': 'ETTh1'},
    
    # Informer - ALL datasets
    {'model': 'Informer', 'dataset': 'ETTm1'},
    {'model': 'Informer', 'dataset': 'ETTm2'},
    {'model': 'Informer', 'dataset': 'ETTh1'},
    {'model': 'Informer', 'dataset': 'ETTh2'},
    {'model': 'Informer', 'dataset': 'Exchange'},
    {'model': 'Informer', 'dataset': 'Weather'},
    {'model': 'Informer', 'dataset': 'Illness'},
    
    # Transformer - ALL datasets
    {'model': 'Transformer', 'dataset': 'ETTm1'},
    {'model': 'Transformer', 'dataset': 'ETTm2'},
    {'model': 'Transformer', 'dataset': 'ETTh1'},
    {'model': 'Transformer', 'dataset': 'ETTh2'},
    {'model': 'Transformer', 'dataset': 'Exchange'},
    {'model': 'Transformer', 'dataset': 'Weather'},
    {'model': 'Transformer', 'dataset': 'Illness'},
    
    # Autoformer - ALL datasets
    {'model': 'Autoformer', 'dataset': 'ETTm1'},
    {'model': 'Autoformer', 'dataset': 'ETTm2'},
    {'model': 'Autoformer', 'dataset': 'ETTh1'},
    {'model': 'Autoformer', 'dataset': 'ETTh2'},
    {'model': 'Autoformer', 'dataset': 'Exchange'},
    {'model': 'Autoformer', 'dataset': 'Weather'},
    {'model': 'Autoformer', 'dataset': 'Illness'},
]

# Add Illness for ALL remaining models
illness_models = ['PatchTST', 'PFB_v2', 'DLinear', 'TiDE', 'TimeXer', 'PatchFusionBERT_BERTOnly',
                  'PatchFusionBERT_v0']

for model in illness_models:
    # Check if not already in list
    if not any(e['model'] == model and e['dataset'] == 'Illness' for e in experiments):
        experiments.append({'model': model, 'dataset': 'Illness'})

print(f"\nTotal experiments planned: {len(experiments)}")
print("\nBreakdown by model:")
from collections import Counter
model_counts = Counter([e['model'] for e in experiments])
for model, count in sorted(model_counts.items()):
    print(f"  {model:30s}: {count} datasets")

print(f"\nEstimated time: {len(experiments) * 20:.0f} - {len(experiments) * 30:.0f} minutes")
print(f"                (~{len(experiments) * 25 / 60:.1f} hours)")

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
        'batch_size': 8
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

# Model parameters
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
    },
    'Informer': {
        'd_model': 128,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 512,
        'dropout': 0.1
    },
    'Transformer': {
        'd_model': 128,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 512,
        'dropout': 0.1
    },
    'Autoformer': {
        'd_model': 128,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 512,
        'dropout': 0.1
    },
    'PatchFusionBERT_v2': {
        'd_model': 128,
        'n_heads': 8,
        'e_layers': 3,
        'd_layers': 1,
        'd_ff': 512,
        'dropout': 0.1
    },
    'PatchFusionBERT_BERTOnly': {
        'd_model': 128,
        'n_heads': 8,
        'e_layers': 3,
        'd_layers': 1,
        'd_ff': 512,
        'dropout': 0.1
    },
    'PatchFusionBERT_v0': {
        'd_model': 128,
        'n_heads': 8,
        'e_layers': 3,
        'd_layers': 1,
        'd_ff': 512,
        'dropout': 0.1
    }
}

print("\n" + "="*80)
response = input("Proceed with comprehensive training campaign? [y/N]: ")
if response.lower() != 'y':
    print("Aborted.")
    exit()

# Track progress
progress_file = Path('./analysis_results/comprehensive_baseline_progress.json')
completed = []
failed = []
start_time = time.time()

# Training loop
for idx, exp in enumerate(experiments, 1):
    model = exp['model']
    dataset = exp['dataset']
    config = dataset_configs[dataset]
    params = model_params[model]
    
    print("\n" + "="*80)
    print(f"EXPERIMENT {idx}/{len(experiments)}")
    print(f"Model: {model}, Dataset: {dataset}")
    print("="*80)
    
    # Determine description based on model type
    if 'PatchFusionBERT' in model:
        if 'BERTOnly' in model:
            des = 'Ablation_BERTOnly'
        elif 'v2' in model:
            des = 'Full_Training'
        else:
            des = 'Full_Training'
    else:
        des = 'baseline'
    
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
        '--des', des,
        '--itr', '1',
        '--use_gpu', '1',
        '--gpu', '0'
    ]
    
    print(f"\nStarting: {model} on {dataset}")
    
    exp_start = time.time()
    
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
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
        'total': len(experiments),
        'remaining': len(experiments) - len(completed) - len(failed),
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    }
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    # Progress update
    elapsed_hours = (time.time() - start_time) / 3600
    remaining = len(experiments) - idx
    if idx > 0:
        avg_time_per_exp = elapsed_hours / idx
        estimated_remaining = remaining * avg_time_per_exp
    else:
        estimated_remaining = 0
    
    print(f"\n{'='*80}")
    print(f"PROGRESS: {idx}/{len(experiments)} experiments")
    print(f"Successful: {len(completed)}, Failed: {len(failed)}")
    print(f"Time elapsed: {elapsed_hours:.2f} hours")
    print(f"Estimated remaining: {estimated_remaining:.2f} hours")
    print(f"{'='*80}")

# Final summary
total_time = (time.time() - start_time) / 3600

print("\n" + "="*80)
print("COMPREHENSIVE BASELINE CAMPAIGN COMPLETE!")
print("="*80)
print(f"\nCampaign duration: {total_time:.2f} hours")
print(f"Total experiments: {len(experiments)}")
print(f"Successful: {len(completed)}")
print(f"Failed: {len(failed)}")

if failed:
    print("\nFailed experiments:")
    for f in failed:
        print(f"  - {f['model']} on {f['dataset']}: {f['error'][:100]}")

print(f"\nResults saved to: {progress_file}")
print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Run: python generate_complete_comparison.py")
print("2. Run: python generate_publication_figures.py")
print("3. Run: python generate_word_report.py")
print("="*80)
