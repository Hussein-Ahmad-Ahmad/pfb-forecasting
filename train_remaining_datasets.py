"""
Train PatchFusionBERT v0 and v2 on Remaining Datasets
For fair comparison with baseline models across all datasets
"""

import os
import subprocess
import time
import torch

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("=" * 80)
print("TRAINING PATCHFUSIONBERT ON REMAINING DATASETS")
print("For fair comparison with baseline models")
print("=" * 80)

def run_training(model, dataset, data_path, data_type, seq_len, label_len, pred_len, 
                 enc_in, dec_in, c_out, batch_size=16, description="Full_Training"):
    """Run a single training experiment"""
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    cmd = (
        f"python run.py --task_name long_term_forecast --is_training 1 "
        f"--root_path ./data/ --data_path {data_path} "
        f"--model_id {dataset}_{model}_full --model {model} "
        f"--data {data_type} --features M "
        f"--seq_len {seq_len} --label_len {label_len} --pred_len {pred_len} "
        f"--e_layers 3 --d_layers 1 "
        f"--enc_in {enc_in} --dec_in {dec_in} --c_out {c_out} "
        f"--d_model 128 --n_heads 8 --d_ff 512 "
        f"--dropout 0.1 --des {description} --itr 1 "
        f"--train_epochs 20 --batch_size {batch_size} "
        f"--learning_rate 0.0001 --patience 5 --num_workers 4 "
        f"--patch_len 16 --stride 8"
    )
    
    print(f"\n{'='*80}")
    print(f"Training: {model} on {dataset}")
    print(f"Config: seq_len={seq_len}, pred_len={pred_len}, batch_size={batch_size}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        elapsed = time.time() - start_time
        
        print(f"\n✓ Completed in {elapsed/60:.1f} minutes")
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}")
        return False, 0

# Dataset configurations
datasets = {
    # ETT datasets (remaining)
    'ETTh2': {
        'data_path': 'ETTh2.csv',
        'data_type': 'ETTh2',
        'seq_len': 336,
        'label_len': 48,
        'pred_len': 96,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'batch_size': 16
    },
    'ETTm2': {
        'data_path': 'ETTm2.csv',
        'data_type': 'ETTm2',
        'seq_len': 336,
        'label_len': 48,
        'pred_len': 96,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'batch_size': 16
    },
    
    # Custom datasets
    'Exchange': {
        'data_path': 'exchange_rate.csv',
        'data_type': 'custom',
        'seq_len': 336,
        'label_len': 48,
        'pred_len': 96,
        'enc_in': 8,
        'dec_in': 8,
        'c_out': 8,
        'batch_size': 16
    },
    'Weather': {
        'data_path': 'weather.csv',
        'data_type': 'custom',
        'seq_len': 336,
        'label_len': 48,
        'pred_len': 96,
        'enc_in': 21,
        'dec_in': 21,
        'c_out': 21,
        'batch_size': 8  # Reduced due to more features
    },
    'Illness': {
        'data_path': 'national_illness.csv',
        'data_type': 'custom',
        'seq_len': 104,  # Different from others
        'label_len': 18,
        'pred_len': 24,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'batch_size': 16
    }
}

# Models to train
models = ['PatchFusionBERT_v0', 'PatchFusionBERT_v2']

# Track results
results = {
    'successful': [],
    'failed': [],
    'total_time': 0
}

total_experiments = len(datasets) * len(models)
current = 0

print(f"\nPlanned experiments: {total_experiments}")
print(f"Models: {', '.join(models)}")
print(f"Datasets: {', '.join(datasets.keys())}")
print(f"\nEstimated time: ~{total_experiments * 0.5:.1f} hours (assuming 30 min/experiment)")

input("\nPress ENTER to start training...")

# Run all experiments
for dataset_name, config in datasets.items():
    for model in models:
        current += 1
        
        print(f"\n\n{'#'*80}")
        print(f"# Experiment {current}/{total_experiments}")
        print(f"# Dataset: {dataset_name}, Model: {model}")
        print(f"{'#'*80}")
        
        success, elapsed = run_training(
            model=model,
            dataset=dataset_name,
            data_path=config['data_path'],
            data_type=config['data_type'],
            seq_len=config['seq_len'],
            label_len=config['label_len'],
            pred_len=config['pred_len'],
            enc_in=config['enc_in'],
            dec_in=config['dec_in'],
            c_out=config['c_out'],
            batch_size=config['batch_size'],
            description="Full_Training"
        )
        
        if success:
            results['successful'].append(f"{dataset_name}_{model}")
            results['total_time'] += elapsed
        else:
            results['failed'].append(f"{dataset_name}_{model}")
        
        # Print progress
        print(f"\n{'='*80}")
        print(f"Progress: {current}/{total_experiments} experiments")
        print(f"Successful: {len(results['successful'])}, Failed: {len(results['failed'])}")
        print(f"Time elapsed: {results['total_time']/3600:.2f} hours")
        print(f"{'='*80}")

# Final summary
print("\n\n" + "="*80)
print("TRAINING CAMPAIGN COMPLETE")
print("="*80)
print(f"\nTotal experiments: {total_experiments}")
print(f"Successful: {len(results['successful'])}")
print(f"Failed: {len(results['failed'])}")
print(f"Total time: {results['total_time']/3600:.2f} hours")

if results['successful']:
    print("\n✓ Successful experiments:")
    for exp in results['successful']:
        print(f"  - {exp}")

if results['failed']:
    print("\n✗ Failed experiments:")
    for exp in results['failed']:
        print(f"  - {exp}")

print("\n" + "="*80)
print("Next steps:")
print("1. Run: python generate_visualizations.py (to update all plots)")
print("2. Check: analysis_results/ for updated tables and figures")
print("3. Review: result_long_term_forecast.txt for all metrics")
print("="*80)
