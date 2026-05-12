"""
COMPREHENSIVE TRAINING CAMPAIGN FOR PUBLICATION
Includes: Full training, ablation studies, and sensitivity analysis on all datasets

Campaign Structure:
1. Full 20-epoch training (v0, v2) - 5 datasets
2. Ablation study (BERTOnly) - 5 datasets  
3. Patch sensitivity (v0 only, 2 key datasets, 4 configs each)
Total: 23 experiments
"""

import os
import subprocess
import time
import torch
import json
from datetime import datetime

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("=" * 80)
print("COMPREHENSIVE TRAINING CAMPAIGN FOR PUBLICATION")
print("Full training + Ablation + Sensitivity on ALL datasets")
print("=" * 80)

def run_training(model, dataset, data_path, data_type, seq_len, label_len, pred_len, 
                 enc_in, dec_in, c_out, batch_size=16, patch_len=16, stride=8, 
                 description="Full_Training"):
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
        f"--patch_len {patch_len} --stride {stride}"
    )
    
    print(f"\n{'='*80}")
    print(f"Training: {model} on {dataset}")
    print(f"Config: seq_len={seq_len}, pred={pred_len}, batch={batch_size}, patch={patch_len}, stride={stride}")
    print(f"Description: {description}")
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
        'seq_len': 104,
        'label_len': 18,
        'pred_len': 24,
        'enc_in': 7,
        'dec_in': 7,
        'c_out': 7,
        'batch_size': 16
    }
}

# Patch sensitivity configurations (for v0 only, on 2 key datasets)
patch_configs = [
    {'patch_len': 8, 'stride': 4},
    {'patch_len': 16, 'stride': 8},
    {'patch_len': 24, 'stride': 12},
    {'patch_len': 32, 'stride': 16}
]

sensitivity_datasets = ['ETTm2', 'Exchange']  # Representative: electricity + finance

# ============================================================================
# EXPERIMENTAL CAMPAIGN PLAN
# ============================================================================

experiments = []

# 1. MAIN TRAINING: v0 and v2 on all 5 datasets (10 experiments)
for dataset_name in datasets.keys():
    for model in ['PatchFusionBERT_v0', 'PatchFusionBERT_v2']:
        experiments.append({
            'type': 'main',
            'model': model,
            'dataset': dataset_name,
            'description': 'Full_Training'
        })

# 2. ABLATION STUDY: BERTOnly on all 5 datasets (5 experiments)
for dataset_name in datasets.keys():
    experiments.append({
        'type': 'ablation',
        'model': 'PatchFusionBERT_BERTOnly',
        'dataset': dataset_name,
        'description': f'Ablation_BERTOnly'
    })

# 3. PATCH SENSITIVITY: v0 on 2 datasets with 4 configs (8 experiments)
for dataset_name in sensitivity_datasets:
    for config in patch_configs:
        experiments.append({
            'type': 'sensitivity',
            'model': 'PatchFusionBERT_v0',
            'dataset': dataset_name,
            'patch_len': config['patch_len'],
            'stride': config['stride'],
            'description': f"PatchSens_p{config['patch_len']}_s{config['stride']}"
        })

total_experiments = len(experiments)

print(f"\n{'='*80}")
print("EXPERIMENTAL CAMPAIGN SUMMARY")
print(f"{'='*80}")
print(f"Main Training:        10 experiments (v0, v2 × 5 datasets)")
print(f"Ablation Study:        5 experiments (BERTOnly × 5 datasets)")
print(f"Sensitivity Analysis:  8 experiments (v0 × 2 datasets × 4 configs)")
print(f"{'='*80}")
print(f"TOTAL:                {total_experiments} experiments")
print(f"\nEstimated time:")
print(f"  Main (10 × 25 min):      ~4.2 hours")
print(f"  Ablation (5 × 25 min):   ~2.1 hours")
print(f"  Sensitivity (8 × 25 min): ~3.3 hours")
print(f"  {'='*60}")
print(f"  TOTAL ESTIMATED:          ~9-10 hours")
print(f"{'='*80}\n")

print("Datasets:")
for ds in datasets.keys():
    print(f"  - {ds}")

print("\nModels:")
print("  - PatchFusionBERT_v0")
print("  - PatchFusionBERT_v2")
print("  - PatchFusionBERT_BERTOnly (ablation)")

print("\nSensitivity Analysis:")
print(f"  Datasets: {', '.join(sensitivity_datasets)}")
print(f"  Configurations: {len(patch_configs)} (patch/stride combinations)")

response = input("\n⚠️  This is a 9-10 hour training campaign. Proceed? [y/N]: ")

if response.lower() != 'y':
    print("\nTraining cancelled. Run again when ready.")
    exit(0)

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

results = {
    'campaign_start': datetime.now().isoformat(),
    'experiments': [],
    'successful': [],
    'failed': [],
    'total_time': 0
}

print(f"\n{'#'*80}")
print(f"# STARTING TRAINING CAMPAIGN")
print(f"# Start time: {results['campaign_start']}")
print(f"{'#'*80}\n")

for idx, exp in enumerate(experiments, 1):
    print(f"\n{'#'*80}")
    print(f"# Experiment {idx}/{total_experiments}")
    print(f"# Type: {exp['type'].upper()}")
    print(f"# Model: {exp['model']}, Dataset: {exp['dataset']}")
    if exp['type'] == 'sensitivity':
        print(f"# Patch Config: {exp['patch_len']}/{exp['stride']}")
    print(f"{'#'*80}")
    
    config = datasets[exp['dataset']]
    
    # Set patch config for sensitivity experiments
    patch_len = exp.get('patch_len', 16)
    stride = exp.get('stride', 8)
    
    success, elapsed = run_training(
        model=exp['model'],
        dataset=exp['dataset'],
        data_path=config['data_path'],
        data_type=config['data_type'],
        seq_len=config['seq_len'],
        label_len=config['label_len'],
        pred_len=config['pred_len'],
        enc_in=config['enc_in'],
        dec_in=config['dec_in'],
        c_out=config['c_out'],
        batch_size=config['batch_size'],
        patch_len=patch_len,
        stride=stride,
        description=exp['description']
    )
    
    exp_result = {
        'experiment_id': idx,
        'type': exp['type'],
        'model': exp['model'],
        'dataset': exp['dataset'],
        'success': success,
        'time_minutes': elapsed / 60 if success else 0
    }
    
    if exp['type'] == 'sensitivity':
        exp_result['patch_len'] = patch_len
        exp_result['stride'] = stride
    
    results['experiments'].append(exp_result)
    
    if success:
        results['successful'].append(f"{exp['dataset']}_{exp['model']}")
        results['total_time'] += elapsed
    else:
        results['failed'].append(f"{exp['dataset']}_{exp['model']}")
    
    # Print progress
    print(f"\n{'='*80}")
    print(f"PROGRESS: {idx}/{total_experiments} experiments")
    print(f"Successful: {len(results['successful'])}, Failed: {len(results['failed'])}")
    print(f"Time elapsed: {results['total_time']/3600:.2f} hours")
    print(f"Estimated remaining: {((total_experiments - idx) * 25)/60:.2f} hours")
    print(f"{'='*80}")
    
    # Save progress checkpoint
    with open('./analysis_results/training_progress.json', 'w') as f:
        json.dump(results, f, indent=2)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

results['campaign_end'] = datetime.now().isoformat()
results['total_hours'] = results['total_time'] / 3600

print("\n\n" + "="*80)
print("COMPREHENSIVE TRAINING CAMPAIGN COMPLETE!")
print("="*80)
print(f"\nCampaign duration: {results['total_hours']:.2f} hours")
print(f"Total experiments: {total_experiments}")
print(f"Successful: {len(results['successful'])}")
print(f"Failed: {len(results['failed'])}")

# Breakdown by type
main_success = sum(1 for e in results['experiments'] if e['type'] == 'main' and e['success'])
ablation_success = sum(1 for e in results['experiments'] if e['type'] == 'ablation' and e['success'])
sensitivity_success = sum(1 for e in results['experiments'] if e['type'] == 'sensitivity' and e['success'])

print(f"\nBreakdown:")
print(f"  Main training:     {main_success}/10")
print(f"  Ablation study:    {ablation_success}/5")
print(f"  Sensitivity:       {sensitivity_success}/8")

if results['failed']:
    print("\n✗ Failed experiments:")
    for exp in results['failed']:
        print(f"  - {exp}")

# Save final results
with open('./analysis_results/training_campaign_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: ./analysis_results/training_campaign_results.json")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Run: python generate_complete_comparison.py")
print("   → Creates comprehensive tables across all 7 datasets")
print("")
print("2. Run: python generate_visualizations.py")
print("   → Updates all ablation and sensitivity plots")
print("")
print("3. Check: analysis_results/ for:")
print("   - Complete comparison tables (7 datasets)")
print("   - Ablation plots (all datasets)")
print("   - Sensitivity plots (ETTm2, Exchange)")
print("   - Ranking analysis")
print("   - Final comprehensive report")
print("="*80)
