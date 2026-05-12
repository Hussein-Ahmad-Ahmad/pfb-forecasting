#!/usr/bin/env python3
"""
Generate Phase 4B Ablation Study Batch Script
Focus: PatchTST hyperparameter sensitivity on ETTm2 (best performing dataset)
"""
import json

# Load ablation configs
with open('phase4b_ablation_configs.json', 'r') as f:
    configs = json.load(f)

baseline = configs['PatchTST']['baseline']
ablations = configs['PatchTST']['ablations']

# Target dataset and horizon
dataset = "ETTm2"
data_path = "ETTm2.csv"
root_path = "./data/ETT/"
horizon = 96
seeds = [2021, 2022, 2023]

script_lines = []
script_lines.append("# Phase 4B: PatchTST Ablation Study on ETTm2 H=96")
script_lines.append("# Total: 24 experiments (8 variants × 3 seeds)")
script_lines.append("")

count = 0

for ablation in ablations:
    variant_name = ablation['name']
    
    # Start with baseline config
    config = baseline.copy()
    
    # Apply ablation changes
    for key, value in ablation.items():
        if key != 'name':
            config[key] = value
    
    for seed in seeds:
        count += 1
        model_id = f"PatchTST_{dataset}_H{horizon}_{variant_name}_seed{seed}_Ablation"
        
        cmd = (
            f"python run.py --task_name long_term_forecast --is_training 1 "
            f"--model_id {model_id} --model PatchTST "
            f"--data {dataset} --root_path {root_path} --data_path {data_path} "
            f"--features M --seq_len 336 --label_len 48 --pred_len {horizon} "
            f"--d_model {config['d_model']} --d_ff {config['d_ff']} "
            f"--e_layers {config['e_layers']} --n_heads {config['n_heads']} "
            f"--train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed {seed}"
        )
        
        script_lines.append(f"Write-Host 'Experiment {count}/24: {variant_name} seed={seed}'")
        script_lines.append(cmd)
        script_lines.append("")

# Write PowerShell script
with open("phase4b_ablation_patchtst_etm2.ps1", "w") as f:
    f.write("\n".join(script_lines))

print("✓ Generated phase4b_ablation_patchtst_etm2.ps1")
print(f"  Total experiments: {count}")
print(f"  Variants: {len(ablations)}")
print(f"  Seeds per variant: {len(seeds)}")
print(f"  Estimated time: {count * 4} - {count * 6} minutes ({count * 4 / 60:.1f} - {count * 6 / 60:.1f} hours)")
print(f"\nAblation variants:")
for i, ablation in enumerate(ablations, 1):
    changes = {k: v for k, v in ablation.items() if k != 'name'}
    print(f"  {i}. {ablation['name']}: {changes}")
print(f"\nTo run: .\\phase4b_ablation_patchtst_etm2.ps1")
