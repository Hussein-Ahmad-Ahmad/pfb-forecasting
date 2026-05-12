#!/usr/bin/env python3
"""
Generate corrected Phase 3B resume script for missing Exchange and Weather experiments
Uses --data custom with --data_path parameter
"""

models = ["DLinear", "PatchTST"]
datasets = {
    "Exchange": "exchange_rate.csv",
    "Weather": "weather.csv"
}
horizons = [96, 336]
seeds = [2021, 2022, 2023]

config_map = {
    "DLinear": "--features M --seq_len 336 --label_len 48 --d_model 512 --d_ff 2048",
    "PatchTST": "--features M --seq_len 336 --label_len 48 --d_model 128 --d_ff 512 --e_layers 3"
}

script_lines = []
script_lines.append("# Phase 3B Resume Script - Missing Exchange and Weather Experiments")
script_lines.append("# Total: 24 experiments (2 models × 2 datasets × 2 horizons × 3 seeds)")
script_lines.append("")

count = 0
for model in models:
    for dataset_name, data_path in datasets.items():
        for horizon in horizons:
            for seed in seeds:
                count += 1
                model_id = f"{model}_{dataset_name}_H{horizon}_seed{seed}"
                config = config_map[model]
                
                cmd = (
                    f"python run.py --task_name long_term_forecast --is_training 1 "
                    f"--model_id {model_id} --model {model} "
                    f"--data custom --data_path {data_path} "
                    f"{config} --pred_len {horizon} --train_epochs 100 --seed {seed}"
                )
                
                script_lines.append(f"Write-Host 'Running experiment {count}/24: {model_id}'")
                script_lines.append(cmd)
                script_lines.append("")

# Write PowerShell script
with open("phase3b_resume_corrected.ps1", "w") as f:
    f.write("\n".join(script_lines))

print(f"✓ Created phase3b_resume_corrected.ps1")
print(f"  Total: {count} experiments")
print(f"  Models: {', '.join(models)}")
print(f"  Datasets: {', '.join(datasets.keys())}")
print(f"  Horizons: {horizons}")
print(f"  Seeds: {seeds}")
print(f"\nTo run: .\\phase3b_resume_corrected.ps1")
