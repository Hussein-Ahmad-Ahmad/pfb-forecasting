#!/usr/bin/env python3
"""
Generate script to run remaining Phase 3B experiments (24 missing)
"""

configs = [
    # DLinear Exchange H=96 (3 seeds)
    ("DLinear", "Exchange", 96, 2021),
    ("DLinear", "Exchange", 96, 2022),
    ("DLinear", "Exchange", 96, 2023),
    # DLinear Exchange H=336 (3 seeds)
    ("DLinear", "Exchange", 336, 2021),
    ("DLinear", "Exchange", 336, 2022),
    ("DLinear", "Exchange", 336, 2023),
    # DLinear Weather H=96 (3 seeds)
    ("DLinear", "Weather", 96, 2021),
    ("DLinear", "Weather", 96, 2022),
    ("DLinear", "Weather", 96, 2023),
    # DLinear Weather H=336 (3 seeds)
    ("DLinear", "Weather", 336, 2021),
    ("DLinear", "Weather", 336, 2022),
    ("DLinear", "Weather", 336, 2023),
    # PatchTST Exchange H=96 (3 seeds)
    ("PatchTST", "Exchange", 96, 2021),
    ("PatchTST", "Exchange", 96, 2022),
    ("PatchTST", "Exchange", 96, 2023),
    # PatchTST Exchange H=336 (3 seeds)
    ("PatchTST", "Exchange", 336, 2021),
    ("PatchTST", "Exchange", 336, 2022),
    ("PatchTST", "Exchange", 336, 2023),
    # PatchTST Weather H=96 (3 seeds)
    ("PatchTST", "Weather", 96, 2021),
    ("PatchTST", "Weather", 96, 2022),
    ("PatchTST", "Weather", 96, 2023),
    # PatchTST Weather H=336 (3 seeds)
    ("PatchTST", "Weather", 336, 2021),
    ("PatchTST", "Weather", 336, 2022),
    ("PatchTST", "Weather", 336, 2023),
]

# Generate PowerShell script
script_lines = ["# Phase 3B Resume: Missing Exchange and Weather experiments (24 runs)"]

for model, dataset, horizon, seed in configs:
    model_id = f"{model}_{dataset}_H{horizon}_seed{seed}"
    
    # Determine paths - all custom datasets use 'custom' as --data parameter
    if dataset in ["Exchange"]:
        root_path = "./data/"
        data_path = "exchange_rate.csv"
        data_arg = "custom"
    elif dataset in ["Weather"]:
        root_path = "./data/"
        data_path = "weather.csv"
        data_arg = "custom"
    else:
        root_path = "./data/ETT/"
        data_path = f"{dataset}.csv"
        data_arg = dataset
    
    cmd = (f"python run.py --task_name long_term_forecast --is_training 1 "
           f"--model_id {model_id} --model {model} --data {data_arg} "
           f"--root_path {root_path} --data_path {data_path} "
           f"--seq_len 336 --label_len 48 --pred_len {horizon} "
           f"--d_model 512 --n_heads 8 --e_layers 2 --d_ff 2048 "
           f"--train_epochs 100 --patience 3 --batch_size 32 --learning_rate 0.0001 --seed {seed}")
    
    script_lines.append(cmd)

# Write script
with open("phase3b_resume_missing.ps1", "w") as f:
    f.write("\n".join(script_lines))

print(f"✓ Generated phase3b_resume_missing.ps1 with {len(configs)} experiments")
