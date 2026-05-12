#!/usr/bin/env python3
"""
Phase 3B: Generate batch scripts for multi-seed validation runs (FIXED).
Target: H=96 and H=336, top 3 models (DLinear, PatchTST, PFB_v0), 
         6 key datasets (ETTh1, ETTh2, ETTm1, ETTm2, Exchange, Weather),
         3 seeds (2021, 2022, 2023)
"""

models = ["DLinear", "PatchTST"]
datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather"]
horizons = [96, 336]
seeds = [2021, 2022, 2023]

# Model-specific configurations (seq_len, label_len, d_model, n_heads, e_layers, d_ff)
config_map = {
    "DLinear": {"seq_len": 336, "label_len": 48, "d_model": 512, "n_heads": 8, "e_layers": 2, "d_ff": 2048},
    "PatchTST": {"seq_len": 336, "label_len": 48, "d_model": 128, "n_heads": 8, "e_layers": 3, "d_ff": 512},
}

data_config = {
    "ETTh1": {"root_path": "./data/ETT/", "data_path": "ETTh1.csv"},
    "ETTh2": {"root_path": "./data/ETT/", "data_path": "ETTh2.csv"},
    "ETTm1": {"root_path": "./data/ETT/", "data_path": "ETTm1.csv"},
    "ETTm2": {"root_path": "./data/ETT/", "data_path": "ETTm2.csv"},
    "Exchange": {"root_path": "./data/", "data_path": "exchange_rate.csv"},
    "Weather": {"root_path": "./data/", "data_path": "weather.csv"},
}

# Generate PowerShell batch script with FIXED --seed argument
with open("phase3b_multiseed_h96_h336.ps1", "w") as f:
    f.write("# Phase 3B: Multi-seed validation runs for H=96 and H=336 (FIXED - PFB_v0 removed)\n")
    f.write("# Targets: DLinear, PatchTST on 6 key datasets, 3 seeds\n\n")
    
    for horizon in horizons:
        f.write(f"\n# ===== Horizon {horizon} =====\n")
        for model in models:
            for dataset in datasets:
                for seed in seeds:
                    cfg = config_map[model]
                    dcfg = data_config[dataset]
                    
                    model_id = f"{model}_{dataset}_H{horizon}_seed{seed}"
                    cmd = (
                        f"python run.py --task_name long_term_forecast --is_training 1 "
                        f"--model_id {model_id} --model {model} "
                        f"--data {dataset} --root_path {dcfg['root_path']} --data_path {dcfg['data_path']} "
                        f"--seq_len {cfg['seq_len']} --label_len {cfg['label_len']} --pred_len {horizon} "
                        f"--d_model {cfg['d_model']} --n_heads {cfg['n_heads']} --e_layers {cfg['e_layers']} --d_ff {cfg['d_ff']} "
                        f"--train_epochs 100 --patience 3 --batch_size 32 --learning_rate 0.0001 "
                        f"--seed {seed}"
                    )
                    f.write(f"{cmd}\n")

print("✓ FIXED phase3b_multiseed_h96_h336.ps1")
print(f"  Changes: Removed PFB_v0 (not in model registry)")
print(f"  Total runs: {len(models)} models × {len(datasets)} datasets × {len(horizons)} horizons × {len(seeds)} seeds")
print(f"             = {len(models) * len(datasets) * len(horizons) * len(seeds)} experiments")
print("\nTo run all experiments sequentially:")
print("  .\\phase3b_multiseed_h96_h336.ps1")
