#!/usr/bin/env python3
"""
Identify missing Phase 3B experiments and generate runner script
"""
import re

# All 72 experiments that should have been run
expected = []
models = ["DLinear", "PatchTST"]
datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather"]
horizons = [96, 336]
seeds = [2021, 2022, 2023]

for model in models:
    for dataset in datasets:
        for horizon in horizons:
            for seed in seeds:
                exp_id = f"long_term_forecast_{model}_{dataset}_H{horizon}_seed{seed}"
                expected.append(exp_id)

# Read results file and extract completed experiments
completed = set()
result_file = "result_long_term_forecast.txt"
pattern = r"long_term_forecast_(DLinear|PatchTST)_(\w+)_H(96|336)_seed(202[1-3])"

try:
    with open(result_file, 'r') as f:
        content = f.read()
        # Find all experiment identifiers with metrics (two-line blocks)
        lines = content.strip().split('\n')
        for i, line in enumerate(lines):
            if 'long_term_forecast_' in line and 'mse:' in lines[i+1] if i+1 < len(lines) else False:
                match = re.search(pattern, line)
                if match:
                    model, dataset, horizon, seed = match.groups()
                    exp_id = f"long_term_forecast_{model}_{dataset}_H{horizon}_seed{seed}"
                    completed.add(exp_id)
except:
    pass

missing = [e for e in expected if e not in completed]
print(f"Completed: {len(completed)}/72")
print(f"Missing: {len(missing)}")
if missing:
    print(f"\nMissing experiments:")
    for m in sorted(missing):
        parts = m.replace("long_term_forecast_", "").split("_")
        print(f"  {parts[0]} {parts[1]} H={parts[2].replace('H', '')} seed={parts[3].replace('seed', '')}")
