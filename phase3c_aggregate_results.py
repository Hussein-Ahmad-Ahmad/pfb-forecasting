#!/usr/bin/env python3
"""
Phase 3C: Aggregate Phase 3B results and generate journal-ready tables.
Run this script after Phase 3B completes to merge new results and create final stats.
"""

import re
import csv
from collections import defaultdict
import numpy as np

RESULT_FILE = "result_long_term_forecast.txt"
CLEAN_CSV = "result_long_term_forecast_clean.csv"
PHASE3C_CSV = "result_long_term_forecast_phase3c_aggregated.csv"
PHASE3C_STATS = "result_long_term_forecast_phase3c_stats.csv"
PHASE3C_SUMMARY = "result_long_term_forecast_phase3c_best_models.csv"

# Parse results from log file
id_pattern = re.compile(
    r"^long_term_forecast_([A-Za-z0-9]+)_\d+epoch_\1_([A-Za-z0-9]+)_.*?_pl(\d+)_.*?_(\d+)\s*$"
)
mse_mae_pattern = re.compile(r"mse:([\d\.eE+-]+),\s*mae:([\d\.eE+-]+)")

results = {}
with open(RESULT_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

i = 0
while i < len(lines) - 1:
    id_line = lines[i]
    metric_line = lines[i + 1]
    
    id_match = id_pattern.match(id_line)
    mse_mae_match = mse_mae_pattern.search(metric_line)
    
    if id_match and mse_mae_match:
        model = id_match.group(1)
        dataset = id_match.group(2)
        horizon = id_match.group(3)
        seed = id_match.group(4)
        mse = float(mse_mae_match.group(1))
        mae = float(mse_mae_match.group(2))
        
        key = (model, dataset, horizon, seed)
        results[key] = (mse, mae)
    
    i += 2

# Write aggregated results
with open(PHASE3C_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Dataset", "Horizon", "Seed", "MSE", "MAE"])
    for (model, dataset, horizon, seed), (mse, mae) in sorted(results.items()):
        writer.writerow([model, dataset, horizon, seed, f"{mse:.10f}", f"{mae:.10f}"])

# Group by (model, dataset, horizon) and compute stats
grouped = defaultdict(list)
for (model, dataset, horizon, seed), (mse, mae) in results.items():
    key = (model, dataset, horizon)
    grouped[key].append((mse, mae))

# Write stats
with open(PHASE3C_STATS, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Dataset", "Horizon", "MSE_mean", "MSE_std", "MAE_mean", "MAE_std", "N_runs"])
    for (model, dataset, horizon), vals in sorted(grouped.items()):
        mses = [v[0] for v in vals]
        maes = [v[1] for v in vals]
        mse_mean = np.mean(mses)
        mse_std = np.std(mses)
        mae_mean = np.mean(maes)
        mae_std = np.std(maes)
        writer.writerow([model, dataset, horizon, f"{mse_mean:.6f}", f"{mse_std:.6f}", f"{mae_mean:.6f}", f"{mae_std:.6f}", len(vals)])

# Find best model per (dataset, horizon)
best_by_dh = {}
for (model, dataset, horizon), vals in grouped.items():
    mses = [v[0] for v in vals]
    mse_mean = np.mean(mses)
    mse_std = np.std(mses)
    maes = [v[1] for v in vals]
    mae_mean = np.mean(maes)
    mae_std = np.std(maes)
    
    key = (dataset, horizon)
    if key not in best_by_dh or mse_std < best_by_dh[key][2]:  # Best by lowest std
        best_by_dh[key] = (model, mse_mean, mse_std, mae_mean, mae_std, len(vals))

# Write summary
with open(PHASE3C_SUMMARY, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset", "Horizon", "Best_Model", "MSE_mean", "MSE_std", "MAE_mean", "MAE_std", "N_runs"])
    for (dataset, horizon), (model, mse_mean, mse_std, mae_mean, mae_std, n_runs) in sorted(best_by_dh.items()):
        writer.writerow([dataset, horizon, model, f"{mse_mean:.6f}", f"{mse_std:.6f}", f"{mae_mean:.6f}", f"{mae_std:.6f}", n_runs])

print(f"✓ Phase 3C aggregation complete")
print(f"  - {PHASE3C_CSV}: All results ({len(results)} unique)")
print(f"  - {PHASE3C_STATS}: Aggregated stats ({len(grouped)} model/dataset/horizon combinations)")
print(f"  - {PHASE3C_SUMMARY}: Best models per dataset/horizon ({len(best_by_dh)} combinations)")
