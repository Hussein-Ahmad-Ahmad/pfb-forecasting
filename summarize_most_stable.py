import csv
from collections import defaultdict

STATS_CSV = "result_long_term_forecast_multiseed_stats.csv"
SUMMARY_CSV = "result_long_term_forecast_most_stable.csv"

# (dataset, horizon) -> (model, mse_std, mae_std, mse_mean, mae_mean, n_runs)
best_by_dataset_horizon = {}

with open(STATS_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    grouped = defaultdict(list)
    for row in reader:
        dataset = row['Dataset']
        horizon = row['Horizon']
        mse_std = float(row['MSE_std'])
        mae_std = float(row['MAE_std'])
        mse_mean = float(row['MSE_mean'])
        mae_mean = float(row['MAE_mean'])
        n_runs = int(row['N_runs'])
        model = row['Model']
        grouped[(dataset, horizon)].append((model, mse_std, mae_std, mse_mean, mae_mean, n_runs))

    for key, vals in grouped.items():
        # Choose the model with the lowest MSE_std (if tie, lowest MAE_std)
        best = min(vals, key=lambda x: (x[1], x[2]))
        best_by_dataset_horizon[key] = best

with open(SUMMARY_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset", "Horizon", "Model", "MSE_mean", "MSE_std", "MAE_mean", "MAE_std", "N_runs"])
    for (dataset, horizon), (model, mse_std, mae_std, mse_mean, mae_mean, n_runs) in sorted(best_by_dataset_horizon.items()):
        writer.writerow([dataset, horizon, model, f"{mse_mean:.6f}", f"{mse_std:.6f}", f"{mae_mean:.6f}", f"{mae_std:.6f}", n_runs])
