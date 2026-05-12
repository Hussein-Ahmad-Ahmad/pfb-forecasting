
import re
import csv
import numpy as np

RESULT_FILE = "result_long_term_forecast.txt"
STATS_CSV = "result_long_term_forecast_multiseed_stats.csv"

# Example identifier:
# long_term_forecast_PatchTST_100epoch_PatchTST_ETTm1_ftM_sl336_ll48_pl96_dm128_nh8_el3_dl1_df512_expand2_dc4_fc1_ebtimeF_dtTrue_baseline_100_0  
# mse:0.2981731593608856, mae:0.3489706814289093, dtw:Not calculated

id_re = re.compile(r"^long_term_forecast_([A-Za-z0-9]+)_.*?([A-Za-z0-9]+)_ftM.*?pl(\d+)_.*$")
mse_mae_re = re.compile(r"mse:([\d\.eE+-]+), mae:([\d\.eE+-]+)")

results = dict()  # (model, dataset, horizon) -> list of (mse, mae)
with open(RESULT_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
    i = 0
    while i < len(lines) - 1:
        id_line = lines[i]
        metric_line = lines[i+1]
        id_match = id_re.match(id_line)
        metric_match = mse_mae_re.search(metric_line)
        if id_match and metric_match:
            model = id_match.group(1)
            dataset = id_match.group(2)
            horizon = id_match.group(3)
            mse = float(metric_match.group(1))
            mae = float(metric_match.group(2))
            key = (model, dataset, horizon)
            results.setdefault(key, []).append((mse, mae))
        i += 2

with open(STATS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Dataset", "Horizon", "MSE_mean", "MSE_std", "MAE_mean", "MAE_std", "N_runs"])
    for key, vals in sorted(results.items()):
        mses = [v[0] for v in vals]
        maes = [v[1] for v in vals]
        mse_mean = np.mean(mses)
        mse_std = np.std(mses)
        mae_mean = np.mean(maes)
        mae_std = np.std(maes)
        writer.writerow([key[0], key[1], key[2], f"{mse_mean:.6f}", f"{mse_std:.6f}", f"{mae_mean:.6f}", f"{mae_std:.6f}", len(vals)])
