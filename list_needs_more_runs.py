import csv

STATS_CSV = "result_long_term_forecast_multiseed_stats.csv"
NEEDS_MORE_RUNS = "result_long_term_forecast_needs_more_runs.csv"

with open(STATS_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = [row for row in reader if int(row['N_runs']) == 1]

with open(NEEDS_MORE_RUNS, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Dataset", "Horizon"])
    for row in rows:
        writer.writerow([row['Model'], row['Dataset'], row['Horizon']])
