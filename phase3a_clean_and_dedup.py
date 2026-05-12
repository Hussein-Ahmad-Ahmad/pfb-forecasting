import re
import csv
from collections import defaultdict

RESULT_FILE = "result_long_term_forecast.txt"
CLEAN_CSV = "result_long_term_forecast_clean.csv"
BUG_REPORT = "phase3a_issues_found.txt"

# Regex to parse identifier and extract key fields
# Format: long_term_forecast_{model}_{epochs}{model}_{dataset}_{config}_pl{horizon}_{more}_seed_{seed}
# Example: long_term_forecast_PatchTST_100epoch_PatchTST_ETTm1_ftM_sl336_ll48_pl96_dm128_..._baseline_100_0
id_pattern = re.compile(
    r"^long_term_forecast_([A-Za-z0-9]+)_\d+epoch_\1_([A-Za-z0-9]+)_.*?_pl(\d+)_.*?_(\d+)\s*$"
)
mse_mae_pattern = re.compile(r"mse:([\d\.eE+-]+),\s*mae:([\d\.eE+-]+)")

# Track results and issues
results = {}  # (model, dataset, horizon, seed) -> (mse, mae)
duplicates = defaultdict(int)  # count duplicates
naming_bugs = []  # H=336 with _192_ in name
issues = []

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
        
        # Check for duplicates
        if key in results:
            duplicates[key] += 1
            issues.append(f"DUPLICATE: {model} {dataset} H={horizon} seed={seed}")
        
        # Check for H=336 with _192_ in identifier (naming bug)
        if horizon == "336" and "_192_" in id_line:
            naming_bugs.append(f"BUG: H=336 but has _192_ in identifier: {id_line[:80]}...")
            issues.append(f"NAMING BUG: {model} {dataset} has H=336 but _192_ in name")
        
        results[key] = (mse, mae)
    
    i += 2

# Write clean CSV
with open(CLEAN_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Dataset", "Horizon", "Seed", "MSE", "MAE"])
    for (model, dataset, horizon, seed), (mse, mae) in sorted(results.items()):
        writer.writerow([model, dataset, horizon, seed, f"{mse:.10f}", f"{mae:.10f}"])

# Write bug report
with open(BUG_REPORT, "w", encoding="utf-8") as f:
    f.write("Phase 3A Issues Report\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Total unique results parsed: {len(results)}\n")
    f.write(f"Duplicates found: {sum(duplicates.values())}\n")
    f.write(f"Naming bugs (H=336 with _192_): {len(naming_bugs)}\n\n")
    
    if naming_bugs:
        f.write("Naming Bugs Detected:\n")
        for bug in naming_bugs[:10]:
            f.write(f"  {bug}\n")
        if len(naming_bugs) > 10:
            f.write(f"  ... and {len(naming_bugs) - 10} more\n")
    
    if duplicates:
        f.write("\nDuplicate entries (should be investigated):\n")
        for (model, dataset, horizon, seed), count in sorted(duplicates.items())[:10]:
            f.write(f"  {model} {dataset} H={horizon} seed={seed}: {count} duplicates\n")
        if len(duplicates) > 10:
            f.write(f"  ... and {len(duplicates) - 10} more\n")
    
    f.write(f"\nAll issues ({len(issues)} total):\n")
    for issue in issues[:20]:
        f.write(f"  {issue}\n")
    if len(issues) > 20:
        f.write(f"  ... and {len(issues) - 20} more\n")

print(f"✓ Clean CSV written: {CLEAN_CSV} ({len(results)} unique results)")
print(f"✓ Bug report written: {BUG_REPORT}")
print(f"  - Duplicates: {sum(duplicates.values())}")
print(f"  - Naming bugs: {len(naming_bugs)}")
