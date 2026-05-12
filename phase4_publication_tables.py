#!/usr/bin/env python3
"""
Phase 4: Publication Tables and Analysis
Generates LaTeX/Markdown tables for journal submission
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path

# Load all results
print("Loading Phase 3 results...")
stats = pd.read_csv('result_long_term_forecast_phase3c_stats.csv')
all_results = pd.read_csv('result_long_term_forecast_phase3c_aggregated.csv')

print(f"Loaded {len(stats)} model/dataset/horizon combinations")
print(f"Total experiments: {len(all_results)}")

# ============================================================================
# TABLE 1: Main Results - Mean ± Std for H=96 (Standard Benchmark)
# ============================================================================
print("\n" + "="*70)
print("TABLE 1: Main Results (H=96) - MSE and MAE with Mean ± Std")
print("="*70)

# Filter for H=96 only
h96_stats = stats[stats['Horizon'] == 96].copy()

# Pivot for MSE
mse_pivot = h96_stats.pivot_table(
    index='Model', 
    columns='Dataset', 
    values=['MSE_mean', 'MSE_std'],
    aggfunc='first'
)

# Pivot for MAE
mae_pivot = h96_stats.pivot_table(
    index='Model',
    columns='Dataset',
    values=['MAE_mean', 'MAE_std'],
    aggfunc='first'
)

# Create formatted table
def format_mean_std(mean, std):
    if pd.isna(mean):
        return "-"
    if std == 0 or pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f}±{std:.4f}"

# Get datasets for H=96
datasets_h96 = sorted(h96_stats['Dataset'].unique())
models = sorted(h96_stats['Model'].unique())

print("\nMSE Results (H=96):")
print("Model".ljust(15), end='')
for ds in datasets_h96:
    print(ds.ljust(20), end='')
print()
print("-" * 100)

for model in models:
    print(model.ljust(15), end='')
    for ds in datasets_h96:
        try:
            mean = mse_pivot.loc[model, ('MSE_mean', ds)]
            std = mse_pivot.loc[model, ('MSE_std', ds)]
            print(format_mean_std(mean, std).ljust(20), end='')
        except:
            print("-".ljust(20), end='')
    print()

print("\nMAE Results (H=96):")
print("Model".ljust(15), end='')
for ds in datasets_h96:
    print(ds.ljust(20), end='')
print()
print("-" * 100)

for model in models:
    print(model.ljust(15), end='')
    for ds in datasets_h96:
        try:
            mean = mae_pivot.loc[model, ('MAE_mean', ds)]
            std = mae_pivot.loc[model, ('MAE_std', ds)]
            print(format_mean_std(mean, std).ljust(20), end='')
        except:
            print("-".ljust(20), end='')
    print()

# ============================================================================
# TABLE 2: LaTeX Format for Paper
# ============================================================================
print("\n" + "="*70)
print("TABLE 2: LaTeX Format (MSE for H=96)")
print("="*70)

latex_lines = []
latex_lines.append("\\begin{table}[t]")
latex_lines.append("\\centering")
latex_lines.append("\\caption{Forecasting Performance (MSE) on H=96 - Mean $\\pm$ Std}")
latex_lines.append("\\label{tab:main_results_h96}")
latex_lines.append("\\begin{tabular}{l" + "c" * len(datasets_h96) + "}")
latex_lines.append("\\toprule")
latex_lines.append("Model & " + " & ".join(datasets_h96) + " \\\\")
latex_lines.append("\\midrule")

for model in models:
    row = [model]
    for ds in datasets_h96:
        try:
            mean = mse_pivot.loc[model, ('MSE_mean', ds)]
            std = mse_pivot.loc[model, ('MSE_std', ds)]
            if std > 0:
                row.append(f"${mean:.4f} \\pm {std:.4f}$")
            else:
                row.append(f"${mean:.4f}$")
        except:
            row.append("-")
    latex_lines.append(" & ".join(row) + " \\\\")

latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\end{table}")

latex_table = "\n".join(latex_lines)
print(latex_table)

# Save LaTeX to file
with open("table_h96_mse_latex.tex", "w") as f:
    f.write(latex_table)
print("\n✓ Saved to: table_h96_mse_latex.tex")

# ============================================================================
# TABLE 3: Model Ranking by Average Performance
# ============================================================================
print("\n" + "="*70)
print("TABLE 3: Model Ranking (Average across all datasets/horizons)")
print("="*70)

ranking = stats.groupby('Model').agg({
    'MSE_mean': ['mean', 'std', 'min', 'max'],
    'MAE_mean': ['mean', 'std', 'min', 'max']
}).round(6)

ranking.columns = ['_'.join(col).strip() for col in ranking.columns.values]
ranking = ranking.sort_values('MSE_mean_mean')
ranking['Rank'] = range(1, len(ranking) + 1)

print(ranking[['Rank', 'MSE_mean_mean', 'MSE_mean_std', 'MAE_mean_mean', 'MAE_mean_std']].to_string())

# Save ranking
ranking.to_csv('phase4_model_ranking.csv')
print("\n✓ Saved to: phase4_model_ranking.csv")

# ============================================================================
# TABLE 4: Best Model per Dataset/Horizon
# ============================================================================
print("\n" + "="*70)
print("TABLE 4: Best Model Selection per Dataset/Horizon")
print("="*70)

best_models = []
for (dataset, horizon), group in stats.groupby(['Dataset', 'Horizon']):
    best_idx = group['MSE_mean'].idxmin()
    best = group.loc[best_idx]
    best_models.append({
        'Dataset': dataset,
        'Horizon': horizon,
        'Best_Model': best['Model'],
        'MSE': f"{best['MSE_mean']:.6f}",
        'MAE': f"{best['MAE_mean']:.6f}",
        'MSE_std': f"{best['MSE_std']:.6f}" if best['MSE_std'] > 0 else "0",
        'Robustness': 'High' if best['MSE_std'] < 0.01 else 'Medium' if best['MSE_std'] < 0.05 else 'Low'
    })

best_df = pd.DataFrame(best_models)
print(best_df.to_string(index=False))

best_df.to_csv('phase4_best_models_per_config.csv', index=False)
print("\n✓ Saved to: phase4_best_models_per_config.csv")

# ============================================================================
# TABLE 5: Statistical Summary
# ============================================================================
print("\n" + "="*70)
print("TABLE 5: Statistical Summary - Win/Lose Matrix")
print("="*70)

# Count wins per model (how many times each model achieved best MSE)
win_counts = best_df['Best_Model'].value_counts().to_dict()
total_configs = len(best_df)

print(f"\nModel Win Counts (out of {total_configs} configurations):")
for model in sorted(models):
    wins = win_counts.get(model, 0)
    pct = (wins / total_configs) * 100
    print(f"{model.ljust(15)}: {wins:2d} wins ({pct:5.1f}%)")

# ============================================================================
# Generate Summary Report
# ============================================================================
print("\n" + "="*70)
print("PHASE 4 SUMMARY")
print("="*70)
print(f"\n✓ Generated 4 publication-ready tables")
print(f"✓ LaTeX table ready for paper submission")
print(f"✓ Model rankings calculated")
print(f"✓ Best model selection per configuration")
print(f"\nFiles created:")
print(f"  - table_h96_mse_latex.tex")
print(f"  - phase4_model_ranking.csv")
print(f"  - phase4_best_models_per_config.csv")

print("\n" + "="*70)
print("NEXT: Ablation Study Planning")
print("="*70)
print(f"Top 3 models for ablation:")
top3 = ranking.head(3).index.tolist()
for i, model in enumerate(top3, 1):
    print(f"  {i}. {model}")
print("\nRecommendation: Focus ablation on PatchTST and iTransformer")
