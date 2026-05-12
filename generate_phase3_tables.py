"""
Generate comprehensive Phase 3 Core Benchmark table
Shows all 198 experiments with performance metrics
"""

import pandas as pd

# Load data
df = pd.read_csv('../complete_results_CORRECTED.csv')

print("Generating Phase 3 Core Benchmark comprehensive table...")
print(f"Total experiments: {len(df)}\n")

# Sort by Dataset, Horizon, MSE
df_sorted = df.sort_values(['Dataset', 'Horizon', 'MSE'])

# Generate markdown table
output = []
output.append("# PHASE 3: CORE BENCHMARK - COMPLETE RESULTS\n")
output.append(f"**Total Experiments**: {len(df)}\n")
output.append(f"**Models**: {df['Model'].nunique()} ({', '.join(sorted(df['Model'].unique()))})\n")
output.append(f"**Datasets**: {df['Dataset'].nunique()} ({', '.join(sorted(df['Dataset'].unique()))})\n")
output.append(f"**Horizons**: {sorted(df['Horizon'].unique())}\n")
output.append("\n---\n\n")

# Table by dataset
for dataset in sorted(df['Dataset'].unique()):
    df_dataset = df[df['Dataset'] == dataset]
    
    output.append(f"\n## {dataset} ({len(df_dataset)} experiments)\n\n")
    
    for horizon in sorted(df_dataset['Horizon'].unique()):
        df_horizon = df_dataset[df_dataset['Horizon'] == horizon].sort_values('MSE')
        
        output.append(f"\n### Horizon = {horizon} ({len(df_horizon)} models)\n\n")
        output.append("| Rank | Model | MSE | MAE | N_runs |\n")
        output.append("|------|-------|-----|-----|--------|\n")
        
        for idx, (i, row) in enumerate(df_horizon.iterrows(), 1):
            output.append(f"| {idx} | {row['Model']:25s} | {row['MSE']:.6f} | {row['MAE']:.6f} | {int(row['N_runs'])} |\n")
    
    output.append("\n---\n")

# Write to file
with open('../PHASE3_CORE_BENCHMARK_FULL_TABLE.md', 'w', encoding='utf-8') as f:
    f.writelines(output)

print("✅ Generated: PHASE3_CORE_BENCHMARK_FULL_TABLE.md")

# Also generate summary table (all datasets × all horizons in one)
output_summary = []
output_summary.append("# PHASE 3: SUMMARY TABLE (All Results)\n\n")

# Pivot table: Models as rows, Dataset×Horizon as columns
output_summary.append("## MSE Results\n\n")
output_summary.append("| Model | " + " | ".join([f"{ds} H={h}" for ds in sorted(df['Dataset'].unique()) for h in sorted(df[df['Dataset']==ds]['Horizon'].unique())]) + " |\n")
output_summary.append("|-------|" + "|".join(["------" for ds in sorted(df['Dataset'].unique()) for h in sorted(df[df['Dataset']==ds]['Horizon'].unique())]) + "|\n")

for model in sorted(df['Model'].unique()):
    row = [model]
    for dataset in sorted(df['Dataset'].unique()):
        for horizon in sorted(df[df['Dataset']==dataset]['Horizon'].unique()):
            match = df[(df['Model']==model) & (df['Dataset']==dataset) & (df['Horizon']==horizon)]
            if len(match) > 0:
                row.append(f"{match.iloc[0]['MSE']:.4f}")
            else:
                row.append("-")
    output_summary.append("| " + " | ".join(row) + " |\n")

# Write summary
with open('../PHASE3_CORE_BENCHMARK_SUMMARY.md', 'w', encoding='utf-8') as f:
    f.writelines(output_summary)

print("✅ Generated: PHASE3_CORE_BENCHMARK_SUMMARY.md")

# Generate ranking summary
output_ranks = []
output_ranks.append("# PHASE 3: MODEL RANKINGS BY DATASET\n\n")

for dataset in sorted(df['Dataset'].unique()):
    df_dataset = df[df['Dataset'] == dataset]
    
    # Calculate average rank across all horizons for this dataset
    ranks = []
    for horizon in sorted(df_dataset['Horizon'].unique()):
        df_h = df_dataset[df_dataset['Horizon'] == horizon].sort_values('MSE').reset_index(drop=True)
        df_h['Rank'] = range(1, len(df_h) + 1)
        ranks.append(df_h[['Model', 'Rank']])
    
    # Merge all horizon ranks
    if len(ranks) > 0:
        rank_df = ranks[0]
        for r in ranks[1:]:
            rank_df = rank_df.merge(r, on='Model', how='outer', suffixes=('', f'_h{len(ranks)}'))
        
        # Calculate average rank
        rank_cols = [c for c in rank_df.columns if c.startswith('Rank')]
        rank_df['Avg_Rank'] = rank_df[rank_cols].mean(axis=1)
        rank_df = rank_df.sort_values('Avg_Rank')
        
        output_ranks.append(f"\n## {dataset}\n\n")
        output_ranks.append("| Rank | Model | Avg Rank |\n")
        output_ranks.append("|------|-------|----------|\n")
        
        for idx, (i, row) in enumerate(rank_df.iterrows(), 1):
            output_ranks.append(f"| {idx} | {row['Model']:25s} | {row['Avg_Rank']:.2f} |\n")

with open('../PHASE3_MODEL_RANKINGS.md', 'w', encoding='utf-8') as f:
    f.writelines(output_ranks)

print("✅ Generated: PHASE3_MODEL_RANKINGS.md")

print("\n" + "="*80)
print("Generated 3 comprehensive tables:")
print("  1. PHASE3_CORE_BENCHMARK_FULL_TABLE.md (detailed by dataset/horizon)")
print("  2. PHASE3_CORE_BENCHMARK_SUMMARY.md (all results in one table)")
print("  3. PHASE3_MODEL_RANKINGS.md (rankings by dataset)")
print("="*80)
