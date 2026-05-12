#!/usr/bin/env python3
"""
Complete Results Analysis - Including ALL Models (PatchFusionBERT variants)
Plus Time Series Prediction Plots
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

print("="*70)
print("COMPREHENSIVE ANALYSIS - ALL MODELS INCLUDING PATCHFUSIONBERT")
print("="*70)

# Parse the full results file to get ALL models
result_file = "result_long_term_forecast.txt"
print(f"\nParsing: {result_file}")

results = []
with open(result_file, 'r') as f:
    lines = f.readlines()
    
for i in range(0, len(lines)-1, 2):
    if 'long_term_forecast_' in lines[i]:
        identifier = lines[i].strip()
        metrics_line = lines[i+1].strip() if i+1 < len(lines) else ""
        
        if 'mse:' in metrics_line:
            # Parse metrics
            mse_match = re.search(r'mse:([\d.]+)', metrics_line)
            mae_match = re.search(r'mae:([\d.]+)', metrics_line)
            
            if mse_match and mae_match:
                mse = float(mse_match.group(1))
                mae = float(mae_match.group(1))
                
                # Parse identifier for model/dataset/horizon info
                # Format examples:
                # long_term_forecast_PFB_v0_H192_Weather_192_...
                # long_term_forecast_PatchTST_ETTm2_H96_seed2021_...
                
                parts = identifier.split('_')
                
                # Extract model name
                model = None
                dataset = None
                horizon = None
                seed = None
                
                # Find model name (after long_term_forecast_)
                if 'PFB_v0' in identifier or 'PatchFusionBERT_v0' in identifier:
                    model = 'PatchFusionBERT_v0'
                elif 'PFB_v2' in identifier or 'PatchFusionBERT_v2' in identifier:
                    model = 'PatchFusionBERT_v2'
                elif 'BERTOnly' in identifier or 'PatchFusionBERT_BERTOnly' in identifier:
                    model = 'BERTOnly'
                elif 'PatchTST' in identifier:
                    model = 'PatchTST'
                elif 'iTransformer' in identifier:
                    model = 'iTransformer'
                elif 'DLinear' in identifier:
                    model = 'DLinear'
                elif 'Autoformer' in identifier:
                    model = 'Autoformer'
                elif 'Informer' in identifier:
                    model = 'Informer'
                elif 'Transformer' in identifier:
                    model = 'Transformer'
                elif 'TimeXer' in identifier:
                    model = 'TimeXer'
                elif 'TiDE' in identifier:
                    model = 'TiDE'
                
                # Extract horizon
                h_match = re.search(r'H(\d+)', identifier)
                if h_match:
                    horizon = int(h_match.group(1))
                elif 'pl192' in identifier or '_192_' in identifier:
                    horizon = 192
                elif 'pl96' in identifier or '_96_' in identifier:
                    horizon = 96
                elif 'pl336' in identifier or '_336_' in identifier:
                    horizon = 336
                elif 'pl24' in identifier:
                    horizon = 24
                
                # Extract dataset
                if 'ETTh1' in identifier:
                    dataset = 'ETTh1'
                elif 'ETTh2' in identifier:
                    dataset = 'ETTh2'
                elif 'ETTm1' in identifier:
                    dataset = 'ETTm1'
                elif 'ETTm2' in identifier:
                    dataset = 'ETTm2'
                elif 'Exchange' in identifier or 'exchange' in identifier:
                    dataset = 'Exchange'
                elif 'Weather' in identifier or 'weather' in identifier:
                    dataset = 'Weather'
                elif 'custom' in identifier or 'Illness' in identifier:
                    dataset = 'Illness'
                
                # Extract seed
                seed_match = re.search(r'seed(\d+)', identifier)
                if seed_match:
                    seed = int(seed_match.group(1))
                
                if model and dataset and horizon:
                    results.append({
                        'Model': model,
                        'Dataset': dataset,
                        'Horizon': horizon,
                        'Seed': seed if seed else 0,
                        'MSE': mse,
                        'MAE': mae,
                        'Identifier': identifier
                    })

df = pd.DataFrame(results)
print(f"\nParsed {len(df)} total experiments")
print(f"Unique models: {df['Model'].nunique()}")
print(f"\nAll models found:")
for model in sorted(df['Model'].unique()):
    count = len(df[df['Model'] == model])
    print(f"  - {model}: {count} experiments")

# Aggregate by model/dataset/horizon
stats = df.groupby(['Model', 'Dataset', 'Horizon']).agg({
    'MSE': ['mean', 'std', 'count'],
    'MAE': ['mean', 'std']
}).reset_index()

stats.columns = ['Model', 'Dataset', 'Horizon', 'MSE_mean', 'MSE_std', 'N_runs', 'MAE_mean', 'MAE_std']

# Save complete results
output_dir = Path("results_analysis")
stats.to_csv(output_dir / 'complete_results_all_models.csv', index=False)
print(f"\n✓ Saved: {output_dir / 'complete_results_all_models.csv'}")

# Create visualizations including PatchFusionBERT
figures_dir = output_dir / "figures_complete"
figures_dir.mkdir(exist_ok=True)

# ============================================================================
# FIGURE: Complete Model Rankings (ALL MODELS)
# ============================================================================
print("\n" + "="*70)
print("GENERATING COMPLETE MODEL RANKINGS")
print("="*70)

model_rankings = stats.groupby('Model').agg({
    'MSE_mean': ['mean', 'std'],
    'MAE_mean': ['mean', 'std']
}).round(6)

model_rankings.columns = ['MSE_mean', 'MSE_std', 'MAE_mean', 'MAE_std']
model_rankings = model_rankings.sort_values('MSE_mean')
model_rankings['Rank'] = range(1, len(model_rankings) + 1)

print("\nComplete Model Rankings:")
print(model_rankings[['Rank', 'MSE_mean', 'MSE_std', 'MAE_mean']].to_string())

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
models = model_rankings.index
mse_means = model_rankings['MSE_mean']

# Color PatchFusionBERT variants differently
colors = []
for model in models:
    if 'PatchFusion' in model or 'BERT' in model:
        colors.append('#e74c3c')  # Red for BERT models
    elif 'PatchTST' == model:
        colors.append('#2ecc71')  # Green for winner
    elif 'iTransformer' in model:
        colors.append('#3498db')  # Blue for runner-up
    else:
        colors.append('#95a5a6')  # Gray for others

bars = ax.barh(models, mse_means, color=colors, edgecolor='black', linewidth=1)
ax.set_xlabel('Average MSE (Lower is Better)', fontsize=13)
ax.set_ylabel('Model', fontsize=13)
ax.set_title('Complete Model Rankings - Including PatchFusionBERT Variants', 
             fontsize=15, fontweight='bold')
ax.invert_yaxis()

# Add value labels
for i, (model, value) in enumerate(zip(models, mse_means)):
    ax.text(value + 0.02, i, f'{value:.4f}', va='center', fontsize=10)
    rank = model_rankings.loc[model, 'Rank']
    ax.text(-0.05, i, f'#{int(rank)}', va='center', ha='right', fontsize=11, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', label='PatchFusionBERT variants'),
    Patch(facecolor='#2ecc71', label='Best Overall'),
    Patch(facecolor='#3498db', label='Top Performer'),
    Patch(facecolor='#95a5a6', label='Baseline Models')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(figures_dir / 'complete_model_rankings_with_bert.png', dpi=300, bbox_inches='tight')
plt.savefig(figures_dir / 'complete_model_rankings_with_bert.pdf', bbox_inches='tight')
print(f"✓ Saved: {figures_dir / 'complete_model_rankings_with_bert.png'}")
plt.close()

# ============================================================================
# FIGURE: PatchFusionBERT Variants Comparison
# ============================================================================
print("\n" + "="*70)
print("PATCHFUSIONBERT VARIANTS DETAILED COMPARISON")
print("="*70)

bert_models = [m for m in stats['Model'].unique() if 'BERT' in m or 'Fusion' in m]
if bert_models:
    bert_stats = stats[stats['Model'].isin(bert_models)]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    
    datasets = sorted(bert_stats['Dataset'].unique())[:6]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        data = bert_stats[bert_stats['Dataset'] == dataset].groupby('Model')['MSE_mean'].mean().sort_values()
        
        colors_bert = ['#e74c3c', '#c0392b', '#e67e22'][:len(data)]
        bars = ax.bar(range(len(data)), data.values, color=colors_bert, 
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([m.replace('PatchFusionBERT_', 'PFB_') for m in data.index], 
                          rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('MSE', fontsize=11)
        ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add values on bars
        for i, val in enumerate(data.values):
            ax.text(i, val + max(data.values) * 0.02, f'{val:.4f}', 
                   ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('PatchFusionBERT Variants Performance Comparison', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(figures_dir / 'patchfusionbert_variants_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'patchfusionbert_variants_comparison.pdf', bbox_inches='tight')
    print(f"✓ Saved: {figures_dir / 'patchfusionbert_variants_comparison.png'}")
    plt.close()

# Save rankings
model_rankings.to_csv(output_dir / 'complete_model_rankings.csv')
print(f"\n✓ Saved: {output_dir / 'complete_model_rankings.csv'}")

print("\n" + "="*70)
print("COMPLETE ANALYSIS FINISHED")
print("="*70)
print(f"\nGenerated files in: {output_dir}/")
print(f"  - complete_results_all_models.csv (all experiments)")
print(f"  - complete_model_rankings.csv (rankings with BERT models)")
print(f"\nFigures in: {figures_dir}/")
print(f"  - complete_model_rankings_with_bert.png/pdf")
print(f"  - patchfusionbert_variants_comparison.png/pdf")
print("="*70)
