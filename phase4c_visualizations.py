#!/usr/bin/env python3
"""
Phase 4C: Comprehensive Results Analysis and Visualization
Creates publication-ready figures and statistical analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Create output directory
output_dir = Path("results_analysis/figures")
output_dir.mkdir(exist_ok=True)

print("="*70)
print("PHASE 4C: COMPREHENSIVE RESULTS ANALYSIS")
print("="*70)

# Load data
stats = pd.read_csv('results_analysis/result_long_term_forecast_phase3c_stats.csv')
ranking = pd.read_csv('results_analysis/phase4_model_ranking.csv', index_col=0)

print(f"\nLoaded {len(stats)} model/dataset/horizon combinations")
print(f"Models: {stats['Model'].nunique()}")
print(f"Datasets: {stats['Dataset'].nunique()}")

# ============================================================================
# FIGURE 1: Model Performance Heatmap
# ============================================================================
print("\n" + "="*70)
print("FIGURE 1: Model Performance Heatmap (MSE)")
print("="*70)

# Filter for H=96 only for cleaner visualization
h96_stats = stats[stats['Horizon'] == 96].copy()

# Create pivot table for heatmap
heatmap_data = h96_stats.pivot_table(
    index='Model',
    columns='Dataset',
    values='MSE_mean',
    aggfunc='first'
)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn_r', 
            cbar_kws={'label': 'MSE'}, ax=ax, linewidths=0.5)
ax.set_title('Model Performance Heatmap (H=96) - Lower is Better', fontsize=14, fontweight='bold')
ax.set_xlabel('Dataset', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / 'fig1_performance_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig1_performance_heatmap.pdf', bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'fig1_performance_heatmap.png'}")
plt.close()

# ============================================================================
# FIGURE 2: Model Ranking Bar Chart
# ============================================================================
print("\n" + "="*70)
print("FIGURE 2: Overall Model Rankings")
print("="*70)

fig, ax = plt.subplots(figsize=(10, 6))
models = ranking.sort_values('MSE_mean_mean').index
mse_means = ranking.loc[models, 'MSE_mean_mean']
colors = ['#2ecc71' if i < 3 else '#3498db' for i in range(len(models))]

bars = ax.barh(models, mse_means, color=colors, edgecolor='black', linewidth=0.7)
ax.set_xlabel('Average MSE (across all datasets/horizons)', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
ax.set_title('Model Performance Rankings - Lower is Better', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# Add value labels
for i, (model, value) in enumerate(zip(models, mse_means)):
    ax.text(value + 0.05, i, f'{value:.3f}', va='center', fontsize=10)

# Add rank labels
for i, model in enumerate(models):
    rank = i + 1
    medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else ''
    ax.text(-0.1, i, f'{medal} #{rank}', va='center', ha='right', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'fig2_model_rankings.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig2_model_rankings.pdf', bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'fig2_model_rankings.png'}")
plt.close()

# ============================================================================
# FIGURE 3: Dataset Difficulty Analysis
# ============================================================================
print("\n" + "="*70)
print("FIGURE 3: Dataset Difficulty Analysis")
print("="*70)

dataset_stats = h96_stats.groupby('Dataset').agg({
    'MSE_mean': ['mean', 'std', 'min', 'max'],
    'MAE_mean': ['mean', 'std']
}).round(4)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Box plot of MSE by dataset
datasets = h96_stats['Dataset'].unique()
mse_by_dataset = [h96_stats[h96_stats['Dataset'] == ds]['MSE_mean'].values for ds in datasets]

bp = ax1.boxplot(mse_by_dataset, labels=datasets, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#3498db')
    patch.set_alpha(0.7)
ax1.set_ylabel('MSE', fontsize=12)
ax1.set_xlabel('Dataset', fontsize=12)
ax1.set_title('MSE Distribution by Dataset (H=96)', fontsize=12, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Bar chart of average MSE
dataset_avg = h96_stats.groupby('Dataset')['MSE_mean'].mean().sort_values()
colors_datasets = plt.cm.viridis(np.linspace(0, 1, len(dataset_avg)))
ax2.barh(dataset_avg.index, dataset_avg.values, color=colors_datasets, edgecolor='black')
ax2.set_xlabel('Average MSE', fontsize=12)
ax2.set_title('Dataset Difficulty Ranking', fontsize=12, fontweight='bold')
ax2.invert_yaxis()

for i, (ds, val) in enumerate(zip(dataset_avg.index, dataset_avg.values)):
    ax2.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'fig3_dataset_difficulty.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig3_dataset_difficulty.pdf', bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'fig3_dataset_difficulty.png'}")
plt.close()

# ============================================================================
# FIGURE 4: Top Models Comparison
# ============================================================================
print("\n" + "="*70)
print("FIGURE 4: Top 4 Models Detailed Comparison")
print("="*70)

top4_models = ranking.head(4).index.tolist()
top4_data = h96_stats[h96_stats['Model'].isin(top4_models)]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, dataset in enumerate(sorted(h96_stats['Dataset'].unique())[:4]):
    ax = axes[idx]
    data = top4_data[top4_data['Dataset'] == dataset].sort_values('MSE_mean')
    
    bars = ax.bar(range(len(data)), data['MSE_mean'], 
                   color=plt.cm.Set3(np.linspace(0, 1, len(data))),
                   edgecolor='black', linewidth=1.2)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data['Model'], rotation=45, ha='right')
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title(f'{dataset} (H=96)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Highlight best model
    best_idx = data['MSE_mean'].idxmin()
    best_model = data.loc[best_idx, 'Model']
    bars[0].set_color('#2ecc71')
    bars[0].set_edgecolor('black')
    bars[0].set_linewidth(2)
    
    # Add value labels
    for i, (_, row) in enumerate(data.iterrows()):
        ax.text(i, row['MSE_mean'] + 0.01, f"{row['MSE_mean']:.3f}", 
                ha='center', va='bottom', fontsize=9)

plt.suptitle('Top 4 Models Performance Comparison', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(output_dir / 'fig4_top_models_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig4_top_models_comparison.pdf', bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'fig4_top_models_comparison.png'}")
plt.close()

# ============================================================================
# FIGURE 5: MSE vs MAE Correlation
# ============================================================================
print("\n" + "="*70)
print("FIGURE 5: MSE vs MAE Analysis")
print("="*70)

fig, ax = plt.subplots(figsize=(10, 7))

for model in stats['Model'].unique():
    model_data = stats[stats['Model'] == model]
    ax.scatter(model_data['MSE_mean'], model_data['MAE_mean'], 
               label=model, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

ax.set_xlabel('MSE', fontsize=12)
ax.set_ylabel('MAE', fontsize=12)
ax.set_title('MSE vs MAE: Model Performance Correlation', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# Add regression line
from scipy.stats import linregress
x = stats['MSE_mean'].values
y = stats['MAE_mean'].values
slope, intercept, r_value, p_value, std_err = linregress(x, y)
line_x = np.linspace(x.min(), x.max(), 100)
line_y = slope * line_x + intercept
ax.plot(line_x, line_y, 'r--', alpha=0.5, linewidth=2, label=f'R²={r_value**2:.3f}')

plt.tight_layout()
plt.savefig(output_dir / 'fig5_mse_mae_correlation.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig5_mse_mae_correlation.pdf', bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'fig5_mse_mae_correlation.png'}")
plt.close()

# ============================================================================
# Summary Statistics Table
# ============================================================================
print("\n" + "="*70)
print("GENERATING SUMMARY STATISTICS")
print("="*70)

summary = {
    'Total_Experiments': len(stats),
    'Models_Tested': stats['Model'].nunique(),
    'Datasets': stats['Dataset'].nunique(),
    'Best_Model': ranking.index[0],
    'Best_MSE': ranking.loc[ranking.index[0], 'MSE_mean_mean'],
    'Easiest_Dataset': h96_stats.groupby('Dataset')['MSE_mean'].mean().idxmin(),
    'Hardest_Dataset': h96_stats.groupby('Dataset')['MSE_mean'].mean().idxmax(),
    'MSE_Range': f"{stats['MSE_mean'].min():.4f} - {stats['MSE_mean'].max():.4f}",
    'MAE_Range': f"{stats['MAE_mean'].min():.4f} - {stats['MAE_mean'].max():.4f}"
}

summary_df = pd.DataFrame([summary]).T
summary_df.columns = ['Value']
summary_df.to_csv(output_dir.parent / 'analysis_summary_stats.csv')
print("\n" + summary_df.to_string())
print(f"\n✓ Saved: {output_dir.parent / 'analysis_summary_stats.csv'}")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*70)
print("PHASE 4C COMPLETE - ALL FIGURES GENERATED")
print("="*70)
print(f"\n📊 Generated 5 publication-ready figures:")
print(f"  1. Performance Heatmap (all models × datasets)")
print(f"  2. Model Rankings Bar Chart (with medals)")
print(f"  3. Dataset Difficulty Analysis (box plots + bars)")
print(f"  4. Top 4 Models Comparison (detailed)")
print(f"  5. MSE vs MAE Correlation (scatter + regression)")
print(f"\n📁 Location: {output_dir}/")
print(f"  - PNG files (300 DPI) for presentations")
print(f"  - PDF files (vector) for publications")
print(f"\n✓ Summary statistics saved to CSV")
print("="*70)
