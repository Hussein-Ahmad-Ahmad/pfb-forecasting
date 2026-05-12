#!/usr/bin/env python3
"""
PUBLICATION-QUALITY ACADEMIC VISUALIZATIONS
Clear, Professional, No Vagueness
Using Real Dataset Timestamps
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Academic style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
})

sns.set_palette("Set2")

print("="*80)
print(" "*15 + "PUBLICATION-QUALITY ACADEMIC VISUALIZATIONS")
print("="*80)

# Load results
df_complete = pd.read_csv('results_analysis/complete_results_all_models.csv')
df_rankings = pd.read_csv('results_analysis/complete_model_rankings.csv')

output_dir = Path('results_analysis/publication_figures')
output_dir.mkdir(exist_ok=True, parents=True)

# Color scheme for models
BERT_MODELS = ['BERTOnly', 'PatchFusionBERT_v0', 'PatchFusionBERT_v2']
COLOR_SCHEME = {
    'BERTOnly': '#D32F2F',  # Red (Winner)
    'PatchFusionBERT_v0': '#C2185B',  # Pink (3rd place)
    'PatchFusionBERT_v2': '#7B1FA2',  # Purple (7th)
    'PatchTST': '#1976D2',  # Blue (2nd place)
    'iTransformer': '#388E3C',  # Green
    'DLinear': '#F57C00',  # Orange
    'TimeXer': '#0097A7',  # Cyan
    'TiDE': '#5D4037',  # Brown
    'Autoformer': '#616161',  # Gray
    'Transformer': '#455A64',  # Blue Gray
    'Informer': '#37474F',  # Dark Gray
}

# ============================================================================
# FIGURE 1: Overall Model Performance Ranking (Bar Chart)
# ============================================================================
print("\n📊 Figure 1: Overall Model Performance Ranking")
print("-" * 80)

fig, ax = plt.subplots(figsize=(10, 7))

models = df_rankings.sort_values('Rank')['Model'].values
mse_means = df_rankings.sort_values('Rank')['MSE_mean'].values
mse_stds = df_rankings.sort_values('Rank')['MSE_std'].values

colors = [COLOR_SCHEME.get(m, '#757575') for m in models]

# Create horizontal bar chart
y_pos = np.arange(len(models))
bars = ax.barh(y_pos, mse_means, xerr=mse_stds, capsize=4,
               color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)

# Customize
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel('Mean Squared Error (MSE) ± Std', fontweight='bold')
ax.set_title('Overall Model Performance Ranking Across All Datasets',
             fontweight='bold', pad=15)

# Add rank numbers and values
for i, (model, mse, std) in enumerate(zip(models, mse_means, mse_stds)):
    rank = i + 1
    # Rank on left
    ax.text(-0.05, i, f'#{rank}', ha='right', va='center',
            fontweight='bold', fontsize=11)
    # Value on right
    ax.text(mse + std + 0.05, i, f'{mse:.4f}',
            ha='left', va='center', fontsize=9)

# Add legend for PatchFusionBERT models
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#D32F2F', label='BERTOnly (Rank #1)', edgecolor='black'),
    Patch(facecolor='#C2185B', label='PatchFusionBERT_v0 (Rank #3)', edgecolor='black'),
    Patch(facecolor='#7B1FA2', label='PatchFusionBERT_v2 (Rank #7)', edgecolor='black'),
    Patch(facecolor='#1976D2', label='PatchTST Baseline (Rank #2)', edgecolor='black'),
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, 
          fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig(output_dir / 'fig1_overall_ranking.png', dpi=300)
plt.savefig(output_dir / 'fig1_overall_ranking.pdf')
print(f"✓ Saved: {output_dir / 'fig1_overall_ranking.png'}")
print(f"✓ Saved: {output_dir / 'fig1_overall_ranking.pdf'}")
plt.close()

# ============================================================================
# FIGURE 2: PatchFusionBERT Variants Performance Heatmap
# ============================================================================
print("\n📊 Figure 2: PatchFusionBERT Variants Performance Heatmap")
print("-" * 80)

bert_df = df_complete[df_complete['Model'].isin(BERT_MODELS)]
pivot_mse = bert_df.pivot_table(values='MSE_mean', index='Model', 
                                 columns='Dataset', aggfunc='mean')

# Reorder models by rank
model_order = ['BERTOnly', 'PatchFusionBERT_v0', 'PatchFusionBERT_v2']
pivot_mse = pivot_mse.reindex(model_order)

fig, ax = plt.subplots(figsize=(10, 5))

# Create heatmap
sns.heatmap(pivot_mse, annot=True, fmt='.4f', cmap='RdYlGn_r',
            cbar_kws={'label': 'MSE (Lower is Better)'},
            linewidths=1.5, linecolor='black',
            vmin=0, vmax=pivot_mse.max().max(),
            ax=ax, square=False, annot_kws={'size': 10, 'weight': 'bold'})

ax.set_title('PatchFusionBERT Variants: Dataset-Wise Performance',
             fontweight='bold', pad=15)
ax.set_xlabel('Dataset', fontweight='bold')
ax.set_ylabel('Model Variant', fontweight='bold')

# Rotate labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(output_dir / 'fig2_bert_variants_heatmap.png', dpi=300)
plt.savefig(output_dir / 'fig2_bert_variants_heatmap.pdf')
print(f"✓ Saved: {output_dir / 'fig2_bert_variants_heatmap.png'}")
print(f"✓ Saved: {output_dir / 'fig2_bert_variants_heatmap.pdf'}")
plt.close()

# ============================================================================
# FIGURE 3: Performance by Dataset (Box Plots)
# ============================================================================
print("\n📊 Figure 3: Performance Distribution by Dataset")
print("-" * 80)

fig, ax = plt.subplots(figsize=(12, 6))

datasets = sorted(df_complete['Dataset'].unique())
data_to_plot = [df_complete[df_complete['Dataset'] == d]['MSE_mean'].values 
                for d in datasets]

bp = ax.boxplot(data_to_plot, labels=datasets, patch_artist=True,
                showmeans=True, meanline=True,
                boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.2),
                medianprops=dict(color='red', linewidth=2),
                meanprops=dict(color='blue', linestyle='--', linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2))

ax.set_xlabel('Dataset', fontweight='bold')
ax.set_ylabel('Mean Squared Error (MSE)', fontweight='bold')
ax.set_title('Performance Distribution Across Datasets (All Models)',
             fontweight='bold', pad=15)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linewidth=2, label='Median'),
    Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean'),
]
ax.legend(handles=legend_elements, loc='upper right')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(output_dir / 'fig3_dataset_distribution.png', dpi=300)
plt.savefig(output_dir / 'fig3_dataset_distribution.pdf')
print(f"✓ Saved: {output_dir / 'fig3_dataset_distribution.png'}")
print(f"✓ Saved: {output_dir / 'fig3_dataset_distribution.pdf'}")
plt.close()

# ============================================================================
# FIGURE 4: Top-3 Models Comparison (Multi-Panel)
# ============================================================================
print("\n📊 Figure 4: Top-3 Models Detailed Comparison")
print("-" * 80)

top3_models = df_rankings.sort_values('Rank').head(3)['Model'].values
top3_df = df_complete[df_complete['Model'].isin(top3_models)]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: MSE Comparison
ax1 = axes[0, 0]
for model in top3_models:
    data = top3_df[top3_df['Model'] == model].groupby('Dataset')['MSE_mean'].mean()
    ax1.plot(data.index, data.values, 'o-', label=model, linewidth=2.5,
             markersize=8, color=COLOR_SCHEME[model])
ax1.set_xlabel('Dataset', fontweight='bold')
ax1.set_ylabel('MSE', fontweight='bold')
ax1.set_title('(a) MSE by Dataset', fontweight='bold', loc='left')
ax1.legend(loc='best', frameon=True)
ax1.tick_params(axis='x', rotation=45)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Panel 2: MAE Comparison
ax2 = axes[0, 1]
for model in top3_models:
    data = top3_df[top3_df['Model'] == model].groupby('Dataset')['MAE_mean'].mean()
    ax2.plot(data.index, data.values, 's-', label=model, linewidth=2.5,
             markersize=8, color=COLOR_SCHEME[model])
ax2.set_xlabel('Dataset', fontweight='bold')
ax2.set_ylabel('MAE', fontweight='bold')
ax2.set_title('(b) MAE by Dataset', fontweight='bold', loc='left')
ax2.legend(loc='best', frameon=True)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Panel 3: Performance by Horizon
ax3 = axes[1, 0]
horizons = sorted(top3_df['Horizon'].unique())
x_pos = np.arange(len(horizons))
width = 0.25

for i, model in enumerate(top3_models):
    data = [top3_df[(top3_df['Model'] == model) & 
                    (top3_df['Horizon'] == h)]['MSE_mean'].mean() 
            for h in horizons]
    ax3.bar(x_pos + i*width, data, width, label=model,
            color=COLOR_SCHEME[model], edgecolor='black', linewidth=1)

ax3.set_xlabel('Forecast Horizon', fontweight='bold')
ax3.set_ylabel('Average MSE', fontweight='bold')
ax3.set_title('(c) Performance by Forecast Horizon', fontweight='bold', loc='left')
ax3.set_xticks(x_pos + width)
ax3.set_xticklabels([f'H={h}' for h in horizons])
ax3.legend(loc='best', frameon=True)

# Panel 4: Win Rate
ax4 = axes[1, 1]
win_counts = {}
for model in top3_models:
    wins = 0
    for dataset in datasets:
        for horizon in horizons:
            subset = df_complete[(df_complete['Dataset'] == dataset) & 
                                (df_complete['Horizon'] == horizon)]
            if len(subset) > 0 and subset['MSE_mean'].min() == \
               subset[subset['Model'] == model]['MSE_mean'].min():
                wins += 1
    win_counts[model] = wins

models_list = list(win_counts.keys())
wins_list = list(win_counts.values())
bars = ax4.bar(models_list, wins_list, color=[COLOR_SCHEME[m] for m in models_list],
               edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Number of Wins', fontweight='bold')
ax4.set_title('(d) Win Count Across Configurations', fontweight='bold', loc='left')
ax4.set_ylim([0, max(wins_list) * 1.2])

# Add value labels on bars
for bar, val in zip(bars, wins_list):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(val)}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Top-3 Models Comprehensive Comparison', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'fig4_top3_comparison.png', dpi=300)
plt.savefig(output_dir / 'fig4_top3_comparison.pdf')
print(f"✓ Saved: {output_dir / 'fig4_top3_comparison.png'}")
print(f"✓ Saved: {output_dir / 'fig4_top3_comparison.pdf'}")
plt.close()

# ============================================================================
# FIGURE 5: Statistical Comparison (Violin + Scatter)
# ============================================================================
print("\n📊 Figure 5: Statistical Performance Analysis")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Violin plot
ax1 = axes[0]
top5_models = df_rankings.sort_values('Rank').head(5)['Model'].values
top5_df = df_complete[df_complete['Model'].isin(top5_models)]

violin_data = [top5_df[top5_df['Model'] == m]['MSE_mean'].values 
               for m in top5_models]
parts = ax1.violinplot(violin_data, positions=range(len(top5_models)),
                       showmeans=True, showmedians=True)

# Color the violins
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(COLOR_SCHEME.get(top5_models[i], '#757575'))
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.2)

ax1.set_xticks(range(len(top5_models)))
ax1.set_xticklabels(top5_models, rotation=45, ha='right')
ax1.set_ylabel('MSE Distribution', fontweight='bold')
ax1.set_title('(a) MSE Distribution - Top 5 Models', fontweight='bold', loc='left')

# MSE vs MAE scatter
ax2 = axes[1]
for model in top5_models:
    model_data = top5_df[top5_df['Model'] == model]
    ax2.scatter(model_data['MSE_mean'], model_data['MAE_mean'],
                label=model, s=100, alpha=0.7, edgecolors='black',
                linewidth=1.5, color=COLOR_SCHEME.get(model, '#757575'))

# Add regression line
mse_all = top5_df['MSE_mean'].values
mae_all = top5_df['MAE_mean'].values
z = np.polyfit(mse_all, mae_all, 1)
p = np.poly1d(z)
ax2.plot(sorted(mse_all), p(sorted(mse_all)), "k--", alpha=0.5, linewidth=2,
         label=f'Trend: MAE = {z[0]:.2f}×MSE + {z[1]:.2f}')

# Calculate correlation
corr = np.corrcoef(mse_all, mae_all)[0, 1]
ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}',
         transform=ax2.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         fontweight='bold')

ax2.set_xlabel('MSE', fontweight='bold')
ax2.set_ylabel('MAE', fontweight='bold')
ax2.set_title('(b) MSE vs MAE Correlation', fontweight='bold', loc='left')
ax2.legend(loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig(output_dir / 'fig5_statistical_analysis.png', dpi=300)
plt.savefig(output_dir / 'fig5_statistical_analysis.pdf')
print(f"✓ Saved: {output_dir / 'fig5_statistical_analysis.png'}")
print(f"✓ Saved: {output_dir / 'fig5_statistical_analysis.pdf'}")
plt.close()

# ============================================================================
# FIGURE 6: Conference vs Journal Comparison
# ============================================================================
print("\n📊 Figure 6: Conference vs Journal Extension Comparison")
print("-" * 80)

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['BERTOnly\n(NEW)', 'PatchFusion\nBERT_v0\n(Conference)', 
              'PatchFusion\nBERT_v2\n(NEW)', 'PatchTST\n(Baseline)']
mse_values = [
    df_rankings[df_rankings['Model'] == 'BERTOnly']['MSE_mean'].values[0],
    df_rankings[df_rankings['Model'] == 'PatchFusionBERT_v0']['MSE_mean'].values[0],
    df_rankings[df_rankings['Model'] == 'PatchFusionBERT_v2']['MSE_mean'].values[0],
    df_rankings[df_rankings['Model'] == 'PatchTST']['MSE_mean'].values[0],
]
ranks = [1, 3, 7, 2]
colors_bar = ['#D32F2F', '#C2185B', '#7B1FA2', '#1976D2']

x_pos = np.arange(len(categories))
bars = ax.bar(x_pos, mse_values, color=colors_bar, edgecolor='black',
              linewidth=2, alpha=0.85, width=0.6)

# Add annotations
for i, (bar, val, rank) in enumerate(zip(bars, mse_values, ranks)):
    height = bar.get_height()
    # MSE value
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'MSE: {val:.4f}', ha='center', va='bottom',
            fontweight='bold', fontsize=10)
    # Rank
    ax.text(bar.get_x() + bar.get_width()/2., height/2,
            f'Rank #{rank}', ha='center', va='center',
            fontweight='bold', fontsize=11, color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontweight='bold')
ax.set_ylabel('Mean Squared Error (MSE)', fontweight='bold')
ax.set_title('Conference Paper vs Journal Extension: Model Variants Comparison',
             fontweight='bold', pad=15)
ax.set_ylim([0, max(mse_values) * 1.15])

# Add improvement annotation
improvement = ((mse_values[3] - mse_values[0]) / mse_values[3]) * 100
ax.annotate(f'BERTOnly improves\nPatchTST by {improvement:.1f}%',
            xy=(0, mse_values[0]), xytext=(1.5, max(mse_values) * 0.9),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'),
            fontweight='bold', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig(output_dir / 'fig6_conference_journal_comparison.png', dpi=300)
plt.savefig(output_dir / 'fig6_conference_journal_comparison.pdf')
print(f"✓ Saved: {output_dir / 'fig6_conference_journal_comparison.png'}")
print(f"✓ Saved: {output_dir / 'fig6_conference_journal_comparison.pdf'}")
plt.close()

print("\n" + "="*80)
print("ALL PUBLICATION FIGURES GENERATED")
print("="*80)
print(f"\nLocation: {output_dir}/")
print("\nFigures Created:")
print("  1. Overall Model Ranking (Bar Chart)")
print("  2. PatchFusionBERT Variants Heatmap")
print("  3. Dataset Performance Distribution (Box Plots)")
print("  4. Top-3 Models Comparison (4-Panel)")
print("  5. Statistical Analysis (Violin + Scatter)")
print("  6. Conference vs Journal Comparison")
print("\nAll figures saved as PNG (300 DPI) and PDF (vector)")
print("="*80)
