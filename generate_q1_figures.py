"""
Q1 Journal-Quality Figure Generation for Manuscript
Excludes H=720, focuses on H={96, 192, 336}
Output: doc.j-8-02/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Model name mapping for cleaner labels
def clean_model_name(model):
    """Convert long model names to short publication-ready labels"""
    mapping = {
        'PatchFusionBERT_v0': 'PFB-v0',
        'PatchFusionBERT_v2': 'PFB-v2',
        'PatchFusionBERT_BERTOnly': 'BERT',
        'PatchTST': 'PatchTST',
        'iTransformer': 'iTransformer',
        'DLinear': 'DLinear',
        'TimeXer': 'TimeXer',
        'TiDE': 'TiDE',
        'Autoformer': 'Autoformer',
        'Informer': 'Informer',
        'Transformer': 'Transformer'
    }
    return mapping.get(model, model)

# Set publication-quality style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Output directory
OUTPUT_DIR = Path('../doc.j-8-02/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv('results_23-01.csv')

# Filter: Exclude H=720 and other non-standard horizons
VALID_HORIZONS = [96, 192, 336]
df = df[df['Horizon'].isin(VALID_HORIZONS)].copy()

print(f"Filtered data: {len(df)} experiments (H={VALID_HORIZONS})")
print(f"Datasets: {df['Dataset'].unique()}")
print(f"Models: {df['Model'].unique()}")

# Model groups for analysis
PFB_MODELS = ['PatchFusionBERT_v0', 'PatchFusionBERT_v2']  # Only 2 fusion variants
BASELINE_MODELS = ['DLinear', 'PatchTST', 'TiDE', 'TimeXer', 'iTransformer', 'PatchFusionBERT_BERTOnly']
ALL_MODELS = PFB_MODELS + BASELINE_MODELS

# Filter to relevant models
df = df[df['Model'].isin(ALL_MODELS)].copy()

print(f"Final dataset: {len(df)} experiments with {len(ALL_MODELS)} models")

# ============================================================================
# Figure 1: Main Performance Comparison (Table as Heatmap)
# ============================================================================
def create_performance_heatmap():
    """Create publication-quality performance heatmap"""
    
    # Aggregate by model and horizon
    perf = df.groupby(['Model', 'Horizon'])['MSE'].mean().unstack()
    perf = perf.reindex(ALL_MODELS)  # Preserve order
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    sns.heatmap(perf, annot=True, fmt='.4f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Mean Squared Error (MSE)'},
                linewidths=0.5, linecolor='white',
                vmin=perf.min().min() * 0.95,
                vmax=perf.max().max() * 1.05,
                ax=ax)
    
    # Formatting
    ax.set_title('(a) Model Performance: MSE by Prediction Horizon', 
                 fontweight='bold', pad=15)
    ax.set_xlabel('Prediction Horizon (time steps)', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    
    # Rotate labels
    ax.set_xticklabels([f'H={h}' for h in perf.columns], rotation=0)
    ax.set_yticklabels([clean_model_name(m) for m in perf.index], rotation=0)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_performance_heatmap.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig1_performance_heatmap.png', bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 1: Performance Heatmap")


# ============================================================================
# Figure 2: Ablation Study - PatchFusionBERT Variants
# ============================================================================
def create_ablation_comparison():
    """Bar chart comparing PFB variants"""
    
    pfb_data = df[df['Model'].isin(PFB_MODELS)].copy()
    metrics = pfb_data.groupby('Model').agg({
        'MSE': ['mean', 'std'],
        'MAE': ['mean', 'std']
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE comparison
    ax = axes[0]
    models = metrics.index
    mse_mean = metrics[('MSE', 'mean')].values
    mse_std = metrics[('MSE', 'std')].values
    
    colors = ['#e74c3c', '#3498db']  # v0, v2
    bars = ax.bar(range(len(models)), mse_mean, yerr=mse_std,
                   color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.2)
    
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(['PFB-v0\n(Concat)', 'PFB-v2\n(Concat+Proj)'], rotation=0, ha='center')
    ax.set_ylabel('Mean Squared Error (MSE)', fontweight='bold')
    ax.set_title('(a) MSE Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, mse_mean)):
        ax.text(bar.get_x() + bar.get_width()/2, val + mse_std[i] + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # MAE comparison
    ax = axes[1]
    mae_mean = metrics[('MAE', 'mean')].values
    mae_std = metrics[('MAE', 'std')].values
    
    bars = ax.bar(range(len(models)), mae_mean, yerr=mae_std,
                   color=colors[:len(models)], alpha=0.8, capsize=5, edgecolor='black', linewidth=1.2)
    
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(['PFB-v0\n(Concat)', 'PFB-v2\n(Concat+Proj)'], rotation=0, ha='center')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontweight='bold')
    ax.set_title('(b) MAE Comparison', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, mae_mean)):
        ax.text(bar.get_x() + bar.get_width()/2, val + mae_std[i] + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Fusion Architecture Comparison (PFB-v0 vs PFB-v2)',
                 fontweight='bold', y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_ablation_study.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_ablation_study.png', bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 2: Ablation Study")


# ============================================================================
# Figure 3: Dataset-wise Performance
# ============================================================================
def create_dataset_performance():
    """Performance breakdown by dataset"""
    
    datasets = sorted(df['Dataset'].unique())
    pfb_only = df[df['Model'].isin(PFB_MODELS)].copy()

    n = len(datasets)
    if n == 0:
        print("[WARN] No datasets available for Figure 3 after filtering")
        return

    # Dynamic grid to avoid empty panels (which look like missing plots)
    ncols = 3 if n <= 6 else 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows))
    axes = np.array(axes).flatten()
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        data = pfb_only[pfb_only['Dataset'] == dataset]
        
        # Group by model
        means = data.groupby('Model')['MSE'].mean().reindex(PFB_MODELS)
        stds = data.groupby('Model')['MSE'].std().reindex(PFB_MODELS)
        
        colors = ['#e74c3c', '#3498db']  # Only 2 variants
        bars = ax.bar(range(len(PFB_MODELS)), means.values, yerr=stds.values,
                       color=colors, alpha=0.8, capsize=4, edgecolor='black', linewidth=1)
        
        ax.set_title(f'{dataset}', fontweight='bold', fontsize=11)
        ax.set_xticks(range(len(PFB_MODELS)))
        ax.set_xticklabels(['v0', 'v2'], rotation=0, fontsize=9)
        ax.set_ylabel('MSE', fontsize=9)
        ax.grid(axis='y', alpha=0.2)
        
        # Highlight best
        best_idx = means.argmin()
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
    
    # Remove any unused subplots
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Fusion Variants Performance by Dataset (v0 vs v2)',
                 fontweight='bold', y=0.995, fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_dataset_performance.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_dataset_performance.png', bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 3: Dataset Performance")


# ============================================================================
# Figure 4: Horizon Scaling Analysis
# ============================================================================
def create_horizon_scaling():
    """Line plot showing performance vs horizon"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE scaling
    ax = axes[0]
    for model in PFB_MODELS:
        model_data = df[df['Model'] == model].groupby('Horizon')['MSE'].mean()
        color = {'PatchFusionBERT_v0': '#e74c3c', 
                 'PatchFusionBERT_v2': '#3498db'}[model]
        label = model.replace('PatchFusionBERT_', 'PFB-')
        ax.plot(model_data.index, model_data.values, marker='o', 
                linewidth=2.5, markersize=8, label=label, color=color)
    
    ax.set_xlabel('Prediction Horizon (time steps)', fontweight='bold')
    ax.set_ylabel('Mean Squared Error (MSE)', fontweight='bold')
    ax.set_title('(a) MSE vs Horizon', fontweight='bold')
    ax.legend(frameon=True, shadow=True, loc='best')
    ax.grid(True, alpha=0.3)
    
    # MAE scaling
    ax = axes[1]
    for model in PFB_MODELS:
        model_data = df[df['Model'] == model].groupby('Horizon')['MAE'].mean()
        color = {'PatchFusionBERT_v0': '#e74c3c',
                 'PatchFusionBERT_v2': '#3498db'}[model]
        label = model.replace('PatchFusionBERT_', 'PFB-')
        ax.plot(model_data.index, model_data.values, marker='s',
                linewidth=2.5, markersize=8, label=label, color=color)
    
    ax.set_xlabel('Prediction Horizon (time steps)', fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontweight='bold')
    ax.set_title('(b) MAE vs Horizon', fontweight='bold')
    ax.legend(frameon=True, shadow=True, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Horizon Scaling Analysis (PFB-v0 vs PFB-v2)', fontsize=14, fontweight='bold')
    plt.savefig(OUTPUT_DIR / 'fig4_horizon_scaling.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig4_horizon_scaling.png', bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 4: Horizon Scaling")


# ============================================================================
# Figure 5: Competitive Analysis vs Baselines
# ============================================================================
def create_competitive_analysis():
    """Compare PFB-BERTOnly against all baselines"""
    
    # Average performance across all experiments
    avg_perf = df.groupby('Model').agg({
        'MSE': 'mean',
        'MAE': 'mean'
    }).sort_values('MSE')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MSE comparison
    ax = axes[0]
    models = avg_perf.index
    mse_vals = avg_perf['MSE'].values
    
    # Color coding: variants in red/blue, BERTOnly in green, baselines in gray
    colors = []
    for m in models:
        if m == 'PatchFusionBERT_v0':
            colors.append('#e74c3c')  # Red
        elif m == 'PatchFusionBERT_v2':
            colors.append('#3498db')  # Blue
        elif m == 'PatchFusionBERT_BERTOnly':
            colors.append('#2ecc71')  # Green (separate baseline)
        else:
            colors.append('#95a5a6')  # Gray for other baselines
    
    bars = ax.barh(range(len(models)), mse_vals, color=colors, alpha=0.85, edgecolor='black')
    
    # Highlight best
    best_idx = mse_vals.argmin()
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([clean_model_name(m) for m in models], fontsize=10)
    ax.set_xlabel('Mean Squared Error (MSE)', fontweight='bold')
    ax.set_title('(a) MSE Ranking', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, mse_vals)):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8, fontweight='bold')
    
    # MAE comparison
    ax = axes[1]
    mae_vals = avg_perf['MAE'].values
    
    bars = ax.barh(range(len(models)), mae_vals, color=colors, alpha=0.85, edgecolor='black')
    
    # Highlight best
    best_idx = mae_vals.argmin()
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([clean_model_name(m) for m in models], fontsize=10)
    ax.set_xlabel('Mean Absolute Error (MAE)', fontweight='bold')
    ax.set_title('(b) MAE Ranking', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, mae_vals)):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8, fontweight='bold')
    
    plt.suptitle('Competitive Analysis - All Models',
                 fontweight='bold', y=0.98, fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_competitive_analysis.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig5_competitive_analysis.png', bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 5: Competitive Analysis")


# ============================================================================
# Figure 6: Win/Loss Matrix - PFB vs Baselines
# ============================================================================
def create_winloss_matrix():
    """Heatmap showing where PFB variants win/lose against each baseline"""

    # Compute win rates on mean MSE per (dataset, horizon, model) to avoid
    # artifacts when multiple runs/seeds exist.
    baselines = [m for m in BASELINE_MODELS if m != 'PatchFusionBERT_BERTOnly']
    pfb_variants = ['PatchFusionBERT_v0', 'PatchFusionBERT_v2']

    mean_df = (
        df.groupby(['Dataset', 'Horizon', 'Model'], as_index=False)
        .agg(MSE=('MSE', 'mean'))
    )

    win_data = {variant: [] for variant in pfb_variants}

    for variant in pfb_variants:
        v = mean_df[mean_df['Model'] == variant][['Dataset', 'Horizon', 'MSE']].rename(columns={'MSE': 'm_variant'})
        for baseline in baselines:
            b = mean_df[mean_df['Model'] == baseline][['Dataset', 'Horizon', 'MSE']].rename(columns={'MSE': 'm_base'})
            j = v.merge(b, on=['Dataset', 'Horizon'], how='inner')

            if len(j) == 0:
                win_data[variant].append(0.0)
                continue

            win_rate = (j['m_variant'] < j['m_base']).mean() * 100.0
            win_data[variant].append(float(win_rate))
    
    # Create dataframe for heatmap
    win_df = pd.DataFrame(win_data, index=[clean_model_name(b) for b in baselines])
    win_df.columns = [clean_model_name(v) for v in pfb_variants]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(win_df, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
                cbar_kws={'label': 'Win Rate (%)'},
                linewidths=0.5, linecolor='white',
                vmin=0, vmax=100, ax=ax)
    
    ax.set_title('Win Rate Against Baselines (%)', fontweight='bold', pad=15)
    ax.set_xlabel('PFB Fusion Variant', fontweight='bold')
    ax.set_ylabel('Baseline Model', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_winloss_matrix.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig6_winloss_matrix.png', bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 6: Win/Loss Matrix")


# ============================================================================
# Figure 7: Per-Dataset Ranking (Where PFB Excels and Fails)
# ============================================================================
def create_dataset_rankings():
    """Show ranking of PFB variants on each dataset"""
    
    datasets = sorted(df['Dataset'].unique())
    pfb_variants = ['PatchFusionBERT_v0', 'PatchFusionBERT_v2']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    ranks_v0 = []
    ranks_v2 = []
    
    for dataset in datasets:
        # Calculate average ranking for each variant on this dataset
        dataset_df = df[df['Dataset'] == dataset]
        
        # Rank by MSE (lower is better)
        avg_mse = dataset_df.groupby('Model')['MSE'].mean().sort_values()
        
        rank_v0 = (avg_mse.index.get_loc('PatchFusionBERT_v0') + 1) if 'PatchFusionBERT_v0' in avg_mse.index else len(avg_mse) + 1
        rank_v2 = (avg_mse.index.get_loc('PatchFusionBERT_v2') + 1) if 'PatchFusionBERT_v2' in avg_mse.index else len(avg_mse) + 1
        
        ranks_v0.append(rank_v0)
        ranks_v2.append(rank_v2)
    
    # Plot
    bars1 = ax.bar(x_pos - width/2, ranks_v0, width, label='PFB-v0', color='#e74c3c', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, ranks_v2, width, label='PFB-v2', color='#3498db', alpha=0.8, edgecolor='black')
    
    # Formatting
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Ranking (1=Best)', fontweight='bold')
    ax.set_title('PFB Variant Rankings by Dataset (Lower is Better)', fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    ax.invert_yaxis()  # Lower rank = better = top of chart
    
    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_dataset_rankings.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig7_dataset_rankings.png', bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 7: Dataset Rankings")


# ============================================================================
# Figure 8: Strengths & Weaknesses (Improvement over BERT baseline)
# ============================================================================
def create_improvement_analysis():
    """Show % improvement of fusion variants over BERT-only baseline"""
    
    datasets = sorted(df['Dataset'].unique())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    improvements_v0 = []
    improvements_v2 = []
    
    for dataset in datasets:
        # Get MSE for each model on this dataset
        bert_mse = df[(df['Model'] == 'PatchFusionBERT_BERTOnly') & (df['Dataset'] == dataset)]['MSE'].mean()
        v0_mse = df[(df['Model'] == 'PatchFusionBERT_v0') & (df['Dataset'] == dataset)]['MSE'].mean()
        v2_mse = df[(df['Model'] == 'PatchFusionBERT_v2') & (df['Dataset'] == dataset)]['MSE'].mean()
        
        # Calculate % improvement (negative = worse than BERT)
        imp_v0 = ((bert_mse - v0_mse) / bert_mse * 100) if not np.isnan(bert_mse) and bert_mse > 0 else 0
        imp_v2 = ((bert_mse - v2_mse) / bert_mse * 100) if not np.isnan(bert_mse) and bert_mse > 0 else 0
        
        improvements_v0.append(imp_v0)
        improvements_v2.append(imp_v2)
    
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    # Color based on positive/negative
    colors_v0 = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements_v0]
    colors_v2 = ['#27ae60' if imp > 0 else '#3498db' for imp in improvements_v2]
    
    bars1 = ax.bar(x_pos - width/2, improvements_v0, width, label='PFB-v0', 
                   color=colors_v0, alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, improvements_v2, width, label='PFB-v2',
                   color=colors_v2, alpha=0.8, edgecolor='black')
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    
    # Formatting
    ax.set_xlabel('Dataset', fontweight='bold')
    ax.set_ylabel('Improvement over BERT (%)', fontweight='bold')
    ax.set_title('Fusion Effectiveness - Where PFB Excels and Fails', fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            label_y = height + (1 if height > 0 else -1)
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig8_improvement_analysis.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig8_improvement_analysis.png', bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 8: Improvement Analysis")


# ============================================================================
# Table 1: Main Results Table (LaTeX)
# ============================================================================
def generate_main_results_table():
    """Generate LaTeX table for main results"""
    
    # Calculate per-horizon averages
    table_data = []
    
    for model in ALL_MODELS:
        row = [clean_model_name(model)]
        
        for horizon in VALID_HORIZONS:
            subset = df[(df['Model'] == model) & (df['Horizon'] == horizon)]
            if len(subset) > 0:
                mse = subset['MSE'].mean()
                mae = subset['MAE'].mean()
                row.append(f'{mse:.4f}')
                row.append(f'{mae:.4f}')
            else:
                row.append('--')
                row.append('--')
        
        # Overall average
        overall_mse = df[df['Model'] == model]['MSE'].mean()
        overall_mae = df[df['Model'] == model]['MAE'].mean()
        row.append(f'{overall_mse:.4f}')
        row.append(f'{overall_mae:.4f}')
        
        table_data.append(row)
    
    # Generate LaTeX
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Main Results: Performance across Prediction Horizons (H=96, 192, 336)}\n"
    latex += "\\label{tab:main_results}\n"
    latex += "\\resizebox{\\textwidth}{!}{\n"
    latex += "\\begin{tabular}{l" + "cc" * len(VALID_HORIZONS) + "cc}\n"
    latex += "\\toprule\n"
    latex += "\\multirow{2}{*}{Model} & \\multicolumn{2}{c}{H=96} & \\multicolumn{2}{c}{H=192} & \\multicolumn{2}{c}{H=336} & \\multicolumn{2}{c}{Average} \\\\\n"
    latex += "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}\n"
    latex += " & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE \\\\\n"
    latex += "\\midrule\n"
    
    for row in table_data:
        latex += " & ".join(row) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "}\n"
    latex += "\\end{table}\n"
    
    # Save
    with open(OUTPUT_DIR / 'table1_main_results.tex', 'w') as f:
        f.write(latex)
    
    print("✓ Created Table 1: Main Results (LaTeX)")


# ============================================================================
# Figure 9: Robustness to Missing Inputs (from robustness_missing_data_results.csv)
# ============================================================================
def create_missingness_robustness_figure():
    """Create robustness figure showing MSE vs missing rate."""

    csv_path = Path('robustness_missing_data_results.csv')
    if not csv_path.exists():
        print("! Skipped Figure 9: robustness_missing_data_results.csv not found")
        return

    rdf = pd.read_csv(csv_path)
    required_cols = {'Model', 'Missing_Rate', 'MSE'}
    missing_cols = required_cols - set(rdf.columns)
    if missing_cols:
        print(f"! Skipped Figure 9: missing columns {sorted(missing_cols)}")
        return

    # If multiple settings exist, prefer a deterministic selection
    if 'Dataset' in rdf.columns and rdf['Dataset'].nunique() > 1:
        dataset = sorted(rdf['Dataset'].unique())[0]
        rdf = rdf[rdf['Dataset'] == dataset].copy()
    else:
        dataset = rdf['Dataset'].iloc[0] if 'Dataset' in rdf.columns and len(rdf) else 'Dataset'

    if 'Horizon' in rdf.columns and rdf['Horizon'].nunique() > 1:
        horizon = sorted(rdf['Horizon'].unique())[0]
        rdf = rdf[rdf['Horizon'] == horizon].copy()
    else:
        horizon = rdf['Horizon'].iloc[0] if 'Horizon' in rdf.columns and len(rdf) else ''

    models_present = list(rdf['Model'].unique())
    preferred_order = ['PatchFusionBERT_v0', 'PatchFusionBERT_v2', 'PatchTST', 'DLinear']
    models_order = [m for m in preferred_order if m in models_present] + [
        m for m in models_present if m not in preferred_order
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    palette = {
        'PatchFusionBERT_v0': '#1f77b4',
        'PatchFusionBERT_v2': '#ff7f0e',
        'PatchTST': '#2ca02c',
        'DLinear': '#d62728'
    }

    for model in models_order:
        sub = rdf[rdf['Model'] == model].sort_values('Missing_Rate')
        if sub.empty:
            continue
        x = (sub['Missing_Rate'].astype(float) * 100.0).values
        y = sub['MSE'].astype(float).values
        ax.plot(
            x,
            y,
            marker='o',
            linewidth=2.0,
            markersize=5.5,
            label=clean_model_name(model),
            color=palette.get(model, None),
        )

    ax.set_xlabel('Missing rate (%)', fontweight='bold')
    ax.set_ylabel('Mean Squared Error (MSE)', fontweight='bold')
    title_suffix = f" ({dataset}, H={horizon})" if horizon != '' else f" ({dataset})"
    ax.set_title('(i) Robustness to Missing Inputs' + title_suffix, fontweight='bold', pad=10)
    ax.set_xticks([0, 10, 20, 30])
    ax.legend(frameon=False, ncols=2, loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig9_missingness_robustness.pdf', bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig9_missingness_robustness.png', bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 9: Missingness Robustness")


# ============================================================================
# Run all generations
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENERATING Q1 JOURNAL-QUALITY FIGURES")
    print("="*70 + "\n")
    
    create_performance_heatmap()
    create_ablation_comparison()
    create_dataset_performance()
    create_horizon_scaling()
    create_competitive_analysis()
    create_winloss_matrix()
    create_dataset_rankings()
    create_improvement_analysis()
    create_missingness_robustness_figure()
    generate_main_results_table()
    
    print("\n" + "="*70)
    print("✓ ALL FIGURES GENERATED SUCCESSFULLY")
    print(f"✓ Output directory: {OUTPUT_DIR.absolute()}")
    print("="*70 + "\n")
    
    print("Files created:")
    for file in sorted(OUTPUT_DIR.glob('*')):
        print(f"  - {file.name}")
