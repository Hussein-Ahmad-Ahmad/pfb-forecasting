"""
Create Patch Sensitivity Plots from Results File
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

# Manual data from results
patch_sensitivity_data = {
    'ETTm1': {
        'configs': [(8, 4), (16, 8), (24, 12), (32, 16)],
        'mse': [0.2972, 0.3037, 0.2957, 0.2947],
        'mae': [0.3491, 0.3479, 0.3432, 0.3433]
    },
    'ETTh1': {
        'configs': [(8, 4), (16, 8), (24, 12), (32, 16)],
        'mse': [0.3790, 0.3675, 0.3655, 0.3729],
        'mae': [0.4045, 0.3952, 0.3953, 0.3990]
    }
}

for dataset in ['ETTm1', 'ETTh1']:
    data = patch_sensitivity_data[dataset]
    labels = [f'P={p}, S={s}' for p, s in data['configs']]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Patch Sensitivity Analysis: {dataset}', fontsize=16, fontweight='bold', y=1.02)
    
    x = np.arange(len(labels))
    width = 0.6
    
    # MSE sensitivity
    ax = axes[0]
    mse_values = data['mse']
    bars = ax.bar(x, mse_values, width, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Patch Configuration', fontsize=13, fontweight='bold')
    ax.set_ylabel('MSE', fontsize=13, fontweight='bold')
    ax.set_title('MSE vs Patch Size', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Highlight best
    best_idx = np.argmin(mse_values)
    bars[best_idx].set_color('#2ecc71')
    bars[best_idx].set_alpha(1.0)
    
    for i, (bar, val) in enumerate(zip(bars, mse_values)):
        height = bar.get_height()
        label = f'{val:.4f}'
        if i == best_idx:
            label += ' ★'
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.003,
               label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # MAE sensitivity
    ax = axes[1]
    mae_values = data['mae']
    bars = ax.bar(x, mae_values, width, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Patch Configuration', fontsize=13, fontweight='bold')
    ax.set_ylabel('MAE', fontsize=13, fontweight='bold')
    ax.set_title('MAE vs Patch Size', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Highlight best
    best_idx = np.argmin(mae_values)
    bars[best_idx].set_color('#2ecc71')
    bars[best_idx].set_alpha(1.0)
    
    for i, (bar, val) in enumerate(zip(bars, mae_values)):
        height = bar.get_height()
        label = f'{val:.4f}'
        if i == best_idx:
            label += ' ★'
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.003,
               label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path = f'./analysis_results/plots/patch_sensitivity_{dataset}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

# Create sensitivity table
import pandas as pd

rows = []
for dataset in ['ETTm1', 'ETTh1']:
    data = patch_sensitivity_data[dataset]
    for i, (patch, stride) in enumerate(data['configs']):
        rows.append({
            'Dataset': dataset,
            'Patch Length': patch,
            'Stride': stride,
            'MSE': data['mse'][i],
            'MAE': data['mae'][i]
        })

df = pd.DataFrame(rows)

csv_path = './analysis_results/tables/patch_sensitivity.csv'
df.to_csv(csv_path, index=False, float_format='%.4f')
print(f"✓ Saved: {csv_path}")

latex_path = './analysis_results/tables/patch_sensitivity.tex'
with open(latex_path, 'w') as f:
    f.write(df.to_latex(index=False, float_format='%.4f', escape=False))
print(f"✓ Saved: {latex_path}")

print("\nPatch sensitivity analysis complete!")
