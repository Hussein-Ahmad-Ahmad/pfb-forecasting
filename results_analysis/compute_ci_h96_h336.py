"""Compute 95% CI for H=96 and H=336 MSE results (N=5 seeds, t-distribution df=4)."""
import pandas as pd
import scipy.stats as stats
import numpy as np

df = pd.read_csv('multiseed_h96_h336_summary.csv')

models = ['PFB-Direct', 'PFB-Projected', 'PatchTST', 'DLinear']
model_short = {
    'PFB-Direct': 'PFB-Direct',
    'PFB-Projected': 'PFB-Projected',
    'PatchTST': 'PatchTST',
    'DLinear': 'DLinear',
}
datasets = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather']

t_val = stats.t.ppf(0.975, df=4)  # 2.7764

for horizon in [96, 336]:
    print(f"\n=== H={horizon} MSE 95% CI (N=5, t={t_val:.4f}) ===")
    print(f"| Dataset | {' | '.join(model_short[m] + ' MSE [±95%CI]' for m in models)} |")
    print(f"|---------|{'|'.join(['------------------' for _ in models])}|")
    for ds in datasets:
        row_parts = []
        for m in models:
            r = df[(df['dataset'] == ds) & (df['model'] == m) & (df['horizon'] == horizon)]
            if len(r):
                mu = float(r['mse_mean'].iloc[0])
                sd = float(r['mse_std'].iloc[0])
                n = int(r['seeds'].iloc[0])
                se = sd / np.sqrt(n)
                ci = se * t_val
                row_parts.append(f'{mu:.4f} [±{ci:.4f}]')
            else:
                row_parts.append('—')
        print(f"| {ds} | {' | '.join(row_parts)} |")

    print(f"\nInterval observations for H={horizon}:")
    # Find best model per dataset and check for non-overlap
    for ds in datasets:
        vals = {}
        for m in models:
            r = df[(df['dataset'] == ds) & (df['model'] == m) & (df['horizon'] == horizon)]
            if len(r):
                mu = float(r['mse_mean'].iloc[0])
                sd = float(r['mse_std'].iloc[0])
                n = int(r['seeds'].iloc[0])
                se = sd / np.sqrt(n)
                ci = se * t_val
                vals[m] = (mu, ci)
        if vals:
            best_m = min(vals, key=lambda x: vals[x][0])
            best_mu, best_ci = vals[best_m]
            best_lo, best_hi = best_mu - best_ci, best_mu + best_ci
            overlaps = []
            non_overlaps = []
            for m in models:
                if m == best_m:
                    continue
                mu, ci = vals[m]
                lo, hi = mu - ci, mu + ci
                if lo > best_hi or hi < best_lo:
                    non_overlaps.append(model_short[m])
                else:
                    overlaps.append(model_short[m])
            best_short = model_short[best_m]
            if non_overlaps:
                print(f"  {ds}: {best_short} [{best_lo:.4f},{best_hi:.4f}] non-overlapping with: {', '.join(non_overlaps)}")
            else:
                print(f"  {ds}: {best_short} [{best_lo:.4f},{best_hi:.4f}] — CIs overlap with all others")
