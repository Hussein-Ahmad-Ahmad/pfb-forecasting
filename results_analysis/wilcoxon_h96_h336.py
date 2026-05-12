"""
Wilcoxon signed-rank significance testing for H=96 and H=336 5-seed data.
Pooled across 6 datasets per horizon (N=30 pairs each).
Holm-Bonferroni + BH-FDR correction applied to 8 tests per horizon.
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

df = pd.read_csv('multiseed_h96_h336_raw.csv')

comparisons = [
    ('PatchFusionBERT_v0', 'PatchTST'),
    ('PatchFusionBERT_v0', 'PatchFusionBERT_v2'),
    ('PatchFusionBERT_v0', 'DLinear'),
    ('PatchFusionBERT_v2', 'PatchTST'),
]
metrics = ['mse', 'mae']
datasets = sorted(df['dataset'].unique())  # 6 datasets

all_results = []

for horizon in [96, 336]:
    dfh = df[df['horizon'] == horizon]
    # Build paired data: for each (dataset, seed) pair, one MSE/MAE per model
    pivot = dfh.pivot_table(index=['dataset', 'seed'], columns='model', values=['mse', 'mae'])
    pivot.columns = ['_'.join(c).strip() for c in pivot.columns]

    tests_per_horizon = []
    for metric in metrics:
        for (m_a, m_b) in comparisons:
            col_a = f'{metric}_{m_a}'
            col_b = f'{metric}_{m_b}'
            if col_a not in pivot.columns or col_b not in pivot.columns:
                continue
            pairs = pivot[[col_a, col_b]].dropna()
            diffs = pairs[col_a] - pairs[col_b]
            n = len(diffs)
            nonzero = diffs[diffs != 0]
            if len(nonzero) == 0:
                stat, pval = np.nan, 1.0
            else:
                stat, pval = stats.wilcoxon(nonzero, alternative='two-sided')
            mean_diff = float(diffs.mean())
            median_diff = float(diffs.median())
            # rank-biserial: r = 1 - 2*W / (n*(n+1)/2)  where n = len(nonzero)
            nn = len(nonzero)
            rb = float(1 - 2 * stat / (nn * (nn + 1) / 2)) if nn > 0 else np.nan
            winner = m_a if mean_diff < 0 else m_b
            tests_per_horizon.append({
                'horizon': horizon,
                'comparison': f'{m_a} vs {m_b}',
                'metric': metric.upper(),
                'n_pairs': n,
                'n_nonzero': nn,
                'statistic': stat,
                'p_value': pval,
                'mean_diff': mean_diff,
                'rank_biserial': rb,
                'winner': winner,
            })

    # Apply multiple-testing correction per horizon (8 tests)
    pvals = [t['p_value'] for t in tests_per_horizon]
    reject_holm, p_holm, _, _ = multipletests(pvals, method='holm')
    reject_bh,   p_bh,   _, _ = multipletests(pvals, method='fdr_bh')

    for i, t in enumerate(tests_per_horizon):
        t['p_holm'] = p_holm[i]
        t['p_fdr_bh'] = p_bh[i]
        t['significant_holm'] = 'Yes' if reject_holm[i] else 'No'
        t['significant_fdr_bh'] = 'Yes' if reject_bh[i] else 'No'
        all_results.append(t)

out = pd.DataFrame(all_results)
out.to_csv('multiseed_h96_h336_significance.csv', index=False)
print(out[['horizon','comparison','metric','n_pairs','p_value','p_holm','p_fdr_bh',
           'significant_holm','significant_fdr_bh','mean_diff']].to_string(index=False, float_format='%.6f'))
