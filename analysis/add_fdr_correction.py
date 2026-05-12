"""
Add Holm-Bonferroni and Benjamini-Hochberg FDR correction to Wilcoxon significance files.
Applies correction within each file (pooled: 8 tests; by_config: 72 tests).
"""
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

def apply_corrections(df, p_col='p_value'):
    """Add p_holm and p_fdr columns using Holm and BH methods."""
    pvals = df[p_col].astype(float).values

    # Holm-Bonferroni (step-down, controls FWER)
    reject_holm, p_holm, _, _ = multipletests(pvals, alpha=0.05, method='holm')
    # Benjamini-Hochberg (controls FDR)
    reject_bh, p_bh, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

    df = df.copy()
    df['p_holm'] = p_holm.round(6)
    df['p_fdr_bh'] = p_bh.round(6)
    df['significant_holm'] = ['Yes' if r else 'No' for r in reject_holm]
    df['significant_fdr_bh'] = ['Yes' if r else 'No' for r in reject_bh]
    return df


# --- Pooled (8 tests) ---
pooled = pd.read_csv('results_analysis/multiseed_5seed_significance_pooled.csv')
pooled_corrected = apply_corrections(pooled)
pooled_corrected.to_csv('results_analysis/multiseed_5seed_significance_pooled.csv', index=False)
print("=== POOLED (8 tests) ===")
print(pooled_corrected[['comparison', 'metric', 'p_value', 'p_holm', 'p_fdr_bh',
                          'significant', 'significant_holm', 'significant_fdr_bh']].to_string(index=False))

# --- By-config (72 tests) ---
by_config = pd.read_csv('results_analysis/multiseed_5seed_significance_by_config.csv')
by_config_corrected = apply_corrections(by_config)
by_config_corrected.to_csv('results_analysis/multiseed_5seed_significance_by_config.csv', index=False)

sig_holm = (by_config_corrected['significant_holm'] == 'Yes').sum()
sig_fdr = (by_config_corrected['significant_fdr_bh'] == 'Yes').sum()
sig_uncorrected = (by_config_corrected['significant'] == 'Yes').sum()
print(f"\n=== BY-CONFIG (72 tests) ===")
print(f"Significant uncorrected (p<0.05): {sig_uncorrected}/72")
print(f"Significant after Holm:           {sig_holm}/72")
print(f"Significant after BH-FDR:         {sig_fdr}/72")
print("\nSurviving Holm correction:")
survivors = by_config_corrected[by_config_corrected['significant_holm'] == 'Yes']
if len(survivors):
    print(survivors[['comparison', 'metric', 'p_value', 'p_holm', 'p_fdr_bh']].to_string(index=False))
else:
    print("  None — all per-config tests lose significance after Holm correction.")

print("\n=== Files updated ===")
print("  results_analysis/multiseed_5seed_significance_pooled.csv")
print("  results_analysis/multiseed_5seed_significance_by_config.csv")
