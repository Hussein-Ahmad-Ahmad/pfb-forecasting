import pandas as pd, scipy.stats as stats, numpy as np

df5 = pd.read_csv('multiseed_5seed_summary.csv')
models = ['PatchFusionBERT_v0','PatchFusionBERT_v2','PatchTST','DLinear']
datasets = ['ETTm1','ETTm2','ETTh1','ETTh2','Exchange','Weather']

rows = []
for ds in datasets:
    for m in models:
        r = df5[(df5['dataset']==ds)&(df5['model']==m)&(df5['horizon']==192)]
        if len(r):
            mu = float(r['mse_mean'].iloc[0])
            sd = float(r['mse_std'].iloc[0])
            n  = int(r['seed_count'].iloc[0])
            se = sd / np.sqrt(n)
            ci = se * stats.t.ppf(0.975, df=n-1)
            rows.append({'dataset':ds,'model':m,'mse_mean':mu,'mse_std':sd,'n':n,'ci95':ci})

out = pd.DataFrame(rows)
print(out.to_string(index=False, float_format='%.4f'))

# Also print illness multiseed
ill = pd.read_csv('illness_multiseed_stability.csv')
print()
print("=== Illness ===")
print(ill.to_string(index=False))

# Pooled significance
sig = pd.read_csv('multiseed_5seed_significance_pooled.csv')
print()
print("=== Pooled Wilcoxon ===")
print(sig[['comparison','metric','p_value','p_holm','p_fdr_bh','significant_holm','significant_fdr_bh']].to_string(index=False))
