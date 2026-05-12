import pandas as pd
df = pd.read_csv('results_analysis/error_by_step.csv')

for ds in ['Weather', 'ETTm2']:
    sub = df[df['dataset'] == ds]
    horizon = sub['horizon'].iloc[0]
    max_step = sub['step'].max()
    early = sub[sub['step'] <= 24].groupby('model')['per_step_mse'].mean()
    late = sub[sub['step'] > max_step - 24].groupby('model')['per_step_mse'].mean()
    print(f'\n=== {ds} H={horizon} ===')
    print('Early steps (1-24) mean MSE:')
    print(early.round(4).to_string())
    print('Late steps mean MSE:')
    print(late.round(4).to_string())
    for m in ['PatchTST', 'DLinear']:
        if m in late.index and 'PatchFusionBERT_v0' in late.index:
            if m in early.index:
                adv_early = (early[m] - early['PatchFusionBERT_v0']) / early[m] * 100
                adv_late = (late[m] - late['PatchFusionBERT_v0']) / late[m] * 100
                print(f'PFBv0 vs {m}: early advantage={adv_early:.1f}%, late advantage={adv_late:.1f}%')
