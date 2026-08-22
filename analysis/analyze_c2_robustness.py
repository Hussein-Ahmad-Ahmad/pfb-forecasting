import pandas as pd

df = pd.read_csv('results_analysis/c2_gaussian_channel_robustness.csv')

clean = df[df['Corruption']=='clean'][['Dataset','Horizon','Model','MSE']].rename(columns={'MSE':'MSE_clean'})
aug = df[df['Corruption']!='clean'].copy()
aug = aug.merge(clean, on=['Dataset','Horizon','Model'])
aug['pct_increase'] = (aug['MSE'] - aug['MSE_clean']) / aug['MSE_clean'] * 100

print('=== GAUSSIAN NOISE (sigma=0.5) - % MSE increase ===')
g = aug[(aug['Corruption']=='gaussian') & (aug['Rate']==0.5)]
pivot = g.pivot_table(index=['Dataset','Horizon'], columns='Model', values='pct_increase').round(1)
print(pivot.to_string())

print()
print('=== CHANNEL DROPOUT (rate=0.1) - % MSE increase ===')
c1 = aug[(aug['Corruption']=='channel_dropout') & (aug['Rate']==0.1)]
pivot1 = c1.pivot_table(index=['Dataset','Horizon'], columns='Model', values='pct_increase').round(1)
print(pivot1.to_string())

print()
print('=== CHANNEL DROPOUT (rate=0.3) - % MSE increase ===')
c3 = aug[(aug['Corruption']=='channel_dropout') & (aug['Rate']==0.3)]
pivot3 = c3.pivot_table(index=['Dataset','Horizon'], columns='Model', values='pct_increase').round(1)
print(pivot3.to_string())

print()
print('=== CLEAN BASELINE MSE ===')
bl = df[df['Corruption']=='clean'][['Dataset','Horizon','Model','MSE']].pivot_table(index=['Dataset','Horizon'], columns='Model', values='MSE').round(4)
print(bl.to_string())
