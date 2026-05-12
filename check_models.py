import pandas as pd

df = pd.read_csv('results_analysis/result_long_term_forecast_phase3c_stats.csv')
print('Models in Phase 3C results:')
print(df['Model'].unique())
print(f'\nTotal models: {df["Model"].nunique()}')
