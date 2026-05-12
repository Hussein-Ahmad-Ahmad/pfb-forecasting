#!/usr/bin/env python3
"""
Generate Phase 3B Final Summary Report
"""
import pandas as pd
import numpy as np

# Load the stats file
stats = pd.read_csv('result_long_term_forecast_phase3c_stats.csv')

print('='*70)
print('PHASE 3B MULTI-SEED VALIDATION - FINAL SUMMARY')
print('='*70)
print(f'\nTotal Model/Dataset/Horizon Combinations: {len(stats)}')
print(f'Models Tested: {stats["Model"].nunique()} ({", ".join(sorted(stats["Model"].unique()))})')
print(f'Datasets: {stats["Dataset"].nunique()}')
print(f'Horizons Tested: {sorted(stats["Horizon"].unique())}')
if len(stats) > 0:
    print(f'Seeds per experiment: {int(stats["N_runs"].iloc[0])}')

print('\n' + '='*70)
print('TOP 10 MOST STABLE MODEL CONFIGURATIONS (Lowest MSE Std)')
print('='*70)
stable = stats.nsmallest(10, 'MSE_std')[['Model', 'Dataset', 'Horizon', 'MSE_mean', 'MSE_std', 'MAE_mean', 'MAE_std']]
print(stable.to_string(index=False))

print('\n' + '='*70)
print('TOP 10 BEST PERFORMING CONFIGURATIONS (Lowest MSE Mean)')
print('='*70)
best = stats.nsmallest(10, 'MSE_mean')[['Model', 'Dataset', 'Horizon', 'MSE_mean', 'MSE_std', 'MAE_mean', 'MAE_std']]
print(best.to_string(index=False))

print('\n' + '='*70)
print('MODEL PERFORMANCE COMPARISON (Average MSE across all datasets/horizons)')
print('='*70)
model_avg = stats.groupby('Model').agg({
    'MSE_mean': 'mean',
    'MSE_std': 'mean',
    'MAE_mean': 'mean',
    'MAE_std': 'mean'
}).round(6).sort_values('MSE_mean')
print(model_avg)

print('\n' + '='*70)
print('DATASET DIFFICULTY RANKING (Average MSE across all models/horizons)')
print('='*70)
dataset_avg = stats.groupby('Dataset').agg({
    'MSE_mean': 'mean',
    'MAE_mean': 'mean'
}).round(6).sort_values('MSE_mean')
print(dataset_avg)

print('\n' + '='*70)
print('FILES GENERATED')
print('='*70)
print('✓ result_long_term_forecast_phase3c_aggregated.csv - All individual results')
print('✓ result_long_term_forecast_phase3c_stats.csv - Mean±Std statistics')
print('✓ result_long_term_forecast_phase3c_best_models.csv - Best model per dataset/horizon')
print('\n' + '='*70)
print('NEXT STEPS: Phase 4 - Ablation Studies and Publication Tables')
print('='*70)
