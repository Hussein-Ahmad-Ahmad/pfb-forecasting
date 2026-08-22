"""
Statistical Analysis of Multi-Seed Validation Results
Extracts MS2021, MS2022, MS2023 results and computes mean±std for publication

Author: Generated for Dynamic Ensemble Paper
Date: January 5, 2026
"""

import re
import numpy as np
from scipy import stats
import pandas as pd

def parse_results_file(filename='result_long_term_forecast.txt'):
    """Parse the result file and extract multi-seed experiments"""
    results = {}
    parsed_count = 0
    skipped_count = 0
    models_found = set()
    datasets_found = set()
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse each result entry (skip blank lines, then read name+metrics)
    i = 0
    while i < len(lines):
        # Skip blank lines
        while i < len(lines) and not lines[i].strip():
            i += 1
        
        if i >= len(lines):
            break
        
        name_line = lines[i].strip()
        i += 1
        
        if i >= len(lines):
            break
            
        metrics_line = lines[i].strip()
        i += 1
        
        # Only process MS2021, MS2022, MS2023 entries
        if not any(seed in name_line for seed in ['MS2021', 'MS2022', 'MS2023']):
            continue
        
        # Extract components using more flexible pattern
        # Format: long_term_forecast_MODEL_MSSEED_DATASET_192...
        parts = name_line.split('_')
        
        # Find MS seed
        seed = None
        seed_idx = None
        for idx, part in enumerate(parts):
            if part.startswith('MS202'):
                seed = part
                seed_idx = idx
                break
        
        if not seed or seed_idx is None:
            skipped_count += 1
            continue
        
        # Model is everything before MS seed (after long_term_forecast)
        model = '_'.join(parts[2:seed_idx])  # Skip 'long', 'term', 'forecast'
        
        # Remove "forecast_" prefix if present
        if model.startswith('forecast_'):
            model = model[9:]  # Remove "forecast_"
        
        # Dataset is right after MS seed
        dataset = parts[seed_idx + 1] if seed_idx + 1 < len(parts) else None
        
        if not model or not dataset:
            skipped_count += 1
            continue
        
        models_found.add(model)
        datasets_found.add(dataset)
        
        # Extract MSE and MAE
        mse_match = re.search(r'mse:([\d\.]+)', metrics_line)
        mae_match = re.search(r'mae:([\d\.]+)', metrics_line)
        
        if not mse_match or not mae_match:
            skipped_count += 1
            continue
        
        mse = float(mse_match.group(1))
        mae = float(mae_match.group(1))
        
        # Store results
        key = (model, dataset)
        if key not in results:
            results[key] = {'MS2021': {}, 'MS2022': {}, 'MS2023': {}}
        
        results[key][seed] = {'mse': mse, 'mae': mae}
        parsed_count += 1
    
    print(f"Parsed {parsed_count} multi-seed entries (skipped {skipped_count})")
    print(f"Models found: {sorted(models_found)}")
    print(f"Datasets found: {sorted(datasets_found)}")
    return results

def compute_statistics(results):
    """Compute mean and std across seeds"""
    stats_table = []
    
    for (model, dataset), seed_results in sorted(results.items()):
        # Check if all 3 seeds are present
        if len(seed_results) != 3:
            print(f"Warning: {model} + {dataset} has only {len(seed_results)} seeds")
            continue
        
        # Extract MSE and MAE values
        mse_values = [seed_results[seed]['mse'] for seed in ['MS2021', 'MS2022', 'MS2023']]
        mae_values = [seed_results[seed]['mae'] for seed in ['MS2021', 'MS2022', 'MS2023']]
        
        # Compute statistics
        mse_mean = np.mean(mse_values)
        mse_std = np.std(mse_values, ddof=1)  # Sample std
        mae_mean = np.mean(mae_values)
        mae_std = np.std(mae_values, ddof=1)
        
        stats_table.append({
            'Model': model,
            'Dataset': dataset,
            'MSE_mean': mse_mean,
            'MSE_std': mse_std,
            'MAE_mean': mae_mean,
            'MAE_std': mae_std,
            'MSE_raw': mse_values,
            'MAE_raw': mae_values
        })
    
    return pd.DataFrame(stats_table)

def wilcoxon_tests(df):
    """Perform Wilcoxon signed-rank tests: PFB-Direct vs baselines"""
    datasets = df['Dataset'].unique()
    
    print("\n" + "="*80)
    print("WILCOXON SIGNED-RANK TESTS (PFB-Direct vs Baselines)")
    print("="*80)
    
    # PFB-Direct vs PatchTST
    print("\n### PFB-Direct vs PatchTST ###")
    pfb_mse = []
    patchtst_mse = []
    
    for dataset in datasets:
        pfb_row = df[(df['Model'] == 'PFB-Direct') & (df['Dataset'] == dataset)]
        tst_row = df[(df['Model'] == 'PatchTST') & (df['Dataset'] == dataset)]
        
        if not pfb_row.empty and not tst_row.empty:
            pfb_mse.extend(pfb_row['MSE_raw'].values[0])
            patchtst_mse.extend(tst_row['MSE_raw'].values[0])
    
    if pfb_mse and patchtst_mse:
        stat, p_value = stats.wilcoxon(pfb_mse, patchtst_mse)
        print(f"  Statistic: {stat:.4f}, p-value: {p_value:.6f}")
        if p_value < 0.05:
            winner = "PFB-Direct" if np.mean(pfb_mse) < np.mean(patchtst_mse) else "PatchTST"
            print(f"  [+] Significant difference (p<0.05): {winner} is better")
        else:
            print(f"  [-] No significant difference (p>=0.05)")
    
    # PFB-Direct vs DLinear
    print("\n### PFB-Direct vs DLinear ###")
    pfb_mse = []
    dlinear_mse = []
    
    for dataset in datasets:
        pfb_row = df[(df['Model'] == 'PFB-Direct') & (df['Dataset'] == dataset)]
        dlin_row = df[(df['Model'] == 'DLinear') & (df['Dataset'] == dataset)]
        
        if not pfb_row.empty and not dlin_row.empty:
            pfb_mse.extend(pfb_row['MSE_raw'].values[0])
            dlinear_mse.extend(dlin_row['MSE_raw'].values[0])
    
    if pfb_mse and dlinear_mse:
        stat, p_value = stats.wilcoxon(pfb_mse, dlinear_mse)
        print(f"  Statistic: {stat:.4f}, p-value: {p_value:.6f}")
        if p_value < 0.05:
            winner = "PFB-Direct" if np.mean(pfb_mse) < np.mean(dlinear_mse) else "DLinear"
            print(f"  [+] Significant difference (p<0.05): {winner} is better")
        else:
            print(f"  [-] No significant difference (p>=0.05)")

def print_publication_table(df):
    """Generate publication-ready LaTeX table"""
    print("\n" + "="*80)
    print("PUBLICATION RESULTS TABLE (Mean ± Std)")
    print("="*80)
    
    datasets = sorted(df['Dataset'].unique())
    models = ['PFB-Direct', 'PatchTST', 'DLinear']
    
    print("\nMSE Results:")
    print("-" * 80)
    print(f"{'Dataset':<12}", end='')
    for model in models:
        print(f"{model:>20}", end='')
    print()
    print("-" * 80)
    
    for dataset in datasets:
        print(f"{dataset:<12}", end='')
        for model in models:
            row = df[(df['Model'] == model) & (df['Dataset'] == dataset)]
            if not row.empty:
                mean = row['MSE_mean'].values[0]
                std = row['MSE_std'].values[0]
                print(f"{mean:>15.4f}±{std:<4.4f}", end='')
            else:
                print(f"{'N/A':>20}", end='')
        print()
    
    print("\n" + "="*80)
    print("\nMAE Results:")
    print("-" * 80)
    print(f"{'Dataset':<12}", end='')
    for model in models:
        print(f"{model:>20}", end='')
    print()
    print("-" * 80)
    
    for dataset in datasets:
        print(f"{dataset:<12}", end='')
        for model in models:
            row = df[(df['Model'] == model) & (df['Dataset'] == dataset)]
            if not row.empty:
                mean = row['MAE_mean'].values[0]
                std = row['MAE_std'].values[0]
                print(f"{mean:>15.4f}±{std:<4.4f}", end='')
            else:
                print(f"{'N/A':>20}", end='')
        print()

def main():
    print("="*80)
    print("MULTI-SEED STATISTICAL ANALYSIS")
    print("="*80)
    print(f"\nParsing results file...")
    
    # Parse results
    results = parse_results_file()
    print(f"Found {len(results)} unique model-dataset combinations")
    
    # Compute statistics
    print("\nComputing statistics across seeds...")
    df = compute_statistics(results)
    
    print(f"\nValid combinations (all 3 seeds present): {len(df)}")
    print(f"Expected: 18 (3 models × 6 datasets)")
    
    # Print publication table
    print_publication_table(df)
    
    # Perform Wilcoxon tests
    wilcoxon_tests(df)
    
    # Save to CSV
    output_file = 'multiseed_statistics.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
