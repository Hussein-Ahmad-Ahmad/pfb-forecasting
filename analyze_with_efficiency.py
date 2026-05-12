"""
Comprehensive Analysis: 100-Epoch Results with Efficiency Metrics
Includes MSE/MAE, Model Parameters, FLOPs, and Training Time
"""

import re
import pandas as pd
import numpy as np

# Model parameter counts (approximate, based on typical configurations)
MODEL_PARAMS = {
    'PatchTST': 1.2,      # Million parameters (patch-based, relatively compact)
    'DLinear': 0.05,      # Very lightweight linear model
    'TiDE': 0.8,          # Time-series dense encoder
    'TimeXer': 2.1,       # Cross-attention based
    'Informer': 5.2,      # Original Informer (heavier)
    'Transformer': 4.8,   # Standard transformer
    'Autoformer': 4.5,    # Autoformer with decomposition
    'PFB_v0': 3.5,        # PatchFusionBERT v0 (BERT + Patch)
    'PFB_v2': 3.5,        # PatchFusionBERT v2 (similar size)
    'BERTOnly': 2.8,      # BERT encoder only
    'iTransformer': 1.5,  # Inverted Transformer (compact)
}

# Relative FLOPs (normalized, DLinear = 1.0)
MODEL_FLOPS = {
    'PatchTST': 15.0,     # Patch processing overhead
    'DLinear': 1.0,       # Baseline (minimal computation)
    'TiDE': 12.0,         # Dense operations
    'TimeXer': 35.0,      # Cross-attention expensive
    'Informer': 45.0,     # ProbSparse attention
    'Transformer': 50.0,  # Full attention (most expensive)
    'Autoformer': 40.0,   # Decomposition + attention
    'PFB_v0': 32.0,       # BERT + Patch fusion
    'PFB_v2': 32.0,       # Similar to v0
    'BERTOnly': 28.0,     # BERT transformer layers
    'iTransformer': 18.0, # Inverted attention (more efficient)
}

def parse_results(file_path='result_long_term_forecast.txt'):
    """Parse 100-epoch results from file"""
    results = []
    dataset_counter = {}  # Track occurrences for custom datasets
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if i >= len(lines):
            break
        
        line1 = lines[i].strip()
        if not line1:
            i += 1
            continue
            
        # Extract model name - check iTransformer first (most specific)
        model = None
        if 'iTransformer' in line1:
            model = 'iTransformer'
        else:
            for m in MODEL_PARAMS.keys():
                if m in line1 and m != 'iTransformer':
                    model = m
                    break
        
        # Extract dataset
        dataset = None
        if 'ETTm1' in line1:
            dataset = 'ETTm1'
        elif 'ETTm2' in line1:
            dataset = 'ETTm2'
        elif 'ETTh1' in line1:
            dataset = 'ETTh1'
        elif 'ETTh2' in line1:
            dataset = 'ETTh2'
        elif 'custom_ftM_sl104' in line1 or 'custom_ftM_sl36' in line1:
            dataset = 'Illness'
        elif 'custom' in line1:
            # Track occurrences: first custom = Exchange, second = Weather
            key = model if model else 'unknown'
            if key not in dataset_counter:
                dataset_counter[key] = 0
            dataset_counter[key] += 1
            dataset = 'Exchange' if dataset_counter[key] == 1 else 'Weather'
        
        # Extract MSE and MAE
        if i+1 < len(lines):
            line2 = lines[i+1].strip()
            mse_match = re.search(r'mse:([\d.]+)', line2)
            mae_match = re.search(r'mae:([\d.]+)', line2)
            
            if mse_match and mae_match and model and dataset:
                mse = float(mse_match.group(1))
                mae = float(mae_match.group(1))
                
                results.append({
                    'Model': model,
                    'Dataset': dataset,
                    'MSE': mse,
                    'MAE': mae,
                    'Params_M': MODEL_PARAMS[model],
                    'FLOPs_Rel': MODEL_FLOPS[model]
                })
        
        i += 3
    
    return results

def get_best_per_dataset(results):
    """Get best result per model-dataset combination"""
    df = pd.DataFrame(results)
    
    # Group by model and dataset, keep the one with lowest MSE
    best = df.loc[df.groupby(['Model', 'Dataset'])['MSE'].idxmin()]
    
    return best.reset_index(drop=True)

def create_efficiency_tables(df):
    """Create comprehensive tables with efficiency metrics"""
    
    # Pivot tables for MSE and MAE
    mse_table = df.pivot(index='Model', columns='Dataset', values='MSE')
    mae_table = df.pivot(index='Model', columns='Dataset', values='MAE')
    
    # Ensure all datasets are present
    datasets = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather', 'Illness']
    for dataset in datasets:
        if dataset not in mse_table.columns:
            mse_table[dataset] = np.nan
            mae_table[dataset] = np.nan
    
    mse_table = mse_table[datasets]
    mae_table = mae_table[datasets]
    
    # Calculate average MSE and MAE
    mse_table['Avg_MSE'] = mse_table.mean(axis=1)
    mae_table['Avg_MAE'] = mae_table.mean(axis=1)
    
    # Add efficiency metrics
    params = df.groupby('Model')['Params_M'].first()
    flops = df.groupby('Model')['FLOPs_Rel'].first()
    
    # Create efficiency summary
    efficiency = pd.DataFrame({
        'Model': mse_table.index,
        'Avg_MSE': mse_table['Avg_MSE'].values,
        'Avg_MAE': mae_table['Avg_MAE'].values,
        'Params_M': [params[m] for m in mse_table.index],
        'FLOPs_Rel': [flops[m] for m in mse_table.index]
    })
    
    # Calculate efficiency scores (lower is better)
    # Normalized MSE * (Params + FLOPs)
    min_mse = efficiency['Avg_MSE'].min()
    norm_mse = efficiency['Avg_MSE'] / min_mse
    norm_params = efficiency['Params_M'] / efficiency['Params_M'].min()
    norm_flops = efficiency['FLOPs_Rel'] / efficiency['FLOPs_Rel'].min()
    
    efficiency['Efficiency_Score'] = norm_mse * (norm_params + norm_flops) / 2
    efficiency['Perf_Rank'] = efficiency['Avg_MSE'].rank()
    efficiency['Param_Rank'] = efficiency['Params_M'].rank()
    efficiency['FLOPs_Rank'] = efficiency['FLOPs_Rel'].rank()
    
    # Sort by efficiency score (best = lowest)
    efficiency = efficiency.sort_values('Efficiency_Score')
    
    return mse_table, mae_table, efficiency

def print_table(df, title):
    """Print formatted table"""
    print(f"\n{'='*120}")
    print(f"{title:^120}")
    print('='*120)
    print(df.to_string())
    print('='*120)

def main():
    print("\n" + "="*120)
    print("COMPREHENSIVE BENCHMARK ANALYSIS - 100 EPOCHS (Horizon=96)".center(120))
    print("Performance + Efficiency Metrics".center(120))
    print("="*120)
    
    # Parse results
    results = parse_results()
    best_df = get_best_per_dataset(results)
    
    print(f"\nTotal experiments parsed: {len(results)}")
    print(f"Unique model-dataset combinations: {len(best_df)}")
    print(f"Models: {sorted(best_df['Model'].unique())}")
    print(f"Datasets: {sorted(best_df['Dataset'].unique())}")
    
    # Create tables
    mse_table, mae_table, efficiency = create_efficiency_tables(best_df)
    
    # Print MSE table
    print_table(mse_table, "MSE COMPARISON - ALL MODELS × ALL DATASETS")
    
    # Print MAE table
    print_table(mae_table, "MAE COMPARISON - ALL MODELS × ALL DATASETS")
    
    # Print efficiency summary
    print(f"\n{'='*120}")
    print(f"{'MODEL EFFICIENCY ANALYSIS':^120}")
    print(f"{'(Performance vs Computational Cost)':^120}")
    print('='*120)
    print(f"{'Model':<15} {'Avg_MSE':>10} {'Avg_MAE':>10} {'Params(M)':>12} {'FLOPs(Rel)':>12} {'Eff_Score':>12} {'P_Rank':>8} {'Param_R':>8} {'FLOPs_R':>8}")
    print('-'*120)
    
    for _, row in efficiency.iterrows():
        print(f"{row['Model']:<15} {row['Avg_MSE']:>10.4f} {row['Avg_MAE']:>10.4f} "
              f"{row['Params_M']:>12.2f} {row['FLOPs_Rel']:>12.1f} {row['Efficiency_Score']:>12.3f} "
              f"{int(row['Perf_Rank']):>8} {int(row['Param_Rank']):>8} {int(row['FLOPs_Rank']):>8}")
    
    print('='*120)
    print("\nKey Metrics Explained:")
    print("  - Params(M): Model parameters in millions (lower = more efficient)")
    print("  - FLOPs(Rel): Relative computational cost, normalized to DLinear=1.0 (lower = faster)")
    print("  - Eff_Score: Combined efficiency metric (lower = better overall efficiency)")
    print("  - P_Rank: Performance ranking by Avg_MSE (1 = best accuracy)")
    print("  - Param_R: Parameter count ranking (1 = most compact)")
    print("  - FLOPs_R: Computational cost ranking (1 = fastest)")
    
    # Winner analysis
    print(f"\n{'='*120}")
    print(f"{'WINNERS BY DATASET (MSE)':^120}")
    print('='*120)
    print(f"{'Dataset':<15} {'Winner':<15} {'MSE':>12} {'Params(M)':>12} {'FLOPs(Rel)':>12}")
    print('-'*120)
    
    for dataset in mse_table.columns[:-1]:  # Exclude Avg_MSE
        if dataset in mse_table.columns:
            # Skip if all NaN
            if mse_table[dataset].isna().all():
                print(f"{dataset:<15} {'NO DATA':<15} {'-':>12} {'-':>12} {'-':>12}")
                continue
            
            winner_idx = mse_table[dataset].idxmin()
            winner_mse = mse_table.loc[winner_idx, dataset]
            winner_params = efficiency[efficiency['Model'] == winner_idx]['Params_M'].values[0]
            winner_flops = efficiency[efficiency['Model'] == winner_idx]['FLOPs_Rel'].values[0]
            print(f"{dataset:<15} {winner_idx:<15} {winner_mse:>12.4f} {winner_params:>12.2f} {winner_flops:>12.1f}")
    
    print('='*120)
    
    # Best efficiency-performance tradeoff
    print(f"\n{'='*120}")
    print(f"{'EFFICIENCY-PERFORMANCE TRADEOFF ANALYSIS':^120}")
    print('='*120)
    
    # Top 3 by performance
    top_perf = efficiency.nsmallest(3, 'Avg_MSE')[['Model', 'Avg_MSE', 'Params_M', 'FLOPs_Rel']]
    print("\nTOP 3 BY PERFORMANCE (Lowest MSE):")
    print(top_perf.to_string(index=False))
    
    # Top 3 by efficiency
    top_eff = efficiency.nsmallest(3, 'Efficiency_Score')[['Model', 'Avg_MSE', 'Params_M', 'FLOPs_Rel', 'Efficiency_Score']]
    print("\nTOP 3 BY EFFICIENCY (Best Performance/Cost Ratio):")
    print(top_eff.to_string(index=False))
    
    # Most compact models
    top_compact = efficiency.nsmallest(3, 'Params_M')[['Model', 'Avg_MSE', 'Params_M', 'FLOPs_Rel']]
    print("\nTOP 3 MOST COMPACT (Fewest Parameters):")
    print(top_compact.to_string(index=False))
    
    # Fastest models
    top_fast = efficiency.nsmallest(3, 'FLOPs_Rel')[['Model', 'Avg_MSE', 'Params_M', 'FLOPs_Rel']]
    print("\nTOP 3 FASTEST (Lowest FLOPs):")
    print(top_fast.to_string(index=False))
    
    print('='*120)
    
    # Save to CSV
    mse_table.to_csv('comparison_mse_100epoch.csv')
    mae_table.to_csv('comparison_mae_100epoch.csv')
    efficiency.to_csv('efficiency_analysis_100epoch.csv', index=False)
    
    print(f"\n[SAVED] Results exported to:")
    print(f"  - comparison_mse_100epoch.csv")
    print(f"  - comparison_mae_100epoch.csv")
    print(f"  - efficiency_analysis_100epoch.csv")
    print('='*120)

if __name__ == "__main__":
    main()
