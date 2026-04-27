import pandas as pd
import numpy as np
from pathlib import Path
import re

def parse_all_results(filepath):
    """Parse all completed experiments from result file"""
    
    results = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('long_term_forecast_'):
            exp_name = line
            
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                
                mse_match = re.search(r'mse:([\d.]+)', next_line)
                mae_match = re.search(r'mae:([\d.]+)', next_line)
                
                if mse_match and mae_match:
                    mse = float(mse_match.group(1))
                    mae = float(mae_match.group(1))
                    
                    # Identify model and dataset
                    model_name = None
                    dataset_name = None
                    training_type = "1-epoch"
                    
                    # Identify training type
                    if 'baseline_100' in exp_name or 'PFB_100' in exp_name or 'Ablation_100' in exp_name:
                        training_type = "100-epoch"
                    elif 'baseline_0' in exp_name or 'Full_Training' in exp_name or 'Ablation' in exp_name or 'PatchSens' in exp_name:
                        training_type = "20-epoch"
                    
                    # Identify model
                    if 'PatchFusionBERT_v0' in exp_name or 'PFB_v0' in exp_name:
                        model_name = 'PFB_v0'
                    elif 'PatchFusionBERT_v2' in exp_name or 'PFB_v2' in exp_name:
                        model_name = 'PFB_v2'
                    elif 'BERTOnly' in exp_name:
                        model_name = 'BERTOnly'
                    elif 'PatchTST' in exp_name:
                        model_name = 'PatchTST'
                    elif 'DLinear' in exp_name:
                        model_name = 'DLinear'
                    elif 'TiDE' in exp_name:
                        model_name = 'TiDE'
                    elif 'TimeXer' in exp_name:
                        model_name = 'TimeXer'
                    elif 'iTransformer' in exp_name:
                        model_name = 'iTransformer'
                    elif 'Informer' in exp_name:
                        model_name = 'Informer'
                    elif 'Transformer' in exp_name:
                        model_name = 'Transformer'
                    elif 'Autoformer' in exp_name:
                        model_name = 'Autoformer'
                    
                    # Identify dataset
                    if 'ETTm1' in exp_name:
                        dataset_name = 'ETTm1'
                    elif 'ETTm2' in exp_name:
                        dataset_name = 'ETTm2'
                    elif 'ETTh1' in exp_name:
                        dataset_name = 'ETTh1'
                    elif 'ETTh2' in exp_name:
                        dataset_name = 'ETTh2'
                    elif 'Exchange' in exp_name:
                        dataset_name = 'Exchange'
                    elif 'Weather' in exp_name:
                        dataset_name = 'Weather'
                    elif 'Illness' in exp_name:
                        dataset_name = 'Illness'
                    
                    if model_name and dataset_name:
                        results.append({
                            'Model': model_name,
                            'Dataset': dataset_name,
                            'MSE': mse,
                            'MAE': mae,
                            'Training': training_type,
                            'Experiment': exp_name
                        })
            
            i += 2
        else:
            i += 1
    
    return pd.DataFrame(results)

def get_best_results_per_model(df):
    """Get best result for each model-dataset combination"""
    
    # For each model-dataset, keep the best MSE result
    best_results = df.loc[df.groupby(['Model', 'Dataset'])['MSE'].idxmin()]
    return best_results.copy()

def create_comparison_tables(df):
    """Create comprehensive comparison tables"""
    
    # MSE pivot
    mse_pivot = df.pivot_table(
        values='MSE',
        index='Dataset',
        columns='Model',
        aggfunc='first'
    )
    
    # MAE pivot
    mae_pivot = df.pivot_table(
        values='MAE',
        index='Dataset',
        columns='Model',
        aggfunc='first'
    )
    
    # Ensure dataset order
    dataset_order = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather', 'Illness']
    mse_pivot = mse_pivot.reindex([d for d in dataset_order if d in mse_pivot.index])
    mae_pivot = mae_pivot.reindex([d for d in dataset_order if d in mae_pivot.index])
    
    return mse_pivot, mae_pivot

def create_rankings(mse_pivot):
    """Create detailed rankings for each dataset"""
    
    rankings = []
    
    for dataset in mse_pivot.index:
        row = mse_pivot.loc[dataset].dropna().sort_values()
        
        for rank, (model, mse_val) in enumerate(row.items(), 1):
            best_mse = row.values[0]
            diff_pct = ((mse_val - best_mse) / best_mse) * 100 if best_mse > 0 else 0
            
            rankings.append({
                'Dataset': dataset,
                'Rank': rank,
                'Model': model,
                'MSE': mse_val,
                'Gap_to_Best_%': diff_pct
            })
    
    return pd.DataFrame(rankings)

def overall_rankings(mse_pivot):
    """Calculate overall model rankings"""
    
    stats = []
    for model in mse_pivot.columns:
        values = mse_pivot[model].dropna()
        if len(values) > 0:
            stats.append({
                'Model': model,
                'Avg_MSE': values.mean(),
                'Std_MSE': values.std(),
                'Min_MSE': values.min(),
                'Max_MSE': values.max(),
                'Datasets': len(values)
            })
    
    stats_df = pd.DataFrame(stats).sort_values('Avg_MSE')
    stats_df['Overall_Rank'] = range(1, len(stats_df) + 1)
    
    return stats_df

def print_table(df, title):
    """Print formatted table"""
    print(f"\n{'=' * 120}")
    print(f"{title:^120}")
    print('=' * 120)
    print(df.to_string())
    print('=' * 120)

# Main execution
if __name__ == '__main__':
    
    print("\n" + "=" * 120)
    print("COMPREHENSIVE RESULTS ANALYSIS - ALL EXPERIMENTS".center(120))
    print("=" * 120)
    
    # Parse all results
    result_file = Path('result_long_term_forecast.txt')
    all_results = parse_all_results(result_file)
    
    print(f"\nTotal experiments parsed: {len(all_results)}")
    print(f"Training types: {all_results['Training'].value_counts().to_dict()}")
    print(f"Models found: {sorted(all_results['Model'].unique())}")
    print(f"Datasets: {sorted(all_results['Dataset'].unique())}")
    
    # Get best results per model-dataset
    best_results = get_best_results_per_model(all_results)
    
    print(f"\nBest results (per model-dataset): {len(best_results)}")
    
    # Create pivot tables
    mse_pivot, mae_pivot = create_comparison_tables(best_results)
    
    # ========================================================================
    # TABLE 1: MSE Comparison
    # ========================================================================
    print_table(mse_pivot, "MSE COMPARISON - BEST RESULTS PER MODEL")
    
    # ========================================================================
    # TABLE 2: MAE Comparison
    # ========================================================================
    print_table(mae_pivot, "MAE COMPARISON - BEST RESULTS PER MODEL")
    
    # ========================================================================
    # TABLE 3: Overall Rankings
    # ========================================================================
    overall = overall_rankings(mse_pivot)
    print_table(overall, "OVERALL MODEL RANKINGS (by Average MSE)")
    
    # ========================================================================
    # TABLE 4: Detailed Rankings Per Dataset
    # ========================================================================
    rankings = create_rankings(mse_pivot)
    print_table(rankings, "DETAILED RANKINGS PER DATASET")
    
    # ========================================================================
    # WINNERS SUMMARY
    # ========================================================================
    print(f"\n{'=' * 120}")
    print("WINNERS SUMMARY".center(120))
    print('=' * 120)
    
    for dataset in mse_pivot.index:
        row = mse_pivot.loc[dataset].dropna().sort_values()
        if len(row) > 0:
            winner = row.index[0]
            winner_mse = row.values[0]
            
            if len(row) > 1:
                second = row.index[1]
                second_mse = row.values[1]
                gap = ((second_mse - winner_mse) / winner_mse) * 100
                print(f"{dataset:12s}: [WINNER] {winner:15s} MSE={winner_mse:.6f}  |  2nd: {second:15s} MSE={second_mse:.6f} (+{gap:.2f}%)")
            else:
                print(f"{dataset:12s}: [WINNER] {winner:15s} MSE={winner_mse:.6f}")
    
    # Count wins
    print(f"\n{'=' * 120}")
    print("WIN COUNT".center(120))
    print('=' * 120)
    
    win_count = {}
    for dataset in mse_pivot.index:
        row = mse_pivot.loc[dataset].dropna().sort_values()
        if len(row) > 0:
            winner = row.index[0]
            win_count[winner] = win_count.get(winner, 0) + 1
    
    for model, wins in sorted(win_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model:20s}: {wins} wins")
    
    # ========================================================================
    # COMPETITIVE ANALYSIS (within 5% of best)
    # ========================================================================
    print(f"\n{'=' * 120}")
    print("COMPETITIVE MODELS (within 5% of best per dataset)".center(120))
    print('=' * 120)
    
    for dataset in mse_pivot.index:
        row = mse_pivot.loc[dataset].dropna().sort_values()
        best_mse = row.values[0]
        competitive = row[row <= best_mse * 1.05]
        
        print(f"\n{dataset}:")
        for model, mse_val in competitive.items():
            diff_pct = ((mse_val - best_mse) / best_mse) * 100
            marker = "[BEST]" if model == row.index[0] else "[COMP]"
            print(f"  {marker} {model:15s}: {mse_val:.6f} (+{diff_pct:.2f}%)")
    
    # ========================================================================
    # TRAINING TYPE BREAKDOWN
    # ========================================================================
    print(f"\n{'=' * 120}")
    print("TRAINING TYPE DISTRIBUTION".center(120))
    print('=' * 120)
    
    training_summary = all_results.groupby(['Model', 'Training']).size().unstack(fill_value=0)
    print(training_summary.to_string())
    
    # ========================================================================
    # 100-EPOCH PROGRESS
    # ========================================================================
    epoch100_results = all_results[all_results['Training'] == '100-epoch']
    
    print(f"\n{'=' * 120}")
    print("100-EPOCH TRAINING PROGRESS".center(120))
    print('=' * 120)
    
    if len(epoch100_results) > 0:
        print(f"\nCompleted: {len(epoch100_results)} / 70 experiments ({len(epoch100_results)/70*100:.1f}%)")
        print(f"\nModels with 100-epoch results:")
        for model in sorted(epoch100_results['Model'].unique()):
            count = len(epoch100_results[epoch100_results['Model'] == model])
            datasets = sorted(epoch100_results[epoch100_results['Model'] == model]['Dataset'].unique())
            print(f"  {model:15s}: {count}/7 datasets - {', '.join(datasets)}")
    else:
        print("\n100-epoch training just started - first experiment running...")
        print("Check back later for results!")
    
    # Save to CSV
    output_dir = Path('.')
    mse_pivot.to_csv(output_dir / 'comparison_mse_all.csv')
    mae_pivot.to_csv(output_dir / 'comparison_mae_all.csv')
    rankings.to_csv(output_dir / 'rankings_detailed.csv', index=False)
    overall.to_csv(output_dir / 'rankings_overall.csv', index=False)
    best_results.to_csv(output_dir / 'best_results_all.csv', index=False)
    
    print(f"\n{'=' * 120}")
    print("Tables saved to CSV files:".center(120))
    print('=' * 120)
    print("  - comparison_mse_all.csv")
    print("  - comparison_mae_all.csv")
    print("  - rankings_detailed.csv")
    print("  - rankings_overall.csv")
    print("  - best_results_all.csv")
    print('=' * 120)
