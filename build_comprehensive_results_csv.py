"""
Build comprehensive results CSV from all experiment result directories
Saves as results_23-01.csv
"""
import os
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

def extract_result_from_npy(result_dir):
    """Extract MSE and MAE from metrics.npy file"""
    import numpy as np
    try:
        metrics_file = result_dir / 'metrics.npy'
        if metrics_file.exists():
            metrics = np.load(metrics_file)
            # metrics is typically [mse, mae, ...]
            if len(metrics) >= 2:
                return float(metrics[0]), float(metrics[1])
        return None, None
    except:
        return None, None

def parse_experiment_name(exp_name):
    """Parse experiment directory name to extract metadata"""
    # Pattern: long_term_forecast_[Model]_[Config]_[Dataset]_[Horizon]_...
    parts = exp_name.split('_')
    
    model = None
    dataset = None
    horizon = None
    seq_len = None
    seed = None
    
    # Extract model (usually after "forecast")
    if 'forecast' in parts:
        idx = parts.index('forecast') + 1
        if idx < len(parts):
            # Handle compound model names
            if parts[idx] in ['PatchFusionBERT', 'Informer', 'Autoformer', 'Transformer']:
                model = parts[idx]
                if idx + 1 < len(parts) and parts[idx + 1] in ['v0', 'v2', 'BERTOnly', 'PatchOnly', 'RefineOnly']:
                    model = f"{parts[idx]}_{parts[idx + 1]}"
            else:
                model = parts[idx]
    
    # Extract dataset
    dataset_keywords = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather', 'Illness', 'ILI', 'custom']
    for part in parts:
        if part in dataset_keywords:
            dataset = part
            if part == 'ILI':
                dataset = 'Illness'
            break
    
    # Extract horizon (pred_len)
    for i, part in enumerate(parts):
        if part.startswith('pl') and i > 0:
            try:
                horizon = int(part[2:])
            except:
                pass
        elif part.startswith('sl') and i > 0:
            try:
                seq_len = int(part[2:])
            except:
                pass
    
    # Try to extract from model_id pattern (e.g., ETTm1_336_96)
    if not horizon or not dataset:
        for i, part in enumerate(parts):
            if part.isdigit() and i > 0:
                prev_part = parts[i-1]
                if prev_part in dataset_keywords or prev_part.isdigit():
                    if i + 1 < len(parts) and parts[i+1].isdigit():
                        if not seq_len:
                            seq_len = int(part)
                        if not horizon:
                            horizon = int(parts[i+1])
    
    return model, dataset, horizon, seq_len

def build_results_csv():
    """Build comprehensive CSV from all result directories"""
    results_dir = Path('results')
    
    if not results_dir.exists():
        print("Error: results directory not found")
        return
    
    all_results = []
    
    # Iterate through all experiment directories
    exp_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(exp_dirs)} experiment directories")
    print("Parsing results...")
    
    for i, exp_dir in enumerate(exp_dirs, 1):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(exp_dirs)} directories...")
        
        exp_name = exp_dir.name
        
        # Parse experiment metadata
        model, dataset, horizon, seq_len = parse_experiment_name(exp_name)
        
        # Extract metrics from metrics.npy file
        mse, mae = extract_result_from_npy(exp_dir)
        
        if mse is not None and mae is not None:
            all_results.append({
                'Model': model,
                'Dataset': dataset,
                'Horizon': horizon,
                'SeqLen': seq_len if seq_len else 336,  # Default to 336
                'MSE': mse,
                'MAE': mae,
                'ExperimentName': exp_name
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    if len(df) == 0:
        print("Warning: No results found!")
        return df
    
    # Sort by Dataset, Horizon, Model (handle None values)
    df['Dataset'] = df['Dataset'].fillna('Unknown')
    df['Horizon'] = df['Horizon'].fillna(0)
    df['Model'] = df['Model'].fillna('Unknown')
    df = df.sort_values(['Dataset', 'Horizon', 'Model']).reset_index(drop=True)
    
    # Save to CSV
    output_file = 'results_23-01.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Successfully created {output_file}")
    print(f"   Total experiments: {len(df)}")
    print(f"\nBreakdown by dataset:")
    for dataset in sorted(df['Dataset'].unique()):
        count = len(df[df['Dataset'] == dataset])
        print(f"   {dataset}: {count} experiments")
    
    print(f"\nBreakdown by model:")
    for model in sorted(df['Model'].unique()):
        count = len(df[df['Model'] == model])
        print(f"   {model}: {count} experiments")
    
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head(10).to_string())
    
    # Also create a summary by Dataset x Horizon
    print("\n" + "="*80)
    print("SUMMARY: Experiments per Dataset x Horizon")
    print("="*80)
    summary = df.groupby(['Dataset', 'Horizon']).size().reset_index(name='Count')
    for dataset in sorted(summary['Dataset'].unique()):
        subset = summary[summary['Dataset'] == dataset]
        print(f"\n{dataset}:")
        for _, row in subset.iterrows():
            print(f"  H={row['Horizon']}: {row['Count']} models")
    
    return df

if __name__ == "__main__":
    print("="*80)
    print("BUILDING COMPREHENSIVE RESULTS CSV - results_23-01.csv")
    print("="*80)
    print()
    
    df = build_results_csv()
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
