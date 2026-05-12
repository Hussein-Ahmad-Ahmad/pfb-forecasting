"""
Build comprehensive results CSV from result_long_term_forecast.txt
Saves as results_23-01.csv
"""
import pandas as pd
import re

def parse_experiment_name(exp_name):
    """Parse experiment name to extract metadata"""
    # Remove "long_term_forecast_" prefix
    exp_name = exp_name.replace('long_term_forecast_', '')
    
    parts = exp_name.split('_')
    
    model = None
    dataset = None
    horizon = None
    seq_len = None
    
    # Extract model - it's usually the FIRST part after long_term_forecast_
    # Handle compound model names (PFB_v0, PFB_v2, PatchFusionBERT_v0, etc.)
    if len(parts) > 0:
        first_part = parts[0]
        
        # Check for compound model names
        if first_part in ['PatchFusionBERT', 'Autoformer', 'Informer', 'Transformer']:
            if len(parts) > 1 and parts[1] in ['v0', 'v2', 'BERTOnly', 'PatchOnly', 'RefineOnly']:
                model = f"{first_part}_{parts[1]}"
            else:
                model = first_part
        elif first_part == 'PFB':
            # PFB_v0, PFB_v2, or just PFB
            if len(parts) > 1 and parts[1] in ['v0', 'v2']:
                model = f"PatchFusionBERT_{parts[1]}"
            else:
                model = 'PatchFusionBERT_v0'  # Default PFB to v0
        elif first_part == 'BERTOnly':
            model = 'PatchFusionBERT_BERTOnly'
        elif first_part in ['DLinear', 'PatchTST', 'TiDE', 'TimeXer', 'iTransformer']:
            model = first_part
        else:
            model = first_part
    
    # Extract dataset
    dataset_map = {
        'ETTm1': 'ETTm1',
        'ETTm2': 'ETTm2',
        'ETTh1': 'ETTh1',
        'ETTh2': 'ETTh2',
        'Exchange': 'Exchange',
        'Weather': 'Weather',
        'Illness': 'Illness',
        'ILI': 'Illness',
        'custom': None  # Will be determined by context
    }
    
    # Look for dataset indicators
    for part in parts:
        if part in dataset_map:
            dataset = dataset_map[part]
            break
    
    # Extract seq_len and pred_len
    for i, part in enumerate(parts):
        if part.startswith('sl') and len(part) > 2:
            try:
                seq_len = int(part[2:])
            except:
                pass
        elif part.startswith('pl') and len(part) > 2:
            try:
                horizon = int(part[2:])
            except:
                pass
    
    # Try pattern matching for model_id (e.g., ETTm1_336_96 or Exchange_336_96)
    for i in range(len(parts) - 2):
        if parts[i] in dataset_map.keys() and parts[i+1].isdigit() and parts[i+2].isdigit():
            if not dataset:
                dataset = dataset_map.get(parts[i], parts[i])
            if not seq_len:
                seq_len = int(parts[i+1])
            if not horizon:
                horizon = int(parts[i+2])
            break
    
    # Handle "custom" dataset - infer from other clues
    if dataset is None and 'custom' in exp_name.lower():
        # Check for typical horizon values to infer dataset
        if horizon == 24:
            dataset = 'Illness'
        elif seq_len in [96, 192, 336, 720]:
            # Could be Exchange or Weather, need more context
            if 'exchange' in exp_name.lower():
                dataset = 'Exchange'
            elif 'weather' in exp_name.lower():
                dataset = 'Weather'
    
    return model, dataset, horizon, seq_len

def build_csv_from_txt():
    """Parse result_long_term_forecast.txt and build CSV"""
    results_file = 'result_long_term_forecast.txt'
    
    all_results = []
    
    with open(results_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line and not line.startswith('mse:'):
            # This is an experiment name
            exp_name = line
            
            # Next line should have metrics
            if i + 1 < len(lines):
                metrics_line = lines[i + 1].strip()
                
                # Extract MSE and MAE
                mse_match = re.search(r'mse:\s*([0-9.]+)', metrics_line)
                mae_match = re.search(r'mae:\s*([0-9.]+)', metrics_line)
                
                if mse_match and mae_match:
                    mse = float(mse_match.group(1))
                    mae = float(mae_match.group(1))
                    
                    # Parse experiment name
                    model, dataset, horizon, seq_len = parse_experiment_name(exp_name)
                    
                    all_results.append({
                        'Model': model,
                        'Dataset': dataset,
                        'Horizon': horizon,
                        'SeqLen': seq_len if seq_len else 336,
                        'MSE': mse,
                        'MAE': mae,
                        'ExperimentName': exp_name
                    })
                
                i += 2  # Skip to next experiment
            else:
                i += 1
        else:
            i += 1
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Clean up None values
    df = df.dropna(subset=['Model', 'Dataset', 'Horizon'])
    
    # Convert types
    df['Horizon'] = df['Horizon'].astype(int)
    df['SeqLen'] = df['SeqLen'].astype(int)
    
    # Sort
    df = df.sort_values(['Dataset', 'Horizon', 'Model']).reset_index(drop=True)
    
    # Save
    output_file = 'results_23-01.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Successfully created {output_file}")
    print(f"   Total experiments: {len(df)}")
    
    print(f"\n{'='*80}")
    print("BREAKDOWN BY DATASET:")
    print(f"{'='*80}")
    for dataset in sorted(df['Dataset'].unique()):
        count = len(df[df['Dataset'] == dataset])
        horizons = sorted(df[df['Dataset'] == dataset]['Horizon'].unique())
        print(f"{dataset:15s}: {count:3d} experiments  (H={horizons})")
    
    print(f"\n{'='*80}")
    print("BREAKDOWN BY MODEL:")
    print(f"{'='*80}")
    for model in sorted(df['Model'].unique()):
        count = len(df[df['Model'] == model])
        print(f"{model:30s}: {count:3d} experiments")
    
    print(f"\n{'='*80}")
    print("SUMMARY BY DATASET × HORIZON:")
    print(f"{'='*80}")
    summary = df.groupby(['Dataset', 'Horizon']).agg({
        'Model': 'count',
        'MSE': ['mean', 'min', 'max']
    }).round(4)
    summary.columns = ['N_Models', 'MSE_Mean', 'MSE_Min', 'MSE_Max']
    print(summary)
    
    print(f"\n{'='*80}")
    print("SAMPLE ROWS:")
    print(f"{'='*80}")
    print(df[['Model', 'Dataset', 'Horizon', 'MSE', 'MAE']].head(15).to_string())
    
    return df

if __name__ == "__main__":
    print("="*80)
    print("BUILDING COMPREHENSIVE RESULTS CSV - results_23-01.csv")
    print("="*80)
    print()
    
    df = build_csv_from_txt()
    
    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}")
