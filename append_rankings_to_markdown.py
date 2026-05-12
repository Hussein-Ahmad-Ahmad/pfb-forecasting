"""
Append best performer rankings to results_23-01.md
Shows 1st, 2nd, 3rd place for each dataset × horizon × metric
"""

import re
from collections import defaultdict

# Define expected configurations
MODELS = [
    'DLinear',
    'PatchTST', 
    'TiDE',
    'TimeXer',
    'iTransformer',
    'PatchFusionBERT_v0',
    'PatchFusionBERT_v2',
    'PatchFusionBERT_BERTOnly'
]

DATASETS = {
    'ETTm1': [96, 192, 336],
    'ETTm2': [96, 192, 336],
    'ETTh1': [96, 192, 336],
    'ETTh2': [96, 192, 336],
    'Exchange': [96, 192, 336],
    'Weather': [96, 192, 336],
    'Illness': [24, 48, 60]
}

def parse_experiment_name(exp_name):
    """Parse experiment name to extract model, dataset, horizon, seq_len."""
    original_exp = exp_name
    
    # Extract seq_len from slXXX pattern
    seq_len = None
    sl_match = re.search(r'_sl(\d+)_', exp_name)
    if sl_match:
        seq_len = int(sl_match.group(1))
    
    # Extract horizon from plXXX pattern
    horizon = None
    pl_match = re.search(r'_pl(\d+)_', exp_name)
    if pl_match:
        horizon = int(pl_match.group(1))
    
    # Remove prefix
    if exp_name.startswith('long_term_forecast_'):
        exp_name = exp_name[len('long_term_forecast_'):]
    
    parts = exp_name.split('_')
    
    model = None
    dataset = None
    
    # Check for format: Exchange_336_96_MODEL or Weather_336_96_MODEL or Illness_336_60_MODEL
    if len(parts) >= 4 and parts[0] in ['Exchange', 'Weather', 'Illness']:
        dataset = parts[0]
        if not seq_len and parts[1].isdigit():
            seq_len = int(parts[1])
        if not horizon and parts[2].isdigit():
            horizon = int(parts[2])
        model_part = parts[3]
        if model_part in MODELS:
            model = model_part
        elif model_part == 'PatchFusionBERT':
            if len(parts) > 4 and parts[4] in ['v0', 'v2', 'BERTOnly', 'RefineOnly']:
                model = f'PatchFusionBERT_{parts[4]}'
        return model, dataset, horizon, seq_len
    
    # Standard format
    first_part = parts[0]
    
    if first_part == 'PFB':
        if len(parts) > 1 and parts[1] in ['v0', 'v2']:
            model = f'PatchFusionBERT_{parts[1]}'
    elif first_part == 'BERTOnly':
        model = 'PatchFusionBERT_BERTOnly'
    elif first_part in MODELS:
        model = first_part
    
    # Find dataset
    for i, part in enumerate(parts):
        if part in DATASETS.keys():
            dataset = part
            break
        elif part == 'custom':
            if 'Exchange' in original_exp:
                dataset = 'Exchange'
            elif 'Weather' in original_exp:
                dataset = 'Weather'
            elif 'Illness' in original_exp:
                dataset = 'Illness'
            elif horizon in [24, 48, 60]:
                dataset = 'Illness'
            else:
                dataset = None
            break
    
    return model, dataset, horizon, seq_len

def parse_results_file(filepath):
    """Parse result_long_term_forecast.txt and extract all experiments."""
    experiments = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('long_term_forecast_'):
            exp_name = line
            model, dataset, horizon, seq_len = parse_experiment_name(exp_name)
            
            if seq_len == 336 and model and dataset and horizon:
                if i + 1 < len(lines):
                    metrics_line = lines[i + 1].strip()
                    
                    mse = None
                    mae = None
                    
                    if 'mse:' in metrics_line and 'mae:' in metrics_line:
                        mse_match = re.search(r'mse:([\d.]+)', metrics_line)
                        mae_match = re.search(r'mae:([\d.]+)', metrics_line)
                        
                        if mse_match:
                            mse = float(mse_match.group(1))
                        if mae_match:
                            mae = float(mae_match.group(1))
                    
                    if mse is not None and mae is not None:
                        key = (model, dataset, horizon)
                        if key not in experiments or mse < experiments[key]['mse']:
                            experiments[key] = {
                                'mse': mse,
                                'mae': mae,
                                'exp_name': exp_name
                            }
        
        i += 1
    
    return experiments

def find_top_performers(experiments):
    """Find top 3 performers for each dataset × horizon × metric."""
    rankings = {}
    
    for dataset, horizons in DATASETS.items():
        rankings[dataset] = {}
        
        for horizon in horizons:
            rankings[dataset][horizon] = {'mse': [], 'mae': []}
            
            # Collect all results for this dataset × horizon
            results_mse = []
            results_mae = []
            
            for model in MODELS:
                key = (model, dataset, horizon)
                if key in experiments:
                    results_mse.append((model, experiments[key]['mse']))
                    results_mae.append((model, experiments[key]['mae']))
            
            # Sort and get top 3
            if results_mse:
                results_mse.sort(key=lambda x: x[1])
                rankings[dataset][horizon]['mse'] = results_mse[:3]
            
            if results_mae:
                results_mae.sort(key=lambda x: x[1])
                rankings[dataset][horizon]['mae'] = results_mae[:3]
    
    return rankings

def build_ranking_section(rankings):
    """Build markdown section with rankings."""
    md = []
    
    md.append("\n---\n")
    md.append("\n# 🏆 Best Performers Rankings\n\n")
    md.append("**Legend:** 🥇 1st Place | 🥈 2nd Place | 🥉 3rd Place\n\n")
    
    for dataset, horizons_dict in rankings.items():
        md.append(f"\n## {dataset}\n\n")
        
        for horizon in sorted(horizons_dict.keys()):
            metrics_dict = horizons_dict[horizon]
            
            md.append(f"### Horizon={horizon}\n\n")
            
            # MSE Rankings
            md.append("**MSE Rankings:**\n\n")
            if metrics_dict['mse']:
                for i, (model, score) in enumerate(metrics_dict['mse']):
                    medal = ['🥇', '🥈', '🥉'][i]
                    md.append(f"- {medal} **{model}**: {score:.4f}\n")
            else:
                md.append("- ❌ No experiments found\n")
            
            md.append("\n")
            
            # MAE Rankings
            md.append("**MAE Rankings:**\n\n")
            if metrics_dict['mae']:
                for i, (model, score) in enumerate(metrics_dict['mae']):
                    medal = ['🥇', '🥈', '🥉'][i]
                    md.append(f"- {medal} **{model}**: {score:.4f}\n")
            else:
                md.append("- ❌ No experiments found\n")
            
            md.append("\n")
    
    return ''.join(md)

def main():
    print("Parsing result_long_term_forecast.txt...")
    experiments = parse_results_file('result_long_term_forecast.txt')
    
    print(f"Found {len(experiments)} experiments with seq_len=336")
    
    print("\nFinding top performers...")
    rankings = find_top_performers(experiments)
    
    print("\nBuilding ranking section...")
    ranking_md = build_ranking_section(rankings)
    
    # Read current markdown
    with open('results_23-01.md', 'r', encoding='utf-8') as f:
        current_content = f.read()
    
    # Append rankings
    new_content = current_content + ranking_md
    
    with open('results_23-01.md', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("\n✅ Appended rankings to results_23-01.md")
    print("\nSample rankings:")
    print(ranking_md[:500] + "...")

if __name__ == '__main__':
    main()
