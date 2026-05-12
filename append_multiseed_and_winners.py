"""
Append multi-seed results and winner counts to results_23-01.md
"""

import re
from collections import defaultdict

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

def normalize_model_name(name):
    """Normalize model name."""
    if 'PatchFusionBERT_v0' in name or name == 'PFB_v0' or name == 'PFB' or 'PFB_v0' in name:
        return 'PatchFusionBERT_v0'
    elif 'PatchFusionBERT_v2' in name or name == 'PFB_v2':
        return 'PatchFusionBERT_v2'
    elif 'BERTOnly' in name:
        return 'PatchFusionBERT_BERTOnly'
    elif name in MODELS:
        return name
    return None

def parse_multiseed_experiments(filepath):
    """Parse multi-seed experiments from result file."""
    experiments = defaultdict(list)  # Key: (model, dataset, horizon, seed) -> metrics
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for multi-seed experiments (MS2021, MS2022, MS2023)
        if line.startswith('long_term_forecast_') and ('MS2021' in line or 'MS2022' in line or 'MS2023' in line):
            exp_name = line
            
            # Extract seed
            seed = None
            if 'MS2021' in exp_name:
                seed = 2021
            elif 'MS2022' in exp_name:
                seed = 2022
            elif 'MS2023' in exp_name:
                seed = 2023
            
            # Extract horizon
            horizon_match = re.search(r'_pl(\d+)_', exp_name)
            horizon = int(horizon_match.group(1)) if horizon_match else None
            
            # Extract seq_len
            sl_match = re.search(r'_sl(\d+)_', exp_name)
            seq_len = int(sl_match.group(1)) if sl_match else None
            
            # Extract dataset and model
            dataset = None
            model = None
            
            # Check for format: Dataset_SeqLen_Horizon_MSSeed_Model (e.g., ETTm1_336_192_MS2021_PatchFusionBERT_v2)
            if '_MS20' in exp_name:
                parts = exp_name.replace('long_term_forecast_', '').split('_')
                # parts[0] might be dataset, parts might contain MS2021/MS2022/MS2023
                for i, part in enumerate(parts):
                    if part.startswith('MS202'):
                        # Found the seed marker, dataset should be before it
                        if i > 0 and parts[0] in DATASETS.keys():
                            dataset = parts[0]
                        # Model should be after the seed marker
                        if i + 1 < len(parts):
                            model_raw = parts[i + 1]
                            if i + 2 < len(parts) and parts[i + 2] in ['v0', 'v2', 'BERTOnly', 'RefineOnly']:
                                model_raw = f"{model_raw}_{parts[i + 2]}"
                            model = normalize_model_name(model_raw)
                        break
            
            # Fallback: look for dataset in exp_name
            if dataset is None:
                for ds in DATASETS.keys():
                    if ds in exp_name:
                        dataset = ds
                        break
            
            # If custom, try to infer
            if dataset is None and 'custom' in exp_name:
                if horizon in [24, 36, 48, 60]:
                    dataset = 'Illness'
                elif 'Exchange' in exp_name:
                    dataset = 'Exchange'
                elif 'Weather' in exp_name:
                    dataset = 'Weather'
            
            # Get metrics
            if i + 1 < len(lines):
                metrics_line = lines[i + 1].strip()
                mse_match = re.search(r'mse:([\d.]+)', metrics_line)
                mae_match = re.search(r'mae:([\d.]+)', metrics_line)
                
                mse = float(mse_match.group(1)) if mse_match else None
                mae = float(mae_match.group(1)) if mae_match else None
                
                if model and dataset and horizon and seed and mse is not None and mae is not None:
                    key = (model, dataset, horizon)
                    experiments[key].append({
                        'seed': seed,
                        'mse': mse,
                        'mae': mae,
                        'seq_len': seq_len
                    })
        
        i += 1
    
    return experiments

def parse_single_seed_experiments(filepath):
    """Parse single-seed experiments."""
    experiments = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('long_term_forecast_') and 'MS2021' not in line and 'MS2022' not in line and 'MS2023' not in line:
            exp_name = line
            
            # Extract horizon
            horizon_match = re.search(r'_pl(\d+)_', exp_name)
            horizon = int(horizon_match.group(1)) if horizon_match else None
            
            # Extract seq_len
            sl_match = re.search(r'_sl(\d+)_', exp_name)
            seq_len = int(sl_match.group(1)) if sl_match else None
            
            if seq_len != 336:
                i += 1
                continue
            
            # Extract model and dataset using same logic as before
            if exp_name.startswith('long_term_forecast_'):
                exp_name_clean = exp_name[len('long_term_forecast_'):]
            
            parts = exp_name_clean.split('_')
            
            model = None
            dataset = None
            
            # Check for format: Exchange_336_96_MODEL
            if len(parts) >= 4 and parts[0] in ['Exchange', 'Weather', 'Illness']:
                dataset = parts[0]
                model_part = parts[3]
                if model_part in MODELS:
                    model = model_part
                elif model_part == 'PatchFusionBERT':
                    if len(parts) > 4 and parts[4] in ['v0', 'v2', 'BERTOnly', 'RefineOnly']:
                        model = f'PatchFusionBERT_{parts[4]}'
            else:
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
                for part in parts:
                    if part in DATASETS.keys():
                        dataset = part
                        break
                    elif part == 'custom':
                        if 'Exchange' in exp_name:
                            dataset = 'Exchange'
                        elif 'Weather' in exp_name:
                            dataset = 'Weather'
                        elif 'Illness' in exp_name or horizon in [24, 48, 60]:
                            dataset = 'Illness'
                        break
            
            # Get metrics
            if i + 1 < len(lines):
                metrics_line = lines[i + 1].strip()
                mse_match = re.search(r'mse:([\d.]+)', metrics_line)
                mae_match = re.search(r'mae:([\d.]+)', metrics_line)
                
                mse = float(mse_match.group(1)) if mse_match else None
                mae = float(mae_match.group(1)) if mae_match else None
                
                if model and dataset and horizon and mse is not None and mae is not None:
                    key = (model, dataset, horizon)
                    experiments[key] = {'mse': mse, 'mae': mae}
        
        i += 1
    
    return experiments

def count_winners(single_seed_experiments):
    """Count how many times each model wins across all configurations."""
    win_counts = defaultdict(lambda: {'mse': 0, 'mae': 0, 'total': 0})
    
    # Group by dataset and horizon
    by_config = defaultdict(list)
    for (model, dataset, horizon), metrics in single_seed_experiments.items():
        config = (dataset, horizon)
        by_config[config].append((model, metrics))
    
    # For each configuration, find winners
    for config, results in by_config.items():
        if not results:
            continue
        
        # Find MSE winner
        mse_results = [(model, metrics['mse']) for model, metrics in results]
        mse_results.sort(key=lambda x: x[1])
        if mse_results:
            mse_winner = mse_results[0][0]
            win_counts[mse_winner]['mse'] += 1
            win_counts[mse_winner]['total'] += 1
        
        # Find MAE winner
        mae_results = [(model, metrics['mae']) for model, metrics in results]
        mae_results.sort(key=lambda x: x[1])
        if mae_results:
            mae_winner = mae_results[0][0]
            win_counts[mae_winner]['mae'] += 1
            win_counts[mae_winner]['total'] += 1
    
    return win_counts

def build_multiseed_section(multiseed_experiments):
    """Build multi-seed coverage section."""
    md = []
    
    md.append("\n---\n")
    md.append("\n# 🔄 Multi-Seed Experiments Coverage\n\n")
    md.append("**Protocol:** 3 seeds (2021, 2022, 2023) for robustness validation\n\n")
    
    # Calculate statistics
    total_configs = 0
    complete_configs = 0
    partial_configs = 0
    
    by_dataset = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for (model, dataset, horizon), seeds_data in multiseed_experiments.items():
        total_configs += 1
        n_seeds = len(seeds_data)
        by_dataset[dataset][horizon][model] = n_seeds
        
        if n_seeds >= 3:
            complete_configs += 1
        elif n_seeds > 0:
            partial_configs += 1
    
    md.append(f"**Overall Multi-Seed Coverage:**\n")
    md.append(f"- Total configurations with multi-seed: {total_configs}\n")
    md.append(f"- Complete (3 seeds): {complete_configs}\n")
    md.append(f"- Partial (1-2 seeds): {partial_configs}\n\n")
    
    # Detailed breakdown by dataset
    for dataset in sorted(DATASETS.keys()):
        horizons = DATASETS[dataset]
        
        md.append(f"\n## {dataset}\n\n")
        
        for horizon in sorted(horizons):
            md.append(f"### Horizon={horizon}\n\n")
            
            found_any = False
            for model in MODELS:
                n_seeds = by_dataset[dataset][horizon].get(model, 0)
                if n_seeds > 0:
                    found_any = True
                    status = "✅ COMPLETE" if n_seeds >= 3 else f"⚠️ PARTIAL ({n_seeds}/3 seeds)"
                    md.append(f"- **{model}**: {status}\n")
            
            if not found_any:
                md.append("- ❌ No multi-seed experiments\n")
            
            # Show missing models
            missing_models = [m for m in MODELS if by_dataset[dataset][horizon].get(m, 0) == 0]
            if missing_models:
                md.append(f"\n  **Missing:** {', '.join(missing_models)}\n")
            
            md.append("\n")
    
    return ''.join(md)

def build_winner_count_section(win_counts):
    """Build winner count section."""
    md = []
    
    md.append("\n---\n")
    md.append("\n# 🏅 Overall Winner Statistics\n\n")
    md.append("**Summary of wins across all dataset × horizon configurations**\n\n")
    
    # Sort by total wins
    sorted_models = sorted(win_counts.items(), key=lambda x: x[1]['total'], reverse=True)
    
    md.append("| Rank | Model | MSE Wins | MAE Wins | Total Wins |\n")
    md.append("|------|-------|----------|----------|------------|\n")
    
    medals = ['🥇', '🥈', '🥉']
    for i, (model, counts) in enumerate(sorted_models):
        rank = medals[i] if i < 3 else f"{i+1}"
        md.append(f"| {rank} | **{model}** | {counts['mse']} | {counts['mae']} | {counts['total']} |\n")
    
    # Add insights
    md.append("\n### Key Insights\n\n")
    
    if sorted_models:
        top_model = sorted_models[0][0]
        top_wins = sorted_models[0][1]['total']
        md.append(f"- 🏆 **Overall Champion:** {top_model} with {top_wins} total wins\n")
        
        # MSE specialist
        mse_sorted = sorted(win_counts.items(), key=lambda x: x[1]['mse'], reverse=True)
        if mse_sorted:
            mse_champ = mse_sorted[0][0]
            mse_wins = mse_sorted[0][1]['mse']
            md.append(f"- 📊 **MSE Specialist:** {mse_champ} with {mse_wins} MSE wins\n")
        
        # MAE specialist
        mae_sorted = sorted(win_counts.items(), key=lambda x: x[1]['mae'], reverse=True)
        if mae_sorted:
            mae_champ = mae_sorted[0][0]
            mae_wins = mae_sorted[0][1]['mae']
            md.append(f"- 📈 **MAE Specialist:** {mae_champ} with {mae_wins} MAE wins\n")
    
    return ''.join(md)

def main():
    print("Parsing result_long_term_forecast.txt...")
    
    print("  - Single-seed experiments...")
    single_seed = parse_single_seed_experiments('result_long_term_forecast.txt')
    print(f"    Found {len(single_seed)} single-seed experiments")
    
    print("  - Multi-seed experiments...")
    multiseed = parse_multiseed_experiments('result_long_term_forecast.txt')
    print(f"    Found {len(multiseed)} multi-seed configurations")
    
    print("\nCounting winners...")
    win_counts = count_winners(single_seed)
    
    print("\nBuilding multi-seed section...")
    multiseed_md = build_multiseed_section(multiseed)
    
    print("Building winner count section...")
    winner_md = build_winner_count_section(win_counts)
    
    # Read current markdown
    with open('results_23-01.md', 'r', encoding='utf-8') as f:
        current_content = f.read()
    
    # Append new sections
    new_content = current_content + multiseed_md + winner_md
    
    with open('results_23-01.md', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("\n✅ Appended multi-seed coverage and winner statistics to results_23-01.md")
    
    # Show top 3 winners
    sorted_models = sorted(win_counts.items(), key=lambda x: x[1]['total'], reverse=True)
    print("\n🏅 Top 3 Overall Winners:")
    for i, (model, counts) in enumerate(sorted_models[:3]):
        medal = ['🥇', '🥈', '🥉'][i]
        print(f"  {medal} {model}: {counts['total']} wins (MSE: {counts['mse']}, MAE: {counts['mae']})")

if __name__ == '__main__':
    main()
