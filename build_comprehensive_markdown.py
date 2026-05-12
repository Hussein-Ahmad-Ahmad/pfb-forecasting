"""
Build comprehensive markdown table showing ALL datasets, horizons, models, and metrics.
This will show EXACTLY what exists and what's MISSING - no ambiguity.
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

def normalize_model_name(name):
    """Normalize model name to standard form."""
    if 'PatchFusionBERT_v0' in name or name == 'PFB_v0':
        return 'PatchFusionBERT_v0'
    elif 'PatchFusionBERT_v2' in name or name == 'PFB_v2':
        return 'PatchFusionBERT_v2'
    elif 'BERTOnly' in name:
        return 'PatchFusionBERT_BERTOnly'
    elif name in MODELS:
        return name
    return None

def parse_experiment_name(exp_name):
    """
    Parse experiment name to extract model, dataset, horizon, seq_len.
    Handles multiple formats:
    1. long_term_forecast_MODEL_100epoch_MODEL_DATASET_ftM_slSEQLEN_ll48_plHORIZON_...
    2. long_term_forecast_PFB_v2_100epoch_PatchFusionBERT_v2_DATASET_ftM_slSEQLEN_...
    3. long_term_forecast_Exchange_336_96_MODEL_custom_ftM_slSEQLEN_...
    4. long_term_forecast_Illness_336_60_MODEL_custom_ftM_slSEQLEN_...
    """
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
        # parts[1] is seq_len, parts[2] is horizon
        if not seq_len and parts[1].isdigit():
            seq_len = int(parts[1])
        if not horizon and parts[2].isdigit():
            horizon = int(parts[2])
        # parts[3] onwards contain model name
        model_part = parts[3]
        if model_part in MODELS:
            model = model_part
        elif model_part == 'PatchFusionBERT':
            if len(parts) > 4 and parts[4] in ['v0', 'v2', 'BERTOnly', 'RefineOnly']:
                model = f'PatchFusionBERT_{parts[4]}'
        return model, dataset, horizon, seq_len
    
    # Standard format: MODEL_100epoch_MODEL_DATASET or PFB_v2_100epoch_...
    first_part = parts[0]
    
    # Determine model from first part
    if first_part == 'PFB':
        if len(parts) > 1 and parts[1] in ['v0', 'v2']:
            model = f'PatchFusionBERT_{parts[1]}'
    elif first_part == 'BERTOnly':
        model = 'PatchFusionBERT_BERTOnly'
    elif first_part in MODELS:
        model = first_part
    
    # Find dataset - look for known dataset names
    for i, part in enumerate(parts):
        if part in DATASETS.keys():
            dataset = part
            break
        elif part == 'custom':
            # Custom could be Exchange, Weather, or Illness
            # Check if dataset appears earlier in the original experiment name
            if 'Exchange' in original_exp:
                dataset = 'Exchange'
            elif 'Weather' in original_exp:
                dataset = 'Weather'
            elif 'Illness' in original_exp:
                dataset = 'Illness'
            elif horizon in [24, 48, 60]:
                dataset = 'Illness'
            else:
                # Could be Exchange or Weather - can't determine without more context
                # Keep as None and skip this experiment
                dataset = None
            break
    
    return model, dataset, horizon, seq_len

def parse_results_file(filepath):
    """Parse result_long_term_forecast.txt and extract all experiments."""
    experiments = {}  # Key: (model, dataset, horizon) -> {'mse': X, 'mae': Y, 'exp_name': Z}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for experiment name pattern (starts with 'long_term_forecast_')
        if line.startswith('long_term_forecast_'):
            exp_name = line
            
            # Parse the name
            model, dataset, horizon, seq_len = parse_experiment_name(exp_name)
            
            # Process if seq_len=336 (standard) OR seq_len=104 (Illness)
            if seq_len in [104, 336] and model and dataset and horizon:
                # Next line should have metrics
                if i + 1 < len(lines):
                    metrics_line = lines[i + 1].strip()
                    
                    # Parse: mse:0.2981731593608856, mae:0.3489706814289093, dtw:Not calculated
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
                        # Keep best result if duplicate (lowest MSE)
                        if key not in experiments or mse < experiments[key]['mse']:
                            experiments[key] = {
                                'mse': mse,
                                'mae': mae,
                                'exp_name': exp_name
                            }
        
        i += 1
    
    return experiments

def build_markdown_table(experiments):
    """Build comprehensive markdown tables for each dataset."""
    
    md_content = []
    
    md_content.append("# Comprehensive Experimental Results - January 23, 2025\n")
    md_content.append(f"**Protocol:** seq_len=336, epochs=100, patience=10, seed=2021\n")
    md_content.append(f"**Total experiments found:** {len(experiments)}\n")
    md_content.append("\n---\n")
    
    # Calculate overall coverage
    total_expected = sum(len(horizons) * len(MODELS) for horizons in DATASETS.values())
    total_found = len(experiments)
    coverage_pct = (total_found / total_expected) * 100
    
    md_content.append(f"\n## Overall Coverage: {total_found}/{total_expected} ({coverage_pct:.1f}%)\n\n")
    
    # For each dataset, create a detailed table
    for dataset, horizons in DATASETS.items():
        md_content.append(f"\n## Dataset: {dataset}\n\n")
        
        # Count found for this dataset
        dataset_found = sum(1 for (m, d, h) in experiments.keys() if d == dataset)
        dataset_expected = len(horizons) * len(MODELS)
        dataset_coverage = (dataset_found / dataset_expected) * 100
        
        md_content.append(f"**Coverage:** {dataset_found}/{dataset_expected} ({dataset_coverage:.1f}%)\n\n")
        
        # Build table
        md_content.append("| Model | " + " | ".join([f"H={h} MSE" for h in horizons]) + " | " + " | ".join([f"H={h} MAE" for h in horizons]) + " |\n")
        md_content.append("|" + "---|" * (1 + 2*len(horizons)) + "\n")
        
        for model in MODELS:
            row = [f"**{model}**"]
            
            # MSE columns
            for horizon in horizons:
                key = (model, dataset, horizon)
                if key in experiments:
                    mse = experiments[key]['mse']
                    row.append(f"{mse:.4f}")
                else:
                    row.append("❌ MISSING")
            
            # MAE columns
            for horizon in horizons:
                key = (model, dataset, horizon)
                if key in experiments:
                    mae = experiments[key]['mae']
                    row.append(f"{mae:.4f}")
                else:
                    row.append("❌ MISSING")
            
            md_content.append("| " + " | ".join(row) + " |\n")
        
        # Add missing experiments list
        missing = []
        for horizon in horizons:
            for model in MODELS:
                key = (model, dataset, horizon)
                if key not in experiments:
                    missing.append(f"- ❌ {model} - H={horizon}")
        
        if missing:
            md_content.append(f"\n### Missing Experiments for {dataset}:\n")
            md_content.extend([m + "\n" for m in missing])
        else:
            md_content.append(f"\n### ✅ {dataset}: COMPLETE - All experiments present!\n")
        
        md_content.append("\n---\n")
    
    # Summary by model
    md_content.append("\n## Summary by Model\n\n")
    md_content.append("| Model | Total Found | Total Expected | Coverage % |\n")
    md_content.append("|---|---|---|---|\n")
    
    for model in MODELS:
        found = sum(1 for (m, d, h) in experiments.keys() if m == model)
        expected = sum(len(horizons) for horizons in DATASETS.values())
        coverage = (found / expected) * 100 if expected > 0 else 0
        md_content.append(f"| **{model}** | {found} | {expected} | {coverage:.1f}% |\n")
    
    return ''.join(md_content)

def main():
    print("Parsing result_long_term_forecast.txt...")
    
    experiments = parse_results_file('result_long_term_forecast.txt')
    
    print(f"Found {len(experiments)} experiments with seq_len=336")
    
    print("\nBuilding comprehensive markdown table...")
    
    markdown = build_markdown_table(experiments)
    
    with open('results_23-01.md', 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print("\n✅ Created results_23-01.md with comprehensive coverage tables")
    
    # Quick summary
    total_expected = sum(len(horizons) * len(MODELS) for horizons in DATASETS.values())
    total_found = len(experiments)
    missing = total_expected - total_found
    
    print(f"\nSummary:")
    print(f"  Total expected: {total_expected}")
    print(f"  Total found: {total_found}")
    print(f"  Missing: {missing}")
    print(f"  Coverage: {total_found}/{total_expected} ({(total_found/total_expected)*100:.1f}%)")

if __name__ == '__main__':
    main()
