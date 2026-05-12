"""
Auto-update results_23-01.md as new experiments complete.
Monitors result_long_term_forecast.txt and updates the markdown tables in real-time.
"""

import re
import time
from datetime import datetime

MODELS = [
    'DLinear', 'PatchTST', 'TiDE', 'TimeXer', 'iTransformer',
    'PatchFusionBERT_v0', 'PatchFusionBERT_v2', 'PatchFusionBERT_BERTOnly'
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
    
    seq_len = None
    sl_match = re.search(r'_sl(\d+)_', exp_name)
    if sl_match:
        seq_len = int(sl_match.group(1))
    
    horizon = None
    pl_match = re.search(r'_pl(\d+)_', exp_name)
    if pl_match:
        horizon = int(pl_match.group(1))
    
    if exp_name.startswith('long_term_forecast_'):
        exp_name = exp_name[len('long_term_forecast_'):]
    
    parts = exp_name.split('_')
    
    model = None
    dataset = None
    
    # Check for format: Exchange_336_96_MODEL
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
    
    first_part = parts[0]
    
    if first_part == 'PFB':
        if len(parts) > 1 and parts[1] in ['v0', 'v2']:
            model = f'PatchFusionBERT_{parts[1]}'
    elif first_part == 'BERTOnly':
        model = 'PatchFusionBERT_BERTOnly'
    elif first_part in MODELS:
        model = first_part
    
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
            break
    
    return model, dataset, horizon, seq_len

def parse_results_file(filepath):
    """Parse result file and return experiments."""
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
                            experiments[key] = {'mse': mse, 'mae': mae, 'exp_name': exp_name}
        
        i += 1
    
    return experiments

def update_markdown_tables(experiments):
    """Rebuild the main results tables in the markdown."""
    
    # Build new tables for each dataset
    tables = {}
    
    for dataset, horizons in DATASETS.items():
        table_lines = []
        
        # Header
        table_lines.append(f"\n## Dataset: {dataset}\n\n")
        
        # Count coverage
        dataset_found = sum(1 for (m, d, h) in experiments.keys() if d == dataset)
        dataset_expected = len(horizons) * len(MODELS)
        dataset_coverage = (dataset_found / dataset_expected) * 100
        
        table_lines.append(f"**Coverage:** {dataset_found}/{dataset_expected} ({dataset_coverage:.1f}%)\n\n")
        
        # Table header
        table_lines.append("| Model | " + " | ".join([f"H={h} MSE" for h in horizons]) + " | " + " | ".join([f"H={h} MAE" for h in horizons]) + " |\n")
        table_lines.append("|" + "---|" * (1 + 2*len(horizons)) + "\n")
        
        # Rows
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
            
            table_lines.append("| " + " | ".join(row) + " |\n")
        
        # Missing list or complete
        missing = []
        for horizon in horizons:
            for model in MODELS:
                key = (model, dataset, horizon)
                if key not in experiments:
                    missing.append(f"- ❌ {model} - H={horizon}")
        
        if missing:
            table_lines.append(f"\n### Missing Experiments for {dataset}:\n")
            table_lines.extend([m + "\n" for m in missing])
        else:
            table_lines.append(f"\n### ✅ {dataset}: COMPLETE - All experiments present!\n")
        
        table_lines.append("\n---\n")
        
        tables[dataset] = ''.join(table_lines)
    
    return tables

def rebuild_markdown(experiments):
    """Rebuild the entire markdown with current results."""
    
    md = []
    
    md.append(f"# Comprehensive Experimental Results - Updated {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    md.append(f"**Protocol:** seq_len=336, epochs=100, patience=10, seed=2021\n")
    md.append(f"**Total experiments found:** {len(experiments)}\n")
    md.append("\n---\n")
    
    # Overall coverage
    total_expected = sum(len(horizons) * len(MODELS) for horizons in DATASETS.values())
    total_found = len(experiments)
    coverage_pct = (total_found / total_expected) * 100
    
    md.append(f"\n## Overall Coverage: {total_found}/{total_expected} ({coverage_pct:.1f}%)\n\n")
    
    # Dataset tables
    tables = update_markdown_tables(experiments)
    for dataset in DATASETS.keys():
        md.append(tables[dataset])
    
    # Summary by model
    md.append("\n## Summary by Model\n\n")
    md.append("| Model | Total Found | Total Expected | Coverage % |\n")
    md.append("|---|---|---|---|\n")
    
    for model in MODELS:
        found = sum(1 for (m, d, h) in experiments.keys() if m == model)
        expected = sum(len(horizons) for horizons in DATASETS.values())
        coverage = (found / expected) * 100 if expected > 0 else 0
        md.append(f"| **{model}** | {found} | {expected} | {coverage:.1f}% |\n")
    
    return ''.join(md)

def main():
    print("🔄 Auto-updating results_23-01.md as experiments complete...")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_count = 0
    update_number = 0
    
    while True:
        try:
            # Parse current results
            experiments = parse_results_file('result_long_term_forecast.txt')
            current_count = len(experiments)
            
            # Check if new experiments completed
            if current_count > last_count:
                new_experiments = current_count - last_count
                update_number += 1
                
                print(f"\n{'='*60}")
                print(f"UPDATE #{update_number} - {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'='*60}")
                print(f"✅ {new_experiments} new experiment(s) completed!")
                print(f"📊 Total: {current_count}/168 experiments")
                print(f"📈 Coverage: {(current_count/168)*100:.1f}%")
                
                # Rebuild markdown
                new_content = rebuild_markdown(experiments)
                
                # Save to markdown
                with open('results_23-01.md', 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"✅ Updated results_23-01.md")
                
                last_count = current_count
            
            # Wait before next check
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n\n🛑 Stopped monitoring")
            print(f"Final count: {last_count}/168 experiments")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(30)

if __name__ == '__main__':
    main()
