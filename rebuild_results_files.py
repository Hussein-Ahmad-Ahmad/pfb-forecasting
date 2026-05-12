#!/usr/bin/env python
"""Rebuild results_23-01.csv and results_23-01.md with latest data including multi-seed experiments."""

import os
import re
from collections import defaultdict
import numpy as np

# Configuration
RESULT_FILE = 'result_long_term_forecast.txt'
OUTPUT_CSV = 'results_23-01.csv'
OUTPUT_MD = 'results_23-01.md'

# Standard datasets use H=96, 192, 336
STANDARD_DATASETS = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather']
STANDARD_HORIZONS = [96, 192, 336]

# Illness uses H=24, 48, 60
ILLNESS_HORIZONS = [24, 48, 60]

# Core models (excluding legacy baselines like Autoformer, Informer, Transformer)
CORE_MODELS = ['DLinear', 'PatchTST', 'TiDE', 'TimeXer', 'iTransformer', 
               'PatchFusionBERT_v0', 'PatchFusionBERT_v2', 'PatchFusionBERT_BERTOnly']

# Top 4 models for multi-seed validation
TOP_MODELS = ['PatchFusionBERT_v0', 'PatchFusionBERT_v2', 'DLinear', 'PatchTST']

def parse_result_file():
    """Parse result file and extract all experiments."""
    experiments = []
    
    with open(RESULT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines or find experiment blocks
    lines = content.strip().split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for experiment name (starts with long_term_forecast)
        if line.startswith('long_term_forecast_'):
            exp_name = line
            
            # Next line should have mse/mae
            if i + 1 < len(lines):
                metrics_line = lines[i + 1].strip()
                
                # Parse mse and mae from "mse:value, mae:value" format
                try:
                    mse_match = re.search(r'mse[:\s]*([\d.]+)', metrics_line)
                    mae_match = re.search(r'mae[:\s]*([\d.]+)', metrics_line)
                    
                    if mse_match and mae_match:
                        mse = float(mse_match.group(1))
                        mae = float(mae_match.group(1))
                        
                        # Extract info from experiment name
                        info = parse_experiment_name(exp_name)
                        if info:
                            info['mse'] = mse
                            info['mae'] = mae
                            info['exp_name'] = exp_name
                            experiments.append(info)
                except (ValueError, AttributeError):
                    pass
                
                i += 2
                continue
        
        i += 1
    
    return experiments

def parse_experiment_name(exp_name):
    """Parse experiment name to extract model, dataset, horizon, seq_len, and seed info."""
    
    # Check for multi-seed experiments
    is_multiseed = False
    seed = None
    
    # Pattern for multi-seed (MS2021, MS2022, MS2023)
    ms_match = re.search(r'_MS(\d{4})_', exp_name)
    if ms_match:
        is_multiseed = True
        seed = int(ms_match.group(1))
    
    # Also check for seed2021, seed2022, seed2023 patterns
    seed_match = re.search(r'_seed(\d{4})_', exp_name)
    if seed_match:
        is_multiseed = True
        seed = int(seed_match.group(1))
    
    # Extract model
    model = None
    for m in CORE_MODELS + ['Autoformer', 'Informer', 'Transformer']:
        if m in exp_name:
            model = m
            break
    
    if not model:
        return None
    
    # Extract dataset
    dataset = None
    for ds in STANDARD_DATASETS + ['Illness']:
        if ds in exp_name:
            dataset = ds
            break
    
    if not dataset:
        return None
    
    # Extract horizon (pl value)
    horizon_match = re.search(r'_pl(\d+)_', exp_name)
    if not horizon_match:
        return None
    horizon = int(horizon_match.group(1))
    
    # Extract seq_len (sl value)
    seqlen_match = re.search(r'_sl(\d+)_', exp_name)
    if not seqlen_match:
        return None
    seq_len = int(seqlen_match.group(1))
    
    return {
        'model': model,
        'dataset': dataset,
        'horizon': horizon,
        'seq_len': seq_len,
        'is_multiseed': is_multiseed,
        'seed': seed
    }

def get_best_single_seed(experiments, model, dataset, horizon):
    """Get the best single-seed result for a model/dataset/horizon combination.
    
    If no true single-seed experiment exists, use seed2021 as the primary.
    """
    # First try to find true single-seed (no seed marker)
    matching = [e for e in experiments 
                if e['model'] == model 
                and e['dataset'] == dataset 
                and e['horizon'] == horizon
                and not e['is_multiseed']]
    
    if matching:
        # Return the one with lowest MSE
        return min(matching, key=lambda x: x['mse'])
    
    # If no true single-seed, try to find seed2021 as the baseline
    seed2021_match = [e for e in experiments 
                      if e['model'] == model 
                      and e['dataset'] == dataset 
                      and e['horizon'] == horizon
                      and e['seed'] == 2021]
    
    if seed2021_match:
        return min(seed2021_match, key=lambda x: x['mse'])
    
    return None

def get_multiseed_results(experiments, model, dataset, horizon):
    """Get all multi-seed results for a model/dataset/horizon combination."""
    matching = [e for e in experiments 
                if e['model'] == model 
                and e['dataset'] == dataset 
                and e['horizon'] == horizon
                and e['is_multiseed']]
    return matching

def compute_multiseed_stats(results):
    """Compute mean and std for multi-seed results."""
    if not results:
        return None
    
    mse_values = [r['mse'] for r in results]
    mae_values = [r['mae'] for r in results]
    
    return {
        'mse_mean': np.mean(mse_values),
        'mse_std': np.std(mse_values),
        'mae_mean': np.mean(mae_values),
        'mae_std': np.std(mae_values),
        'count': len(results),
        'seeds': sorted(set(r['seed'] for r in results if r['seed']))
    }

def build_csv(experiments):
    """Build the CSV file."""
    lines = ['Model,Dataset,Horizon,SeqLen,MSE,MAE,ExperimentName']
    
    # Sort experiments
    sorted_exps = sorted(experiments, key=lambda x: (x['dataset'], x['horizon'], x['model']))
    
    for exp in sorted_exps:
        line = f"{exp['model']},{exp['dataset']},{exp['horizon']},{exp['seq_len']},{exp['mse']},{exp['mae']},{exp['exp_name']}"
        lines.append(line)
    
    with open(OUTPUT_CSV, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Written {len(sorted_exps)} experiments to {OUTPUT_CSV}")

def build_markdown(experiments):
    """Build comprehensive markdown with all results."""
    lines = []
    
    # Header
    lines.append("# Comprehensive Experimental Results - January 24, 2025")
    lines.append("**Protocol:** seq_len=336 (standard) / 104 (Illness), epochs=100, patience=10, seed=2021")
    
    # Count unique experiments
    unique_exps = set()
    for exp in experiments:
        if not exp['is_multiseed']:
            unique_exps.add((exp['model'], exp['dataset'], exp['horizon']))
    
    lines.append(f"**Total single-seed experiments:** {len(unique_exps)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Overall coverage
    expected = len(CORE_MODELS) * (len(STANDARD_DATASETS) * len(STANDARD_HORIZONS) + len(ILLNESS_HORIZONS))
    coverage_pct = len(unique_exps) / expected * 100 if expected > 0 else 0
    lines.append(f"## Overall Coverage: {len(unique_exps)}/{expected} ({coverage_pct:.1f}%)")
    lines.append("")
    
    # Build tables for each dataset
    for dataset in STANDARD_DATASETS:
        horizons = STANDARD_HORIZONS
        lines.extend(build_dataset_table(experiments, dataset, horizons))
    
    # Illness dataset
    lines.extend(build_dataset_table(experiments, 'Illness', ILLNESS_HORIZONS))
    
    # Summary by model
    lines.extend(build_model_summary(experiments))
    
    # Best performers rankings
    lines.extend(build_rankings(experiments))
    
    # Multi-seed detailed results
    lines.extend(build_multiseed_section(experiments))
    
    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Written comprehensive results to {OUTPUT_MD}")

def build_dataset_table(experiments, dataset, horizons):
    """Build a table for a single dataset."""
    lines = []
    lines.append(f"## Dataset: {dataset}")
    lines.append("")
    
    # Count coverage
    found = 0
    expected = len(CORE_MODELS) * len(horizons)
    
    for model in CORE_MODELS:
        for h in horizons:
            if get_best_single_seed(experiments, model, dataset, h):
                found += 1
    
    coverage_pct = found / expected * 100 if expected > 0 else 0
    lines.append(f"**Coverage:** {found}/{expected} ({coverage_pct:.1f}%)")
    lines.append("")
    
    # Build header
    if dataset == 'Illness':
        header = "| Model |"
        for h in horizons:
            header += f" H={h} MSE |"
        for h in horizons:
            header += f" H={h} MAE |"
    else:
        header = "| Model | H=96 MSE | H=192 MSE | H=336 MSE | H=96 MAE | H=192 MAE | H=336 MAE |"
    
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (1 + 2 * len(horizons))) + "|")
    
    # Build rows
    for model in CORE_MODELS:
        row = f"| **{model}** |"
        
        # MSE columns
        for h in horizons:
            exp = get_best_single_seed(experiments, model, dataset, h)
            if exp:
                row += f" {exp['mse']:.4f} |"
            else:
                row += " - |"
        
        # MAE columns
        for h in horizons:
            exp = get_best_single_seed(experiments, model, dataset, h)
            if exp:
                row += f" {exp['mae']:.4f} |"
            else:
                row += " - |"
        
        lines.append(row)
    
    lines.append("")
    if found >= expected:
        lines.append(f"### ✅ {dataset}: COMPLETE - All experiments present!")
    else:
        lines.append(f"### ⚠️ {dataset}: {expected - found} experiments missing")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    return lines

def build_model_summary(experiments):
    """Build summary table by model."""
    lines = []
    lines.append("## Summary by Model")
    lines.append("")
    lines.append("| Model | Total Found | Total Expected | Coverage % |")
    lines.append("|---|---|---|---|")
    
    for model in CORE_MODELS:
        found = 0
        expected = len(STANDARD_DATASETS) * len(STANDARD_HORIZONS) + len(ILLNESS_HORIZONS)
        
        for dataset in STANDARD_DATASETS:
            for h in STANDARD_HORIZONS:
                if get_best_single_seed(experiments, model, dataset, h):
                    found += 1
        
        for h in ILLNESS_HORIZONS:
            if get_best_single_seed(experiments, model, 'Illness', h):
                found += 1
        
        coverage_pct = found / expected * 100 if expected > 0 else 0
        lines.append(f"| **{model}** | {found} | {expected} | {coverage_pct:.1f}% |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    return lines

def build_rankings(experiments):
    """Build best performers rankings."""
    lines = []
    lines.append("# 🏆 Best Performers Rankings")
    lines.append("")
    lines.append("**Legend:** 🥇 1st Place | 🥈 2nd Place | 🥉 3rd Place")
    lines.append("")
    
    for dataset in STANDARD_DATASETS + ['Illness']:
        lines.append(f"## {dataset}")
        lines.append("")
        
        horizons = ILLNESS_HORIZONS if dataset == 'Illness' else STANDARD_HORIZONS
        
        for h in horizons:
            lines.append(f"### Horizon={h}")
            lines.append("")
            
            # Get all results for this dataset/horizon
            results = []
            for model in CORE_MODELS:
                exp = get_best_single_seed(experiments, model, dataset, h)
                if exp:
                    results.append(exp)
            
            if results:
                # MSE rankings
                lines.append("**MSE Rankings:**")
                lines.append("")
                sorted_by_mse = sorted(results, key=lambda x: x['mse'])[:3]
                medals = ['🥇', '🥈', '🥉']
                for i, exp in enumerate(sorted_by_mse):
                    lines.append(f"- {medals[i]} **{exp['model']}**: {exp['mse']:.4f}")
                lines.append("")
                
                # MAE rankings
                lines.append("**MAE Rankings:**")
                lines.append("")
                sorted_by_mae = sorted(results, key=lambda x: x['mae'])[:3]
                for i, exp in enumerate(sorted_by_mae):
                    lines.append(f"- {medals[i]} **{exp['model']}**: {exp['mae']:.4f}")
                lines.append("")
            else:
                lines.append("*No results available*")
                lines.append("")
    
    lines.append("---")
    lines.append("")
    
    return lines

def build_multiseed_section(experiments):
    """Build detailed multi-seed experiment section."""
    lines = []
    lines.append("# 🔬 Multi-Seed Experiment Results (Detailed)")
    lines.append("")
    
    # Count total multi-seed experiments
    multiseed_exps = [e for e in experiments if e['is_multiseed']]
    lines.append(f"**Total Multi-Seed Runs:** {len(multiseed_exps)}")
    lines.append("**Seeds:** 2021, 2022, 2023")
    lines.append("**Protocol:** seq_len=336 (standard) / 104 (Illness), epochs=100, patience=10")
    lines.append("")
    
    # Standard datasets section (H=192)
    lines.append("## Standard Datasets (H=192)")
    lines.append("")
    lines.append("| Dataset | Model | Seeds | MSE (mean±std) | MAE (mean±std) | Status |")
    lines.append("|---------|-------|-------|----------------|----------------|--------|")
    
    for dataset in STANDARD_DATASETS:
        for model in TOP_MODELS:
            results = get_multiseed_results(experiments, model, dataset, 192)
            stats = compute_multiseed_stats(results)
            
            if stats and stats['count'] > 0:
                status = "✅ Complete" if stats['count'] >= 3 else f"⚠️ {stats['count']}/3"
                mse_str = f"{stats['mse_mean']:.4f}±{stats['mse_std']:.4f}"
                mae_str = f"{stats['mae_mean']:.4f}±{stats['mae_std']:.4f}"
                lines.append(f"| {dataset} | {model} | {stats['count']}/3 | {mse_str} | {mae_str} | {status} |")
            else:
                lines.append(f"| {dataset} | {model} | 0/3 | - | - | ❌ Missing |")
    
    lines.append("")
    
    # Illness dataset section (H=24, 48, 60)
    lines.append("## Illness Dataset (H=24, 48, 60)")
    lines.append("")
    lines.append("| Horizon | Model | Seeds | MSE (mean±std) | MAE (mean±std) | Status |")
    lines.append("|---------|-------|-------|----------------|----------------|--------|")
    
    for h in ILLNESS_HORIZONS:
        for model in TOP_MODELS:
            results = get_multiseed_results(experiments, model, 'Illness', h)
            stats = compute_multiseed_stats(results)
            
            if stats and stats['count'] > 0:
                status = "✅ Complete" if stats['count'] >= 3 else f"⚠️ {stats['count']}/3"
                mse_str = f"{stats['mse_mean']:.4f}±{stats['mse_std']:.4f}"
                mae_str = f"{stats['mae_mean']:.4f}±{stats['mae_std']:.4f}"
                lines.append(f"| H={h} | {model} | {stats['count']}/3 | {mse_str} | {mae_str} | {status} |")
            else:
                lines.append(f"| H={h} | {model} | 0/3 | - | - | ❌ Missing |")
    
    lines.append("")
    
    # Coverage summary
    lines.append("## Multi-Seed Coverage Summary")
    lines.append("")
    lines.append("| Model | Standard (6 datasets × H=192) | Illness (3 horizons) | Total | Status |")
    lines.append("|-------|-------------------------------|----------------------|-------|--------|")
    
    for model in TOP_MODELS:
        standard_count = 0
        illness_count = 0
        
        for dataset in STANDARD_DATASETS:
            results = get_multiseed_results(experiments, model, dataset, 192)
            if results and len(results) >= 3:
                standard_count += 1
        
        for h in ILLNESS_HORIZONS:
            results = get_multiseed_results(experiments, model, 'Illness', h)
            if results and len(results) >= 3:
                illness_count += 1
        
        total = standard_count + illness_count
        status = "✅ COMPLETE" if total >= 9 else f"⚠️ {total}/9"
        lines.append(f"| {model} | {standard_count}/6 | {illness_count}/3 | {total}/9 | {status} |")
    
    lines.append("")
    
    # Overall coverage
    total_configs = 0
    complete_configs = 0
    
    for model in TOP_MODELS:
        for dataset in STANDARD_DATASETS:
            total_configs += 1
            results = get_multiseed_results(experiments, model, dataset, 192)
            if results and len(results) >= 3:
                complete_configs += 1
        
        for h in ILLNESS_HORIZONS:
            total_configs += 1
            results = get_multiseed_results(experiments, model, 'Illness', h)
            if results and len(results) >= 3:
                complete_configs += 1
    
    coverage_pct = complete_configs / total_configs * 100 if total_configs > 0 else 0
    lines.append(f"**Overall Multi-Seed Coverage:** {complete_configs}/{total_configs} configurations complete ({coverage_pct:.1f}%)")
    lines.append("")
    
    return lines

def main():
    print("=" * 60)
    print("Rebuilding results_23-01.csv and results_23-01.md")
    print("=" * 60)
    
    # Parse all experiments
    print("\nParsing result file...")
    experiments = parse_result_file()
    print(f"Found {len(experiments)} total experiment results")
    
    # Count by type
    single_seed = [e for e in experiments if not e['is_multiseed']]
    multi_seed = [e for e in experiments if e['is_multiseed']]
    print(f"  - Single-seed: {len(single_seed)}")
    print(f"  - Multi-seed: {len(multi_seed)}")
    
    # Build CSV
    print("\nBuilding CSV...")
    build_csv(experiments)
    
    # Build Markdown
    print("\nBuilding Markdown...")
    build_markdown(experiments)
    
    print("\n" + "=" * 60)
    print("✅ DONE! Both files have been updated.")
    print("=" * 60)

if __name__ == '__main__':
    main()
