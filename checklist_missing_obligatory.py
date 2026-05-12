"""
Generate checklist of missing experiments for OBLIGATORY phases
(Phases 0, 1, 2, 3, 4A, 9, 10)
Excluding multi-seed for now
"""
import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('results_23-01.csv')

print("="*80)
print("MISSING EXPERIMENTS CHECKLIST - OBLIGATORY PHASES")
print("="*80)
print()
print("Scope: 8 models (excluding Informer due to unreliable results)")
print("Models: DLinear, PatchTST, TiDE, TimeXer, iTransformer, PFB_v0, PFB_v2, BERTOnly")
print("Protocol: seq_len=336, single-seed (2021)")
print()

# Define expected model names (normalized)
expected_models_8 = [
    'DLinear',
    'PatchTST', 
    'TiDE',
    'TimeXer',
    'iTransformer',
    'PatchFusionBERT_v0',
    'PatchFusionBERT_v2',
    'PatchFusionBERT_BERTOnly'
]

# Model name normalization mapping
model_name_map = {
    'PFB': 'PatchFusionBERT_v0',  # Assume PFB is v0
    'BERTOnly': 'PatchFusionBERT_BERTOnly',
    'PatchFusionBERT': 'PatchFusionBERT_v0',
}

# Normalize model names in dataframe
df['ModelNorm'] = df['Model'].map(lambda x: model_name_map.get(x, x))

# Filter to seq_len=336 only (protocol requirement)
df_336 = df[df['SeqLen'] == 336].copy()

print(f"Total experiments in CSV: {len(df)}")
print(f"Experiments with seq_len=336: {len(df_336)}")
print()

# Define expected configurations
standard_datasets = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather']
standard_horizons = [96, 192, 336]
illness_horizons = [24, 48, 60]

# Expected experiments
expected_experiments = []

# Standard datasets
for dataset in standard_datasets:
    for horizon in standard_horizons:
        for model in expected_models_8:
            expected_experiments.append((dataset, horizon, model))

# Illness dataset  
for horizon in illness_horizons:
    for model in expected_models_8:
        expected_experiments.append(('Illness', horizon, model))

total_expected = len(expected_experiments)
print(f"Expected total experiments (8-model scope): {total_expected}")
print(f"  - Standard datasets: {len(standard_datasets)} × {len(standard_horizons)} horizons × {len(expected_models_8)} models = {len(standard_datasets) * len(standard_horizons) * len(expected_models_8)}")
print(f"  - Illness dataset: {len(illness_horizons)} horizons × {len(expected_models_8)} models = {len(illness_horizons) * len(expected_models_8)}")
print()

# Check which experiments exist
existing_experiments = set()
for _, row in df_336.iterrows():
    existing_experiments.add((row['Dataset'], row['Horizon'], row['ModelNorm']))

# Find missing experiments
missing_experiments = []
for exp in expected_experiments:
    if exp not in existing_experiments:
        missing_experiments.append(exp)

completed = total_expected - len(missing_experiments)
print(f"✅ COMPLETED: {completed}/{total_expected} ({100*completed/total_expected:.1f}%)")
print(f"❌ MISSING: {len(missing_experiments)}/{total_expected} ({100*len(missing_experiments)/total_expected:.1f}%)")
print()

if len(missing_experiments) == 0:
    print("🎉 ALL OBLIGATORY EXPERIMENTS COMPLETE!")
else:
    # Organize missing by priority
    print("="*80)
    print("MISSING EXPERIMENTS CHECKLIST")
    print("="*80)
    print()
    
    # Group by dataset and horizon
    missing_by_dataset = {}
    for dataset, horizon, model in missing_experiments:
        key = (dataset, horizon)
        if key not in missing_by_dataset:
            missing_by_dataset[key] = []
        missing_by_dataset[key].append(model)
    
    # Count by model
    missing_by_model = {}
    for dataset, horizon, model in missing_experiments:
        if model not in missing_by_model:
            missing_by_model[model] = 0
        missing_by_model[model] += 1
    
    # Print summary by model
    print("MISSING BY MODEL:")
    print("-" * 80)
    for model in sorted(missing_by_model.keys(), key=lambda x: missing_by_model[x], reverse=True):
        count = missing_by_model[model]
        pct = 100 * count / len(missing_experiments)
        print(f"  {model:30s}: {count:3d} missing ({pct:5.1f}% of total missing)")
    print()
    
    # Print detailed checklist by dataset
    print("DETAILED CHECKLIST BY DATASET:")
    print("="*80)
    
    for dataset in standard_datasets + ['Illness']:
        dataset_missing = [(h, m) for d, h, m in missing_experiments if d == dataset]
        if dataset_missing:
            print(f"\n📊 {dataset.upper()}")
            print("-" * 80)
            
            # Group by horizon
            by_horizon = {}
            for horizon, model in dataset_missing:
                if horizon not in by_horizon:
                    by_horizon[horizon] = []
                by_horizon[horizon].append(model)
            
            for horizon in sorted(by_horizon.keys()):
                models = by_horizon[horizon]
                print(f"\n  H={horizon} ({len(models)} missing):")
                for i, model in enumerate(sorted(models), 1):
                    print(f"    ❌ [{i}] {model}")
    
    # Create prioritized action list
    print("\n" + "="*80)
    print("PRIORITIZED ACTION LIST")
    print("="*80)
    print()
    
    # Priority 1: Phase 4A critical - BERTOnly for ablation
    bertonly_missing = [(d, h) for d, h, m in missing_experiments if m == 'PatchFusionBERT_BERTOnly']
    if bertonly_missing:
        print("🔴 PRIORITY 1: Phase 4A Ablation (BERTOnly missing)")
        print("-" * 80)
        print(f"Count: {len(bertonly_missing)} experiments")
        print("Critical: Needed to validate fusion advantage")
        for dataset, horizon in sorted(bertonly_missing):
            print(f"  ❌ BERTOnly - {dataset} - H={horizon}")
        print()
    
    # Priority 2: Fusion models (PFB_v0, PFB_v2) - core contribution
    fusion_missing = [(d, h, m) for d, h, m in missing_experiments 
                      if m in ['PatchFusionBERT_v0', 'PatchFusionBERT_v2']]
    if fusion_missing:
        print("🟠 PRIORITY 2: Fusion Models (Core Contribution)")
        print("-" * 80)
        print(f"Count: {len(fusion_missing)} experiments")
        for dataset, horizon, model in sorted(fusion_missing):
            print(f"  ❌ {model} - {dataset} - H={horizon}")
        print()
    
    # Priority 3: Modern baselines (TiDE, TimeXer, iTransformer)
    modern_missing = [(d, h, m) for d, h, m in missing_experiments 
                      if m in ['TiDE', 'TimeXer', 'iTransformer']]
    if modern_missing:
        print("🟡 PRIORITY 3: Modern Baselines")
        print("-" * 80)
        print(f"Count: {len(modern_missing)} experiments")
        for dataset, horizon, model in sorted(modern_missing):
            print(f"  ❌ {model} - {dataset} - H={horizon}")
        print()
    
    # Priority 4: Standard baselines (DLinear, PatchTST)
    standard_missing = [(d, h, m) for d, h, m in missing_experiments 
                        if m in ['DLinear', 'PatchTST']]
    if standard_missing:
        print("🟢 PRIORITY 4: Standard Baselines")
        print("-" * 80)
        print(f"Count: {len(standard_missing)} experiments")
        for dataset, horizon, model in sorted(standard_missing):
            print(f"  ❌ {model} - {dataset} - H={horizon}")
        print()
    
    # Generate batch file template
    print("="*80)
    print("BATCH FILE GENERATION SUGGESTION")
    print("="*80)
    print()
    print(f"Create batch file with {len(missing_experiments)} experiments")
    print(f"Estimated time: {len(missing_experiments) * 0.5:.1f} hours (~30 min per experiment)")
    print()
    print("Would you like me to generate a batch file for these missing experiments?")

# Also check Phase 4A ablation coverage specifically
print("\n" + "="*80)
print("PHASE 4A: ABLATION ANALYSIS COVERAGE")
print("="*80)
print()

# For ablation, we need BERTOnly wherever we have PFB_v0 or PFB_v2
fusion_configs = set()
for _, row in df_336.iterrows():
    if row['ModelNorm'] in ['PatchFusionBERT_v0', 'PatchFusionBERT_v2']:
        fusion_configs.add((row['Dataset'], row['Horizon']))

bertonly_configs = set()
for _, row in df_336.iterrows():
    if row['ModelNorm'] == 'PatchFusionBERT_BERTOnly':
        bertonly_configs.add((row['Dataset'], row['Horizon']))

missing_ablation = fusion_configs - bertonly_configs

print(f"Fusion model configurations: {len(fusion_configs)}")
print(f"BERTOnly configurations: {len(bertonly_configs)}")
print(f"Missing BERTOnly for ablation: {len(missing_ablation)}")
print()

if missing_ablation:
    print("❌ INCOMPLETE ABLATION - Missing BERTOnly for:")
    for dataset, horizon in sorted(missing_ablation):
        print(f"  - {dataset} H={horizon}")
else:
    print("✅ ABLATION COMPLETE - BERTOnly exists for all fusion configurations")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"8-model scope (no Informer): {completed}/{total_expected} complete ({100*completed/total_expected:.1f}%)")
print(f"Missing experiments: {len(missing_experiments)}")
print(f"Estimated time to complete: {len(missing_experiments) * 0.5:.1f} hours")
print("="*80)
