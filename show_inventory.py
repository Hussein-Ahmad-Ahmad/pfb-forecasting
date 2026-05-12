"""
Show exactly what we have: Models, Horizons, Datasets
"""
import pandas as pd
import numpy as np

df = pd.read_csv('results_23-01.csv')

# Filter to seq_len=336 (our protocol)
df_336 = df[df['SeqLen'] == 336].copy()

print("="*80)
print("COMPLETE INVENTORY - WHAT WE HAVE")
print("="*80)
print(f"\nTotal experiments: {len(df_336)} (seq_len=336)")
print()

# ============================================================================
# DATASETS
# ============================================================================
print("="*80)
print("1. DATASETS")
print("="*80)
datasets = sorted(df_336['Dataset'].unique())
print(f"\nTotal: {len(datasets)} datasets\n")
for i, ds in enumerate(datasets, 1):
    count = len(df_336[df_336['Dataset'] == ds])
    horizons = sorted(df_336[df_336['Dataset'] == ds]['Horizon'].unique())
    print(f"{i:2d}. {ds:15s}: {count:3d} experiments, Horizons={horizons}")

# ============================================================================
# MODELS
# ============================================================================
print("\n" + "="*80)
print("2. MODELS")
print("="*80)

# Categorize models
fusion_models = ['PatchFusionBERT_v0', 'PatchFusionBERT_v2', 'PatchFusionBERT_BERTOnly']
modern_models = ['TiDE', 'TimeXer', 'iTransformer']
standard_models = ['DLinear', 'PatchTST']
legacy_models = ['Autoformer', 'Informer', 'Transformer']

all_models = sorted(df_336['Model'].unique())
print(f"\nTotal unique model names: {len(all_models)}\n")

print("FUSION MODELS (Our contribution):")
for model in all_models:
    if any(fm in model for fm in fusion_models):
        count = len(df_336[df_336['Model'] == model])
        print(f"  ✓ {model:35s}: {count:3d} experiments")

print("\nMODERN BASELINES:")
for model in modern_models:
    if model in all_models:
        count = len(df_336[df_336['Model'] == model])
        print(f"  ✓ {model:35s}: {count:3d} experiments")

print("\nSTANDARD BASELINES:")
for model in standard_models:
    if model in all_models:
        count = len(df_336[df_336['Model'] == model])
        print(f"  ✓ {model:35s}: {count:3d} experiments")

print("\nLEGACY BASELINES:")
for model in legacy_models:
    if model in all_models:
        count = len(df_336[df_336['Model'] == model])
        print(f"  ✓ {model:35s}: {count:3d} experiments")

print("\nOTHER (likely parsing errors - dataset names as models):")
other_models = [m for m in all_models if m not in fusion_models + modern_models + standard_models + legacy_models]
for model in other_models:
    count = len(df_336[df_336['Model'] == model])
    print(f"  ⚠️  {model:35s}: {count:3d} experiments (PARSING ERROR?)")

# ============================================================================
# COVERAGE MATRIX
# ============================================================================
print("\n" + "="*80)
print("3. DETAILED COVERAGE MATRIX BY DATASET × HORIZON")
print("="*80)

# Focus on the 8 expected models
expected_8_models = [
    'DLinear',
    'PatchTST',
    'TiDE',
    'TimeXer',
    'iTransformer',
    'PatchFusionBERT_v0',
    'PatchFusionBERT_v2',
    'PatchFusionBERT_BERTOnly'
]

standard_datasets = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather']
standard_horizons = [96, 192, 336]

for dataset in standard_datasets:
    print(f"\n{'='*80}")
    print(f"{dataset.upper()}")
    print(f"{'='*80}")
    
    df_ds = df_336[df_336['Dataset'] == dataset]
    
    for horizon in standard_horizons:
        df_h = df_ds[df_ds['Horizon'] == horizon]
        
        print(f"\n  H={horizon}:")
        print(f"  {'-'*76}")
        
        found_models = []
        missing_models = []
        
        for model in expected_8_models:
            if model in df_h['Model'].values:
                count = len(df_h[df_h['Model'] == model])
                found_models.append(model)
                print(f"    ✓ {model:35s} ({count} runs)")
            else:
                missing_models.append(model)
        
        if missing_models:
            print(f"\n    MISSING ({len(missing_models)}):")
            for model in missing_models:
                print(f"      ❌ {model}")
        
        print(f"\n  Coverage: {len(found_models)}/8 models")

# Illness separately
print(f"\n{'='*80}")
print(f"ILLNESS")
print(f"{'='*80}")

df_illness = df_336[df_336['Dataset'] == 'Illness']
illness_horizons = sorted(df_illness['Horizon'].unique())

print(f"\nHorizons found: {illness_horizons}")

for horizon in [24, 48, 60]:
    df_h = df_illness[df_illness['Horizon'] == horizon]
    
    print(f"\n  H={horizon}:")
    print(f"  {'-'*76}")
    
    if len(df_h) == 0:
        print(f"    ❌ NO EXPERIMENTS")
        print(f"    Missing: All 8 models")
    else:
        found_models = []
        missing_models = []
        
        for model in expected_8_models:
            if model in df_h['Model'].values:
                count = len(df_h[df_h['Model'] == model])
                found_models.append(model)
                print(f"    ✓ {model:35s} ({count} runs)")
            else:
                missing_models.append(model)
        
        if missing_models:
            print(f"\n    MISSING ({len(missing_models)}):")
            for model in missing_models:
                print(f"      ❌ {model}")
        
        print(f"\n  Coverage: {len(found_models)}/8 models")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "="*80)
print("4. SUMMARY TABLE - MODEL × DATASET COVERAGE")
print("="*80)
print()

summary_data = []
for model in expected_8_models:
    row = {'Model': model}
    for dataset in standard_datasets + ['Illness']:
        count = len(df_336[(df_336['Model'] == model) & (df_336['Dataset'] == dataset)])
        row[dataset] = count
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("5. QUICK FACTS")
print("="*80)
print()
print(f"✓ Total datasets: {len(datasets)}")
print(f"✓ Standard datasets: {len(standard_datasets)}")
print(f"✓ Illness dataset: 1")
print()
print(f"✓ Models (expected 8-model scope): {len(expected_8_models)}")
print(f"✓ Fusion models (our contribution): 3")
print(f"✓ Modern baselines: 3")
print(f"✓ Standard baselines: 2")
print()
print(f"✓ Standard horizons: {standard_horizons}")
print(f"✓ Illness horizons: [24, 48, 60]")
print()

# Calculate overall coverage
total_expected = len(standard_datasets) * len(standard_horizons) * len(expected_8_models) + 3 * len(expected_8_models)
total_found = len([1 for model in expected_8_models for ds in standard_datasets + ['Illness'] 
                   for h in (standard_horizons if ds != 'Illness' else [24, 48, 60])
                   if len(df_336[(df_336['Model'] == model) & (df_336['Dataset'] == ds) & (df_336['Horizon'] == h)]) > 0])

print(f"Overall coverage: {total_found}/{total_expected} ({100*total_found/total_expected:.1f}%)")
print()
print("="*80)
