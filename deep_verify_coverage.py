"""
DEEP VERIFICATION - Double check all experiments
Check raw data, model name variants, and actual coverage
"""
import pandas as pd
import re

# Load results
df = pd.read_csv('results_23-01.csv')

print("="*80)
print("DEEP VERIFICATION - ACTUAL COVERAGE CHECK")
print("="*80)
print()

# First, let's see what model names actually exist
print("STEP 1: ACTUAL MODEL NAMES IN CSV")
print("-"*80)
unique_models = sorted(df['Model'].unique())
print(f"Total unique model names: {len(unique_models)}")
for model in unique_models:
    count = len(df[df['Model'] == model])
    print(f"  {model:40s}: {count:4d} experiments")
print()

# Check for PFB variants specifically
print("="*80)
print("STEP 2: PATCHFUSIONBERT VARIANTS ANALYSIS")
print("-"*80)
pfb_related = df[df['Model'].str.contains('PFB|PatchFusion|Fusion|BERT', case=False, na=False)]
pfb_model_names = sorted(pfb_related['Model'].unique())
print(f"Models containing 'PFB/PatchFusion/BERT': {len(pfb_model_names)}")
for model in pfb_model_names:
    count = len(pfb_related[pfb_related['Model'] == model])
    print(f"  {model:40s}: {count:4d} experiments")
print()

# Show sample experiment names for PFB
print("Sample PFB experiment names:")
pfb_samples = pfb_related['ExperimentName'].head(10)
for i, name in enumerate(pfb_samples, 1):
    print(f"  {i}. {name[:100]}...")
print()

# Check ETT datasets specifically for PFB_v2
print("="*80)
print("STEP 3: ETT DATASETS - CHECKING PFB_v2 COVERAGE")
print("-"*80)

ett_datasets = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2']
horizons = [96, 192, 336]

for dataset in ett_datasets:
    print(f"\n{dataset}:")
    df_ett = df[(df['Dataset'] == dataset) & (df['SeqLen'] == 336)]
    
    for horizon in horizons:
        df_h = df_ett[df_ett['Horizon'] == horizon]
        
        # Check for any PFB variant
        pfb_any = df_h[df_h['Model'].str.contains('PFB|PatchFusion', case=False, na=False)]
        pfb_v2 = df_h[df_h['Model'].str.contains('v2', case=False, na=False)]
        
        print(f"  H={horizon}: Total={len(df_h)}, PFB_any={len(pfb_any)}, PFB_v2={len(pfb_v2)}")
        
        if len(pfb_any) > 0:
            models = pfb_any['Model'].unique()
            print(f"         Models: {', '.join(models)}")

# Now let's look at raw experiment names for ETT + PFB_v2
print("\n" + "="*80)
print("STEP 4: SEARCHING RAW EXPERIMENT NAMES FOR 'v2' IN ETT")
print("-"*80)

ett_df = df[df['Dataset'].isin(ett_datasets)]
ett_v2 = ett_df[ett_df['ExperimentName'].str.contains('v2', case=False)]

print(f"ETT experiments containing 'v2' in name: {len(ett_v2)}")
if len(ett_v2) > 0:
    print("\nSample experiment names:")
    for i, row in ett_v2.head(10).iterrows():
        print(f"  {row['Dataset']} H={row['Horizon']}: {row['ExperimentName'][:80]}...")

# Check the result_long_term_forecast.txt file directly
print("\n" + "="*80)
print("STEP 5: CHECKING RAW TXT FILE FOR PFB_v2 ON ETT")
print("-"*80)

with open('result_long_term_forecast.txt', 'r') as f:
    content = f.read()

# Search for PFB_v2 on ETT datasets
for dataset in ett_datasets:
    pattern = f'PatchFusionBERT_v2.*{dataset}'
    matches = re.findall(pattern, content)
    print(f"{dataset}: Found {len(matches)} PatchFusionBERT_v2 experiments")
    
    if len(matches) > 0:
        print(f"  First match: {matches[0][:80]}...")

# Also check for just "v2" with ETT
print("\nSearching for 'v2' pattern with ETT:")
for dataset in ett_datasets:
    pattern = f'.*v2.*{dataset}.*'
    matches = re.findall(pattern, content, re.IGNORECASE)
    print(f"{dataset}: Found {len(matches)} lines with 'v2'")

# Check model name parsing logic
print("\n" + "="*80)
print("STEP 6: TESTING MODEL NAME PARSING")
print("-"*80)

test_names = [
    'long_term_forecast_PatchFusionBERT_v2_ETTm1_336_96',
    'long_term_forecast_PFB_v2_100epoch_ETTm1_336_96', 
    'PatchFusionBERT_v2_100epoch_PatchFusionBERT_v2_ETTm1_ftM_sl336',
    'long_term_forecast_Exchange_336_96_PatchFusionBERT_v2_custom',
]

print("Testing experiment name parsing:")
for name in test_names:
    name_clean = name.replace('long_term_forecast_', '')
    parts = name_clean.split('_')
    print(f"\nName: {name[:60]}...")
    print(f"  Parts: {parts[:10]}")
    print(f"  First part: '{parts[0] if parts else 'none'}'")
    if len(parts) > 1:
        print(f"  Second part: '{parts[1]}'")

# COMPREHENSIVE COVERAGE CHECK
print("\n" + "="*80)
print("STEP 7: COMPREHENSIVE COVERAGE MATRIX")
print("="*80)
print()

# Expected 8 models
models_8 = ['DLinear', 'PatchTST', 'TiDE', 'TimeXer', 'iTransformer', 
            'PatchFusionBERT_v0', 'PatchFusionBERT_v2', 'PatchFusionBERT_BERTOnly']

# Create mapping for all possible names
model_aliases = {
    'PFB': 'PatchFusionBERT_v0',
    'BERTOnly': 'PatchFusionBERT_BERTOnly',
}

df_336 = df[df['SeqLen'] == 336].copy()

print("COVERAGE BY DATASET × HORIZON:")
print()

all_datasets = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather', 'Illness']

for dataset in all_datasets:
    print(f"\n{dataset.upper()}")
    print("-" * 80)
    
    df_ds = df_336[df_336['Dataset'] == dataset]
    
    if dataset == 'Illness':
        horizons_check = [24, 48, 60]
    else:
        horizons_check = [96, 192, 336]
    
    for horizon in horizons_check:
        df_h = df_ds[df_ds['Horizon'] == horizon]
        
        print(f"H={horizon}: {len(df_h)} total experiments")
        
        # Show all models for this config
        models_found = sorted(df_h['Model'].unique())
        for model in models_found:
            count = len(df_h[df_h['Model'] == model])
            print(f"  ✓ {model:40s} ({count} runs)")

print("\n" + "="*80)
print("FINAL VERDICT - RE-COUNTING WITH BETTER LOGIC")
print("="*80)
