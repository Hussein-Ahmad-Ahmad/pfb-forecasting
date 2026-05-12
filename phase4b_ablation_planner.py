#!/usr/bin/env python3
"""
Phase 4B: Ablation Study Planner
Plans ablation experiments for top-performing models
"""
import pandas as pd
import json

print("="*70)
print("PHASE 4B: ABLATION STUDY PLANNER")
print("="*70)

# Load rankings
ranking = pd.read_csv('phase4_model_ranking.csv', index_col=0)
best_models = pd.read_csv('phase4_best_models_per_config.csv')

print("\nTop 3 Models for Ablation Studies:")
print("-" * 70)
top3 = ranking.head(3)
for i, (model, row) in enumerate(top3.iterrows(), 1):
    print(f"{i}. {model}: MSE={row['MSE_mean_mean']:.6f}, MAE={row['MAE_mean_mean']:.6f}")

# ============================================================================
# Ablation Study 1: Hyperparameter Sensitivity
# ============================================================================
print("\n" + "="*70)
print("ABLATION STUDY 1: Hyperparameter Sensitivity")
print("="*70)

ablation_configs = {
    "PatchTST": {
        "baseline": {
            "d_model": 128,
            "d_ff": 512,
            "e_layers": 3,
            "n_heads": 8,
            "patch_len": 16,
            "stride": 8
        },
        "ablations": [
            {"name": "Small_Model", "d_model": 64, "d_ff": 256},
            {"name": "Large_Model", "d_model": 256, "d_ff": 1024},
            {"name": "Deep_Network", "e_layers": 6},
            {"name": "Shallow_Network", "e_layers": 1},
            {"name": "More_Heads", "n_heads": 16},
            {"name": "Fewer_Heads", "n_heads": 4},
            {"name": "Large_Patch", "patch_len": 32, "stride": 16},
            {"name": "Small_Patch", "patch_len": 8, "stride": 4}
        ]
    },
    "iTransformer": {
        "baseline": {
            "d_model": 128,
            "d_ff": 512,
            "e_layers": 2,
            "n_heads": 8
        },
        "ablations": [
            {"name": "Small_Model", "d_model": 64, "d_ff": 256},
            {"name": "Large_Model", "d_model": 256, "d_ff": 1024},
            {"name": "Deep_Network", "e_layers": 4},
            {"name": "More_Heads", "n_heads": 16},
            {"name": "Fewer_Heads", "n_heads": 4}
        ]
    }
}

print("\nPatchTST Ablation Variants:")
for i, ablation in enumerate(ablation_configs["PatchTST"]["ablations"], 1):
    print(f"  {i}. {ablation['name']}: {ablation}")

print("\niTransformer Ablation Variants:")
for i, ablation in enumerate(ablation_configs["iTransformer"]["ablations"], 1):
    print(f"  {i}. {ablation['name']}: {ablation}")

# Save ablation configs
with open('phase4b_ablation_configs.json', 'w') as f:
    json.dump(ablation_configs, f, indent=2)
print("\n✓ Saved ablation configs to: phase4b_ablation_configs.json")

# ============================================================================
# Ablation Study 2: Component Analysis
# ============================================================================
print("\n" + "="*70)
print("ABLATION STUDY 2: Component Analysis (Architecture Variants)")
print("="*70)

component_studies = {
    "PatchTST": [
        "Remove positional encoding",
        "Replace attention with FFN only",
        "Remove normalization layers",
        "Change activation function (ReLU vs GELU)",
        "Remove residual connections"
    ],
    "General": [
        "Training: Different learning rates (1e-3, 1e-4, 1e-5)",
        "Training: Different batch sizes (16, 32, 64, 128)",
        "Training: With/without learning rate scheduler",
        "Data: Different sequence lengths (168, 336, 512)",
        "Data: Different train/test splits"
    ]
}

print("\nPatchTST Component Ablations:")
for i, study in enumerate(component_studies["PatchTST"], 1):
    print(f"  {i}. {study}")

print("\nGeneral Ablations (applicable to all models):")
for i, study in enumerate(component_studies["General"], 1):
    print(f"  {i}. {study}")

# ============================================================================
# Ablation Study 3: Dataset-Specific Analysis
# ============================================================================
print("\n" + "="*70)
print("ABLATION STUDY 3: Dataset-Specific Performance Analysis")
print("="*70)

# Find which datasets each model performs best on
model_best_datasets = {}
for _, row in best_models.iterrows():
    model = row['Best_Model']
    if model not in model_best_datasets:
        model_best_datasets[model] = []
    model_best_datasets[model].append(f"{row['Dataset']}_H{row['Horizon']}")

print("\nDatasets where each model excels:")
for model, datasets in sorted(model_best_datasets.items()):
    print(f"\n{model}:")
    for ds in datasets:
        print(f"  - {ds}")

# ============================================================================
# Generate Experiment Commands for Ablation
# ============================================================================
print("\n" + "="*70)
print("SAMPLE ABLATION EXPERIMENT COMMANDS")
print("="*70)

print("\nExample: PatchTST with Small_Model variant on ETTm2:")
print("-" * 70)
cmd = """python run.py --task_name long_term_forecast --is_training 1 \\
  --model_id PatchTST_ETTm2_H96_SmallModel_Ablation \\
  --model PatchTST --data ETTm2 --features M \\
  --seq_len 336 --label_len 48 --pred_len 96 \\
  --d_model 64 --d_ff 256 --e_layers 3 --n_heads 8 \\
  --train_epochs 100 --batch_size 32 --learning_rate 0.0001"""
print(cmd)

print("\nExample: iTransformer with Deep_Network variant on custom:")
print("-" * 70)
cmd = """python run.py --task_name long_term_forecast --is_training 1 \\
  --model_id iTransformer_custom_H96_DeepNetwork_Ablation \\
  --model iTransformer --data custom --data_path weather.csv --features M \\
  --seq_len 336 --label_len 48 --pred_len 96 \\
  --d_model 128 --d_ff 512 --e_layers 4 --n_heads 8 \\
  --train_epochs 100 --batch_size 32 --learning_rate 0.0001"""
print(cmd)

# ============================================================================
# Estimation
# ============================================================================
print("\n" + "="*70)
print("ABLATION STUDY SCOPE ESTIMATION")
print("="*70)

# Count total ablation experiments needed
patchtst_variants = len(ablation_configs["PatchTST"]["ablations"])
itransformer_variants = len(ablation_configs["iTransformer"]["ablations"])
datasets_to_test = 3  # Test on 3 key datasets (e.g., ETTm1, ETTm2, custom)
horizons = 1  # Focus on H=96
seeds = 3  # Multi-seed for robustness

total_experiments = (patchtst_variants + itransformer_variants) * datasets_to_test * horizons * seeds

print(f"\nPatchTST variants: {patchtst_variants}")
print(f"iTransformer variants: {itransformer_variants}")
print(f"Datasets to test: {datasets_to_test}")
print(f"Horizons: {horizons} (H=96)")
print(f"Seeds per config: {seeds}")
print(f"\nTotal ablation experiments: {total_experiments}")
print(f"Estimated time: {total_experiments * 4} - {total_experiments * 6} minutes")
print(f"                ({total_experiments * 4 / 60:.1f} - {total_experiments * 6 / 60:.1f} hours)")

# ============================================================================
# Recommendations
# ============================================================================
print("\n" + "="*70)
print("RECOMMENDATIONS FOR PHASE 4B")
print("="*70)

recommendations = """
1. START WITH: Hyperparameter sensitivity on best dataset (ETTm2)
   - Test PatchTST ablations (8 variants × 3 seeds = 24 experiments)
   - Estimated time: ~2 hours

2. THEN: Component analysis on PatchTST
   - Test architectural changes (5 variants × 3 seeds = 15 experiments)
   - Estimated time: ~1 hour

3. FINALLY: Cross-model comparison on multiple datasets
   - Compare top 3 models on all 5 datasets
   - Already done in Phase 3B!

4. OPTIONAL: Training hyperparameter ablation
   - Different learning rates, batch sizes
   - Can be done quickly with existing checkpoints

PRIORITY: Focus on PatchTST since it won 33.3% of configurations
          and iTransformer for diversity (different architecture)
"""

print(recommendations)

print("\n" + "="*70)
print("FILES GENERATED")
print("="*70)
print("✓ phase4b_ablation_configs.json - Hyperparameter configurations")
print("\nReady to proceed with ablation studies!")
print("="*70)
