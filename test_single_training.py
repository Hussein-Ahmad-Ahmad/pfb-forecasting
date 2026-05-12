"""
Quick test training to verify pipeline works
Single experiment: PatchTST on ETTm1
"""

import subprocess
import torch

print("="*80)
print("TEST TRAINING - PatchTST on ETTm1")
print("="*80)

# Clear GPU cache
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.cuda.empty_cache()
else:
    print("WARNING: No GPU available!")

# Build command
cmd = [
    'python', 'run.py',
    '--task_name', 'long_term_forecast',
    '--is_training', '1',
    '--root_path', './data/',
    '--data_path', 'ETTm1.csv',
    '--model_id', 'TEST_ETTm1_PatchTST',
    '--model', 'PatchTST',
    '--data', 'ETTm1',
    '--features', 'M',
    '--seq_len', '336',
    '--label_len', '48',
    '--pred_len', '96',
    '--enc_in', '7',
    '--dec_in', '7',
    '--c_out', '7',
    '--d_model', '128',
    '--n_heads', '8',
    '--e_layers', '3',
    '--d_layers', '1',
    '--d_ff', '512',
    '--dropout', '0.1',
    '--batch_size', '16',
    '--learning_rate', '0.0001',
    '--train_epochs', '3',
    '--patience', '3',
    '--des', 'test',
    '--itr', '1',
    '--use_gpu', '1',
    '--gpu', '0'
]

print("\nCommand:")
print(' '.join(cmd))
print("\n" + "="*80)
print("Starting training (3 epochs max)...")
print("="*80 + "\n")

try:
    result = subprocess.run(cmd, text=True, timeout=1800)
    
    if result.returncode == 0:
        print("\n" + "="*80)
        print("✓ TEST TRAINING SUCCESSFUL!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("✗ TEST TRAINING FAILED")
        print(f"Exit code: {result.returncode}")
        print("="*80)
        
except subprocess.TimeoutExpired:
    print("\n" + "="*80)
    print("✗ TEST TIMEOUT (>30 minutes)")
    print("="*80)
    
except Exception as e:
    print("\n" + "="*80)
    print(f"✗ ERROR: {str(e)}")
    print("="*80)
