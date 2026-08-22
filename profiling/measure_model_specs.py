"""
Measure Model Parameters and Inference Time
"""

import torch
import sys
import os
import time
import numpy as np
from argparse import Namespace

# Add models to path
sys.path.append('.')

def count_parameters(model):
    """Count trainable parameters in millions"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000

def measure_inference_time(model, input_shape, device='cuda', iterations=100):
    """Measure average inference time in milliseconds"""
    model.eval()
    model.to(device)
    
    # Create dummy inputs
    batch_x = torch.randn(input_shape[0], input_shape[1], input_shape[2]).to(device)
    batch_x_mark = torch.randn(input_shape[0], input_shape[1], 4).to(device)  # Assuming 4 time features
    dec_inp = torch.randn(input_shape[0], 48 + 96, input_shape[2]).to(device)  # label_len + pred_len
    batch_y_mark = torch.randn(input_shape[0], 48 + 96, 4).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
    # Measure
    if device == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.time()
            _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)  # Convert to ms
    
    return np.mean(times), np.std(times)

# Configuration for models
configs = Namespace(
    task_name='long_term_forecast',
    seq_len=336,
    label_len=48,
    pred_len=96,
    enc_in=7,  # ETTm1/ETTh1 have 7 features
    dec_in=7,
    c_out=7,
    d_model=128,
    n_heads=8,
    e_layers=3,
    d_layers=1,
    d_ff=512,
    dropout=0.1,
    embed='timeF',
    freq='h',
    activation='gelu',
    output_attention=False,
    patch_len=16,
    stride=8,
    top_k=5,
    num_kernels=6,
    factor=1,
    moving_avg=25,
    distil=True
)

print("=" * 80)
print("MEASURING MODEL PARAMETERS AND INFERENCE TIME")
print("=" * 80)

results = {}

# Test models
models_to_test = {
    'PatchFusionBERT_v0': 'PatchFusionBERT_v0',
    'PatchFusionBERT_v2': 'PatchFusionBERT_v2',
    'PatchFusionBERT_BERTOnly': 'PatchFusionBERT_BERTOnly',
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")
print(f"Input shape: (batch=16, seq_len=336, features=7)\n")

for model_name, module_name in models_to_test.items():
    print(f"Testing {model_name}...")
    
    try:
        # Import model
        exec(f"from models import {module_name}")
        model_class = eval(f"{module_name}.Model")
        
        # Create model
        model = model_class(configs)
        
        # Count parameters
        params_M = count_parameters(model)
        print(f"  Parameters: {params_M:.2f}M")
        
        # Measure inference time
        input_shape = (16, 336, 7)  # batch, seq_len, features
        
        try:
            avg_time, std_time = measure_inference_time(model, input_shape, device=device, iterations=50)
            print(f"  Inference time: {avg_time:.2f} ± {std_time:.2f} ms")
            
            results[model_name] = {
                'params_M': params_M,
                'inference_time_ms': avg_time,
                'inference_std_ms': std_time
            }
        except Exception as e:
            print(f"  ⚠ Error measuring inference time: {e}")
            results[model_name] = {
                'params_M': params_M,
                'inference_time_ms': None,
                'inference_std_ms': None
            }
        
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        continue

# Save results
import json

output_file = './analysis_results/model_specs.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {output_file}")

# Create summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Model':<30} {'Parameters (M)':<20} {'Inference Time (ms)':<25}")
print("-" * 80)

for model_name, data in results.items():
    params = f"{data['params_M']:.2f}"
    if data['inference_time_ms'] is not None:
        inf_time = f"{data['inference_time_ms']:.2f} ± {data['inference_std_ms']:.2f}"
    else:
        inf_time = "N/A"
    
    print(f"{model_name:<30} {params:<20} {inf_time:<25}")

print("=" * 80)
