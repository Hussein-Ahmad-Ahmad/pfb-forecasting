"""
Phase 8: Efficiency Measurement (Latency/VRAM)
Measures inference performance for top 5 models
Fixed setup: batch_size=16, seq_len=336, pred_len=192
"""

import torch
import numpy as np
import time
from models import PatchTST, DLinear, TiDE, PatchFusionBERT_v0, PatchFusionBERT_v2

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
SEQ_LEN = 336
PRED_LEN = 192
ENC_IN = 7  # ETT datasets
WARMUP_RUNS = 10
MEASURE_RUNS = 100

print(f"Device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
print("\n" + "="*80)

def measure_model_efficiency(model_class, model_name, model_kwargs):
    """Measure latency, throughput, and peak VRAM for a model"""
    print(f"\n📊 Measuring {model_name}...")
    
    # Initialize model
    model = model_class(**model_kwargs).to(DEVICE)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create dummy input
    x_enc = torch.randn(BATCH_SIZE, SEQ_LEN, ENC_IN).to(DEVICE)
    x_mark_enc = torch.randn(BATCH_SIZE, SEQ_LEN, 4).to(DEVICE)  # time features
    x_dec = torch.randn(BATCH_SIZE, 48 + PRED_LEN, ENC_IN).to(DEVICE)  # label_len=48
    x_mark_dec = torch.randn(BATCH_SIZE, 48 + PRED_LEN, 4).to(DEVICE)
    
    # Reset VRAM tracking
    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    # Warmup
    print(f"  Warming up ({WARMUP_RUNS} runs)...")
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
    
    # Measure latency
    print(f"  Measuring latency ({MEASURE_RUNS} runs)...")
    latencies = []
    
    with torch.no_grad():
        for _ in range(MEASURE_RUNS):
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
                start = time.perf_counter()
            else:
                start = time.perf_counter()
            
            _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    # Get peak VRAM
    if DEVICE == 'cuda':
        peak_vram_bytes = torch.cuda.max_memory_allocated()
        peak_vram_gb = peak_vram_bytes / (1024**3)
    else:
        peak_vram_gb = 0.0
    
    # Calculate statistics
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    median_latency = np.median(latencies)
    throughput = (BATCH_SIZE * 1000) / mean_latency  # samples/sec
    
    results = {
        'model': model_name,
        'params_total': total_params,
        'params_trainable': trainable_params,
        'latency_mean_ms': mean_latency,
        'latency_std_ms': std_latency,
        'latency_median_ms': median_latency,
        'throughput_samples_sec': throughput,
        'peak_vram_gb': peak_vram_gb
    }
    
    print(f"  ✅ {model_name}:")
    print(f"     Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"     Latency: {mean_latency:.2f} ± {std_latency:.2f} ms")
    print(f"     Throughput: {throughput:.1f} samples/sec")
    print(f"     Peak VRAM: {peak_vram_gb:.2f} GB")
    
    # Clean up
    del model, x_enc, x_mark_enc, x_dec, x_mark_dec
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    
    return results


# Model configurations (matching your experimental setup)
models_to_test = [
    {
        'class': DLinear,
        'name': 'DLinear',
        'kwargs': {
            'configs': type('Args', (), {
                'seq_len': SEQ_LEN,
                'pred_len': PRED_LEN,
                'enc_in': ENC_IN,
                'individual': False
            })()
        }
    },
    {
        'class': PatchTST,
        'name': 'PatchTST',
        'kwargs': {
            'configs': type('Args', (), {
                'seq_len': SEQ_LEN,
                'pred_len': PRED_LEN,
                'enc_in': ENC_IN,
                'c_out': ENC_IN,
                'd_model': 128,
                'n_heads': 16,
                'e_layers': 3,
                'd_ff': 256,
                'dropout': 0.2,
                'fc_dropout': 0.2,
                'head_dropout': 0.0,
                'patch_len': 16,
                'stride': 8,
                'activation': 'gelu',
                'output_attention': False
            })()
        }
    },
    {
        'class': TiDE,
        'name': 'TiDE',
        'kwargs': {
            'configs': type('Args', (), {
                'seq_len': SEQ_LEN,
                'pred_len': PRED_LEN,
                'enc_in': ENC_IN,
                'c_out': ENC_IN,
                'num_layers': 2,
                'hidden_dim': 256,
                'decoder_output_dim': 8,
                'temporal_decoder_dim': 32,
                'dropout': 0.3
            })()
        }
    },
    {
        'class': PatchFusionBERT_v0,
        'name': 'PFB_v0',
        'kwargs': {
            'configs': type('Args', (), {
                'seq_len': SEQ_LEN,
                'pred_len': PRED_LEN,
                'enc_in': ENC_IN,
                'c_out': ENC_IN,
                'd_model': 128,
                'n_heads': 8,
                'e_layers': 2,
                'd_ff': 256,
                'dropout': 0.1,
                'patch_len': 16,
                'stride': 8,
                'activation': 'gelu'
            })()
        }
    },
    {
        'class': PatchFusionBERT_v2,
        'name': 'PFB_v2',
        'kwargs': {
            'configs': type('Args', (), {
                'seq_len': SEQ_LEN,
                'pred_len': PRED_LEN,
                'enc_in': ENC_IN,
                'c_out': ENC_IN,
                'd_model': 128,
                'n_heads': 8,
                'e_layers': 2,
                'd_ff': 256,
                'dropout': 0.1,
                'patch_len': 16,
                'stride': 8,
                'activation': 'gelu'
            })()
        }
    }
]

# Run measurements
all_results = []

for model_config in models_to_test:
    try:
        results = measure_model_efficiency(
            model_config['class'],
            model_config['name'],
            model_config['kwargs']
        )
        all_results.append(results)
    except Exception as e:
        print(f"  ❌ Error measuring {model_config['name']}: {e}")
        continue

# Print summary table
print("\n" + "="*80)
print("\n📋 EFFICIENCY SUMMARY TABLE\n")
print(f"{'Model':<12} {'Params(M)':<12} {'Latency(ms)':<15} {'Throughput':<15} {'VRAM(GB)':<10}")
print("-" * 80)

for r in all_results:
    print(f"{r['model']:<12} "
          f"{r['params_total']/1e6:>10.2f}  "
          f"{r['latency_mean_ms']:>8.2f}±{r['latency_std_ms']:>4.2f}  "
          f"{r['throughput_samples_sec']:>12.1f}  "
          f"{r['peak_vram_gb']:>8.2f}")

# Save to CSV
import csv
output_file = 'results_efficiency.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)

print(f"\n✅ Results saved to: {output_file}")

# Generate LaTeX table
latex_output = """
\\begin{table}[t]
\\centering
\\caption{Inference Efficiency Comparison (batch\\_size=16, seq\\_len=336, pred\\_len=192)}
\\label{tab:efficiency}
\\scriptsize
\\begin{tabular}{lrrrr}
\\toprule
\\textbf{Model} & \\textbf{Params(M)} & \\textbf{Latency(ms)} & \\textbf{Throughput} & \\textbf{VRAM(GB)} \\\\
\\midrule
"""

for r in all_results:
    latex_output += f"{r['model']} & {r['params_total']/1e6:.2f} & {r['latency_mean_ms']:.2f}$\\pm${r['latency_std_ms']:.2f} & {r['throughput_samples_sec']:.1f} & {r['peak_vram_gb']:.2f} \\\\\n"

latex_output += """\\bottomrule
\\end{tabular}
\\end{table}
"""

latex_file = 'efficiency_table.tex'
with open(latex_file, 'w') as f:
    f.write(latex_output)

print(f"✅ LaTeX table saved to: {latex_file}")
print("\n" + "="*80)
print("Phase 8 (Efficiency) COMPLETE ✅")
