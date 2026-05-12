"""
Check ALL Illness experiments in result_long_term_forecast.txt
Including multi-seed experiments and different seq_len values.
"""

import re

def parse_illness_experiments(filepath):
    """Find all Illness experiments regardless of seed or seq_len."""
    
    illness_experiments = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for experiments containing 'Illness' or horizon patterns for illness (24, 48, 60)
        if line.startswith('long_term_forecast_') and ('Illness' in line or '_pl24_' in line or '_pl48_' in line or '_pl60_' in line):
            exp_name = line
            
            # Extract details
            # Get horizon from plXXX
            horizon_match = re.search(r'_pl(\d+)_', exp_name)
            horizon = int(horizon_match.group(1)) if horizon_match else None
            
            # Get seq_len from slXXX
            sl_match = re.search(r'_sl(\d+)_', exp_name)
            seq_len = int(sl_match.group(1)) if sl_match else None
            
            # Check if multi-seed
            is_multiseed = 'MS2021' in exp_name or 'MS2022' in exp_name or 'MS2023' in exp_name
            
            # Get model name
            parts = exp_name.replace('long_term_forecast_', '').split('_')
            model = parts[0]
            if 'PFB' in exp_name:
                if 'v0' in exp_name:
                    model = 'PatchFusionBERT_v0'
                elif 'v2' in exp_name:
                    model = 'PatchFusionBERT_v2'
            elif 'BERTOnly' in exp_name:
                model = 'PatchFusionBERT_BERTOnly'
            
            # Get metrics from next line
            mse, mae = None, None
            if i + 1 < len(lines):
                metrics_line = lines[i + 1].strip()
                mse_match = re.search(r'mse:([\d.]+)', metrics_line)
                mae_match = re.search(r'mae:([\d.]+)', metrics_line)
                
                if mse_match:
                    mse = float(mse_match.group(1))
                if mae_match:
                    mae = float(mae_match.group(1))
            
            illness_experiments.append({
                'exp_name': exp_name,
                'model': model,
                'horizon': horizon,
                'seq_len': seq_len,
                'is_multiseed': is_multiseed,
                'mse': mse,
                'mae': mae
            })
        
        i += 1
    
    return illness_experiments

def main():
    print("=" * 80)
    print("CHECKING ALL ILLNESS EXPERIMENTS")
    print("=" * 80)
    
    experiments = parse_illness_experiments('result_long_term_forecast.txt')
    
    print(f"\nTotal Illness experiments found: {len(experiments)}\n")
    
    # Group by type
    normal_experiments = [e for e in experiments if not e['is_multiseed']]
    multiseed_experiments = [e for e in experiments if e['is_multiseed']]
    
    print(f"Normal experiments (single seed): {len(normal_experiments)}")
    print(f"Multi-seed experiments: {len(multiseed_experiments)}")
    
    # Group normal experiments by seq_len
    print("\n" + "=" * 80)
    print("NORMAL EXPERIMENTS (SINGLE SEED)")
    print("=" * 80)
    
    by_seqlen = {}
    for exp in normal_experiments:
        sl = exp['seq_len']
        if sl not in by_seqlen:
            by_seqlen[sl] = []
        by_seqlen[sl].append(exp)
    
    for seq_len in sorted(by_seqlen.keys()):
        exps = by_seqlen[seq_len]
        print(f"\nseq_len={seq_len} ({len(exps)} experiments):")
        
        # Group by horizon
        by_horizon = {}
        for exp in exps:
            h = exp['horizon']
            if h not in by_horizon:
                by_horizon[h] = []
            by_horizon[h].append(exp)
        
        for horizon in sorted(by_horizon.keys()):
            h_exps = by_horizon[horizon]
            models = [e['model'] for e in h_exps]
            print(f"  Horizon={horizon}: {len(h_exps)} experiments")
            print(f"    Models: {', '.join(sorted(set(models)))}")
    
    # Show multi-seed summary
    print("\n" + "=" * 80)
    print("MULTI-SEED EXPERIMENTS")
    print("=" * 80)
    
    if multiseed_experiments:
        by_seqlen_ms = {}
        for exp in multiseed_experiments:
            sl = exp['seq_len']
            if sl not in by_seqlen_ms:
                by_seqlen_ms[sl] = []
            by_seqlen_ms[sl].append(exp)
        
        for seq_len in sorted(by_seqlen_ms.keys()):
            exps = by_seqlen_ms[seq_len]
            print(f"\nseq_len={seq_len} ({len(exps)} multi-seed experiments):")
            
            # Group by horizon
            by_horizon = {}
            for exp in exps:
                h = exp['horizon']
                if h not in by_horizon:
                    by_horizon[h] = []
                by_horizon[h].append(exp)
            
            for horizon in sorted(by_horizon.keys()):
                h_exps = by_horizon[horizon]
                models = [e['model'] for e in h_exps]
                seeds = ['MS2021' if 'MS2021' in e['exp_name'] else 'MS2022' if 'MS2022' in e['exp_name'] else 'MS2023' for e in h_exps]
                print(f"  Horizon={horizon}: {len(h_exps)} experiments")
                print(f"    Models: {', '.join(sorted(set(models)))}")
                print(f"    Seeds: {', '.join(sorted(set(seeds)))}")
    
    # Summary for seq_len=336 (our target)
    print("\n" + "=" * 80)
    print("FOCUS: seq_len=336 (Our standard protocol)")
    print("=" * 80)
    
    sl336_normal = [e for e in normal_experiments if e['seq_len'] == 336]
    
    if sl336_normal:
        print(f"\nTotal seq_len=336 normal experiments: {len(sl336_normal)}")
        
        # Group by horizon
        by_horizon = {}
        for exp in sl336_normal:
            h = exp['horizon']
            if h not in by_horizon:
                by_horizon[h] = []
            by_horizon[h].append(exp)
        
        for horizon in sorted(by_horizon.keys()):
            h_exps = by_horizon[horizon]
            models = sorted(set([e['model'] for e in h_exps]))
            print(f"\n  Horizon={horizon}: {len(h_exps)}/{8} expected models")
            print(f"    Found: {', '.join(models)}")
            
            # Expected models
            expected = ['DLinear', 'PatchTST', 'TiDE', 'TimeXer', 'iTransformer', 
                       'PatchFusionBERT_v0', 'PatchFusionBERT_v2', 'PatchFusionBERT_BERTOnly']
            missing = [m for m in expected if m not in models]
            if missing:
                print(f"    Missing: {', '.join(missing)}")
            else:
                print(f"    ✅ COMPLETE")
    else:
        print("\n❌ NO seq_len=336 normal experiments found for Illness!")
    
    # Detailed list for seq_len=336
    if sl336_normal:
        print("\n" + "=" * 80)
        print("DETAILED LIST: seq_len=336 Illness experiments")
        print("=" * 80)
        
        for exp in sorted(sl336_normal, key=lambda x: (x['horizon'], x['model'])):
            print(f"\nModel: {exp['model']}")
            print(f"  Horizon: {exp['horizon']}, seq_len: {exp['seq_len']}")
            print(f"  MSE: {exp['mse']:.4f}, MAE: {exp['mae']:.4f}")

if __name__ == '__main__':
    main()
