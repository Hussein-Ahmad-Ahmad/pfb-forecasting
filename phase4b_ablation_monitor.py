#!/usr/bin/env python3
"""
Phase 4B Ablation Progress Monitor
Tracks PatchTST ablation experiments on ETTm2
"""
import re
from datetime import datetime

def parse_ablation_results():
    """Parse result_long_term_forecast.txt for ablation experiments"""
    result_file = "result_long_term_forecast.txt"
    
    # Pattern for ablation experiments
    pattern = r"long_term_forecast_PatchTST_ETTm2_H96_(\w+)_seed(202[1-3])_Ablation"
    
    variants = {
        'Small_Model': [0, 0, 0],  # seeds 2021, 2022, 2023
        'Large_Model': [0, 0, 0],
        'Deep_Network': [0, 0, 0],
        'Shallow_Network': [0, 0, 0],
        'More_Heads': [0, 0, 0],
        'Fewer_Heads': [0, 0, 0],
        'Large_Patch': [0, 0, 0],
        'Small_Patch': [0, 0, 0]
    }
    
    try:
        with open(result_file, 'r') as f:
            content = f.read()
            matches = re.findall(pattern, content)
            
            for variant, seed in matches:
                if variant in variants:
                    seed_idx = int(seed) - 2021
                    variants[variant][seed_idx] = 1
    except FileNotFoundError:
        print("Results file not found!")
        return None
    
    total = 0
    print(f"\n{'='*70}")
    print(f"PHASE 4B ABLATION PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"\nPatchTST Variants on ETTm2 H=96:")
    print(f"{'Variant':<20} {'2021':<6} {'2022':<6} {'2023':<6} {'Total':<6}")
    print(f"{'-'*50}")
    
    for variant, seeds in variants.items():
        completed = sum(seeds)
        total += completed
        status = ['✓' if s else '✗' for s in seeds]
        print(f"{variant:<20} {status[0]:<6} {status[1]:<6} {status[2]:<6} {completed}/3")
    
    print(f"{'-'*50}")
    print(f"{'TOTAL':<20} {'':<6} {'':<6} {'':<6} {total}/24")
    
    pct = (total / 24) * 100
    print(f"\nProgress: {pct:.1f}% complete")
    
    if total < 24:
        remaining = 24 - total
        est_time = remaining * 4  # 4 min per experiment
        print(f"Remaining: {remaining} experiments (~{est_time} minutes)")
    else:
        print(f"\n✓ PHASE 4B ABLATION COMPLETE!")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    parse_ablation_results()
