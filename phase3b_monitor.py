#!/usr/bin/env python3
"""
Phase 3B Progress Monitor
Tracks execution of 72 multi-seed H=96/H=336 experiments
"""
import re
from collections import defaultdict
from datetime import datetime

def parse_results_file():
    """Parse result_long_term_forecast.txt and count Phase 3B experiments"""
    result_file = "result_long_term_forecast.txt"
    
    h96_dlinear = 0
    h96_patchtst = 0
    h336_dlinear = 0
    h336_patchtst = 0
    
    # Pattern to match Phase 3B experiments (H96 or H336 with seed 2021/2022/2023)
    pattern = r"long_term_forecast_(DLinear|PatchTST)_(\w+)_H(96|336)_seed(202[1-3])"
    
    try:
        with open(result_file, 'r') as f:
            content = f.read()
            matches = re.findall(pattern, content)
            
            for model, dataset, horizon, seed in matches:
                if horizon == "96":
                    if model == "DLinear":
                        h96_dlinear += 1
                    elif model == "PatchTST":
                        h96_patchtst += 1
                elif horizon == "336":
                    if model == "DLinear":
                        h336_dlinear += 1
                    elif model == "PatchTST":
                        h336_patchtst += 1
    except FileNotFoundError:
        print("Results file not found!")
        return None
    
    total = h96_dlinear + h96_patchtst + h336_dlinear + h336_patchtst
    
    print(f"\n{'='*60}")
    print(f"PHASE 3B PROGRESS MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"\nExperiments Completed:")
    print(f"  H=96  DLinear:   {h96_dlinear:2d}/18")
    print(f"  H=96  PatchTST:  {h96_patchtst:2d}/18")
    print(f"  H=336 DLinear:   {h336_dlinear:2d}/18")
    print(f"  H=336 PatchTST:  {h336_patchtst:2d}/18")
    print(f"  {'-'*40}")
    print(f"  TOTAL:          {total:2d}/72 ({100*total/72:.1f}%)")
    print(f"\nEstimated Completion:")
    if total > 0 and total < 72:
        # Rough estimate: ~3-5 min per experiment
        remaining = 72 - total
        est_time_min = remaining * 4  # 4 min per experiment average
        print(f"  Remaining: {remaining} experiments (~{est_time_min} minutes)")
    elif total == 72:
        print(f"  ✓ PHASE 3B COMPLETE!")
    else:
        print(f"  Starting up...")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parse_results_file()
