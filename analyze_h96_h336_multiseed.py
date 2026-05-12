"""
Analyze multi-seed H=96 and H=336 results.
Tag in results file: des=multiseed_h96_h336
Output: results_analysis/multiseed_h96_h336_summary.md + .csv
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_FILE = 'result_long_term_forecast.txt'
OUT_DIR = Path('results_analysis')
OUT_DIR.mkdir(exist_ok=True)

TAG = 'multiseed_h96_h336'

SEEDS = [2021, 2022, 2023, 2024, 2025]
MODELS = ['PatchFusionBERT_v0', 'PatchFusionBERT_v2', 'PatchTST', 'DLinear']
DATASETS = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather']
HORIZONS = [96, 336]

MODEL_ALIASES = {
    'PFBv0': 'PatchFusionBERT_v0',
    'PFBv2': 'PatchFusionBERT_v2',
}

def parse_results(filename):
    records = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        name_line = lines[i].strip()
        i += 1
        if i >= len(lines):
            break
        metrics_line = lines[i].strip()
        i += 1

        if TAG not in name_line:
            continue

        # Extract MSE and MAE
        mse_match = re.search(r'mse:([\d.]+)', metrics_line)
        mae_match = re.search(r'mae:([\d.]+)', metrics_line)
        if not mse_match or not mae_match:
            continue
        mse = float(mse_match.group(1))
        mae = float(mae_match.group(1))

        # Extract seed from MS#### pattern
        seed_match = re.search(r'MS(\d{4})', name_line)
        if not seed_match:
            continue
        seed = int(seed_match.group(1))

        # Extract horizon (pred_len) from pl#### pattern
        pl_match = re.search(r'_pl(\d+)_', name_line)
        if not pl_match:
            continue
        horizon = int(pl_match.group(1))
        if horizon not in HORIZONS:
            continue

        # Extract model — try model_id pattern first: PFBv0_MS... or PFBv2_MS... or PatchTST_MS... or DLinear_MS...
        model = None
        for alias, full in MODEL_ALIASES.items():
            if f'_{alias}_MS' in name_line:
                model = full
                break
        if model is None:
            for m in ['PatchTST', 'DLinear']:
                if f'_{m}_MS' in name_line:
                    model = m
                    break
        if model is None:
            continue

        # Extract dataset — look for known dataset names after MS####_
        dataset = None
        ms_pos = name_line.find(f'MS{seed}')
        after_ms = name_line[ms_pos + len(f'MS{seed}') + 1:]  # skip underscore
        for ds in DATASETS:
            if after_ms.startswith(ds):
                dataset = ds
                break
        if dataset is None:
            continue

        records.append({
            'dataset': dataset,
            'horizon': horizon,
            'model': model,
            'seed': seed,
            'mse': mse,
            'mae': mae,
        })

    return pd.DataFrame(records)


def build_summary(df):
    rows = []
    for horizon in HORIZONS:
        for dataset in DATASETS:
            for model in MODELS:
                sub = df[(df['dataset'] == dataset) &
                         (df['horizon'] == horizon) &
                         (df['model'] == model)]
                n = len(sub)
                if n == 0:
                    rows.append({'dataset': dataset, 'horizon': horizon, 'model': model,
                                 'seeds': 0, 'mse': None, 'mae': None, 'status': 'MISSING'})
                    continue
                mse_mean = sub['mse'].mean()
                mse_std  = sub['mse'].std(ddof=1) if n > 1 else 0.0
                mae_mean = sub['mae'].mean()
                mae_std  = sub['mae'].std(ddof=1) if n > 1 else 0.0
                rows.append({
                    'dataset': dataset,
                    'horizon': horizon,
                    'model': model,
                    'seeds': n,
                    'mse': f'{mse_mean:.4f}±{mse_std:.4f}',
                    'mae': f'{mae_mean:.4f}±{mae_std:.4f}',
                    'mse_mean': mse_mean,
                    'mae_mean': mae_mean,
                    'mse_std': mse_std,
                    'mae_std': mae_std,
                    'status': 'COMPLETE' if n == 5 else f'PARTIAL({n}/5)',
                })
    return pd.DataFrame(rows)


def write_markdown(summary_df, raw_df):
    lines = []
    lines.append('# Multi-Seed H=96 and H=336 Results')
    lines.append(f'\nSeeds: {SEEDS}')
    lines.append(f'Models: {", ".join(MODELS)}')
    lines.append(f'Datasets: {", ".join(DATASETS)}')
    lines.append(f'Horizons: H=96, H=336\n')

    total = len(MODELS) * len(DATASETS) * len(HORIZONS)
    complete = (summary_df['status'] == 'COMPLETE').sum()
    lines.append(f'Coverage: {complete}/{total} configurations complete ({100*complete/total:.1f}%)\n')

    for horizon in HORIZONS:
        lines.append(f'## H={horizon}\n')
        sub = summary_df[summary_df['horizon'] == horizon]
        lines.append('| Dataset | Model | Seeds | MSE | MAE | Status |')
        lines.append('|---------|-------|-------|-----|-----|--------|')
        for dataset in DATASETS:
            for model in MODELS:
                row = sub[(sub['dataset'] == dataset) & (sub['model'] == model)]
                if row.empty:
                    lines.append(f'| {dataset} | {model} | - | - | - | MISSING |')
                    continue
                r = row.iloc[0]
                lines.append(f'| {dataset} | {model} | {r["seeds"]}/5 | {r["mse"]} | {r["mae"]} | {r["status"]} |')
        lines.append('')

    # Best model per config
    lines.append('## Best Model Per Config\n')
    lines.append('| Dataset | Horizon | Best Model | Best MSE | vs DLinear ΔMSE | vs PatchTST ΔMSE |')
    lines.append('|---------|---------|------------|----------|-----------------|-----------------|')

    for horizon in HORIZONS:
        for dataset in DATASETS:
            sub = summary_df[(summary_df['horizon'] == horizon) &
                             (summary_df['dataset'] == dataset) &
                             (summary_df['status'] == 'COMPLETE')]
            if sub.empty:
                continue
            best_idx = sub['mse_mean'].idxmin()
            best = sub.loc[best_idx]
            dlinear_row = sub[sub['model'] == 'DLinear']
            patchtst_row = sub[sub['model'] == 'PatchTST']
            pfbv0_row = sub[sub['model'] == 'PatchFusionBERT_v0']

            dl_mse = dlinear_row['mse_mean'].values[0] if not dlinear_row.empty else None
            pt_mse = patchtst_row['mse_mean'].values[0] if not patchtst_row.empty else None
            pfb_mse = pfbv0_row['mse_mean'].values[0] if not pfbv0_row.empty else None

            dl_delta = f'{pfb_mse - dl_mse:+.4f}' if dl_mse and pfb_mse else 'N/A'
            pt_delta = f'{pfb_mse - pt_mse:+.4f}' if pt_mse and pfb_mse else 'N/A'

            lines.append(f'| {dataset} | {horizon} | {best["model"]} | {best["mse_mean"]:.4f} | {dl_delta} | {pt_delta} |')
    lines.append('')

    return '\n'.join(lines)


def main():
    print(f'Parsing {RESULTS_FILE} for tag: {TAG}')
    df = parse_results(RESULTS_FILE)
    print(f'  Found {len(df)} records')

    if df.empty:
        print('ERROR: No records found. Check tag or results file.')
        return

    print(f'  Models found: {sorted(df["model"].unique())}')
    print(f'  Datasets found: {sorted(df["dataset"].unique())}')
    print(f'  Horizons found: {sorted(df["horizon"].unique())}')
    print(f'  Seeds found: {sorted(df["seed"].unique())}')

    summary = build_summary(df)

    # Save raw CSV
    raw_out = OUT_DIR / 'multiseed_h96_h336_raw.csv'
    df.to_csv(raw_out, index=False)
    print(f'  Raw CSV saved: {raw_out}')

    # Save summary CSV
    sum_out = OUT_DIR / 'multiseed_h96_h336_summary.csv'
    summary.to_csv(sum_out, index=False)
    print(f'  Summary CSV saved: {sum_out}')

    # Save markdown
    md = write_markdown(summary, df)
    md_out = OUT_DIR / 'multiseed_h96_h336_summary.md'
    md_out.write_text(md, encoding='utf-8')
    print(f'  Markdown saved: {md_out}')

    # Print quick overview
    print('\n=== QUICK OVERVIEW ===')
    complete = summary[summary['status'] == 'COMPLETE']
    missing = summary[summary['status'] == 'MISSING']
    print(f'  Complete configs: {len(complete)}/{len(summary)}')
    if not missing.empty:
        print(f'  Missing:')
        for _, r in missing.iterrows():
            print(f'    {r["dataset"]} H={r["horizon"]} {r["model"]}')

    print('\n=== H=96 SUMMARY (MSE mean) ===')
    for dataset in DATASETS:
        sub = summary[(summary['horizon'] == 96) & (summary['dataset'] == dataset) & (summary['status'] == 'COMPLETE')]
        if sub.empty:
            continue
        vals = {r['model']: r['mse_mean'] for _, r in sub.iterrows()}
        print(f'  {dataset}: PFBv0={vals.get("PatchFusionBERT_v0", "?"):.4f}  '
              f'PFBv2={vals.get("PatchFusionBERT_v2", "?"):.4f}  '
              f'PatchTST={vals.get("PatchTST", "?"):.4f}  '
              f'DLinear={vals.get("DLinear", "?"):.4f}')

    print('\n=== H=336 SUMMARY (MSE mean) ===')
    for dataset in DATASETS:
        sub = summary[(summary['horizon'] == 336) & (summary['dataset'] == dataset) & (summary['status'] == 'COMPLETE')]
        if sub.empty:
            continue
        vals = {r['model']: r['mse_mean'] for _, r in sub.iterrows()}
        print(f'  {dataset}: PFBv0={vals.get("PatchFusionBERT_v0", "?"):.4f}  '
              f'PFBv2={vals.get("PatchFusionBERT_v2", "?"):.4f}  '
              f'PatchTST={vals.get("PatchTST", "?"):.4f}  '
              f'DLinear={vals.get("DLinear", "?"):.4f}')


if __name__ == '__main__':
    main()
