"""
Analyze bridging LR unified experiment results.
Tag in results file: model_id starts with BridgeLR_
Output: results_analysis/bridging_lr_summary.md + .csv
"""

import re
import pandas as pd
from pathlib import Path

RESULTS_FILE = 'result_long_term_forecast.txt'
OUT_DIR = Path('results_analysis')
OUT_DIR.mkdir(exist_ok=True)

TAG = 'BridgeLR_'

MODEL_MAP = {
    'PFB-Direct': 'PFB-Direct',
    'PFB-Projected': 'PFB-Projected',
    'PatchTST': 'PatchTST',
    'DLinear': 'DLinear',
}
MODELS_ORDER = ['PFB-Direct', 'PFB-Projected', 'PatchTST', 'DLinear']
DATASETS = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather', 'Illness']
STD_HORIZONS = [96, 192, 336, 720]
ILL_HORIZONS = [24, 48, 60]


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

        mse_match = re.search(r'mse:([\d.]+)', metrics_line)
        mae_match = re.search(r'mae:([\d.]+)', metrics_line)
        if not mse_match or not mae_match:
            continue
        mse = float(mse_match.group(1))
        mae = float(mae_match.group(1))

        # model_id format: BridgeLR_PFB-Direct_ETTm2_192 or BridgeLR_PatchTST_Weather_720
        bridge_match = re.search(r'BridgeLR_([^_]+)_([^_]+)_(\d+)', name_line)
        if not bridge_match:
            continue

        short_model = bridge_match.group(1)
        dataset = bridge_match.group(2)
        horizon = int(bridge_match.group(3))

        model = MODEL_MAP.get(short_model)
        if model is None:
            continue
        if dataset not in DATASETS:
            continue

        records.append({
            'dataset': dataset,
            'horizon': horizon,
            'model': model,
            'mse': mse,
            'mae': mae,
        })

    return pd.DataFrame(records)


def write_markdown(df):
    lines = []
    lines.append('# Bridging LR Unified Experiment Results')
    lines.append('\nProtocol: Single seed (2021). Unified LR — DLinear: 1e-2, all others: 1e-4.')
    lines.append('Purpose: Verify main benchmark results under unified LR protocol.\n')
    lines.append(f'Total runs: {len(df)}/108\n')

    # Standard datasets
    lines.append('## Standard Datasets (H=96/192/336/720)\n')
    for dataset in [d for d in DATASETS if d != 'Illness']:
        lines.append(f'### {dataset}\n')
        lines.append('| Horizon | PFB-Direct MSE | PFB-Direct MAE | PFB-Projected MSE | PatchTST MSE | DLinear MSE | PFB-Direct vs DLinear | PFB-Direct vs PatchTST |')
        lines.append('|---------|-----------|-----------|-----------|--------------|-------------|------------------|-------------------|')
        for h in STD_HORIZONS:
            sub = df[(df['dataset'] == dataset) & (df['horizon'] == h)]
            if sub.empty:
                lines.append(f'| {h} | - | - | - | - | - | - | - |')
                continue
            def get(m): 
                r = sub[sub['model'] == m]
                return r['mse'].values[0] if not r.empty else None
            def getmae(m):
                r = sub[sub['model'] == m]
                return r['mae'].values[0] if not r.empty else None

            pfb_direct = get('PFB-Direct')
            pfb_direct_mae = getmae('PFB-Direct')
            pfb_projected = get('PFB-Projected')
            ptst = get('PatchTST')
            dl = get('DLinear')

            dl_delta = f'{pfb_direct - dl:+.4f}' if pfb_direct and dl else 'N/A'
            pt_delta = f'{pfb_direct - ptst:+.4f}' if pfb_direct and ptst else 'N/A'

            pfb_projected_s = f'{pfb_projected:.4f}' if pfb_projected is not None else 'N/A'
            ptst_s = f'{ptst:.4f}' if ptst is not None else 'N/A'
            dl_s = f'{dl:.4f}' if dl is not None else 'N/A'
            lines.append(f'| {h} | {pfb_direct:.4f} | {pfb_direct_mae:.4f} | {pfb_projected_s} | {ptst_s} | {dl_s} | {dl_delta} | {pt_delta} |')
        lines.append('')

    # Illness
    lines.append('## Illness Dataset (H=24/48/60)\n')
    lines.append('| Horizon | PFB-Direct MSE | PFB-Direct MAE | PFB-Projected MSE | PatchTST MSE | DLinear MSE |')
    lines.append('|---------|-----------|-----------|-----------|--------------|-------------|')
    for h in ILL_HORIZONS:
        sub = df[(df['dataset'] == 'Illness') & (df['horizon'] == h)]
        if sub.empty:
            lines.append(f'| {h} | - | - | - | - | - |')
            continue
        def get(m):
            r = sub[sub['model'] == m]
            return r['mse'].values[0] if not r.empty else None
        def getmae(m):
            r = sub[sub['model'] == m]
            return r['mae'].values[0] if not r.empty else None
        pfb_direct = get('PFB-Direct')
        pfb_direct_mae = getmae('PFB-Direct')
        pfb_projected = get('PFB-Projected')
        ptst = get('PatchTST')
        dl = get('DLinear')
        pfb_direct_s = f'{pfb_direct:.4f}' if pfb_direct is not None else 'N/A'
        pfb_direct_mae_s = f'{pfb_direct_mae:.4f}' if pfb_direct_mae is not None else 'N/A'
        pfb_projected_s = f'{pfb_projected:.4f}' if pfb_projected is not None else 'N/A'
        ptst_s = f'{ptst:.4f}' if ptst is not None else 'N/A'
        dl_s = f'{dl:.4f}' if dl is not None else 'N/A'
        lines.append(f'| {h} | {pfb_direct_s} | {pfb_direct_mae_s} | {pfb_projected_s} | {ptst_s} | {dl_s} |')
    lines.append('')

    # Summary: consistent advantage check
    lines.append('## PFB-Direct vs DLinear — Bridging LR Direction Check\n')
    lines.append('Negative ΔMSE = PFB-Direct wins. Positive = DLinear wins.\n')
    lines.append('| Dataset | H=96 | H=192 | H=336 | H=720 | Pattern |')
    lines.append('|---------|------|-------|-------|-------|---------|')
    for dataset in [d for d in DATASETS if d != 'Illness']:
        deltas = []
        for h in STD_HORIZONS:
            sub = df[(df['dataset'] == dataset) & (df['horizon'] == h)]
            pfb_direct_r = sub[sub['model'] == 'PFB-Direct']
            dl_r = sub[sub['model'] == 'DLinear']
            if not pfb_direct_r.empty and not dl_r.empty:
                d = pfb_direct_r['mse'].values[0] - dl_r['mse'].values[0]
                deltas.append(f'{d:+.4f}')
            else:
                deltas.append('N/A')
        wins = sum(1 for d in deltas if d != 'N/A' and float(d) < 0)
        total = sum(1 for d in deltas if d != 'N/A')
        pattern = f'PFB-Direct wins {wins}/{total}' if total > 0 else 'N/A'
        row = f'| {dataset} | ' + ' | '.join(deltas) + f' | {pattern} |'
        lines.append(row)

    return '\n'.join(lines)


def main():
    print(f'Parsing {RESULTS_FILE} for tag: {TAG}')
    df = parse_results(RESULTS_FILE)
    print(f'  Found {len(df)} records')

    if df.empty:
        print('ERROR: No records found.')
        return

    print(f'  Models: {sorted(df["model"].unique())}')
    print(f'  Datasets: {sorted(df["dataset"].unique())}')
    print(f'  Horizons: {sorted(df["horizon"].unique())}')

    # Save raw CSV
    raw_out = OUT_DIR / 'bridging_lr_raw.csv'
    df.to_csv(raw_out, index=False)
    print(f'  Raw CSV: {raw_out}')

    # Save summary CSV
    sum_out = OUT_DIR / 'bridging_lr_summary.csv'
    df.to_csv(sum_out, index=False)
    print(f'  Summary CSV: {sum_out}')

    # Save markdown
    md = write_markdown(df)
    md_out = OUT_DIR / 'bridging_lr_summary.md'
    md_out.write_text(md, encoding='utf-8')
    print(f'  Markdown: {md_out}')

    # Quick console summary
    print('\n=== PFB-Direct vs DLinear DIRECTION (BRIDGING LR) ===')
    for dataset in [d for d in DATASETS if d != 'Illness']:
        row_parts = []
        for h in STD_HORIZONS:
            sub = df[(df['dataset'] == dataset) & (df['horizon'] == h)]
            pfb = sub[sub['model'] == 'PFB-Direct']
            dl = sub[sub['model'] == 'DLinear']
            if not pfb.empty and not dl.empty:
                delta = pfb['mse'].values[0] - dl['mse'].values[0]
                row_parts.append(f'H{h}:{delta:+.4f}')
        print(f'  {dataset}: {" | ".join(row_parts)}')

    print('\n=== PFB-Direct vs PatchTST DIRECTION (BRIDGING LR) ===')
    for dataset in [d for d in DATASETS if d != 'Illness']:
        row_parts = []
        for h in STD_HORIZONS:
            sub = df[(df['dataset'] == dataset) & (df['horizon'] == h)]
            pfb = sub[sub['model'] == 'PFB-Direct']
            pt = sub[sub['model'] == 'PatchTST']
            if not pfb.empty and not pt.empty:
                delta = pfb['mse'].values[0] - pt['mse'].values[0]
                row_parts.append(f'H{h}:{delta:+.4f}')
        print(f'  {dataset}: {" | ".join(row_parts)}')


if __name__ == '__main__':
    main()
