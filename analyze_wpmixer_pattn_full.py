from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULT_LOG = ROOT / 'result_long_term_forecast.txt'
REPORT_MD = ROOT / 'results_23-01.md'
OUT_DIR = ROOT / 'results_analysis'
OUT_ALL = OUT_DIR / 'wpmixer_pattn_full_comparison.csv'
OUT_SUMMARY = OUT_DIR / 'wpmixer_pattn_full_summary.csv'

STANDARD_DATASETS = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather']
ILLNESS = 'Illness'
HORIZONS = {
    'ETTm1': [96, 192, 336],
    'ETTm2': [96, 192, 336],
    'ETTh1': [96, 192, 336],
    'ETTh2': [96, 192, 336],
    'Exchange': [96, 192, 336],
    'Weather': [96, 192, 336],
    'Illness': [24, 48, 60],
}


def parse_report_tables() -> pd.DataFrame:
    text = REPORT_MD.read_text(encoding='utf-8')
    rows: list[dict] = []

    for dataset in STANDARD_DATASETS:
        pattern = re.compile(
            rf"## Dataset: {re.escape(dataset)}\n.*?\n\| Model \| H=96 MSE \| H=192 MSE \| H=336 MSE \| H=96 MAE \| H=192 MAE \| H=336 MAE \|\n\|---\|---\|---\|---\|---\|---\|---\|\n(.*?)\n###",
            re.S,
        )
        m = pattern.search(text)
        if not m:
            raise RuntimeError(f'Could not parse report table for {dataset}')
        for line in [ln.strip() for ln in m.group(1).splitlines() if ln.strip().startswith('|')]:
            parts = [p.strip() for p in line.strip('|').split('|')]
            model = parts[0].replace('**', '')
            mse_vals = list(map(float, parts[1:4]))
            mae_vals = list(map(float, parts[4:7]))
            for horizon, mse, mae in zip(HORIZONS[dataset], mse_vals, mae_vals):
                rows.append({
                    'Dataset': dataset,
                    'Horizon': horizon,
                    'Model': model,
                    'MSE': mse,
                    'MAE': mae,
                    'Source': 'results_23-01.md',
                })

    ill_pattern = re.compile(
        r"## Dataset: Illness\n.*?\n\| Model \| H=24 MSE \| H=48 MSE \| H=60 MSE \| H=24 MAE \| H=48 MAE \| H=60 MAE \|\n\|---\|---\|---\|---\|---\|---\|---\|\n(.*?)\n###",
        re.S,
    )
    m = ill_pattern.search(text)
    if not m:
        raise RuntimeError('Could not parse report table for Illness')
    for line in [ln.strip() for ln in m.group(1).splitlines() if ln.strip().startswith('|')]:
        parts = [p.strip() for p in line.strip('|').split('|')]
        model = parts[0].replace('**', '')
        mse_vals = list(map(float, parts[1:4]))
        mae_vals = list(map(float, parts[4:7]))
        for horizon, mse, mae in zip(HORIZONS[ILLNESS], mse_vals, mae_vals):
            rows.append({
                'Dataset': ILLNESS,
                'Horizon': horizon,
                'Model': model,
                'MSE': mse,
                'MAE': mae,
                'Source': 'results_23-01.md',
            })

    return pd.DataFrame(rows)


def parse_new_runs() -> pd.DataFrame:
    lines = RESULT_LOG.read_text(encoding='utf-8').splitlines()
    pattern = re.compile(
        r'long_term_forecast_(WPMixer|PAttn)_Full_(ETTm1|ETTm2|ETTh1|ETTh2|Exchange|Weather|Illness)_H(24|48|60|96|192|336)_'
    )
    rows: list[dict] = []
    for i, line in enumerate(lines):
        m = pattern.search(line.strip())
        if not m:
            continue
        metrics = lines[i + 1].strip() if i + 1 < len(lines) else ''
        mse_m = re.search(r'mse:([\d.]+)', metrics)
        mae_m = re.search(r'mae:([\d.]+)', metrics)
        if not (mse_m and mae_m):
            continue
        rows.append({
            'Dataset': m.group(2),
            'Horizon': int(m.group(3)),
            'Model': m.group(1),
            'MSE': float(mse_m.group(1)),
            'MAE': float(mae_m.group(1)),
            'Source': 'result_long_term_forecast.txt',
        })
    return pd.DataFrame(rows)


def build_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = parse_report_tables()
    new_runs = parse_new_runs()
    combined = pd.concat([baseline, new_runs], ignore_index=True)

    combined['MSE_Rank'] = combined.groupby(['Dataset', 'Horizon'])['MSE'].rank(method='min')
    combined['MAE_Rank'] = combined.groupby(['Dataset', 'Horizon'])['MAE'].rank(method='min')

    best_existing = (
        baseline.sort_values(['Dataset', 'Horizon', 'MSE'])
        .groupby(['Dataset', 'Horizon'], as_index=False)
        .first()[['Dataset', 'Horizon', 'Model', 'MSE', 'MAE']]
        .rename(columns={'Model': 'BestExistingModel', 'MSE': 'BestExistingMSE', 'MAE': 'BestExistingMAE'})
    )

    comparison = combined.merge(best_existing, on=['Dataset', 'Horizon'], how='left')
    comparison['DeltaVsBestExisting_MSE'] = comparison['MSE'] - comparison['BestExistingMSE']
    comparison['DeltaVsBestExisting_MAE'] = comparison['MAE'] - comparison['BestExistingMAE']

    target_models = comparison[comparison['Model'].isin(['WPMixer', 'PAttn'])].copy()
    summary = (
        target_models.groupby('Model', as_index=False)
        .agg(
            Settings=('Model', 'size'),
            AvgMSE=('MSE', 'mean'),
            AvgMAE=('MAE', 'mean'),
            AvgMSE_Rank=('MSE_Rank', 'mean'),
            AvgMAE_Rank=('MAE_Rank', 'mean'),
            MSE_Wins=('MSE_Rank', lambda s: int((s == 1).sum())),
            MAE_Wins=('MAE_Rank', lambda s: int((s == 1).sum())),
            AvgDeltaVsBestExisting_MSE=('DeltaVsBestExisting_MSE', 'mean'),
            AvgDeltaVsBestExisting_MAE=('DeltaVsBestExisting_MAE', 'mean'),
        )
        .sort_values(['AvgMSE_Rank', 'AvgMSE'])
        .reset_index(drop=True)
    )

    return comparison.sort_values(['Dataset', 'Horizon', 'MSE']), summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison, summary = build_comparison()
    comparison.to_csv(OUT_ALL, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f'Wrote: {OUT_ALL}')
    print(f'Wrote: {OUT_SUMMARY}')
    print('\nSummary:')
    print(summary.to_string(index=False))
    print('\nPer-setting comparison for WPMixer/PAttn:')
    cols = [
        'Dataset', 'Horizon', 'Model', 'MSE', 'MAE', 'MSE_Rank', 'MAE_Rank',
        'BestExistingModel', 'BestExistingMSE', 'DeltaVsBestExisting_MSE'
    ]
    print(comparison[comparison['Model'].isin(['WPMixer', 'PAttn'])][cols].to_string(index=False))


if __name__ == '__main__':
    main()