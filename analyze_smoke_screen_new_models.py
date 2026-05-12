from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RESULT_FILES = [
    Path('result_long_term_forecast.txt'),
    Path('..') / 'result_long_term_forecast.txt',
]

OUT_DIR = Path('results_analysis')
OUT_RESULTS = OUT_DIR / 'smoke_screen_new_models_results.csv'
OUT_SUMMARY = OUT_DIR / 'smoke_screen_new_models_summary.csv'

MODELS = [
    'TimeMixer',
    'MultiPatchFormer',
    'WPMixer',
    'MSGNet',
    'PAttn',
    'SegRNN',
    'MambaSimple',
]
DATASETS = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather', 'Illness']


def parse_runs() -> pd.DataFrame:
    rows: list[dict] = []
    pattern = re.compile(
        r'Smoke_(TimeMixer|MultiPatchFormer|WPMixer|MSGNet|PAttn|SegRNN|MambaSimple)_(ETTm1|ETTm2|ETTh1|ETTh2|Exchange|Weather|Illness)'
    )
    for path in RESULT_FILES:
        if not path.exists():
            continue
        lines = path.read_text(encoding='utf-8').splitlines()
        for i, line in enumerate(lines):
            s = line.strip()
            if not s.startswith('long_term_forecast_') or 'Smoke_' not in s:
                continue

            m = pattern.search(s)
            if not m:
                continue

            metrics_text = s
            if 'mse:' not in metrics_text or 'mae:' not in metrics_text:
                nxt = lines[i + 1].strip() if i + 1 < len(lines) else ''
                metrics_text = f'{metrics_text} {nxt}'.strip()

            mse_m = re.search(r'mse:([\d.]+)', metrics_text)
            mae_m = re.search(r'mae:([\d.]+)', metrics_text)
            if not (mse_m and mae_m):
                continue

            rows.append(
                {
                    'Model': m.group(1),
                    'Dataset': m.group(2),
                    'MSE': float(mse_m.group(1)),
                    'MAE': float(mae_m.group(1)),
                    'ExperimentName': s,
                    'LineNo': i,
                    'Source': str(path),
                }
            )

    if not rows:
        raise RuntimeError('No smoke screening runs found in result logs.')

    df = pd.DataFrame(rows).sort_values(['Source', 'LineNo'])
    df = df.groupby(['Model', 'Dataset'], as_index=False).tail(1)
    return df.sort_values(['Model', 'Dataset']).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot(index='Dataset', columns='Model', values='MSE')
    ranks = pivot.rank(axis=1, method='average')

    rows: list[dict] = []
    for model in MODELS:
        present = int(df[df['Model'] == model]['Dataset'].nunique())
        avg_mse = df[df['Model'] == model]['MSE'].mean()
        avg_mae = df[df['Model'] == model]['MAE'].mean()
        avg_rank = ranks[model].mean() if model in ranks.columns else float('nan')
        wins = int((ranks[model] == 1).sum()) if model in ranks.columns else 0
        rows.append(
            {
                'Model': model,
                'DatasetsCompleted': present,
                'ExpectedDatasets': len(DATASETS),
                'AvgMSE': avg_mse,
                'AvgMAE': avg_mae,
                'AvgRank': avg_rank,
                'DatasetWins': wins,
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(['DatasetsCompleted', 'AvgRank', 'AvgMSE'], ascending=[False, True, True]).reset_index(drop=True)
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = parse_runs()
    summary = summarize(df)
    df.to_csv(OUT_RESULTS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f'Wrote: {OUT_RESULTS}')
    print(f'Wrote: {OUT_SUMMARY}')
    print('\nPer-run results:')
    print(df.to_string(index=False))
    print('\nSummary:')
    print(summary.to_string(index=False))
    print('\nRecommended full-training candidates:')
    print(summary.head(3)[['Model', 'DatasetsCompleted', 'AvgRank', 'AvgMSE']].to_string(index=False))


if __name__ == '__main__':
    main()