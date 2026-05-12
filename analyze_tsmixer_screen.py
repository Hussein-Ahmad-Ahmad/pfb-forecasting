from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RESULT_FILES = [
    Path('result_long_term_forecast.txt'),
    Path('..') / 'result_long_term_forecast.txt',
]

OUT_DIR = Path('results_analysis')
OUT_SCREEN = OUT_DIR / 'tsmixer_screen_results.csv'
OUT_COMPARE = OUT_DIR / 'tsmixer_vs_current_benchmark.csv'
OUT_SUMMARY = OUT_DIR / 'tsmixer_screen_summary.csv'

TARGET_MODELS = [
    'DLinear',
    'PatchTST',
    'TiDE',
    'TimeXer',
    'iTransformer',
    'PatchFusionBERT_v0',
    'PatchFusionBERT_v2',
    'PatchFusionBERT_BERTOnly',
]

KEY_DATASETS = {'Weather', 'ETTm2'}


def parse_tsmixer_runs() -> pd.DataFrame:
    rows: list[dict] = []
    for path in RESULT_FILES:
        if not path.exists():
            continue
        lines = path.read_text(encoding='utf-8').splitlines()
        for i, line in enumerate(lines):
            s = line.strip()
            if not s.startswith('long_term_forecast_'):
                continue
            if 'TSMixerScreen_' not in s and 'TSMixerScreenV2_' not in s:
                continue

            metrics_text = s
            if 'mse:' not in metrics_text or 'mae:' not in metrics_text:
                nxt = lines[i + 1].strip() if i + 1 < len(lines) else ''
                metrics_text = f'{metrics_text} {nxt}'.strip()

            mse_m = re.search(r'mse:([\d.]+)', metrics_text)
            mae_m = re.search(r'mae:([\d.]+)', metrics_text)
            id_m = re.search(r'TSMixerScreen(?:V2)?_(ETTm1|ETTm2|ETTh1|ETTh2|Exchange|Weather|Illness)_(\d+)', s)
            sl_m = re.search(r'_sl(\d+)_', s)
            if not (mse_m and mae_m and id_m and sl_m):
                continue

            rows.append(
                {
                    'Dataset': id_m.group(1),
                    'Horizon': int(id_m.group(2)),
                    'SeqLen': int(sl_m.group(1)),
                    'Model': 'TSMixer',
                    'MSE': float(mse_m.group(1)),
                    'MAE': float(mae_m.group(1)),
                    'ExperimentName': s,
                    'LineNo': i,
                    'Source': str(path),
                }
            )

    if not rows:
        raise RuntimeError('No TSMixer screening runs were found in the result logs.')

    df = pd.DataFrame(rows)
    df = df.sort_values(['Source', 'LineNo'])
    df = df.groupby(['Dataset', 'Horizon'], as_index=False).tail(1)
    return df.sort_values(['Dataset', 'Horizon']).reset_index(drop=True)


def load_current_benchmark() -> pd.DataFrame:
    df = pd.read_csv('results_23-01.csv')
    df = df[df['Model'].isin(TARGET_MODELS)].copy()
    df = df[
        ((df['Dataset'] != 'Illness') & (df['SeqLen'] == 336) & (df['Horizon'].isin([96, 192, 336])))
        | ((df['Dataset'] == 'Illness') & (df['SeqLen'] == 104) & (df['Horizon'].isin([24, 48, 60])))
    ].copy()

    exclude_patterns = [
        'MS20',
        'CapMatch_',
        'KDepth_',
        'Align_',
        'PatchSens_',
        'WallClock_',
        '_test_',
        'PREFLIGHT',
        'CALIBRATE',
        'illness_horizons',
    ]
    mask = pd.Series(False, index=df.index)
    for pat in exclude_patterns:
        mask = mask | df['ExperimentName'].str.contains(pat, regex=False)
    df = df[~mask].copy()

    # The archived CSV contains a small number of exact duplicate rows.
    df = df.drop_duplicates(subset=['Dataset', 'Horizon', 'Model', 'SeqLen', 'ExperimentName']).copy()

    # For Illness, keep the direct single-seed benchmark rows whose experiment names
    # match the reported protocol: long_term_forecast_Illness_<seq>_<horizon>_<model>...
    illness_mask = df['Dataset'] == 'Illness'
    illness_direct = illness_mask & df['ExperimentName'].str.contains(r'long_term_forecast_Illness_104_(?:24|48|60)_', regex=True)
    df = pd.concat([
        df[~illness_mask],
        df[illness_direct],
    ], ignore_index=True)

    counts = df.groupby(['Dataset', 'Horizon', 'Model']).size()
    dupes = counts[counts > 1]
    if not dupes.empty:
        raise RuntimeError(f'Unexpected duplicate benchmark rows remain:\n{dupes.to_string()}')

    return df.sort_values(['Dataset', 'Horizon', 'Model']).reset_index(drop=True)


def summarize(ts: pd.DataFrame, base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot_mse = base.pivot(index=['Dataset', 'Horizon'], columns='Model', values='MSE').reset_index()
    pivot_mae = base.pivot(index=['Dataset', 'Horizon'], columns='Model', values='MAE').reset_index()
    compare = ts.merge(pivot_mse, on=['Dataset', 'Horizon']).merge(pivot_mae, on=['Dataset', 'Horizon'], suffixes=('_mse', '_mae'))

    model_cols = [f'{m}_mse' for m in TARGET_MODELS if f'{m}_mse' in compare.columns]
    compare['BestExistingModel'] = compare[model_cols].idxmin(axis=1).str.replace('_mse', '', regex=False)
    compare['BestExistingMSE'] = compare[model_cols].min(axis=1)
    compare['TSMixer_Beats_PFB_v0'] = compare['MSE'] < compare['PatchFusionBERT_v0_mse']
    compare['TSMixer_Beats_PFB_v2'] = compare['MSE'] < compare['PatchFusionBERT_v2_mse']
    compare['TSMixer_Beats_BothFusion'] = compare['TSMixer_Beats_PFB_v0'] & compare['TSMixer_Beats_PFB_v2']
    compare['FusionBestMSE'] = compare[['PatchFusionBERT_v0_mse', 'PatchFusionBERT_v2_mse']].min(axis=1)
    compare['TSMixer_NearTie_Fusion_1pct'] = compare['MSE'] <= compare['FusionBestMSE'] * 1.01
    compare['TSMixer_Wins_All'] = compare['MSE'] < compare['BestExistingMSE']
    compare['TSMixer_RankIfAdded'] = compare[model_cols + ['MSE']].rank(axis=1, method='min').loc[:, 'MSE'].astype(int)

    summary_rows = []
    tsm_avg_mse = compare['MSE'].mean()
    tsm_avg_mae = compare['MAE'].mean()
    pfb_v0_avg_mse = base[base['Model'] == 'PatchFusionBERT_v0']['MSE'].mean()
    pfb_v2_avg_mse = base[base['Model'] == 'PatchFusionBERT_v2']['MSE'].mean()

    summary_rows.append({'Metric': 'settings', 'Value': len(compare)})
    summary_rows.append({'Metric': 'tsmixer_avg_mse', 'Value': tsm_avg_mse})
    summary_rows.append({'Metric': 'tsmixer_avg_mae', 'Value': tsm_avg_mae})
    summary_rows.append({'Metric': 'pfb_v0_avg_mse', 'Value': pfb_v0_avg_mse})
    summary_rows.append({'Metric': 'pfb_v2_avg_mse', 'Value': pfb_v2_avg_mse})
    summary_rows.append({'Metric': 'beats_pfb_v0_mse', 'Value': int(compare['TSMixer_Beats_PFB_v0'].sum())})
    summary_rows.append({'Metric': 'beats_pfb_v2_mse', 'Value': int(compare['TSMixer_Beats_PFB_v2'].sum())})
    summary_rows.append({'Metric': 'beats_both_fusion_mse', 'Value': int(compare['TSMixer_Beats_BothFusion'].sum())})
    summary_rows.append({'Metric': 'near_tie_or_better_bothfusion_1pct', 'Value': int(compare['TSMixer_NearTie_Fusion_1pct'].sum())})
    summary_rows.append({'Metric': 'wins_all_models_mse', 'Value': int(compare['TSMixer_Wins_All'].sum())})
    summary_rows.append({'Metric': 'key_regime_beats_both_fusion', 'Value': int(compare[compare['Dataset'].isin(KEY_DATASETS)]['TSMixer_Beats_BothFusion'].sum())})
    summary_rows.append({'Metric': 'key_regime_near_tie_or_better_1pct', 'Value': int(compare[compare['Dataset'].isin(KEY_DATASETS)]['TSMixer_NearTie_Fusion_1pct'].sum())})

    summary = pd.DataFrame(summary_rows)
    return compare.sort_values(['Dataset', 'Horizon']).reset_index(drop=True), summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = parse_tsmixer_runs()
    base = load_current_benchmark()

    compare, summary = summarize(ts, base)

    ts.to_csv(OUT_SCREEN, index=False)
    compare.to_csv(OUT_COMPARE, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f'Wrote: {OUT_SCREEN}')
    print(f'Wrote: {OUT_COMPARE}')
    print(f'Wrote: {OUT_SUMMARY}')
    print('\nTSMixer vs current benchmark (MSE):')
    cols = [
        'Dataset', 'Horizon', 'MSE', 'MAE', 'PatchFusionBERT_v0_mse', 'PatchFusionBERT_v2_mse',
        'BestExistingModel', 'BestExistingMSE', 'TSMixer_Beats_BothFusion',
        'TSMixer_NearTie_Fusion_1pct', 'TSMixer_RankIfAdded'
    ]
    print(compare[cols].to_string(index=False))
    print('\nSummary:')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
