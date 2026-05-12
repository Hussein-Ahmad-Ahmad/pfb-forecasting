"""Generate LaTeX tables for doc.j-8-02/name.tex from results_23-01.csv.

Goal: make manuscript tables reproducible and perfectly consistent with the CSV.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT_OUT = Path('..') / 'doc.j-8-02' / 'tables'
ROOT_OUT.mkdir(parents=True, exist_ok=True)

CSV_PATH = Path('results_23-01.csv')

MAIN_H = [96, 192, 336]
# Illness: some models are missing H=36 in results_23-01.csv.
# To keep the manuscript table fully reproducible (no missing cells) across the
# included Illness models, we report horizons with complete coverage.
ILL_H = [24, 48, 60]

MODELS_MAIN = [
    'DLinear',
    'PatchTST',
    'TiDE',
    'TimeXer',
    'iTransformer',
    'PatchFusionBERT_BERTOnly',
    'PatchFusionBERT_v0',
    'PatchFusionBERT_v2',
]

MODELS_ILL = [
    'DLinear',
    'PatchTST',
    'TiDE',
    'TimeXer',
    'iTransformer',
    'PatchFusionBERT_BERTOnly',
    'PatchFusionBERT_v0',
    'PatchFusionBERT_v2',
]


def fmt(x: float) -> str:
    return f"{x:.4f}"


def best_second_mask(values: pd.Series) -> tuple[pd.Index, pd.Index]:
    s = values.sort_values()
    if len(s) == 0:
        return pd.Index([]), pd.Index([])
    best = pd.Index([s.index[0]])
    second = pd.Index([s.index[1]]) if len(s) > 1 else pd.Index([])
    return best, second


def table_main_results(df: pd.DataFrame) -> str:
    # Mean across repeated runs/seeds per dataset/horizon/model
    agg = (
        df[df['Horizon'].isin(MAIN_H) & df['Model'].isin(MODELS_MAIN) & (df['Dataset'] != 'Illness')]
        .groupby(['Dataset', 'Horizon', 'Model'], as_index=False)
        .agg(MSE=('MSE', 'mean'), MAE=('MAE', 'mean'))
    )

    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Weather']
    # Keep only datasets present
    datasets = [d for d in datasets if d in set(agg['Dataset'])]

    lines: list[str] = []
    lines.append('\\begin{table*}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Forecasting performance across datasets and horizons (reported as MSE/MAE). Best and second-best are highlighted. Values are mean over available runs/seeds in results\\_23-01.csv.}')
    lines.append('\\label{tab:main_results}')
    # Slight downscaling to keep the (expanded) table within page height.
    lines.append('\\resizebox{0.98\\textwidth}{!}{%')
    lines.append('\\begin{tabular}{@{}llccc@{}}')
    lines.append('\\toprule')
    lines.append('Dataset & Model & $H=96$ (MSE/MAE) & $H=192$ (MSE/MAE) & $H=336$ (MSE/MAE) \\\\')
    lines.append('\\midrule')

    for ds in datasets:
        sub_ds = agg[agg['Dataset'] == ds]

        # Determine best/second per horizon by MSE
        highlight: dict[tuple[int, str], str] = {}
        for h in MAIN_H:
            sub_h = sub_ds[sub_ds['Horizon'] == h].set_index('Model')
            if len(sub_h) == 0:
                continue
            best, second = best_second_mask(sub_h['MSE'])
            for m in best:
                highlight[(h, m)] = 'best'
            for m in second:
                highlight[(h, m)] = 'second'

        def cell(h: int, m: str) -> str:
            r = sub_ds[(sub_ds['Horizon'] == h) & (sub_ds['Model'] == m)]
            if len(r) == 0:
                return '---'
            mse = float(r['MSE'].iloc[0])
            mae = float(r['MAE'].iloc[0])
            mse_s = fmt(mse)
            mae_s = fmt(mae)
            tag = highlight.get((h, m))
            if tag == 'best':
                mse_s = f"\\best{{{mse_s}}}"
            elif tag == 'second':
                mse_s = f"\\second{{{mse_s}}}"
            # We only highlight MSE for clean reading (consistent with many TS papers)
            return f"{mse_s} / {mae_s}"

        model_labels = {
            'DLinear': '\\dlinear',
            'PatchTST': '\\patchtst',
            'TiDE': 'TiDE',
            'TimeXer': 'TimeXer',
            'iTransformer': 'iTransformer',
            'PatchFusionBERT_BERTOnly': '\\bertonly',
            'PatchFusionBERT_v0': '\\modelvzero',
            'PatchFusionBERT_v2': '\\modelvtwo',
        }

        lines.append(f"\\multirow{{{len(MODELS_MAIN)}}}{{*}}{{{ds}}}")
        for i, m in enumerate(MODELS_MAIN):
            prefix = '&' if i > 0 else '&'
            label = model_labels[m]
            lines.append(
                f"{prefix} {label} & {cell(96, m)} & {cell(192, m)} & {cell(336, m)} \\\\"
            )
        lines.append('\\midrule')

    # Replace last midrule with bottomrule
    if lines[-1] == '\\midrule':
        lines[-1] = '\\bottomrule'
    else:
        lines.append('\\bottomrule')

    lines.append('\\end{tabular}%')
    lines.append('}')
    lines.append('\\end{table*}')

    return '\n'.join(lines) + '\n'


def table_illness(df: pd.DataFrame) -> str:
    agg = (
        df[(df['Dataset'] == 'Illness') & df['Horizon'].isin(ILL_H) & df['Model'].isin(MODELS_ILL)]
        .groupby(['Horizon', 'Model'], as_index=False)
        .agg(MSE=('MSE', 'mean'))
    )

    # Determine best per horizon
    highlight: dict[tuple[int, str], bool] = {}
    for h in ILL_H:
        sub = agg[agg['Horizon'] == h].set_index('Model')
        if len(sub) == 0:
            continue
        best, _ = best_second_mask(sub['MSE'])
        for m in best:
            highlight[(h, m)] = True

    def mse(h: int, m: str) -> float | None:
        r = agg[(agg['Horizon'] == h) & (agg['Model'] == m)]
        if len(r) == 0:
            return None
        return float(r['MSE'].iloc[0])

    def cell(h: int, m: str) -> str:
        v = mse(h, m)
        if v is None:
            return '---'
        s = fmt(v)
        if highlight.get((h, m), False):
            s = f"\\textbf{{{s}}}"
        return s

    # % change vs DLinear
    def rel(h: int, m: str) -> str:
        if m == 'DLinear':
            return ''
        d = mse(h, 'DLinear')
        v = mse(h, m)
        if d is None or v is None or d <= 0:
            return ''
        pct = (v - d) / d * 100.0
        return f" ({pct:+.1f}\\%)"

    model_labels = {
        'DLinear': 'DLinear',
        'PatchTST': 'PatchTST',
        'TiDE': 'TiDE',
        'TimeXer': 'TimeXer',
        'iTransformer': 'iTransformer',
        'PatchFusionBERT_BERTOnly': '\\bertonly',
        'PatchFusionBERT_v0': '\\modelvzero',
        'PatchFusionBERT_v2': '\\modelvtwo',
    }

    lines: list[str] = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Illness benchmark performance (MSE). Relative change vs.\\ DLinear is shown in parentheses. Values are mean over available runs/seeds in results\\_23-01.csv.}')
    lines.append('\\label{tab:illness}')
    lines.append('\\small')
    col_spec = 'l' + ('c' * len(ILL_H))
    lines.append(f'\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}')
    lines.append('\\toprule')
    header = 'Model' + ''.join([f' & $H={h}$' for h in ILL_H]) + ' \\\\'
    lines.append(header)
    lines.append('\\midrule')

    for m in MODELS_ILL:
        label = model_labels[m]
        row = [label]
        for h in ILL_H:
            row.append(cell(h, m) + rel(h, m))
        lines.append(' & '.join(row) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    return '\n'.join(lines) + '\n'


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    out_main = ROOT_OUT / 'table_main_results_by_dataset.tex'
    out_ill = ROOT_OUT / 'table_illness.tex'

    out_main.write_text(table_main_results(df), encoding='utf-8')
    out_ill.write_text(table_illness(df), encoding='utf-8')

    print('Wrote:')
    print(' -', out_main)
    print(' -', out_ill)


if __name__ == '__main__':
    main()
