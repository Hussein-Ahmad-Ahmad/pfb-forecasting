from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / 'results_analysis'
RAW_CSV = OUT_DIR / 'b2_broader_robustness_raw.csv'
SUMMARY_CSV = OUT_DIR / 'b2_broader_robustness_summary.csv'
SUMMARY_MD = OUT_DIR / 'b2_broader_robustness_summary.md'
SUMMARY_TEX = OUT_DIR / 'b2_broader_robustness_summary.tex'

MODEL_ORDER = ['DLinear', 'PatchTST', 'PFB-Direct', 'PFB-Projected']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyze B2 broader robustness results')
    parser.add_argument('--input', type=Path, default=RAW_CSV)
    parser.add_argument('--summary_csv', type=Path, default=SUMMARY_CSV)
    parser.add_argument('--summary_md', type=Path, default=SUMMARY_MD)
    parser.add_argument('--summary_tex', type=Path, default=SUMMARY_TEX)
    return parser.parse_args()


def build_markdown(summary: pd.DataFrame) -> str:
    lines = ['# B2 Broader Robustness Matrix', '']
    lines.append('Scope: Weather and ETTm2, H=96 and H=192, clean plus random/block missingness.')
    lines.append('')

    for dataset in ['ETTm2', 'Weather']:
        dataset_df = summary[summary['Dataset'] == dataset]
        if dataset_df.empty:
            continue
        for horizon in sorted(dataset_df['Horizon'].unique()):
            subset = dataset_df[dataset_df['Horizon'] == horizon]
            lines.append(f'## {dataset} (H={horizon})')
            lines.append('')
            for corruption in ['random', 'block']:
                lines.append(f'### {corruption.title()} missingness')
                lines.append('')
                corr_df = subset[subset['Corruption'] == corruption]
                if corr_df.empty:
                    lines.append('- No results found.')
                    lines.append('')
                    continue
                for rate in sorted(corr_df['Missing_Rate'].unique()):
                    rate_df = corr_df[corr_df['Missing_Rate'] == rate].sort_values('Model')
                    best_idx = rate_df['MSE'].idxmin()
                    best_model = rate_df.loc[best_idx, 'Model']
                    lines.append(f'- Rate {rate:.1f}: best MSE = {best_model}')
                    for _, row in rate_df.iterrows():
                        lines.append(
                            '  {model}: mse={mse:.4f}, mae={mae:.4f}, DeltaMSE={delta:+.1f}%'.format(
                                model=row['Model'],
                                mse=row['MSE'],
                                mae=row['MAE'],
                                delta=row['MSE_DegradationPct'],
                            )
                        )
                    lines.append('')
    return '\n'.join(lines).strip() + '\n'


def build_latex(summary: pd.DataFrame) -> str:
    lines = []
    lines.append('\\begin{tabular}{lllrlll}')
    lines.append('\\toprule')
    lines.append('Dataset & Horizon & Corruption & Rate & Model & MSE & $\\Delta$MSE \\\\')
    lines.append('\\midrule')
    for _, row in summary.iterrows():
        lines.append(
            f"{row['Dataset']} & {int(row['Horizon'])} & {row['Corruption']} & {row['Missing_Rate']:.1f} & {row['Model']} & {row['MSE']:.4f} & {row['MSE_DegradationPct']:+.1f}\\% \\\\"
        )
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f'Missing raw robustness results: {args.input}')

    df = pd.read_csv(args.input)
    clean = df[df['Corruption'] == 'clean'][['Dataset', 'Horizon', 'Model', 'MSE', 'MAE']].rename(
        columns={'MSE': 'Clean_MSE', 'MAE': 'Clean_MAE'}
    )
    summary = df[df['Corruption'] != 'clean'].merge(clean, on=['Dataset', 'Horizon', 'Model'], how='left')
    summary['MSE_DegradationPct'] = (summary['MSE'] - summary['Clean_MSE']) / summary['Clean_MSE'] * 100.0
    summary['MAE_DegradationPct'] = (summary['MAE'] - summary['Clean_MAE']) / summary['Clean_MAE'] * 100.0
    summary['Model'] = pd.Categorical(summary['Model'], categories=MODEL_ORDER, ordered=True)
    summary = summary.sort_values(['Dataset', 'Horizon', 'Corruption', 'Missing_Rate', 'Model']).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    args.summary_md.write_text(build_markdown(summary), encoding='utf-8')
    args.summary_tex.write_text(build_latex(summary), encoding='utf-8')

    print(f'Wrote: {args.summary_csv}')
    print(f'Wrote: {args.summary_md}')
    print(f'Wrote: {args.summary_tex}')
    print('\nSummary:')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()