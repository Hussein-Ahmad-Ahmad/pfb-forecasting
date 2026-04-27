from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch


RESULT_FILES = [
    Path('result_long_term_forecast.txt'),
    Path('..') / 'result_long_term_forecast.txt',
]

OUT_DIR = Path('results_analysis')
OUT_CSV = OUT_DIR / 'b1_component_ablation.csv'
OUT_MD = OUT_DIR / 'b1_component_ablation.md'
OUT_TEX = OUT_DIR / 'b1_component_ablation.tex'

COMPONENT_ORDER = ['PatchOnly', 'RefineOnly', 'FusionOnly', 'Full']
COMPONENT_LABELS = {
    'PatchOnly': 'Backbone only',
    'RefineOnly': 'Backbone + refinement (no fusion)',
    'FusionOnly': 'Backbone + fusion (no deeper refinement)',
    'Full': 'Full model',
}

DATASET_CONFIGS = {
    'ETTm2': {'enc_in': 7},
    'ETTh2': {'enc_in': 7},
    'Weather': {'enc_in': 21},
}


@dataclass(frozen=True)
class ParsedRun:
    seed: int
    dataset: str
    horizon: int
    component: str
    mse: float
    mae: float
    source: str
    line_no: int


class _Args:
    def __init__(self, *, enc_in: int, pfb_k: int = 0):
        self.task_name = 'long_term_forecast'
        self.seq_len = 336
        self.pred_len = 192
        self.label_len = 96
        self.enc_in = enc_in
        self.dec_in = enc_in
        self.c_out = enc_in
        self.d_model = 128
        self.n_heads = 8
        self.e_layers = 3
        self.d_layers = 1
        self.d_ff = 512
        self.factor = 3
        self.dropout = 0.1
        self.activation = 'gelu'
        self.embed = 'timeF'
        self.distil = True
        self.patch_len = 16
        self.stride = 8
        self.pfb_k = pfb_k


def _iter_runs(path: Path):
    lines = path.read_text(encoding='utf-8').splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if not s.startswith('long_term_forecast_'):
            continue
        if 'B1Chain_MS' not in s:
            continue
        metrics_text = s
        if 'mse:' not in metrics_text or 'mae:' not in metrics_text:
            maybe_next = lines[i + 1].strip() if i + 1 < len(lines) else ''
            metrics_text = f'{metrics_text} {maybe_next}'.strip()
        mse_m = re.search(r'mse:([\d.]+)', metrics_text)
        mae_m = re.search(r'mae:([\d.]+)', metrics_text)
        if not (mse_m and mae_m):
            continue
        yield i, s, float(mse_m.group(1)), float(mae_m.group(1))


def _parse_run(exp_name: str, mse: float, mae: float, source: str, line_no: int) -> Optional[ParsedRun]:
    match = re.search(r'B1Chain_MS(\d{4})_(ETTm2|ETTh2|Weather)_(\d+)_(PatchOnly|RefineOnly|FusionOnly|Full)', exp_name)
    if not match:
        return None
    return ParsedRun(
        seed=int(match.group(1)),
        dataset=match.group(2),
        horizon=int(match.group(3)),
        component=match.group(4),
        mse=mse,
        mae=mae,
        source=source,
        line_no=line_no,
    )


def _count_params(component: str, enc_in: int) -> int:
    from models import PatchFusionBERT_PatchOnly, PatchFusionBERT_RefineOnly, PatchFusionBERT_v0

    if component == 'PatchOnly':
        model = PatchFusionBERT_PatchOnly.Model(_Args(enc_in=enc_in, pfb_k=0)).float()
    elif component == 'RefineOnly':
        model = PatchFusionBERT_RefineOnly.Model(_Args(enc_in=enc_in, pfb_k=3)).float()
    elif component == 'FusionOnly':
        model = PatchFusionBERT_v0.Model(_Args(enc_in=enc_in, pfb_k=0)).float()
    elif component == 'Full':
        model = PatchFusionBERT_v0.Model(_Args(enc_in=enc_in, pfb_k=3)).float()
    else:
        raise ValueError(component)

    return int(sum(p.numel() for p in model.parameters()))


def _build_markdown(summary: pd.DataFrame) -> str:
    lines = ['# B1 Component Ablation Chain', '', 'Chain:', '']
    for component in COMPONENT_ORDER:
        lines.append(f"- {component}: {COMPONENT_LABELS[component]}")
    lines.append('')

    for dataset in ['ETTm2', 'ETTh2', 'Weather']:
        subset = summary[summary['Dataset'] == dataset]
        if subset.empty:
            continue
        lines.append(f'## {dataset} (H=192)')
        lines.append('')
        for _, row in subset.iterrows():
            lines.append(
                '- {label}: params={params:,}, mse={mse:.4f} ± {mse_std:.4f}, mae={mae:.4f} ± {mae_std:.4f}'.format(
                    label=COMPONENT_LABELS[row['Component']],
                    params=int(row['Params']),
                    mse=row['MSE_mean'],
                    mse_std=row['MSE_std'],
                    mae=row['MAE_mean'],
                    mae_std=row['MAE_std'],
                )
            )
        lines.append('')
    return '\n'.join(lines).strip() + '\n'


def _build_latex(summary: pd.DataFrame) -> str:
    lines = []
    lines.append('\\begin{tabular}{llrrrr}')
    lines.append('\\toprule')
    lines.append('Dataset & Component & Params & MSE & MSE std & MAE \\\\')
    lines.append('\\midrule')
    for _, row in summary.iterrows():
        lines.append(
            f"{row['Dataset']} & {row['Component']} & {int(row['Params'])} & {row['MSE_mean']:.4f} & {row['MSE_std']:.4f} & {row['MAE_mean']:.4f} \\\\"
        )
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    return '\n'.join(lines) + '\n'


def main() -> None:
    rows: List[ParsedRun] = []
    for path in RESULT_FILES:
        if not path.exists():
            continue
        for line_no, exp_name, mse, mae in _iter_runs(path):
            parsed = _parse_run(exp_name, mse, mae, str(path), line_no)
            if parsed is not None:
                rows.append(parsed)

    if not rows:
        raise RuntimeError('No B1 component ablation runs found in the result logs.')

    df = pd.DataFrame([r.__dict__ for r in rows])
    df = df.sort_values(['source', 'line_no']).groupby(['seed', 'dataset', 'horizon', 'component'], as_index=False).tail(1)

    params_map: Dict[tuple[str, str], int] = {}
    for dataset in sorted(df['dataset'].unique()):
        enc_in = DATASET_CONFIGS[dataset]['enc_in']
        for component in COMPONENT_ORDER:
            params_map[(dataset, component)] = _count_params(component, enc_in)

    df['Params'] = df.apply(lambda row: params_map[(row['dataset'], row['component'])], axis=1)

    summary = (
        df.groupby(['dataset', 'horizon', 'component', 'Params'], as_index=False)
        .agg(
            MSE_mean=('mse', 'mean'),
            MSE_std=('mse', 'std'),
            MAE_mean=('mae', 'mean'),
            MAE_std=('mae', 'std'),
            Seeds=('seed', 'nunique'),
        )
        .rename(columns={'dataset': 'Dataset', 'horizon': 'Horizon', 'component': 'Component'})
    )
    summary['MSE_std'] = summary['MSE_std'].fillna(0.0)
    summary['MAE_std'] = summary['MAE_std'].fillna(0.0)
    summary['Component'] = pd.Categorical(summary['Component'], categories=COMPONENT_ORDER, ordered=True)
    summary = summary.sort_values(['Dataset', 'Horizon', 'Component']).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(_build_markdown(summary), encoding='utf-8')
    OUT_TEX.write_text(_build_latex(summary), encoding='utf-8')

    print(f'Wrote: {OUT_CSV}')
    print(f'Wrote: {OUT_MD}')
    print(f'Wrote: {OUT_TEX}')
    print('\nSummary:')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()