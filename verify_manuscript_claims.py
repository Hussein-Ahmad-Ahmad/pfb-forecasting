import pandas as pd
import numpy as np

VALID_H_MAIN = [96, 192, 336]
# Manuscript reports Illness at horizons with complete coverage across the
# included models in results_23-01.csv.
VALID_H_ILLNESS = [24, 48, 60]

PFB_V0 = 'PatchFusionBERT_v0'
PFB_V2 = 'PatchFusionBERT_v2'
BERT = 'PatchFusionBERT_BERTOnly'

BASELINES = ['DLinear', 'PatchTST', 'TiDE', 'TimeXer', 'iTransformer']
ALL = [PFB_V0, PFB_V2, BERT] + BASELINES


def clean_model_name(model: str) -> str:
    mapping = {
        PFB_V0: 'PFB-v0',
        PFB_V2: 'PFB-v2',
        BERT: 'BERT',
    }
    return mapping.get(model, model)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # Mean across repeated runs/seeds per dataset/horizon/model
    return (
        df.groupby(['Dataset', 'Horizon', 'Model'], as_index=False)
        .agg(MSE=('MSE', 'mean'), MAE=('MAE', 'mean'), n=('MSE', 'size'))
    )


def best_second_counts(agg_df: pd.DataFrame) -> tuple[int, dict, dict]:
    best = {m: 0 for m in ALL}
    second = {m: 0 for m in ALL}
    total = 0

    for (ds, h), sub in agg_df.groupby(['Dataset', 'Horizon']):
        sub = sub[sub['Model'].isin(ALL)].sort_values('MSE')
        if len(sub) < 2:
            continue
        total += 1
        best[sub.iloc[0]['Model']] += 1
        second[sub.iloc[1]['Model']] += 1

    return total, best, second


def win_rates(agg_df: pd.DataFrame, variant: str) -> dict:
    out = {}
    v = agg_df[agg_df.Model == variant][['Dataset', 'Horizon', 'MSE']].rename(columns={'MSE': 'v'})
    for base in BASELINES:
        b = agg_df[agg_df.Model == base][['Dataset', 'Horizon', 'MSE']].rename(columns={'MSE': 'b'})
        j = v.merge(b, on=['Dataset', 'Horizon'], how='inner')
        if len(j) == 0:
            out[base] = (np.nan, 0)
        else:
            out[base] = ((j.v < j.b).mean() * 100.0, len(j))
    return out


def main() -> None:
    df = pd.read_csv('results_23-01.csv')
    df = df[df['Model'].isin(ALL)].copy()

    # MAIN suite stats (H=96/192/336)
    main_df = df[df['Horizon'].isin(VALID_H_MAIN)].copy()
    main_agg = aggregate(main_df)

    print('=== DATA OVERVIEW ===')
    print('rows:', len(df))
    print('datasets:', sorted(df.Dataset.unique()))
    print('horizons:', sorted(df.Horizon.unique()))
    print('models:', sorted(df.Model.unique()))

    print('\n=== MAIN SUITE (H=96/192/336) ===')
    print('datasets in main suite:', sorted(main_agg.Dataset.unique()))
    print('settings present:', main_agg.groupby(['Dataset', 'Horizon']).size().shape[0])

    total_settings, best, second = best_second_counts(main_agg)
    print('settings with >=2 models:', total_settings)
    print('best counts (top):')
    for m, c in sorted(best.items(), key=lambda x: (-x[1], x[0])):
        if c:
            print(f'  {clean_model_name(m):>10s}: {c}')

    # Overall average MSE
    overall = main_agg.groupby('Model').MSE.mean().sort_values()
    print('\nOverall avg MSE ranking (main suite):')
    for i, (m, v) in enumerate(overall.items(), start=1):
        print(f'  #{i:>2d} {clean_model_name(m):>10s}: {v:.6f}')

    # Win rates
    for variant in [PFB_V0, PFB_V2]:
        print(f"\nWin rates vs baselines for {clean_model_name(variant)} (main suite)")
        wr = win_rates(main_agg, variant)
        for base, (rate, n) in wr.items():
            r = 'n/a' if np.isnan(rate) else f'{rate:.1f}%'
            print(f'  vs {clean_model_name(base):<12s}: {r:>6s} over {n} settings')

    # Illness-only check (exists?)
    illness_df = df[(df['Dataset'] == 'Illness') & (df['Horizon'].isin(VALID_H_ILLNESS))].copy()
    if len(illness_df) == 0:
        print('\n=== ILLNESS (H=24/36/48/60) ===')
        print('No Illness rows for these models in results_23-01.csv after filtering to ALL models.')
    else:
        ill_agg = aggregate(illness_df)
        print('\n=== ILLNESS (H=24/36/48/60) ===')
        print('models present:', sorted(ill_agg.Model.unique()))
        print('horizons present:', sorted(ill_agg.Horizon.unique()))
        # Example reductions vs DLinear for PFB-v0
        def mse(ds, h, model):
            sub = ill_agg[(ill_agg.Dataset == ds) & (ill_agg.Horizon == h) & (ill_agg.Model == model)]
            return float(sub.MSE.iloc[0]) if len(sub) else np.nan

        for h in sorted(ill_agg.Horizon.unique()):
            d = mse('Illness', h, 'DLinear')
            v0 = mse('Illness', h, PFB_V0)
            if np.isfinite(d) and np.isfinite(v0) and d > 0:
                red = (d - v0) / d * 100.0
                print(f'  H={h}: DLinear={d:.4f}, PFB-v0={v0:.4f}, reduction={red:.1f}%')


if __name__ == '__main__':
    main()
