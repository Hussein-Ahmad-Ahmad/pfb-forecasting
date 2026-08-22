"""
Dataset-regime diagnostic table for all 7 datasets.
Computes:
- Mean lag-1 autocorrelation (average across all target variables, train split)
- Dominant period (peak FFT period in hours/samples)
- Spectral entropy (0=pure tone, 1=white noise; avg across variables)
- Num variables
- Dataset length (train rows)
"""
import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'D:\Hussein-experiments\dynamic-ensemble-paper\Ts-library\Time-Series-Library\data'

# Dataset configs: name, file, target_cols (None = all numeric except date), train_frac
datasets = [
    dict(name='ETTh1',    file='ETTh1.csv',            date_col='date', train_frac=0.7),
    dict(name='ETTh2',    file='ETTh2.csv',            date_col='date', train_frac=0.7),
    dict(name='ETTm1',    file='ETTm1.csv',            date_col='date', train_frac=0.7),
    dict(name='ETTm2',    file='ETTm2.csv',            date_col='date', train_frac=0.7),
    dict(name='Exchange', file='exchange_rate.csv',    date_col='date', train_frac=0.7),
    dict(name='Weather',  file='weather.csv',          date_col='date', train_frac=0.7),
    dict(name='Illness',  file='national_illness.csv', date_col='date', train_frac=0.7),
]

def lag1_autocorr(series):
    """Pearson lag-1 autocorrelation."""
    s = np.array(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 3:
        return np.nan
    return float(np.corrcoef(s[:-1], s[1:])[0, 1])

def spectral_entropy(series, n_freq_bins=50):
    """
    Normalised spectral entropy:
    0 = single pure tone, 1 = white noise (uniform power spectrum).
    """
    s = np.array(series, dtype=float)
    s = s - np.nanmean(s)
    s = np.where(np.isnan(s), 0, s)
    ps = np.abs(np.fft.rfft(s)) ** 2
    ps = ps[1:]  # remove DC
    if ps.sum() == 0:
        return np.nan
    ps = ps / ps.sum()
    # bin into n_freq_bins to reduce noise
    if len(ps) > n_freq_bins:
        ps = pd.Series(ps).groupby(np.arange(len(ps)) * n_freq_bins // len(ps)).mean().values
    ps = ps / ps.sum()
    se = float(-np.sum(ps * np.log(ps + 1e-12)) / np.log(len(ps)))
    return se

def dominant_period(series, min_period=2, max_period=2000):
    """Return dominant period in samples, searching within [min_period, max_period]."""
    s = np.array(series, dtype=float)
    s = s - np.nanmean(s)
    s = np.where(np.isnan(s), 0, s)
    ps = np.abs(np.fft.rfft(s)) ** 2
    freqs = np.fft.rfftfreq(len(s))
    # zero out DC and frequencies outside window
    periods = np.where(freqs[1:] > 0, 1.0 / freqs[1:], np.inf)
    mask = (periods >= min_period) & (periods <= max_period)
    masked_ps = ps[1:].copy()
    masked_ps[~mask] = 0
    if masked_ps.max() == 0:
        return None  # no clear period in range
    idx = np.argmax(masked_ps)
    return round(periods[idx])

rows = []
for cfg in datasets:
    import os
    path = os.path.join(DATA_DIR, cfg['file'])
    df = pd.read_csv(path)
    date_col = cfg['date_col']
    num_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    n_train = int(len(df) * cfg['train_frac'])
    train = df[num_cols].iloc[:n_train]

    ac1_vals  = [lag1_autocorr(train[c])     for c in num_cols]
    se_vals   = [spectral_entropy(train[c])  for c in num_cols]
    dp_vals   = [dominant_period(train[c])   for c in num_cols]

    mean_ac1 = float(np.nanmean(ac1_vals))
    mean_se  = float(np.nanmean(se_vals))
    # most common dominant period (mode)
    dp_arr   = [x for x in dp_vals if x is not None and not np.isnan(x)]
    if dp_arr:
        dp_counts = pd.Series(dp_arr).value_counts()
        dom_period = int(dp_counts.index[0])
    else:
        dom_period = None

    rows.append({
        'Dataset': cfg['name'],
        'N_vars': len(num_cols),
        'Train_rows': n_train,
        'Mean_lag1_AC': round(mean_ac1, 3),
        'Spectral_entropy': round(mean_se, 3),
        'Dominant_period_samples': dom_period,
    })
    print(f"{cfg['name']:10s}: n_vars={len(num_cols):3d}  train_rows={n_train:6d}  "
          f"lag1_AC={mean_ac1:.3f}  spec_ent={mean_se:.3f}  dom_period={dom_period}")

out = pd.DataFrame(rows)
out.to_csv('dataset_characteristics_all7.csv', index=False)
print("\nSaved to dataset_characteristics_all7.csv")
print(out.to_string(index=False))
