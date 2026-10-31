"""Reproduce the absolute-MSE topology plot from its retained numerical CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--values-csv', type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.source, float_precision='round_trip')
    required = {'dataset', 'horizon', 'pfb_direct_mse', 'serial_mse'}
    if not required.issubset(frame) or len(frame) != 18 or frame.duplicated(['dataset', 'horizon']).any():
        raise ValueError('Expected exactly 18 distinct retained dataset/horizon pairs.')
    if not np.isfinite(frame[['pfb_direct_mse', 'serial_mse']].to_numpy()).all():
        raise ValueError('Nonfinite MSE input.')
    frame['absolute_delta_mse'] = frame.pfb_direct_mse - frame.serial_mse
    order = frame.sort_values('absolute_delta_mse', ascending=False, kind='stable')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.size': 11, 'axes.titlesize': 14}):
        fig, ax = plt.subplots(figsize=(9.5, 8), dpi=300)
        bars = ax.barh(order.dataset + '-' + order.horizon.astype(str), order.absolute_delta_mse, color='#1f77b4')
        ax.invert_yaxis()
        ax.axvline(0, color='#1f77b4', linewidth=1)
        ax.set_title('Serial versus parallel topology across 18 tasks')
        ax.set_xlabel(r'$\Delta$MSE = PFB-Direct $-$ serial matched control')
        assert not ax.texts
        np.testing.assert_array_equal([bar.get_width() for bar in bars], order.absolute_delta_mse)
        fig.tight_layout()
        fig.savefig(args.output, dpi=300)
        plt.close(fig)
    if args.values_csv:
        args.values_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.values_csv, index=False, float_format='%.15g')
    print('Replotted absolute MSE differences; numerical inputs and historical hardware labels are unchanged.')


if __name__ == '__main__':
    main()
