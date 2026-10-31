"""Verify the published scalar records and paired effects without training."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / 'results_summary'


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_metrics(path, row, fields=('mae', 'mse')):
    values = np.load(path, allow_pickle=False)
    assert values.shape == (5,) and np.isfinite(values).all(), path
    indices = dict(mae=0, mse=1, rmse=2, mape=3, mspe=4)
    for field in fields:
        assert math.isclose(float(values[indices[field]]), float(row[field]), rel_tol=1e-11, abs_tol=1e-12), (path, field)


def main():
    manifest = json.loads((SUMMARY / 'ARTIFACT_SHA256.json').read_text())
    for name, digest in manifest['files'].items():
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT.resolve()), name
        assert sha(path) == digest, name
    metadata = json.loads((SUMMARY / 'PROVENANCE.json').read_text())
    review = rows(SUMMARY / 'review_campaigns/completed_results_181026.csv')
    assert len(review) == 3942
    hardware = Counter(r['run_by_gpu_watermark'] for r in review)
    assert hardware == {'EXP_RUN_BY_RTX4090_GPU': 3807, '': 135}
    for row in review:
        run_name = Path(row['metrics_path'].replace('\\', '/')).parent.name
        check_metrics(SUMMARY / 'review_campaigns/metrics' / run_name / 'metrics.npy', row,
                      ('mae', 'mse', 'rmse', 'mape', 'mspe'))
    verified = rows(SUMMARY / 'review_campaigns/verified_rtx4090_results.csv')
    assert verified == [r for r in review if r['run_by_gpu_watermark'] == 'EXP_RUN_BY_RTX4090_GPU']

    prospective = rows(SUMMARY / 'controlled_addons/addon_results_231026.csv')
    assert len(prospective) == 1039
    assert Counter(r['study'] for r in prospective) == {'hpo': 504, 'hpo_test': 210, 'projected': 165, 'preprocessing': 160}
    fingerprint_checks = set()
    for row in prospective:
        relative = row['metrics_path'].replace('\\', '/').split('controlled_addons_v1/', 1)[1]
        metric = SUMMARY / 'controlled_addons' / relative
        check_metrics(metric, row)
        receipt = json.loads(metric.with_name('complete.json').read_text())
        assert receipt['spec']['run_id'] == row['run_id']
        for name, digest in receipt['sha256'].items():
            if name == 'checkpoint.pth':
                continue  # Weights are explicitly outside the scalar export.
            assert sha(metric.parent / name) == digest
        provenance = json.loads(metric.with_name('run_provenance.json').read_text())
        assert row['run_by_gpu_watermark'] == provenance['run_by_gpu_watermark'] == 'EXP_RUN_BY_RTX4090_GPU'
        assert provenance['actual_cuda_used'] and '4090' in provenance['actual_cuda_device']
        study = 'hpo' if row['study'].startswith('hpo') else row['study']
        environment = metadata['recorded_protocols'][study]['environment']
        for name, digest in environment['source_sha256'].items():
            if (name, digest) not in fingerprint_checks:
                assert sha(ROOT / metadata['exact_source_snapshot'] / name) == digest
                fingerprint_checks.add((name, digest))
    assert len(fingerprint_checks) == 50

    effects = rows(SUMMARY / 'controlled_addons/addon_paired_effects_231026.csv')
    assert len(effects) == 360
    for effect in effects:
        def find(side):
            group = effect['study']
            key = effect[side]
            candidates = [r for r in prospective if r['dataset'] == effect['dataset'] and r['seed'] == effect['seed']]
            if group == 'projected':
                candidates = [r for r in candidates if r['study'] == 'projected' and r['variant'] == key]
            elif group == 'hpo':
                candidates = [r for r in candidates if r['study'] == 'hpo_test' and r['family'] == key]
            else:
                family = effect['description'] if group == 'normalization' else key
                variant = key if group == 'normalization' else 'norm_on'
                candidates = [r for r in candidates if r['study'] == 'preprocessing' and r['family'] == family and r['variant'] == variant]
            assert len(candidates) == 1, effect
            return candidates[0]
        left, right = find('lhs'), find('rhs')
        for key in ('data_sha256', 'source_sha256', 'software_sha256', 'protocol_sha256'):
            assert left[key] == right[key]
        delta = float(left['mse']) - float(right['mse'])
        assert math.isclose(delta, float(effect['delta_mse']), abs_tol=1e-12)
        assert math.isclose(100*delta/float(right['mse']), float(effect['effect_pct']), abs_tol=1e-10)
        assert math.isclose(float(left['mae'])-float(right['mae']), float(effect['delta_mae']), abs_tol=1e-12)

    factorial = rows(SUMMARY / 'factorial_interval_audit.csv')
    cells = rows(SUMMARY / 'factorial_cell_summary.csv')
    assert len(factorial) == 42 and len(cells) == 24
    counts = {}
    for row in factorial:
        reference = math.fsum(float(r[row['metric']+'_mean']) for r in cells if r['Dataset'] == row['dataset']) / 8
        assert reference == float(row['reference_metric_mean'])
        low, high, band = float(row['ci95_low']), float(row['ci95_high']), .01*reference
        classification = 'adverse' if low > band else 'favorable' if high < -band else 'band-contained' if low >= -band and high <= band else 'indeterminate'
        assert classification == row['interval_classification']
        counts.setdefault(row['metric'], Counter())[classification] += 1
        assert row['experiment_run_on'] == 'NVIDIA GeForce RTX 2080 Ti'
    assert counts['MSE'] == {'indeterminate': 19, 'band-contained': 2}
    assert counts['MAE'] == {'indeterminate': 17, 'band-contained': 3, 'adverse': 1}
    winners = Counter(r['winner'] for r in rows(SUMMARY / 'capacity_task_winners.csv'))
    assert winners == {'PFB-Direct': 5, 'PFB-Projected': 4}
    print(json.dumps(dict(checksummed_files=len(manifest['files']), review_metric_arrays=3942,
        prospective_metric_arrays=1039, exact_source_fingerprints=50, paired_effects=360,
        factorial_contrasts=42, hardware_labels_preserved=True, training_launched=False), indent=2))


if __name__ == '__main__':
    main()
