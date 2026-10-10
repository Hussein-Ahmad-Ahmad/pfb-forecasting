"""CPU-only checks for combined forecasting campaign interfaces and provenance."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from layers.OuterWindowNorm import ReversibleOuterWindowNorm
from utils.result_provenance import RTX4090_WATERMARK, build_result_provenance


class TinyForecaster(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, *args, **kwargs):
        return self.linear(x)


def experiment(directory, **overrides):
    # Confirm each automatic cleanup target is a generated child of the temp root.
    assert Path(directory).resolve().parent == Path(tempfile.gettempdir()).resolve()
    exp = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
    args = dict(model='Tiny', use_gpu=False, gpu_type='cuda', device=torch.device('cpu'),
                checkpoints=str(Path(directory) / 'checkpoints'), artifacts_root=str(directory),
                learning_rate=1e-4, optimizer='Adam', weight_decay=0.0,
                train_epochs=1, patience=2, use_amp=False, pred_len=2, label_len=2,
                features='M', inverse=False, use_dtw=False, lradj='type1', skip_prediction_arrays=True,
                selection_only=False, validation_only=False, defer_test_until_after_training=False,
                use_multi_gpu=False, outer_window_norm=False, uniform_outer_normalization=False)
    args.update(overrides)
    exp.args = SimpleNamespace(**args)
    exp.device = torch.device('cpu')
    exp.model = TinyForecaster()
    exp.model_dict = {'Tiny': SimpleNamespace(Model=TinyForecaster)}
    exp.requested_splits = []
    def data(flag):
        exp.requested_splits.append(flag)
        x = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4, 1)
        return SimpleNamespace(scale=False), [(x, x.clone(), torch.zeros_like(x), torch.zeros_like(x))]
    exp._get_data = data
    return exp


class ReviewCampaignTests(unittest.TestCase):
    def test_old_and_new_cli_flags_coexist(self):
        proc = subprocess.run([sys.executable, str(ROOT / 'run.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for flag in ('--selection_only', '--report_split', '--outer_window_norm', '--validation_only',
                     '--optimizer', '--artifacts_root', '--skip_prediction_arrays', '--timesqueeze_tau'):
            self.assertIn(flag, proc.stdout)

    def test_selection_cli_rejects_test_reporting(self):
        proc = subprocess.run([sys.executable, str(ROOT / 'run.py'), '--task_name', 'long_term_forecast',
                               '--is_training', '1', '--model_id', 'unit', '--model', 'PatchTST',
                               '--data', 'ETTh1', '--selection_only'], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn('--selection_only requires --report_split val', proc.stderr)

    def test_each_validation_control_withholds_test(self):
        for flag in ('selection_only', 'validation_only', 'defer_test_until_after_training'):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                exp = experiment(directory, **{flag: True})
                with contextlib.redirect_stdout(io.StringIO()):
                    exp.train('unit')
                self.assertEqual(exp.requested_splits, ['train', 'val'])
                self.assertTrue(np.isfinite(exp.best_validation_loss))
                self.assertEqual(exp.best_validation_epoch, 1)

    def test_normal_training_retains_test_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = experiment(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                exp.train('unit')
            self.assertEqual(exp.requested_splits, ['train', 'val', 'test'])

    def test_validation_artifacts_and_prediction_array_option(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = experiment(directory)
            original_open = open
            def local_open(name, *args, **kwargs):
                if name == 'result_long_term_forecast.txt':
                    name = str(Path(directory) / name)
                return original_open(name, *args, **kwargs)
            with patch('builtins.open', side_effect=local_open), patch('exp.exp_long_term_forecasting.visual'), \
                    contextlib.redirect_stdout(io.StringIO()):
                exp.test('unit', split='val')
            output = Path(directory) / 'val_results/unit'
            self.assertEqual(exp.requested_splits, ['val'])
            self.assertTrue((output / 'metrics.npy').is_file())
            self.assertTrue((output / 'run_provenance.json').is_file())
            self.assertFalse((output / 'pred.npy').exists())
            self.assertFalse((Path(directory) / 'results').exists())
            self.assertEqual(json.loads((output / 'run_provenance.json').read_text())['run_by_gpu_watermark'], 'EXP_RUN_BY_CPU')

    def test_remote_optimizer_options_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = experiment(directory, optimizer='AdamW', weight_decay=0.01)
            optimizer = exp._select_optimizer()
            self.assertIsInstance(optimizer, torch.optim.AdamW)
            self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.01)

    def test_normalization_is_applied_once(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = experiment(directory, outer_window_norm=True, uniform_outer_normalization=True)
            model = exp._build_model()
            self.assertIsInstance(model, ReversibleOuterWindowNorm)
            self.assertIsInstance(model.model, TinyForecaster)

    def test_watermark_reflects_actual_execution_device(self):
        for use_gpu, device, expected in ((True, 'NVIDIA GeForce RTX 4090', RTX4090_WATERMARK),
                                          (True, 'NVIDIA GeForce RTX 2080 Ti', 'EXP_RUN_BY_OTHER_CUDA_GPU'),
                                          (False, 'NVIDIA GeForce RTX 4090', 'EXP_RUN_BY_CPU')):
            with self.subTest(use_gpu=use_gpu, device=device), patch('torch.cuda.is_available', return_value=True), \
                    patch('torch.cuda.current_device', return_value=0), patch('torch.cuda.get_device_name', return_value=device):
                result = build_result_provenance(SimpleNamespace(use_gpu=use_gpu, gpu_type='cuda'), 'test')
                self.assertEqual(result['run_by_gpu_watermark'], expected)


if __name__ == '__main__':
    unittest.main()
