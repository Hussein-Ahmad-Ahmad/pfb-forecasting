"""
Train PatchTST_LargeHead on ETTm2 / ETTh2 / Weather at H=192 (seed=2021).
Appends results to result_long_term_forecast.txt just like the other ablation runs.
"""
import subprocess, sys, os

os.chdir(r'D:\Hussein-experiments\dynamic-ensemble-paper\Ts-library\Time-Series-Library')

dataset_configs = {
    'ETTm2':   dict(data='ETTm2',  data_path='ETTm2.csv',   enc_in=7,  batch_size=16, label_len=96),
    'ETTh2':   dict(data='ETTh2',  data_path='ETTh2.csv',   enc_in=7,  batch_size=16, label_len=96),
    'Weather': dict(data='custom', data_path='weather.csv', enc_in=21, batch_size=8,  label_len=96),
}

seed = 2021
horizon = 192
results = {}

for ds_name, cfg in dataset_configs.items():
    model_id = f'B1Chain_LargeHead_MS{seed}_{ds_name}_{horizon}'
    args = [
        sys.executable, '-u', 'run.py',
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--model_id', model_id,
        '--model', 'PatchTST_LargeHead',
        '--data', cfg['data'],
        '--root_path', './data/',
        '--data_path', cfg['data_path'],
        '--features', 'M',
        '--seq_len', '336',
        '--label_len', str(cfg['label_len']),
        '--pred_len', str(horizon),
        '--enc_in', str(cfg['enc_in']),
        '--dec_in', str(cfg['enc_in']),
        '--c_out', str(cfg['enc_in']),
        '--patch_len', '16',
        '--stride', '8',
        '--d_model', '128',
        '--n_heads', '8',
        '--e_layers', '3',
        '--d_layers', '1',
        '--d_ff', '512',
        '--factor', '3',
        '--dropout', '0.1',
        '--itr', '1',
        '--train_epochs', '100',
        '--patience', '10',
        '--batch_size', str(cfg['batch_size']),
        '--learning_rate', '0.0001',
        '--num_workers', '0',
        '--seed', str(seed),
        '--des', 'b1_largehead',
    ]
    print(f'\n{"="*60}')
    print(f'Running: {ds_name} H={horizon} seed={seed}')
    print(f'{"="*60}')
    ret = subprocess.run(args, capture_output=False, text=True)
    results[ds_name] = 'OK' if ret.returncode == 0 else f'ERROR (code {ret.returncode})'
    print(f'>>> {ds_name}: {results[ds_name]}')

print('\n=== Summary ===')
for k, v in results.items():
    print(f'  {k}: {v}')
