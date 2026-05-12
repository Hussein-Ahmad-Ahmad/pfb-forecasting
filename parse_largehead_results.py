"""
Parse PatchTST_LargeHead ablation results from result_long_term_forecast.txt.
"""
import re, os

results_file = r'D:\Hussein-experiments\dynamic-ensemble-paper\Ts-library\Time-Series-Library\result_long_term_forecast.txt'

if not os.path.exists(results_file):
    print('Results file not found')
    exit()

text = open(results_file, encoding='utf-8').read()

# Find all LargeHead entries
pattern = r'(B1Chain_LargeHead_MS\d+_(\w+)_(\d+)_PatchTST_LargeHead[^\n]+)\n.*?mse:([\d.]+), mae:([\d.]+)'
matches = re.findall(pattern, text, re.DOTALL)

if not matches:
    # Try alternative pattern
    blocks = text.split('\n\n')
    for block in blocks:
        if 'LargeHead' in block:
            print('=== Found LargeHead block ===')
            print(block[:500])
else:
    print('=== PatchTST_LargeHead Results ===')
    for m in matches:
        full_id, dataset, horizon, mse, mae = m
        print(f'  {dataset} H={horizon}: MSE={mse}, MAE={mae}')
