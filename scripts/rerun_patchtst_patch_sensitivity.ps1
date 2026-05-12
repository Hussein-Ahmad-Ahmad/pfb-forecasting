# Re-run PatchTST patch sensitivity after fixing the patch_len bug.
# Bug: PatchTST.__init__ ignored configs.patch_len and always used default 16.
# Fix: Added patch_len = getattr(configs, 'patch_len', patch_len) to PatchTST.py.
# This script overwrites the previous (invalid) PatchTST PatchSens results.
#
# 8 runs total: patch_lengths (8, 16, 32, 64) x datasets (ETTm1, ETTh1)
# Protocol: matches original patch_sensitivity_analysis.py
#   - seq_len=336, label_len=96, pred_len=192, d_model=128, e_layers=3, d_ff=512
#   - train_epochs=100, patience=10, lr=0.0001
# Run from Time-Series-Library/

$ErrorActionPreference = "Stop"
$python = "C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe"
$scriptRoot = "d:\Hussein-experiments\dynamic-ensemble-paper\Ts-library\Time-Series-Library"
Set-Location $scriptRoot

Write-Host "Re-running PatchTST patch sensitivity (bug fix: patch_len now reads from configs)"
Write-Host "Runs: patch_len in {8,16,32,64} x dataset in {ETTm1,ETTh1}"
Write-Host ""

$patch_lengths = @(8, 16, 32, 64)
$datasets = @("ETTm1", "ETTh1")
$data_map = @{ "ETTm1" = "ETTm1"; "ETTh1" = "ETTh1" }

foreach ($dataset in $datasets) {
    foreach ($patch_len in $patch_lengths) {
        $stride = [math]::Max($patch_len / 2, 1)
        $model_id = "PatchSens_P${patch_len}_${dataset}_H192"
        $des = "PatchSensitivity_P${patch_len}"
        $data_type = $data_map[$dataset]

        Write-Host "Running PatchTST | ${dataset} | patch_len=$patch_len stride=$stride"

        & $python -u run.py `
            --task_name long_term_forecast --is_training 1 `
            --root_path ./data/ --data_path "${dataset}.csv" `
            --model_id $model_id --model PatchTST --data $data_type `
            --features M --seq_len 336 --label_len 96 --pred_len 192 `
            --enc_in 7 --dec_in 7 --c_out 7 `
            --d_model 128 --n_heads 8 --e_layers 3 --d_layers 1 --d_ff 512 `
            --factor 3 --dropout 0.1 `
            --patch_len $patch_len --stride $stride `
            --train_epochs 100 --patience 10 --batch_size 32 --learning_rate 0.0001 `
            --itr 1 --num_workers 0 --des $des

        if ($LASTEXITCODE -ne 0) { throw "Failed: PatchTST ${dataset} patch_len=$patch_len" }
        Write-Host "  Done."
    }
}

Write-Host ""
Write-Host "All 8 PatchTST patch sensitivity runs complete."
Write-Host "Next: python analyze_patch_sensitivity.py"
