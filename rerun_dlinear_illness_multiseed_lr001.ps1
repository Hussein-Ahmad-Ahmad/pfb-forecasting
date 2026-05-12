# Re-run DLinear Illness multi-seed experiments with the original single-seed LR=0.01
# Purpose: replace excluded legacy DLinear Illness multi-seed rows with protocol-aligned runs.

$ErrorActionPreference = "Stop"

$python = "C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe"
$scriptRoot = $PSScriptRoot
$runPy = Join-Path $scriptRoot "run.py"
$dataRoot = Join-Path $scriptRoot "data"

Set-Location $scriptRoot

& $python -c "import sys, torch; print('exe:', sys.executable); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "Python check failed" }

$seeds = @(2021, 2022, 2023)
$horizons = @(24, 48, 60)

foreach ($seed in $seeds) {
    foreach ($predLen in $horizons) {
        $labelLen = 18
        $modelId = "DLinearAlign_MS$seed`_Illness_$predLen"
        Write-Host "Running: $modelId"

        & $python -u $runPy `
            --task_name long_term_forecast --is_training 1 `
            --model_id $modelId --model DLinear --data custom `
            --root_path $dataRoot --data_path national_illness.csv `
            --features M --seq_len 104 --label_len $labelLen --pred_len $predLen `
            --enc_in 7 --dec_in 7 --c_out 7 `
            --itr 1 --train_epochs 100 --patience 10 --batch_size 16 `
            --learning_rate 0.01 --num_workers 0 --seed $seed `
            --des illness_align_lr001

        if ($LASTEXITCODE -ne 0) {
            throw "Run failed: $modelId"
        }
    }
}

Write-Host "All DLinear Illness alignment runs completed successfully."
