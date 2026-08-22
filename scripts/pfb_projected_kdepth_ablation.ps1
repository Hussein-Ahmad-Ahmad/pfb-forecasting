# Experiment: K-depth refinement ablation for PFB-Projected
# Mirrors the PFB-Direct K-depth ablation with PFB-Projected.
# Runs:
#   - Datasets: ETTh2, Weather
#   - Horizons: 96, 192, 336
#   - K: 1, 2, 3
#   - Seed: single (2021) to match the direct-fusion ablation for comparison
# Protocol identical to capmatch_controls.ps1 (lr=1e-4, epochs=100, patience=10).

$ErrorActionPreference = "Stop"

$python = "python"
$fallbackPython = "C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe"

$scriptRoot = $PSScriptRoot
$runPy = Join-Path $scriptRoot "run.py"
$dataRoot = Join-Path $scriptRoot "data"

Set-Location $scriptRoot

& $python -c "import sys, torch; print('exe:', sys.executable); print('cuda:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    $python = $fallbackPython
    & $python -c "import sys, torch; print('exe:', sys.executable); print('cuda:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw "Python check failed" }
}

$task          = "long_term_forecast"
$itr           = 1
$train_epochs  = 100
$patience      = 10
$learning_rate = 0.0001
$dropout       = 0.1
$n_heads       = 8
$factor        = 3
$patch_len     = 16
$stride        = 8
$d_model       = 128
$e_layers      = 3
$d_ff          = 512
$seed          = 2021
$ks            = @(1, 2, 3)
$horizons      = @(96, 192, 336)

function Invoke-ProjectedKDepthRun {
    param(
        [string]$Data,
        [string]$RootPath,
        [string]$DataPath,
        [string]$DatasetTag,
        [int]$SeqLen,
        [int]$LabelLen,
        [int]$PredLen,
        [int]$EncIn,
        [int]$BatchSize,
        [int]$Seed,
        [int]$K
    )

    $model   = "PFB-Projected"
    $modelId = "KDepthProjected_MS$Seed`_${DatasetTag}_$PredLen`_K$K"

    Write-Host "Running: $modelId  model=$model  pred_len=$PredLen  K=$K"

    & $python -u $runPy `
        --task_name $task --is_training 1 `
        --model_id $modelId --model $model --data $Data `
        --root_path $RootPath --data_path $DataPath `
        --features M --seq_len $SeqLen --label_len $LabelLen --pred_len $PredLen `
        --enc_in $EncIn --dec_in $EncIn --c_out $EncIn `
        --patch_len $patch_len --stride $stride `
        --d_model $d_model --n_heads $n_heads --e_layers $e_layers --d_layers 1 --d_ff $d_ff --factor $factor `
        --dropout $dropout `
        --pfb_k $K `
        --itr $itr --train_epochs $train_epochs --patience $patience --batch_size $BatchSize --learning_rate $learning_rate `
        --num_workers 0 --seed $Seed `
        --des kdepth_projected_ablation

    if ($LASTEXITCODE -ne 0) { throw "Training failed for $modelId" }
}

# ETTh2
foreach ($h in $horizons) {
    foreach ($k in $ks) {
        Invoke-ProjectedKDepthRun -Data "ETTh2" -RootPath $dataRoot -DataPath "ETTh2.csv" -DatasetTag "ETTh2" `
            -SeqLen 336 -LabelLen 96 -PredLen $h -EncIn 7 -BatchSize 16 -Seed $seed -K $k
    }
}

# Weather
foreach ($h in $horizons) {
    foreach ($k in $ks) {
        Invoke-ProjectedKDepthRun -Data "custom" -RootPath $dataRoot -DataPath "weather.csv" -DatasetTag "Weather" `
            -SeqLen 336 -LabelLen 96 -PredLen $h -EncIn 21 -BatchSize 8 -Seed $seed -K $k
    }
}

# Parse results immediately after training
Write-Host "Parsing PFB-Projected K-depth results..."
$py310 = "C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe"
if (Test-Path $py310) {
    & $py310 .\analyze_pfb_kdepth_ablation.py
} else {
    & $python .\analyze_pfb_kdepth_ablation.py
}

Write-Host "Done. Results saved to results_analysis/pfb_projected_kdepth_ablation.csv"
