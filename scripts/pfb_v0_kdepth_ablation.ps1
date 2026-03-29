# Experiment 2: K-depth refinement ablation for PatchFusionBERT_v0
# Runs:
#   - Datasets: ETTh2, Weather
#   - Horizons: 96, 192, 336
#   - K: 1, 2, 3
#   - Seed: single (default 2021)
# Protocol matches CapMatch scripts (epochs/lr/dropout/patching/etc).

$ErrorActionPreference = "Stop"

$python = "python"  # must point to your GPU env python (torch+cuda)
$fallbackPython = "C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe"

$scriptRoot = $PSScriptRoot
$packageRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
$runPy = Join-Path $packageRoot "run.py"
$dataRoot = Join-Path $packageRoot "data"

# Ensure all outputs go under the package root.
Set-Location $packageRoot

Write-Host "Using python (initial): $python"
& $python -c "import sys, torch; print('exe:', sys.executable); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python check failed; falling back to: $fallbackPython"
    $python = $fallbackPython
    & $python -c "import sys, torch; print('exe:', sys.executable); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw "Python check failed (is torch installed in this env?)" }
}

# Common protocol (keep aligned with capmatch_controls.ps1)
$task = "long_term_forecast"
$itr = 1
$train_epochs = 100
$patience = 10
$learning_rate = 0.0001
$dropout = 0.1
$n_heads = 8
$factor = 3
$patch_len = 16
$stride = 8

# Base model capacity (same as prior PFB runs)
$d_model = 128
$e_layers = 3
$d_ff = 512

$seed = 2021
$ks = @(1, 2, 3)
$horizons = @(96, 192, 336)

function Invoke-KDepthRun {
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

    $model = "PatchFusionBERT_v0"
    $modelId = "KDepth_MS$Seed`_${DatasetTag}_$PredLen`_K$K"

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
        --des kdepth_ablation

    if ($LASTEXITCODE -ne 0) { throw "Training failed for $modelId" }
}

# ETTh2
$ettRoot = $dataRoot
$ettSeq = 336
$ettLabel = 96
$ettEnc = 7
$ettBatch = 16
foreach ($h in $horizons) {
    foreach ($k in $ks) {
        Invoke-KDepthRun -Data "ETTh2" -RootPath $ettRoot -DataPath "ETTh2.csv" -DatasetTag "ETTh2" `
            -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $h -EncIn $ettEnc -BatchSize $ettBatch -Seed $seed -K $k
    }
}

# Weather
$weatherRoot = $dataRoot
$weatherSeq = 336
$weatherLabel = 96
$weatherEnc = 21
$weatherBatch = 8
foreach ($h in $horizons) {
    foreach ($k in $ks) {
        Invoke-KDepthRun -Data "custom" -RootPath $weatherRoot -DataPath "weather.csv" -DatasetTag "Weather" `
            -SeqLen $weatherSeq -LabelLen $weatherLabel -PredLen $h -EncIn $weatherEnc -BatchSize $weatherBatch -Seed $seed -K $k
    }
}

Write-Host "Done. Next: parse results from result_long_term_forecast.txt"
