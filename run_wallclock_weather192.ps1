$ErrorActionPreference = "Stop"

$python = "python"
$fallbackPython = "C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe"

$scriptRoot = $PSScriptRoot
$runPy = Join-Path $scriptRoot "run.py"
$dataRoot = Join-Path $scriptRoot "data"
$logDir = Join-Path $scriptRoot "results_analysis\wallclock_logs"

Set-Location $scriptRoot

if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

Write-Host "Checking Python environment..."
& $python -c "import sys, torch; print('exe:', sys.executable); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Falling back to: $fallbackPython"
    $python = $fallbackPython
    & $python -c "import sys, torch; print('exe:', sys.executable); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    if ($LASTEXITCODE -ne 0) { throw "No usable torch environment found." }
}

$gpuName = & $python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
$gpuName | Set-Content -Path (Join-Path $logDir "gpu_name.txt")

$task = "long_term_forecast"
$seqLen = 336
$labelLen = 96
$predLen = 192
$encIn = 21
$batchSize = 8
$trainEpochs = 100
$patience = 10
$learningRate = 0.0001
$patchLen = 16
$stride = 8
$dropout = 0.1
$dModel = 128
$eLayers = 3
$dff = 512
$nHeads = 8
$factor = 3
$seeds = @(2021, 2022, 2023)

$models = @(
    @{ Name = "DLinear"; ModelIdPrefix = "WallClock" },
    @{ Name = "PatchTST_base"; ModelIdPrefix = "WallClock" },
    @{ Name = "PatchFusionBERT_v0"; ModelIdPrefix = "WallClock" },
    @{ Name = "PatchFusionBERT_v2"; ModelIdPrefix = "WallClock" }
)

foreach ($modelInfo in $models) {
    foreach ($seed in $seeds) {
        $model = $modelInfo.Name
        $modelId = "$($modelInfo.ModelIdPrefix)_MS$seed`_Weather_192_$model"
        $logFile = Join-Path $logDir "$model`_MS$seed`_Weather_192.log"

        Write-Host "============================================================"
        Write-Host "Running model=$model seed=$seed"
        Write-Host "Log: $logFile"

        "MODEL=$model" | Set-Content -Path $logFile
        "SEED=$seed" | Add-Content -Path $logFile
        "GPU=$gpuName" | Add-Content -Path $logFile
        "START=$(Get-Date -Format o)" | Add-Content -Path $logFile

        & $python -u $runPy `
            --task_name $task --is_training 1 `
            --model_id $modelId --model $model --data custom `
            --root_path $dataRoot --data_path weather.csv `
            --features M --seq_len $seqLen --label_len $labelLen --pred_len $predLen `
            --enc_in $encIn --dec_in $encIn --c_out $encIn `
            --patch_len $patchLen --stride $stride `
            --d_model $dModel --n_heads $nHeads --e_layers $eLayers --d_layers 1 --d_ff $dff --factor $factor `
            --dropout $dropout `
            --itr 1 --train_epochs $trainEpochs --patience $patience --batch_size $batchSize --learning_rate $learningRate `
            --num_workers 0 --seed $seed `
            --des wallclock 2>&1 | Tee-Object -FilePath $logFile -Append

        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $model seed $seed"
        }

        "END=$(Get-Date -Format o)" | Add-Content -Path $logFile
    }
}

Write-Host "Wall-clock experiment finished. Next: python .\analyze_wallclock_weather192.py"