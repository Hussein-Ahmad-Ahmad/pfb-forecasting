# Capacity-matched PatchTST controls vs PatchFusionBERT on Exchange
# Exchange: seq_len=336, label_len=96, pred_len=192, enc_in=8
#
# Capacity match (python capacity_match_patchtst.py --dataset Exchange
#                 --seq_len 336 --pred_len 192 --enc_in 8):
#   PatchTST-base (d_model=128,e_layers=3,d_ff=512): 1,629,504 params
#   PatchFusionBERT_v0 target:                        3,256,768 params
#   Best match: d_model=192, e_layers=5, d_ff=512     3,283,456 params (0.82% off)
#
# Models: PatchTST_base, PatchTST_capacity, PatchFusionBERT_v0, PatchFusionBERT_v2
# Seeds: 2021, 2022, 2023
# Run from Time-Series-Library/

$ErrorActionPreference = "Stop"

$python = "python"
$fallbackPython = "C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe"

$scriptRoot = $PSScriptRoot
$runPy = Join-Path $scriptRoot "run.py"
$dataRoot = Join-Path $scriptRoot "data"

Set-Location $scriptRoot

Write-Host "Using python (initial): $python"
& $python -c "import sys, torch; print('exe:', sys.executable); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python check failed; falling back to: $fallbackPython"
    $python = $fallbackPython
    & $python -c "import sys, torch; print('exe:', sys.executable); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw "Python check failed (is torch installed in this env?)" }
}

$seeds = @(2021, 2022, 2023)

# Common protocol (matches capmatch_controls.ps1)
$task = "long_term_forecast"
$itr = 1
$train_epochs = 100
$patience = 10
$dropout = 0.1
$n_heads = 8
$factor = 3
$patch_len = 16
$stride = 8

# Base capacity (same for all non-capacity models)
$base_d_model = 128
$base_e_layers = 3
$base_d_ff = 512

# Capacity-matched PatchTST for Exchange (seq_len=336, pred_len=192, enc_in=8)
$cap_d_model = 192
$cap_e_layers = 5
$cap_d_ff = 512

# Exchange dataset settings
$exSeq = 336
$exLabel = 96
$exPred = 192
$exEnc = 8
$exBatch = 16

function Get-LearningRate {
    param([string]$Model)
    switch ($Model) {
        'DLinear' { return 0.01 }
        default   { return 0.0001 }
    }
}

function Invoke-CapMatchRun {
    param(
        [string]$Model,
        [string]$Data,
        [string]$RootPath,
        [string]$DataPath,
        [string]$ModelId,
        [int]$SeqLen,
        [int]$LabelLen,
        [int]$PredLen,
        [int]$EncIn,
        [int]$BatchSize,
        [int]$DModel,
        [int]$ELayers,
        [int]$Dff,
        [int]$Seed
    )

    $learning_rate = Get-LearningRate -Model $Model

    Write-Host "Running: Model=$Model  ModelId=$ModelId  pred_len=$PredLen  seed=$Seed"

    & $python -u $runPy `
        --task_name $task --is_training 1 `
        --model_id $ModelId --model $Model --data $Data `
        --root_path $RootPath --data_path $DataPath `
        --features M --seq_len $SeqLen --label_len $LabelLen --pred_len $PredLen `
        --enc_in $EncIn --dec_in $EncIn --c_out $EncIn `
        --patch_len $patch_len --stride $stride `
        --d_model $DModel --n_heads $n_heads --e_layers $ELayers --d_layers 1 --d_ff $Dff --factor $factor `
        --dropout $dropout `
        --itr $itr --train_epochs $train_epochs --patience $patience --batch_size $BatchSize --learning_rate $learning_rate `
        --num_workers 0 --seed $Seed `
        --des capmatch

    if ($LASTEXITCODE -ne 0) { throw "Training failed for $ModelId" }
}

# --------------------------
# Exchange: H=192
# --------------------------
foreach ($seed in $seeds) {
    Invoke-CapMatchRun -Model "PatchTST_base" -Data "custom" -RootPath $dataRoot -DataPath "exchange_rate.csv" `
        -ModelId "CapMatch_MS$seed`_Exchange_$exPred" -SeqLen $exSeq -LabelLen $exLabel -PredLen $exPred -EncIn $exEnc -BatchSize $exBatch `
        -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

    Invoke-CapMatchRun -Model "PatchTST_capacity" -Data "custom" -RootPath $dataRoot -DataPath "exchange_rate.csv" `
        -ModelId "CapMatch_MS$seed`_Exchange_$exPred" -SeqLen $exSeq -LabelLen $exLabel -PredLen $exPred -EncIn $exEnc -BatchSize $exBatch `
        -DModel $cap_d_model -ELayers $cap_e_layers -Dff $cap_d_ff -Seed $seed

    Invoke-CapMatchRun -Model "PatchFusionBERT_v0" -Data "custom" -RootPath $dataRoot -DataPath "exchange_rate.csv" `
        -ModelId "CapMatch_MS$seed`_Exchange_$exPred" -SeqLen $exSeq -LabelLen $exLabel -PredLen $exPred -EncIn $exEnc -BatchSize $exBatch `
        -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

    Invoke-CapMatchRun -Model "PatchFusionBERT_v2" -Data "custom" -RootPath $dataRoot -DataPath "exchange_rate.csv" `
        -ModelId "CapMatch_MS$seed`_Exchange_$exPred" -SeqLen $exSeq -LabelLen $exLabel -PredLen $exPred -EncIn $exEnc -BatchSize $exBatch `
        -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed
}

Write-Host "Done. Next: python .\analyze_capmatch_controls.py"
