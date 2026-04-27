# Capacity-matched PatchTST controls vs PatchFusionBERT
# Runs on the most important regimes: ETTh2, ETTm2, Weather, Illness
# Models in table:
#   - PatchTST_base (baseline)
#   - PatchTST_capacity (capacity-matched; no fusion)
#   - PatchFusionBERT_v0
#   - PatchFusionBERT_v2
#
# Notes:
# - Uses the same training protocol across models; only PatchTST capacity changes.
# - Uses 3 seeds (2021/2022/2023) consistent with existing multi-seed runs.
# - Assumes you run this from Time-Series-Library/.

$ErrorActionPreference = "Stop"

$python = "python"  # must point to your GPU env python (torch+cuda)
$fallbackPython = "C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe"

$scriptRoot = $PSScriptRoot
$runPy = Join-Path $scriptRoot "run.py"
$dataRoot = Join-Path $scriptRoot "data"

# Ensure all outputs are written under Time-Series-Library/ even if the script is launched elsewhere.
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

# Common protocol
$task = "long_term_forecast"
$itr = 1
$train_epochs = 100
$patience = 10
$dropout = 0.1
$n_heads = 8
$factor = 3
$patch_len = 16
$stride = 8

# Base model capacity (PatchTST-base and PFB configs)
$base_d_model = 128
$base_e_layers = 3
$base_d_ff = 512

# Capacity-matched PatchTST configs
# IMPORTANT: matching depends on (seq_len, pred_len) because the head size changes.
# ETT/Weather regime used here: seq_len=336, pred_len=192
$cap_d_model_336_192 = 192
$cap_e_layers_336_192 = 5
$cap_d_ff_336_192 = 512

# Illness regime used here: seq_len=104, pred_len in {24,48,60}
$cap_d_model_104_24 = 128
$cap_e_layers_104_24 = 6
$cap_d_ff_104_24 = 512

$cap_d_model_104_48 = 160
$cap_e_layers_104_48 = 3
$cap_d_ff_104_48 = 1024

$cap_d_model_104_60 = 128
$cap_e_layers_104_60 = 4
$cap_d_ff_104_60 = 1024

function Get-LearningRate {
    param(
        [string]$Model
    )

    switch ($Model) {
        'DLinear' { return 0.01 }
        default { return 0.0001 }
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

    Write-Host "Running: Model=$Model  DatasetTag=$ModelId  pred_len=$PredLen  seed=$Seed"

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
# ETT (ETTh2, ETTm2): H=192
# --------------------------
$ettRoot = $dataRoot
$ettSeq = 336
$ettLabel = 96
$ettPred = 192
$ettEnc = 7
$ettBatch = 16

foreach ($seed in $seeds) {
    # ETTh2
    Invoke-CapMatchRun -Model "PatchTST_base" -Data "ETTh2" -RootPath $ettRoot -DataPath "ETTh2.csv" `
        -ModelId "CapMatch_MS$seed`_ETTh2_$ettPred" -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $ettPred -EncIn $ettEnc -BatchSize $ettBatch `
        -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

    Invoke-CapMatchRun -Model "PatchTST_capacity" -Data "ETTh2" -RootPath $ettRoot -DataPath "ETTh2.csv" `
        -ModelId "CapMatch_MS$seed`_ETTh2_$ettPred" -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $ettPred -EncIn $ettEnc -BatchSize $ettBatch `
        -DModel $cap_d_model_336_192 -ELayers $cap_e_layers_336_192 -Dff $cap_d_ff_336_192 -Seed $seed

    Invoke-CapMatchRun -Model "PatchFusionBERT_v0" -Data "ETTh2" -RootPath $ettRoot -DataPath "ETTh2.csv" `
        -ModelId "CapMatch_MS$seed`_ETTh2_$ettPred" -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $ettPred -EncIn $ettEnc -BatchSize $ettBatch `
        -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

    Invoke-CapMatchRun -Model "PatchFusionBERT_v2" -Data "ETTh2" -RootPath $ettRoot -DataPath "ETTh2.csv" `
        -ModelId "CapMatch_MS$seed`_ETTh2_$ettPred" -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $ettPred -EncIn $ettEnc -BatchSize $ettBatch `
        -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

    # ETTm2
    Invoke-CapMatchRun -Model "PatchTST_base" -Data "ETTm2" -RootPath $ettRoot -DataPath "ETTm2.csv" `
        -ModelId "CapMatch_MS$seed`_ETTm2_$ettPred" -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $ettPred -EncIn $ettEnc -BatchSize $ettBatch `
        -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

    Invoke-CapMatchRun -Model "PatchTST_capacity" -Data "ETTm2" -RootPath $ettRoot -DataPath "ETTm2.csv" `
        -ModelId "CapMatch_MS$seed`_ETTm2_$ettPred" -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $ettPred -EncIn $ettEnc -BatchSize $ettBatch `
        -DModel $cap_d_model_336_192 -ELayers $cap_e_layers_336_192 -Dff $cap_d_ff_336_192 -Seed $seed

    Invoke-CapMatchRun -Model "PatchFusionBERT_v0" -Data "ETTm2" -RootPath $ettRoot -DataPath "ETTm2.csv" `
        -ModelId "CapMatch_MS$seed`_ETTm2_$ettPred" -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $ettPred -EncIn $ettEnc -BatchSize $ettBatch `
        -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

    Invoke-CapMatchRun -Model "PatchFusionBERT_v2" -Data "ETTm2" -RootPath $ettRoot -DataPath "ETTm2.csv" `
        -ModelId "CapMatch_MS$seed`_ETTm2_$ettPred" -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $ettPred -EncIn $ettEnc -BatchSize $ettBatch `
        -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed
}

# --------------------------
# Weather: H=192 (custom)
# --------------------------
$weatherRoot = $dataRoot
$weatherSeq = 336
$weatherLabel = 96
$weatherPred = 192
$weatherEnc = 21
$weatherBatch = 8

foreach ($seed in $seeds) {
    Invoke-CapMatchRun -Model "PatchTST_base" -Data "custom" -RootPath $weatherRoot -DataPath "weather.csv" `
        -ModelId "CapMatch_MS$seed`_Weather_$weatherPred" -SeqLen $weatherSeq -LabelLen $weatherLabel -PredLen $weatherPred -EncIn $weatherEnc -BatchSize $weatherBatch `
        -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

    Invoke-CapMatchRun -Model "PatchTST_capacity" -Data "custom" -RootPath $weatherRoot -DataPath "weather.csv" `
        -ModelId "CapMatch_MS$seed`_Weather_$weatherPred" -SeqLen $weatherSeq -LabelLen $weatherLabel -PredLen $weatherPred -EncIn $weatherEnc -BatchSize $weatherBatch `
        -DModel $cap_d_model_336_192 -ELayers $cap_e_layers_336_192 -Dff $cap_d_ff_336_192 -Seed $seed

    Invoke-CapMatchRun -Model "PatchFusionBERT_v0" -Data "custom" -RootPath $weatherRoot -DataPath "weather.csv" `
        -ModelId "CapMatch_MS$seed`_Weather_$weatherPred" -SeqLen $weatherSeq -LabelLen $weatherLabel -PredLen $weatherPred -EncIn $weatherEnc -BatchSize $weatherBatch `
        -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

    Invoke-CapMatchRun -Model "PatchFusionBERT_v2" -Data "custom" -RootPath $weatherRoot -DataPath "weather.csv" `
        -ModelId "CapMatch_MS$seed`_Weather_$weatherPred" -SeqLen $weatherSeq -LabelLen $weatherLabel -PredLen $weatherPred -EncIn $weatherEnc -BatchSize $weatherBatch `
        -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed
}

# --------------------------
# Illness: H=24/48/60 (custom)
# --------------------------
$illRoot = $dataRoot
$illSeq = 104
$illLabel = 18
$illEnc = 7
$illBatch = 16
$illHorizons = @(24, 48, 60)

foreach ($seed in $seeds) {
    foreach ($h in $illHorizons) {
        $illCapDModel = $cap_d_model_104_24
        $illCapELayers = $cap_e_layers_104_24
        $illCapDff = $cap_d_ff_104_24
        if ($h -eq 48) {
            $illCapDModel = $cap_d_model_104_48
            $illCapELayers = $cap_e_layers_104_48
            $illCapDff = $cap_d_ff_104_48
        } elseif ($h -eq 60) {
            $illCapDModel = $cap_d_model_104_60
            $illCapELayers = $cap_e_layers_104_60
            $illCapDff = $cap_d_ff_104_60
        }

        Invoke-CapMatchRun -Model "PatchTST_base" -Data "custom" -RootPath $illRoot -DataPath "national_illness.csv" `
            -ModelId "CapMatch_MS$seed`_Illness_$h" -SeqLen $illSeq -LabelLen $illLabel -PredLen $h -EncIn $illEnc -BatchSize $illBatch `
            -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

        Invoke-CapMatchRun -Model "PatchTST_capacity" -Data "custom" -RootPath $illRoot -DataPath "national_illness.csv" `
            -ModelId "CapMatch_MS$seed`_Illness_$h" -SeqLen $illSeq -LabelLen $illLabel -PredLen $h -EncIn $illEnc -BatchSize $illBatch `
            -DModel $illCapDModel -ELayers $illCapELayers -Dff $illCapDff -Seed $seed

        Invoke-CapMatchRun -Model "PatchFusionBERT_v0" -Data "custom" -RootPath $illRoot -DataPath "national_illness.csv" `
            -ModelId "CapMatch_MS$seed`_Illness_$h" -SeqLen $illSeq -LabelLen $illLabel -PredLen $h -EncIn $illEnc -BatchSize $illBatch `
            -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

        Invoke-CapMatchRun -Model "PatchFusionBERT_v2" -Data "custom" -RootPath $illRoot -DataPath "national_illness.csv" `
            -ModelId "CapMatch_MS$seed`_Illness_$h" -SeqLen $illSeq -LabelLen $illLabel -PredLen $h -EncIn $illEnc -BatchSize $illBatch `
            -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed
    }
}

Write-Host "Done. Next: python .\analyze_capmatch_controls.py"
