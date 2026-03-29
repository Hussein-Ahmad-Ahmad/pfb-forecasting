# Extra horizons for CapMatch controls: ETT/Weather at H=96 and H=336
# This complements capmatch_controls.ps1 (which already ran H=192 + Illness H=24/48/60).
#
# Models:
#   - PatchTST_base
#   - PatchTST_capacity (capacity-matched per horizon)
#   - PatchFusionBERT_v0
#   - PatchFusionBERT_v2

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

$seeds = @(2021, 2022, 2023)

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

# Base capacity
$base_d_model = 128
$base_e_layers = 3
$base_d_ff = 512

# Capacity-matched PatchTST configs (matched to PatchFusionBERT_v0)
# ETT/Weather regime: seq_len=336
# H=96:  PatchTST_capacity d_model=224, e_layers=3, d_ff=512  (0.87% rel diff)
# H=192: PatchTST_capacity d_model=192, e_layers=5, d_ff=512  (0.82% rel diff)  [already run elsewhere]
# H=336: PatchTST_capacity d_model=224, e_layers=3, d_ff=768  (0.06% rel diff)
$cap_336_96_d_model = 224
$cap_336_96_e_layers = 3
$cap_336_96_d_ff = 512

$cap_336_336_d_model = 224
$cap_336_336_e_layers = 3
$cap_336_336_d_ff = 768

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
# ETT (ETTh2, ETTm2): H=96 and H=336
# --------------------------
$ettRoot = $dataRoot
$ettSeq = 336
$ettLabel = 96
$ettEnc = 7
$ettBatch = 16
$ettHorizons = @(96, 336)

foreach ($seed in $seeds) {
    foreach ($h in $ettHorizons) {
        if ($h -eq 96) {
            $capDModel = $cap_336_96_d_model
            $capELayers = $cap_336_96_e_layers
            $capDff = $cap_336_96_d_ff
        } else {
            $capDModel = $cap_336_336_d_model
            $capELayers = $cap_336_336_e_layers
            $capDff = $cap_336_336_d_ff
        }

        # ETTh2
        Invoke-CapMatchRun -Model "PatchTST_base" -Data "ETTh2" -RootPath $ettRoot -DataPath "ETTh2.csv" `
            -ModelId "CapMatch_MS$seed`_ETTh2_$h" -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $h -EncIn $ettEnc -BatchSize $ettBatch `
            -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

        Invoke-CapMatchRun -Model "PatchTST_capacity" -Data "ETTh2" -RootPath $ettRoot -DataPath "ETTh2.csv" `
            -ModelId "CapMatch_MS$seed`_ETTh2_$h" -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $h -EncIn $ettEnc -BatchSize $ettBatch `
            -DModel $capDModel -ELayers $capELayers -Dff $capDff -Seed $seed

        Invoke-CapMatchRun -Model "PatchFusionBERT_v0" -Data "ETTh2" -RootPath $ettRoot -DataPath "ETTh2.csv" `
            -ModelId "CapMatch_MS$seed`_ETTh2_$h" -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $h -EncIn $ettEnc -BatchSize $ettBatch `
            -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

        Invoke-CapMatchRun -Model "PatchFusionBERT_v2" -Data "ETTh2" -RootPath $ettRoot -DataPath "ETTh2.csv" `
            -ModelId "CapMatch_MS$seed`_ETTh2_$h" -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $h -EncIn $ettEnc -BatchSize $ettBatch `
            -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

        # ETTm2
        Invoke-CapMatchRun -Model "PatchTST_base" -Data "ETTm2" -RootPath $ettRoot -DataPath "ETTm2.csv" `
            -ModelId "CapMatch_MS$seed`_ETTm2_$h" -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $h -EncIn $ettEnc -BatchSize $ettBatch `
            -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

        Invoke-CapMatchRun -Model "PatchTST_capacity" -Data "ETTm2" -RootPath $ettRoot -DataPath "ETTm2.csv" `
            -ModelId "CapMatch_MS$seed`_ETTm2_$h" -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $h -EncIn $ettEnc -BatchSize $ettBatch `
            -DModel $capDModel -ELayers $capELayers -Dff $capDff -Seed $seed

        Invoke-CapMatchRun -Model "PatchFusionBERT_v0" -Data "ETTm2" -RootPath $ettRoot -DataPath "ETTm2.csv" `
            -ModelId "CapMatch_MS$seed`_ETTm2_$h" -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $h -EncIn $ettEnc -BatchSize $ettBatch `
            -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

        Invoke-CapMatchRun -Model "PatchFusionBERT_v2" -Data "ETTm2" -RootPath $ettRoot -DataPath "ETTm2.csv" `
            -ModelId "CapMatch_MS$seed`_ETTm2_$h" -SeqLen $ettSeq -LabelLen $ettLabel -PredLen $h -EncIn $ettEnc -BatchSize $ettBatch `
            -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed
    }
}

# --------------------------
# Weather (custom): H=96 and H=336
# --------------------------
$weatherRoot = $dataRoot
$weatherSeq = 336
$weatherLabel = 96
$weatherEnc = 21
$weatherBatch = 8
$weatherHorizons = @(96, 336)

foreach ($seed in $seeds) {
    foreach ($h in $weatherHorizons) {
        if ($h -eq 96) {
            $capDModel = $cap_336_96_d_model
            $capELayers = $cap_336_96_e_layers
            $capDff = $cap_336_96_d_ff
        } else {
            $capDModel = $cap_336_336_d_model
            $capELayers = $cap_336_336_e_layers
            $capDff = $cap_336_336_d_ff
        }

        Invoke-CapMatchRun -Model "PatchTST_base" -Data "custom" -RootPath $weatherRoot -DataPath "weather.csv" `
            -ModelId "CapMatch_MS$seed`_Weather_$h" -SeqLen $weatherSeq -LabelLen $weatherLabel -PredLen $h -EncIn $weatherEnc -BatchSize $weatherBatch `
            -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

        Invoke-CapMatchRun -Model "PatchTST_capacity" -Data "custom" -RootPath $weatherRoot -DataPath "weather.csv" `
            -ModelId "CapMatch_MS$seed`_Weather_$h" -SeqLen $weatherSeq -LabelLen $weatherLabel -PredLen $h -EncIn $weatherEnc -BatchSize $weatherBatch `
            -DModel $capDModel -ELayers $capELayers -Dff $capDff -Seed $seed

        Invoke-CapMatchRun -Model "PatchFusionBERT_v0" -Data "custom" -RootPath $weatherRoot -DataPath "weather.csv" `
            -ModelId "CapMatch_MS$seed`_Weather_$h" -SeqLen $weatherSeq -LabelLen $weatherLabel -PredLen $h -EncIn $weatherEnc -BatchSize $weatherBatch `
            -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed

        Invoke-CapMatchRun -Model "PatchFusionBERT_v2" -Data "custom" -RootPath $weatherRoot -DataPath "weather.csv" `
            -ModelId "CapMatch_MS$seed`_Weather_$h" -SeqLen $weatherSeq -LabelLen $weatherLabel -PredLen $h -EncIn $weatherEnc -BatchSize $weatherBatch `
            -DModel $base_d_model -ELayers $base_e_layers -Dff $base_d_ff -Seed $seed
    }
}

Write-Host "Done. Next: python .\analyze_capmatch_controls.py"
