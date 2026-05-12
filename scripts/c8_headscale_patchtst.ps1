# C.C.8 -- Head-scaled PatchTST capacity control
#
# Tests attention-head expansion as a third scaling axis (beyond width and depth).
# PatchTST d_model=128, e_layers=3 fixed. n_heads varies: 8 (base), 16, 32.
# Note: in standard MHA, changing n_heads does not change parameter count
# (head_dim = d_model / n_heads shrinks proportionally). This isolates attention
# granularity from capacity, testing whether more heads improve PatchTST.
#
# Datasets: ETTh2 H=192, Weather H=192 (main comparison datasets).
# One seed (2021) for this minor ablation.
# Run from: Time-Series-Library/

$ErrorActionPreference = "Stop"
$python = "C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe"
$scriptRoot = $PSScriptRoot
Set-Location $scriptRoot

Write-Host "Head-scaled PatchTST -- C.C.8"
& $python -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'Python/torch not available' }

$task       = "long_term_forecast"
$seq_len    = 336
$label_len  = 96
$d_model    = 128
$e_layers   = 3
$d_ff       = 512
$dropout    = 0.1
$factor     = 3
$patch_len  = 16
$stride     = 8
$batch_size = 128
$lr         = 0.0001
$epochs     = 100
$patience   = 10
$seed       = 2021

$datasets = @(
    @{ name="ETTh2";   data="ETTh2";  data_path="ETTh2.csv";   enc_in=7;  pred_len=192 },
    @{ name="Weather"; data="custom"; data_path="weather.csv";  enc_in=21; pred_len=192 }
)

$n_heads_list = @(8, 16, 32)

foreach ($ds in $datasets) {
    foreach ($nh in $n_heads_list) {
        $model_id = "HeadScale_MS${seed}_$($ds.name)_192_NH${nh}_PatchTST"
        Write-Host ""
        Write-Host "=== $model_id ==="

        & $python run.py `
            --task_name        $task `
            --is_training      1 `
            --root_path        ./data/ `
            --data_path        $ds.data_path `
            --model_id         $model_id `
            --model            PatchTST `
            --data             $ds.data `
            --features         M `
            --seq_len          $seq_len `
            --label_len        $label_len `
            --pred_len         $ds.pred_len `
            --e_layers         $e_layers `
            --d_layers         1 `
            --factor           $factor `
            --enc_in           $ds.enc_in `
            --dec_in           $ds.enc_in `
            --c_out            $ds.enc_in `
            --d_model          $d_model `
            --d_ff             $d_ff `
            --n_heads          $nh `
            --patch_len        $patch_len `
            --stride           $stride `
            --dropout          $dropout `
            --batch_size       $batch_size `
            --learning_rate    $lr `
            --train_epochs     $epochs `
            --patience         $patience `
            --itr              1 `
            --use_gpu          True `
            --gpu              0 `
            --seed             $seed `
            --des              headscale_ablation

        if ($LASTEXITCODE -ne 0) { Write-Warning ("Run failed: " + $model_id) }
    }
}

Write-Host ""
Write-Host "=== C.C.8 Head-scaled PatchTST complete ==="
Write-Host "Result folders with HeadScale tag:"
Get-ChildItem results -Directory | Where-Object { $_.Name -match 'HeadScale' } | Select-Object -ExpandProperty Name
