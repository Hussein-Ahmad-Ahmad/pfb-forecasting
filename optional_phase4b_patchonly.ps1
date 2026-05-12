# Optional Phase 4B: PatchOnly Ablation Study
# Activates conda environment and runs experiments

# Activate conda
$env:Path = "C:\Users\lab2\anaconda3\Scripts;" + $env:Path
conda activate ts_lib

Set-Location "D:\Hussein-experiments\dynamic-ensemble-paper\Ts-library\Time-Series-Library"

Write-Host "============================================================"
Write-Host "OPTIONAL PHASE 4B: PatchOnly Ablation"
Write-Host "Testing Patch-Only baseline (no BERT component)"
Write-Host "============================================================"

# Standard datasets - H=192
$datasets = @("ETTm1", "ETTm2", "ETTh1", "ETTh2", "Exchange", "Weather")

foreach ($d in $datasets) {
    Write-Host ""
    Write-Host "=== PatchOnly on $d H=192 ==="
    
    $dataPath = "$d.csv"
    if ($d -eq "Exchange" -or $d -eq "Weather") {
        $dataType = "custom"
    } else {
        $dataType = $d
    }
    
    python -u run.py `
        --task_name long_term_forecast `
        --is_training 1 `
        --root_path ./data/ `
        --data_path $dataPath `
        --model_id "PatchOnly_H192_$d" `
        --model PatchFusionBERT_PatchOnly `
        --data $dataType `
        --features M `
        --seq_len 336 `
        --label_len 96 `
        --pred_len 192 `
        --e_layers 3 `
        --d_model 128 `
        --d_ff 512 `
        --n_heads 8 `
        --dropout 0.1 `
        --batch_size 32 `
        --learning_rate 0.0001 `
        --train_epochs 100 `
        --patience 10 `
        --des Ablation_PatchOnly
}

# Illness dataset - H=24, 48, 60
$horizons = @(24, 48, 60)

foreach ($h in $horizons) {
    Write-Host ""
    Write-Host "=== PatchOnly on Illness H=$h ==="
    
    python -u run.py `
        --task_name long_term_forecast `
        --is_training 1 `
        --root_path ./data/ `
        --data_path national_illness.csv `
        --model_id "PatchOnly_H${h}_Illness" `
        --model PatchFusionBERT_PatchOnly `
        --data custom `
        --features M `
        --seq_len 104 `
        --label_len 18 `
        --pred_len $h `
        --e_layers 3 `
        --d_model 128 `
        --d_ff 512 `
        --n_heads 8 `
        --dropout 0.1 `
        --batch_size 32 `
        --learning_rate 0.0001 `
        --train_epochs 100 `
        --patience 10 `
        --des Ablation_PatchOnly
}

Write-Host ""
Write-Host "============================================================"
Write-Host "PHASE 4B COMPLETE!"
Write-Host "PatchOnly experiments finished for 9 configurations."
Write-Host "============================================================"
