@echo off
cd /d "%~dp0.."
REM ==========================================================================
REM PHASE 3: MULTI-SEED VALIDATION - H192, Seed 2023
REM Purpose: Statistical validation with meanÂ±std for key baselines
REM Models: PFB_v0, PatchTST, DLinear (3 models)
REM Datasets: 6 datasets (ETTm1, ETTm2, ETTh1, ETTh2, Exchange, Weather)
REM Note: Illness EXCLUDED - only supports Hâ‰¤60, separate batch for proper horizons
REM Horizon: 192 (most representative mid-range)
REM Seed: 2023 (third of three: 2021, 2022, 2023)
REM Total: 18 experiments = 3 models Ã— 6 datasets
REM Expected time: ~30-40 mins per experiment (optimized)
REM ==========================================================================

echo ============================================================================
echo MULTI-SEED VALIDATION - H192, Seed 2023
echo Started: %date% %time%
echo ============================================================================
echo.

REM ========== ETTm1 Dataset (3 models) ==========
echo [1/21] PFB_v0 + ETTm1 + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTm1.csv --model_id PFB_v0_MS2023_ETTm1_192 --model PatchFusionBERT_v0 --data ETTm1 --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des multiseed_h192 --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [2/21] PatchTST + ETTm1 + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTm1.csv --model_id PatchTST_MS2023_ETTm1_192 --model PatchTST --data ETTm1 --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des multiseed_h192 --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [3/21] DLinear + ETTm1 + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTm1.csv --model_id DLinear_MS2023_ETTm1_192 --model DLinear --data ETTm1 --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des multiseed_h192 --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

REM ========== ETTm2 Dataset (3 models) ==========
echo [4/21] PFB_v0 + ETTm2 + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTm2.csv --model_id PFB_v0_MS2023_ETTm2_192 --model PatchFusionBERT_v0 --data ETTm2 --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des multiseed_h192 --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [5/21] PatchTST + ETTm2 + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTm2.csv --model_id PatchTST_MS2023_ETTm2_192 --model PatchTST --data ETTm2 --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des multiseed_h192 --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [6/21] DLinear + ETTm2 + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTm2.csv --model_id DLinear_MS2023_ETTm2_192 --model DLinear --data ETTm2 --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des multiseed_h192 --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

REM ========== ETTh1 Dataset (3 models) ==========
echo [7/21] PFB_v0 + ETTh1 + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTh1.csv --model_id PFB_v0_MS2023_ETTh1_192 --model PatchFusionBERT_v0 --data ETTh1 --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des multiseed_h192 --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [8/21] PatchTST + ETTh1 + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTh1.csv --model_id PatchTST_MS2023_ETTh1_192 --model PatchTST --data ETTh1 --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des multiseed_h192 --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [9/21] DLinear + ETTh1 + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTh1.csv --model_id DLinear_MS2023_ETTh1_192 --model DLinear --data ETTh1 --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des multiseed_h192 --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

REM ========== ETTh2 Dataset (3 models) ==========
echo [10/21] PFB_v0 + ETTh2 + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTh2.csv --model_id PFB_v0_MS2023_ETTh2_192 --model PatchFusionBERT_v0 --data ETTh2 --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des multiseed_h192 --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [11/21] PatchTST + ETTh2 + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTh2.csv --model_id PatchTST_MS2023_ETTh2_192 --model PatchTST --data ETTh2 --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des multiseed_h192 --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [12/21] DLinear + ETTh2 + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTh2.csv --model_id DLinear_MS2023_ETTh2_192 --model DLinear --data ETTh2 --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des multiseed_h192 --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

REM ========== Exchange Dataset (3 models) ==========
echo [13/21] PFB_v0 + Exchange + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path exchange_rate.csv --model_id PFB_v0_MS2023_Exchange_192 --model PatchFusionBERT_v0 --data custom --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --des multiseed_h192 --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [14/21] PatchTST + Exchange + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path exchange_rate.csv --model_id PatchTST_MS2023_Exchange_192 --model PatchTST --data custom --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --des multiseed_h192 --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [15/21] DLinear + Exchange + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path exchange_rate.csv --model_id DLinear_MS2023_Exchange_192 --model DLinear --data custom --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 8 --dec_in 8 --c_out 8 --des multiseed_h192 --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

REM ========== Weather Dataset (3 models) ==========
echo [16/21] PFB_v0 + Weather + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path weather.csv --model_id PFB_v0_MS2023_Weather_192 --model PatchFusionBERT_v0 --data custom --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --des multiseed_h192 --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 8 --learning_rate 0.0001 --seed 2023

echo [17/21] PatchTST + Weather + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path weather.csv --model_id PatchTST_MS2023_Weather_192 --model PatchTST --data custom --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --des multiseed_h192 --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 8 --learning_rate 0.0001 --seed 2023

echo [18/18] DLinear + Weather + H192 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path weather.csv --model_id DLinear_MS2023_Weather_192 --model DLinear --data custom --features M --seq_len 336 --label_len 96 --pred_len 192 --e_layers 2 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --des multiseed_h192 --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 8 --learning_rate 0.0001 --seed 2023

echo.
echo ============================================================================
echo MULTI-SEED VALIDATION SEED 2023 COMPLETED
echo Ended: %date% %time%
echo ============================================================================
echo.
echo VERIFICATION CHECKLIST:
echo 1. Check result_long_term_forecast.txt - should have 21 new entries
echo 2. Verify no missing experiments (3 models Ã— 7 datasets = 21)
echo 3. Check for any crashes or errors
echo 4. Review MSE/MAE ranges - should match Phase 0 single-seed baselines
echo.
echo If all checks pass: Run multiseed_H192_complete.txt
echo ============================================================================
18 new entries
echo 2. Verify no missing experiments (3 models Ã— 6 datasets = 18
