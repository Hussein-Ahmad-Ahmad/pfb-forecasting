@echo off
cd /d "%~dp0.."
REM ============================================================================
REM MULTI-SEED VALIDATION: PatchFusionBERT_v2 (27 EXPERIMENTS)
REM 3 seeds (2021, 2022, 2023) Ã— 9 configurations
REM 6 standard datasets @ H=192 + Illness @ 3 horizons
REM ============================================================================

echo ============================================================================
echo MULTI-SEED VALIDATION: PatchFusionBERT_v2
echo ============================================================================
echo.
echo Configuration: 27 experiments
echo   - 6 standard datasets (ETTm1, ETTm2, ETTh1, ETTh2, Exchange, Weather) x H=192 x 3 seeds = 18
echo   - Illness dataset x H=24/48/60 x 3 seeds = 9
echo Expected runtime: ~6-12 hours
echo.

REM ============================================================================
REM ETTm1 - H=192 (3 seeds)
REM ============================================================================

echo [1/27] PFB_v2 - ETTm1 - H=192 - Seed 2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTm1.csv --model_id ETTm1_336_192_MS2021 --model PatchFusionBERT_v2 --data ETTm1 --features M --seq_len 336 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2021
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [2/27] PFB_v2 - ETTm1 - H=192 - Seed 2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTm1.csv --model_id ETTm1_336_192_MS2022 --model PatchFusionBERT_v2 --data ETTm1 --features M --seq_len 336 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2022
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [3/27] PFB_v2 - ETTm1 - H=192 - Seed 2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTm1.csv --model_id ETTm1_336_192_MS2023 --model PatchFusionBERT_v2 --data ETTm1 --features M --seq_len 336 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2023
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

REM ============================================================================
REM ETTm2 - H=192 (3 seeds)
REM ============================================================================

echo [4/27] PFB_v2 - ETTm2 - H=192 - Seed 2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTm2.csv --model_id ETTm2_336_192_MS2021 --model PatchFusionBERT_v2 --data ETTm2 --features M --seq_len 336 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2021
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [5/27] PFB_v2 - ETTm2 - H=192 - Seed 2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTm2.csv --model_id ETTm2_336_192_MS2022 --model PatchFusionBERT_v2 --data ETTm2 --features M --seq_len 336 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2022
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [6/27] PFB_v2 - ETTm2 - H=192 - Seed 2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTm2.csv --model_id ETTm2_336_192_MS2023 --model PatchFusionBERT_v2 --data ETTm2 --features M --seq_len 336 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2023
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

REM ============================================================================
REM ETTh1 - H=192 (3 seeds)
REM ============================================================================

echo [7/27] PFB_v2 - ETTh1 - H=192 - Seed 2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTh1.csv --model_id ETTh1_336_192_MS2021 --model PatchFusionBERT_v2 --data ETTh1 --features M --seq_len 336 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2021
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [8/27] PFB_v2 - ETTh1 - H=192 - Seed 2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTh1.csv --model_id ETTh1_336_192_MS2022 --model PatchFusionBERT_v2 --data ETTh1 --features M --seq_len 336 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2022
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [9/27] PFB_v2 - ETTh1 - H=192 - Seed 2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTh1.csv --model_id ETTh1_336_192_MS2023 --model PatchFusionBERT_v2 --data ETTh1 --features M --seq_len 336 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2023
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

REM ============================================================================
REM ETTh2 - H=192 (3 seeds)
REM ============================================================================

echo [10/27] PFB_v2 - ETTh2 - H=192 - Seed 2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTh2.csv --model_id ETTh2_336_192_MS2021 --model PatchFusionBERT_v2 --data ETTh2 --features M --seq_len 336 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2021
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [11/27] PFB_v2 - ETTh2 - H=192 - Seed 2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTh2.csv --model_id ETTh2_336_192_MS2022 --model PatchFusionBERT_v2 --data ETTh2 --features M --seq_len 336 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2022
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [12/27] PFB_v2 - ETTh2 - H=192 - Seed 2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path ETTh2.csv --model_id ETTh2_336_192_MS2023 --model PatchFusionBERT_v2 --data ETTh2 --features M --seq_len 336 --label_len 48 --pred_len 192 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2023
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

REM ============================================================================
REM Exchange - H=192 (3 seeds)
REM ============================================================================

echo [13/27] PFB_v2 - Exchange - H=192 - Seed 2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path exchange_rate.csv --model_id Exchange_336_192_MS2021 --model PatchFusionBERT_v2 --data custom --features M --seq_len 336 --label_len 48 --pred_len 192 --enc_in 8 --dec_in 8 --c_out 8 --d_model 128 --d_ff 512 --e_layers 3 --d_layers 1 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2021
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [14/27] PFB_v2 - Exchange - H=192 - Seed 2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path exchange_rate.csv --model_id Exchange_336_192_MS2022 --model PatchFusionBERT_v2 --data custom --features M --seq_len 336 --label_len 48 --pred_len 192 --enc_in 8 --dec_in 8 --c_out 8 --d_model 128 --d_ff 512 --e_layers 3 --d_layers 1 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2022
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [15/27] PFB_v2 - Exchange - H=192 - Seed 2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path exchange_rate.csv --model_id Exchange_336_192_MS2023 --model PatchFusionBERT_v2 --data custom --features M --seq_len 336 --label_len 48 --pred_len 192 --enc_in 8 --dec_in 8 --c_out 8 --d_model 128 --d_ff 512 --e_layers 3 --d_layers 1 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2023
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

REM ============================================================================
REM Weather - H=192 (3 seeds)
REM ============================================================================

echo [16/27] PFB_v2 - Weather - H=192 - Seed 2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path weather.csv --model_id Weather_336_192_MS2021 --model PatchFusionBERT_v2 --data custom --features M --seq_len 336 --label_len 48 --pred_len 192 --enc_in 21 --dec_in 21 --c_out 21 --d_model 128 --d_ff 512 --e_layers 3 --d_layers 1 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2021
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [17/27] PFB_v2 - Weather - H=192 - Seed 2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path weather.csv --model_id Weather_336_192_MS2022 --model PatchFusionBERT_v2 --data custom --features M --seq_len 336 --label_len 48 --pred_len 192 --enc_in 21 --dec_in 21 --c_out 21 --d_model 128 --d_ff 512 --e_layers 3 --d_layers 1 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2022
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [18/27] PFB_v2 - Weather - H=192 - Seed 2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path weather.csv --model_id Weather_336_192_MS2023 --model PatchFusionBERT_v2 --data custom --features M --seq_len 336 --label_len 48 --pred_len 192 --enc_in 21 --dec_in 21 --c_out 21 --d_model 128 --d_ff 512 --e_layers 3 --d_layers 1 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --seed 2023
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

REM ============================================================================
REM Illness - H=24 (3 seeds)
REM ============================================================================

echo [19/27] PFB_v2 - Illness - H=24 - Seed 2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_24_MS2021 --model PatchFusionBERT_v2 --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0 --seed 2021
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [20/27] PFB_v2 - Illness - H=24 - Seed 2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_24_MS2022 --model PatchFusionBERT_v2 --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0 --seed 2022
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [21/27] PFB_v2 - Illness - H=24 - Seed 2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_24_MS2023 --model PatchFusionBERT_v2 --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0 --seed 2023
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

REM ============================================================================
REM Illness - H=48 (3 seeds)
REM ============================================================================

echo [22/27] PFB_v2 - Illness - H=48 - Seed 2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_48_MS2021 --model PatchFusionBERT_v2 --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0 --seed 2021
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [23/27] PFB_v2 - Illness - H=48 - Seed 2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_48_MS2022 --model PatchFusionBERT_v2 --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0 --seed 2022
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [24/27] PFB_v2 - Illness - H=48 - Seed 2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_48_MS2023 --model PatchFusionBERT_v2 --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0 --seed 2023
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

REM ============================================================================
REM Illness - H=60 (3 seeds)
REM ============================================================================

echo [25/27] PFB_v2 - Illness - H=60 - Seed 2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_60_MS2021 --model PatchFusionBERT_v2 --data custom --features M --seq_len 104 --label_len 18 --pred_len 60 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0 --seed 2021
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [26/27] PFB_v2 - Illness - H=60 - Seed 2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_60_MS2022 --model PatchFusionBERT_v2 --data custom --features M --seq_len 104 --label_len 18 --pred_len 60 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0 --seed 2022
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

echo [27/27] PFB_v2 - Illness - H=60 - Seed 2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_60_MS2023 --model PatchFusionBERT_v2 --data custom --features M --seq_len 104 --label_len 18 --pred_len 60 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0 --seed 2023
if %errorlevel% neq 0 (echo FAILED && pause && exit /b 1)

REM ============================================================================
REM COMPLETION
REM ============================================================================

echo.
echo ============================================================================
echo âœ… ALL 27 MULTI-SEED PFB_v2 EXPERIMENTS COMPLETED!
echo ============================================================================
echo.
echo Multi-seed validation: COMPLETE
echo   - 6 standard datasets @ H=192 x 3 seeds = 18 experiments
echo   - Illness @ 3 horizons x 3 seeds = 9 experiments
echo   - Total: 27 experiments
echo.
echo ðŸŽ‰ PHASE 2 NOW 100%% COMPLETE!
echo ðŸ† ALL EXPERIMENTAL PHASES FINISHED!
echo.
echo Next: Update results_23-01.md with new multi-seed data
echo.
pause

