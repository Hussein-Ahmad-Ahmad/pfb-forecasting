@echo off
cd /d "%~dp0.."
REM ============================================================================
REM COMPLETE ILLNESS DATASET - 19 MISSING EXPERIMENTS
REM seq_len=104 (special for Illness), epochs=100, patience=10, seed=2021
REM ============================================================================

echo ============================================================================
echo COMPLETING ILLNESS DATASET - 19 MISSING EXPERIMENTS
echo ============================================================================
echo.
echo Models: All 8 models (PatchTST, DLinear, PFB_v0, PFB_v2, TiDE, iTransformer, TimeXer, BERTOnly)
echo Horizons: H=24 (8 exp), H=48 (8 exp), H=60 (3 exp)
echo Expected runtime: ~6-10 hours total
echo.

REM ============================================================================
REM HORIZON 24 - ALL 8 MODELS (currently 0/8)
REM ============================================================================

echo.
echo [GROUP 1/3] Illness H=24 (8 experiments)
echo ============================================================================

echo [1/19] PatchTST - Illness - H=24
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_24 --model PatchTST --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: PatchTST Illness H=24 && pause && exit /b 1)

echo [2/19] DLinear - Illness - H=24
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_24 --model DLinear --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --enc_in 7 --dec_in 7 --c_out 7 --des Exp --itr 1 --learning_rate 0.01 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: DLinear Illness H=24 && pause && exit /b 1)

echo [3/19] PatchFusionBERT_v0 - Illness - H=24
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_24 --model PatchFusionBERT_v0 --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: PFB_v0 Illness H=24 && pause && exit /b 1)

echo [4/19] PatchFusionBERT_v2 - Illness - H=24
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_24 --model PatchFusionBERT_v2 --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: PFB_v2 Illness H=24 && pause && exit /b 1)

echo [5/19] TiDE - Illness - H=24
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_24 --model TiDE --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --enc_in 7 --dec_in 7 --c_out 7 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: TiDE Illness H=24 && pause && exit /b 1)

echo [6/19] iTransformer - Illness - H=24
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_24 --model iTransformer --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 512 --dropout 0.1 --batch_size 16 --learning_rate 0.0001 --train_epochs 100 --patience 10 --des Exp --itr 1 --use_gpu 1 --gpu 0 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: iTransformer Illness H=24 && pause && exit /b 1)

echo [7/19] TimeXer - Illness - H=24
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_24 --model TimeXer --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: TimeXer Illness H=24 && pause && exit /b 1)

echo [8/19] PatchFusionBERT_BERTOnly - Illness - H=24
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_24 --model PatchFusionBERT_BERTOnly --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: BERTOnly Illness H=24 && pause && exit /b 1)

REM ============================================================================
REM HORIZON 48 - ALL 8 MODELS (currently 0/8)
REM ============================================================================

echo.
echo [GROUP 2/3] Illness H=48 (8 experiments)
echo ============================================================================

echo [9/19] PatchTST - Illness - H=48
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_48 --model PatchTST --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: PatchTST Illness H=48 && pause && exit /b 1)

echo [10/19] DLinear - Illness - H=48
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_48 --model DLinear --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --enc_in 7 --dec_in 7 --c_out 7 --des Exp --itr 1 --learning_rate 0.01 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: DLinear Illness H=48 && pause && exit /b 1)

echo [11/19] PatchFusionBERT_v0 - Illness - H=48
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_48 --model PatchFusionBERT_v0 --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: PFB_v0 Illness H=48 && pause && exit /b 1)

echo [12/19] PatchFusionBERT_v2 - Illness - H=48
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_48 --model PatchFusionBERT_v2 --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: PFB_v2 Illness H=48 && pause && exit /b 1)

echo [13/19] TiDE - Illness - H=48
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_48 --model TiDE --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --enc_in 7 --dec_in 7 --c_out 7 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: TiDE Illness H=48 && pause && exit /b 1)

echo [14/19] iTransformer - Illness - H=48
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_48 --model iTransformer --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 512 --dropout 0.1 --batch_size 16 --learning_rate 0.0001 --train_epochs 100 --patience 10 --des Exp --itr 1 --use_gpu 1 --gpu 0 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: iTransformer Illness H=48 && pause && exit /b 1)

echo [15/19] TimeXer - Illness - H=48
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_48 --model TimeXer --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: TimeXer Illness H=48 && pause && exit /b 1)

echo [16/19] PatchFusionBERT_BERTOnly - Illness - H=48
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_48 --model PatchFusionBERT_BERTOnly --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: BERTOnly Illness H=48 && pause && exit /b 1)

REM ============================================================================
REM HORIZON 60 - REMAINING 3 MODELS (currently 5/8 - need PatchTST, DLinear, PFB_v0)
REM ============================================================================

echo.
echo [GROUP 3/3] Illness H=60 (3 remaining experiments)
echo ============================================================================

echo [17/19] PatchTST - Illness - H=60
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_60 --model PatchTST --data custom --features M --seq_len 104 --label_len 18 --pred_len 60 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: PatchTST Illness H=60 && pause && exit /b 1)

echo [18/19] DLinear - Illness - H=60
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_60 --model DLinear --data custom --features M --seq_len 104 --label_len 18 --pred_len 60 --enc_in 7 --dec_in 7 --c_out 7 --des Exp --itr 1 --learning_rate 0.01 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: DLinear Illness H=60 && pause && exit /b 1)

echo [19/19] PatchFusionBERT_v0 - Illness - H=60
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id Illness_104_60 --model PatchFusionBERT_v0 --data custom --features M --seq_len 104 --label_len 18 --pred_len 60 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 512 --des Exp --itr 1 --learning_rate 0.0001 --train_epochs 100 --patience 10 --batch_size 16 --num_workers 0
if %errorlevel% neq 0 (echo FAILED: PFB_v0 Illness H=60 && pause && exit /b 1)

REM ============================================================================
REM COMPLETION
REM ============================================================================

echo.
echo ============================================================================
echo âœ… ALL 19 ILLNESS EXPERIMENTS COMPLETED!
echo ============================================================================
echo.
echo Coverage: 24/24 Illness experiments (100%%)
echo Total dataset coverage: 168/168 (100%%)
echo.
echo Next steps:
echo   1. Check results_23-01.md (auto-updated by monitoring script)
echo   2. Proceed with multi-seed validation (PFB_v2 priority!)
echo.
pause

