@echo off
cd /d "%~dp0.."
REM ==========================================================================
REM ILLNESS DATASET - PROPER HORIZONS Multi-Seed
REM Purpose: Illness dataset with horizons it actually supports
REM Models: PFB_v0, PatchTST, DLinear (3 models)
REM Dataset: Illness (national_illness.csv)
REM Horizons: 24, 36, 48, 60 (realistic for weekly ILI data)
REM Seeds: 2021, 2022, 2023
REM Total: 3 models Ã— 4 horizons Ã— 3 seeds = 36 experiments
REM Expected time: ~18-20 hours (36 Ã— ~30 mins)
REM ==========================================================================

echo ============================================================================
echo ILLNESS HORIZONS MULTI-SEED VALIDATION
echo Started: %date% %time%
echo ============================================================================
echo.

REM ========== Horizon 24 (3 models Ã— 3 seeds = 9 exp) ==========
echo [1/36] PFB_v0 + Illness + H24 + seed=2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PFB_v0_MS2021_Illness_24 --model PatchFusionBERT_v0 --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2021

echo [2/36] PatchTST + Illness + H24 + seed=2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PatchTST_MS2021_Illness_24 --model PatchTST --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2021

echo [3/36] DLinear + Illness + H24 + seed=2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id DLinear_MS2021_Illness_24 --model DLinear --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2021

echo [4/36] PFB_v0 + Illness + H24 + seed=2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PFB_v0_MS2022_Illness_24 --model PatchFusionBERT_v0 --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2022

echo [5/36] PatchTST + Illness + H24 + seed=2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PatchTST_MS2022_Illness_24 --model PatchTST --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2022

echo [6/36] DLinear + Illness + H24 + seed=2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id DLinear_MS2022_Illness_24 --model DLinear --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2022

echo [7/36] PFB_v0 + Illness + H24 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PFB_v0_MS2023_Illness_24 --model PatchFusionBERT_v0 --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [8/36] PatchTST + Illness + H24 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PatchTST_MS2023_Illness_24 --model PatchTST --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [9/36] DLinear + Illness + H24 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id DLinear_MS2023_Illness_24 --model DLinear --data custom --features M --seq_len 104 --label_len 18 --pred_len 24 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

REM ========== Horizon 36 (3 models Ã— 3 seeds = 9 exp) ==========
echo [10/36] PFB_v0 + Illness + H36 + seed=2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PFB_v0_MS2021_Illness_36 --model PatchFusionBERT_v0 --data custom --features M --seq_len 104 --label_len 18 --pred_len 36 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2021

echo [11/36] PatchTST + Illness + H36 + seed=2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PatchTST_MS2021_Illness_36 --model PatchTST --data custom --features M --seq_len 104 --label_len 18 --pred_len 36 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2021

echo [12/36] DLinear + Illness + H36 + seed=2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id DLinear_MS2021_Illness_36 --model DLinear --data custom --features M --seq_len 104 --label_len 18 --pred_len 36 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2021

echo [13/36] PFB_v0 + Illness + H36 + seed=2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PFB_v0_MS2022_Illness_36 --model PatchFusionBERT_v0 --data custom --features M --seq_len 104 --label_len 18 --pred_len 36 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2022

echo [14/36] PatchTST + Illness + H36 + seed=2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PatchTST_MS2022_Illness_36 --model PatchTST --data custom --features M --seq_len 104 --label_len 18 --pred_len 36 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2022

echo [15/36] DLinear + Illness + H36 + seed=2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id DLinear_MS2022_Illness_36 --model DLinear --data custom --features M --seq_len 104 --label_len 18 --pred_len 36 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2022

echo [16/36] PFB_v0 + Illness + H36 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PFB_v0_MS2023_Illness_36 --model PatchFusionBERT_v0 --data custom --features M --seq_len 104 --label_len 18 --pred_len 36 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [17/36] PatchTST + Illness + H36 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PatchTST_MS2023_Illness_36 --model PatchTST --data custom --features M --seq_len 104 --label_len 18 --pred_len 36 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [18/36] DLinear + Illness + H36 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id DLinear_MS2023_Illness_36 --model DLinear --data custom --features M --seq_len 104 --label_len 18 --pred_len 36 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

REM ========== Horizon 48 (3 models Ã— 3 seeds = 9 exp) ==========
echo [19/36] PFB_v0 + Illness + H48 + seed=2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PFB_v0_MS2021_Illness_48 --model PatchFusionBERT_v0 --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2021

echo [20/36] PatchTST + Illness + H48 + seed=2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PatchTST_MS2021_Illness_48 --model PatchTST --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2021

echo [21/36] DLinear + Illness + H48 + seed=2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id DLinear_MS2021_Illness_48 --model DLinear --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2021

echo [22/36] PFB_v0 + Illness + H48 + seed=2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PFB_v0_MS2022_Illness_48 --model PatchFusionBERT_v0 --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2022

echo [23/36] PatchTST + Illness + H48 + seed=2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PatchTST_MS2022_Illness_48 --model PatchTST --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2022

echo [24/36] DLinear + Illness + H48 + seed=2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id DLinear_MS2022_Illness_48 --model DLinear --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2022

echo [25/36] PFB_v0 + Illness + H48 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PFB_v0_MS2023_Illness_48 --model PatchFusionBERT_v0 --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [26/36] PatchTST + Illness + H48 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PatchTST_MS2023_Illness_48 --model PatchTST --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [27/36] DLinear + Illness + H48 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id DLinear_MS2023_Illness_48 --model DLinear --data custom --features M --seq_len 104 --label_len 18 --pred_len 48 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

REM ========== Horizon 60 (3 models Ã— 3 seeds = 9 exp) ==========
echo [28/36] PFB_v0 + Illness + H60 + seed=2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PFB_v0_MS2021_Illness_60 --model PatchFusionBERT_v0 --data custom --features M --seq_len 104 --label_len 18 --pred_len 60 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2021

echo [29/36] PatchTST + Illness + H60 + seed=2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PatchTST_MS2021_Illness_60 --model PatchTST --data custom --features M --seq_len 104 --label_len 18 --pred_len 60 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2021

echo [30/36] DLinear + Illness + H60 + seed=2021
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id DLinear_MS2021_Illness_60 --model DLinear --data custom --features M --seq_len 104 --label_len 18 --pred_len 60 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2021

echo [31/36] PFB_v0 + Illness + H60 + seed=2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PFB_v0_MS2022_Illness_60 --model PatchFusionBERT_v0 --data custom --features M --seq_len 104 --label_len 18 --pred_len 60 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2022

echo [32/36] PatchTST + Illness + H60 + seed=2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PatchTST_MS2022_Illness_60 --model PatchTST --data custom --features M --seq_len 104 --label_len 18 --pred_len 60 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2022

echo [33/36] DLinear + Illness + H60 + seed=2022
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id DLinear_MS2022_Illness_60 --model DLinear --data custom --features M --seq_len 104 --label_len 18 --pred_len 60 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2022

echo [34/36] PFB_v0 + Illness + H60 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PFB_v0_MS2023_Illness_60 --model PatchFusionBERT_v0 --data custom --features M --seq_len 104 --label_len 18 --pred_len 60 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --n_heads 8 --patch_len 16 --stride 8 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [35/36] PatchTST + Illness + H60 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id PatchTST_MS2023_Illness_60 --model PatchTST --data custom --features M --seq_len 104 --label_len 18 --pred_len 60 --e_layers 3 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 128 --d_ff 512 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo [36/36] DLinear + Illness + H60 + seed=2023
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./data/ --data_path national_illness.csv --model_id DLinear_MS2023_Illness_60 --model DLinear --data custom --features M --seq_len 104 --label_len 18 --pred_len 60 --e_layers 2 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des illness_horizons --d_model 512 --d_ff 2048 --itr 1 --train_epochs 100 --patience 10 --batch_size 16 --learning_rate 0.0001 --seed 2023

echo.
echo ============================================================================
echo ILLNESS HORIZONS MULTI-SEED COMPLETED
echo Ended: %date% %time%
echo ============================================================================
echo.
echo VERIFICATION CHECKLIST:
echo 1. Check result_long_term_forecast.txt - should have 36 new entries
echo 2. Verify: 3 models Ã— 4 horizons Ã— 3 seeds = 36 experiments
echo 3. All horizons should show H={24,36,48,60} in model_id
echo 4. No H=192 or H=336 mislabeling
echo.
echo NOTE: Illness results go in separate table with proper horizon labels

