# Phase 3B Resume Script - Missing Exchange and Weather Experiments
# Total: 24 experiments (2 models × 2 datasets × 2 horizons × 3 seeds)

Write-Host 'Running experiment 1/24: DLinear_Exchange_H96_seed2021'
python run.py --task_name long_term_forecast --is_training 1 --model_id DLinear_Exchange_H96_seed2021 --model DLinear --data custom --data_path exchange_rate.csv --features M --seq_len 336 --label_len 48 --d_model 512 --d_ff 2048 --pred_len 96 --train_epochs 100 --seed 2021

Write-Host 'Running experiment 2/24: DLinear_Exchange_H96_seed2022'
python run.py --task_name long_term_forecast --is_training 1 --model_id DLinear_Exchange_H96_seed2022 --model DLinear --data custom --data_path exchange_rate.csv --features M --seq_len 336 --label_len 48 --d_model 512 --d_ff 2048 --pred_len 96 --train_epochs 100 --seed 2022

Write-Host 'Running experiment 3/24: DLinear_Exchange_H96_seed2023'
python run.py --task_name long_term_forecast --is_training 1 --model_id DLinear_Exchange_H96_seed2023 --model DLinear --data custom --data_path exchange_rate.csv --features M --seq_len 336 --label_len 48 --d_model 512 --d_ff 2048 --pred_len 96 --train_epochs 100 --seed 2023

Write-Host 'Running experiment 4/24: DLinear_Exchange_H336_seed2021'
python run.py --task_name long_term_forecast --is_training 1 --model_id DLinear_Exchange_H336_seed2021 --model DLinear --data custom --data_path exchange_rate.csv --features M --seq_len 336 --label_len 48 --d_model 512 --d_ff 2048 --pred_len 336 --train_epochs 100 --seed 2021

Write-Host 'Running experiment 5/24: DLinear_Exchange_H336_seed2022'
python run.py --task_name long_term_forecast --is_training 1 --model_id DLinear_Exchange_H336_seed2022 --model DLinear --data custom --data_path exchange_rate.csv --features M --seq_len 336 --label_len 48 --d_model 512 --d_ff 2048 --pred_len 336 --train_epochs 100 --seed 2022

Write-Host 'Running experiment 6/24: DLinear_Exchange_H336_seed2023'
python run.py --task_name long_term_forecast --is_training 1 --model_id DLinear_Exchange_H336_seed2023 --model DLinear --data custom --data_path exchange_rate.csv --features M --seq_len 336 --label_len 48 --d_model 512 --d_ff 2048 --pred_len 336 --train_epochs 100 --seed 2023

Write-Host 'Running experiment 7/24: DLinear_Weather_H96_seed2021'
python run.py --task_name long_term_forecast --is_training 1 --model_id DLinear_Weather_H96_seed2021 --model DLinear --data custom --data_path weather.csv --features M --seq_len 336 --label_len 48 --d_model 512 --d_ff 2048 --pred_len 96 --train_epochs 100 --seed 2021

Write-Host 'Running experiment 8/24: DLinear_Weather_H96_seed2022'
python run.py --task_name long_term_forecast --is_training 1 --model_id DLinear_Weather_H96_seed2022 --model DLinear --data custom --data_path weather.csv --features M --seq_len 336 --label_len 48 --d_model 512 --d_ff 2048 --pred_len 96 --train_epochs 100 --seed 2022

Write-Host 'Running experiment 9/24: DLinear_Weather_H96_seed2023'
python run.py --task_name long_term_forecast --is_training 1 --model_id DLinear_Weather_H96_seed2023 --model DLinear --data custom --data_path weather.csv --features M --seq_len 336 --label_len 48 --d_model 512 --d_ff 2048 --pred_len 96 --train_epochs 100 --seed 2023

Write-Host 'Running experiment 10/24: DLinear_Weather_H336_seed2021'
python run.py --task_name long_term_forecast --is_training 1 --model_id DLinear_Weather_H336_seed2021 --model DLinear --data custom --data_path weather.csv --features M --seq_len 336 --label_len 48 --d_model 512 --d_ff 2048 --pred_len 336 --train_epochs 100 --seed 2021

Write-Host 'Running experiment 11/24: DLinear_Weather_H336_seed2022'
python run.py --task_name long_term_forecast --is_training 1 --model_id DLinear_Weather_H336_seed2022 --model DLinear --data custom --data_path weather.csv --features M --seq_len 336 --label_len 48 --d_model 512 --d_ff 2048 --pred_len 336 --train_epochs 100 --seed 2022

Write-Host 'Running experiment 12/24: DLinear_Weather_H336_seed2023'
python run.py --task_name long_term_forecast --is_training 1 --model_id DLinear_Weather_H336_seed2023 --model DLinear --data custom --data_path weather.csv --features M --seq_len 336 --label_len 48 --d_model 512 --d_ff 2048 --pred_len 336 --train_epochs 100 --seed 2023

Write-Host 'Running experiment 13/24: PatchTST_Exchange_H96_seed2021'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_Exchange_H96_seed2021 --model PatchTST --data custom --data_path exchange_rate.csv --features M --seq_len 336 --label_len 48 --d_model 128 --d_ff 512 --e_layers 3 --pred_len 96 --train_epochs 100 --seed 2021

Write-Host 'Running experiment 14/24: PatchTST_Exchange_H96_seed2022'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_Exchange_H96_seed2022 --model PatchTST --data custom --data_path exchange_rate.csv --features M --seq_len 336 --label_len 48 --d_model 128 --d_ff 512 --e_layers 3 --pred_len 96 --train_epochs 100 --seed 2022

Write-Host 'Running experiment 15/24: PatchTST_Exchange_H96_seed2023'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_Exchange_H96_seed2023 --model PatchTST --data custom --data_path exchange_rate.csv --features M --seq_len 336 --label_len 48 --d_model 128 --d_ff 512 --e_layers 3 --pred_len 96 --train_epochs 100 --seed 2023

Write-Host 'Running experiment 16/24: PatchTST_Exchange_H336_seed2021'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_Exchange_H336_seed2021 --model PatchTST --data custom --data_path exchange_rate.csv --features M --seq_len 336 --label_len 48 --d_model 128 --d_ff 512 --e_layers 3 --pred_len 336 --train_epochs 100 --seed 2021

Write-Host 'Running experiment 17/24: PatchTST_Exchange_H336_seed2022'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_Exchange_H336_seed2022 --model PatchTST --data custom --data_path exchange_rate.csv --features M --seq_len 336 --label_len 48 --d_model 128 --d_ff 512 --e_layers 3 --pred_len 336 --train_epochs 100 --seed 2022

Write-Host 'Running experiment 18/24: PatchTST_Exchange_H336_seed2023'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_Exchange_H336_seed2023 --model PatchTST --data custom --data_path exchange_rate.csv --features M --seq_len 336 --label_len 48 --d_model 128 --d_ff 512 --e_layers 3 --pred_len 336 --train_epochs 100 --seed 2023

Write-Host 'Running experiment 19/24: PatchTST_Weather_H96_seed2021'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_Weather_H96_seed2021 --model PatchTST --data custom --data_path weather.csv --features M --seq_len 336 --label_len 48 --d_model 128 --d_ff 512 --e_layers 3 --pred_len 96 --train_epochs 100 --seed 2021

Write-Host 'Running experiment 20/24: PatchTST_Weather_H96_seed2022'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_Weather_H96_seed2022 --model PatchTST --data custom --data_path weather.csv --features M --seq_len 336 --label_len 48 --d_model 128 --d_ff 512 --e_layers 3 --pred_len 96 --train_epochs 100 --seed 2022

Write-Host 'Running experiment 21/24: PatchTST_Weather_H96_seed2023'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_Weather_H96_seed2023 --model PatchTST --data custom --data_path weather.csv --features M --seq_len 336 --label_len 48 --d_model 128 --d_ff 512 --e_layers 3 --pred_len 96 --train_epochs 100 --seed 2023

Write-Host 'Running experiment 22/24: PatchTST_Weather_H336_seed2021'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_Weather_H336_seed2021 --model PatchTST --data custom --data_path weather.csv --features M --seq_len 336 --label_len 48 --d_model 128 --d_ff 512 --e_layers 3 --pred_len 336 --train_epochs 100 --seed 2021

Write-Host 'Running experiment 23/24: PatchTST_Weather_H336_seed2022'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_Weather_H336_seed2022 --model PatchTST --data custom --data_path weather.csv --features M --seq_len 336 --label_len 48 --d_model 128 --d_ff 512 --e_layers 3 --pred_len 336 --train_epochs 100 --seed 2022

Write-Host 'Running experiment 24/24: PatchTST_Weather_H336_seed2023'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_Weather_H336_seed2023 --model PatchTST --data custom --data_path weather.csv --features M --seq_len 336 --label_len 48 --d_model 128 --d_ff 512 --e_layers 3 --pred_len 336 --train_epochs 100 --seed 2023
