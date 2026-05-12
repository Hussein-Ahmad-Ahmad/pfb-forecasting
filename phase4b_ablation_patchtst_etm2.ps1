# Phase 4B: PatchTST Ablation Study on ETTm2 H=96
# Total: 24 experiments (8 variants � 3 seeds)
# Ensure we're in the correct directory
Set-Location "d:\Hussein-experiments\dynamic-ensemble-paper\Ts-library\Time-Series-Library"
Write-Host 'Experiment 1/24: Small_Model seed=2021'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Small_Model_seed2021_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 64 --d_ff 256 --e_layers 3 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2021

Write-Host 'Experiment 2/24: Small_Model seed=2022'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Small_Model_seed2022_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 64 --d_ff 256 --e_layers 3 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2022

Write-Host 'Experiment 3/24: Small_Model seed=2023'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Small_Model_seed2023_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 64 --d_ff 256 --e_layers 3 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2023

Write-Host 'Experiment 4/24: Large_Model seed=2021'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Large_Model_seed2021_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 256 --d_ff 1024 --e_layers 3 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2021

Write-Host 'Experiment 5/24: Large_Model seed=2022'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Large_Model_seed2022_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 256 --d_ff 1024 --e_layers 3 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2022

Write-Host 'Experiment 6/24: Large_Model seed=2023'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Large_Model_seed2023_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 256 --d_ff 1024 --e_layers 3 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2023

Write-Host 'Experiment 7/24: Deep_Network seed=2021'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Deep_Network_seed2021_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 6 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2021

Write-Host 'Experiment 8/24: Deep_Network seed=2022'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Deep_Network_seed2022_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 6 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2022

Write-Host 'Experiment 9/24: Deep_Network seed=2023'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Deep_Network_seed2023_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 6 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2023

Write-Host 'Experiment 10/24: Shallow_Network seed=2021'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Shallow_Network_seed2021_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 1 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2021

Write-Host 'Experiment 11/24: Shallow_Network seed=2022'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Shallow_Network_seed2022_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 1 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2022

Write-Host 'Experiment 12/24: Shallow_Network seed=2023'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Shallow_Network_seed2023_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 1 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2023

Write-Host 'Experiment 13/24: More_Heads seed=2021'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_More_Heads_seed2021_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 3 --n_heads 16 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2021

Write-Host 'Experiment 14/24: More_Heads seed=2022'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_More_Heads_seed2022_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 3 --n_heads 16 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2022

Write-Host 'Experiment 15/24: More_Heads seed=2023'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_More_Heads_seed2023_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 3 --n_heads 16 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2023

Write-Host 'Experiment 16/24: Fewer_Heads seed=2021'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Fewer_Heads_seed2021_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 3 --n_heads 4 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2021

Write-Host 'Experiment 17/24: Fewer_Heads seed=2022'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Fewer_Heads_seed2022_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 3 --n_heads 4 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2022

Write-Host 'Experiment 18/24: Fewer_Heads seed=2023'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Fewer_Heads_seed2023_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 3 --n_heads 4 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2023

Write-Host 'Experiment 19/24: Large_Patch seed=2021'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Large_Patch_seed2021_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 3 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2021

Write-Host 'Experiment 20/24: Large_Patch seed=2022'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Large_Patch_seed2022_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 3 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2022

Write-Host 'Experiment 21/24: Large_Patch seed=2023'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Large_Patch_seed2023_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 3 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2023

Write-Host 'Experiment 22/24: Small_Patch seed=2021'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Small_Patch_seed2021_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 3 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2021

Write-Host 'Experiment 23/24: Small_Patch seed=2022'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Small_Patch_seed2022_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 3 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2022

Write-Host 'Experiment 24/24: Small_Patch seed=2023'
python run.py --task_name long_term_forecast --is_training 1 --model_id PatchTST_ETTm2_H96_Small_Patch_seed2023_Ablation --model PatchTST --data ETTm2 --root_path ./data/ETT/ --data_path ETTm2.csv --features M --seq_len 336 --label_len 48 --pred_len 96 --d_model 128 --d_ff 512 --e_layers 3 --n_heads 8 --train_epochs 100 --batch_size 32 --learning_rate 0.0001 --seed 2023
