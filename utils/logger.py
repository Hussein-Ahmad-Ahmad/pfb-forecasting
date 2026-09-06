"""
Enhanced logging for multi-seed experiments with full reproducibility tracking.
"""

import csv
import os
import subprocess
from datetime import datetime
import torch


class ExperimentLogger:
    """
    Comprehensive experiment logger for benchmark campaigns.
    Tracks: results, runtime, resources, reproducibility info.
    """
    
    def __init__(self, log_file='results_detailed.csv'):
        self.log_file = log_file
        self.headers = [
            'timestamp',
            'model',
            'dataset', 
            'horizon',
            'seed',
            'mse',
            'mae',
            'params_M',
            'flops_relative',
            'wall_time_mins',
            'epochs_trained',
            'early_stop_epoch',
            'peak_vram_gb',
            'git_hash'
        ]
        
        # Create file with headers if doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
    
    def get_git_hash(self):
        """Get current git commit hash for reproducibility."""
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL
            ).decode().strip()
            return git_hash[:8]  # Short hash
        except:
            return 'unknown'
    
    def get_gpu_memory(self):
        """Get peak GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
        return 0.0
    
    def log_experiment(self, 
                      model, dataset, horizon, seed,
                      mse, mae, 
                      params_M, flops_relative,
                      wall_time_mins, epochs_trained, early_stop_epoch=-1):
        """
        Log single experiment results.
        
        Args:
            model: Model name (e.g., 'PFB-Direct', 'PatchTST')
            dataset: Dataset name (e.g., 'ETTm1', 'Weather')
            horizon: Prediction horizon (96, 192, 336, 720)
            seed: Random seed used
            mse: Test MSE
            mae: Test MAE
            params_M: Model parameters in millions
            flops_relative: FLOPs relative to baseline
            wall_time_mins: Total training time in minutes
            epochs_trained: Actual number of epochs run
            early_stop_epoch: Epoch where early stopping occurred (-1 if hit max)
        """
        row = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            model,
            dataset,
            horizon,
            seed,
            f'{mse:.6f}',
            f'{mae:.6f}',
            f'{params_M:.2f}',
            f'{flops_relative:.1f}',
            f'{wall_time_mins:.2f}',
            epochs_trained,
            early_stop_epoch,
            f'{self.get_gpu_memory():.2f}',
            self.get_git_hash()
        ]
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        print(f'Logged: {model} | {dataset} | H{horizon} | seed={seed} | MSE={mse:.4f}')
    
    def log_from_result_file(self, result_line, model, dataset, horizon, seed,
                            params_M, flops_relative, wall_time_mins, 
                            epochs_trained, early_stop_epoch=-1):
        """
        Parse result from result_long_term_forecast.txt format and log.
        
        Result line format: "mse:X.XXX, mae:Y.YYY"
        """
        try:
            parts = result_line.split(',')
            mse = float(parts[0].split(':')[1])
            mae = float(parts[1].split(':')[1])
            
            self.log_experiment(
                model, dataset, horizon, seed,
                mse, mae,
                params_M, flops_relative,
                wall_time_mins, epochs_trained, early_stop_epoch
            )
        except Exception as e:
            print(f'Error parsing result line: {e}')


# Usage example:
"""
from utils.logger import ExperimentLogger

logger = ExperimentLogger('results_multiseed.csv')

# After experiment completes:
logger.log_experiment(
    model='PFB-Direct',
    dataset='ETTm2', 
    horizon=192,
    seed=2021,
    mse=0.234,
    mae=0.312,
    params_M=3.5,
    flops_relative=32.0,
    wall_time_mins=38.2,
    epochs_trained=45,
    early_stop_epoch=35
)
"""
