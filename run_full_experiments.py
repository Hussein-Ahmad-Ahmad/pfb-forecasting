"""
Full Training Script with Comprehensive Analysis
Runs complete training sessions with ablation studies and visualizations
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
from comprehensive_analysis import ComprehensiveAnalyzer

# Change to script directory to ensure relative paths work
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_full_training(args, analyzer):
    """
    Run full training with specified configuration
    """
    print(f"\n{'='*80}")
    print(f"FULL TRAINING: {args.model} on {args.data}")
    print(f"{'='*80}\n")
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Build command
    cmd = f"python run.py --task_name long_term_forecast --is_training 1 " \
          f"--root_path ./data/ --data_path {args.data_path} " \
          f"--model_id {args.data}_{args.model}_full --model {args.model} " \
          f"--data {args.data_type} --features M " \
          f"--seq_len {args.seq_len} --label_len {args.label_len} --pred_len {args.pred_len} " \
          f"--e_layers {args.e_layers} --d_layers {args.d_layers} " \
          f"--enc_in {args.enc_in} --dec_in {args.dec_in} --c_out {args.c_out} " \
          f"--d_model {args.d_model} --n_heads {args.n_heads} --d_ff {args.d_ff} " \
          f"--dropout {args.dropout} --des {args.des} --itr {args.itr} " \
          f"--train_epochs {args.train_epochs} --batch_size {args.batch_size} " \
          f"--learning_rate {args.learning_rate} --patience {args.patience} --num_workers 4"
    
    if hasattr(args, 'patch_len'):
        cmd += f" --patch_len {args.patch_len} --stride {args.stride}"
    
    print(f"Command: {cmd}\n")
    
    # Record start time
    start_time = time.time()
    
    # Execute
    exit_code = os.system(cmd)
    
    # Record end time
    end_time = time.time()
    training_time = end_time - start_time
    
    if exit_code == 0:
        print(f"\n✓ Training completed in {training_time:.2f} seconds")
        
        # Log efficiency metrics
        exp_key = f"{args.data}_{args.model}"
        analyzer.results['efficiency_metrics'][exp_key] = {
            'training_time': training_time,
            'training_time_per_epoch': training_time / args.train_epochs
        }
        
        return True
    else:
        print(f"\n✗ Training failed with exit code {exit_code}")
        return False


def run_ablation_study(dataset_configs, analyzer):
    """
    Run complete ablation study:
    - Patch-only
    - BERT-only  
    - Fusion v0
    - Fusion v2
    """
    print(f"\n{'='*80}")
    print("ABLATION STUDY: Dual-Stream Fusion Analysis")
    print(f"{'='*80}\n")
    
    ablation_models = [
        'PatchFusionBERT_PatchOnly',
        'PatchFusionBERT_BERTOnly',
        'PatchFusionBERT_v0',
        'PatchFusionBERT_v2'
    ]
    
    for dataset_name, config in dataset_configs.items():
        print(f"\n📊 Running ablation study for dataset: {dataset_name}")
        print("-" * 80)
        
        for model in ablation_models:
            config_copy = argparse.Namespace(**vars(config))
            config_copy.model = model
            config_copy.des = f'Ablation_{model}'
            
            success = run_full_training(config_copy, analyzer)
            
            if success:
                # Parse results from result file
                try:
                    result_file = './result_long_term_forecast.txt'
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            lines = f.readlines()
                            # Find the last result for this model
                            for line in reversed(lines):
                                if model in line:
                                    # Parse MSE and MAE
                                    if 'mse:' in line and 'mae:' in line:
                                        mse_str = line.split('mse:')[1].split(',')[0]
                                        mae_str = line.split('mae:')[1].split(',')[0]
                                        mse = float(mse_str)
                                        mae = float(mae_str)
                                        
                                        exp_key = f"{dataset_name}_{model}"
                                        analyzer.results['final_metrics'][exp_key] = {
                                            'mse': mse,
                                            'mae': mae
                                        }
                                        print(f"  ✓ {model}: MAE={mae:.4f}, MSE={mse:.4f}")
                                        break
                except Exception as e:
                    print(f"  ⚠ Warning: Could not parse results for {model}: {e}")
            
            time.sleep(2)  # Brief pause between experiments


def run_patch_sensitivity_study(dataset_configs, analyzer):
    """
    Run patch length and stride sensitivity analysis
    """
    print(f"\n{'='*80}")
    print("PATCH SENSITIVITY STUDY")
    print(f"{'='*80}\n")
    
    patch_configs = [
        {'patch_len': 8, 'stride': 4},
        {'patch_len': 16, 'stride': 8},
        {'patch_len': 24, 'stride': 12},
        {'patch_len': 32, 'stride': 16},
    ]
    
    base_model = 'PatchFusionBERT_v0'
    
    for dataset_name, config in dataset_configs.items():
        print(f"\n📊 Patch sensitivity for dataset: {dataset_name}")
        print("-" * 80)
        
        for patch_config in patch_configs:
            config_copy = argparse.Namespace(**vars(config))
            config_copy.model = base_model
            config_copy.patch_len = patch_config['patch_len']
            config_copy.stride = patch_config['stride']
            config_copy.des = f'PatchSens_p{patch_config["patch_len"]}_s{patch_config["stride"]}'
            
            print(f"  Testing patch_len={patch_config['patch_len']}, stride={patch_config['stride']}")
            
            success = run_full_training(config_copy, analyzer)
            
            if success:
                try:
                    result_file = './result_long_term_forecast.txt'
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            lines = f.readlines()
                            for line in reversed(lines):
                                if base_model in line and config_copy.des in line:
                                    if 'mse:' in line and 'mae:' in line:
                                        mse_str = line.split('mse:')[1].split(',')[0]
                                        mae_str = line.split('mae:')[1].split(',')[0]
                                        mse = float(mse_str)
                                        mae = float(mae_str)
                                        
                                        exp_key = f"{dataset_name}_{base_model}_p{patch_config['patch_len']}_s{patch_config['stride']}"
                                        analyzer.results['final_metrics'][exp_key] = {
                                            'mse': mse,
                                            'mae': mae,
                                            'patch_len': patch_config['patch_len'],
                                            'stride': patch_config['stride']
                                        }
                                        print(f"    ✓ MAE={mae:.4f}, MSE={mse:.4f}")
                                        break
                except Exception as e:
                    print(f"    ⚠ Warning: Could not parse results: {e}")
            
            time.sleep(2)


def create_dataset_configs():
    """Create standard configurations for all datasets"""
    configs = {}
    
    # ETTm1
    configs['ETTm1'] = argparse.Namespace(
        data='ETTm1',
        data_type='ETTm1',
        data_path='ETTm1.csv',
        seq_len=336,
        label_len=48,
        pred_len=96,
        e_layers=3,
        d_layers=1,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=128,
        n_heads=8,
        d_ff=512,
        dropout=0.1,
        des='Full_Training',
        itr=1,
        train_epochs=20,  # Full training
        batch_size=16,  # Reduced from 32 to avoid GPU OOM
        learning_rate=0.0001,
        patience=5,
        patch_len=16,
        stride=8
    )
    
    # ETTh1
    configs['ETTh1'] = argparse.Namespace(
        data='ETTh1',
        data_type='ETTh1',
        data_path='ETTh1.csv',
        seq_len=336,
        label_len=48,
        pred_len=96,
        e_layers=3,
        d_layers=1,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=128,
        n_heads=8,
        d_ff=512,
        dropout=0.1,
        des='Full_Training',
        itr=1,
        train_epochs=20,
        batch_size=16,  # Reduced from 32 to avoid GPU OOM
        learning_rate=0.0001,
        patience=5,
        patch_len=16,
        stride=8
    )
    
    # ETTh2
    configs['ETTh2'] = argparse.Namespace(
        data='ETTh2',
        data_type='ETTh2',
        data_path='ETTh2.csv',
        seq_len=336,
        label_len=48,
        pred_len=96,
        e_layers=3,
        d_layers=1,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=128,
        n_heads=8,
        d_ff=512,
        dropout=0.1,
        des='Full_Training',
        itr=1,
        train_epochs=20,
        batch_size=32,
        learning_rate=0.0001,
        patience=5,
        patch_len=16,
        stride=8
    )
    
    # Weather
    configs['Weather'] = argparse.Namespace(
        data='Weather',
        data_type='custom',
        data_path='weather.csv',
        seq_len=336,
        label_len=48,
        pred_len=96,
        e_layers=3,
        d_layers=1,
        enc_in=21,
        dec_in=21,
        c_out=21,
        d_model=128,
        n_heads=8,
        d_ff=512,
        dropout=0.1,
        des='Full_Training',
        itr=1,
        train_epochs=20,
        batch_size=32,
        learning_rate=0.0001,
        patience=5,
        patch_len=16,
        stride=8
    )
    
    return configs


def main():
    parser = argparse.ArgumentParser(description='Full Training with Comprehensive Analysis')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'ablation', 'sensitivity', 'all'],
                       help='Experiment mode')
    parser.add_argument('--datasets', type=str, nargs='+', default=['ETTm1', 'ETTh1'],
                       help='Datasets to test')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                       help='Output directory for analysis')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ComprehensiveAnalyzer(args.output_dir)
    
    # Create dataset configurations
    all_configs = create_dataset_configs()
    selected_configs = {k: v for k, v in all_configs.items() if k in args.datasets}
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TRAINING AND ANALYSIS PIPELINE")
    print("="*80)
    print(f"\nMode: {args.mode}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Run experiments based on mode
    if args.mode in ['full', 'all']:
        print("\n🚀 Starting full training experiments...")
        for dataset_name, config in selected_configs.items():
            for model in ['PatchFusionBERT_v0', 'PatchFusionBERT_v2']:
                config_copy = argparse.Namespace(**vars(config))
                config_copy.model = model
                run_full_training(config_copy, analyzer)
    
    if args.mode in ['ablation', 'all']:
        print("\n🔬 Starting ablation studies...")
        run_ablation_study(selected_configs, analyzer)
    
    if args.mode in ['sensitivity', 'all']:
        print("\n📏 Starting patch sensitivity analysis...")
        run_patch_sensitivity_study(selected_configs, analyzer)
    
    # Generate visualizations and reports
    print("\n" + "="*80)
    print("GENERATING ANALYSIS AND VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Generate plots
    if args.mode in ['ablation', 'all']:
        for dataset in args.datasets:
            analyzer.plot_ablation_comparison(dataset)
    
    # Generate comparison tables
    models = ['PatchFusionBERT_v0', 'PatchFusionBERT_v2', 
              'PatchFusionBERT_PatchOnly', 'PatchFusionBERT_BERTOnly']
    df = analyzer.generate_comparison_table(args.datasets, models)
    
    # Generate efficiency analysis
    analyzer.plot_efficiency_analysis(models)
    
    # Save all results
    analyzer.save_results()
    analyzer.generate_summary_report()
    
    print("\n" + "="*80)
    print("✓ COMPREHENSIVE ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📁 All results saved to: {args.output_dir}")
    print(f"  • Plots: {os.path.join(args.output_dir, 'plots')}")
    print(f"  • Tables: {os.path.join(args.output_dir, 'tables')}")
    print(f"  • JSON: {os.path.join(args.output_dir, 'all_results.json')}")
    print(f"  • Report: {os.path.join(args.output_dir, 'summary_report.txt')}\n")


if __name__ == '__main__':
    main()
