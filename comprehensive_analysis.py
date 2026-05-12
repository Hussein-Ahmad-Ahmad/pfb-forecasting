"""
Comprehensive Training and Analysis Script for PatchFusionBERT
Includes full training, ablation studies, visualization, and analysis
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class ComprehensiveAnalyzer:
    """Comprehensive analysis and visualization for time series forecasting experiments"""
    
    def __init__(self, output_dir='./analysis_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
        
        self.results = {
            'training_curves': {},
            'final_metrics': {},
            'efficiency_metrics': {},
            'ablation_results': {}
        }
    
    def log_training_curve(self, model_name, epoch, metrics):
        """Log training curve data"""
        if model_name not in self.results['training_curves']:
            self.results['training_curves'][model_name] = []
        
        self.results['training_curves'][model_name].append({
            'epoch': epoch,
            **metrics
        })
    
    def plot_training_curves(self, models=None):
        """Plot training curves for multiple models"""
        if models is None:
            models = list(self.results['training_curves'].keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['train_loss', 'val_loss', 'val_mae', 'val_mse']
        titles = ['Training Loss', 'Validation Loss', 'Validation MAE', 'Validation MSE']
        
        for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
            ax = axes[idx // 2, idx % 2]
            
            for model_name in models:
                if model_name in self.results['training_curves']:
                    data = self.results['training_curves'][model_name]
                    epochs = [d['epoch'] for d in data]
                    values = [d.get(metric, np.nan) for d in data]
                    ax.plot(epochs, values, marker='o', label=model_name, linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'plots', 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {save_path}")
        plt.close()
    
    def plot_ablation_comparison(self, dataset_name):
        """Plot ablation study results (Patch-only vs BERT-only vs Fusion)"""
        ablation_models = ['PatchOnly', 'BERTOnly', 'PatchFusionBERT_v0', 'PatchFusionBERT_v2']
        
        # Extract metrics for ablation models
        metrics_data = {
            'MAE': [],
            'MSE': [],
            'Training Time (s)': [],
            'Parameters (M)': []
        }
        model_labels = []
        
        for model in ablation_models:
            key = f"{dataset_name}_{model}"
            if key in self.results['final_metrics']:
                data = self.results['final_metrics'][key]
                metrics_data['MAE'].append(data.get('mae', 0))
                metrics_data['MSE'].append(data.get('mse', 0))
                
                if key in self.results['efficiency_metrics']:
                    eff_data = self.results['efficiency_metrics'][key]
                    metrics_data['Training Time (s)'].append(eff_data.get('training_time', 0))
                    metrics_data['Parameters (M)'].append(eff_data.get('params_M', 0))
                else:
                    metrics_data['Training Time (s)'].append(0)
                    metrics_data['Parameters (M)'].append(0)
                
                # Clean model name for display
                if model == 'PatchOnly':
                    model_labels.append('Patch-Only')
                elif model == 'BERTOnly':
                    model_labels.append('BERT-Only')
                elif model == 'PatchFusionBERT_v0':
                    model_labels.append('Fusion-v0')
                elif model == 'PatchFusionBERT_v2':
                    model_labels.append('Fusion-v2')
        
        if not model_labels:
            print(f"⚠ No ablation data found for {dataset_name}")
            return
        
        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Ablation Study: {dataset_name}', fontsize=16, fontweight='bold')
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        # MAE comparison
        ax = axes[0, 0]
        bars = ax.bar(model_labels, metrics_data['MAE'], color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
        ax.set_title('Mean Absolute Error', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, metrics_data['MAE'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # MSE comparison
        ax = axes[0, 1]
        bars = ax.bar(model_labels, metrics_data['MSE'], color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('MSE', fontsize=12, fontweight='bold')
        ax.set_title('Mean Squared Error', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, metrics_data['MSE'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Training time comparison
        ax = axes[1, 0]
        bars = ax.bar(model_labels, metrics_data['Training Time (s)'], color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Training Time per Epoch', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, metrics_data['Training Time (s)'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{val:.1f}s', ha='center', va='bottom', fontsize=10)
        
        # Parameters comparison
        ax = axes[1, 1]
        bars = ax.bar(model_labels, metrics_data['Parameters (M)'], color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Parameters (M)', fontsize=12, fontweight='bold')
        ax.set_title('Model Parameters', fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, metrics_data['Parameters (M)'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                   f'{val:.2f}M', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'plots', f'ablation_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Ablation plot saved to {save_path}")
        plt.close()
    
    def plot_patch_sensitivity(self, dataset_name, model_name='PatchFusionBERT_v0'):
        """Plot patch length and stride sensitivity analysis"""
        # This will be populated during sensitivity experiments
        patch_configs = []
        mae_values = []
        
        # Search for sensitivity results
        for key in self.results['final_metrics']:
            if dataset_name in key and model_name in key and 'patch' in key:
                data = self.results['final_metrics'][key]
                # Extract patch config from key if available
                mae_values.append(data.get('mae', 0))
        
        if not mae_values:
            print(f"⚠ No patch sensitivity data found for {dataset_name}/{model_name}")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # Placeholder for actual sensitivity plot
        ax.text(0.5, 0.5, 'Patch Sensitivity Analysis\n(Requires multiple runs with different configs)', 
               ha='center', va='center', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'plots', f'patch_sensitivity_{dataset_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comparison_table(self, datasets, models):
        """Generate comprehensive comparison table"""
        rows = []
        
        for dataset in datasets:
            for model in models:
                key = f"{dataset}_{model}"
                if key in self.results['final_metrics']:
                    metrics = self.results['final_metrics'][key]
                    eff = self.results['efficiency_metrics'].get(key, {})
                    
                    rows.append({
                        'Dataset': dataset,
                        'Model': model,
                        'MAE': metrics.get('mae', np.nan),
                        'MSE': metrics.get('mse', np.nan),
                        'Params (M)': eff.get('params_M', np.nan),
                        'Train Time (s)': eff.get('training_time', np.nan),
                        'Inference Time (ms)': eff.get('inference_time_ms', np.nan)
                    })
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'tables', 'comprehensive_results.csv')
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"✓ Results table saved to {csv_path}")
        
        # Create formatted LaTeX table
        latex_path = os.path.join(self.output_dir, 'tables', 'comprehensive_results.tex')
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, float_format='%.4f'))
        print(f"✓ LaTeX table saved to {latex_path}")
        
        return df
    
    def plot_efficiency_analysis(self, models):
        """Plot parameter count vs performance trade-off"""
        params = []
        mae = []
        labels = []
        
        for model in models:
            model_keys = [k for k in self.results['final_metrics'] if model in k]
            if model_keys:
                # Average across datasets
                avg_mae = np.mean([self.results['final_metrics'][k].get('mae', np.nan) 
                                  for k in model_keys])
                avg_params = np.mean([self.results['efficiency_metrics'].get(k, {}).get('params_M', np.nan) 
                                     for k in model_keys if k in self.results['efficiency_metrics']])
                
                if not np.isnan(avg_mae) and not np.isnan(avg_params):
                    params.append(avg_params)
                    mae.append(avg_mae)
                    labels.append(model)
        
        if not params:
            print("⚠ No efficiency data available")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(params, mae, s=200, alpha=0.6, c=range(len(params)), cmap='viridis', edgecolors='black')
        
        for i, label in enumerate(labels):
            ax.annotate(label, (params[i], mae[i]), fontsize=10, ha='center', va='bottom')
        
        ax.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average MAE', fontsize=12, fontweight='bold')
        ax.set_title('Model Efficiency: Parameters vs Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'plots', 'efficiency_tradeoff.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Efficiency plot saved to {save_path}")
        plt.close()
    
    def save_results(self):
        """Save all results to JSON"""
        json_path = os.path.join(self.output_dir, 'all_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ All results saved to {json_path}")
    
    def generate_summary_report(self):
        """Generate text summary report"""
        report_path = os.path.join(self.output_dir, 'summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE EXPERIMENT RESULTS SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. MODELS EVALUATED\n")
            f.write("-"*80 + "\n")
            models = set()
            for key in self.results['final_metrics'].keys():
                model = '_'.join(key.split('_')[1:])  # Extract model name
                models.add(model)
            for model in sorted(models):
                f.write(f"  • {model}\n")
            f.write("\n")
            
            f.write("2. DATASETS TESTED\n")
            f.write("-"*80 + "\n")
            datasets = set()
            for key in self.results['final_metrics'].keys():
                dataset = key.split('_')[0]  # Extract dataset name
                datasets.add(dataset)
            for dataset in sorted(datasets):
                f.write(f"  • {dataset}\n")
            f.write("\n")
            
            f.write("3. BEST RESULTS PER DATASET\n")
            f.write("-"*80 + "\n")
            dataset_best = {}
            for key, metrics in self.results['final_metrics'].items():
                dataset = key.split('_')[0]
                if dataset not in dataset_best or metrics.get('mae', float('inf')) < dataset_best[dataset]['mae']:
                    dataset_best[dataset] = {
                        'model': '_'.join(key.split('_')[1:]),
                        'mae': metrics.get('mae', float('inf')),
                        'mse': metrics.get('mse', float('inf'))
                    }
            
            for dataset in sorted(dataset_best.keys()):
                best = dataset_best[dataset]
                f.write(f"  {dataset:15s}: {best['model']:30s} (MAE: {best['mae']:.4f}, MSE: {best['mse']:.4f})\n")
            f.write("\n")
            
            f.write("="*80 + "\n")
        
        print(f"✓ Summary report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Analysis and Visualization')
    parser.add_argument('--output_dir', type=str, default='./analysis_results', help='Output directory')
    parser.add_argument('--plot_ablation', action='store_true', help='Generate ablation plots')
    parser.add_argument('--plot_training', action='store_true', help='Generate training curve plots')
    parser.add_argument('--generate_tables', action='store_true', help='Generate comparison tables')
    parser.add_argument('--datasets', type=str, nargs='+', default=['ETTm1', 'ETTh1', 'ETTh2', 'ETTm2', 'Weather', 'Exchange', 'Illness'])
    
    args = parser.parse_args()
    
    analyzer = ComprehensiveAnalyzer(args.output_dir)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS TOOL FOR PATCHFUSIONBERT")
    print("="*80 + "\n")
    
    # Example usage - in practice, this would be called from training scripts
    print("📊 Analyzer initialized. Ready to process experimental results.")
    print(f"📁 Output directory: {args.output_dir}")
    print("\nTo use this analyzer:")
    print("  1. Run full training experiments")
    print("  2. Log results using analyzer.log_training_curve() and analyzer.results")
    print("  3. Call visualization methods to generate plots\n")


if __name__ == '__main__':
    main()
