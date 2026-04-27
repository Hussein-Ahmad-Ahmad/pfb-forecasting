# Results Analysis Folder - Index

**Location**: `Time-Series-Library/results_analysis/`  
**Date**: January 16, 2026

---

## 📊 Main Results Files

### 1. Phase 3C Outputs (Multi-Seed Validation)

**`result_long_term_forecast_phase3c_stats.csv`** ⭐ **MOST IMPORTANT**
- Mean ± Std for all model/dataset/horizon combinations
- 48 rows (8 models × 6 datasets/horizons)
- Columns: Model, Dataset, Horizon, MSE_mean, MSE_std, MAE_mean, MAE_std, N_runs
- **Use this for**: Main results table in your paper

**`result_long_term_forecast_phase3c_aggregated.csv`**
- Individual experiment results (all seeds)
- Raw data before aggregation
- **Use this for**: Detailed analysis, checking specific runs

**`result_long_term_forecast_phase3c_best_models.csv`**
- Best performing model for each dataset/horizon
- Quick reference for model selection
- **Use this for**: Discussion section, model recommendations

---

## 📈 Phase 4 Analysis Files

### 2. Model Rankings

**`phase4_model_ranking.csv`** ⭐ **KEY FILE**
- Overall model performance across all datasets
- Rankings from best to worst
- Includes mean, std, min, max for MSE and MAE
- **Winner**: PatchTST (Rank 1)
- **Use this for**: Model comparison section

**`phase4_best_models_per_config.csv`**
- Best model per dataset/horizon with robustness score
- Shows which model excels where
- **Use this for**: Dataset-specific recommendations

### 2B. Mechanistic Analysis (C1)

**`c1_mechanistic_analysis_summary.md`** ⭐ **NEW**
- Mechanistic explanation for why fusion helps more on ETTm2 and Weather than on Exchange
- Uses dataset characteristics instead of expensive model-internal probing
- **Use this for**: Results discussion, mechanism section

**`c1_dataset_characteristics.csv`**
- Dataset-level structure metrics: channel count, autocorrelation, redundancy, diversity
- **Use this for**: Quantitative support for mechanism claims

**`c1_fusion_gain_summary.csv`**
- Fusion-vs-baseline gains for the H=192 5-seed focus plus broader horizon checks
- **Use this for**: Supporting tables and figure captions

---

## 📝 Publication-Ready Materials

### 3. LaTeX Tables

**`table_h96_mse_latex.tex`** ⭐ **READY FOR PAPER**
- Complete LaTeX table for H=96 results
- Formatted with MSE values
- **Action**: Copy directly into your paper!

**`publication_figures/fig7_mechanistic_analysis.png`** ⭐ **NEW**
- Compact publication-style C1 figure linking dataset structure to fusion gains
- **Action**: Use in the mechanism/analysis subsection

---

## 📋 Reference Documents

**`PHASE4_COMPLETE_SUMMARY.md`**
- Complete overview of all phases
- Key findings summary
- Next steps and recommendations

**`phase4b_ablation_configs.json`**
- Hyperparameter configurations for ablation studies
- Reference for ablation experiments

---

## 🎯 Quick Start Guide

### For Writing Results Section:
1. Open: `result_long_term_forecast_phase3c_stats.csv`
2. Copy table from: `table_h96_mse_latex.tex`
3. Reference rankings from: `phase4_model_ranking.csv`

### For Model Comparison:
- Primary: `phase4_model_ranking.csv`
- Supporting: `phase4_best_models_per_config.csv`

### For Fusion Mechanism Discussion:
- Primary: `c1_mechanistic_analysis_summary.md`
- Quantitative support: `c1_dataset_characteristics.csv`
- Figure: `publication_figures/fig7_mechanistic_analysis.png`

### For Statistical Analysis:
- Main data: `result_long_term_forecast_phase3c_stats.csv`
- Individual runs: `result_long_term_forecast_phase3c_aggregated.csv`

---

## 📊 Key Findings (Quick Reference)

**Top 3 Models:**
1. PatchTST - MSE: 0.591
2. iTransformer - MSE: 0.597  
3. TimeXer - MSE: 0.618

**Easiest Dataset**: ETTm2 (MSE: 0.248)  
**Hardest Dataset**: custom/Illness (MSE: 1.797)

**Model Wins:**
- PatchTST: 2 wins (33.3%)
- DLinear: 2 wins (33.3%)
- TiDE: 1 win (16.7%)
- TimeXer: 1 win (16.7%)

**Mechanistic C1 Finding:**
- Fusion gains are positive on Weather (+1.82%) and ETTm2 (+0.91%) at the 5-seed H=192 focus, but negative on Exchange (-13.06%), consistent with Exchange being shorter and more redundant.

---

## 💡 What to Do Next

### Option 1: Start Writing Paper ✍️
- All tables ready in this folder
- Open LaTeX table and copy to paper
- Use stats CSV for any additional tables

### Option 2: Create Visualizations 📊
- Load `result_long_term_forecast_phase3c_stats.csv` in Python
- Create heatmaps, bar charts, box plots
- Compare model performance visually

### Option 3: Statistical Tests 📈
- Use aggregated CSV for paired tests
- Compare top models (PatchTST vs iTransformer)
- Calculate confidence intervals

---

**All files organized and ready for analysis!** 🎉
