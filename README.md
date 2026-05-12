# PatchFusionBERT: Post-Encoder Fusion for Patch-Based Long-Horizon Time Series Forecasting

> **Hussein Ahmad, Seyyed Kasra Mortazavi, Taha Benarbia, Fadi Al Machot, Kyandoghere Kyamakya**  
> Institute for Smart Systems Technologies, Universität Klagenfurt

This repository contains the code and results for the paper **"PatchFusionBERT: Post-Encoder Fusion for Patch-Based Long-Horizon Time Series Forecasting"**.  
The implementation is built on top of [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library).

---

## Overview

**PatchFusionBERT** augments a PatchTST-style patch encoder with:
1. A compact post-encoder Transformer (BERT-style) refinement stage
2. Explicit adaptive fusion between the refined patch representations and a direct linear path

Two fusion variants are studied:
- **PFBv0** — direct additive gate fusion
- **PFBv2** — attention-weighted projection fusion

The architecture targets long-horizon forecasting (H = 96/192/336) across seven public benchmarks under a unified training protocol (single seed 2021; DLinear LR = 1e-2, patch models LR = 1e-4; seq\_len = 336).

---

## Main Results (MSE / MAE, single seed, seq\_len = 336)

| Dataset | H | PFBv0 | PFBv2 | PatchTST | iTransformer | TiDE | TimeXer | DLinear |
|---------|---|-------|-------|----------|-------------|------|---------|---------|
| ETTh1 | 96 | **0.3675**/0.3952 | 0.3863/0.4053 | 0.3814/0.4063 | 0.3997/0.4174 | 0.3958/0.4141 | 0.3909/0.4076 | 0.3711/**0.3927** |
| ETTh1 | 192 | **0.4043**/0.4242 | 0.4198/0.4266 | 0.4312/0.4325 | 0.4495/0.4506 | 0.4283/0.4327 | 0.4239/0.4269 | **0.4043**/**0.4128** |
| ETTh1 | 336 | 0.4378/0.4476 | 0.4411/0.4404 | 0.4934/0.4901 | 0.4636/0.4645 | 0.4502/0.4466 | 0.4575/0.4532 | **0.4345**/**0.4352** |
| ETTh2 | 96 | 0.3054/0.3524 | 0.2893/0.3474 | 0.3145/0.3647 | 0.3062/0.3606 | 0.2914/0.3515 | 0.2969/0.3604 | **0.2836**/**0.3473** |
| ETTh2 | 192 | 0.3720/0.4004 | **0.3673**/**0.3952** | 0.3887/0.4118 | 0.3667/0.4009 | **0.3517**/**0.3903** | 0.3614/0.3972 | 0.3797/0.4151 |
| ETTh2 | 336 | **0.3814**/**0.4139** | 0.3955/0.4225 | 0.4059/0.4293 | 0.3988/0.4229 | **0.3736**/**0.4109** | 0.3945/0.4265 | 0.4257/0.4555 |
| ETTm2 | 96 | **0.1679**/0.2571 | 0.1695/0.2567 | 0.1728/0.2640 | 0.1749/0.2660 | 0.1690/0.2595 | 0.1716/**0.2562** | **0.1653**/0.2574 |
| ETTm2 | 192 | 0.2347/**0.2997** | 0.2347/0.3001 | 0.2282/0.3018 | 0.2475/0.3139 | **0.2241**/**0.2965** | 0.2311/0.2986 | **0.2273**/0.3077 |
| ETTm2 | 336 | **0.2739**/**0.3291** | 0.3014/0.3496 | 0.2935/0.3457 | 0.3020/0.3484 | 0.2780/0.3314 | 0.2870/0.3351 | 0.2930/0.3524 |
| Weather | 96 | **0.1496**/0.2007 | 0.1522/**0.2000** | 0.1526/0.2019 | 0.1603/0.2096 | 0.1764/0.2267 | 0.1508/0.2024 | 0.1782/0.2432 |
| Weather | 192 | **0.1936**/**0.2387** | 0.1938/0.2394 | 0.1962/0.2429 | 0.2031/0.2487 | 0.2185/0.2612 | **0.1942**/**0.2409** | 0.2179/0.2775 |
| Weather | 336 | 0.2517/0.2839 | **0.2446**/**0.2802** | 0.2479/0.2814 | 0.2527/0.2867 | 0.2659/0.2957 | **0.2453**/0.2811 | 0.2612/0.3116 |

**Bold** = best; second-best underlined in the paper (see `paper/txt.tex`). Results use the unified BridgeLR protocol (single seed 2021). Full results including ETTm1, Exchange, and Illness are in the paper.

---

## Statistical Validation (5 seeds × 6 datasets, Wilcoxon signed-rank, Holm-corrected)

| Horizon pool | Comparison | MSE p (Holm) | MAE p (Holm) | Significant? |
|---|---|---|---|---|
| H=192 (9 configs) | PFBv0 vs DLinear | 0.027 | 0.00039 | ✅ Both |
| H=96 (6 datasets) | PFBv0 vs PatchTST | 0.056 | 0.00028 | ✅ MAE |
| H=336 (6 datasets) | PFBv0 vs PatchTST | 0.015 | 0.0044 | ✅ Both |
| H=336 (6 datasets) | PFBv0 vs DLinear | 0.070 | 0.028 | ✅ MAE |

---

## Repository Structure

```
models/
  PatchFusionBERT_v0.py        # PFBv0 — additive gate fusion (main variant)
  PatchFusionBERT_v2.py        # PFBv2 — attention-weighted projection fusion
  PatchFusionBERT.py           # shared base class
  PatchFusionBERT_BERTOnly.py  # ablation: refinement only (no linear path)
  PatchFusionBERT_PatchOnly.py # ablation: backbone only (no refinement)
  PatchFusionBERT_RefineOnly.py# ablation: backbone + BERT, no fusion gate
  PatchTST_LargeHead.py        # head-capacity control (3.45M params)
  PatchTST.py / DLinear.py / iTransformer.py / TiDE.py / TimeXer.py  # baselines
exp/ data_provider/ layers/ utils/  # TSLib framework
bridging_lr_unified.ps1        # main benchmark (Table 2) — single seed 2021
multiseed_h96_h336_5seed.ps1   # 5-seed H=96/336 experiments (Tables 3/14)
run_multiseed_5seed_campaign.ps1  # 5-seed H=192 (Table 3)
run_largehead_ablation.py      # PatchTST_LargeHead ablation (Table 8)
robustness_missing_data.py     # random/block missingness (Table 16)
c2_gaussian_channel_robustness.py # Gaussian noise + channel dropout
c4_cka_representation_similarity.py # CKA analysis (Table 11)
analyze_bridging_lr.py         # parse bridging LR results → Table 2
analyze_h96_h336_multiseed.py  # parse H=96/336 multiseed results
statistical_significance_multiseed_5seed.py # Wilcoxon tests (pooled H=192)
add_fdr_correction.py          # Holm + BH-FDR correction
count_params.py                # parameter count verification
measure_flops_macs.py          # FLOPs/MACs measurement (Table 13)
benchmark_training_weather192.py / benchmark_inference_weather192.py
results_analysis/              # parsed CSVs and LaTeX tables used in paper
paper/                         # manuscript (txt.tex)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Datasets (ETTh1/2, ETTm1/2, Weather, Exchange, National Illness) follow the [TSLib data preparation guide](https://github.com/thuml/Time-Series-Library#data-preparation). Place CSV files under `data/`.

---

## Reproducing the Main Benchmark (Table 2)

```powershell
# Windows PowerShell — runs all 108 single-seed experiments
.\bridging_lr_unified.ps1
# Then parse results
python analyze_bridging_lr.py
```

## Reproducing Multi-Seed Experiments (Tables 3 / 14 / appendix)

```powershell
# H=192, 5 seeds
.\run_multiseed_5seed_campaign.ps1
# H=96 and H=336, 5 seeds
.\multiseed_h96_h336_5seed.ps1
# Statistical tests
python statistical_significance_multiseed_5seed.py
python add_fdr_correction.py
python results_analysis/wilcoxon_h96_h336.py
```

## Reproducing Ablation (Table 8)

```powershell
.\b1_component_ablation_chain.ps1
python run_largehead_ablation.py
python analyze_b1_component_ablation.py
```

## Capacity-Matched Controls (Table 6)

```powershell
.\capmatch_controls.ps1
.\capmatch_controls_ett_weather_96_336.ps1
python analyze_capmatch_controls.py
```

---

## Figures

### Architecture (Fig. 1)
![Architecture](paper/figures/Fig1_architecture.png)

### MSE Heatmap across datasets and horizons (Fig. 2)
![MSE Heatmap](paper/figures/Fig2_mse_heatmap.png)

### Capacity-Matched Control Results (Fig. 3)
![Capacity Control](paper/figures/Fig3_capacity_control.png)

### Efficiency Pareto — Accuracy vs. Inference Cost (Fig. 4)
![Efficiency Pareto](paper/figures/Fig4_efficiency_pareto.png)

### Component Ablation Chain (Fig. 5)
![Ablation](paper/figures/Fig5_ablation_chain.png)

### Robustness Under Missing Data (Fig. 6)
![Robustness](paper/figures/Fig6_robustness_curves.png)

### CKA Representation Similarity (Fig. 7)
![CKA](paper/figures/Fig7_cka_similarity.png)

### Appendix figures
| | |
|---|---|
| ![A1](paper/figures/A1_full_errorbars.png) **A1** — 5-seed error bars | ![A2](paper/figures/A2_significance.png) **A2** — Significance summary |
| ![A3](paper/figures/A3_per_variable_weather.png) **A3** — Per-variable Weather error | ![A4](paper/figures/A4_error_by_step.png) **A4** — Error by forecast step |
| ![A5](paper/figures/A5_kdepth_sensitivity.png) **A5** — K-depth sensitivity | ![A6](paper/figures/A6_bridging_lr.png) **A6** — Bridging LR results |
| ![A7](paper/figures/A7_robustness_matrix.png) **A7** — Robustness matrix | ![A8](paper/figures/A8_gaussian_channel_robustness.png) **A8** — Gaussian & channel robustness |

---

## Citation

```bibtex
@article{ahmad2026patchfusionbert,
  title  = {PatchFusionBERT: Post-Encoder Fusion for Patch-Based Long-Horizon Time Series Forecasting},
  author = {Ahmad, Hussein and Mortazavi, Seyyed Kasra and Benarbia, Taha and Al Machot, Fadi and Kyamakya, Kyandoghere},
  journal= {IEEE Access},
  year   = {2026}
}
```

---

## Acknowledgements

This codebase is built on [Time-Series-Library](https://github.com/thuml/Time-Series-Library) by THUML @ Tsinghua University.
