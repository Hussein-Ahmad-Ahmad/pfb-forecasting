# C.C.2 Gaussian Noise + Channel Dropout Robustness
**Date:** 27 April 2026  
**Script:** `c2_gaussian_channel_robustness.py`  
**Output CSV:** `results_analysis/c2_gaussian_channel_robustness.csv`  
**Datasets:** ETTm2 + Weather, H=96+192, seed=2021  
**Models:** DLinear (CI), PatchTST (MS), PFBv0 (MS), PFBv2 (MS)

---

## Clean Baseline MSE

| Dataset | H | DLinear | PatchTST | PFBv0 | PFBv2 |
|---------|---|---------|----------|-------|-------|
| ETTm2 | 96 | 0.1694 | 0.1742 | **0.1679** | 0.1695 |
| ETTm2 | 192 | 0.2357 | **0.2282** | 0.2347 | 0.2347 |
| Weather | 96 | 0.1743 | 0.1526 | **0.1498** | 0.1496 |
| Weather | 192 | 0.2181 | 0.1962 | **0.1936** | 0.1938 |

---

## Gaussian Noise — % MSE Increase vs Clean

### σ = 0.5 (strongest noise)

| Dataset | H | DLinear | PatchTST | PFBv0 | PFBv2 |
|---------|---|---------|----------|-------|-------|
| ETTm2 | 96 | +2.2% | +2.2% | +3.2% | **+1.8%** |
| ETTm2 | 192 | +1.4% | +2.2% | **-0.6%*** | +1.9% |
| Weather | 96 | **+0.8%** | +8.9% | +10.7% | +10.1% |
| Weather | 192 | **+0.5%** | +7.7% | +6.2% | +5.7% |

*\* PFBv0 ETTm2 H=192 at σ=0.5 shows marginal MSE reduction — within stochastic variation.*

**Key finding — Gaussian noise:**
- On **ETTm2** (low-variance, univariate-like): all models are highly robust (<3.5% degradation at σ=0.5).
- On **Weather** (21 channels, high feature diversity): **DLinear is far more robust** (<1%) than patch-based models (5.7–10.7%). DLinear's per-channel linear design inherently averages out noise; patch-based MS models embed channels jointly and propagate noise across the shared representation.
- PFBv2 is slightly more robust than PFBv0 on Weather (5.7% vs 6.2% at H=192; 10.1% vs 10.7% at H=96) — marginal difference.

---

## Channel Dropout — % MSE Increase vs Clean

### 10% channels zeroed

| Dataset | H | DLinear | PatchTST | PFBv0 | PFBv2 |
|---------|---|---------|----------|-------|-------|
| ETTm2 | 96 | 242.5% | 235.0% | 244.6% | 242.0% |
| ETTm2 | 192 | 170.8% | 177.2% | 171.8% | 171.5% |
| Weather | 96 | **23.2%** | 27.4% | 28.3% | 28.1% |
| Weather | 192 | **16.8%** | 19.5% | 19.8% | 19.6% |

### 30% channels zeroed

| Dataset | H | DLinear | PatchTST | PFBv0 | PFBv2 |
|---------|---|---------|----------|-------|-------|
| ETTm2 | 96 | 500.4% | 485.4% | 505.0% | **499.8%** |
| ETTm2 | 192 | **349.7%** | 363.0% | 351.7% | 351.7% |
| Weather | 96 | **75.9%** | 91.2% | 93.6% | 93.6% |
| Weather | 192 | **54.9%** | 64.2% | 65.4% | 65.4% |

**Key finding — Channel dropout:**
- **ETTm2 (7 channels)**: 10% dropout removes ~1 channel → catastrophic for all models (~235–245% increase). 30% removes ~2 channels → >485% increase. No model is resilient. Differences between models are negligible.
- **Weather (21 channels)**: Much more gradual degradation due to redundancy across 21 channels. At 30%, models lose ~6 channels; 55–94% MSE increase.
- **DLinear consistently least affected**: DLinear uses channel-independent (CI) training — each channel has its own linear weights, so zeroing channels only degrades the predictions for those specific channels and leaves remaining channels unaffected. PatchTST and PFB are MS-mode (multivariate), so channel dropout corrupts the shared representation and degrades all channel predictions.
- **PFBv0/PFBv2 are not more robust than PatchTST** under channel dropout — the cross-channel BERT refinement does not provide resilience to missing variables.

---

## Summary for Manuscript

### Narrative (for robustness section)

Two additional corruption types were evaluated beyond zero-masking: Gaussian noise (σ ∈ {0.1, 0.3, 0.5} × per-feature std) and channel dropout (complete zeroing of 10%/20%/30% of channels, drawn uniformly at random at test time without retraining).

**Gaussian noise:** All models are highly resilient on ETTm2 (<3.5% MSE degradation at σ=0.5). On Weather, DLinear's channel-independent linear structure provides near-perfect noise rejection (<1%), while patch-based models degrade 6–11%. No fusion-based architecture advantage is observed under additive noise.

**Channel dropout:** Degradation scales with the fraction of channels zeroed and is critically dependent on the number of available channels. On ETTm2 (7 channels), even 10% dropout is catastrophic for all models equally (235–245% MSE increase). On Weather (21 channels), 30% dropout yields 55–94% degradation with DLinear consistently least affected due to per-channel independence. PFBv0's cross-channel refinement provides no robustness benefit over PatchTST under this corruption pattern.

**Scope note:** These robustness evaluations are a case study at seed 2021, not a comprehensive per-dataset survey. Results are consistent with the C.B.1 zero-masking findings: no architecture dominates robustness across corruption types.

---

## Checklist Update

- ✅ **C.C.2** — Gaussian noise + channel dropout robustness complete
- CSV: `results_analysis/c2_gaussian_channel_robustness.csv` (112 rows)
- Summary: this file
