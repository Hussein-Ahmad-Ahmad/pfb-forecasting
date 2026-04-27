# Manuscript Integration Packet
**Generated:** 2025-04-27  
**Purpose:** Copy-ready text and tables for integrating all new experimental evidence into access (29).pdf.  
Each section maps to a specific manuscript location.

---

## 1. Efficiency Table (Section A4 → Results / Efficiency subsection)

Replace or extend the existing efficiency table with the following. This adds PatchTSTcap and FLOPs — both explicitly requested by the professor.

### Table: Efficiency Benchmark (Weather, H=192, repeated 3-run median)

| Model | Params | MACs | Train (s) | Infer (ms) | Throughput (s/s) | Train GPU (MB) | Infer GPU (MB) |
|---|---|---|---|---|---|---|---|
| DLinear | 129,408 | 2.72M | 321 ± 34 | 0.33 ± 0.00 | 23,821 | 1,966 | 10 |
| PatchTST | 1,629,504 | 546.87M | 1,018 ± 96 | 3.25 ± 0.25 | 2,382 | 2,812 | 62 |
| PatchTST_cap | **3,283,456** | **1.560G** | 1,378 ± 61 | 6.64 ± 0.03 | 1,198 | 3,107 | 193 |
| PatchFusionBERT v0 | 3,256,768 | **1.092G** | 1,375 ± 72 | **5.87 ± 0.01** | **1,362** | 3,205 | 72 |
| PatchFusionBERT v2 | 2,456,000 | 1.274G | 1,461 ± 3 | 6.75 ± 0.01 | 1,183 | 1,994 | 69 |

**Key sentence for manuscript:**
> "At matched parameter count (~3.26M), PatchFusionBERT v0 requires 1.092G MACs per forward pass compared to 1.560G for PatchTST_cap — a 30% FLOP reduction — while achieving competitive inference throughput (1,362 vs 1,198 samples/s) and lower GPU inference memory (72 MB vs 193 MB)."

**Source:** `results_analysis/efficiency_weather192_summary.md`, `results_analysis/flops_macs.csv`

---

## 2. Statistical Significance — Corrected P-values (Section A2 → Results)

### Pooled Wilcoxon signed-rank (N=45 pairs, 8 comparisons, α=0.05)

| Comparison | Metric | p_raw | p_holm | p_fdr_bh | Sig (Holm) | Sig (BH-FDR) |
|---|---|---|---|---|---|---|
| PFBv0 vs DLinear | MSE | 0.003907 | **0.027** | **0.016** | **Yes** | **Yes** |
| PFBv0 vs DLinear | MAE | 0.000048 | **0.000** | **0.000** | **Yes** | **Yes** |
| PFBv0 vs PatchTST | MAE | 0.011624 | 0.070 | **0.031** | No | **Yes** |
| PFBv0 vs PatchTST | MSE | 0.034875 | 0.174 | 0.070 | No | No |
| PFBv0 vs PFBv2 | MSE | 0.047536 | 0.190 | 0.076 | No | No |
| PFBv0 vs PFBv2 | MAE | 0.480683 | 0.961 | 0.538 | No | No |
| PFBv2 vs PatchTST | MSE | 0.538281 | 0.961 | 0.538 | No | No |
| PFBv2 vs PatchTST | MAE | 0.134285 | 0.403 | 0.179 | No | No |

**Per-config (72 tests, N=5 each):** Minimum resolvable p = 0.0625. No per-config comparison reaches uncorrected significance. All per-config differences should be described as directional trends.

**Key sentences for manuscript (Results section):**
> "After Holm-Bonferroni correction for 8 simultaneous tests, PatchFusionBERT v0 significantly outperforms DLinear on both MSE (p_holm=0.027) and MAE (p_holm<0.001). The comparison against PatchTST is borderline: significant under Benjamini-Hochberg FDR for MAE (p_fdr=0.031) but not under the more conservative Holm correction."

**Key sentence for Limitations section:**
> "Per-configuration Wilcoxon tests (N=5 per config) operate at minimum statistical resolution (p≥0.0625), precluding per-dataset significance claims. All per-configuration differences are reported as directional trends. FLOP-matched capacity control was not run; measured FLOPs show PatchFusionBERT v0 uses 30% fewer MACs than PatchTST_cap at equal parameter count."

**Source:** `results_analysis/multiseed_5seed_significance_pooled.csv` (columns `p_holm`, `p_fdr_bh`)

---

## 3. Error-by-Forecast-Step Curves (Section A6 → Mechanistic Analysis)

**Data in:** `results_analysis/error_by_step.csv` — 192 steps × 4 models × 2 datasets (Weather, ETTm2).

### Key finding — fusion advantage grows at longer steps:

**Weather H=192:**
| Steps | DLinear MSE | PatchTST MSE | PFBv0 MSE | PFBv0 vs PatchTST |
|---|---|---|---|---|
| Steps 1–24 (early) | 0.1060 | 0.0933 | 0.0937 | −0.4% |
| Steps 169–192 (late) | 0.2909 | 0.2730 | **0.2702** | **+1.1%** |

**ETTm2 H=192:**
| Steps | DLinear MSE | PatchTST MSE | PFBv0 MSE | PFBv0 vs PatchTST | PFBv0 vs DLinear |
|---|---|---|---|---|---|
| Steps 1–24 (early) | 0.0987 | 0.1024 | 0.1008 | +1.5% | −2.1% |
| Steps 169–192 (late) | 0.3222 | 0.3200 | **0.3061** | **+4.4%** | **+5.0%** |

**Key sentences for manuscript (Mechanistic Analysis / Discussion):**
> "Error-by-forecast-step analysis reveals that the benefit of post-encoder fusion is concentrated at longer forecast horizons. On ETTm2, PatchFusionBERT v0 trails DLinear by 2.1% at early steps (1–24) but leads by 5.0% at steps 169–192. A similar pattern holds for Weather, where the PatchTST advantage at early steps (0.4%) reverses to a PatchFusionBERT v0 advantage at late steps (1.1%). This is consistent with the hypothesis that the refinement stack provides longer-range contextual integration that standard patch-attention alone does not capture."

**Figure recommendation:** Line plot, x-axis = forecast step (1–192), y-axis = per-step MSE, 4 lines (DLinear/PatchTST/PFBv0/PFBv2), two panels side-by-side (Weather left, ETTm2 right). Vertical dashed line at step 24 and step 168 to delineate early/late regions.

**Source:** `results_analysis/error_by_step.csv`

---

## 4. Capacity Control on Exchange — Failure Case Analysis (Section A3 → Capacity Control)

### Table: Capacity-Matched Models on Exchange H=192 (3-run mean)

| Model | Params | MSE | MAE |
|---|---|---|---|
| PatchTST_base | 1,629,504 | **0.1904** | **0.3164** |
| PatchFusionBERT v2 | 2,456,000 | 0.1958 | 0.3178 |
| PatchTST_cap (width-matched) | 3,283,456 | 0.2182 | 0.3400 |
| PatchFusionBERT v0 | 3,256,768 | 0.2222 | 0.3377 |

**Interpretation for manuscript:**
Exchange exhibits high inter-feature collinearity and low temporal variability (see `c1_dataset_characteristics.csv`). At matched parameter count, both PFBv0 and PatchTSTcap underperform the lighter PatchTST_base. This rules out parameter count as the explanation for the performance gap — the failure is architectural: the refinement+fusion pipeline adds noise in low-diversity, high-redundancy regimes.

**Key sentence for Capacity Control section:**
> "On Exchange — the primary failure-case dataset — capacity-matched controls confirm that the performance gap is not an artefact of increased model capacity. PatchTST_base (1.6M parameters) achieves MSE=0.190, while both the capacity-matched PatchTST_cap (3.3M, MSE=0.218) and PatchFusionBERT v0 (3.3M, MSE=0.222) perform worse. This indicates that adding depth and fusion degrades performance in low-diversity, high-collinearity regimes regardless of parameter budget."

**Source:** `results_analysis/capmatch_controls_results.csv`

---

## 5. Robustness on Exchange and Illness (Section A5 → Robustness)

### Exchange H=192 — Random Zero-Masking

| Model | Clean MSE | 10% mask | 20% mask | 30% mask | Δ at 30% |
|---|---|---|---|---|---|
| DLinear | 0.187 | 0.249 (+33%) | 0.361 (+93%) | 0.521 (+179%) | +179% |
| PatchTST | 0.191 | 0.624 (+226%) | 1.389 (+626%) | **2.264 (+1084%)** | **+1084%** |
| PFBv0 | 0.218 | 0.348 (+59%) | 0.555 (+154%) | 0.810 (+271%) | +271% |
| PFBv2 | 0.188 | 0.354 (+89%) | 0.561 (+199%) | 0.854 (+355%) | +355% |

**Key finding:** PatchTST collapses catastrophically on Exchange under corruption (×11 MSE at 30% masking). PFBv0 is notably more robust than PatchTST on this dataset despite having worse clean performance. This is consistent with the refinement stack providing redundant signal paths that partially compensate for corrupted patches.

**Key sentence for manuscript (Robustness / Discussion):**
> "On Exchange — where PatchFusionBERT v0 has lower clean performance than PatchTST — the robustness profile reverses: PatchTST shows catastrophic degradation at 30% random masking (MSE ×11.8), while PatchFusionBERT v0 degrades more gracefully (MSE ×3.7). This dissociation between clean accuracy and robustness suggests the refinement mechanism introduces redundant patch-level representations that confer corruption resistance even in regimes where the fusion itself is net-negative for clean forecasting."

**Note:** Illness robustness available in `b2_robustness_exchange_illness_summary.csv`. Include if needed for section robustness.

**Source:** `results_analysis/b2_robustness_exchange_illness_summary.csv`

---

## 6. Patch Sensitivity (Section A9 → Ablation / Sensitivity)

PatchTST bug fixed (hardcoded `patch_len` in `models/PatchTST.py.__init__`). Now shows genuine variation:

### ETTm1 H=192

| Patch len | PFBv0 MSE | PatchTST MSE |
|---|---|---|
| 8 | 0.3402 | 0.3327 |
| 16 | 0.3361 | 0.3327 |
| **32** | **0.3293** | **0.3280** |
| 64 | 0.3331 | 0.3392 |

Both models peak at patch=32. PatchTST is more sensitive to patch length (Δ=0.0112 from best to worst vs PFBv0 Δ=0.0109).

**Manuscript note:** The previous version that showed PatchTST as insensitive (identical MSE=0.3327 across all patch lengths) was a code bug — the `patch_len` argument was ignored. After correction, both models show similar sensitivity profiles.

**Source:** `results_analysis/patch_sensitivity_results.csv`

---

## 7. FLOP-Matched Capacity Control — Limitation Statement (Section B7 / B10)

FLOPs are measured but a FLOP-matched training experiment was not run. Use the following text in the limitations section:

> "Although we measured FLOPs for all evaluated models (PFBv0: 1.092G MACs, PatchTST_cap: 1.560G MACs at equal parameter count), we did not train a FLOP-matched PatchTST variant. A compute-budget-controlled comparison remains an open experimental gap. The measured FLOP gap (PFBv0 uses ~30% fewer MACs than PatchTST_cap) is directionally favourable for PatchFusionBERT, but this does not substitute for a controlled FLOP-matched experiment."

**Source:** `results_analysis/flops_macs.csv`

---

## 8. Summary of Corrected Checklist Status (as of 2025-04-27)

| Priority A Item | Experiment Status | Manuscript Integration |
|---|---|---|
| Patch sensitivity fix (C.A.1) | ✅ Done | ❌ Needs integration |
| PatchTSTcap efficiency (C.A.2) | ✅ Done | ❌ Needs table update |
| Exchange capacity control (C.A.3) | ✅ Done | ❌ Needs paragraph |
| Error-by-step curves (C.A.4) | ✅ Done | ❌ Needs figure + text |
| FDR/Holm correction (C.A.5) | ✅ Done | ❌ Needs wording update |
| Robustness Exchange+Illness (C.B.1) | ✅ Done | ❌ Needs section |
| FLOPs/MACs (C.B.5) | ✅ Done | ❌ Needs table + limitation |

**All remaining gaps are manuscript writing tasks. No further GPU experiments are required for Priority A+B coverage.**
