# Full GPU pipeline - runs all remaining experiment jobs in sequence.
# Start this once capmatch_controls_exchange.ps1 finishes.
#
# Job order:
#   1. b2_robustness_exchange_illness.ps1       (~30-60 min, inference-only)
#   2. rerun_patchtst_patch_sensitivity.ps1     (~2-3h, 8 training runs)
#   3. benchmark_training_weather192.py         (~2-3h, PatchTSTcap efficiency)
#   4. pfb_v2_kdepth_ablation.ps1              (~10-15h, 18 training runs)
#
# Run from Time-Series-Library\ as:
#   .\run_remaining_pipeline.ps1
#
# Jobs 3+4 use start/end timestamps so you can see how long each took.

$ErrorActionPreference = "Stop"
$python = "C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe"
$scriptRoot = "d:\Hussein-experiments\dynamic-ensemble-paper\Ts-library\Time-Series-Library"
Set-Location $scriptRoot

function Write-Banner {
    param([string]$Title)
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $Title  [$ts]" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

# -------------------------------------------------------------------
Write-Banner "JOB 1 - Robustness Exchange + Illness (inference only)"
# -------------------------------------------------------------------
.\b2_robustness_exchange_illness.ps1
if ($LASTEXITCODE -ne 0) { throw "Job 1 failed: b2_robustness_exchange_illness.ps1" }
Write-Host "Job 1 DONE."

# -------------------------------------------------------------------
Write-Banner "JOB 2 - PatchTST patch sensitivity rerun (8 training runs)"
# -------------------------------------------------------------------
.\rerun_patchtst_patch_sensitivity.ps1
if ($LASTEXITCODE -ne 0) { throw "Job 2 failed: rerun_patchtst_patch_sensitivity.ps1" }
# Regenerate analysis CSV
& $python analyze_patch_sensitivity.py
if ($LASTEXITCODE -ne 0) { throw "Job 2 post-analysis failed: analyze_patch_sensitivity.py" }
Write-Host "Job 2 DONE."

# -------------------------------------------------------------------
Write-Banner "JOB 3 - PatchTSTcap efficiency benchmark (training timing)"
# -------------------------------------------------------------------
& $python benchmark_training_weather192.py
if ($LASTEXITCODE -ne 0) { throw "Job 3 failed: benchmark_training_weather192.py" }
& $python benchmark_inference_weather192.py
if ($LASTEXITCODE -ne 0) { throw "Job 3 failed: benchmark_inference_weather192.py" }
& $python analyze_efficiency_weather192.py
if ($LASTEXITCODE -ne 0) { throw "Job 3 failed: analyze_efficiency_weather192.py" }
Write-Host "Job 3 DONE."

# -------------------------------------------------------------------
Write-Banner "JOB 4 - PFBv2 K-depth ablation (18 training runs)"
# -------------------------------------------------------------------
.\pfb_v2_kdepth_ablation.ps1
if ($LASTEXITCODE -ne 0) { throw "Job 4 failed: pfb_v2_kdepth_ablation.ps1" }
& $python analyze_pfb_kdepth_ablation.py
if ($LASTEXITCODE -ne 0) { throw "Job 4 failed: analyze_pfb_kdepth_ablation.py" }
Write-Host "Job 4 DONE."

# -------------------------------------------------------------------
Write-Banner "ALL JOBS COMPLETE"
# -------------------------------------------------------------------
Write-Host ""
Write-Host "Post-pipeline: run analyze_capmatch_controls.py once Exchange capmatch is verified."
Write-Host "Then run statistical_significance_multiseed_5seed.py to regenerate significance tables."
Write-Host ""
Get-Date -Format "yyyy-MM-dd HH:mm:ss"
