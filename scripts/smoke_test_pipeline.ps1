param(
    [switch]$SkipTraining,
    [switch]$UseCpu,
    [switch]$IncludeAnalysis
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$smokeDir = Join-Path $root 'results\smoke'
New-Item -ItemType Directory -Force -Path $smokeDir | Out-Null

$summaryPath = Join-Path $smokeDir 'smoke_test_summary.txt'
$logPath = Join-Path $smokeDir 'smoke_test_pipeline.log'

$results = New-Object System.Collections.Generic.List[object]

function Add-Result {
    param(
        [string]$Step,
        [string]$Status,
        [string]$Detail
    )

    $results.Add([pscustomobject]@{
        Step = $Step
        Status = $Status
        Detail = $Detail
    }) | Out-Null
}

function Test-PythonCandidate {
    param([string]$PythonExe)

    if (-not (Test-Path $PythonExe)) {
        return $false
    }

    try {
        & $PythonExe -c "import torch, pandas, numpy, sklearn, scipy, einops; print(torch.__version__)" *> $null
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    Write-Host "[SMOKE] $Name"
    try {
        $detail = & $Action
        if ([string]::IsNullOrWhiteSpace($detail)) {
            $detail = 'ok'
        }
        Add-Result -Step $Name -Status 'PASS' -Detail $detail
    }
    catch {
        $detail = $_.Exception.Message
        Add-Result -Step $Name -Status 'FAIL' -Detail $detail
    }
}

$pythonCandidates = @(
    'C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe',
    (Join-Path $root '..\.venv\Scripts\python.exe'),
    (Join-Path $root '.venv\Scripts\python.exe')
)

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
    if (Test-PythonCandidate -PythonExe $candidate) {
        $pythonExe = $candidate
        break
    }
}

Invoke-Step -Name 'Python environment' -Action {
    if (-not $pythonExe) {
        throw 'No Python interpreter with torch+pandas+numpy+sklearn+scipy+einops was found.'
    }
    $version = & $pythonExe -c "import sys, torch; print(sys.version.split()[0] + ' | torch=' + torch.__version__)"
    "using $pythonExe ($version)"
}

Invoke-Step -Name 'Package layout' -Action {
    $requiredPaths = @(
        'run.py',
        'scripts\run_main_benchmark.py',
        'analysis\analyze_multiseed_results.py',
        'analysis\analyze_capmatch_controls.py',
        'analysis\analyze_illness_multiseed_stability.py',
        'results\result_long_term_forecast.txt',
        'data\ETTm1.csv',
        'data\ETTm2.csv',
        'data\ETTh1.csv',
        'data\ETTh2.csv',
        'data\exchange_rate.csv',
        'data\weather.csv',
        'data\national_illness.csv'
    )
    $missing = @()
    foreach ($relPath in $requiredPaths) {
        $fullPath = Join-Path $root $relPath
        if (-not (Test-Path $fullPath)) {
            $missing += $relPath
        }
    }
    if ($missing.Count -gt 0) {
        throw ('Missing required paths: ' + ($missing -join ', '))
    }
    'required code, results, and bundled datasets are present'
}

Invoke-Step -Name 'Core imports' -Action {
    & $pythonExe -c "from exp.exp_basic import Exp_Basic; from models import PatchFusionBERT_v0, PatchFusionBERT_v2, PatchTST, DLinear; print('imports_ok')"
    if ($LASTEXITCODE -ne 0) {
        throw 'Core import check failed.'
    }
    'model and experiment imports succeeded'
}

Invoke-Step -Name 'Benchmark launcher dry-run' -Action {
    & $pythonExe .\scripts\run_main_benchmark.py --phase h96 --models DLinear,PatchFusionBERT_v0 --datasets ETTm1,Illness --dry-run *> $logPath
    if ($LASTEXITCODE -ne 0) {
        throw 'run_main_benchmark.py dry-run failed.'
    }
    'launcher generated commands for standard and illness settings'
}

if (-not $SkipTraining) {
    $gpuFlag = @()
    if ($UseCpu) {
        $gpuFlag = @('--gpu_type', 'cpu')
    }

    $trainRuns = @(
        @{
            Name = 'Tiny training: DLinear ETTm1';
            ModelId = 'SmokePipeline_DLinear_ETTm1';
            Args = @(
                '--task_name', 'long_term_forecast',
                '--is_training', '1',
                '--model_id', 'SmokePipeline_DLinear_ETTm1',
                '--model', 'DLinear',
                '--data', 'ETTm1',
                '--root_path', '.\data',
                '--data_path', 'ETTm1.csv',
                '--features', 'M',
                '--freq', 't',
                '--seq_len', '336',
                '--label_len', '48',
                '--pred_len', '96',
                '--enc_in', '7',
                '--dec_in', '7',
                '--c_out', '7',
                '--d_model', '512',
                '--n_heads', '8',
                '--e_layers', '2',
                '--d_layers', '1',
                '--d_ff', '2048',
                '--factor', '3',
                '--dropout', '0.1',
                '--batch_size', '8',
                '--train_epochs', '1',
                '--patience', '1',
                '--itr', '1',
                '--num_workers', '0',
                '--learning_rate', '0.0001',
                '--seed', '2021',
                '--des', 'smoke_pipeline'
            ) + $gpuFlag
        },
        @{
            Name = 'Tiny training: PatchFusionBERT_v0 ETTm1';
            ModelId = 'SmokePipeline_PFBv0_ETTm1';
            Args = @(
                '--task_name', 'long_term_forecast',
                '--is_training', '1',
                '--model_id', 'SmokePipeline_PFBv0_ETTm1',
                '--model', 'PatchFusionBERT_v0',
                '--data', 'ETTm1',
                '--root_path', '.\data',
                '--data_path', 'ETTm1.csv',
                '--features', 'M',
                '--freq', 't',
                '--seq_len', '336',
                '--label_len', '48',
                '--pred_len', '96',
                '--enc_in', '7',
                '--dec_in', '7',
                '--c_out', '7',
                '--d_model', '128',
                '--n_heads', '8',
                '--e_layers', '3',
                '--d_layers', '1',
                '--d_ff', '512',
                '--factor', '3',
                '--dropout', '0.1',
                '--patch_len', '16',
                '--stride', '8',
                '--batch_size', '8',
                '--train_epochs', '1',
                '--patience', '1',
                '--itr', '1',
                '--num_workers', '0',
                '--learning_rate', '0.0001',
                '--seed', '2021',
                '--des', 'smoke_pipeline'
            ) + $gpuFlag
        }
    )

    foreach ($run in $trainRuns) {
        Invoke-Step -Name $run.Name -Action {
            $runLog = Join-Path $smokeDir ($run.ModelId + '.log')
            & $pythonExe .\run.py @($run.Args) *> $runLog
            if ($LASTEXITCODE -ne 0) {
                throw ($run.Name + ' failed. See ' + $runLog)
            }
            $resultFile = Join-Path $root 'results\result_long_term_forecast.txt'
            if (-not (Select-String -Path $resultFile -Pattern $run.ModelId -Quiet)) {
                throw ($run.ModelId + ' finished but did not append to results\result_long_term_forecast.txt')
            }
            'training+test completed and result log updated'
        }
    }
}

$analysisScripts = @(
    'analysis\analyze_multiseed_results.py',
    'analysis\analyze_capmatch_controls.py',
    'analysis\analyze_illness_multiseed_stability.py',
    'analysis\analyze_pfb_kdepth_ablation.py'
)

if ($IncludeAnalysis) {
    foreach ($script in $analysisScripts) {
        Invoke-Step -Name ("Analysis: $script") -Action {
            $analysisLog = Join-Path $smokeDir ((Split-Path $script -Leaf) + '.log')
            & $pythonExe $script *> $analysisLog
            if ($LASTEXITCODE -ne 0) {
                throw ('Analysis failed. See ' + $analysisLog)
            }
            'analysis script completed'
        }
    }
}
else {
    Add-Result -Step 'Analysis suite' -Status 'SKIP' -Detail 'Skipped by default; use -IncludeAnalysis for extended package validation.'
}

$failed = @($results | Where-Object { $_.Status -eq 'FAIL' })
$coreFailed = @(
    $results | Where-Object {
        $_.Status -eq 'FAIL' -and -not $_.Step.StartsWith('Analysis:')
    }
)

$summaryLines = @()
$summaryLines += 'PatchFusionBERT smoke test summary'
$summaryLines += ('Generated: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
$summaryLines += ''
$summaryLines += ($results | Format-Table -AutoSize | Out-String).TrimEnd()
$summaryLines += ''
$summaryLines += ('Core pipeline status: ' + ($(if ($coreFailed.Count -eq 0) { 'PASS' } else { 'FAIL' })))
$summaryLines += ('Extended status: ' + ($(if ($failed.Count -eq 0) { 'PASS' } else { 'FAIL' })))

$summaryLines -join [Environment]::NewLine | Set-Content -Path $summaryPath

Write-Host ''
Write-Host ($results | Format-Table -AutoSize | Out-String)
Write-Host "Summary written to $summaryPath"

if ($coreFailed.Count -gt 0) {
    exit 1
}