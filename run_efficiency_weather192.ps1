param(
    [switch]$SkipCompleted,
    [switch]$InferenceOnly,
    [switch]$AnalysisOnly
)

$ErrorActionPreference = 'Stop'

$python = 'python'
$fallbackPython = 'C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe'
$scriptRoot = $PSScriptRoot

Set-Location $scriptRoot

Write-Host "Checking Python environment..."
& $python -c "import sys, torch; print('exe:', sys.executable); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Falling back to: $fallbackPython"
    $python = $fallbackPython
    & $python -c "import sys, torch; print('exe:', sys.executable); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) {
        throw 'No usable torch environment found.'
    }
}

if (-not $InferenceOnly -and -not $AnalysisOnly) {
    $trainArgs = @('.\benchmark_training_weather192.py')
    if ($SkipCompleted) {
        $trainArgs += '--skip_completed'
    }
    & $python @trainArgs
    if ($LASTEXITCODE -ne 0) {
        throw 'Training efficiency benchmark failed.'
    }
}

if (-not $AnalysisOnly) {
    & $python .\benchmark_inference_weather192.py
    if ($LASTEXITCODE -ne 0) {
        throw 'Inference efficiency benchmark failed.'
    }
}

& $python .\analyze_efficiency_weather192.py
if ($LASTEXITCODE -ne 0) {
    throw 'Efficiency benchmark analysis failed.'
}

Write-Host 'Done. Outputs are under results_analysis/efficiency_weather192_*'