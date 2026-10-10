param(
    [switch]$Cpu
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py310 = 'C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe'
$workspacePython = Join-Path $root '..\.venv\Scripts\python.exe'

if (Test-Path $py310) {
    $python = $py310
} elseif (Test-Path $workspacePython) {
    $python = (Resolve-Path $workspacePython).Path
} else {
    $python = 'python'
}

$device = if ($Cpu) { 'cpu' } else { 'cuda' }

Push-Location $root
try {
    # Exchange: only H=192 has multiseed checkpoints
    # Illness:  H=24, H=48, H=60 have multiseed checkpoints (illness_horizons tag)
    # --skip_missing handles cross-product combos with no checkpoint gracefully
    & $python .\b2_broader_robustness_matrix.py `
        --device $device `
        --datasets Exchange Illness `
        --horizons 24 48 60 192 `
        --corruptions random block `
        --rates 0.1 0.2 0.3 `
        --skip_missing `
        --output results_analysis\b2_robustness_exchange_illness_raw.csv

    & $python .\analyze_b2_broader_robustness.py `
        --input results_analysis\b2_robustness_exchange_illness_raw.csv `
        --summary_md results_analysis\b2_robustness_exchange_illness_summary.md `
        --summary_csv results_analysis\b2_robustness_exchange_illness_summary.csv `
        --summary_tex results_analysis\b2_robustness_exchange_illness_summary.tex
}
finally {
    Pop-Location
}
