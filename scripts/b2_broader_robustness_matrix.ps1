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
    & $python .\b2_broader_robustness_matrix.py --device $device
    & $python .\analyze_b2_broader_robustness.py
}
finally {
    Pop-Location
}