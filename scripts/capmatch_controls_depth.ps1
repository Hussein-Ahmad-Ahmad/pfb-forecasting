param(
    [int[]]$Seeds = @(2021, 2022, 2023),
    [string[]]$Models = @('PatchTST_base', 'PatchTST_depth', 'PFB-Direct', 'PFB-Projected'),
    [string]$ResultsFile = 'result_long_term_forecast.txt',
    [switch]$SkipCompleted,
    [switch]$DryRun
)

# Depth-only PatchTST controls vs PFB.
# Keeps PatchTST width fixed at the base setting and only changes e_layers.

$ErrorActionPreference = 'Stop'

$python = 'python'
$fallbackPython = 'C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe'

$scriptRoot = $PSScriptRoot
$repoRoot = Split-Path -Parent $scriptRoot
$runPy = Join-Path $repoRoot 'run.py'

Set-Location $repoRoot

$env:USE_TF = '0'
$env:TRANSFORMERS_NO_TF = '1'
$env:TRANSFORMERS_NO_FLAX = '1'
$env:TRANSFORMERS_NO_JAX = '1'

Write-Host "Using python (initial): $python"
& $python -c "import sys, torch; print('exe:', sys.executable); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python check failed; falling back to: $fallbackPython"
    $python = $fallbackPython
    & $python -c "import sys, torch; print('exe:', sys.executable); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) {
        throw 'Python check failed (is torch installed in this env?)'
    }
}

$task = 'long_term_forecast'
$itr = 1
$train_epochs = 100
$patience = 10
$dropout = 0.1
$n_heads = 8
$factor = 3
$patch_len = 16
$stride = 8
$base_d_model = 128
$base_e_layers = 3
$base_d_ff = 512

$depthConfigs = @{
    'ETTh2_96'  = @{ e_layers = 11 }
    'ETTh2_192' = @{ e_layers = 11 }
    'ETTh2_336' = @{ e_layers = 11 }
    'ETTm2_96'  = @{ e_layers = 11 }
    'ETTm2_192' = @{ e_layers = 11 }
    'ETTm2_336' = @{ e_layers = 11 }
    'Weather_96'  = @{ e_layers = 11 }
    'Weather_192' = @{ e_layers = 11 }
    'Weather_336' = @{ e_layers = 11 }
    'Illness_24' = @{ e_layers = 6 }
    'Illness_48' = @{ e_layers = 6 }
    'Illness_60' = @{ e_layers = 7 }
}

function Get-LearningRate {
    param(
        [string]$Model
    )

    switch ($Model) {
        'DLinear' { return 0.01 }
        default { return 0.0001 }
    }
}

function New-RunSpec {
    param(
        [string]$DatasetTag,
        [string]$Data,
        [string]$DataPath,
        [int]$SeqLen,
        [int]$LabelLen,
        [int]$PredLen,
        [int]$EncIn,
        [int]$BatchSize,
        [int]$Seed
    )

    return [pscustomobject]@{
        DatasetTag = $DatasetTag
        Data = $Data
        DataPath = $DataPath
        SeqLen = $SeqLen
        LabelLen = $LabelLen
        PredLen = $PredLen
        EncIn = $EncIn
        BatchSize = $BatchSize
        Seed = $Seed
    }
}

function Get-ModelArgs {
    param(
        [pscustomobject]$RunSpec,
        [string]$Model
    )

    $learning_rate = Get-LearningRate -Model $Model

    $commonArgs = @(
        '-u',
        $runPy,
        '--task_name', $task,
        '--is_training', '1',
        '--root_path', './data/',
        '--data_path', $RunSpec.DataPath,
        '--data', $RunSpec.Data,
        '--features', 'M',
        '--seq_len', [string]$RunSpec.SeqLen,
        '--label_len', [string]$RunSpec.LabelLen,
        '--pred_len', [string]$RunSpec.PredLen,
        '--enc_in', [string]$RunSpec.EncIn,
        '--dec_in', [string]$RunSpec.EncIn,
        '--c_out', [string]$RunSpec.EncIn,
        '--patch_len', [string]$patch_len,
        '--stride', [string]$stride,
        '--n_heads', [string]$n_heads,
        '--d_layers', '1',
        '--factor', [string]$factor,
        '--dropout', [string]$dropout,
        '--itr', [string]$itr,
        '--train_epochs', [string]$train_epochs,
        '--patience', [string]$patience,
        '--batch_size', [string]$RunSpec.BatchSize,
        '--learning_rate', [string]$learning_rate,
        '--num_workers', '0',
        '--seed', [string]$RunSpec.Seed,
        '--des', 'capmatch_depth'
    )

    $modelId = "CapMatchDepth_MS$($RunSpec.Seed)_$($RunSpec.DatasetTag)_$($RunSpec.PredLen)"

    switch ($Model) {
        'PatchTST_base' {
            return $commonArgs + @(
                '--model_id', $modelId,
                '--model', 'PatchTST_base',
                '--d_model', [string]$base_d_model,
                '--e_layers', [string]$base_e_layers,
                '--d_ff', [string]$base_d_ff
            )
        }
        'PatchTST_depth' {
            $cfgKey = "$($RunSpec.DatasetTag)_$($RunSpec.PredLen)"
            $depthCfg = $depthConfigs[$cfgKey]
            if ($null -eq $depthCfg) {
                throw "Missing depth config for $cfgKey"
            }

            return $commonArgs + @(
                '--model_id', $modelId,
                '--model', 'PatchTST_depth',
                '--d_model', [string]$base_d_model,
                '--e_layers', [string]$depthCfg.e_layers,
                '--d_ff', [string]$base_d_ff
            )
        }
        'PFB-Direct' {
            return $commonArgs + @(
                '--model_id', $modelId,
                '--model', 'PFB-Direct',
                '--d_model', [string]$base_d_model,
                '--e_layers', [string]$base_e_layers,
                '--d_ff', [string]$base_d_ff
            )
        }
        'PFB-Projected' {
            return $commonArgs + @(
                '--model_id', $modelId,
                '--model', 'PFB-Projected',
                '--d_model', [string]$base_d_model,
                '--e_layers', [string]$base_e_layers,
                '--d_ff', [string]$base_d_ff
            )
        }
        default {
            throw "Unsupported model '$Model'."
        }
    }
}

$models = $Models

$runSpecs = New-Object System.Collections.Generic.List[object]
foreach ($seed in $Seeds) {
    $runSpecs.Add((New-RunSpec -DatasetTag 'ETTh2' -Data 'ETTh2' -DataPath 'ETTh2.csv' -SeqLen 336 -LabelLen 96 -PredLen 96  -EncIn 7 -BatchSize 16 -Seed $seed))
    $runSpecs.Add((New-RunSpec -DatasetTag 'ETTh2' -Data 'ETTh2' -DataPath 'ETTh2.csv' -SeqLen 336 -LabelLen 96 -PredLen 192 -EncIn 7 -BatchSize 16 -Seed $seed))
    $runSpecs.Add((New-RunSpec -DatasetTag 'ETTh2' -Data 'ETTh2' -DataPath 'ETTh2.csv' -SeqLen 336 -LabelLen 96 -PredLen 336 -EncIn 7 -BatchSize 16 -Seed $seed))
    $runSpecs.Add((New-RunSpec -DatasetTag 'ETTm2' -Data 'ETTm2' -DataPath 'ETTm2.csv' -SeqLen 336 -LabelLen 96 -PredLen 96  -EncIn 7 -BatchSize 16 -Seed $seed))
    $runSpecs.Add((New-RunSpec -DatasetTag 'ETTm2' -Data 'ETTm2' -DataPath 'ETTm2.csv' -SeqLen 336 -LabelLen 96 -PredLen 192 -EncIn 7 -BatchSize 16 -Seed $seed))
    $runSpecs.Add((New-RunSpec -DatasetTag 'ETTm2' -Data 'ETTm2' -DataPath 'ETTm2.csv' -SeqLen 336 -LabelLen 96 -PredLen 336 -EncIn 7 -BatchSize 16 -Seed $seed))
    $runSpecs.Add((New-RunSpec -DatasetTag 'Weather' -Data 'custom' -DataPath 'weather.csv' -SeqLen 336 -LabelLen 96 -PredLen 96  -EncIn 21 -BatchSize 8 -Seed $seed))
    $runSpecs.Add((New-RunSpec -DatasetTag 'Weather' -Data 'custom' -DataPath 'weather.csv' -SeqLen 336 -LabelLen 96 -PredLen 192 -EncIn 21 -BatchSize 8 -Seed $seed))
    $runSpecs.Add((New-RunSpec -DatasetTag 'Weather' -Data 'custom' -DataPath 'weather.csv' -SeqLen 336 -LabelLen 96 -PredLen 336 -EncIn 21 -BatchSize 8 -Seed $seed))
    $runSpecs.Add((New-RunSpec -DatasetTag 'Illness' -Data 'custom' -DataPath 'national_illness.csv' -SeqLen 104 -LabelLen 18 -PredLen 24 -EncIn 7 -BatchSize 16 -Seed $seed))
    $runSpecs.Add((New-RunSpec -DatasetTag 'Illness' -Data 'custom' -DataPath 'national_illness.csv' -SeqLen 104 -LabelLen 18 -PredLen 48 -EncIn 7 -BatchSize 16 -Seed $seed))
    $runSpecs.Add((New-RunSpec -DatasetTag 'Illness' -Data 'custom' -DataPath 'national_illness.csv' -SeqLen 104 -LabelLen 18 -PredLen 60 -EncIn 7 -BatchSize 16 -Seed $seed))
}

$runs = New-Object System.Collections.Generic.List[object]
foreach ($runSpec in $runSpecs) {
    foreach ($model in $models) {
        $args = Get-ModelArgs -RunSpec $runSpec -Model $model
        $modelIdIndex = [array]::IndexOf($args, '--model_id')
        $modelIndex = [array]::IndexOf($args, '--model')

        $runs.Add([pscustomobject]@{
            Dataset = $runSpec.DatasetTag
            Horizon = $runSpec.PredLen
            Seed = $runSpec.Seed
            Model = $args[$modelIndex + 1]
            ModelId = $args[$modelIdIndex + 1]
            Args = $args
        })
    }
}

$initialRunCount = $runs.Count
if ($SkipCompleted) {
    if (-not (Test-Path $ResultsFile)) {
        throw "Results file '$ResultsFile' was not found."
    }

    $resultsText = Get-Content $ResultsFile -Raw
    $remainingRuns = New-Object System.Collections.Generic.List[object]
    foreach ($run in $runs) {
        $needle = "{0}_{1}_" -f $run.ModelId, $run.Model
        if ($resultsText -notmatch [regex]::Escape($needle)) {
            $remainingRuns.Add($run)
        }
    }
    $runs = $remainingRuns
}

Write-Host '========================================================================='
Write-Host 'DEPTH-ONLY CAPACITY CONTROL CAMPAIGN'
Write-Host '========================================================================='
Write-Host ("Seeds: {0}" -f (($Seeds | Sort-Object) -join ', '))
Write-Host ("Models: {0}" -f ($Models -join ', '))
Write-Host ("Total selected runs: {0}" -f $initialRunCount)
if ($SkipCompleted) {
    Write-Host ("Remaining after skip: {0}" -f $runs.Count)
}
else {
    Write-Host ("Total runs: {0}" -f $runs.Count)
}
if ($DryRun) {
    Write-Host 'Mode: DRY RUN'
}
Write-Host '========================================================================='

if ($runs.Count -eq 0) {
    Write-Host 'No remaining runs.'
    exit 0
}

foreach ($run in $runs) {
    Write-Host ''
    Write-Host ("[{0}] {1} | H={2} | seed={3}" -f $run.Model, $run.Dataset, $run.Horizon, $run.Seed)
    Write-Host ("python {0}" -f ($run.Args -join ' '))

    if ($DryRun) {
        continue
    }

    & $python @($run.Args)
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $($run.ModelId) / $($run.Model)"
    }
}

Write-Host ''
Write-Host 'Done. Next: python .\analyze_capmatch_controls.py'