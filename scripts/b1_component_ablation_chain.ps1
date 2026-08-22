param(
    [int[]]$Seeds = @(2021),
    [string[]]$Datasets = @('ETTm2', 'ETTh2', 'Weather'),
    [int]$Horizon = 192,
    [int]$RefinementDepth = 3,
    [string]$ResultsFile = 'result_long_term_forecast.txt',
    [switch]$SkipCompleted,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
Push-Location $repoRoot

try {
    $env:USE_TF = '0'
    $env:TRANSFORMERS_NO_TF = '1'
    $env:TRANSFORMERS_NO_FLAX = '1'
    $env:TRANSFORMERS_NO_JAX = '1'

    $python = 'python'
    $fallbackPython = 'C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe'

    & $python -c "import sys, torch; print('exe:', sys.executable); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) {
        $python = $fallbackPython
        & $python -c "import sys, torch; print('exe:', sys.executable); print('torch:', torch.__version__); print('cuda_available:', torch.cuda.is_available())"
        if ($LASTEXITCODE -ne 0) {
            throw 'No usable torch environment found.'
        }
    }

    $datasetConfigs = @{
        ETTm2 = @{ data = 'ETTm2'; dataPath = 'ETTm2.csv'; encIn = 7; batchSize = 16; labelLen = 96 }
        ETTh2 = @{ data = 'ETTh2'; dataPath = 'ETTh2.csv'; encIn = 7; batchSize = 16; labelLen = 96 }
        Weather = @{ data = 'custom'; dataPath = 'weather.csv'; encIn = 21; batchSize = 8; labelLen = 96 }
    }

    $components = @(
        [pscustomobject]@{ Name = 'PatchOnly'; Model = 'PFB-PatchOnly'; ExtraArgs = @() },
        [pscustomobject]@{ Name = 'SerialRefinement'; Model = 'PFB-SerialRefinement'; ExtraArgs = @('--pfb_k', [string]$RefinementDepth) },
        [pscustomobject]@{ Name = 'FusionOnly'; Model = 'PFB-Direct'; ExtraArgs = @('--pfb_k', '0') },
        [pscustomobject]@{ Name = 'Full'; Model = 'PFB-Direct'; ExtraArgs = @('--pfb_k', [string]$RefinementDepth) }
    )

    $runs = New-Object System.Collections.Generic.List[object]
    foreach ($seed in $Seeds) {
        foreach ($dataset in $Datasets) {
            if (-not $datasetConfigs.ContainsKey($dataset)) {
                throw "Unsupported dataset '$dataset'."
            }

            $cfg = $datasetConfigs[$dataset]
            foreach ($component in $components) {
                $modelId = "B1Chain_MS${seed}_${dataset}_${Horizon}_$($component.Name)"
                $args = @(
                    '-u',
                    'run.py',
                    '--task_name', 'long_term_forecast',
                    '--is_training', '1',
                    '--model_id', $modelId,
                    '--model', $component.Model,
                    '--data', $cfg.data,
                    '--root_path', './data/',
                    '--data_path', $cfg.dataPath,
                    '--features', 'M',
                    '--seq_len', '336',
                    '--label_len', [string]$cfg.labelLen,
                    '--pred_len', [string]$Horizon,
                    '--enc_in', [string]$cfg.encIn,
                    '--dec_in', [string]$cfg.encIn,
                    '--c_out', [string]$cfg.encIn,
                    '--patch_len', '16',
                    '--stride', '8',
                    '--d_model', '128',
                    '--n_heads', '8',
                    '--e_layers', '3',
                    '--d_layers', '1',
                    '--d_ff', '512',
                    '--factor', '3',
                    '--dropout', '0.1',
                    '--itr', '1',
                    '--train_epochs', '100',
                    '--patience', '10',
                    '--batch_size', [string]$cfg.batchSize,
                    '--learning_rate', '0.0001',
                    '--num_workers', '0',
                    '--seed', [string]$seed,
                    '--des', 'b1_component_chain'
                ) + $component.ExtraArgs

                $runs.Add([pscustomobject]@{
                    Seed = $seed
                    Dataset = $dataset
                    Horizon = $Horizon
                    Component = $component.Name
                    Model = $component.Model
                    ModelId = $modelId
                    Args = $args
                })
            }
        }
    }

    if ($SkipCompleted) {
        if (-not (Test-Path $ResultsFile)) {
            throw "Results file '$ResultsFile' was not found."
        }
        $resultsText = Get-Content $ResultsFile -Raw
        $remaining = New-Object System.Collections.Generic.List[object]
        foreach ($run in $runs) {
            if ($resultsText -notmatch [regex]::Escape($run.ModelId)) {
                $remaining.Add($run)
            }
        }
        $runs = $remaining
    }

    Write-Host '========================================================================='
    Write-Host 'B1 COMPONENT ABLATION CHAIN'
    Write-Host '========================================================================='
    Write-Host ("Seeds: {0}" -f (($Seeds | Sort-Object) -join ', '))
    Write-Host ("Datasets: {0}" -f ($Datasets -join ', '))
    Write-Host ("Horizon: {0}" -f $Horizon)
    Write-Host ("Refinement depth: {0}" -f $RefinementDepth)
    Write-Host ("Runs queued: {0}" -f $runs.Count)
    if ($DryRun) {
        Write-Host 'Mode: DRY RUN'
    }
    Write-Host '========================================================================='

    if ($runs.Count -eq 0) {
        Write-Host 'Nothing to run.'
        return
    }

    $index = 0
    foreach ($run in $runs) {
        $index += 1
        Write-Host ("[{0}/{1}] {2} | {3} | seed={4}" -f $index, $runs.Count, $run.Dataset, $run.Component, $run.Seed)
        if ($DryRun) {
            Write-Host ((@($python) + $run.Args) -join ' ')
            continue
        }

        & $python @($run.Args)
        if ($LASTEXITCODE -ne 0) {
            throw "Run failed for $($run.ModelId)"
        }
    }

    Write-Host 'Done. Next: python analyze_b1_component_ablation.py'
}
finally {
    Pop-Location
}