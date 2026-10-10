# multiseed_h96_h336_5seed.ps1
# Multi-seed validation at H=96 and H=336 (complementing the existing H=192 campaign)
# Runs: 5 seeds x 4 models x 6 datasets x 2 horizons = 240 runs
# Purpose: Answer reviewer challenge "are gains stable across the full evaluation grid?"

param(
    [int[]]$Seeds        = @(2021, 2022, 2023, 2024, 2025),
    [string[]]$Models    = @('PFB-Direct', 'PFB-Projected', 'PatchTST', 'DLinear'),
    [string[]]$Datasets  = @('ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather'),
    [int[]]$Horizons     = @(96, 336),
    [string]$ResultsFile = 'result_long_term_forecast.txt',
    [switch]$SkipCompleted,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repoRoot

try {
    $env:USE_TF               = '0'
    $env:TRANSFORMERS_NO_TF   = '1'
    $env:TRANSFORMERS_NO_FLAX = '1'
    $env:TRANSFORMERS_NO_JAX  = '1'

    # --- Dataset configs ------------------------------------------------------──
    $datasetConfigs = @{
        ETTm1    = @{ data='ETTm1';  dataPath='ETTm1.csv';          encIn=7;  batchSize=16 }
        ETTm2    = @{ data='ETTm2';  dataPath='ETTm2.csv';          encIn=7;  batchSize=16 }
        ETTh1    = @{ data='ETTh1';  dataPath='ETTh1.csv';          encIn=7;  batchSize=16 }
        ETTh2    = @{ data='ETTh2';  dataPath='ETTh2.csv';          encIn=7;  batchSize=16 }
        Exchange = @{ data='custom'; dataPath='exchange_rate.csv';  encIn=8;  batchSize=16 }
        Weather  = @{ data='custom'; dataPath='weather.csv';        encIn=21; batchSize=8  }
    }

    # --- Learning rate (matches unified multi-seed protocol) -----------------──
    function Get-LR([string]$model) {
        if ($model -eq 'DLinear') { '0.01' } else { '0.0001' }
    }

    # --- Build argument list for one run ------------------------------------
    function Get-RunArgs([string]$model, [string]$dataset, [int]$horizon, [int]$seed) {
        $cfg = $datasetConfigs[$dataset]
        $modelId = "${model}_MS${seed}_${dataset}_${horizon}"
        # Normalise short model name for model_id readability
        if ($model -eq 'PFB-Direct') { $modelId = "PFB-Direct_MS${seed}_${dataset}_${horizon}" }
        if ($model -eq 'PFB-Projected') { $modelId = "PFB-Projected_MS${seed}_${dataset}_${horizon}" }

        $common = @(
            '-u', 'run.py',
            '--task_name',     'long_term_forecast',
            '--is_training',   '1',
            '--root_path',     './data/',
            '--data_path',     $cfg.dataPath,
            '--data',          $cfg.data,
            '--features',      'M',
            '--seq_len',       '336',
            '--pred_len',      [string]$horizon,
            '--enc_in',        [string]$cfg.encIn,
            '--dec_in',        [string]$cfg.encIn,
            '--c_out',         [string]$cfg.encIn,
            '--num_workers',   '0',
            '--itr',           '1',
            '--train_epochs',  '100',
            '--patience',      '10',
            '--batch_size',    [string]$cfg.batchSize,
            '--learning_rate', (Get-LR $model),
            '--seed',          [string]$seed,
            '--model_id',      $modelId,
            '--des',           'multiseed_h96_h336'
        )

        $modelArgs = switch ($model) {
            'PFB-Direct' {
                @('--model','PFB-Direct','--label_len','96',
                  '--e_layers','3','--d_layers','1','--factor','3',
                  '--d_model','128','--d_ff','512','--n_heads','8',
                  '--patch_len','16','--stride','8')
            }
            'PFB-Projected' {
                $fArgs = @('--model','PFB-Projected','--label_len','48',
                           '--e_layers','3','--d_layers','1',
                           '--d_model','128','--d_ff','512','--n_heads','8')
                if ($dataset -in @('ETTm1','ETTm2','ETTh1','ETTh2')) {
                    $fArgs += @('--factor','3')
                }
                $fArgs
            }
            'PatchTST' {
                @('--model','PatchTST','--label_len','96',
                  '--e_layers','3','--d_layers','1','--factor','3',
                  '--d_model','128','--d_ff','512','--n_heads','8')
            }
            'DLinear' {
                @('--model','DLinear','--label_len','96',
                  '--e_layers','2','--d_layers','1','--factor','3',
                  '--d_model','512','--d_ff','2048','--n_heads','8')
            }
        }

        return $common + $modelArgs
    }

    # --- Build run list ----------------------------------------------------
    $runs = [System.Collections.Generic.List[object]]::new()
    foreach ($horizon in $Horizons) {
        foreach ($seed in $Seeds) {
            foreach ($dataset in $Datasets) {
                foreach ($model in $Models) {
                    $args = Get-RunArgs -model $model -dataset $dataset -horizon $horizon -seed $seed
                    $idIdx = [array]::IndexOf($args, '--model_id')
                    $modelId = $args[$idIdx + 1]
                    $runs.Add([pscustomobject]@{
                        Horizon  = $horizon
                        Seed     = $seed
                        Dataset  = $dataset
                        Model    = $model
                        ModelId  = $modelId
                        Args     = $args
                    })
                }
            }
        }
    }

    $total    = $runs.Count
    $done     = 0
    $skipped  = 0
    $failed   = 0

    Write-Host "=== Multiseed H=96/H=336 Campaign ===" -ForegroundColor Cyan
    Write-Host "Total planned runs: $total" -ForegroundColor Cyan
    if ($DryRun) { Write-Host '(DRY RUN - no training will execute)' -ForegroundColor Yellow }

    # ── Load completed IDs once ────────────────────────────────────────────────
    $completedIds = [System.Collections.Generic.HashSet[string]]::new()
    if ($SkipCompleted -and (Test-Path $ResultsFile)) {
        $content = Get-Content $ResultsFile -Raw
        $matches = [regex]::Matches($content, 'long_term_forecast_(\S+?)(?:\s+mse:|\s*$)')
        foreach ($m in $matches) { [void]$completedIds.Add($m.Groups[1].Value) }
        Write-Host "Loaded $($completedIds.Count) completed model IDs from $ResultsFile" -ForegroundColor Green
    }

    $i = 0
    foreach ($run in $runs) {
        $i++
        $prefix = "[$i/$total] H=$($run.Horizon) | $($run.Dataset) | $($run.Model) | seed=$($run.Seed)"

        if ($SkipCompleted -and $completedIds.Contains($run.ModelId)) {
            Write-Host "SKIP  $prefix (already done)" -ForegroundColor DarkGray
            $skipped++
            continue
        }

        Write-Host "RUN   $prefix" -ForegroundColor White

        if (-not $DryRun) {
            try {
                & python @($run.Args)
                if ($LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }
                $done++
                # Refresh completed IDs from file after each run
                if (Test-Path $ResultsFile) {
                    $content = Get-Content $ResultsFile -Raw
                    $matches = [regex]::Matches($content, 'long_term_forecast_(\S+?)(?:\s+mse:|\s*$)')
                    $completedIds.Clear()
                    foreach ($m in $matches) { [void]$completedIds.Add($m.Groups[1].Value) }
                }
            }
            catch {
                Write-Warning "FAILED $prefix : $_"
                $failed++
            }
        }
        else {
            $done++
        }
    }

    Write-Host "`n=== Campaign complete ===" -ForegroundColor Cyan
    $color = if ($failed -gt 0) { 'Yellow' } else { 'Green' }
    Write-Host "Ran: $done  |  Skipped: $skipped  |  Failed: $failed" -ForegroundColor $color

}
finally {
    Pop-Location
}
