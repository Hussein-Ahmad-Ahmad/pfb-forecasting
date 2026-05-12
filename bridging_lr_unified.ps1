# bridging_lr_unified.ps1
# Bridging LR experiment: rerun the full main benchmark at unified LR=1e-4
# Purpose: Resolve Prof. Kyamakya Major Revision 3 ??? harmonize LR protocol
#   Original benchmark: PFBv0/PFBv2 at LR=1e-3, PatchTST at LR=1e-3, DLinear at LR=1e-2
#   Unified protocol:   all attention models at LR=1e-4, DLinear at LR=1e-2
# Scope: 4 models x 6 standard datasets x 4 horizons + 4 models x Illness x 3 horizons = 108 runs
# Single seed (2021) ??? statistical comparison against existing multi-seed H=192 results

param(
    [int]$Seed               = 2021,
    [string[]]$Models        = @('PatchFusionBERT_v0', 'PatchFusionBERT_v2', 'PatchTST', 'DLinear'),
    [string[]]$StandardDatasets = @('ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather'),
    [int[]]$StandardHorizons = @(96, 192, 336, 720),
    [int[]]$IllnessHorizons  = @(24, 48, 60),
    [switch]$SkipStandard,
    [switch]$SkipIllness,
    [string]$ResultsFile     = 'result_long_term_forecast.txt',
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

    # ?????? Dataset configs ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    $datasetConfigs = @{
        ETTm1    = @{ data='ETTm1';  dataPath='ETTm1.csv';          encIn=7;  batchSize=16; type='standard' }
        ETTm2    = @{ data='ETTm2';  dataPath='ETTm2.csv';          encIn=7;  batchSize=16; type='standard' }
        ETTh1    = @{ data='ETTh1';  dataPath='ETTh1.csv';          encIn=7;  batchSize=16; type='standard' }
        ETTh2    = @{ data='ETTh2';  dataPath='ETTh2.csv';          encIn=7;  batchSize=16; type='standard' }
        Exchange = @{ data='custom'; dataPath='exchange_rate.csv';  encIn=8;  batchSize=16; type='standard' }
        Weather  = @{ data='custom'; dataPath='weather.csv';        encIn=21; batchSize=8;  type='standard' }
        Illness  = @{ data='custom'; dataPath='national_illness.csv'; encIn=7; batchSize=16; type='illness' }
    }

    # ?????? Unified LR: all attention models at 1e-4; DLinear keeps its natural rate
    function Get-LR([string]$model) {
        if ($model -eq 'DLinear') { '0.01' } else { '0.0001' }
    }

    # ?????? Build argument list ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    function Get-RunArgs([string]$model, [string]$dataset, [int]$horizon, [int]$seed) {
        $cfg      = $datasetConfigs[$dataset]
        $isIllness = ($cfg.type -eq 'illness')

        # Stable model_id prefix for easy extraction from results file
        $shortModel = switch ($model) {
            'PatchFusionBERT_v0' { 'PFBv0' }
            'PatchFusionBERT_v2' { 'PFBv2' }
            default              { $model  }
        }
        $modelId = "BridgeLR_${shortModel}_${dataset}_${horizon}"

        $seqLen   = if ($isIllness) { '104' } else { '336' }
        $labelLen = if ($isIllness) { '18'  } else { '96'  }   # overridden per model below

        $common = @(
            '-u', 'run.py',
            '--task_name',     'long_term_forecast',
            '--is_training',   '1',
            '--root_path',     './data/',
            '--data_path',     $cfg.dataPath,
            '--data',          $cfg.data,
            '--features',      'M',
            '--seq_len',       $seqLen,
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
            '--des',           'bridging_lr'
        )

        $modelArgs = switch ($model) {
            'PatchFusionBERT_v0' {
                $ll = if ($isIllness) { '18' } else { '96' }
                @('--model','PatchFusionBERT_v0','--label_len',$ll,
                  '--e_layers','3','--d_layers','1','--factor','3',
                  '--d_model','128','--d_ff','512','--n_heads','8',
                  '--patch_len','16','--stride','8')
            }
            'PatchFusionBERT_v2' {
                $ll = if ($isIllness) { '18' } else { '48' }
                $fArgs = @('--model','PatchFusionBERT_v2','--label_len',$ll,
                           '--e_layers','3','--d_layers','1',
                           '--d_model','128','--d_ff','512','--n_heads','8')
                if ($dataset -in @('ETTm1','ETTm2','ETTh1','ETTh2','Illness')) {
                    $fArgs += @('--factor','3')
                }
                $fArgs
            }
            'PatchTST' {
                $ll = if ($isIllness) { '18' } else { '96' }
                @('--model','PatchTST','--label_len',$ll,
                  '--e_layers','3','--d_layers','1','--factor','3',
                  '--d_model','128','--d_ff','512','--n_heads','8')
            }
            'DLinear' {
                $ll = if ($isIllness) { '18' } else { '96' }
                @('--model','DLinear','--label_len',$ll,
                  '--e_layers','2','--d_layers','1','--factor','3',
                  '--d_model','512','--d_ff','2048','--n_heads','8')
            }
        }

        return $common + $modelArgs
    }

    # ?????? Build run list ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    $runs = [System.Collections.Generic.List[object]]::new()

    if (-not $SkipStandard) {
        foreach ($horizon in $StandardHorizons) {
            foreach ($dataset in $StandardDatasets) {
                foreach ($model in $Models) {
                    $args    = Get-RunArgs -model $model -dataset $dataset -horizon $horizon -seed $Seed
                    $idIdx   = [array]::IndexOf($args, '--model_id')
                    $modelId = $args[$idIdx + 1]
                    $runs.Add([pscustomobject]@{
                        Type    = 'Standard'
                        Horizon = $horizon; Dataset = $dataset
                        Model   = $model;   ModelId = $modelId; Args = $args
                    })
                }
            }
        }
    }

    if (-not $SkipIllness) {
        foreach ($horizon in $IllnessHorizons) {
            foreach ($model in $Models) {
                $args    = Get-RunArgs -model $model -dataset 'Illness' -horizon $horizon -seed $Seed
                $idIdx   = [array]::IndexOf($args, '--model_id')
                $modelId = $args[$idIdx + 1]
                $runs.Add([pscustomobject]@{
                    Type    = 'Illness'
                    Horizon = $horizon; Dataset = 'Illness'
                    Model   = $model;   ModelId = $modelId; Args = $args
                })
            }
        }
    }

    $total   = $runs.Count
    $done    = 0
    $skipped = 0
    $failed  = 0

    Write-Host "=== Bridging LR Unified Experiment ===" -ForegroundColor Cyan
    Write-Host "Seed: $Seed  |  LR: attention=1e-4, DLinear=1e-2  |  Total runs: $total" -ForegroundColor Cyan
    if ($DryRun) { Write-Host "(DRY RUN ??? no training will execute)" -ForegroundColor Yellow }

    # ?????? Load completed IDs ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
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
        $prefix = "[$i/$total] H=$($run.Horizon) | $($run.Dataset) | $($run.Model)"

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

    Write-Host "`n=== Bridging LR campaign complete ===" -ForegroundColor Cyan
    Write-Host "Ran: $done  |  Skipped: $skipped  |  Failed: $failed" -ForegroundColor $(if ($failed -gt 0) { 'Yellow' } else { 'Green' })
    Write-Host "`nExtract results with:" -ForegroundColor Cyan
    Write-Host "  Select-String 'BridgeLR_' result_long_term_forecast.txt" -ForegroundColor Gray

}
finally {
    Pop-Location
}
