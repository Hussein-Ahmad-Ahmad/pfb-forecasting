param(
    [int[]]$Seeds = @(2021, 2022, 2023, 2024, 2025),
    [string[]]$Models = @('PFB-Direct', 'PFB-Projected', 'PatchTST', 'DLinear'),
    [string[]]$StandardDatasets = @('ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather'),
    [int[]]$IllnessHorizons = @(24, 48, 60),
    [string]$ResultsFile = 'result_long_term_forecast.txt',
    [switch]$SkipCompleted,
    [switch]$SkipStandard,
    [switch]$SkipIllness,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $repoRoot

try {
    $env:USE_TF = '0'
    $env:TRANSFORMERS_NO_TF = '1'
    $env:TRANSFORMERS_NO_FLAX = '1'
    $env:TRANSFORMERS_NO_JAX = '1'

    $standardDatasetsOrdered = @('ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Exchange', 'Weather')
    $illnessDataset = 'Illness'

    $datasetConfigs = @{
        ETTm1 = @{ data = 'ETTm1'; dataPath = 'ETTm1.csv'; encIn = 7; batchSize = 16; type = 'standard' }
        ETTm2 = @{ data = 'ETTm2'; dataPath = 'ETTm2.csv'; encIn = 7; batchSize = 16; type = 'standard' }
        ETTh1 = @{ data = 'ETTh1'; dataPath = 'ETTh1.csv'; encIn = 7; batchSize = 16; type = 'standard' }
        ETTh2 = @{ data = 'ETTh2'; dataPath = 'ETTh2.csv'; encIn = 7; batchSize = 16; type = 'standard' }
        Exchange = @{ data = 'custom'; dataPath = 'exchange_rate.csv'; encIn = 8; batchSize = 16; type = 'standard' }
        Weather = @{ data = 'custom'; dataPath = 'weather.csv'; encIn = 21; batchSize = 8; type = 'standard' }
        Illness = @{ data = 'custom'; dataPath = 'national_illness.csv'; encIn = 7; batchSize = 16; type = 'illness' }
    }

    function Get-LearningRate {
        param(
            [string]$ModelName
        )

        switch ($ModelName) {
            'DLinear' { return '0.01' }
            default { return '0.0001' }
        }
    }

    function Get-ModelArguments {
        param(
            [string]$ModelName,
            [string]$DatasetName,
            [int]$Horizon,
            [int]$Seed
        )

        $datasetCfg = $datasetConfigs[$DatasetName]
        $commonArgs = @(
            '-u',
            'run.py',
            '--task_name', 'long_term_forecast',
            '--is_training', '1',
            '--root_path', './data/',
            '--data_path', $datasetCfg.dataPath,
            '--data', $datasetCfg.data,
            '--features', 'M',
            '--enc_in', [string]$datasetCfg.encIn,
            '--dec_in', [string]$datasetCfg.encIn,
            '--c_out', [string]$datasetCfg.encIn,
            '--num_workers', '0',
            '--itr', '1',
            '--train_epochs', '100',
            '--patience', '10',
            '--batch_size', [string]$datasetCfg.batchSize,
            '--learning_rate', (Get-LearningRate -ModelName $ModelName),
            '--seed', [string]$Seed
        )

        if ($DatasetName -eq 'Illness') {
            $baseArgs = @(
                '--seq_len', '104',
                '--label_len', '18',
                '--pred_len', [string]$Horizon
            )
        }
        else {
            $baseArgs = @(
                '--seq_len', '336',
                '--pred_len', [string]$Horizon
            )
        }

        switch ($ModelName) {
            'PFB-Direct' {
                $labelLen = if ($DatasetName -eq 'Illness') { '18' } else { '96' }
                $des = if ($DatasetName -eq 'Illness') { 'illness_horizons' } else { 'multiseed_h192' }
                $modelId = "PFB-Direct_MS${Seed}_${DatasetName}_${Horizon}"
                return $commonArgs + @(
                    '--model_id', $modelId,
                    '--model', 'PFB-Direct',
                    '--label_len', $labelLen,
                    '--e_layers', '3',
                    '--d_layers', '1',
                    '--factor', '3',
                    '--d_model', '128',
                    '--d_ff', '512',
                    '--n_heads', '8',
                    '--patch_len', '16',
                    '--stride', '8',
                    '--des', $des
                ) + $baseArgs
            }
            'PFB-Projected' {
                $labelLen = if ($DatasetName -eq 'Illness') { '18' } else { '48' }
                $des = 'Exp'
                $modelId = if ($DatasetName -eq 'Illness') {
                    "Illness_104_${Horizon}_MS${Seed}"
                }
                else {
                    "${DatasetName}_336_${Horizon}_MS${Seed}"
                }

                $v2Args = @(
                    '--model_id', $modelId,
                    '--model', 'PFB-Projected',
                    '--label_len', $labelLen,
                    '--e_layers', '3',
                    '--d_layers', '1',
                    '--d_model', '128',
                    '--d_ff', '512',
                    '--n_heads', '8',
                    '--des', $des
                )

                if ($DatasetName -in @('ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Illness')) {
                    $v2Args += @('--factor', '3')
                }

                return $commonArgs + $v2Args + $baseArgs
            }
            'PatchTST' {
                $labelLen = if ($DatasetName -eq 'Illness') { '18' } else { '96' }
                $des = if ($DatasetName -eq 'Illness') { 'illness_horizons' } else { 'multiseed_h192' }
                $modelId = "PatchTST_MS${Seed}_${DatasetName}_${Horizon}"
                return $commonArgs + @(
                    '--model_id', $modelId,
                    '--model', 'PatchTST',
                    '--label_len', $labelLen,
                    '--e_layers', '3',
                    '--d_layers', '1',
                    '--factor', '3',
                    '--d_model', '128',
                    '--d_ff', '512',
                    '--n_heads', '8',
                    '--des', $des
                ) + $baseArgs
            }
            'DLinear' {
                $labelLen = if ($DatasetName -eq 'Illness') { '18' } else { '96' }
                $des = if ($DatasetName -eq 'Illness') { 'illness_horizons' } else { 'multiseed_h192' }
                $modelId = "DLinear_MS${Seed}_${DatasetName}_${Horizon}"
                return $commonArgs + @(
                    '--model_id', $modelId,
                    '--model', 'DLinear',
                    '--label_len', $labelLen,
                    '--e_layers', '2',
                    '--d_layers', '1',
                    '--factor', '3',
                    '--d_model', '512',
                    '--d_ff', '2048',
                    '--n_heads', '8',
                    '--des', $des
                ) + $baseArgs
            }
            default {
                throw "Unsupported model '$ModelName'."
            }
        }
    }

    $requestedStandard = $standardDatasetsOrdered | Where-Object { $_ -in $StandardDatasets }
    $requestedIllnessHorizons = $IllnessHorizons | Sort-Object

    $runs = New-Object System.Collections.Generic.List[object]

    if (-not $SkipStandard) {
        foreach ($seed in $Seeds) {
            foreach ($dataset in $requestedStandard) {
                foreach ($model in $Models) {
                    $modelArgs = Get-ModelArguments -ModelName $model -DatasetName $dataset -Horizon 192 -Seed $seed
                    $modelIdIndex = [array]::IndexOf($modelArgs, '--model_id')
                    if ($modelIdIndex -lt 0 -or $modelIdIndex + 1 -ge $modelArgs.Count) {
                        throw "Unable to extract --model_id for $model / $dataset / $seed."
                    }

                    $runs.Add([pscustomobject]@{
                        Seed = $seed
                        Dataset = $dataset
                        Horizon = 192
                        Model = $model
                        ModelId = [string]$modelArgs[$modelIdIndex + 1]
                        Args = $modelArgs
                    })
                }
            }
        }
    }

    if (-not $SkipIllness) {
        foreach ($seed in $Seeds) {
            foreach ($horizon in $requestedIllnessHorizons) {
                foreach ($model in $Models) {
                    $modelArgs = Get-ModelArguments -ModelName $model -DatasetName $illnessDataset -Horizon $horizon -Seed $seed
                    $modelIdIndex = [array]::IndexOf($modelArgs, '--model_id')
                    if ($modelIdIndex -lt 0 -or $modelIdIndex + 1 -ge $modelArgs.Count) {
                        throw "Unable to extract --model_id for $model / $illnessDataset / $horizon / $seed."
                    }

                    $runs.Add([pscustomobject]@{
                        Seed = $seed
                        Dataset = $illnessDataset
                        Horizon = $horizon
                        Model = $model
                        ModelId = [string]$modelArgs[$modelIdIndex + 1]
                        Args = $modelArgs
                    })
                }
            }
        }
    }

    if ($runs.Count -eq 0) {
        throw 'No runs selected. Adjust the dataset, horizon, or skip parameters.'
    }

    $initialRunCount = $runs.Count
    if ($SkipCompleted) {
        if (-not (Test-Path $ResultsFile)) {
            throw "Results file '$ResultsFile' was not found."
        }

        $resultsText = Get-Content $ResultsFile -Raw
        $remainingRuns = New-Object System.Collections.Generic.List[object]
        foreach ($run in $runs) {
            if ($resultsText -notmatch [regex]::Escape($run.ModelId)) {
                $remainingRuns.Add($run)
            }
        }

        $runs = $remainingRuns
    }

    Write-Host '========================================================================='
    Write-Host '5-SEED MULTI-SEED CAMPAIGN'
    Write-Host '========================================================================='
    Write-Host ("Seeds: {0}" -f (($Seeds | Sort-Object) -join ', '))
    Write-Host ("Models: {0}" -f ($Models -join ', '))
    Write-Host ("Standard datasets: {0}" -f ($(if ($SkipStandard) { 'skipped' } else { $requestedStandard -join ', ' })))
    Write-Host ("Illness horizons: {0}" -f ($(if ($SkipIllness) { 'skipped' } else { ($requestedIllnessHorizons -join ', ') })))
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
        Write-Host 'No remaining runs to execute.'
        return
    }

    $index = 0
    foreach ($run in $runs) {
        $index += 1
        $summary = "[{0}/{1}] {2} | {3} | H={4} | seed={5}" -f $index, $runs.Count, $run.Model, $run.Dataset, $run.Horizon, $run.Seed
        Write-Host $summary

        if ($DryRun) {
            Write-Host ("python {0}" -f ($run.Args -join ' '))
            continue
        }

        & python @($run.Args)
        if ($LASTEXITCODE -ne 0) {
            throw "Run failed: $summary"
        }
    }

    Write-Host '========================================================================='
    Write-Host 'MULTI-SEED CAMPAIGN COMPLETE'
    Write-Host '========================================================================='
}
finally {
    Pop-Location
}