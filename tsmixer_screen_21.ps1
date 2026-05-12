$ErrorActionPreference = 'Stop'

Set-Location $PSScriptRoot

$pythonCandidates = @(
    'C:\Users\lab2\AppData\Local\Programs\Python\Python310\python.exe',
    (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe')
)

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
    if (Test-Path $candidate) {
        try {
            $torchOk = & $candidate -c "import torch; print(torch.__version__)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $pythonExe = $candidate
                break
            }
        }
        catch {
        }
    }
}

if (-not $pythonExe) {
    throw 'No Python environment with torch was found.'
}

Write-Host "Using Python: $pythonExe"

$logDir = Join-Path $PSScriptRoot 'results_analysis\tsmixer_screen_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$commonArgs = @(
    '--task_name', 'long_term_forecast',
    '--is_training', '1',
    '--model', 'TSMixer',
    '--features', 'M',
    '--n_heads', '8',
    '--d_layers', '1',
    '--factor', '3',
    '--patch_len', '16',
    '--stride', '8',
    '--itr', '1',
    '--train_epochs', '100',
    '--patience', '10',
    '--num_workers', '0',
    '--seed', '2021',
    '--des', 'tsmixer_screen_v2'
)

$runs = @(
    @{ DatasetTag = 'ETTm1'; Data = 'ETTm1'; RootPath = '.\data\'; DataPath = 'ETTm1.csv'; SeqLen = 336; LabelLen = 48; PredLen = 96; EncIn = 7; DecIn = 7; COut = 7; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },
    @{ DatasetTag = 'ETTm1'; Data = 'ETTm1'; RootPath = '.\data\'; DataPath = 'ETTm1.csv'; SeqLen = 336; LabelLen = 96; PredLen = 192; EncIn = 7; DecIn = 7; COut = 7; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },
    @{ DatasetTag = 'ETTm1'; Data = 'ETTm1'; RootPath = '.\data\'; DataPath = 'ETTm1.csv'; SeqLen = 336; LabelLen = 168; PredLen = 336; EncIn = 7; DecIn = 7; COut = 7; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },

    @{ DatasetTag = 'ETTm2'; Data = 'ETTm2'; RootPath = '.\data\'; DataPath = 'ETTm2.csv'; SeqLen = 336; LabelLen = 48; PredLen = 96; EncIn = 7; DecIn = 7; COut = 7; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },
    @{ DatasetTag = 'ETTm2'; Data = 'ETTm2'; RootPath = '.\data\'; DataPath = 'ETTm2.csv'; SeqLen = 336; LabelLen = 96; PredLen = 192; EncIn = 7; DecIn = 7; COut = 7; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },
    @{ DatasetTag = 'ETTm2'; Data = 'ETTm2'; RootPath = '.\data\'; DataPath = 'ETTm2.csv'; SeqLen = 336; LabelLen = 168; PredLen = 336; EncIn = 7; DecIn = 7; COut = 7; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },

    @{ DatasetTag = 'ETTh1'; Data = 'ETTh1'; RootPath = '.\data\'; DataPath = 'ETTh1.csv'; SeqLen = 336; LabelLen = 48; PredLen = 96; EncIn = 7; DecIn = 7; COut = 7; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },
    @{ DatasetTag = 'ETTh1'; Data = 'ETTh1'; RootPath = '.\data\'; DataPath = 'ETTh1.csv'; SeqLen = 336; LabelLen = 96; PredLen = 192; EncIn = 7; DecIn = 7; COut = 7; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },
    @{ DatasetTag = 'ETTh1'; Data = 'ETTh1'; RootPath = '.\data\'; DataPath = 'ETTh1.csv'; SeqLen = 336; LabelLen = 168; PredLen = 336; EncIn = 7; DecIn = 7; COut = 7; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },

    @{ DatasetTag = 'ETTh2'; Data = 'ETTh2'; RootPath = '.\data\'; DataPath = 'ETTh2.csv'; SeqLen = 336; LabelLen = 48; PredLen = 96; EncIn = 7; DecIn = 7; COut = 7; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },
    @{ DatasetTag = 'ETTh2'; Data = 'ETTh2'; RootPath = '.\data\'; DataPath = 'ETTh2.csv'; SeqLen = 336; LabelLen = 96; PredLen = 192; EncIn = 7; DecIn = 7; COut = 7; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },
    @{ DatasetTag = 'ETTh2'; Data = 'ETTh2'; RootPath = '.\data\'; DataPath = 'ETTh2.csv'; SeqLen = 336; LabelLen = 168; PredLen = 336; EncIn = 7; DecIn = 7; COut = 7; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },

    @{ DatasetTag = 'Exchange'; Data = 'custom'; RootPath = '.\data\'; DataPath = 'exchange_rate.csv'; SeqLen = 336; LabelLen = 48; PredLen = 96; EncIn = 8; DecIn = 8; COut = 8; DModel = 512; ELayers = 2; DFF = 2048; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },
    @{ DatasetTag = 'Exchange'; Data = 'custom'; RootPath = '.\data\'; DataPath = 'exchange_rate.csv'; SeqLen = 336; LabelLen = 96; PredLen = 192; EncIn = 8; DecIn = 8; COut = 8; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },
    @{ DatasetTag = 'Exchange'; Data = 'custom'; RootPath = '.\data\'; DataPath = 'exchange_rate.csv'; SeqLen = 336; LabelLen = 168; PredLen = 336; EncIn = 8; DecIn = 8; COut = 8; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },

    @{ DatasetTag = 'Weather'; Data = 'custom'; RootPath = '.\data\'; DataPath = 'weather.csv'; SeqLen = 336; LabelLen = 48; PredLen = 96; EncIn = 21; DecIn = 21; COut = 21; DModel = 512; ELayers = 2; DFF = 2048; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },
    @{ DatasetTag = 'Weather'; Data = 'custom'; RootPath = '.\data\'; DataPath = 'weather.csv'; SeqLen = 336; LabelLen = 96; PredLen = 192; EncIn = 21; DecIn = 21; COut = 21; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },
    @{ DatasetTag = 'Weather'; Data = 'custom'; RootPath = '.\data\'; DataPath = 'weather.csv'; SeqLen = 336; LabelLen = 168; PredLen = 336; EncIn = 21; DecIn = 21; COut = 21; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 32; LearningRate = 0.0001 },

    @{ DatasetTag = 'Illness'; Data = 'custom'; RootPath = '.\data\'; DataPath = 'national_illness.csv'; SeqLen = 104; LabelLen = 18; PredLen = 24; EncIn = 7; DecIn = 7; COut = 7; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 16; LearningRate = 0.0001 },
    @{ DatasetTag = 'Illness'; Data = 'custom'; RootPath = '.\data\'; DataPath = 'national_illness.csv'; SeqLen = 104; LabelLen = 18; PredLen = 48; EncIn = 7; DecIn = 7; COut = 7; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 16; LearningRate = 0.0001 },
    @{ DatasetTag = 'Illness'; Data = 'custom'; RootPath = '.\data\'; DataPath = 'national_illness.csv'; SeqLen = 104; LabelLen = 18; PredLen = 60; EncIn = 7; DecIn = 7; COut = 7; DModel = 128; ELayers = 2; DFF = 512; Dropout = 0.1; BatchSize = 16; LearningRate = 0.0001 }
)

$total = $runs.Count
$index = 0
foreach ($run in $runs) {
    $index += 1
    $modelId = "TSMixerScreenV2_$($run.DatasetTag)_$($run.PredLen)"
    $logPath = Join-Path $logDir ($modelId + '.log')

    Write-Host ''
    Write-Host ('=' * 100)
    Write-Host "[$index/$total] Running $modelId"
    Write-Host ('=' * 100)

    $args = @(
        '-u', '.\run.py',
        '--model_id', $modelId,
        '--data', $run.Data,
        '--root_path', $run.RootPath,
        '--data_path', $run.DataPath,
        '--seq_len', "$($run.SeqLen)",
        '--label_len', "$($run.LabelLen)",
        '--pred_len', "$($run.PredLen)",
        '--enc_in', "$($run.EncIn)",
        '--dec_in', "$($run.DecIn)",
        '--c_out', "$($run.COut)",
        '--d_model', "$($run.DModel)",
        '--e_layers', "$($run.ELayers)",
        '--d_ff', "$($run.DFF)",
        '--dropout', "$($run.Dropout)",
        '--batch_size', "$($run.BatchSize)",
        '--learning_rate', "$($run.LearningRate)"
    ) + $commonArgs

    & $pythonExe @args 2>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "TSMixer screening run failed: $modelId"
    }
}

Write-Host ''
Write-Host 'All 21 TSMixer screening runs completed successfully.'
