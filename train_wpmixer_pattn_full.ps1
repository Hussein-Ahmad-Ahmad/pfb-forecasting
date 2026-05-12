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

$logDir = Join-Path $PSScriptRoot 'results_analysis\wpmixer_pattn_full_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$resultFile = Join-Path $PSScriptRoot 'result_long_term_forecast.txt'

$models = @('WPMixer', 'PAttn')

$commonArgs = @(
    '--task_name', 'long_term_forecast',
    '--is_training', '1',
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
    '--des', 'full_screen_top2',
    '--learning_rate', '0.0001',
    '--dropout', '0.1',
    '--d_model', '128',
    '--e_layers', '2',
    '--d_ff', '512'
)

$runs = @(
    @{ DatasetTag = 'ETTm1';   Data = 'ETTm1';  RootPath = '.\data\'; DataPath = 'ETTm1.csv';            SeqLen = 336; LabelLen = 48;  PredLen = 96;  EncIn = 7;  DecIn = 7;  COut = 7;  BatchSize = 32 },
    @{ DatasetTag = 'ETTm1';   Data = 'ETTm1';  RootPath = '.\data\'; DataPath = 'ETTm1.csv';            SeqLen = 336; LabelLen = 96;  PredLen = 192; EncIn = 7;  DecIn = 7;  COut = 7;  BatchSize = 32 },
    @{ DatasetTag = 'ETTm1';   Data = 'ETTm1';  RootPath = '.\data\'; DataPath = 'ETTm1.csv';            SeqLen = 336; LabelLen = 168; PredLen = 336; EncIn = 7;  DecIn = 7;  COut = 7;  BatchSize = 32 },
    @{ DatasetTag = 'ETTm2';   Data = 'ETTm2';  RootPath = '.\data\'; DataPath = 'ETTm2.csv';            SeqLen = 336; LabelLen = 48;  PredLen = 96;  EncIn = 7;  DecIn = 7;  COut = 7;  BatchSize = 32 },
    @{ DatasetTag = 'ETTm2';   Data = 'ETTm2';  RootPath = '.\data\'; DataPath = 'ETTm2.csv';            SeqLen = 336; LabelLen = 96;  PredLen = 192; EncIn = 7;  DecIn = 7;  COut = 7;  BatchSize = 32 },
    @{ DatasetTag = 'ETTm2';   Data = 'ETTm2';  RootPath = '.\data\'; DataPath = 'ETTm2.csv';            SeqLen = 336; LabelLen = 168; PredLen = 336; EncIn = 7;  DecIn = 7;  COut = 7;  BatchSize = 32 },
    @{ DatasetTag = 'ETTh1';   Data = 'ETTh1';  RootPath = '.\data\'; DataPath = 'ETTh1.csv';            SeqLen = 336; LabelLen = 48;  PredLen = 96;  EncIn = 7;  DecIn = 7;  COut = 7;  BatchSize = 32 },
    @{ DatasetTag = 'ETTh1';   Data = 'ETTh1';  RootPath = '.\data\'; DataPath = 'ETTh1.csv';            SeqLen = 336; LabelLen = 96;  PredLen = 192; EncIn = 7;  DecIn = 7;  COut = 7;  BatchSize = 32 },
    @{ DatasetTag = 'ETTh1';   Data = 'ETTh1';  RootPath = '.\data\'; DataPath = 'ETTh1.csv';            SeqLen = 336; LabelLen = 168; PredLen = 336; EncIn = 7;  DecIn = 7;  COut = 7;  BatchSize = 32 },
    @{ DatasetTag = 'ETTh2';   Data = 'ETTh2';  RootPath = '.\data\'; DataPath = 'ETTh2.csv';            SeqLen = 336; LabelLen = 48;  PredLen = 96;  EncIn = 7;  DecIn = 7;  COut = 7;  BatchSize = 32 },
    @{ DatasetTag = 'ETTh2';   Data = 'ETTh2';  RootPath = '.\data\'; DataPath = 'ETTh2.csv';            SeqLen = 336; LabelLen = 96;  PredLen = 192; EncIn = 7;  DecIn = 7;  COut = 7;  BatchSize = 32 },
    @{ DatasetTag = 'ETTh2';   Data = 'ETTh2';  RootPath = '.\data\'; DataPath = 'ETTh2.csv';            SeqLen = 336; LabelLen = 168; PredLen = 336; EncIn = 7;  DecIn = 7;  COut = 7;  BatchSize = 32 },
    @{ DatasetTag = 'Exchange'; Data = 'custom'; RootPath = '.\data\'; DataPath = 'exchange_rate.csv';   SeqLen = 336; LabelLen = 48;  PredLen = 96;  EncIn = 8;  DecIn = 8;  COut = 8;  BatchSize = 32 },
    @{ DatasetTag = 'Exchange'; Data = 'custom'; RootPath = '.\data\'; DataPath = 'exchange_rate.csv';   SeqLen = 336; LabelLen = 96;  PredLen = 192; EncIn = 8;  DecIn = 8;  COut = 8;  BatchSize = 32 },
    @{ DatasetTag = 'Exchange'; Data = 'custom'; RootPath = '.\data\'; DataPath = 'exchange_rate.csv';   SeqLen = 336; LabelLen = 168; PredLen = 336; EncIn = 8;  DecIn = 8;  COut = 8;  BatchSize = 32 },
    @{ DatasetTag = 'Weather';  Data = 'custom'; RootPath = '.\data\'; DataPath = 'weather.csv';         SeqLen = 336; LabelLen = 48;  PredLen = 96;  EncIn = 21; DecIn = 21; COut = 21; BatchSize = 16 },
    @{ DatasetTag = 'Weather';  Data = 'custom'; RootPath = '.\data\'; DataPath = 'weather.csv';         SeqLen = 336; LabelLen = 96;  PredLen = 192; EncIn = 21; DecIn = 21; COut = 21; BatchSize = 16 },
    @{ DatasetTag = 'Weather';  Data = 'custom'; RootPath = '.\data\'; DataPath = 'weather.csv';         SeqLen = 336; LabelLen = 168; PredLen = 336; EncIn = 21; DecIn = 21; COut = 21; BatchSize = 16 },
    @{ DatasetTag = 'Illness';  Data = 'custom'; RootPath = '.\data\'; DataPath = 'national_illness.csv'; SeqLen = 104; LabelLen = 18; PredLen = 24; EncIn = 7; DecIn = 7; COut = 7; BatchSize = 16 },
    @{ DatasetTag = 'Illness';  Data = 'custom'; RootPath = '.\data\'; DataPath = 'national_illness.csv'; SeqLen = 104; LabelLen = 18; PredLen = 48; EncIn = 7; DecIn = 7; COut = 7; BatchSize = 16 },
    @{ DatasetTag = 'Illness';  Data = 'custom'; RootPath = '.\data\'; DataPath = 'national_illness.csv'; SeqLen = 104; LabelLen = 18; PredLen = 60; EncIn = 7; DecIn = 7; COut = 7; BatchSize = 16 }
)

$total = $models.Count * $runs.Count
$index = 0

foreach ($model in $models) {
    foreach ($run in $runs) {
        $index += 1
        $modelId = "${model}_Full_$($run.DatasetTag)_H$($run.PredLen)"
        $logPath = Join-Path $logDir ($modelId + '.log')
        $stdoutPath = Join-Path $logDir ($modelId + '.stdout.log')
        $stderrPath = Join-Path $logDir ($modelId + '.stderr.log')

        if ((Test-Path $resultFile) -and (Select-String -Path $resultFile -Pattern $modelId -Quiet)) {
            Write-Host "[$index/$total] Skipping completed run: $modelId"
            continue
        }

        Write-Host ''
        Write-Host ('=' * 100)
        Write-Host "[$index/$total] Running $modelId"
        Write-Host ('=' * 100)

        if (Test-Path $stdoutPath) { Remove-Item $stdoutPath -Force -ErrorAction SilentlyContinue }
        if (Test-Path $stderrPath) { Remove-Item $stderrPath -Force -ErrorAction SilentlyContinue }

        $args = @(
            '-u', '.\run.py',
            '--model_id', $modelId,
            '--model', $model,
            '--data', $run.Data,
            '--root_path', $run.RootPath,
            '--data_path', $run.DataPath,
            '--seq_len', "$($run.SeqLen)",
            '--label_len', "$($run.LabelLen)",
            '--pred_len', "$($run.PredLen)",
            '--enc_in', "$($run.EncIn)",
            '--dec_in', "$($run.DecIn)",
            '--c_out', "$($run.COut)",
            '--batch_size', "$($run.BatchSize)"
        ) + $commonArgs

        $proc = Start-Process -FilePath $pythonExe -ArgumentList $args -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

        $stdout = if (Test-Path $stdoutPath) { Get-Content $stdoutPath -Raw } else { '' }
        $stderr = if (Test-Path $stderrPath) { Get-Content $stderrPath -Raw } else { '' }
        ($stdout + $stderr) | Set-Content -Path $logPath

        if ($stdout) {
            Write-Host $stdout
        }
        if ($stderr) {
            Write-Warning $stderr
        }

        if ($proc.ExitCode -ne 0) {
            Write-Warning "Full run failed: $modelId (exit code $($proc.ExitCode))"
        }
    }
}

Write-Host ''
Write-Host 'WPMixer/PAttn full training runs finished.'