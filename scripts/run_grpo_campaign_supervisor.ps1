param(
    [string]$Tag = "gdpo-sft-target-apr24",
    [int]$MaxIterations = 60,
    [string]$SearchSpace = "post_target",
    [string]$EvalPromptProfile = "full",
    [int]$EvalMaxNewTokens = 1024,
    [double]$TargetImprovement = 0.02,
    [int]$RestartDelaySeconds = 15,
    [string]$PythonPath = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

if ([System.IO.Path]::IsPathRooted($PythonPath)) {
    $ResolvedPythonPath = $PythonPath
} else {
    $ResolvedPythonPath = Join-Path $Root $PythonPath
}

$SupervisorLog = Join-Path $Root "artifacts\logs\$Tag-supervisor.log"
New-Item -ItemType Directory -Force -Path (Split-Path $SupervisorLog) | Out-Null

function Write-SupervisorLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp $Message" | Add-Content -Path $SupervisorLog -Encoding UTF8
}

function Get-CompletedIterationCount {
    $experimentLog = Join-Path $Root "research\campaigns\$Tag\experiment_log.jsonl"
    if (-not (Test-Path $experimentLog)) {
        return 0
    }
    $count = 0
    foreach ($line in Get-Content $experimentLog) {
        if (-not $line.Trim()) {
            continue
        }
        $record = $line | ConvertFrom-Json
        if ($record.status -ne "baseline") {
            $count += 1
        }
    }
    return $count
}

while ($true) {
    $completed = Get-CompletedIterationCount
    if ($completed -ge $MaxIterations) {
        Write-SupervisorLog "completed=$completed max=$MaxIterations; stopping"
        break
    }

    Write-SupervisorLog "starting campaign completed=$completed max=$MaxIterations search_space=$SearchSpace"
    & $ResolvedPythonPath `
        -m decomp_clarifier.research.grpo_campaign `
        --tag $Tag `
        --max-iterations $MaxIterations `
        --target-improvement $TargetImprovement `
        --eval-prompt-profile $EvalPromptProfile `
        --eval-max-new-tokens $EvalMaxNewTokens `
        --no-stop-on-target `
        --search-space $SearchSpace

    $exitCode = $LASTEXITCODE
    $completed = Get-CompletedIterationCount
    Write-SupervisorLog "campaign exited code=$exitCode completed=$completed max=$MaxIterations"
    if ($completed -ge $MaxIterations) {
        break
    }
    Start-Sleep -Seconds $RestartDelaySeconds
}
