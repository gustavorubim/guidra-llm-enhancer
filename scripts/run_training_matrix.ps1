param(
    [string]$AppProfile = "default",
    [string]$Split = "val",
    [int]$SampleLimit = 0,
    [int]$InspectionSampleCount = 8,
    [int]$MaxNewTokens = 384,
    [double]$Temperature = 0.0,
    [switch]$SkipReport,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Resolve-Path (Join-Path $repoRoot ".venv\Scripts\python.exe")
$env:PYTHONPATH = (Resolve-Path (Join-Path $repoRoot "src")).Path

$modelGroups = @(
    [ordered]@{
        Name = "2b"
        SftProfiles = @("sft_qwen35_2b", "sft_gemma4_e2b_it")
        GrpoProfiles = @("grpo_qwen35_2b", "grpo_gemma4_e2b_it")
    },
    [ordered]@{
        Name = "4b"
        SftProfiles = @("sft_qwen35_4b", "sft_gemma4_e4b_it")
        GrpoProfiles = @("grpo_qwen35_4b", "grpo_gemma4_e4b_it")
    }
)

function Write-JsonFile {
    param(
        [string]$Path,
        [object]$Payload
    )

    $parent = Split-Path -Parent $Path
    if ($parent) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -Path $Path -Encoding UTF8
}

function Resolve-ManifestPathFromOutput {
    param([string[]]$Lines)

    for ($index = $Lines.Count - 1; $index -ge 0; $index--) {
        $candidate = $Lines[$index].Trim()
        if (-not $candidate) {
            continue
        }
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "CLI step did not emit an existing manifest path."
}

function Invoke-MatrixStep {
    param(
        [hashtable]$Step,
        [int]$StepIndex,
        [int]$TotalSteps,
        [string]$StepsDir,
        [string]$PythonExe
    )

    $logPath = Join-Path $StepsDir ("{0:D2}-{1}.log" -f $StepIndex, $Step.Label)
    $command = @("-m", "decomp_clarifier.cli") + $Step.CliArgs

    Write-Host ""
    Write-Host "[$StepIndex/$TotalSteps] $($Step.Label)" -ForegroundColor Cyan
    Write-Host ("Command: {0} {1}" -f $PythonExe, ($command -join " "))
    Write-Host "Log: $logPath"
    Write-Host ("-" * 80)

    $record = [ordered]@{
        label = $Step.Label
        command = @($PythonExe) + $command
        log = $logPath
    }

    if ($DryRun) {
        $record.status = "planned"
        return $record
    }

    $startedAt = Get-Date
    $output = & $PythonExe @command 2>&1 | Tee-Object -FilePath $logPath
    $returnCode = $LASTEXITCODE
    $elapsedSeconds = [Math]::Round(((Get-Date) - $startedAt).TotalSeconds, 2)
    $lines = @($output | ForEach-Object { $_.ToString() })

    $record.returncode = $returnCode
    $record.elapsed_seconds = $elapsedSeconds

    if ($returnCode -ne 0) {
        $record.status = "failed"
        $record.tail = @($lines | Select-Object -Last 40)
        Write-Host ("-" * 80)
        Write-Host "[$StepIndex/$TotalSteps] $($Step.Label) failed after $elapsedSeconds s" -ForegroundColor Red
        throw ($record | ConvertTo-Json -Depth 10)
    }

    $record.status = "completed"
    if ($Step.ExpectsManifest) {
        $record.manifest_path = Resolve-ManifestPathFromOutput -Lines $lines
    }

    Write-Host ("-" * 80)
    Write-Host "[$StepIndex/$TotalSteps] $($Step.Label) completed in $elapsedSeconds s" -ForegroundColor Green
    return $record
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runId = "train-matrix-$timestamp"
$runDir = Join-Path $repoRoot "artifacts\runs\$runId"
$stepsDir = Join-Path $runDir "steps"
New-Item -ItemType Directory -Path $stepsDir -Force | Out-Null

$stepDefinitions = New-Object System.Collections.Generic.List[object]
foreach ($group in $modelGroups) {
    foreach ($profile in $group.SftProfiles) {
        $stepDefinitions.Add([ordered]@{
            Label = "train-$profile"
            ExpectsManifest = $true
            CliArgs = @(
                "train-sft",
                "--training-profile", $profile,
                "--app-profile", $AppProfile
            )
        })
    }
    foreach ($profile in $group.GrpoProfiles) {
        $stepDefinitions.Add([ordered]@{
            Label = "train-$profile"
            ExpectsManifest = $true
            CliArgs = @(
                "train-grpo",
                "--training-profile", $profile,
                "--app-profile", $AppProfile
            )
        })
    }
    foreach ($profile in $group.SftProfiles) {
        $cliArgs = @(
            "eval-sft-checkpoint",
            "--training-profile", $profile,
            "--split", $Split,
            "--inspection-sample-count", "$InspectionSampleCount",
            "--max-new-tokens", "$MaxNewTokens",
            "--temperature", "$Temperature",
            "--app-profile", $AppProfile
        )
        if ($SampleLimit -gt 0) {
            $cliArgs += @("--sample-limit", "$SampleLimit")
        }
        $stepDefinitions.Add([ordered]@{
            Label = "eval-$profile"
            ExpectsManifest = $true
            CliArgs = $cliArgs
        })
    }
    foreach ($profile in $group.GrpoProfiles) {
        $cliArgs = @(
            "eval-grpo-checkpoint",
            "--training-profile", $profile,
            "--split", $Split,
            "--inspection-sample-count", "$InspectionSampleCount",
            "--max-new-tokens", "$MaxNewTokens",
            "--temperature", "$Temperature",
            "--app-profile", $AppProfile
        )
        if ($SampleLimit -gt 0) {
            $cliArgs += @("--sample-limit", "$SampleLimit")
        }
        $stepDefinitions.Add([ordered]@{
            Label = "eval-$profile"
            ExpectsManifest = $true
            CliArgs = $cliArgs
        })
    }
}

$state = [ordered]@{
    run_id = $runId
    started_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
    root = $repoRoot.Path
    app_profile = $AppProfile
    split = $Split
    sample_limit = if ($SampleLimit -gt 0) { $SampleLimit } else { $null }
    inspection_sample_count = $InspectionSampleCount
    max_new_tokens = $MaxNewTokens
    temperature = $Temperature
    execution_groups = @(
        foreach ($group in $modelGroups) {
            [ordered]@{
                name = $group.Name
                sft_profiles = @($group.SftProfiles)
                grpo_profiles = @($group.GrpoProfiles)
            }
        }
    )
    steps = @()
}

$manifestPath = Join-Path $runDir "matrix_run_manifest.json"
Write-JsonFile -Path $manifestPath -Payload $state

$evalManifestPaths = [ordered]@{}

try {
    $totalSteps = $stepDefinitions.Count + $(if ($SkipReport) { 0 } else { 1 })
    $stepIndex = 1

    foreach ($step in $stepDefinitions) {
        $record = Invoke-MatrixStep `
            -Step $step `
            -StepIndex $stepIndex `
            -TotalSteps $totalSteps `
            -StepsDir $stepsDir `
            -PythonExe $python.Path
        $state.steps += $record

        if ($record.status -eq "completed" -and $step.Label.StartsWith("eval-")) {
            $profile = $step.Label.Substring(5)
            $evalManifestPaths[$profile] = $record.manifest_path
        }

        Write-JsonFile -Path $manifestPath -Payload $state
        $stepIndex += 1
    }

    if (-not $SkipReport) {
        $reportArgs = @((Join-Path $repoRoot "scripts\build_model_matrix_summary.py"), "--app-profile", $AppProfile)
        foreach ($entry in $evalManifestPaths.GetEnumerator()) {
            $reportArgs += @("--eval-manifest", "$($entry.Key)=$($entry.Value)")
        }

        $reportStep = [ordered]@{
            Label = "build-model-matrix-summary"
            ExpectsManifest = $false
            CliArgs = @()
        }

        $reportLog = Join-Path $stepsDir ("{0:D2}-{1}.log" -f $stepIndex, $reportStep.Label)

        Write-Host ""
        Write-Host "[$stepIndex/$totalSteps] $($reportStep.Label)" -ForegroundColor Cyan
        Write-Host ("Command: {0} {1}" -f $python.Path, ($reportArgs -join " "))
        Write-Host "Log: $reportLog"
        Write-Host ("-" * 80)

        if ($DryRun) {
            $record = [ordered]@{
                label = $reportStep.Label
                command = @($python.Path) + $reportArgs
                log = $reportLog
                status = "planned"
            }
        } else {
            $startedAt = Get-Date
            $output = & $python.Path @reportArgs 2>&1 | Tee-Object -FilePath $reportLog
            $returnCode = $LASTEXITCODE
            $elapsedSeconds = [Math]::Round(((Get-Date) - $startedAt).TotalSeconds, 2)
            $lines = @($output | ForEach-Object { $_.ToString() })
            $record = [ordered]@{
                label = $reportStep.Label
                command = @($python.Path) + $reportArgs
                log = $reportLog
                returncode = $returnCode
                elapsed_seconds = $elapsedSeconds
            }
            if ($returnCode -ne 0) {
                $record.status = "failed"
                $record.tail = @($lines | Select-Object -Last 40)
                Write-Host ("-" * 80)
                Write-Host "[$stepIndex/$totalSteps] $($reportStep.Label) failed after $elapsedSeconds s" -ForegroundColor Red
                throw ($record | ConvertTo-Json -Depth 10)
            }

            $record.status = "completed"
            $state.summary_report = [ordered]@{
                model_matrix_markdown = (Join-Path $repoRoot "artifacts\reports\model_matrix_summary.md")
                model_matrix_json = (Join-Path $repoRoot "artifacts\reports\model_matrix_summary.json")
                target_comparison_markdown = (Join-Path $repoRoot "artifacts\reports\target_comparison_table.md")
                target_comparison_json = (Join-Path $repoRoot "artifacts\reports\target_comparison_table.json")
            }
            Write-Host ("-" * 80)
            Write-Host "[$stepIndex/$totalSteps] $($reportStep.Label) completed in $elapsedSeconds s" -ForegroundColor Green
        }

        $state.steps += $record
        Write-JsonFile -Path $manifestPath -Payload $state
    }

    $state.completed_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
    $state.status = "completed"
    Write-JsonFile -Path $manifestPath -Payload $state

    Write-Host ""
    Write-Host "Matrix run manifest: $manifestPath"
    if ($state.summary_report) {
        Write-Host "Model matrix markdown: $($state.summary_report.model_matrix_markdown)"
        Write-Host "Target comparison markdown: $($state.summary_report.target_comparison_markdown)"
    }
}
catch {
    $state.completed_at = (Get-Date).ToString("yyyy-MM-ddTHH:mm:sszzz")
    $state.status = "failed"
    $state.error = $_.Exception.Message
    Write-JsonFile -Path $manifestPath -Payload $state
    throw
}
