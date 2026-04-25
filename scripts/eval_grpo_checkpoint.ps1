param(
    [string]$CheckpointDir = "",
    [string]$TrainingProfile = "grpo_qwen35_2b",
    [string]$AppProfile = "default",
    [string]$Split = "val",
    [int]$InspectionSampleCount = 8,
    [int]$MaxNewTokens = 384,
    [double]$Temperature = 0.0,
    [string]$PromptProfile = "stage",
    [switch]$Thinking,
    [int]$SampleLimit = 0
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Resolve-Path (Join-Path $repoRoot ".venv\Scripts\python.exe")
$env:PYTHONPATH = (Resolve-Path (Join-Path $repoRoot "src")).Path

$args = @(
    "-m", "decomp_clarifier.cli",
    "eval-grpo-checkpoint",
    "--training-profile", $TrainingProfile,
    "--app-profile", $AppProfile,
    "--split", $Split,
    "--inspection-sample-count", $InspectionSampleCount,
    "--max-new-tokens", $MaxNewTokens,
    "--temperature", $Temperature,
    "--prompt-profile", $PromptProfile
)

if ($CheckpointDir) {
    $args += @("--checkpoint-dir", $CheckpointDir)
}
if ($SampleLimit -gt 0) {
    $args += @("--sample-limit", $SampleLimit)
}
if ($Thinking) {
    $args += @("--thinking")
}

& $python.Path @args
$returnCode = $LASTEXITCODE
if ($returnCode -ne 0) {
    throw "eval-grpo-checkpoint failed with exit code $returnCode"
}
