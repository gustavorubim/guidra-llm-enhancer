param(
    [string]$CheckpointDir = "",
    [string]$TrainingProfile = "grpo_qwen35_2b",
    [string]$Split = "val",
    [int]$InspectionSampleCount = 8,
    [int]$MaxNewTokens = 384,
    [double]$Temperature = 0.0,
    [int]$SampleLimit = 0
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = (Resolve-Path (Join-Path $repoRoot "src")).Path

$args = @(
    "-m", "decomp_clarifier.cli",
    "eval-grpo-checkpoint",
    "--training-profile", $TrainingProfile,
    "--split", $Split,
    "--inspection-sample-count", $InspectionSampleCount,
    "--max-new-tokens", $MaxNewTokens,
    "--temperature", $Temperature
)

if ($CheckpointDir) {
    $args += @("--checkpoint-dir", $CheckpointDir)
}
if ($SampleLimit -gt 0) {
    $args += @("--sample-limit", $SampleLimit)
}

& python @args
