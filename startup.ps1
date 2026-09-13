# Start WebbDuck from this repository directory (Windows / PowerShell).
# Usage: .\startup.ps1 [-EnvName webbduck] [-OutputDir ...] [-Port 8010] [-NoConda] [-- <extra run.py args>]
param(
    [string]$EnvName = $env:WEBBDUCK_CONDA_ENV,
    [string]$OutputDir = $env:WEBBDUCK_OUTPUT_DIR,
    [string]$ModelsDir = $env:WEBBDUCK_MODELS_DIR,
    [string]$HfCacheDir = $env:WEBBDUCK_HF_CACHE_DIR,
    [int]$Port = 8010,
    [switch]$NoConda,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$Root = (Get-Location).Path

function Fail([string]$message) {
    Write-Host "[startup] ERROR: $message" -ForegroundColor Red
    exit 1
}

function Info([string]$message) {
    Write-Host "[startup] $message"
}

if ($null -eq $ExtraArgs) { $ExtraArgs = @() }
if ($ExtraArgs.Length -gt 0 -and $ExtraArgs[0] -eq "--") {
    $ExtraArgs = $ExtraArgs[1..($ExtraArgs.Length - 1)]
}

if ($OutputDir -eq "") { $OutputDir = Join-Path $Root "outputs" }
if ($EnvName -eq "") { $EnvName = "webbduck" }

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TQDM_DISABLE = "1"

# Source runtime locations persisted by tools\prepare_model_runtimes.py.
$runtimeHome = $env:WEBBDUCK_RUNTIME_HOME
if ($runtimeHome -eq "" -or $null -eq $runtimeHome) {
    $runtimeHome = Join-Path $env:USERPROFILE ".local\share\webbduck\runtimes"
}
$env:WEBBDUCK_RUNTIME_HOME = $runtimeHome
$envFile = Join-Path $runtimeHome "webbduck_runtime.env.ps1"
if (Test-Path $envFile) {
    Info "Sourcing runtime env: $envFile"
    . $envFile
} else {
    Info "No runtime env file at $envFile (install engines with tools\prepare_model_runtimes.py)"
}

if (-not $NoConda) {
    $condaFound = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -eq $condaFound) { Fail "Conda is not available in PATH (use -NoConda to skip activation)" }
    conda activate $EnvName
    if ($LASTEXITCODE -ne 0) { Fail "Failed to activate conda env: $EnvName" }
}

if ($null -eq (Get-Command python -ErrorAction SilentlyContinue)) {
    Fail "python not found after environment setup"
}

Info "Repository: $Root"
Info "Conda env: $EnvName"
Info "Output: $OutputDir"
if ($ModelsDir) { Info "Models root: $ModelsDir" }
if ($HfCacheDir) { Info "HF cache root: $HfCacheDir" }
Info "Port: $Port"

$runArgs = @("--output", $OutputDir, "--port", "$Port")
if ($ModelsDir) { $runArgs += @("--models", $ModelsDir) }
if ($HfCacheDir) { $runArgs += @("--hf-cache", $HfCacheDir) }
$runArgs += $ExtraArgs

& python run.py @runArgs
exit $LASTEXITCODE