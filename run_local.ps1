param([ValidateSet('scan','convert','batch')][string]$Command = 'scan', [string]$Config = '')
$ErrorActionPreference = 'Stop'
if (-not $Config) { $Config = Join-Path $PSScriptRoot 'configs\local.json' }
if (-not (Test-Path -LiteralPath $Config)) { throw 'Copy configs/example.json to configs/local.json and edit the dataset paths first.' }
Push-Location -LiteralPath $PSScriptRoot
try {
    python -m worldbridge $Command --config $Config
    if ($LASTEXITCODE -ne 0) { throw "WorldBridge exited with code $LASTEXITCODE" }
} finally { Pop-Location }

