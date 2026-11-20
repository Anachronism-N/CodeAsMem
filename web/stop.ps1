$ErrorActionPreference = 'Stop'
$statePath = Join-Path $PSScriptRoot '.local\server.json'
if (-not (Test-Path -LiteralPath $statePath)) { Write-Output 'No saved server process.'; exit 0 }
$saved = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
try { $health = Invoke-RestMethod -Uri ($saved.url + '/api/health') -TimeoutSec 3 }
catch { Write-Output 'Saved server is not responding. No process stopped.'; exit 0 }
if ($health.app -ne 'codeasmem-local-v1' -or $health.pid -ne $saved.pid -or $health.web_root -ne $PSScriptRoot) {
    throw 'Saved PID does not match WorldBridge health. No process stopped.'
}
$owned = Get-CimInstance Win32_Process -Filter ("ProcessId=" + [int]$saved.pid)
if ($null -eq $owned -or $owned.CommandLine -notmatch 'app\.py\s+--port') {
    throw 'Process identity check failed. No process stopped.'
}
Stop-Process -Id ([int]$saved.pid)
Write-Output 'WorldBridge stopped.'
