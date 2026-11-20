param([int]$Port = 8877, [string]$DataRoot = '', [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$webRoot = $PSScriptRoot
$pythonExe = (Get-Command python -ErrorAction Stop).Source
$localState = Join-Path $webRoot '.local'
New-Item -ItemType Directory -Path $localState -Force | Out-Null
$statePath = Join-Path $localState 'server.json'
if (-not $DataRoot) { $DataRoot = $env:CODEASMEM_DATA_ROOT }
if (-not $DataRoot) { $DataRoot = Join-Path (Split-Path $webRoot -Parent) 'data\unified_worlds' }
$resolvedData = [System.IO.Path]::GetFullPath($DataRoot)
if (Test-Path -LiteralPath $statePath) {
    $existing = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    try {
        $health = Invoke-RestMethod -Uri ($existing.url + '/api/health') -TimeoutSec 3
        if ($health.app -eq 'codeasmem-local-v1' -and $health.pid -eq $existing.pid) {
            if ($health.data_root -ne $resolvedData -or $health.web_root -ne $webRoot) {
                throw 'This saved server uses another dataset or checkout. Stop it before changing DataRoot.'
            }
            Write-Output ('WorldBridge is running: ' + $existing.url)
            if (-not $NoBrowser) { Start-Process $existing.url }
            exit 0
        }
    } catch { }
}
$listener = $null
try {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    $listener.Start()
} catch { throw "Port $Port is in use. Run start.ps1 -Port 8878 to choose another port." }
finally { if ($listener) { $listener.Stop() } }
$env:CODEASMEM_DATA_ROOT = $resolvedData
$proc = Start-Process -FilePath $pythonExe -ArgumentList @('app.py','--port',"$Port") -WorkingDirectory $webRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $localState 'server.stdout.log') -RedirectStandardError (Join-Path $localState 'server.stderr.log') -PassThru
$url = 'http://127.0.0.1:' + $Port
$ready = $false
for ($attempt = 0; $attempt -lt 50; $attempt++) {
    Start-Sleep -Milliseconds 200
    if ($proc.HasExited) { throw 'Server stopped during startup. See server.stderr.log.' }
    try {
        $health = Invoke-RestMethod -Uri ($url + '/api/health') -TimeoutSec 2
        if ($health.app -eq 'codeasmem-local-v1' -and $health.pid -eq $proc.Id) { $ready = $true; break }
    } catch { }
}
if (-not $ready) { throw 'Startup timed out. See server.stderr.log.' }
@{url=$url;pid=$proc.Id;app='codeasmem-local-v1';data_root=$resolvedData;started_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
Write-Output ('WorldBridge started: ' + $url)
if (-not $NoBrowser) { Start-Process $url }
