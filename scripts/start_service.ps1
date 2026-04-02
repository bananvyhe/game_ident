$ErrorActionPreference = 'Stop'

if (-not (Test-Path .env)) {
  Copy-Item .env.example .env
  Write-Host 'Created .env from .env.example. Update tokens and ports before production use.'
}

$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
  throw "Virtual environment python not found: $python. Run installation first."
}

try {
  $Host.UI.RawUI.WindowTitle = 'game-id service'
} catch {
}

$logDir = Join-Path (Get-Location) 'logs'
if (-not (Test-Path $logDir)) {
  New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

$logPath = Join-Path $logDir ("service-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "[service] logging to $logPath"
Start-Transcript -Path $logPath | Out-Null
try {
  Write-Host '[service] starting uvicorn on 127.0.0.1:8018'
  & $python -m uvicorn app.main:app --host 127.0.0.1 --port 8018
} finally {
  Stop-Transcript | Out-Null
}
