$ErrorActionPreference = 'Stop'

function Stop-ProcessTree {
  param(
    [int]$ProcessId,
    [string]$Label
  )

  if ($ProcessId -le 0) {
    return
  }

  try {
    & taskkill /PID $ProcessId /T /F | Out-Null
    Write-Host "Stopped $Label PID $ProcessId"
  } catch {
    Write-Warning ("Failed to stop {0} PID {1}: {2}" -f $Label, $ProcessId, $_.Exception.Message)
  }
}

$root = $PSScriptRoot | Split-Path -Parent
$pidFiles = @(
  @{ label = 'service'; paths = @((Join-Path $root 'logs\game_id_service.pid')) },
  @{ label = 'tunnel'; paths = @((Join-Path $root 'logs\game_id_tunnel.pid')) }
)

foreach ($entry in $pidFiles) {
  $path = $entry.paths | Where-Object { Test-Path $_ } | Select-Object -First 1
  if (-not $path) { continue }

  $raw = Get-Content $path -ErrorAction SilentlyContinue | Select-Object -First 1
  $targetPid = 0
  if (-not [string]::IsNullOrWhiteSpace($raw)) {
    [void][int]::TryParse($raw.Trim(), [ref]$targetPid)
  }

  Stop-ProcessTree -ProcessId $targetPid -Label $entry.label
  foreach ($candidate in $entry.paths) {
    Remove-Item $candidate -Force -ErrorAction SilentlyContinue
  }
}

Write-Host 'Game ID service stop request complete.'
