param(
  [string]$VpsHost,
  [string]$VpsUser,
  [int]$RemotePort,
  [string]$RemoteBindAddress,
  [int]$LocalPort,
  [string]$SshKeyPath
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot | Split-Path -Parent

function Import-DotEnv {
  param([string]$Path)

  if (-not (Test-Path $Path)) { return }

  Get-Content $Path | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith('#')) { return }
    $idx = $line.IndexOf('=')
    if ($idx -lt 1) { return }
    $name = $line.Substring(0, $idx).Trim()
    $value = $line.Substring($idx + 1)
    if (-not [string]::IsNullOrWhiteSpace($name) -and [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, 'Process'))) {
      [Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
  }
}

function Get-EnvInt {
  param(
    [string]$Name,
    [int]$DefaultValue
  )

  $raw = [Environment]::GetEnvironmentVariable($Name, 'Process')
  if ([string]::IsNullOrWhiteSpace($raw)) {
    return $DefaultValue
  }

  try {
    return [int]$raw
  } catch {
    return $DefaultValue
  }
}

Import-DotEnv -Path (Join-Path $root '.env')

if ([string]::IsNullOrWhiteSpace($VpsHost)) { $VpsHost = $env:VPS_HOST }
if ([string]::IsNullOrWhiteSpace($VpsUser)) { $VpsUser = $env:VPS_USER }
if (-not $PSBoundParameters.ContainsKey('RemotePort')) { $RemotePort = Get-EnvInt -Name 'REMOTE_PORT' -DefaultValue 19192 }
if ([string]::IsNullOrWhiteSpace($RemoteBindAddress)) { $RemoteBindAddress = $env:REMOTE_BIND_ADDRESS }
if (-not $PSBoundParameters.ContainsKey('LocalPort')) { $LocalPort = Get-EnvInt -Name 'LOCAL_PORT' -DefaultValue 8018 }
if ([string]::IsNullOrWhiteSpace($SshKeyPath)) { $SshKeyPath = $env:SSH_KEY_PATH }

if ([string]::IsNullOrWhiteSpace($VpsHost)) { $VpsHost = '81.163.29.109' }
if ([string]::IsNullOrWhiteSpace($VpsUser)) { $VpsUser = 'root' }
if ([string]::IsNullOrWhiteSpace($RemoteBindAddress)) { $RemoteBindAddress = '127.0.0.1' }
if ([string]::IsNullOrWhiteSpace($SshKeyPath)) { $SshKeyPath = "$env:USERPROFILE\.ssh\farmspot_vps_ed25519" }
if (-not $RemotePort) { $RemotePort = 19192 }
if (-not $LocalPort) { $LocalPort = 8018 }

$powershellExe = Join-Path $PSHOME 'powershell.exe'
if (-not (Test-Path (Join-Path $root 'logs'))) {
  New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null
}

function Start-VisibleWindow {
  param(
    [string]$Title,
    [string]$ScriptPath,
    [string[]]$Arguments = @()
  )

  $scriptArgs = @('-NoLogo', '-NoExit', '-ExecutionPolicy', 'Bypass', '-File', $ScriptPath) + $Arguments
  $process = Start-Process -FilePath $powershellExe -ArgumentList $scriptArgs -WorkingDirectory $root -PassThru
  $pidPath = Join-Path $root "logs\$($Title -replace '[^a-zA-Z0-9]+','_').pid"
  Set-Content -Path $pidPath -Value $process.Id
}

Start-VisibleWindow -Title 'game-id service' -ScriptPath (Join-Path $root 'scripts\start_service.ps1')
Start-Sleep -Seconds 4
Start-VisibleWindow -Title 'game-id tunnel' -ScriptPath (Join-Path $root 'scripts\start_tunnel.ps1') -Arguments @(
  '-VpsHost', $VpsHost,
  '-VpsUser', $VpsUser,
  '-RemotePort', "$RemotePort",
  '-RemoteBindAddress', $RemoteBindAddress,
  '-LocalPort', "$LocalPort",
  '-SshKeyPath', $SshKeyPath
)

Write-Host "Started game-id service on 127.0.0.1:$LocalPort and reverse tunnel to VPS 127.0.0.1:$RemotePort."
