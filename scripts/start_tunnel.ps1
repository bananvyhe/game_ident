param(
  [Parameter(Mandatory=$true)][string]$VpsHost,
  [Parameter(Mandatory=$true)][string]$VpsUser,
  [int]$RemotePort = 19192,
  [string]$RemoteBindAddress = "127.0.0.1",
  [int]$LocalPort = 8018,
  [string]$SshKeyPath = "$HOME\.ssh\farmspot_vps_ed25519",
  [string]$SshPassword = $null,
  [string]$VpsHostKeySha256 = $null,
  [switch]$SkipHostKeyCheck,
  [string]$KnownHostsPath = "$HOME\.ssh\farmspot_known_hosts"
)

$ErrorActionPreference = 'Stop'

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
    if (-not [string]::IsNullOrWhiteSpace($name)) {
      [Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
  }
}

function Get-ExpectedFingerprints {
  param([string]$FingerprintSha256)

  if ([string]::IsNullOrWhiteSpace($FingerprintSha256)) {
    return @()
  }

  $values = @()
  foreach ($raw in ($FingerprintSha256 -split '[,;\s]+')) {
    $item = $raw.Trim()
    if ([string]::IsNullOrWhiteSpace($item)) { continue }
    if ($item -match '(SHA256:[A-Za-z0-9+/=]+)') {
      $values += $Matches[1]
    }
  }

  return @($values | Select-Object -Unique)
}

function Resolve-KnownHost {
  param(
    [string]$HostName,
    [string]$FingerprintSha256,
    [string]$OutputPath
  )

  $expectedFingerprints = Get-ExpectedFingerprints -FingerprintSha256 $FingerprintSha256
  if (-not $expectedFingerprints.Count) {
    return $null
  }

  $scanPath = Join-Path $env:TEMP "gameid_ssh_scan_$(Get-Random).txt"
  try {
    $scanOutput = & cmd /c "ssh-keyscan -T 5 $HostName 2>nul"
    $scanLines = @($scanOutput | Where-Object { $_ -and -not $_.StartsWith('#') })
    if (-not $scanLines.Count) {
      Write-Warning "Unable to fetch SSH host keys from $HostName. Continuing without host key pinning."
      return $null
    }

    $scanLines | Set-Content -Path $scanPath
    if (-not (Test-Path $scanPath) -or (Get-Item $scanPath).Length -eq 0) {
      Write-Warning "Unable to fetch SSH host keys from $HostName. Continuing without host key pinning."
      return $null
    }

    $fingerprints = @(& ssh-keygen -lf $scanPath 2>$null)
    $matchedIndex = $null
    $seenFingerprints = @()

    for ($i = 0; $i -lt $fingerprints.Count; $i++) {
      $line = $fingerprints[$i]
      if ($line -match '(SHA256:[A-Za-z0-9+/=]+)') {
        $fp = $Matches[1]
        $seenFingerprints += $fp
        if ($expectedFingerprints -contains $fp) {
          $matchedIndex = $i
          break
        }
      }
    }

    if ($null -eq $matchedIndex) {
      $seenText = if ($seenFingerprints.Count) { ($seenFingerprints -join ', ') } else { '<none>' }
      $expectedText = ($expectedFingerprints -join ', ')
      throw "SSH host fingerprint(s) $expectedText were not found for $HostName. Seen fingerprints: $seenText"
    }

    $matchedLine = $scanLines[$matchedIndex]
    $knownHostsDir = Split-Path $OutputPath -Parent
    if (-not (Test-Path $knownHostsDir)) {
      New-Item -ItemType Directory -Force -Path $knownHostsDir | Out-Null
    }

    if (-not (Test-Path $OutputPath)) {
      Set-Content -Path $OutputPath -Value $matchedLine
    } else {
      $existing = Get-Content $OutputPath -ErrorAction SilentlyContinue
      if ($existing -notcontains $matchedLine) {
        Add-Content -Path $OutputPath -Value $matchedLine
      }
    }

    return $OutputPath
  } finally {
    if (Test-Path $scanPath) {
      Remove-Item $scanPath -Force -ErrorAction SilentlyContinue
    }
  }
}

Import-DotEnv -Path (Join-Path (Get-Location) '.env')

if ($SkipHostKeyCheck) {
  $VpsHostKeySha256 = $null
}

if ([string]::IsNullOrWhiteSpace($SshPassword)) {
  $SshPassword = $env:SSH_PASSWD
}
if ([string]::IsNullOrWhiteSpace($VpsHostKeySha256)) {
  $VpsHostKeySha256 = $env:VPS_HOSTKEY_SHA256
}

$askpass = Join-Path (Get-Location) 'scripts\ssh_askpass.cmd'
$knownHosts = $null
if (-not $SkipHostKeyCheck) {
  $knownHosts = Resolve-KnownHost -HostName $VpsHost -FingerprintSha256 $VpsHostKeySha256 -OutputPath $KnownHostsPath
}

$logDir = Join-Path (Get-Location) 'logs'
if (-not (Test-Path $logDir)) {
  New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

$logPath = Join-Path $logDir ("tunnel-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Write-Host "[tunnel] logging to $logPath"
Start-Transcript -Path $logPath | Out-Null
try {
  while ($true) {
    Write-Host "[tunnel] starting reverse tunnel $VpsUser@$VpsHost (remote ${RemoteBindAddress}:$RemotePort -> local 127.0.0.1:$LocalPort)"

    $sshArgs = @(
      '-N', '-T',
      '-v',
      '-o', 'ExitOnForwardFailure=yes',
      '-o', 'ServerAliveInterval=30',
      '-o', 'ServerAliveCountMax=3',
      '-o', 'ConnectTimeout=15',
      '-R', "${RemoteBindAddress}:${RemotePort}:127.0.0.1:${LocalPort}"
    )

    if ($knownHosts) {
      $sshArgs += @('-o', "UserKnownHostsFile=$knownHosts", '-o', 'StrictHostKeyChecking=yes')
    }

    if (-not [string]::IsNullOrWhiteSpace($SshPassword)) {
      $env:SSH_PASSWD = $SshPassword
      $env:SSH_ASKPASS = $askpass
      $env:SSH_ASKPASS_REQUIRE = 'force'
      $env:DISPLAY = 'none'
      $sshArgs += @('-o', 'PreferredAuthentications=password', '-o', 'PubkeyAuthentication=no', '-o', 'BatchMode=no')
    } elseif (Test-Path $SshKeyPath) {
      $sshArgs += @('-i', $SshKeyPath)
    } else {
      throw "No SSH auth method available. Set SSH_PASSWD in .env or provide key at $SshKeyPath."
    }

    $sshArgs += "$VpsUser@$VpsHost"

    & ssh.exe @sshArgs

    Write-Host '[tunnel] disconnected, retry in 5s...'
    Start-Sleep -Seconds 5
  }
} finally {
  Stop-Transcript | Out-Null
}
