@echo off
setlocal
cd /d "%~dp0"
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_windows_game_id.ps1" %*
endlocal
