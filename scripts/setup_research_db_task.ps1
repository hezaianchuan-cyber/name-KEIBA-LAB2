# Run from the local clone of KEIBALAB research/keibalab-db-foundation.
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = (Get-Command python -ErrorAction Stop).Source
$Script = Join-Path $Root "scripts\sync_research_db_local.py"
$Action = New-ScheduledTaskAction -Execute $Python -Argument ('"' + $Script + '"') -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger -Daily -At "2:15AM"
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName "KEIBALAB Research DB Daily Sync" -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Description "Fetch official NAR odds snapshots and frozen model predictions into local SQLite." -Force | Out-Null
Write-Host "Registered daily sync at 02:15. It runs while this Windows account is logged on."
Write-Host "Run once now: python $Script"
