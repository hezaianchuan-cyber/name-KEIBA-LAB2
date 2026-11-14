# Run from the local clone of KEIBALAB research/keibalab-db-foundation.
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runner = Join-Path $PSScriptRoot "run_research_db_sync.cmd"
$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument ('/c "' + $Runner + '"') -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger -Daily -At "12:15AM" -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Hours 23)
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName "KEIBALAB Research DB Daily Sync" -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Description "Fetch official NAR odds snapshots and frozen model predictions into local SQLite." -Force | Out-Null
Write-Host "Registered hourly sync from 00:15 to 23:15 local time. It runs while this Windows account is logged on."
Write-Host "Run once now: scripts\run_research_db_sync.cmd"
Write-Host "Log file: data\research-db-sync.log"
