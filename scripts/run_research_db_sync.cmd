@echo off
setlocal
set "ROOT=%~dp0.."
cd /d "%ROOT%"
if not exist "data" mkdir "data"
echo ==== %DATE% %TIME% ====>> "data\research-db-sync.log"
python "scripts\sync_research_db_local.py" >> "data\research-db-sync.log" 2>&1
exit /b %ERRORLEVEL%
