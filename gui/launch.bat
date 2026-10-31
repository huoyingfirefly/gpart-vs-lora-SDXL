@echo off
REM Launch GPart Studio from anywhere; the working directory must be the repo
REM root because every path inside app.py is resolved relative to it.
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" gui\app.py %*
) else (
    python gui\app.py %*
)
if errorlevel 1 pause
