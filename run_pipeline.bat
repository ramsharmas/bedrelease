@echo off
REM Re-creates every file in outputs\ from the raw data (about 2-3 minutes). Add --download to re-fetch the raw CSV.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (echo Run setup.bat first & pause & exit /b 1)
call .venv\Scripts\activate.bat
python run_pipeline.py %*
pause
