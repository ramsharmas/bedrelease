@echo off
REM One-time setup for Windows: creates a virtual environment and installs the packages.
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% --version || (echo Python 3.11-3.13 not found. Install from python.org and tick "Add to PATH". & pause & exit /b 1)
%PY% -m venv .venv || (echo Could not create venv & pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || (echo pip install failed & pause & exit /b 1)
echo.
echo Setup done. Next: double-click run_dashboard.bat  (outputs are already included; run_pipeline.bat retrains from scratch, about 2-3 minutes)
pause
