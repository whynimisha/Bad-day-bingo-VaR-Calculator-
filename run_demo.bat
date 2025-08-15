@echo off
setlocal enableextensions

REM --- Always run from this file's folder ---
cd /d "%~dp0"

REM --- Activate virtual environment ---
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [ERROR] Could not find .venv\Scripts\activate.bat
    echo Create/restore your venv, e.g.:  python -m venv .venv
    pause
    exit /b 1
)

REM --- Fix portfolio.csv if it was accidentally created as a folder ---
if exist "portfolio.csv\" (
    echo Found a folder named portfolio.csv. Deleting it...
    rmdir /s /q "portfolio.csv"
)

REM --- Create a default portfolio.csv if missing ---
if not exist "portfolio.csv" (
    echo Creating default portfolio.csv...
    > "portfolio.csv" (
        echo symbol,qty,price,multiplier
        echo RELIANCE.NS,10,,1
        echo TCS.NS,6,,1
        echo HDFCBANK.NS,12,,1
    )
)

REM --- Ensure outputs/ exists (for plots, json, etc.) ---
if not exist "outputs" mkdir "outputs"

REM --- Ensure yfinance is installed ---
pip show yfinance >nul 2>&1
if errorlevel 1 (
    echo Installing yfinance...
    pip install yfinance
)

REM --- Run the demo ---
echo Running: Student-t, 1-day, 20k scenarios, with KDE...
python var_engine.py --mode t --df 6 --horizon 1 --scenarios 20000 --kde

echo.
echo Done. Check:
echo   outputs\results.json
echo   outputs\pnl_hist_1d_t.png
echo   outputs\pnl_hist_5d_t.png
echo   outputs\pnl_hist_10d_t.png
echo   outputs\pnl_hist_20d_t.png
pause
