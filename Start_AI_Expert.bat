@echo off
title AI Expert - Data Scientist
echo ================================================
echo    AI Expert - Data Scientist
echo    خبير البيانات الذكي
echo ================================================
echo.
echo Starting application...
echo.

cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from python.org
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
)

REM Run the launcher
REM Run Streamlit Directly
streamlit run app.py --server.headless=false --browser.gatherUsageStats=false

pause
