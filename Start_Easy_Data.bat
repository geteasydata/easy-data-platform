@echo off
chcp 65001 >nul
title Easy Data - AI-Powered Data Analysis
echo ════════════════════════════════════════════════════════════
echo              💎 Easy Data - البيانات السهلة
echo    AI-Powered Data Science ^& Machine Learning Platform
echo ════════════════════════════════════════════════════════════
echo.

cd /d "%~dp0"

REM Check if venv exists and activate it
if exist ".venv\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call ".venv\Scripts\activate.bat"
) else (
    echo ⚠️ Virtual environment not found, using system Python...
    REM Check if Python is available
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ❌ ERROR: Python is not installed or not in PATH
        echo    Please install Python from python.org
        pause
        exit /b 1
    )
)

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing required packages...
    pip install -r requirements.txt
)

echo.
echo 🌐 Opening Landing Page...
start "" "%~dp0landing\index.html"

echo.
echo 🚀 Starting Streamlit Server (background)...
echo    App available at: http://localhost:8501
echo.
echo ────────────────────────────────────────────────────────────
echo    Click "ابدأ مجاناً" in the landing page to open the app
echo    Press Ctrl+C to stop the server
echo ────────────────────────────────────────────────────────────
echo.

REM Run Streamlit WITHOUT opening browser automatically
streamlit run app.py --server.headless=true --browser.gatherUsageStats=false --server.port=8501

pause
