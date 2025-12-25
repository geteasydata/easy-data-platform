@echo off
title Data Science Hub - Expert AI Edition
color 0A

echo ========================================================
echo   Data Science Hub - Ultimate Edition (Analyst + Scientist)
echo ========================================================
echo.

cd /d "%~dp0"

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] First time setup: Installing dependencies...
    pip install streamlit pandas numpy plotly scikit-learn openpyxl python-docx reportlab shap optuna prophet google-generativeai openai -q
) else (
    echo [INFO] Dependencies verified.
)

echo.
echo [INFO] Starting the AI Engine...
echo.

streamlit run main.py --server.port 8501 --server.address localhost --theme.base "dark"

pause
