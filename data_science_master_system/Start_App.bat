@echo off
TITLE Omni-Logic AI Data Scientist
COLOR 0B
echo ==================================================
echo      STARTING OMNI-LOGIC DASHBOARD
echo ==================================================
echo.
echo Launching Web Interface...
cd /d "%~dp0"
streamlit run dashboard.py --browser.gatherUsageStats=false --server.headless=false
pause
