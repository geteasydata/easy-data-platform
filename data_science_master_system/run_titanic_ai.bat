@echo off
TITLE Omni-Logic Engine - Titanic Optimizer
COLOR 0A
echo ==================================================
echo      INITIALIZING OMNI-LOGIC ENGINE (8 LAYERS)
echo ==================================================
echo.
echo Loading Python Environment...
cd /d "%~dp0"

echo Running Optimization Pipeline...
python scripts/optimize_titanic_final.py

echo.
echo ==================================================
echo      MISSION COMPLETE
echo ==================================================
pause
