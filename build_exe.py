"""
Build EXE - Create standalone Windows executable
Run this script to create AI_Expert.exe
"""

import subprocess
import sys
import shutil
import os
from pathlib import Path


def build_exe():
    """Build the executable using PyInstaller."""
    
    print("=" * 60)
    print("🔨 Building AI Expert Executable")
    print("=" * 60)
    
    script_dir = Path(__file__).parent.absolute()
    
    # Build-time dependencies for XGBoost collection
    for pkg in ["pyinstaller", "hypothesis", "pytest"]:
        try:
            __import__(pkg)
            print(f"✅ {pkg} found")
        except ImportError:
            print(f"📦 Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)
    
    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",  # Create a directory with exe and dependencies
        "--name", "Easy_Data", # Unified Project Name
        "--icon", "NONE",
        
        # Bundle all 3 Core Folders (Relative to build_exe.py in AI_Expert_App)
        "--add-data", f"app.py{os.pathsep}.",
        "--add-data", f"core{os.pathsep}core",
        "--add-data", f"reports{os.pathsep}reports",
        "--add-data", f"translations.py{os.pathsep}.",
        "--add-data", f"..{os.sep}DataAnalystProject{os.pathsep}DataAnalystProject",
        "--add-data", f"..{os.sep}data_science_master_system{os.pathsep}data_science_master_system",
        
        # Hidden Imports and Collection
        "--hidden-import", "streamlit",
        "--hidden-import", "sklearn",
        "--hidden-import", "pandas",
        "--hidden-import", "numpy",
        "--hidden-import", "plotly",
        "--hidden-import", "xlsxwriter",
        "--hidden-import", "xgboost",
        "--collect-all", "streamlit",
        "--collect-all", "plotly",
        "--collect-all", "sklearn",
        "--collect-all", "xgboost",
        "--collect-all", "pandas",
        
        "--clean",
        "launcher.py"
    ]
    
    print("\n🔧 Running PyInstaller...")
    print("   This may take 5-15 minutes...\n")
    
    try:
        result = subprocess.run(cmd, cwd=str(script_dir), check=True)
        print("\n" + "=" * 60)
        print("✅ Build successful!")
        print(f"📁 Executable created in: {script_dir / 'dist' / 'Easy_Data'}")
        print("=" * 60)
        
        # Create a run.bat in the dist folder
        run_bat = script_dir / "dist" / "Easy_Data" / "Run_Easy_Data.bat"
        with open(run_bat, 'w') as f:
            f.write('@echo off\n')
            f.write('title AI Expert - Data Scientist\n')
            f.write('echo Starting Easy Data...\n')
            f.write('Easy_Data.exe\n')
            f.write('pause\n')
        
        print("\n📌 How to use:")
        print(f"   1. Go to: {script_dir / 'dist' / 'Easy_Data'}")
        print("   2. Double-click 'Run_Easy_Data.bat' or 'Easy_Data.exe'")
        print("   3. Your browser will open automatically")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed: {e}")
        print("\n💡 Alternative: Use Start_AI_Expert.bat to run without building")
        return False
    
    return True


def build_simple():
    """Simpler build using batch file wrapper."""
    
    print("=" * 60)
    print("🔨 Creating Simple Launcher")
    print("=" * 60)
    
    script_dir = Path(__file__).parent.absolute()
    
    # Create a comprehensive batch launcher instead
    bat_content = '''@echo off
setlocal enabledelayedexpansion

title AI Expert - Data Scientist
color 0A

echo.
echo  ================================================
echo     AI Expert - Data Scientist
echo     خبير البيانات الذكي
echo  ================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python from: https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo [OK] Python found
echo.

REM Check if dependencies are installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [INFO] First time setup - Installing dependencies...
    echo        This may take a few minutes...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo.
    echo [OK] Dependencies installed
)

echo [INFO] Starting AI Expert...
echo        Your browser will open automatically.
echo.
echo [HINT] To stop the server, close this window or press Ctrl+C
echo.

python launcher.py

pause
'''
    
    launcher_bat = script_dir / "AI_Expert_Launcher.bat"
    with open(launcher_bat, 'w', encoding='utf-8') as f:
        f.write(bat_content)
    
    print(f"✅ Created: {launcher_bat}")
    print("\n📌 To run the application:")
    print(f"   Double-click: {launcher_bat}")
    
    return True


if __name__ == "__main__":
    print("\n🤖 Easy Data - Build Tool\n")
    # Automatically running option 1 (Build standalone EXE)
    build_exe()
