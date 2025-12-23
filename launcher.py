"""
AI Expert Launcher - Opens the app in browser (PyInstaller Friendly)
"""

import sys
import os
import streamlit.web.cli as stcli
from pathlib import Path

def main():
    """Launch the Streamlit app in a way that works for PyInstaller."""
    
    # 1. Determine the path to app.py
    if getattr(sys, 'frozen', False):
        # Running as compiled EXE
        # The app.py will be in the _internal folder (sys._MEIPASS) or same dir
        base_dir = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(sys.executable).parent
    else:
        # Running as script
        base_dir = Path(__file__).parent.absolute()
        
    app_path = base_dir / "app.py"
    
    print(f"🚀 Launching from: {base_dir}")
    print(f"📄 App path: {app_path}")
    
    if not app_path.exists():
        # Fallback: Check if it's in the current directory
        fallback = Path.cwd() / "app.py"
        if fallback.exists():
            app_path = fallback
            print(f"⚠️  Found app.py in CWD: {app_path}")
        else:
            print("❌ Error: app.py not found in bundle!")
            input("Press Enter to exit...")
            sys.exit(1)

    # 2. Configure Streamlit arguments
    # We manipulate sys.argv to mimic "streamlit run app.py"
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode=false",
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
        "--theme.primaryColor=#667eea",
        "--theme.backgroundColor=#ffffff",
        "--theme.secondaryBackgroundColor=#f8f9fa",
        "--theme.textColor=#2d3436"
    ]
    
    print("🌟 Starting Streamlit Server...")
    print("   Close this window to stop the application.")
    
    # 3. Invoke Streamlit CLI directly
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
