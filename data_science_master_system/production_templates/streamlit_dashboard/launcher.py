import os
import sys
import subprocess
import time
import webbrowser

def main():
    print("🚀 Starting Data Scientist AI App...")
    
    # Path to the app.py file
    # If frozen (exe), we need to look in the internal folder or the same dir
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
    
    app_path = os.path.join(application_path, 'app.py')
    
    # Verify app exists
    if not os.path.exists(app_path):
         # Try internal temporary dir (if --onefile)
         app_path = os.path.join(sys._MEIPASS, 'app.py') if hasattr(sys, '_MEIPASS') else app_path
    
    if not os.path.exists(app_path):
        print(f"❌ Error: Could not find app.py at {app_path}")
        time.sleep(5)
        return

    # Run Streamlit
    # We call it as a subprocess ensuring we use the same python environment logic if possible, 
    # but for a portable exe we rely on the bundled python.
    # However, since Streamlit is complex to bundle in onefile, 
    # the reliable way for this specific user env is to run the INSTALLED streamlit.
    # But if they want a standalone EXE to give to others, that's different.
    # Assuming they just want a clickable exe for THEMSELVES:
    
    print("✅ Loading Intelligent Engine...")
    cmd = ["streamlit", "run", app_path]
    
    # Run command
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error launching app: {e}")
        time.sleep(10)

if __name__ == "__main__":
    main()
