
import streamlit.web.cli as stcli
import os, sys
from pathlib import Path

def resolve_path(path):
    if getattr(sys, "frozen", False):
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
    return os.path.join(basedir, path)

if __name__ == "__main__":
    # Ensure all imports from current directory work
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Locate main.py
    main_script = resolve_path("main.py")
    
    # Set arguments for streamlit run
    sys.argv = [
        "streamlit",
        "run",
        main_script,
        "--global.developmentMode=false",
    ]
    
    print(f"Starting Data Science Hub from: {main_script}")
    sys.exit(stcli.main())
