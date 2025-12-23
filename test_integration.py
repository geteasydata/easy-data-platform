
import sys
import os

print("--- Testing Easy Data Integration ---")

# 1. Setup Paths
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"Detected Root: {root_path}")

# Add all project paths
projects = {
    "AI_Expert_App": os.path.join(root_path, "AI_Expert_App"),
    "DataAnalystProject": os.path.join(root_path, "DataAnalystProject"),
    "data_science_master_system": os.path.join(root_path, "data_science_master_system", "data_science_master_system")
}

for name, path in projects.items():
    if os.path.exists(path):
        print(f"✅ Found project: {name}")
        if path not in sys.path:
            sys.path.insert(0, path)
    else:
        print(f"❌ Missing project: {name} at {path}")

# Also add the root of data_science_master_system for its nested structure
dsms_root = os.path.join(root_path, "data_science_master_system")
if dsms_root not in sys.path:
    sys.path.insert(0, dsms_root)

# 2. Try Imports
print("\n--- Testing Imports ---")

# Test Brain (Omni-Logic)
try:
    from logic import AnalyticalLogic
    print("✅ Brain (Omni-Logic) import SUCCESS")
except Exception as e:
    print(f"❌ Brain (Omni-Logic) import FAILED: {e}")

# Test DataAnalystProject
try:
    from main import show_data_analyst_path
    print("✅ DataAnalystProject.main import SUCCESS")
except Exception as e:
    try:
        # Try just importing main directly
        import main as dap_main
        print("✅ DataAnalystProject.main import SUCCESS (direct)")
    except Exception as e2:
        print(f"❌ DataAnalystProject import FAILED: {e2}")

# Test Streamlit
try:
    import streamlit as st
    print("✅ Streamlit import SUCCESS")
except Exception as e:
    print(f"❌ Streamlit import FAILED: {e}")

print("\n--- Integration Test Finished ---")
