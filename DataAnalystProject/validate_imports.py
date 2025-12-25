
import sys
import os
from pathlib import Path

# Setup paths as in main.py
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

BRAIN_PATH = Path(__file__).parent.parent / "data_science_master_system"
if str(BRAIN_PATH) not in sys.path:
    sys.path.append(str(BRAIN_PATH))

print(f"Checking imports with sys.path: {sys.path}")

try:
    print("Importing AI Ensemble...")
    from paths.ai.ai_ensemble import get_ensemble
    print("✅ AI Ensemble imported")
except ImportError as e:
    print(f"❌ AI Ensemble failed: {e}")

try:
    print("Importing Chief Data Scientist...")
    from paths.ai.chief_data_scientist import ChiefDataScientist
    print("✅ Chief Data Scientist imported")
except ImportError as e:
    print(f"❌ Chief Data Scientist failed: {e}")

try:
    print("Importing Omni-Logic...")
    from data_science_master_system.logic import AnalyticalLogic, EthicalLogic, CausalLogic
    print("✅ Omni-Logic imported")
except ImportError as e:
    print(f"❌ Omni-Logic failed: {e}")
    # List contents of brain path to debug
    if BRAIN_PATH.exists():
        print(f"Contents of {BRAIN_PATH}:")
        for p in BRAIN_PATH.iterdir():
            print(f" - {p}")
        
        # Check logic folder
        logic_path = BRAIN_PATH / "data_science_master_system" / "logic"
        if (BRAIN_PATH / "data_science_master_system").exists():
             print(f"Contents of nested data_science_master_system: {(BRAIN_PATH / 'data_science_master_system').is_dir()}")
        else:
             print("Nested data_science_master_system not found")

