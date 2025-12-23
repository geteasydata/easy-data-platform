import sys
import os

print("Testing imports...")
brain_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_science_master_system'))
print(f"Adding path: {brain_path}")
if brain_path not in sys.path:
    sys.path.append(brain_path)

try:
    from data_science_master_system.logic import AnalyticalLogic
    print("SUCCESS: AnalyticalLogic imported.")
except Exception as e:
    print(f"FAILURE: {e}")
