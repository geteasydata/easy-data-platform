import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Path: {sys.path[0]}")

try:
    print("Importing core.exceptions...")
    from data_science_master_system.core import exceptions
    print("Importing core.base_classes...")
    from data_science_master_system.core import base_classes
    print("Importing titanic_features...")
    from data_science_master_system.features.engineering.titanic_features import TitanicFeatureGenerator
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
