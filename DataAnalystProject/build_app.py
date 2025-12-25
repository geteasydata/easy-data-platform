
import PyInstaller.__main__
import os
import shutil
from pathlib import Path





# Define paths
BASE_DIR = Path(__file__).parent
OUTPUT_NAME = "DataScienceHub_v3"
BRAIN_PATH = BASE_DIR.parent / "data_science_master_system"

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

print(f"Building {OUTPUT_NAME} from {BASE_DIR}")
print(f"Including Brain Path: {BRAIN_PATH}")

# Collect all sklearn submodules
sklearn_hidden_imports = collect_submodules('sklearn')

# Collect xgboost binaries and data (SAFER METHOD)
# collect_all was pulling in test files that crashed the build
xgboost_binaries = collect_dynamic_libs('xgboost')
xgboost_datas = collect_data_files('xgboost')
xgboost_hidden_imports = ['xgboost', 'xgboost.core', 'xgboost.training', 'xgboost.sklearn']

# List hidden imports
hidden_imports = [
    'streamlit',
    'pandas',
    'numpy',
    'plotly',
    'altgraph', 
    'scipy',
    'sklearn',
    'sklearn.utils._cython_blas', 
    'sklearn.neighbors.typedefs',
    'sklearn.neighbors.quad_tree',
    'sklearn.tree._utils',
    'joblib',
    'google.generativeai',
    'groq',
    'requests',
    'optuna',
    'statsmodels',
    'pathlib',
    # Add project specific modules
    'config',
    'translations',
    'scripts.load_data',
    'paths',
    'paths.ai.chief_data_scientist',
    'paths.ai.ai_ensemble',
    # Add Omni-Logic
    'data_science_master_system.logic'
] + sklearn_hidden_imports + xgboost_hidden_imports

# Define data files to include
datas = [
    # Include the entire project source as data for Streamlit to read
    (str(BASE_DIR / "main.py"), "."),
    (str(BASE_DIR / "config.py"), "."),
    (str(BASE_DIR / "translations.py"), "."),
    (str(BASE_DIR / "paths"), "paths"),
    (str(BASE_DIR / "scripts"), "scripts"),
    (str(BASE_DIR / "domains"), "domains"),
    (str(BASE_DIR / "outputs"), "outputs"),
    (str(BASE_DIR / "templates"), "templates"),
    # Include Omni-Logic as package source
    (str(BRAIN_PATH / "data_science_master_system"), "data_science_master_system"),
] + collect_data_files('sklearn') + xgboost_datas


# Clean previous build
if (BASE_DIR / "build").exists():
    shutil.rmtree(str(BASE_DIR / "build"))
# Do not clean dist to preserve previous builds

# Run PyInstaller
PyInstaller.__main__.run([
    'run_exe.py',                     # Script to run
    f'--name={OUTPUT_NAME}',          # Executable name
    '--onefile',                      # Create a single executable file
    '--clean',                        # Clean cache
    '--noconfirm',                    # Do not ask for confirmation
    '--windowed',                     # Hide console (remove this if you want to see errors)
    # '--console',                    # Use console for debugging 
    
    # Add hidden imports
    *[f'--hidden-import={mod}' for mod in hidden_imports],
    

    # Add data files
    *[f'--add-data={src}{os.pathsep}{dst}' for src, dst in datas],

    # Add binaries (specifically for xgboost)
    *[f'--add-binary={src}{os.pathsep}{dst}' for src, dst in xgboost_binaries],
    
    # Streamlit specific
    '--copy-metadata=streamlit',
    '--copy-metadata=tqdm',
    '--copy-metadata=regex',
    '--copy-metadata=requests',
    '--copy-metadata=packaging',
    '--collect-all=streamlit',
    '--collect-all=altair',
    '--collect-all=plotly',
])

print("\n✅ Build complete!")
print(f"Executable created at: {BASE_DIR / 'dist' / (OUTPUT_NAME + '.exe')}")
