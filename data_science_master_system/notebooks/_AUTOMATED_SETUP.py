"""
Automated Setup Script - Data Science Master System

Detects and installs missing packages, downloads datasets, configures kernels.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_package(package):
    """Check if package is installed."""
    try:
        __import__(package.replace('-', '_').split('[')[0])
        return True
    except ImportError:
        return False


def install_package(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])


def setup_environment():
    """Complete environment setup."""
    print("🚀 Data Science Master System - Automated Setup")
    print("=" * 50)
    
    # Core packages
    core = ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn', 'jupyter', 'jupyterlab']
    
    # ML packages
    ml = ['xgboost', 'lightgbm', 'optuna', 'joblib']
    
    # Deep learning (optional)
    dl = ['torch', 'torchvision', 'transformers']
    
    # Production
    prod = ['fastapi', 'uvicorn', 'mlflow']
    
    print("\n📦 Installing Core Packages...")
    for pkg in core:
        if not check_package(pkg):
            print(f"  Installing {pkg}...")
            try:
                install_package(pkg)
                print(f"  ✅ {pkg}")
            except:
                print(f"  ❌ Failed: {pkg}")
        else:
            print(f"  ✅ {pkg} (already installed)")
    
    print("\n📦 Installing ML Packages...")
    for pkg in ml:
        if not check_package(pkg):
            print(f"  Installing {pkg}...")
            try:
                install_package(pkg)
                print(f"  ✅ {pkg}")
            except:
                print(f"  ❌ Failed: {pkg}")
        else:
            print(f"  ✅ {pkg} (already installed)")
    
    print("\n📦 Optional - Deep Learning (may take time)...")
    response = input("Install PyTorch & Transformers? [y/N]: ")
    if response.lower() == 'y':
        for pkg in dl:
            if not check_package(pkg):
                print(f"  Installing {pkg}...")
                try:
                    install_package(pkg)
                except:
                    print(f"  ⚠️ Failed: {pkg}")
    
    # Generate sample data
    print("\n📊 Generating Sample Data...")
    data_script = Path(__file__).parent.parent / 'data' / 'generate_sample_data.py'
    if data_script.exists():
        subprocess.run([sys.executable, str(data_script)])
        print("  ✅ Sample data generated")
    
    # Install Jupyter kernel
    print("\n🔧 Setting up Jupyter Kernel...")
    try:
        subprocess.run([
            sys.executable, '-m', 'ipykernel', 'install', 
            '--user', '--name=dsms', '--display-name=Data Science Master System'
        ], check=True, capture_output=True)
        print("  ✅ Kernel installed: 'Data Science Master System'")
    except:
        print("  ⚠️ Kernel installation skipped")
    
    print("\n" + "=" * 50)
    print("✅ Setup Complete!")
    print("\nNext steps:")
    print("  1. Start Jupyter: jupyter lab")
    print("  2. Select kernel: 'Data Science Master System'")
    print("  3. Open: notebooks/00_getting_started/00_installation_setup.ipynb")


def check_gpu():
    """Check GPU availability."""
    print("\n🖥️ Hardware Detection:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ⚪ No GPU detected (CPU mode)")
    except ImportError:
        print("  ⚪ PyTorch not installed")


if __name__ == "__main__":
    setup_environment()
    check_gpu()
