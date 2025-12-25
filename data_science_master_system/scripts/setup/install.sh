#!/bin/bash
# Data Science Master System - Installation Script

echo "🚀 Installing Data Science Master System..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Generate sample data
echo "📊 Generating sample data..."
python data/generate_sample_data.py

# Setup Jupyter kernel
python -m ipykernel install --user --name=dsms --display-name="Data Science Master System"

echo "✅ Installation complete!"
echo ""
echo "To get started:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Start Jupyter: jupyter lab"
echo "  3. Open: notebooks/00_getting_started/00_installation_setup.ipynb"
