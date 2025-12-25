"""
Dashboard Generation Script
Creates dashboards in various formats
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.data_analyst.dashboard_gen import DashboardGenerator
from paths.data_analyst.analyzer import DataAnalyzer
from scripts.load_data import DataLoader


def generate_dashboard(input_file: str, format: str = 'all',
                       domain: str = 'custom', output_dir: str = None):
    """
    Generate dashboards from data
    
    Args:
        input_file: Path to input data file
        format: Output format ('jupyter', 'excel', 'powerbi', 'all')
        domain: Business domain
        output_dir: Output directory
    """
    # Load data
    loader = DataLoader()
    df = loader.load(input_file)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Analyze data first
    analyzer = DataAnalyzer()
    report = analyzer.analyze(df)
    analysis = analyzer.get_summary()
    
    # Set output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path(__file__).parent.parent / "outputs" / "dashboards"
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate dashboards
    generator = DashboardGenerator(output_dir=output_path)
    
    print(f"\n📊 Generating dashboards for {domain} domain...")
    
    if format == 'all':
        results = generator.generate_all(df, analysis, domain)
        for fmt, path in results.items():
            if path:
                print(f"   ✅ {fmt.upper()}: {path}")
            else:
                print(f"   ❌ {fmt.upper()}: Failed to generate")
    else:
        if format == 'jupyter':
            path = generator.generate_jupyter_notebook(df, analysis, domain)
        elif format == 'excel':
            path = generator.generate_excel_dashboard_template(df, analysis, domain)
        elif format == 'powerbi':
            path = generator.generate_powerbi_template(df, analysis, domain)
        
        print(f"   ✅ {format.upper()}: {path}")
    
    print(f"\n📁 All outputs saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate dashboards')
    parser.add_argument('--file', '-f', required=True, help='Input data file')
    parser.add_argument('--format', '-fmt', default='all',
                        choices=['jupyter', 'excel', 'powerbi', 'all'],
                        help='Dashboard format')
    parser.add_argument('--domain', '-d', default='custom', help='Business domain')
    parser.add_argument('--output', '-o', help='Output directory')
    
    args = parser.parse_args()
    generate_dashboard(args.file, args.format, args.domain, args.output)
