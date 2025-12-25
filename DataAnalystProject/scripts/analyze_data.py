"""
Data Analysis Script
Orchestrates comprehensive data analysis
"""

import pandas as pd
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.data_analyst.analyzer import DataAnalyzer
from paths.data_analyst.insights import InsightsGenerator
from scripts.load_data import DataLoader


def analyze_data(input_file: str, domain: str = 'custom', 
                 output_dir: str = None):
    """
    Perform comprehensive data analysis
    
    Args:
        input_file: Path to input data file
        domain: Business domain for insights
        output_dir: Directory for output reports
    """
    # Load data
    loader = DataLoader()
    df = loader.load(input_file)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Analyze data
    print("\n" + "="*50)
    print("📊 DATA QUALITY ANALYSIS")
    print("="*50)
    
    analyzer = DataAnalyzer()
    report = analyzer.analyze(df)
    
    # Print summary
    summary = analyzer.get_summary()
    print(f"\n📋 Overview:")
    print(f"   Rows: {summary['rows']:,}")
    print(f"   Columns: {summary['columns']}")
    print(f"   Missing: {summary['missing_percent']}%")
    print(f"   Duplicates: {summary['duplicate_rows']:,}")
    print(f"   Numeric columns: {summary['numeric_columns']}")
    print(f"   Categorical columns: {summary['categorical_columns']}")
    
    # Print issues
    if report.data_issues:
        print(f"\n⚠️ Data Issues Found ({len(report.data_issues)}):")
        for issue in report.data_issues[:5]:
            icon = "🔴" if issue['severity'] == 'critical' else "🟡" if issue['severity'] == 'warning' else "ℹ️"
            print(f"   {icon} {issue['message']}")
    
    # Print correlations
    if report.strong_correlations:
        print(f"\n🔗 Strong Correlations:")
        for col1, col2, corr in report.strong_correlations[:5]:
            print(f"   {col1} ↔ {col2}: {corr}")
    
    # Generate insights
    print("\n" + "="*50)
    print("💡 DOMAIN INSIGHTS")
    print("="*50)
    
    insights_gen = InsightsGenerator(domain=domain)
    insights = insights_gen.generate_insights(df, summary)
    
    for insight in insights_gen.get_top_insights(5):
        icon = "✅" if insight.severity == 'success' else "⚠️" if insight.severity == 'warning' else "🔴" if insight.severity == 'critical' else "ℹ️"
        print(f"\n   {icon} {insight.title}")
        print(f"      {insight.description}")
        if insight.recommendation:
            print(f"      💡 {insight.recommendation}")
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save analysis report
        report_data = {
            'summary': summary,
            'issues': report.data_issues,
            'correlations': report.strong_correlations,
            'insights': insights_gen.to_dict()
        }
        
        with open(output_path / f"analysis_{timestamp}.json", 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: {output_path}")
    
    return report, insights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze data file')
    parser.add_argument('--file', '-f', required=True, help='Input data file')
    parser.add_argument('--domain', '-d', default='custom',
                        choices=['hr', 'finance', 'healthcare', 'retail', 
                                'marketing', 'education', 'custom'],
                        help='Business domain')
    parser.add_argument('--output', '-o', help='Output directory')
    
    args = parser.parse_args()
    analyze_data(args.file, args.domain, args.output)
