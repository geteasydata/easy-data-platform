"""
Data Cleaning Script
Orchestrates data cleaning operations
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paths.data_analyst.cleaner import DataCleaner, CleaningTool
from scripts.load_data import DataLoader


def clean_data(input_file: str, output_file: str = None, 
               tool: str = 'python', operations: list = None):
    """
    Clean data using specified tool and operations
    
    Args:
        input_file: Path to input data file
        output_file: Path for cleaned output file
        tool: Cleaning tool ('python', 'excel', 'powerbi')
        operations: List of operations to perform
    """
    # Load data
    loader = DataLoader()
    df = loader.load(input_file)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Initialize cleaner
    tool_map = {
        'python': CleaningTool.PYTHON,
        'excel': CleaningTool.EXCEL,
        'powerbi': CleaningTool.POWERBI
    }
    cleaner = DataCleaner(tool=tool_map.get(tool, CleaningTool.PYTHON))
    
    # Default operations
    if operations is None:
        operations = ['missing_values', 'duplicates', 'outliers']
    
    # Clean data
    cleaned_df = cleaner.clean(df, operations)
    
    # Get summary
    summary = cleaner.get_summary()
    print(f"\nCleaning Summary:")
    print(f"  Operations performed: {summary['total_operations']}")
    print(f"  Rows: {summary['rows_before']} -> {summary['rows_after']}")
    print(f"  Columns: {summary['columns_before']} -> {summary['columns_after']}")
    
    # Print generated code
    print(f"\n{'='*50}")
    print(f"Generated {tool.upper()} Code:")
    print('='*50)
    print(cleaner.get_code())
    
    # Save cleaned data (for Python tool)
    if output_file and tool == 'python':
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_file, index=False)
        print(f"\nCleaned data saved to: {output_file}")
    
    return cleaned_df, cleaner.get_code()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean data file')
    parser.add_argument('--file', '-f', required=True, help='Input data file')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--tool', '-t', default='python', 
                        choices=['python', 'excel', 'powerbi'],
                        help='Cleaning tool to use')
    parser.add_argument('--operations', '-ops', nargs='+',
                        default=['missing_values', 'duplicates', 'outliers'],
                        help='Cleaning operations to perform')
    
    args = parser.parse_args()
    clean_data(args.file, args.output, args.tool, args.operations)
