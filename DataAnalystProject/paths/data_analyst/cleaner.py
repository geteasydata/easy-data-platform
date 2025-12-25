"""
Data Cleaner Module
Multi-tool data cleaning with Python, Excel Power Query, and Power BI code generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class CleaningTool(Enum):
    PYTHON = "python"
    EXCEL = "excel"
    POWERBI = "powerbi"


@dataclass
class CleaningOperation:
    """Represents a single cleaning operation"""
    operation_type: str
    column: Optional[str]
    parameters: Dict[str, Any]
    python_code: str
    excel_code: str
    powerbi_code: str
    description: str


class DataCleaner:
    """
    Multi-tool Data Cleaner
    Generates cleaning code for Python, Excel Power Query, and Power BI
    """
    
    def __init__(self, tool: CleaningTool = CleaningTool.PYTHON):
        self.tool = tool
        self.operations: List[CleaningOperation] = []
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        
    def set_tool(self, tool: CleaningTool):
        """Switch cleaning tool"""
        self.tool = tool
        logger.info(f"Switched cleaning tool to: {tool.value}")
        
    def clean(self, df: pd.DataFrame, operations: List[str] = None) -> pd.DataFrame:
        """
        Apply cleaning operations to dataframe
        Returns cleaned dataframe (for Python) or generates code (for Excel/PowerBI)
        """
        self.original_df = df.copy()
        self.cleaned_df = df.copy()
        self.operations = []
        
        if operations is None:
            operations = ['missing_values', 'duplicates', 'outliers', 'normalize']
        
        for op in operations:
            if op == 'missing_values':
                self._handle_missing_values()
            elif op == 'duplicates':
                self._remove_duplicates()
            elif op == 'outliers':
                self._handle_outliers()
            elif op == 'normalize':
                self._normalize_numeric()
            elif op == 'encode':
                self._encode_categorical()
            elif op == 'standardize':
                self._standardize_numeric()
        
        return self.cleaned_df
    
    def _handle_missing_values(self):
        """Handle missing values with multiple strategies"""
        df = self.cleaned_df
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
                
            missing_pct = missing_count / len(df)
            
            # Determine strategy based on data type and missing percentage
            if missing_pct > 0.5:
                strategy = "drop_column"
            elif pd.api.types.is_numeric_dtype(df[col]):
                strategy = "median"
            else:
                strategy = "mode"
            
            # Generate code for each tool
            python_code = ""
            excel_code = ""
            powerbi_code = ""
            
            if strategy == "drop_column":
                python_code = f"df = df.drop(columns=['{col}'])"
                excel_code = f'= Table.RemoveColumns(Source, {{"{col}"}})'
                powerbi_code = f'= Table.RemoveColumns(Source, {{"{col}"}})'
                if self.tool == CleaningTool.PYTHON:
                    self.cleaned_df = self.cleaned_df.drop(columns=[col])
            elif strategy == "median":
                median_val = df[col].median()
                python_code = f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())"
                excel_code = f'= Table.ReplaceValue(Source, null, List.Median(Source[{col}]), Replacer.ReplaceValue, {{"{col}"}})'
                powerbi_code = f'{col} Filled = IF(ISBLANK([{col}]), MEDIAN(\'{col}\'), [{col}])'
                if self.tool == CleaningTool.PYTHON:
                    self.cleaned_df[col] = self.cleaned_df[col].fillna(median_val)
            elif strategy == "mode":
                mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else ""
                python_code = f"df['{col}'] = df['{col}'].fillna(df['{col}'].mode().iloc[0])"
                excel_code = f'= Table.ReplaceValue(Source, null, "{mode_val}", Replacer.ReplaceValue, {{"{col}"}})'
                powerbi_code = f'{col} Filled = IF(ISBLANK([{col}]), "{mode_val}", [{col}])'
                if self.tool == CleaningTool.PYTHON:
                    self.cleaned_df[col] = self.cleaned_df[col].fillna(mode_val)
            
            self.operations.append(CleaningOperation(
                operation_type="handle_missing",
                column=col,
                parameters={"strategy": strategy, "missing_count": missing_count},
                python_code=python_code,
                excel_code=excel_code,
                powerbi_code=powerbi_code,
                description=f"Handle {missing_count} missing values in '{col}' using {strategy}"
            ))
    
    def _remove_duplicates(self):
        """Remove duplicate rows"""
        df = self.cleaned_df
        duplicate_count = df.duplicated().sum()
        
        if duplicate_count == 0:
            return
        
        python_code = "df = df.drop_duplicates()"
        excel_code = "= Table.Distinct(Source)"
        powerbi_code = "= Table.Distinct(Source)"
        
        if self.tool == CleaningTool.PYTHON:
            self.cleaned_df = self.cleaned_df.drop_duplicates()
        
        self.operations.append(CleaningOperation(
            operation_type="remove_duplicates",
            column=None,
            parameters={"duplicate_count": duplicate_count},
            python_code=python_code,
            excel_code=excel_code,
            powerbi_code=powerbi_code,
            description=f"Remove {duplicate_count} duplicate rows"
        ))
    
    def _handle_outliers(self):
        """Handle outliers using IQR method"""
        df = self.cleaned_df
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            if len(outliers) == 0:
                continue
            
            # Cap outliers instead of removing
            python_code = f"""df['{col}'] = df['{col}'].clip(lower={lower:.2f}, upper={upper:.2f})"""
            excel_code = f"""= Table.TransformColumns(Source, {{{{"{col}", each if _ < {lower:.2f} then {lower:.2f} else if _ > {upper:.2f} then {upper:.2f} else _}}}})"""
            powerbi_code = f"""{col} Capped = IF([{col}] < {lower:.2f}, {lower:.2f}, IF([{col}] > {upper:.2f}, {upper:.2f}, [{col}]))"""
            
            if self.tool == CleaningTool.PYTHON:
                self.cleaned_df[col] = self.cleaned_df[col].clip(lower=lower, upper=upper)
            
            self.operations.append(CleaningOperation(
                operation_type="handle_outliers",
                column=col,
                parameters={"lower_bound": lower, "upper_bound": upper, "outlier_count": len(outliers)},
                python_code=python_code,
                excel_code=excel_code,
                powerbi_code=powerbi_code,
                description=f"Cap {len(outliers)} outliers in '{col}' to range [{lower:.2f}, {upper:.2f}]"
            ))
    
    def _normalize_numeric(self):
        """Normalize numeric columns to 0-1 range"""
        df = self.cleaned_df
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            
            if min_val == max_val:
                continue
            
            python_code = f"df['{col}_normalized'] = (df['{col}'] - {min_val}) / ({max_val} - {min_val})"
            excel_code = f"""= Table.AddColumn(Source, "{col}_normalized", each ([{col}] - {min_val}) / ({max_val} - {min_val}))"""
            powerbi_code = f"""{col} Normalized = ([{col}] - {min_val}) / ({max_val} - {min_val})"""
            
            if self.tool == CleaningTool.PYTHON:
                self.cleaned_df[f'{col}_normalized'] = (self.cleaned_df[col] - min_val) / (max_val - min_val)
            
            self.operations.append(CleaningOperation(
                operation_type="normalize",
                column=col,
                parameters={"min": min_val, "max": max_val},
                python_code=python_code,
                excel_code=excel_code,
                powerbi_code=powerbi_code,
                description=f"Normalize '{col}' to 0-1 range"
            ))
    
    def _standardize_numeric(self):
        """Standardize numeric columns (z-score)"""
        df = self.cleaned_df
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val == 0:
                continue
            
            python_code = f"df['{col}_standardized'] = (df['{col}'] - {mean_val:.4f}) / {std_val:.4f}"
            excel_code = f"""= Table.AddColumn(Source, "{col}_standardized", each ([{col}] - {mean_val:.4f}) / {std_val:.4f})"""
            powerbi_code = f"""{col} Standardized = ([{col}] - {mean_val:.4f}) / {std_val:.4f}"""
            
            if self.tool == CleaningTool.PYTHON:
                self.cleaned_df[f'{col}_standardized'] = (self.cleaned_df[col] - mean_val) / std_val
            
            self.operations.append(CleaningOperation(
                operation_type="standardize",
                column=col,
                parameters={"mean": mean_val, "std": std_val},
                python_code=python_code,
                excel_code=excel_code,
                powerbi_code=powerbi_code,
                description=f"Standardize '{col}' (mean={mean_val:.2f}, std={std_val:.2f})"
            ))
    
    def _encode_categorical(self):
        """Encode categorical variables"""
        df = self.cleaned_df
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_vals = df[col].dropna().unique()
            
            if len(unique_vals) > 10:
                continue  # Skip high cardinality columns
            
            encoding_map = {val: i for i, val in enumerate(unique_vals)}
            
            python_code = f"df['{col}_encoded'] = df['{col}'].map({encoding_map})"
            excel_code = self._generate_excel_encoding(col, encoding_map)
            powerbi_code = self._generate_powerbi_encoding(col, encoding_map)
            
            if self.tool == CleaningTool.PYTHON:
                self.cleaned_df[f'{col}_encoded'] = self.cleaned_df[col].map(encoding_map)
            
            self.operations.append(CleaningOperation(
                operation_type="encode_categorical",
                column=col,
                parameters={"encoding_map": encoding_map},
                python_code=python_code,
                excel_code=excel_code,
                powerbi_code=powerbi_code,
                description=f"Label encode '{col}' ({len(unique_vals)} categories)"
            ))
    
    def _generate_excel_encoding(self, col: str, encoding_map: Dict) -> str:
        """Generate Excel Power Query code for encoding"""
        conditions = []
        for val, code in encoding_map.items():
            conditions.append(f'if [{{column}}] = "{val}" then {code}')
        return f"""= Table.AddColumn(Source, "{col}_encoded", each {" else ".join(conditions)} else null)""".replace("{column}", col)
    
    def _generate_powerbi_encoding(self, col: str, encoding_map: Dict) -> str:
        """Generate Power BI DAX code for encoding"""
        conditions = []
        for val, code in encoding_map.items():
            conditions.append(f'[{col}] = "{val}", {code}')
        return f"""{col} Encoded = SWITCH(TRUE(), {", ".join(conditions)})"""
    
    def get_code(self, tool: CleaningTool = None) -> str:
        """Get all cleaning code for specified tool"""
        tool = tool or self.tool
        
        code_lines = []
        if tool == CleaningTool.PYTHON:
            code_lines.append("import pandas as pd")
            code_lines.append("import numpy as np")
            code_lines.append("")
            code_lines.append("# Data Cleaning Operations")
            for op in self.operations:
                code_lines.append(f"\n# {op.description}")
                code_lines.append(op.python_code)
        elif tool == CleaningTool.EXCEL:
            code_lines.append("// Power Query M Code")
            code_lines.append("let")
            code_lines.append('    Source = Excel.CurrentWorkbook(){[Name="Data"]}[Content],')
            for i, op in enumerate(self.operations):
                step_name = f"Step{i+1}"
                code_lines.append(f"    // {op.description}")
                code_lines.append(f"    {step_name} {op.excel_code},")
            code_lines.append("    Result = Step" + str(len(self.operations)))
            code_lines.append("in")
            code_lines.append("    Result")
        elif tool == CleaningTool.POWERBI:
            code_lines.append("// Power BI DAX Measures and Calculated Columns")
            for op in self.operations:
                code_lines.append(f"\n// {op.description}")
                code_lines.append(op.powerbi_code)
        
        return "\n".join(code_lines)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of cleaning operations"""
        return {
            "total_operations": len(self.operations),
            "operations": [
                {
                    "type": op.operation_type,
                    "column": op.column,
                    "description": op.description
                }
                for op in self.operations
            ],
            "rows_before": len(self.original_df) if self.original_df is not None else 0,
            "rows_after": len(self.cleaned_df) if self.cleaned_df is not None else 0,
            "columns_before": len(self.original_df.columns) if self.original_df is not None else 0,
            "columns_after": len(self.cleaned_df.columns) if self.cleaned_df is not None else 0
        }
