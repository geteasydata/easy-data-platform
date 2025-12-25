"""
Dashboard Generator Module
Generates dashboards for Jupyter, Excel, and Power BI
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """
    Multi-format Dashboard Generator
    Creates dashboards for Jupyter Notebooks, Excel, and Power BI
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("outputs/dashboards")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_jupyter_notebook(self, df: pd.DataFrame, analysis: Dict, 
                                   domain: str = "custom", 
                                   filename: str = None) -> str:
        """Generate Jupyter Notebook with analysis and visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"analysis_{domain}_{timestamp}.ipynb"
        
        cells = []
        
        # Title cell
        cells.append(self._create_markdown_cell(f"""# Data Analysis Report - {domain.upper()} Domain
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Dataset Overview:**
- Rows: {len(df):,}
- Columns: {len(df.columns)}
"""))
        
        # Import cell
        cells.append(self._create_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
"""))
        
        # Load data cell
        cells.append(self._create_markdown_cell("## 1. Data Loading"))
        cells.append(self._create_code_cell("""# Load your data
# df = pd.read_csv('your_data.csv')

# For this analysis, data is pre-loaded
print(f"Dataset Shape: {df.shape}")
df.head()
"""))
        
        # Data overview cell
        cells.append(self._create_markdown_cell("## 2. Data Overview"))
        cells.append(self._create_code_cell("""# Data types and info
print("Data Types:")
print(df.dtypes)
print("\\n" + "="*50 + "\\n")
print("Summary Statistics:")
df.describe(include='all').T
"""))
        
        # Missing values analysis
        cells.append(self._create_markdown_cell("## 3. Missing Values Analysis"))
        cells.append(self._create_code_cell("""# Missing values heatmap
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)

if len(missing_df) > 0:
    fig = px.bar(missing_df, x=missing_df.index, y='Missing %', 
                 title='Missing Values by Column',
                 color='Missing %', color_continuous_scale='Reds')
    fig.show()
    print(missing_df)
else:
    print("No missing values found!")
"""))
        
        # Distribution plots
        cells.append(self._create_markdown_cell("## 4. Distribution Analysis"))
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
        
        if numeric_cols:
            cells.append(self._create_code_cell(f"""# Distribution of numeric columns
numeric_cols = {numeric_cols}

fig = make_subplots(rows=2, cols=2, subplot_titles=numeric_cols[:4])
for i, col in enumerate(numeric_cols[:4]):
    row = i // 2 + 1
    col_idx = i % 2 + 1
    fig.add_trace(go.Histogram(x=df[col], name=col, nbinsx=30), row=row, col=col_idx)

fig.update_layout(height=600, title_text="Distribution of Numeric Variables", showlegend=False)
fig.show()
"""))
        
        # Correlation analysis
        cells.append(self._create_markdown_cell("## 5. Correlation Analysis"))
        cells.append(self._create_code_cell("""# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
if len(numeric_df.columns) > 1:
    corr_matrix = numeric_df.corr()
    
    fig = px.imshow(corr_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title='Correlation Matrix')
    fig.show()
    
    # Strong correlations
    print("\\nStrong Correlations (|r| > 0.7):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                print(f"  {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")
"""))
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()[:4]
        if categorical_cols:
            cells.append(self._create_markdown_cell("## 6. Categorical Analysis"))
            cells.append(self._create_code_cell(f"""# Top categories distribution
categorical_cols = {categorical_cols}

for col in categorical_cols[:4]:
    value_counts = df[col].value_counts().head(10)
    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                 title=f'Top 10 Categories in {{col}}',
                 labels={{'x': col, 'y': 'Count'}})
    fig.show()
"""))
        
        # Summary and insights
        cells.append(self._create_markdown_cell("""## 7. Key Insights

Based on the analysis above, here are the key findings:

1. **Data Quality**: Review missing values and duplicates
2. **Distributions**: Check for skewness and outliers
3. **Correlations**: Consider multicollinearity for modeling
4. **Categories**: Balance of categorical variables

### Recommendations:
- Handle missing values appropriately
- Consider transformations for skewed distributions
- Remove or engineer highly correlated features
- Balance categorical variables if needed for modeling
"""))
        
        # Build notebook structure
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.9.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        # Save notebook
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        
        logger.info(f"Generated Jupyter notebook: {output_path}")
        return str(output_path)
    
    def _create_markdown_cell(self, content: str) -> Dict:
        """Create a markdown cell"""
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": content.split('\n')
        }
    
    def _create_code_cell(self, content: str) -> Dict:
        """Create a code cell"""
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": content.split('\n')
        }
    
    def generate_excel_dashboard_template(self, df: pd.DataFrame, analysis: Dict,
                                           domain: str = "custom",
                                           filename: str = None) -> str:
        """Generate Excel dashboard using the Premium Engine"""
        try:
            # Import the premium generator internally to avoid circular imports
            from paths.reports.excel_generator import create_excel_report
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filename or f"dashboard_{domain}_{timestamp}.xlsx"
            output_path = self.output_dir / filename
            
            # Prepare arguments for the premium generator
            # We map the available data to the expected format
            metrics = {
                'rows': len(df),
                'columns': len(df.columns),
                'missing': df.isnull().sum().sum(),
                'duplicates': df.duplicated().sum()
            }
            
            # Generate bytes using the premium engine
            excel_bytes = create_excel_report(
                df_clean=df,
                df_predictions=None,
                feature_importance=None, # Feature importance not available in this flow yet
                metrics=metrics,
                analysis=analysis,
                lang='en' # Default to EN or pass lang if available
            )
            
            # Write bytes to file
            with open(output_path, 'wb') as f:
                f.write(excel_bytes)
                
            logger.info(f"Generated Premium Excel dashboard: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Premium generation failed, falling back to basic: {e}")
            # Fallback logic could go here, or just raise
            raise e
    
    def _generate_excel_instructions(self, df: pd.DataFrame, analysis: Dict, domain: str) -> str:
        """Generate Excel dashboard creation instructions"""
        instructions = f"""
# Excel Dashboard Instructions - {domain.upper()} Domain

## Data Preparation
1. Open Excel and create a new workbook
2. Rename Sheet1 to "Data"
3. Import your data into the Data sheet

## Dashboard Layout (Sheet2 - Dashboard)

### Row 1: Title
- Cell A1: "{domain.upper()} Analytics Dashboard"
- Merge A1:H1, Font: 20pt Bold

### Row 3-5: Key Metrics Cards
Create 4 metric cards across columns A-H:
- Total Records: {len(df):,}
- Total Columns: {len(df.columns)}
- Missing Values: {df.isnull().sum().sum():,}
- Duplicate Rows: {df.duplicated().sum():,}

### Row 7-20: Charts Section
1. Column Distribution (Bar Chart)
2. Category Breakdown (Pie Chart)
3. Trend Line (if date column exists)

## Pivot Tables (Sheet3)
1. Insert > PivotTable
2. Select Data range
3. Create summary by main categories

## Recommended Visualizations
- Bar chart for categorical counts
- Line chart for time series
- Scatter plot for correlations
"""
        output_path = self.output_dir / f"excel_instructions_{domain}.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(instructions)
        return str(output_path)
    
    def generate_powerbi_template(self, df: pd.DataFrame, analysis: Dict,
                                   domain: str = "custom") -> str:
        """Generate Power BI template with DAX measures"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate DAX measures
        dax_measures = []
        
        # Basic measures
        dax_measures.append(f"""
// Total Records
Total Records = COUNTROWS('{domain}Data')
""")
        
        # Numeric column measures
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols[:5]:
            safe_col = col.replace(' ', '_')
            dax_measures.append(f"""
// Average {col}
Avg {safe_col} = AVERAGE('{domain}Data'[{col}])

// Sum {col}  
Total {safe_col} = SUM('{domain}Data'[{col}])

// YTD {col}
YTD {safe_col} = TOTALYTD(SUM('{domain}Data'[{col}]), 'Calendar'[Date])
""")
        
        # Generate Power Query M code
        m_code = f"""
// Power Query M Code for Data Loading
let
    Source = Csv.Document(File.Contents("data.csv"), [Delimiter=",", Encoding=65001]),
    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers", {{
"""
        
        for col in df.columns[:10]:
            dtype = str(df[col].dtype)
            pq_type = "type text"
            if 'int' in dtype or 'float' in dtype:
                pq_type = "type number"
            elif 'datetime' in dtype:
                pq_type = "type datetime"
            m_code += f'        {{"{col}", {pq_type}}},\n'
        
        m_code += """    })
in
    #"Changed Type"
"""
        
        # Save template files
        output_dir = self.output_dir / f"powerbi_{domain}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "dax_measures.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(dax_measures))
        
        with open(output_dir / "power_query.m", 'w', encoding='utf-8') as f:
            f.write(m_code)
        
        # Generate report layout JSON
        report_layout = {
            "pages": [
                {
                    "name": "Overview",
                    "displayName": f"{domain.upper()} Overview",
                    "visuals": [
                        {"type": "card", "title": "Total Records", "measure": "Total Records"},
                        {"type": "barChart", "title": "Category Distribution"},
                        {"type": "lineChart", "title": "Trend Over Time"},
                        {"type": "pieChart", "title": "Breakdown"}
                    ]
                },
                {
                    "name": "Details",
                    "displayName": "Detailed Analysis",
                    "visuals": [
                        {"type": "table", "title": "Data Table"},
                        {"type": "matrix", "title": "Summary Matrix"}
                    ]
                }
            ]
        }
        
        with open(output_dir / "report_layout.json", 'w', encoding='utf-8') as f:
            json.dump(report_layout, f, indent=2)
        
        logger.info(f"Generated Power BI template: {output_dir}")
        return str(output_dir)
    
    def generate_all(self, df: pd.DataFrame, analysis: Dict, 
                     domain: str = "custom") -> Dict[str, str]:
        """Generate all dashboard formats"""
        results = {}
        
        try:
            results["jupyter"] = self.generate_jupyter_notebook(df, analysis, domain)
        except Exception as e:
            logger.error(f"Jupyter generation failed: {e}")
            results["jupyter"] = None
        
        try:
            results["excel"] = self.generate_excel_dashboard_template(df, analysis, domain)
        except Exception as e:
            logger.error(f"Excel generation failed: {e}")
            results["excel"] = None
        
        try:
            results["powerbi"] = self.generate_powerbi_template(df, analysis, domain)
        except Exception as e:
            logger.error(f"Power BI generation failed: {e}")
            results["powerbi"] = None
        
        return results
