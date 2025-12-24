"""
Power BI Export Generator - Prepares data for seamless Power BI Import
"""

import pandas as pd
import io
import zipfile
import json
from typing import Dict, Any

def create_powerbi_package(
    df_clean: pd.DataFrame,
    df_predictions: pd.DataFrame = None,
    analysis: Dict[str, Any] = None,
    lang: str = 'ar'
) -> bytes:
    """
    Create a professional Enterprise ZIP for Power BI.
    """
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Main Data
        data_filename = 'Enterprise_Data.csv'
        zip_file.writestr(data_filename, df_clean.to_csv(index=False))
        
        # 2. DAX MEASURES (THE EXPERT TOUCH)
        dax_content = """
-- Suggested Measures for Power BI
Total Records = COUNTROWS('Enterprise_Data')
Unique Values Count = DISTINCTCOUNT('Enterprise_Data'[ID]) -- Update ID to your key
        """
        if 'target_col' in analysis:
            target = analysis['target_col']
            dax_content += f"\nTarget Mean = AVERAGE('Enterprise_Data'[{target}])"
            dax_content += f"\nTarget Variance = VARX.P('Enterprise_Data', 'Enterprise_Data'[{target}])"
            
        zip_file.writestr('DAX_Measures.txt', dax_content.strip())
        
        # 3. RELATIONSHIP SCHEMA
        schema = {
            "version": "1.0",
            "recommended_relationships": [
                {"from": "Enterprise_Data", "to": "DateTable", "on": "DateColumn"},
                {"type": "Star Schema Recommended"}
            ],
            "visual_recommendations": [
                {"chart": "Slicer", "field": "Categorical Columns"},
                {"chart": "Line Chart", "field": "Time Components"}
            ]
        }
        zip_file.writestr('BI_Schema_Map.json', json.dumps(schema, indent=4))

        # 4. ADVANCED README
        instructions = "1. Load Enterprise_Data.csv\n2. Open DAX_Measures.txt to copy-paste core calculations."
        zip_file.writestr('INSTRUCTIONS_PRO.txt', instructions)

    return buffer.getvalue()
