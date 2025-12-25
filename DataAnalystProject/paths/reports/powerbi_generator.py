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
    Create a ZIP file containing:
    1. Cleaned Data (CSV)
    2. Predictions (CSV)
    3. Metadata/Instructions (JSON/Text)
    
    Args:
        df_clean: The main cleaned dataset
        df_predictions: Predictions dataframe
        analysis: Analysis metadata
        lang: Language ('ar' or 'en')
        
    Returns:
        Bytes of the ZIP file
    """
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Main Data
        data_filename = 'clean_data.csv'
        zip_file.writestr(data_filename, df_clean.to_csv(index=False))
        
        # 2. Predictions (if any)
        if df_predictions is not None:
            pred_filename = 'predictions.csv'
            zip_file.writestr(pred_filename, df_predictions.to_csv(index=False))
            
        # 3. Instructions / Metadata
        if lang == 'ar':
            read_me = f"""
# تعليمات الاستيراد إلى Power BI

1. قم بفك ضغط هذا الملف.
2. افتح Power BI Desktop.
3. اختر 'Get Data' ثم 'Text/CSV'.
4. اختر ملف '{data_filename}'.
5. اضغط 'Load'.

تم تجهيز هذه البيانات وتنظيفها آلياً لتكون جاهزة للتحليل المباشر.
            """
        else:
            read_me = f"""
# Power BI Import Instructions

1. Unzip this file.
2. Open Power BI Desktop.
3. Select 'Get Data' -> 'Text/CSV'.
4. Select '{data_filename}'.
5. Click 'Load'.

This data has been auto-cleaned and is ready for immediate dashboarding.
            """
            
        zip_file.writestr('README_PowerBI.txt', read_me.strip())
        
        # 4. Metadata (for advanced users)
        if analysis:
            meta = {
                'generated_by': 'Data Science Hub',
                'columns': list(df_clean.columns),
                'rows': len(df_clean),
                'analysis_summary': analysis.get('overview', {})
            }
            zip_file.writestr('metadata.json', json.dumps(meta, indent=2))
            
    return buffer.getvalue()
