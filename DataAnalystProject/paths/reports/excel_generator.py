"""
Excel Output Generator - Premium Visual Dashboard
Unified Design System (Dark Mode)
"""

import pandas as pd
import io
from typing import Dict, Any, Optional
import xlsxwriter

def create_excel_report(
    df_clean: pd.DataFrame,
    df_predictions: Optional[pd.DataFrame],
    feature_importance: pd.DataFrame,
    metrics: Dict[str, Any],
    analysis: Dict[str, Any],
    lang: str = 'ar'
) -> bytes:
    """
    Create a Board-Room ready Excel Dashboard with "Dark Mode" aesthetics.
    Compatible with Data Analyst Path.
    """
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # =========================================================================================
        # 1. THEME & FORMATTING (DARK MODE PRESET)
        # =========================================================================================
        
        # Colors
        C_BG_MAIN = '#0F172A'    # Dark Blue/Slate Background
        C_BG_CARD = '#1E293B'    # Lighter Slate for Cards
        C_TEXT_MAIN = '#F8FAFC'  # White/Light Gray Text
        C_TEXT_SUB = '#94A3B8'   # Muted Text
        C_ACCENT_1 = '#6366F1'   # Indigo (Primary)
        C_ACCENT_2 = '#2DD4BF'   # Teal (Secondary)
        C_ACCENT_3 = '#F472B6'   # Pink (Tertiary)
        C_BORDER = '#334155'     # Border Color

        # Formats
        fmt_bg_global = workbook.add_format({'bg_color': C_BG_MAIN})
        
        fmt_title = workbook.add_format({
            'bold': True, 'font_size': 20, 'font_color': C_TEXT_MAIN,
            'bg_color': C_BG_MAIN, 'valign': 'vcenter'
        })
        
        fmt_card_kpi_val = workbook.add_format({
            'bold': True, 'font_size': 18, 'font_color': C_ACCENT_2,
            'bg_color': C_BG_CARD, 'align': 'center', 'valign': 'vcenter',
            'border': 1, 'border_color': C_BORDER
        })
        
        fmt_card_kpi_label = workbook.add_format({
            'bold': False, 'font_size': 10, 'font_color': C_TEXT_SUB,
            'bg_color': C_BG_CARD, 'align': 'center', 'valign': 'top',
            'border': 1, 'border_color': C_BORDER
        })
        
        fmt_grid_header = workbook.add_format({
            'bold': True, 'font_size': 11, 'font_color': C_TEXT_MAIN,
            'bg_color': '#334155', 'border': 1, 'border_color': '#475569'
        })
        
        # =========================================================================================
        # SHEET NAMES
        # =========================================================================================
        s_dash = 'Dashboard 📊'
        s_data = 'Clean Data 🗄️'
        s_insights = 'Insights 💡'
        s_calc = 'Calculations ⚙️' # Hidden sheet for charts

        # -----------------------------------------------------------------------------------------
        # SHEET 1: CALCULATIONS (Hidden Data for Charts)
        # -----------------------------------------------------------------------------------------
        # We need this first to reference it in charts
        ws_calc = workbook.add_worksheet(s_calc)
        
        # Prepare Data for Charts
        # 1. Feature Importance (Top 10)
        if feature_importance is not None and not feature_importance.empty:
            fi_top = feature_importance.head(10).sort_values('Importance', ascending=True) # Sort for bar chart
            ws_calc.write_row('A1', ['Feature', 'Importance'])
            ws_calc.write_column('A2', fi_top['Feature'])
            ws_calc.write_column('B2', fi_top['Importance'])
        
        # 2. Metric Distribution (Dummy for distribution if needed or reuse logic)
        # For Analyst path, we might not always have a target. Let's use first numeric col distribution.
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            target_name = numeric_cols[0]
            counts, bins = pd.cut(df_clean[target_name], bins=10, retbins=True)
            dist_data = counts.value_counts().sort_index()
            bin_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
            dist_counts = dist_data.values
            
            ws_calc.write_row('D1', ['Bin', 'Count'])
            ws_calc.write_column('D2', bin_labels)
            ws_calc.write_column('E2', dist_counts)
        else:
            bin_labels = []
            dist_counts = []
        
        ws_calc.hide()

        # -----------------------------------------------------------------------------------------
        # SHEET 2: VISUAL DASHBOARD
        # -----------------------------------------------------------------------------------------
        ws_dash = workbook.add_worksheet(s_dash)
        ws_dash.set_zoom(85)
        ws_dash.hide_gridlines(2)
        
        # Set Dark Background
        ws_dash.set_column('A:Z', 2.3) # Narrow grid columns for layout
        for r in range(100):
            ws_dash.set_row(r, 15, fmt_bg_global)
            
        # --- TITLE ---
        report_title = "ANALYST INTELLIGENCE DASHBOARD" if lang == 'en' else "لوحة تحليل البيانات الذكية"
        ws_dash.merge_range('B2:Z3', report_title, fmt_title)
        
        # --- KPI CARDS (Row 5-7) ---
        # Card Helper
        def add_kpi_card(col_start, label, value, fmt_val=None):
            # 5 cols wide, 3 rows high
            c1 = col_start
            c2 = col_start + 4
            ws_dash.merge_range(4, c1, 5, c2, value, fmt_card_kpi_val)
            ws_dash.merge_range(6, c1, 6, c2, label, fmt_card_kpi_label)

        # 1. Total Records
        add_kpi_card(1, "Total Records" if lang == 'en' else "إجمالي السجلات", f"{len(df_clean):,}")
        
        # 2. Variables
        add_kpi_card(7, "Variables" if lang == 'en' else "المتغيرات", len(df_clean.columns))
        
        # 3. Missing Data %
        missing_count = df_clean.isnull().sum().sum()
        total_cells = df_clean.size
        missing_pct = (missing_count / total_cells)
        add_kpi_card(13, "Data Completeness" if lang == 'en' else "اكتمال البيانات", f"{1-missing_pct:.1%}")
        
        # 4. Processing Status
        add_kpi_card(19, "Analysis Status" if lang == 'en' else "حالة التحليل", "COMPLETE" if lang=='en' else "مكتمل")

        # --- CHARTS AREA (Starts Row 9) ---
        
        # CHART 1: Feature Importance (Bar Chart) - IF AVAILABLE
        if feature_importance is not None and not feature_importance.empty:
            chart_fi = workbook.add_chart({'type': 'bar'})
            chart_fi.add_series({
                'name': 'Impact',
                'categories': [s_calc, 1, 0, len(fi_top), 0], # B2:B11
                'values':     [s_calc, 1, 1, len(fi_top), 1], # C2:C11
                'fill':       {'color': C_ACCENT_1},
                'border':     {'none': True},
                'data_labels': {'value': True, 'font': {'color': C_TEXT_SUB}}
            })
            chart_fi.set_title({
                'name': 'Key Insights' if lang == 'en' else 'أهم الرؤى',
                'name_font': {'color': C_TEXT_MAIN, 'size': 12, 'bold': True}
            })
            chart_fi.set_chartarea({'fill': {'color': C_BG_CARD}, 'border': {'color': C_BORDER}})
            chart_fi.set_plotarea({'fill': {'color': C_BG_CARD}, 'border': {'none': True}})
            # Dark Mode Axes
            axis_font = {'name_font': {'color': C_TEXT_SUB}, 'num_font': {'color': C_TEXT_SUB}}
            chart_fi.set_x_axis({'major_gridlines': {'visible': False}, **axis_font})
            chart_fi.set_y_axis({'major_gridlines': {'visible': False}, **axis_font})
            chart_fi.set_legend({'none': True})
            
            ws_dash.insert_chart('B9', chart_fi, {'x_scale': 1.5, 'y_scale': 1.5})


        # CHART 2: Data Distribution
        if bin_labels:
            chart_dist = workbook.add_chart({'type': 'column'})
            chart_dist.add_series({
                'name': 'Count',
                'categories': [s_calc, 1, 3, 1 + len(bin_labels) -1, 3], # D2..
                'values':     [s_calc, 1, 4, 1 + len(dist_counts) -1, 4], # E2..
                'fill':       {'color': C_ACCENT_2},
                'gap':        30
            })
            chart_dist.set_title({
                'name': 'Data Distribution' if lang == 'en' else 'توزيع البيانات',
                'name_font': {'color': C_TEXT_MAIN, 'size': 12, 'bold': True}
            })
            chart_dist.set_chartarea({'fill': {'color': C_BG_CARD}, 'border': {'color': C_BORDER}})
            chart_dist.set_plotarea({'fill': {'color': C_BG_CARD}, 'border': {'none': True}})
            chart_dist.set_x_axis({'major_gridlines': {'visible': False}, **axis_font})
            chart_dist.set_y_axis({'major_gridlines': {'visible': True, 'line': {'color': '#334155'}}, **axis_font})
            chart_dist.set_legend({'none': True})
            
            ws_dash.insert_chart('J9', chart_dist, {'x_scale': 1.5, 'y_scale': 1.5})


        # -----------------------------------------------------------------------------------------
        # SHEET 3: DATA
        # -----------------------------------------------------------------------------------------
        # Write actual data
        df_clean.to_excel(writer, sheet_name=s_data, index=False)
        
        # Apply gentle formatting
        ws_data = writer.sheets[s_data]
        ws_data.set_column('A:Z', 18)
        ws_data.autofilter(0, 0, len(df_clean), len(df_clean.columns)-1)
        
        # Header Format
        for col_num, value in enumerate(df_clean.columns.values):
            ws_data.write(0, col_num, value, fmt_grid_header)


    return buffer.getvalue()
