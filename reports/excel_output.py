"""
Excel Output Generator - Professional Visual Dashboard
"""

import pandas as pd
import io
from typing import Dict, Any, Optional
import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell

def create_excel_report(
    df_clean: pd.DataFrame,
    df_predictions: Optional[pd.DataFrame],
    feature_importance: pd.DataFrame,
    metrics: Dict[str, Any],
    insights: str,
    lang: str = 'ar'
) -> bytes:
    """
    Create a Board-Room ready Excel Dashboard with "Dark Mode" aesthetics.
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
        
        fmt_grid_data = workbook.add_format({
            'font_size': 10, 'border': 1, 'border_color': '#E2E8F0' # Light border for data sheet
        })

        # =========================================================================================
        # SHEET NAMES
        # =========================================================================================
        s_dash = 'Dashboard 📊'
        s_data = 'Data 🗄️'
        s_insights = 'Insights 💡'
        s_calc = 'Calculations ⚙️' # Hidden sheet for charts

        # -----------------------------------------------------------------------------------------
        # SHEET 1: CALCULATIONS (Hidden Data for Charts)
        # -----------------------------------------------------------------------------------------
        # We need this first to reference it in charts
        ws_calc = workbook.add_worksheet(s_calc)
        
        # Prepare Data for Charts
        # 1. Feature Importance (Top 10)
        fi_top = feature_importance.head(10).sort_values('Importance', ascending=True) # Sort for bar chart
        ws_calc.write_row('A1', ['Feature', 'Importance'])
        ws_calc.write_column('A2', fi_top['Feature'])
        ws_calc.write_column('B2', fi_top['Importance'])
        
        # 2. Target Distribution (Numerical or Categorical)
        # Simple Histogram bins or Category Counts
        target_name = df_clean.columns[-1] # Assuming last col is target (rough heuristic) from flow
        
        if pd.api.types.is_numeric_dtype(df_clean[target_name]):
            # Create Histogram Bins
            counts, bins = pd.cut(df_clean[target_name], bins=10, retbins=True)
            dist_data = counts.value_counts().sort_index()
            # Format bins as strings "10-20"
            bin_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
            dist_counts = dist_data.values
        else:
            # Categorical Counts
            dist_vc = df_clean[target_name].value_counts().head(10)
            bin_labels = dist_vc.index.astype(str).tolist()
            dist_counts = dist_vc.values
            
        ws_calc.write_row('D1', ['Bin', 'Count'])
        ws_calc.write_column('D2', bin_labels)
        ws_calc.write_column('E2', dist_counts)
        
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
        report_title = "AI EXECUTIVE DASHBOARD" if lang == 'en' else "لوحة القيادة التنفيذية الذكية"
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
        
        # 3. Model Accuracy (if exists)
        acc_val = metrics.get('accuracy', metrics.get('r2', 0))
        acc_str = f"{acc_val:.1%}" if 'accuracy' in metrics else f"{acc_val:.2f}"
        add_kpi_card(13, "Model Confidence" if lang == 'en' else "دقة النموذج", acc_str)
        
        # 4. Impact (Sum of top 3 features importance)
        top_imp = feature_importance['Importance'].nlargest(3).sum() if not feature_importance.empty else 0
        add_kpi_card(19, "Top Drivers Impact" if lang == 'en' else "تأثير العوامل الرئيسية", f"{top_imp:.1%}")

        # --- CHARTS AREA (Starts Row 9) ---
        
        # CHART 1: Feature Importance (Bar Chart)
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
            'name': 'Top Decision Drivers' if lang == 'en' else 'أهم عوامل التأثير',
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


        # CHART 2: Target Distribution (Donut/Pie or Column)
        chart_dist = workbook.add_chart({'type': 'column'})
        chart_dist.add_series({
            'name': 'Count',
            'categories': [s_calc, 1, 3, 1 + len(bin_labels) -1, 3], # D2..
            'values':     [s_calc, 1, 4, 1 + len(dist_counts) -1, 4], # E2..
            'fill':       {'color': C_ACCENT_2},
            'gap':        30
        })
        chart_dist.set_title({
            'name': 'Outcome Distribution' if lang == 'en' else 'توزيع النتائج',
            'name_font': {'color': C_TEXT_MAIN, 'size': 12, 'bold': True}
        })
        chart_dist.set_chartarea({'fill': {'color': C_BG_CARD}, 'border': {'color': C_BORDER}})
        chart_dist.set_plotarea({'fill': {'color': C_BG_CARD}, 'border': {'none': True}})
        chart_dist.set_x_axis({'major_gridlines': {'visible': False}, **axis_font})
        chart_dist.set_y_axis({'major_gridlines': {'visible': True, 'line': {'color': '#334155'}}, **axis_font})
        chart_dist.set_legend({'none': True})
        
        ws_dash.insert_chart('J9', chart_dist, {'x_scale': 1.5, 'y_scale': 1.5})


        # --- TEXT INSIGHTS BOX ---
        # Merge range below charts
        ws_dash.merge_range('B26:Q32', insights, workbook.add_format({
            'bg_color': C_BG_CARD, 'font_color': C_TEXT_MAIN,
            'text_wrap': True, 'valign': 'top', 'border': 1, 'border_color': C_BORDER,
            'font_size': 11
        }))
        ws_dash.write('B25', "Strategic Insights" if lang=='en' else "الرؤى الاستراتيجية", 
                      workbook.add_format({'font_color': C_ACCENT_3, 'bold': True, 'bg_color': C_BG_MAIN, 'font_size': 12}))


        # -----------------------------------------------------------------------------------------
        # SHEET 3: PROCESSED DATA
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

        # -----------------------------------------------------------------------------------------
        # SHEET 4: STRATEGIC INSIGHTS TEXT
        # -----------------------------------------------------------------------------------------
        ws_txt = workbook.add_worksheet(s_insights)
        ws_txt.hide_gridlines(2)
        ws_txt.set_column('A:A', 5)
        ws_txt.set_column('B:B', 100)
        
        ws_txt.write('B2', "Detailed AI Analysis", workbook.add_format({'bold': True, 'font_size': 16}))
        ws_txt.write('B4', insights, workbook.add_format({'text_wrap': True, 'font_size': 12}))


    return buffer.getvalue()
