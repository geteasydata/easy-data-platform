"""
PDF Report Generator - Professional PDF Reports
"""

from typing import Dict, Any, List, Optional
import io
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def create_pdf_report(
    analysis: Dict[str, Any],
    results: Dict[str, Any],
    target_col: str,
    insights: str,
    lang: str = 'ar'
) -> Optional[bytes]:
    """
    Create a professional PDF report.
    
    Args:
        analysis: Data analysis dictionary
        results: ML results dictionary
        target_col: Target column name
        insights: AI-generated insights
        lang: Language ('ar' or 'en')
    
    Returns:
        PDF file as bytes, or None if reportlab not available
    """
    if not HAS_REPORTLAB:
        return None
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2E86AB')
    )
    
    # Heading style
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#2E86AB')
    )
    
    # Normal style
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        leading=16
    )
    
    # Build content
    story = []
    
    # Title
    if lang == 'ar':
        story.append(Paragraph("AI Expert - Data Science Report", title_style))
        story.append(Paragraph(f"Target: {target_col}", normal_style))
    else:
        story.append(Paragraph("AI Expert - Data Science Report", title_style))
        story.append(Paragraph(f"Target Variable: {target_col}", normal_style))
    
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Spacer(1, 20))
    
    # Data Summary Section
    if lang == 'ar':
        story.append(Paragraph("1. Data Summary", heading_style))
    else:
        story.append(Paragraph("1. Data Summary", heading_style))
    
    # Handle nested analysis structure
    overview = analysis.get('overview', analysis)
    cols_val = overview.get('columns', analysis.get('columns', 0))
    cols_count = len(cols_val) if isinstance(cols_val, dict) else cols_val
    
    summary_data = [
        ['Metric', 'Value'],
        ['Rows', str(overview.get('rows', 0))],
        ['Columns', str(cols_count)],
        ['Numeric Columns', str(len(analysis.get('numeric_columns', [])))],
        ['Categorical Columns', str(len(analysis.get('categorical_columns', [])))],
        ['Missing Values', str(overview.get('total_missing', 0))],
        ['Duplicates', str(overview.get('duplicates', 0))],
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
        ('GRID', (0, 0), (-1, -1), 1, colors.white),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Model Results Section
    if lang == 'ar':
        story.append(Paragraph("2. Model Performance", heading_style))
    else:
        story.append(Paragraph("2. Model Performance", heading_style))
    
    story.append(Paragraph(f"Problem Type: {results['problem_type'].title()}", normal_style))
    story.append(Paragraph(f"Best Model: {results['best_model']}", normal_style))
    story.append(Spacer(1, 10))
    
    metrics_data = [['Metric', 'Value']]
    for key, value in results['metrics'].items():
        if isinstance(value, float):
            metrics_data.append([key.replace('_', ' ').title(), f"{value:.4f}"])
        else:
            metrics_data.append([key.replace('_', ' ').title(), str(value)])
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A3BE8C')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
        ('GRID', (0, 0), (-1, -1), 1, colors.white),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # Feature Importance Section
    if lang == 'ar':
        story.append(Paragraph("3. Feature Importance", heading_style))
    else:
        story.append(Paragraph("3. Feature Importance", heading_style))
    
    importance_data = [['Rank', 'Feature', 'Importance']]
    for idx, row in results['feature_importance'].head(10).iterrows():
        importance_data.append([
            str(idx + 1),
            str(row['Feature']),
            f"{row['Importance']:.4f}"
        ])
    
    importance_table = Table(importance_data, colWidths=[0.8*inch, 3*inch, 1.5*inch])
    importance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#BF616A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
        ('GRID', (0, 0), (-1, -1), 1, colors.white),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    story.append(importance_table)
    story.append(Spacer(1, 20))
    
    # AI Insights Section
    if lang == 'ar':
        story.append(Paragraph("4. AI Expert Insights", heading_style))
    else:
        story.append(Paragraph("4. AI Expert Insights", heading_style))
    
    # Clean insights for PDF
    clean_insights = insights.replace('**', '').replace('*', '').replace('#', '')
    for line in clean_insights.split('\n'):
        if line.strip():
            story.append(Paragraph(line.strip(), normal_style))
    story.append(Spacer(1, 20))
    
    # Cleaning Steps
    if lang == 'ar':
        story.append(Paragraph("5. Data Cleaning Steps", heading_style))
    else:
        story.append(Paragraph("5. Data Cleaning Steps", heading_style))
    
    for step in results['cleaning_steps']:
        clean_step = step.replace('✅', '[OK]').replace('🗑️', '[DEL]').replace('🔧', '[FIX]').replace('🏷️', '[ENC]').replace('🎯', '[TGT]').replace('✨', '[OK]')
        story.append(Paragraph(f"• {clean_step}", normal_style))
    
    # Build PDF
    doc.build(story)
    
    return buffer.getvalue()
