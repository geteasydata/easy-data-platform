"""
PDF Report Generator
Professional PDF reports with Arabic/English support
"""

import io
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, ListFlowable, ListItem
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


@dataclass
class ReportConfig:
    """Report configuration"""
    title: str = "Data Analysis Report"
    subtitle: str = ""
    author: str = "Data Science Hub"
    lang: str = "en"
    include_charts: bool = True
    include_recommendations: bool = True


class PDFReportGenerator:
    """
    Professional PDF Report Generator
    
    Features:
    - Bilingual support (Arabic/English)
    - Executive summary
    - Statistical tables
    - Charts and visualizations
    - Recommendations section
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = None
        
        if HAS_REPORTLAB:
            self.styles = getSampleStyleSheet()
            self._setup_styles()
    
    def _setup_styles(self):
        """Setup custom styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1a1a2e'),
            alignment=1  # Center
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            spaceAfter=20,
            textColor=colors.HexColor('#667eea'),
            alignment=1
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#16213e')
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=10,
            leading=14
        ))
    
    def generate_report(self,
                        data_summary: Dict[str, Any],
                        insights: List[Dict],
                        statistics: Dict[str, Any],
                        config: ReportConfig = None) -> str:
        """
        Generate PDF report
        
        Args:
            data_summary: Summary of dataset
            insights: List of insights
            statistics: Statistical summaries
            config: Report configuration
            
        Returns:
            Path to generated PDF
        """
        if not HAS_REPORTLAB:
            raise ImportError("reportlab is required for PDF generation")
        
        if config is None:
            config = ReportConfig()
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"report_{timestamp}.pdf"
        filepath = self.output_dir / filename
        
        # Create document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build content
        elements = []
        
        # Title page
        elements.extend(self._create_title_page(config))
        
        # Executive summary
        elements.extend(self._create_executive_summary(data_summary, config))
        
        # Data overview
        elements.extend(self._create_data_overview(data_summary, config))
        
        # Statistics section
        elements.extend(self._create_statistics_section(statistics, config))
        
        # Insights section
        if config.include_recommendations:
            elements.extend(self._create_insights_section(insights, config))
        
        # Build PDF
        doc.build(elements)
        
        return str(filepath)
    
    def _create_title_page(self, config: ReportConfig) -> List:
        """Create title page"""
        elements = []
        
        # Add spacing
        elements.append(Spacer(1, 2*inch))
        
        # Title
        elements.append(Paragraph(config.title, self.styles['CustomTitle']))
        
        # Subtitle
        if config.subtitle:
            elements.append(Paragraph(config.subtitle, self.styles['CustomSubtitle']))
        
        # Spacing
        elements.append(Spacer(1, inch))
        
        # Date and author
        date_str = datetime.now().strftime('%Y-%m-%d')
        if config.lang == 'ar':
            meta_text = f"التاريخ: {date_str}<br/>المؤلف: {config.author}"
        else:
            meta_text = f"Date: {date_str}<br/>Author: {config.author}"
        
        elements.append(Paragraph(meta_text, self.styles['BodyText']))
        
        # Page break
        elements.append(PageBreak())
        
        return elements
    
    def _create_executive_summary(self, data_summary: Dict, config: ReportConfig) -> List:
        """Create executive summary section"""
        elements = []
        
        # Section header
        header = "الملخص التنفيذي" if config.lang == 'ar' else "Executive Summary"
        elements.append(Paragraph(header, self.styles['SectionHeader']))
        
        # Summary text
        rows = data_summary.get('rows', 0)
        cols = data_summary.get('columns', 0)
        missing = data_summary.get('missing_percent', 0)
        
        if config.lang == 'ar':
            summary = f"""
            تم تحليل مجموعة بيانات تحتوي على {rows:,} صف و {cols} عمود.
            نسبة البيانات المفقودة: {missing}%.
            """
        else:
            summary = f"""
            Analyzed a dataset containing {rows:,} rows and {cols} columns.
            Missing data percentage: {missing}%.
            """
        
        elements.append(Paragraph(summary, self.styles['BodyText']))
        elements.append(Spacer(1, 0.5*inch))
        
        return elements
    
    def _create_data_overview(self, data_summary: Dict, config: ReportConfig) -> List:
        """Create data overview section"""
        elements = []
        
        # Section header
        header = "نظرة عامة على البيانات" if config.lang == 'ar' else "Data Overview"
        elements.append(Paragraph(header, self.styles['SectionHeader']))
        
        # Create table
        if config.lang == 'ar':
            headers = ['المقياس', 'القيمة']
            data = [
                headers,
                ['إجمالي الصفوف', f"{data_summary.get('rows', 0):,}"],
                ['إجمالي الأعمدة', str(data_summary.get('columns', 0))],
                ['الأعمدة الرقمية', str(data_summary.get('numeric_cols', 0))],
                ['الأعمدة النصية', str(data_summary.get('categorical_cols', 0))],
                ['نسبة المفقود', f"{data_summary.get('missing_percent', 0)}%"],
                ['الصفوف المكررة', f"{data_summary.get('duplicate_rows', 0):,}"]
            ]
        else:
            headers = ['Metric', 'Value']
            data = [
                headers,
                ['Total Rows', f"{data_summary.get('rows', 0):,}"],
                ['Total Columns', str(data_summary.get('columns', 0))],
                ['Numeric Columns', str(data_summary.get('numeric_cols', 0))],
                ['Categorical Columns', str(data_summary.get('categorical_cols', 0))],
                ['Missing %', f"{data_summary.get('missing_percent', 0)}%"],
                ['Duplicate Rows', f"{data_summary.get('duplicate_rows', 0):,}"]
            ]
        
        table = Table(data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6'))
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.5*inch))
        
        return elements
    
    def _create_statistics_section(self, statistics: Dict, config: ReportConfig) -> List:
        """Create statistics section"""
        elements = []
        
        # Section header
        header = "الإحصائيات" if config.lang == 'ar' else "Statistics"
        elements.append(Paragraph(header, self.styles['SectionHeader']))
        
        if not statistics:
            no_stats = "لا توجد إحصائيات" if config.lang == 'ar' else "No statistics available"
            elements.append(Paragraph(no_stats, self.styles['BodyText']))
            return elements
        
        # Create table for numeric stats
        for col_name, stats in list(statistics.items())[:5]:
            elements.append(Paragraph(f"<b>{col_name}</b>", self.styles['BodyText']))
            
            if config.lang == 'ar':
                data = [
                    ['المتوسط', 'الوسيط', 'الانحراف', 'الأدنى', 'الأقصى'],
                    [
                        str(stats.get('mean', '-')),
                        str(stats.get('median', '-')),
                        str(stats.get('std', '-')),
                        str(stats.get('min', '-')),
                        str(stats.get('max', '-'))
                    ]
                ]
            else:
                data = [
                    ['Mean', 'Median', 'Std', 'Min', 'Max'],
                    [
                        str(stats.get('mean', '-')),
                        str(stats.get('median', '-')),
                        str(stats.get('std', '-')),
                        str(stats.get('min', '-')),
                        str(stats.get('max', '-'))
                    ]
                ]
            
            table = Table(data, colWidths=[1.2*inch]*5)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9ecef')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6'))
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 0.25*inch))
        
        return elements
    
    def _create_insights_section(self, insights: List[Dict], config: ReportConfig) -> List:
        """Create insights section"""
        elements = []
        
        # Section header
        header = "الرؤى والتوصيات" if config.lang == 'ar' else "Insights & Recommendations"
        elements.append(Paragraph(header, self.styles['SectionHeader']))
        
        if not insights:
            no_insights = "لا توجد رؤى" if config.lang == 'ar' else "No insights available"
            elements.append(Paragraph(no_insights, self.styles['BodyText']))
            return elements
        
        # Add insights
        for i, insight in enumerate(insights[:10], 1):
            title = insight.get('title', '')
            description = insight.get('description', '')
            recommendation = insight.get('recommendation', '')
            severity = insight.get('severity', 'info')
            
            # Icon based on severity
            icon = '✅' if severity == 'success' else '⚠️' if severity == 'warning' else '🔴' if severity == 'critical' else 'ℹ️'
            
            text = f"<b>{i}. {icon} {title}</b><br/>{description}"
            if recommendation:
                rec_label = "التوصية" if config.lang == 'ar' else "Recommendation"
                text += f"<br/><i>💡 {rec_label}: {recommendation}</i>"
            
            elements.append(Paragraph(text, self.styles['BodyText']))
            elements.append(Spacer(1, 0.15*inch))
        
        return elements


def generate_quick_pdf(data_summary: Dict, 
                       insights: List, 
                       output_path: str = None,
                       lang: str = 'en') -> str:
    """Quick function to generate PDF report"""
    
    generator = PDFReportGenerator()
    
    config = ReportConfig(
        title="تقرير تحليل البيانات" if lang == 'ar' else "Data Analysis Report",
        lang=lang
    )
    
    # Convert insights to dict format if needed
    insights_dicts = []
    for insight in insights:
        if hasattr(insight, 'title'):
            insights_dicts.append({
                'title': insight.title,
                'description': insight.description,
                'severity': insight.severity,
                'recommendation': getattr(insight, 'recommendation', None)
            })
        else:
            insights_dicts.append(insight)
    
    return generator.generate_report(
        data_summary=data_summary,
        insights=insights_dicts,
        statistics={},
        config=config
    )
