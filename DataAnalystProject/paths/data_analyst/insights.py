"""
Domain-Specific Insights Generator
Generates intelligent insights based on domain and data characteristics
Bilingual Support (English/Arabic)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# Bilingual insight texts
INSIGHT_TEXTS = {
    'en': {
        # Data Quality
        'high_missing_title': 'High Missing Data Rate',
        'high_missing_desc': 'Dataset has {pct:.1f}% missing values ({count:,} cells)',
        'high_missing_rec': 'Consider data imputation or investigate data collection issues',
        'excellent_data_title': 'Excellent Data Completeness',
        'excellent_data_desc': 'Dataset has only {pct:.2f}% missing values',
        'duplicate_title': 'Duplicate Records Detected',
        'duplicate_desc': 'Found {count:,} duplicate rows ({pct:.1f}%)',
        'duplicate_rec': 'Review and remove duplicates to ensure data integrity',
        
        # Statistical
        'skewed_title': 'Skewed Distribution: {col}',
        'skewed_desc': "Column '{col}' is {direction} skewed (skewness: {skew:.2f})",
        'skewed_rec': 'Consider log transformation for modeling',
        'positive': 'positively',
        'negative': 'negatively',
        'correlation_title': 'Strong Correlation Detected',
        'correlation_desc': "'{col1}' and '{col2}' are highly correlated (r={corr:.3f})",
        'correlation_rec': 'Consider removing one variable to avoid multicollinearity',
        
        # Time Series
        'timeseries_title': 'Time Series Data Detected',
        'timeseries_desc': 'Found {count} date column(s): {cols}',
        'timeseries_rec': 'Consider time-based analysis and forecasting',
        
        # HR Domain
        'comp_variance_title': 'Compensation Variance',
        'comp_variance_desc': 'High salary variation detected (CV: {cv:.1f}%)',
        'comp_variance_rec': 'Review pay equity across departments and roles',
        
        # Finance/Retail
        'revenue_title': 'Revenue Summary',
        'revenue_desc': 'Total revenue: ${total:,.2f}',
    },
    'ar': {
        # Data Quality
        'high_missing_title': 'نسبة عالية من البيانات المفقودة',
        'high_missing_desc': 'مجموعة البيانات تحتوي على {pct:.1f}% قيم مفقودة ({count:,} خلية)',
        'high_missing_rec': 'فكر في ملء البيانات أو التحقق من مشاكل جمع البيانات',
        'excellent_data_title': 'اكتمال ممتاز للبيانات',
        'excellent_data_desc': 'مجموعة البيانات تحتوي فقط على {pct:.2f}% قيم مفقودة',
        'duplicate_title': 'تم اكتشاف سجلات مكررة',
        'duplicate_desc': 'وُجد {count:,} صف مكرر ({pct:.1f}%)',
        'duplicate_rec': 'راجع وأزل المكررات لضمان سلامة البيانات',
        
        # Statistical
        'skewed_title': 'توزيع منحرف: {col}',
        'skewed_desc': "العمود '{col}' منحرف {direction} (الانحراف: {skew:.2f})",
        'skewed_rec': 'فكر في التحويل اللوغاريتمي للنمذجة',
        'positive': 'إيجابياً',
        'negative': 'سلبياً',
        'correlation_title': 'تم اكتشاف ارتباط قوي',
        'correlation_desc': "'{col1}' و '{col2}' مرتبطان بشكل كبير (r={corr:.3f})",
        'correlation_rec': 'فكر في إزالة أحد المتغيرات لتجنب التعدد الخطي',
        
        # Time Series
        'timeseries_title': 'تم اكتشاف بيانات سلسلة زمنية',
        'timeseries_desc': 'وُجد {count} عمود(أعمدة) تاريخ: {cols}',
        'timeseries_rec': 'فكر في التحليل الزمني والتنبؤ',
        
        # HR Domain
        'comp_variance_title': 'تباين الرواتب',
        'comp_variance_desc': 'تم اكتشاف تباين عالي في الرواتب (CV: {cv:.1f}%)',
        'comp_variance_rec': 'راجع العدالة في الرواتب عبر الأقسام والأدوار',
        
        # Finance/Retail
        'revenue_title': 'ملخص الإيرادات',
        'revenue_desc': 'إجمالي الإيرادات: ${total:,.2f}',
    }
}


def get_text(key: str, lang: str = 'en', **kwargs) -> str:
    """Get translated text with formatting"""
    text = INSIGHT_TEXTS.get(lang, INSIGHT_TEXTS['en']).get(key, key)
    try:
        return text.format(**kwargs) if kwargs else text
    except:
        return text


@dataclass
class Insight:
    """Represents a single insight"""
    title: str
    description: str
    severity: str  # info, warning, critical, success
    category: str
    metric_value: Optional[float] = None
    metric_label: Optional[str] = None
    recommendation: Optional[str] = None
    chart_type: Optional[str] = None
    chart_data: Optional[Dict] = None


class InsightsGenerator:
    """
    Domain-Specific Insights Generator
    Generates senior-analyst level insights based on data and domain
    """
    
    def __init__(self, domain: str = "custom", lang: str = "en"):
        self.domain = domain
        self.lang = lang
        self.insights: List[Insight] = []
        self.domain_configs = self._load_domain_configs()
        
    def _load_domain_configs(self) -> Dict[str, Dict]:
        """Load domain-specific configurations"""
        return {
            "hr": {
                "key_columns": ["salary", "department", "tenure", "performance", "satisfaction"],
                "metrics": {
                    "turnover_threshold": 0.15,
                    "satisfaction_target": 4.0,
                    "tenure_healthy": 3.0
                }
            },
            "healthcare": {
                "key_columns": ["diagnosis", "treatment", "outcome", "length_of_stay", "readmission"],
                "metrics": {
                    "readmission_threshold": 0.10,
                    "los_target": 5.0,
                    "mortality_threshold": 0.02
                }
            },
            "finance": {
                "key_columns": ["revenue", "expense", "profit", "transaction_amount", "category"],
                "metrics": {
                    "profit_margin_target": 0.15,
                    "expense_ratio": 0.70,
                    "growth_target": 0.10
                }
            },
            "retail": {
                "key_columns": ["sales", "quantity", "price", "customer_id", "product"],
                "metrics": {
                    "conversion_target": 0.03,
                    "aov_target": 50.0,
                    "return_threshold": 0.10
                }
            },
            "marketing": {
                "key_columns": ["campaign", "spend", "impressions", "clicks", "conversions"],
                "metrics": {
                    "ctr_target": 0.02,
                    "cpa_target": 50.0,
                    "roas_target": 3.0
                }
            },
            "education": {
                "key_columns": ["grade", "attendance", "course", "student_id", "score"],
                "metrics": {
                    "pass_rate_target": 0.80,
                    "attendance_target": 0.90,
                    "dropout_threshold": 0.05
                }
            }
        }
    
    def generate_insights(self, df: pd.DataFrame, analysis_report: Dict = None, lang: str = None) -> List[Insight]:
        """Generate domain-specific insights from data"""
        if lang:
            self.lang = lang
        self.insights = []
        
        # General data quality insights
        self._generate_data_quality_insights(df, analysis_report)
        
        # Domain-specific insights
        domain_config = self.domain_configs.get(self.domain, {})
        if domain_config:
            self._generate_domain_insights(df, domain_config)
        
        # Statistical insights
        self._generate_statistical_insights(df)
        
        # Trend insights
        self._generate_trend_insights(df)
        
        logger.info(f"Generated {len(self.insights)} insights for domain: {self.domain}")
        return self.insights
    
    def _generate_data_quality_insights(self, df: pd.DataFrame, analysis_report: Dict):
        """Generate data quality insights"""
        # Missing values insight
        total_missing = df.isnull().sum().sum()
        total_cells = df.size
        missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        
        if missing_pct > 10:
            self.insights.append(Insight(
                title=get_text('high_missing_title', self.lang),
                description=get_text('high_missing_desc', self.lang, pct=missing_pct, count=total_missing),
                severity="warning",
                category="data_quality",
                metric_value=missing_pct,
                metric_label="Missing %",
                recommendation=get_text('high_missing_rec', self.lang)
            ))
        elif missing_pct < 1:
            self.insights.append(Insight(
                title=get_text('excellent_data_title', self.lang),
                description=get_text('excellent_data_desc', self.lang, pct=missing_pct),
                severity="success",
                category="data_quality",
                metric_value=missing_pct,
                metric_label="Missing %"
            ))
        
        # Duplicate insight
        dup_count = df.duplicated().sum()
        dup_pct = (dup_count / len(df)) * 100 if len(df) > 0 else 0
        
        if dup_pct > 5:
            self.insights.append(Insight(
                title=get_text('duplicate_title', self.lang),
                description=get_text('duplicate_desc', self.lang, count=dup_count, pct=dup_pct),
                severity="warning",
                category="data_quality",
                metric_value=dup_pct,
                metric_label="Duplicate %",
                recommendation=get_text('duplicate_rec', self.lang)
            ))
    
    def _generate_domain_insights(self, df: pd.DataFrame, config: Dict):
        """Generate domain-specific insights"""
        # Generate metric-based insights
        for col in df.columns:
            col_lower = col.lower()
            
            # Salary/compensation insights for HR
            if self.domain == "hr" and any(x in col_lower for x in ["salary", "comp", "pay"]):
                if pd.api.types.is_numeric_dtype(df[col]):
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
                    
                    if cv > 30:
                        self.insights.append(Insight(
                            title=get_text('comp_variance_title', self.lang),
                            description=get_text('comp_variance_desc', self.lang, cv=cv),
                            severity="info",
                            category="hr_compensation",
                            metric_value=cv,
                            metric_label="Coefficient of Variation",
                            recommendation=get_text('comp_variance_rec', self.lang)
                        ))
            
            # Revenue insights for retail/finance
            if self.domain in ["retail", "finance"] and any(x in col_lower for x in ["revenue", "sales", "amount"]):
                if pd.api.types.is_numeric_dtype(df[col]):
                    total = df[col].sum()
                    self.insights.append(Insight(
                        title=get_text('revenue_title', self.lang),
                        description=get_text('revenue_desc', self.lang, total=total),
                        severity="info",
                        category="finance_summary",
                        metric_value=total,
                        metric_label="Total Revenue",
                        chart_type="bar"
                    ))
    
    def _generate_statistical_insights(self, df: pd.DataFrame):
        """Generate statistical insights"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            # Skewness insight
            try:
                skew = df[col].skew()
                if abs(skew) > 2:
                    direction = get_text('positive', self.lang) if skew > 0 else get_text('negative', self.lang)
                    self.insights.append(Insight(
                        title=get_text('skewed_title', self.lang, col=col),
                        description=get_text('skewed_desc', self.lang, col=col, direction=direction, skew=skew),
                        severity="info",
                        category="distribution",
                        metric_value=skew,
                        metric_label="Skewness",
                        recommendation=get_text('skewed_rec', self.lang)
                    ))
            except:
                pass
        
        # Correlation insights
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        corr = corr_matrix.loc[col1, col2]
                        if abs(corr) > 0.8:
                            self.insights.append(Insight(
                                title=get_text('correlation_title', self.lang),
                                description=get_text('correlation_desc', self.lang, col1=col1, col2=col2, corr=corr),
                                severity="info",
                                category="correlation",
                                metric_value=corr,
                                metric_label="Correlation",
                                recommendation=get_text('correlation_rec', self.lang)
                            ))
            except:
                pass
    
    def _generate_trend_insights(self, df: pd.DataFrame):
        """Generate trend-based insights"""
        # Look for date columns
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Also check object columns that might be dates
        for col in df.select_dtypes(include=['object']).columns:
            try:
                pd.to_datetime(df[col].head(100))
                date_cols.append(col)
            except:
                pass
        
        if date_cols and len(df) > 10:
            cols_str = ', '.join(date_cols[:3])
            self.insights.append(Insight(
                title=get_text('timeseries_title', self.lang),
                description=get_text('timeseries_desc', self.lang, count=len(date_cols), cols=cols_str),
                severity="info",
                category="time_series",
                recommendation=get_text('timeseries_rec', self.lang)
            ))
    
    def get_top_insights(self, n: int = 5) -> List[Insight]:
        """Get top N most important insights"""
        severity_order = {"critical": 0, "warning": 1, "success": 2, "info": 3}
        sorted_insights = sorted(
            self.insights,
            key=lambda x: severity_order.get(x.severity, 4)
        )
        return sorted_insights[:n]
    
    def get_insights_by_category(self, category: str) -> List[Insight]:
        """Get insights filtered by category"""
        return [i for i in self.insights if i.category == category]
    
    def to_dict(self) -> List[Dict]:
        """Convert insights to dictionary format"""
        return [
            {
                "title": i.title,
                "description": i.description,
                "severity": i.severity,
                "category": i.category,
                "metric_value": i.metric_value,
                "metric_label": i.metric_label,
                "recommendation": i.recommendation
            }
            for i in self.insights
        ]
