"""
Auto EDA (Exploratory Data Analysis)
Comprehensive automated data analysis with one click
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')


@dataclass
class EDAReport:
    """Complete EDA report structure"""
    # Basic info
    n_rows: int = 0
    n_cols: int = 0
    memory_mb: float = 0
    
    # Data types
    numeric_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    datetime_cols: List[str] = field(default_factory=list)
    boolean_cols: List[str] = field(default_factory=list)
    
    # Quality
    missing_summary: Dict[str, float] = field(default_factory=dict)
    duplicate_rows: int = 0
    duplicate_percent: float = 0
    
    # Statistics
    numeric_stats: Dict[str, Dict] = field(default_factory=dict)
    categorical_stats: Dict[str, Dict] = field(default_factory=dict)
    
    # Correlations
    high_correlations: List[Dict] = field(default_factory=list)
    
    # Outliers
    outlier_summary: Dict[str, int] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class AutoEDA:
    """
    Automated Exploratory Data Analysis
    Generates comprehensive insights with one function call
    """
    
    def __init__(self, lang: str = 'en'):
        self.lang = lang
        self.report: EDAReport = None
        
    def analyze(self, df: pd.DataFrame) -> EDAReport:
        """
        Run complete EDA on dataframe
        
        Args:
            df: Input DataFrame
            
        Returns:
            EDAReport with all analysis results
        """
        self.report = EDAReport()
        
        # Basic info
        self._analyze_basic_info(df)
        
        # Data types
        self._analyze_data_types(df)
        
        # Quality
        self._analyze_quality(df)
        
        # Statistics
        self._analyze_statistics(df)
        
        # Correlations
        self._analyze_correlations(df)
        
        # Outliers
        self._analyze_outliers(df)
        
        # Generate recommendations
        self._generate_recommendations()
        
        return self.report
    
    def _analyze_basic_info(self, df: pd.DataFrame):
        """Basic dataset information"""
        self.report.n_rows = len(df)
        self.report.n_cols = len(df.columns)
        self.report.memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    def _analyze_data_types(self, df: pd.DataFrame):
        """Categorize columns by data type"""
        for col in df.columns:
            dtype = df[col].dtype
            
            if pd.api.types.is_bool_dtype(dtype):
                self.report.boolean_cols.append(col)
            elif pd.api.types.is_numeric_dtype(dtype):
                self.report.numeric_cols.append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                self.report.datetime_cols.append(col)
            else:
                # Check if it could be datetime
                try:
                    pd.to_datetime(df[col].head(100))
                    self.report.datetime_cols.append(col)
                except:
                    self.report.categorical_cols.append(col)
    
    def _analyze_quality(self, df: pd.DataFrame):
        """Analyze data quality"""
        # Missing values
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 0:
                self.report.missing_summary[col] = round(missing_pct, 2)
        
        # Duplicates
        self.report.duplicate_rows = df.duplicated().sum()
        self.report.duplicate_percent = round((self.report.duplicate_rows / len(df)) * 100, 2)
    
    def _analyze_statistics(self, df: pd.DataFrame):
        """Calculate statistics for all columns"""
        # Numeric columns
        for col in self.report.numeric_cols:
            try:
                series = df[col].dropna()
                self.report.numeric_stats[col] = {
                    'mean': round(float(series.mean()), 4),
                    'median': round(float(series.median()), 4),
                    'std': round(float(series.std()), 4),
                    'min': round(float(series.min()), 4),
                    'max': round(float(series.max()), 4),
                    'q25': round(float(series.quantile(0.25)), 4),
                    'q75': round(float(series.quantile(0.75)), 4),
                    'skewness': round(float(series.skew()), 4),
                    'kurtosis': round(float(series.kurtosis()), 4),
                    'zeros': int((series == 0).sum()),
                    'negatives': int((series < 0).sum())
                }
            except:
                pass
        
        # Categorical columns
        for col in self.report.categorical_cols:
            try:
                series = df[col].dropna()
                value_counts = series.value_counts()
                self.report.categorical_stats[col] = {
                    'unique': int(series.nunique()),
                    'top_value': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'top_freq': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'top_percent': round((value_counts.iloc[0] / len(series)) * 100, 2) if len(value_counts) > 0 else 0,
                    'distribution': value_counts.head(5).to_dict()
                }
            except:
                pass
    
    def _analyze_correlations(self, df: pd.DataFrame):
        """Find high correlations"""
        if len(self.report.numeric_cols) < 2:
            return
        
        try:
            corr_matrix = df[self.report.numeric_cols].corr()
            
            for i, col1 in enumerate(self.report.numeric_cols):
                for col2 in self.report.numeric_cols[i+1:]:
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > 0.7:
                        self.report.high_correlations.append({
                            'col1': col1,
                            'col2': col2,
                            'correlation': round(corr, 4),
                            'strength': 'strong' if abs(corr) > 0.9 else 'moderate'
                        })
        except:
            pass
    
    def _analyze_outliers(self, df: pd.DataFrame):
        """Detect outliers using IQR method"""
        for col in self.report.numeric_cols:
            try:
                series = df[col].dropna()
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                outliers = ((series < lower) | (series > upper)).sum()
                if outliers > 0:
                    self.report.outlier_summary[col] = int(outliers)
            except:
                pass
    
    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        recs = []
        
        # Missing values
        high_missing = [col for col, pct in self.report.missing_summary.items() if pct > 30]
        if high_missing:
            if self.lang == 'ar':
                recs.append(f"🔴 الأعمدة التالية بها نسبة مفقودة عالية (>30%): {', '.join(high_missing[:3])}")
            else:
                recs.append(f"🔴 High missing values (>30%) in: {', '.join(high_missing[:3])}")
        
        # Duplicates
        if self.report.duplicate_percent > 5:
            if self.lang == 'ar':
                recs.append(f"⚠️ توجد {self.report.duplicate_rows:,} صف مكرر ({self.report.duplicate_percent}%)")
            else:
                recs.append(f"⚠️ Found {self.report.duplicate_rows:,} duplicate rows ({self.report.duplicate_percent}%)")
        
        # High correlations
        if self.report.high_correlations:
            if self.lang == 'ar':
                recs.append(f"📊 وجد {len(self.report.high_correlations)} ارتباط قوي - فكر في إزالة التكرار")
            else:
                recs.append(f"📊 Found {len(self.report.high_correlations)} high correlations - consider feature selection")
        
        # Outliers
        if self.report.outlier_summary:
            total_outliers = sum(self.report.outlier_summary.values())
            if self.lang == 'ar':
                recs.append(f"📈 تم اكتشاف {total_outliers:,} قيمة شاذة في {len(self.report.outlier_summary)} عمود")
            else:
                recs.append(f"📈 Detected {total_outliers:,} outliers across {len(self.report.outlier_summary)} columns")
        
        # Skewed distributions
        skewed = [col for col, stats in self.report.numeric_stats.items() 
                  if abs(stats.get('skewness', 0)) > 2]
        if skewed:
            if self.lang == 'ar':
                recs.append(f"📉 أعمدة منحرفة تحتاج تحويل: {', '.join(skewed[:3])}")
            else:
                recs.append(f"📉 Skewed columns need transformation: {', '.join(skewed[:3])}")
        
        # Good quality
        if not recs:
            if self.lang == 'ar':
                recs.append("✅ جودة البيانات ممتازة!")
            else:
                recs.append("✅ Data quality looks good!")
        
        self.report.recommendations = recs
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Get summary as dictionary"""
        if not self.report:
            return {}
        
        return {
            'rows': self.report.n_rows,
            'columns': self.report.n_cols,
            'memory_mb': round(self.report.memory_mb, 2),
            'numeric_cols': len(self.report.numeric_cols),
            'categorical_cols': len(self.report.categorical_cols),
            'datetime_cols': len(self.report.datetime_cols),
            'missing_cols': len(self.report.missing_summary),
            'duplicate_rows': self.report.duplicate_rows,
            'high_correlations': len(self.report.high_correlations),
            'outlier_cols': len(self.report.outlier_summary),
            'recommendations': self.report.recommendations
        }
