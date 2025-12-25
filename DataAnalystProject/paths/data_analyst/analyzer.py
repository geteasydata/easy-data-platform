"""
Data Analyzer Module
Comprehensive data analysis with missing values, outliers, correlations, and statistics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import logging
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ColumnAnalysis:
    """Analysis results for a single column"""
    name: str
    dtype: str
    missing_count: int
    missing_percent: float
    unique_count: int
    unique_percent: float
    sample_values: List[Any]
    
    # Numeric stats
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    q1: Optional[float] = None
    q3: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    outliers_count: int = 0
    outliers_percent: float = 0.0
    
    # Categorical stats
    top_values: Dict[str, int] = field(default_factory=dict)
    value_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Overall data quality report"""
    total_rows: int
    total_columns: int
    total_missing: int
    missing_percent: float
    duplicate_rows: int
    duplicate_percent: float
    memory_usage: str
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    columns_analysis: Dict[str, ColumnAnalysis] = field(default_factory=dict)
    correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    strong_correlations: List[Tuple[str, str, float]] = field(default_factory=list)
    data_issues: List[Dict[str, Any]] = field(default_factory=list)


class DataAnalyzer:
    """
    Comprehensive Data Analyzer
    Analyzes data for quality, patterns, and insights
    """
    
    def __init__(self, 
                 missing_warning_threshold: float = 0.05,
                 missing_critical_threshold: float = 0.30,
                 outlier_iqr_multiplier: float = 1.5,
                 correlation_threshold: float = 0.7):
        self.missing_warning = missing_warning_threshold
        self.missing_critical = missing_critical_threshold
        self.iqr_multiplier = outlier_iqr_multiplier
        self.correlation_threshold = correlation_threshold
        self.report: Optional[DataQualityReport] = None
        
    def analyze(self, df: pd.DataFrame) -> DataQualityReport:
        """Perform comprehensive data analysis"""
        logger.info(f"Starting analysis on dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Basic stats
        total_rows = len(df)
        total_columns = len(df.columns)
        total_missing = df.isnull().sum().sum()
        missing_percent = (total_missing / (total_rows * total_columns)) * 100
        duplicate_rows = df.duplicated().sum()
        duplicate_percent = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
        
        # Memory usage
        memory_bytes = df.memory_usage(deep=True).sum()
        if memory_bytes < 1024:
            memory_usage = f"{memory_bytes} B"
        elif memory_bytes < 1024**2:
            memory_usage = f"{memory_bytes/1024:.2f} KB"
        elif memory_bytes < 1024**3:
            memory_usage = f"{memory_bytes/1024**2:.2f} MB"
        else:
            memory_usage = f"{memory_bytes/1024**3:.2f} GB"
        
        # Column type separation
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Analyze each column
        columns_analysis = {}
        for col in df.columns:
            columns_analysis[col] = self._analyze_column(df[col])
        
        # Correlation analysis
        correlations = {}
        strong_correlations = []
        if len(numeric_columns) > 1:
            corr_matrix = df[numeric_columns].corr()
            correlations = corr_matrix.to_dict()
            
            # Find strong correlations
            for i, col1 in enumerate(numeric_columns):
                for col2 in numeric_columns[i+1:]:
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) >= self.correlation_threshold:
                        strong_correlations.append((col1, col2, round(corr_value, 3)))
        
        # Identify data issues
        data_issues = self._identify_issues(df, columns_analysis)
        
        self.report = DataQualityReport(
            total_rows=total_rows,
            total_columns=total_columns,
            total_missing=total_missing,
            missing_percent=round(missing_percent, 2),
            duplicate_rows=duplicate_rows,
            duplicate_percent=round(duplicate_percent, 2),
            memory_usage=memory_usage,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            columns_analysis=columns_analysis,
            correlations=correlations,
            strong_correlations=strong_correlations,
            data_issues=data_issues
        )
        
        logger.info(f"Analysis complete. Found {len(data_issues)} potential issues")
        return self.report
    
    def _analyze_column(self, series: pd.Series) -> ColumnAnalysis:
        """Analyze a single column"""
        name = series.name
        dtype = str(series.dtype)
        missing_count = series.isnull().sum()
        missing_percent = round((missing_count / len(series)) * 100, 2) if len(series) > 0 else 0
        unique_count = series.nunique()
        unique_percent = round((unique_count / len(series)) * 100, 2) if len(series) > 0 else 0
        sample_values = series.dropna().head(5).tolist()
        
        analysis = ColumnAnalysis(
            name=name,
            dtype=dtype,
            missing_count=missing_count,
            missing_percent=missing_percent,
            unique_count=unique_count,
            unique_percent=unique_percent,
            sample_values=sample_values
        )
        
        # Numeric analysis - exclude boolean columns
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            clean_series = series.dropna()
            if len(clean_series) > 0:
                analysis.mean = round(float(clean_series.mean()), 4)
                analysis.median = round(float(clean_series.median()), 4)
                analysis.std = round(float(clean_series.std()), 4)
                analysis.min_val = round(float(clean_series.min()), 4)
                analysis.max_val = round(float(clean_series.max()), 4)
                analysis.q1 = round(float(clean_series.quantile(0.25)), 4)
                analysis.q3 = round(float(clean_series.quantile(0.75)), 4)
                
                if len(clean_series) > 2:
                    analysis.skewness = round(float(clean_series.skew()), 4)
                    analysis.kurtosis = round(float(clean_series.kurtosis()), 4)
                
                # Outlier detection using IQR
                iqr = analysis.q3 - analysis.q1
                lower_bound = analysis.q1 - (self.iqr_multiplier * iqr)
                upper_bound = analysis.q3 + (self.iqr_multiplier * iqr)
                outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
                analysis.outliers_count = len(outliers)
                analysis.outliers_percent = round((len(outliers) / len(clean_series)) * 100, 2)
        
        # Boolean columns - treat as categorical
        elif pd.api.types.is_bool_dtype(series):
            value_counts = series.value_counts()
            analysis.top_values = {str(k): int(v) for k, v in value_counts.head(10).items()}
            analysis.value_counts = {str(k): int(v) for k, v in value_counts.items()}
        
        # Categorical analysis
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            value_counts = series.value_counts()
            analysis.top_values = value_counts.head(10).to_dict()
            analysis.value_counts = value_counts.to_dict()
        
        return analysis
    
    def _identify_issues(self, df: pd.DataFrame, columns_analysis: Dict[str, ColumnAnalysis]) -> List[Dict[str, Any]]:
        """Identify data quality issues"""
        issues = []
        
        for col, analysis in columns_analysis.items():
            # Missing value issues
            if analysis.missing_percent >= self.missing_critical * 100:
                issues.append({
                    "type": "critical_missing",
                    "column": col,
                    "severity": "critical",
                    "message": f"Column '{col}' has {analysis.missing_percent}% missing values (critical threshold: {self.missing_critical*100}%)",
                    "recommendation": "Consider dropping this column or using advanced imputation"
                })
            elif analysis.missing_percent >= self.missing_warning * 100:
                issues.append({
                    "type": "warning_missing",
                    "column": col,
                    "severity": "warning",
                    "message": f"Column '{col}' has {analysis.missing_percent}% missing values",
                    "recommendation": "Investigate missing pattern and apply appropriate imputation"
                })
            
            # Outlier issues
            if analysis.outliers_percent > 5:
                issues.append({
                    "type": "outliers",
                    "column": col,
                    "severity": "warning",
                    "message": f"Column '{col}' has {analysis.outliers_count} outliers ({analysis.outliers_percent}%)",
                    "recommendation": "Review outliers for data entry errors or legitimate extreme values"
                })
            
            # High cardinality categorical
            if analysis.dtype == 'object' and analysis.unique_percent > 90:
                issues.append({
                    "type": "high_cardinality",
                    "column": col,
                    "severity": "info",
                    "message": f"Column '{col}' has very high cardinality ({analysis.unique_count} unique values)",
                    "recommendation": "Consider if this column needs encoding or can be grouped"
                })
            
            # Constant columns
            if analysis.unique_count <= 1:
                issues.append({
                    "type": "constant_column",
                    "column": col,
                    "severity": "warning",
                    "message": f"Column '{col}' has only {analysis.unique_count} unique value(s)",
                    "recommendation": "Consider dropping this column as it provides no information"
                })
            
            # Skewness issues
            if analysis.skewness is not None and abs(analysis.skewness) > 2:
                issues.append({
                    "type": "high_skewness",
                    "column": col,
                    "severity": "info",
                    "message": f"Column '{col}' is highly skewed (skewness: {analysis.skewness})",
                    "recommendation": "Consider log or Box-Cox transformation"
                })
        
        # Duplicate rows
        if df.duplicated().sum() > 0:
            issues.append({
                "type": "duplicates",
                "column": None,
                "severity": "warning",
                "message": f"Dataset contains {df.duplicated().sum()} duplicate rows",
                "recommendation": "Review and remove duplicate records if appropriate"
            })
        
        return issues
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis"""
        if not self.report:
            return {"error": "No analysis performed yet"}
        
        return {
            "rows": self.report.total_rows,
            "columns": self.report.total_columns,
            "missing_percent": self.report.missing_percent,
            "duplicate_rows": self.report.duplicate_rows,
            "numeric_columns": len(self.report.numeric_columns),
            "categorical_columns": len(self.report.categorical_columns),
            "issues_count": len(self.report.data_issues),
            "strong_correlations": len(self.report.strong_correlations)
        }
    
    def get_column_stats(self, column: str) -> Optional[ColumnAnalysis]:
        """Get detailed statistics for a specific column"""
        if self.report and column in self.report.columns_analysis:
            return self.report.columns_analysis[column]
        return None
    
    def detect_trends(self, df: pd.DataFrame, date_column: str = None, value_column: str = None) -> Dict[str, Any]:
        """Detect trends in time series data"""
        trends = {"detected": False}
        
        if date_column and value_column:
            try:
                df_sorted = df.sort_values(date_column)
                values = df_sorted[value_column].dropna()
                
                if len(values) > 10:
                    # Simple linear trend
                    x = np.arange(len(values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                    
                    trends = {
                        "detected": True,
                        "slope": round(slope, 4),
                        "direction": "increasing" if slope > 0 else "decreasing",
                        "strength": round(abs(r_value), 4),
                        "p_value": round(p_value, 4),
                        "significant": p_value < 0.05
                    }
            except Exception as e:
                logger.error(f"Trend detection failed: {e}")
        
        return trends
