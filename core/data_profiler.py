"""
Professional Data Profiler - Deep Exploratory Data Analysis
Enterprise-grade data understanding like a 30+ year expert
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class DataProfiler:
    """
    Professional Data Profiler - performs deep EDA automatically.
    Understands data like a senior data scientist would.
    """
    
    def __init__(self):
        self.profile = {}
        self.warnings = []
        self.recommendations = []
        self.quality_score = 0
        
    def profile_dataset(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete data profiling - understands everything about the data.
        
        Returns comprehensive profile with:
        - Basic stats
        - Column analysis
        - Data quality metrics
        - Correlations
        - Outliers
        - Recommendations
        """
        self.warnings = []
        self.recommendations = []
        
        profile = {
            'overview': self._get_overview(df),
            'columns': self._analyze_columns(df),
            'quality': self._assess_quality(df),
            'correlations': self._analyze_correlations(df),
            'outliers': self._detect_outliers(df),
            'target_analysis': self._analyze_target(df, target_col) if target_col else None,
            'warnings': [],
            'recommendations': [],
            'quality_score': 0
        }
        
        # Calculate overall quality score
        profile['quality_score'] = self._calculate_quality_score(profile)
        profile['warnings'] = self.warnings
        profile['recommendations'] = self.recommendations
        
        self.profile = profile
        return profile
    
    def _get_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get dataset overview."""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'duplicates': int(df.duplicated().sum()),
            'duplicate_pct': round(df.duplicated().sum() / len(df) * 100, 2),
            'total_missing': int(df.isnull().sum().sum()),
            'missing_pct': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
            'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_cols': len(df.select_dtypes(include=['datetime64']).columns),
        }
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze each column in detail."""
        columns = {}
        
        for col in df.columns:
            col_data = df[col]
            col_info = {
                'dtype': str(col_data.dtype),
                'missing': int(col_data.isnull().sum()),
                'missing_pct': round(col_data.isnull().sum() / len(df) * 100, 2),
                'unique': int(col_data.nunique()),
                'unique_pct': round(col_data.nunique() / len(df) * 100, 2),
            }
            
            # Detect actual type
            col_info['inferred_type'] = self._infer_type(col_data)
            
            # Numeric analysis
            if col_data.dtype in ['int64', 'float64', 'int32', 'float32']:
                col_info.update(self._analyze_numeric(col_data))
            
            # Categorical analysis
            elif col_data.dtype == 'object' or str(col_data.dtype) == 'category':
                col_info.update(self._analyze_categorical(col_data))
            
            columns[col] = col_info
        
        return columns
    
    def _infer_type(self, series: pd.Series) -> str:
        """Infer the actual semantic type of a column."""
        if series.dtype in ['int64', 'float64']:
            # Check if it's actually categorical
            if series.nunique() <= 10:
                return 'categorical_numeric'
            # Check if it's binary
            if series.nunique() == 2:
                return 'binary'
            # Check if it's ID
            if series.nunique() == len(series):
                return 'id'
            return 'numeric'
        
        elif series.dtype == 'object':
            # Sample non-null values
            sample = series.dropna().head(100)
            
            # Check if numeric string
            try:
                pd.to_numeric(sample.str.replace(r'[$€£¥,]', '', regex=True), errors='raise')
                return 'numeric_string'
            except:
                pass
            
            # Check if date
            try:
                pd.to_datetime(sample, errors='raise', infer_datetime_format=True)
                return 'datetime_string'
            except:
                pass
            
            # Check if email
            if sample.str.contains(r'@.*\.', regex=True, na=False).mean() > 0.5:
                return 'email'
            
            # Check if URL
            if sample.str.contains(r'http|www', regex=True, na=False).mean() > 0.5:
                return 'url'
            
            # Check if ID-like
            if series.nunique() == len(series.dropna()):
                return 'id'
            
            return 'categorical'
        
        return str(series.dtype)
    
    def _analyze_numeric(self, series: pd.Series) -> Dict:
        """Analyze numeric column."""
        clean = series.dropna()
        
        if len(clean) == 0:
            return {'stats': 'no valid values'}
        
        analysis = {
            'min': float(clean.min()),
            'max': float(clean.max()),
            'mean': float(clean.mean()),
            'median': float(clean.median()),
            'std': float(clean.std()),
            'skewness': float(clean.skew()) if len(clean) > 2 else 0,
            'kurtosis': float(clean.kurtosis()) if len(clean) > 3 else 0,
            'zeros': int((clean == 0).sum()),
            'zeros_pct': round((clean == 0).sum() / len(clean) * 100, 2),
            'negatives': int((clean < 0).sum()),
        }
        
        # Percentiles
        try:
            analysis['percentiles'] = {
                '25%': float(clean.quantile(0.25)),
                '50%': float(clean.quantile(0.50)),
                '75%': float(clean.quantile(0.75)),
                '95%': float(clean.quantile(0.95)),
                '99%': float(clean.quantile(0.99)),
            }
        except:
            pass
        
        # Distribution type detection
        if len(clean) >= 20:
            try:
                _, p_normal = stats.normaltest(clean)
                analysis['is_normal'] = p_normal > 0.05
            except:
                analysis['is_normal'] = None
        
        return analysis
    
    def _analyze_categorical(self, series: pd.Series) -> Dict:
        """Analyze categorical column."""
        clean = series.dropna()
        value_counts = clean.value_counts()
        
        analysis = {
            'top_values': value_counts.head(10).to_dict(),
            'cardinality': 'low' if series.nunique() <= 10 else 'medium' if series.nunique() <= 50 else 'high',
        }
        
        # Check imbalance
        if len(value_counts) >= 2:
            top_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
            analysis['imbalance_ratio'] = round(top_ratio, 2)
            
            if top_ratio > 10:
                self.warnings.append(f"Column '{series.name}' is highly imbalanced (ratio: {top_ratio:.1f})")
        
        return analysis
    
    def _assess_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        quality = {
            'completeness': round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 2),
            'uniqueness': round((1 - df.duplicated().sum() / len(df)) * 100, 2),
        }
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            quality['constant_columns'] = constant_cols
            self.warnings.append(f"Found {len(constant_cols)} constant columns (no information)")
            self.recommendations.append(f"Remove constant columns: {constant_cols}")
        
        # Check for high-cardinality categoricals
        high_card = [col for col in df.select_dtypes(include=['object']).columns 
                     if df[col].nunique() > 50]
        if high_card:
            quality['high_cardinality'] = high_card
            self.recommendations.append(f"Consider encoding or grouping high-cardinality columns: {high_card}")
        
        # Check for ID columns
        id_cols = [col for col in df.columns if df[col].nunique() == len(df)]
        if id_cols:
            quality['potential_id_columns'] = id_cols
            self.recommendations.append(f"Remove potential ID columns: {id_cols}")
        
        return quality
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {'status': 'not enough numeric columns'}
        
        try:
            corr_matrix = numeric_df.corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j:
                        corr = corr_matrix.loc[col1, col2]
                        if abs(corr) > 0.8:
                            high_corr_pairs.append({
                                'col1': col1,
                                'col2': col2,
                                'correlation': round(corr, 3)
                            })
            
            if high_corr_pairs:
                self.warnings.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
                self.recommendations.append("Consider removing one feature from each highly correlated pair")
            
            return {
                'high_correlations': high_corr_pairs,
                'matrix_summary': {
                    'min': round(corr_matrix.min().min(), 3),
                    'max': round(corr_matrix.max().max(), 3),
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numeric columns."""
        outliers = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            
            if len(series) < 4:
                continue
            
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()
            outlier_pct = outlier_count / len(series) * 100
            
            if outlier_count > 0:
                outliers[col] = {
                    'count': int(outlier_count),
                    'percentage': round(outlier_pct, 2),
                    'lower_bound': round(lower_bound, 3),
                    'upper_bound': round(upper_bound, 3),
                }
            
            if outlier_pct > 5:
                self.warnings.append(f"Column '{col}' has {outlier_pct:.1f}% outliers")
        
        return outliers
    
    def _analyze_target(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Analyze the target variable."""
        if target_col not in df.columns:
            return {'error': 'Target column not found'}
        
        target = df[target_col]
        
        analysis = {
            'dtype': str(target.dtype),
            'missing': int(target.isnull().sum()),
            'unique': int(target.nunique()),
        }
        
        # Determine problem type
        if target.dtype == 'object' or target.nunique() <= 20:
            analysis['problem_type'] = 'classification'
            analysis['classes'] = target.value_counts().to_dict()
            
            # Check class imbalance
            value_counts = target.value_counts()
            if len(value_counts) >= 2:
                imbalance_ratio = value_counts.max() / value_counts.min()
                analysis['imbalance_ratio'] = round(imbalance_ratio, 2)
                
                if imbalance_ratio > 3:
                    self.warnings.append(f"Target is imbalanced (ratio: {imbalance_ratio:.1f})")
                    self.recommendations.append("Consider using SMOTE or class weights")
        else:
            analysis['problem_type'] = 'regression'
            analysis['stats'] = self._analyze_numeric(target)
        
        return analysis
    
    def _calculate_quality_score(self, profile: Dict) -> float:
        """Calculate overall data quality score (0-100)."""
        score = 100
        
        # Penalize for missing values
        missing_pct = profile['overview']['missing_pct']
        score -= missing_pct * 0.5
        
        # Penalize for duplicates
        dup_pct = profile['overview']['duplicate_pct']
        score -= dup_pct * 0.3
        
        # Penalize for each warning
        score -= len(self.warnings) * 2
        
        return max(0, min(100, round(score, 1)))
    
    def get_summary(self, lang: str = 'ar') -> str:
        """Get human-readable summary."""
        if not self.profile:
            return "No profile generated yet."
        
        p = self.profile
        o = p['overview']
        
        if lang == 'ar':
            summary = f"""
📊 **ملخص البيانات**
- الصفوف: {o['rows']:,}
- الأعمدة: {o['columns']}
- الذاكرة: {o['memory_mb']} MB
- جودة البيانات: {p['quality_score']}%

📈 **الأعمدة**
- رقمية: {o['numeric_cols']}
- فئوية: {o['categorical_cols']}

⚠️ **تحذيرات**: {len(p['warnings'])}
💡 **توصيات**: {len(p['recommendations'])}
"""
        else:
            summary = f"""
📊 **Data Summary**
- Rows: {o['rows']:,}
- Columns: {o['columns']}
- Memory: {o['memory_mb']} MB
- Quality Score: {p['quality_score']}%

📈 **Columns**
- Numeric: {o['numeric_cols']}
- Categorical: {o['categorical_cols']}

⚠️ **Warnings**: {len(p['warnings'])}
💡 **Recommendations**: {len(p['recommendations'])}
"""
        return summary


def profile_data(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function for quick profiling."""
    profiler = DataProfiler()
    return profiler.profile_dataset(df, target_col)
