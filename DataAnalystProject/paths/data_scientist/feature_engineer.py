"""
Feature Engineering Module
Automated feature creation, selection, and dimensionality reduction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, 
    mutual_info_regression, RFE, VarianceThreshold
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class FeatureInfo:
    """Information about an engineered feature"""
    name: str
    source_columns: List[str]
    operation: str
    importance: Optional[float] = None


class FeatureEngineer:
    """
    Automated Feature Engineering
    Creates, selects, and transforms features for ML models
    """
    
    def __init__(self):
        self.created_features: List[FeatureInfo] = []
        self.selected_features: List[str] = []
        self.transformers: Dict[str, Any] = {}
        self.scaler = StandardScaler()
        
    def engineer_features(self, df: pd.DataFrame, target: str = None,
                          operations: List[str] = None) -> pd.DataFrame:
        """Apply feature engineering operations"""
        if operations is None:
            operations = ['basic_stats', 'interactions', 'bins', 'datetime']
        
        result_df = df.copy()
        self.created_features = []
        
        for op in operations:
            if op == 'basic_stats':
                result_df = self._create_basic_stats(result_df)
            elif op == 'interactions':
                result_df = self._create_interactions(result_df, max_features=5)
            elif op == 'bins':
                result_df = self._create_bins(result_df)
            elif op == 'datetime':
                result_df = self._extract_datetime_features(result_df)
            elif op == 'polynomial':
                result_df = self._create_polynomial_features(result_df)
            elif op == 'aggregations':
                result_df = self._create_aggregations(result_df, target)
        
        logger.info(f"Created {len(self.created_features)} new features")
        return result_df
    
    def _create_basic_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic statistical features for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Row-wise statistics
            df['_row_mean'] = df[numeric_cols].mean(axis=1)
            df['_row_std'] = df[numeric_cols].std(axis=1)
            df['_row_min'] = df[numeric_cols].min(axis=1)
            df['_row_max'] = df[numeric_cols].max(axis=1)
            df['_row_range'] = df['_row_max'] - df['_row_min']
            
            for feat in ['_row_mean', '_row_std', '_row_min', '_row_max', '_row_range']:
                self.created_features.append(FeatureInfo(
                    name=feat,
                    source_columns=numeric_cols,
                    operation='basic_stats'
                ))
        
        return df
    
    def _create_interactions(self, df: pd.DataFrame, max_features: int = 5) -> pd.DataFrame:
        """Create interaction features between numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Filter out already created features
        numeric_cols = [c for c in numeric_cols if not c.startswith('_')][:max_features]
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Multiplication
                new_col = f'{col1}_x_{col2}'
                df[new_col] = df[col1] * df[col2]
                self.created_features.append(FeatureInfo(
                    name=new_col,
                    source_columns=[col1, col2],
                    operation='multiplication'
                ))
                
                # Ratio (with safe division)
                if (df[col2] != 0).all():
                    ratio_col = f'{col1}_div_{col2}'
                    df[ratio_col] = df[col1] / df[col2].replace(0, np.nan)
                    self.created_features.append(FeatureInfo(
                        name=ratio_col,
                        source_columns=[col1, col2],
                        operation='division'
                    ))
        
        return df
    
    def _create_bins(self, df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
        """Create binned features for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if not c.startswith('_')]
        
        for col in numeric_cols[:10]:  # Limit to first 10 columns
            try:
                bin_col = f'{col}_bin'
                df[bin_col] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
                self.created_features.append(FeatureInfo(
                    name=bin_col,
                    source_columns=[col],
                    operation='binning'
                ))
            except Exception:
                pass  # Skip if binning fails
        
        return df
    
    def _extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns"""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Also check object columns that might be dates
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_datetime(df[col])
                datetime_cols.append(col)
            except:
                pass
        
        for col in datetime_cols:
            prefix = col
            df[f'{prefix}_year'] = df[col].dt.year
            df[f'{prefix}_month'] = df[col].dt.month
            df[f'{prefix}_day'] = df[col].dt.day
            df[f'{prefix}_dayofweek'] = df[col].dt.dayofweek
            df[f'{prefix}_hour'] = df[col].dt.hour
            df[f'{prefix}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
            
            for feat in ['year', 'month', 'day', 'dayofweek', 'hour', 'is_weekend']:
                self.created_features.append(FeatureInfo(
                    name=f'{prefix}_{feat}',
                    source_columns=[col],
                    operation='datetime_extraction'
                ))
        
        return df
    
    def _create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if not c.startswith('_')][:5]
        
        if len(numeric_cols) < 2:
            return df
        
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
        poly_features = poly.fit_transform(df[numeric_cols].fillna(0))
        feature_names = poly.get_feature_names_out(numeric_cols)
        
        # Only add new features (not the original ones)
        for i, name in enumerate(feature_names):
            if name not in numeric_cols:
                safe_name = name.replace(' ', '_')
                df[f'poly_{safe_name}'] = poly_features[:, i]
                self.created_features.append(FeatureInfo(
                    name=f'poly_{safe_name}',
                    source_columns=numeric_cols,
                    operation='polynomial'
                ))
        
        self.transformers['polynomial'] = poly
        return df
    
    def _create_aggregations(self, df: pd.DataFrame, target: str = None) -> pd.DataFrame:
        """Create aggregation features based on categorical columns"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target and not c.startswith('_')]
        
        for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical
            for num_col in numeric_cols[:3]:  # and first 3 numeric
                # Group mean
                group_mean = df.groupby(cat_col)[num_col].transform('mean')
                mean_col = f'{cat_col}_{num_col}_mean'
                df[mean_col] = group_mean
                self.created_features.append(FeatureInfo(
                    name=mean_col,
                    source_columns=[cat_col, num_col],
                    operation='group_mean'
                ))
                
                # Group count
                group_count = df.groupby(cat_col)[num_col].transform('count')
                count_col = f'{cat_col}_count'
                if count_col not in df.columns:
                    df[count_col] = group_count
                    self.created_features.append(FeatureInfo(
                        name=count_col,
                        source_columns=[cat_col],
                        operation='group_count'
                    ))
        
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                        method: str = 'importance',
                        n_features: int = None,
                        threshold: float = None) -> List[str]:
        """Select best features using various methods"""
        # Only use numeric features
        X_numeric = X.select_dtypes(include=[np.number]).copy()
        X_numeric = X_numeric.fillna(X_numeric.median())
        
        if n_features is None:
            n_features = min(20, len(X_numeric.columns))
        
        # Detect task type
        is_classification = y.dtype == 'object' or y.nunique() <= 10
        
        if method == 'importance':
            selected = self._select_by_importance(X_numeric, y, n_features, is_classification)
        elif method == 'mutual_info':
            selected = self._select_by_mutual_info(X_numeric, y, n_features, is_classification)
        elif method == 'rfe':
            selected = self._select_by_rfe(X_numeric, y, n_features, is_classification)
        elif method == 'variance':
            selected = self._select_by_variance(X_numeric, threshold or 0.01)
        else:
            selected = X_numeric.columns.tolist()[:n_features]
        
        self.selected_features = selected
        logger.info(f"Selected {len(selected)} features using {method}")
        return selected
    
    def _select_by_importance(self, X: pd.DataFrame, y: pd.Series, 
                              n_features: int, is_classification: bool) -> List[str]:
        """Select features by tree-based importance"""
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X, y)
        importance = pd.Series(model.feature_importances_, index=X.columns)
        return importance.nlargest(n_features).index.tolist()
    
    def _select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series,
                               n_features: int, is_classification: bool) -> List[str]:
        """Select features by mutual information"""
        if is_classification:
            selector = SelectKBest(mutual_info_classif, k=n_features)
        else:
            selector = SelectKBest(mutual_info_regression, k=n_features)
        
        selector.fit(X, y)
        mask = selector.get_support()
        return X.columns[mask].tolist()
    
    def _select_by_rfe(self, X: pd.DataFrame, y: pd.Series,
                       n_features: int, is_classification: bool) -> List[str]:
        """Select features by recursive feature elimination"""
        if is_classification:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        
        rfe = RFE(estimator, n_features_to_select=n_features, step=1)
        rfe.fit(X, y)
        return X.columns[rfe.support_].tolist()
    
    def _select_by_variance(self, X: pd.DataFrame, threshold: float) -> List[str]:
        """Select features by variance threshold"""
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        return X.columns[selector.get_support()].tolist()
    
    def reduce_dimensions(self, X: pd.DataFrame, n_components: int = None,
                          variance_ratio: float = 0.95) -> Tuple[np.ndarray, Dict]:
        """Apply PCA for dimensionality reduction"""
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        X_scaled = self.scaler.fit_transform(X_numeric)
        
        if n_components is None:
            # Find n_components to explain variance_ratio
            pca_full = PCA()
            pca_full.fit(X_scaled)
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_ratio) + 1
        
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_scaled)
        
        self.transformers['pca'] = pca
        
        info = {
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_variance_explained': sum(pca.explained_variance_ratio_),
            'original_features': X_numeric.columns.tolist()
        }
        
        logger.info(f"Reduced from {X_numeric.shape[1]} to {n_components} dimensions "
                   f"({info['total_variance_explained']:.2%} variance explained)")
        
        return X_reduced, info
    
    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary of created features"""
        data = []
        for feat in self.created_features:
            data.append({
                'Feature': feat.name,
                'Source Columns': ', '.join(feat.source_columns),
                'Operation': feat.operation,
                'Importance': feat.importance
            })
        return pd.DataFrame(data)
