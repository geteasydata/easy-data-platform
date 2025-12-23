"""
Professional Feature Selector - Scientific Feature Selection
Selects the most important features using multiple methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression,
    mutual_info_classif, mutual_info_regression, RFE
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Professional Feature Selector - uses multiple scientific methods.
    Selects features like a senior data scientist would.
    """
    
    def __init__(self):
        self.selected_features = []
        self.feature_scores = {}
        self.selection_log = []
        self.removed_features = []
        
    def log(self, message: str):
        """Add to selection log."""
        self.selection_log.append(f"🎯 {message}")
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                        problem_type: str = 'classification',
                        n_features: Optional[int] = None,
                        methods: List[str] = ['variance', 'correlation', 'importance']) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select best features using multiple methods.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            problem_type: 'classification' or 'regression'
            n_features: Number of features to keep (None = auto)
            methods: List of selection methods to use
            
        Returns:
            (X_selected, selected_columns)
        """
        X = X.copy()
        self.selection_log = []
        self.removed_features = []
        
        original_count = len(X.columns)
        
        # Determine number of features to keep
        if n_features is None:
            n_features = max(5, min(50, len(X.columns) // 2))
        
        # 1. Remove zero/low variance features
        if 'variance' in methods:
            X = self._remove_low_variance(X)
        
        # 2. Remove highly correlated features
        if 'correlation' in methods:
            X = self._remove_high_correlation(X)
        
        # 3. Select by importance
        if 'importance' in methods and len(X.columns) > n_features:
            X = self._select_by_importance(X, y, problem_type, n_features)
        
        # 4. Mutual Information (optional)
        if 'mutual_info' in methods and len(X.columns) > n_features:
            X = self._select_by_mutual_info(X, y, problem_type, n_features)
        
        self.selected_features = list(X.columns)
        self.log(f"Reduced features from {original_count} to {len(X.columns)}")
        
        return X, self.selected_features
    
    def _remove_low_variance(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove features with very low variance."""
        numeric_X = X.select_dtypes(include=[np.number])
        
        if len(numeric_X.columns) == 0:
            return X
        
        try:
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(numeric_X)
            
            low_var_cols = numeric_X.columns[~selector.get_support()].tolist()
            
            if low_var_cols:
                X = X.drop(columns=low_var_cols)
                self.removed_features.extend(low_var_cols)
                self.log(f"Removed {len(low_var_cols)} low-variance features")
        except:
            pass
        
        return X
    
    def _remove_high_correlation(self, X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features."""
        numeric_X = X.select_dtypes(include=[np.number])
        
        if len(numeric_X.columns) < 2:
            return X
        
        try:
            corr_matrix = numeric_X.corr().abs()
            
            # Get upper triangle
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find columns with correlation > threshold
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            
            if to_drop:
                X = X.drop(columns=to_drop)
                self.removed_features.extend(to_drop)
                self.log(f"Removed {len(to_drop)} highly correlated features")
        except:
            pass
        
        return X
    
    def _select_by_importance(self, X: pd.DataFrame, y: pd.Series, 
                              problem_type: str, n_features: int) -> pd.DataFrame:
        """Select features by tree-based importance."""
        numeric_X = X.select_dtypes(include=[np.number])
        
        if len(numeric_X.columns) <= n_features:
            return X
        
        try:
            # Prepare data
            X_clean = numeric_X.fillna(0)
            y_clean = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
            
            # Use Random Forest for importance
            if problem_type == 'classification':
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            
            model.fit(X_clean, y_clean)
            
            # Get importance
            importances = pd.Series(model.feature_importances_, index=numeric_X.columns)
            importances = importances.sort_values(ascending=False)
            
            # Keep top n features
            top_features = importances.head(n_features).index.tolist()
            
            # Also keep non-numeric columns
            non_numeric = [c for c in X.columns if c not in numeric_X.columns]
            keep_cols = top_features + non_numeric
            
            removed = [c for c in X.columns if c not in keep_cols]
            if removed:
                X = X[keep_cols]
                self.removed_features.extend(removed)
                self.log(f"Selected top {n_features} features by importance")
            
            # Store scores
            self.feature_scores = importances.to_dict()
            
        except Exception as e:
            self.log(f"Importance selection skipped: {str(e)[:50]}")
        
        return X
    
    def _select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series,
                               problem_type: str, n_features: int) -> pd.DataFrame:
        """Select features by mutual information."""
        numeric_X = X.select_dtypes(include=[np.number])
        
        if len(numeric_X.columns) <= n_features:
            return X
        
        try:
            X_clean = numeric_X.fillna(0)
            y_clean = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
            
            if problem_type == 'classification':
                mi_scores = mutual_info_classif(X_clean, y_clean, random_state=42)
            else:
                mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
            
            mi_series = pd.Series(mi_scores, index=numeric_X.columns)
            mi_series = mi_series.sort_values(ascending=False)
            
            top_features = mi_series.head(n_features).index.tolist()
            non_numeric = [c for c in X.columns if c not in numeric_X.columns]
            keep_cols = top_features + non_numeric
            
            removed = [c for c in X.columns if c not in keep_cols]
            if removed:
                X = X[keep_cols]
                self.log(f"Selected top {n_features} features by mutual information")
                
        except Exception as e:
            self.log(f"MI selection skipped: {str(e)[:50]}")
        
        return X
    
    def get_feature_scores(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_scores
    
    def get_log(self) -> List[str]:
        """Get selection log."""
        return self.selection_log
    
    def get_removed_features(self) -> List[str]:
        """Get list of removed features."""
        return self.removed_features


def select_features(X: pd.DataFrame, y: pd.Series, 
                   problem_type: str = 'classification',
                   n_features: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Convenience function for feature selection."""
    selector = FeatureSelector()
    return selector.select_features(X, y, problem_type, n_features)
