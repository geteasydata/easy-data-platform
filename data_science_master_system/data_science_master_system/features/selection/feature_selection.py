"""
Feature Selection for Data Science Master System.

Provides comprehensive feature selection methods:
- Filter methods: Variance, correlation, statistical tests
- Wrapper methods: RFE, forward/backward selection
- Embedded methods: L1 regularization, tree importance

Example:
    >>> selector = FeatureSelector()
    >>> selected = selector.select(X, y, method="embedded", n_features=20)
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from data_science_master_system.core.base_classes import BaseProcessor
from data_science_master_system.core.exceptions import FeatureSelectionError
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector(BaseProcessor):
    """
    Unified feature selector with multiple methods.
    
    Supports:
    - Filter methods: variance, correlation, chi-square, mutual info
    - Wrapper methods: RFE, forward/backward selection
    - Embedded methods: L1, tree importance
    
    Example:
        >>> selector = FeatureSelector(method="filter")
        >>> selector.fit(X, y)
        >>> X_selected = selector.transform(X)
        >>> 
        >>> # Or use select() for one-step selection
        >>> selected_features = selector.select(X, y, n_features=20)
    """
    
    def __init__(
        self,
        method: str = "embedded",
        n_features: Optional[int] = None,
        threshold: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ("filter", "wrapper", "embedded")
            n_features: Number of features to select
            threshold: Score threshold for selection
            config: Additional configuration
        """
        super().__init__(config)
        self.method = method.lower()
        self.n_features = n_features
        self.threshold = threshold
        self._selector = None
        self._selected_features: List[str] = []
        self._feature_scores: Dict[str, float] = {}
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> "FeatureSelector":
        """
        Fit feature selector.
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional arguments
            
        Returns:
            Self
        """
        if self.method == "filter":
            selector = FilterSelector(
                n_features=self.n_features,
                threshold=self.threshold,
                config=self.config,
            )
        elif self.method == "wrapper":
            selector = WrapperSelector(
                n_features=self.n_features,
                config=self.config,
            )
        elif self.method == "embedded":
            selector = EmbeddedSelector(
                n_features=self.n_features,
                config=self.config,
            )
        else:
            raise FeatureSelectionError(f"Unknown method: {self.method}")
        
        selector.fit(X, y, **kwargs)
        self._selector = selector
        self._selected_features = selector.selected_features
        self._feature_scores = selector.feature_scores
        self._fitted = True
        
        return self
    
    def transform(self, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Transform data using selected features.
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with selected features only
        """
        if not self._fitted:
            raise FeatureSelectionError("Selector not fitted. Call fit() first.")
        
        return X[self._selected_features]
    
    def select(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        n_features: Optional[int] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        if n_features:
            self.n_features = n_features
        
        self.fit(X, y, **kwargs)
        return self.transform(X)
    
    @property
    def selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        return self._selected_features
    
    @property
    def feature_scores(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self._feature_scores


class FilterSelector(BaseProcessor):
    """
    Filter-based feature selection.
    
    Methods:
    - variance: Remove low-variance features
    - correlation: Remove highly correlated features
    - chi2: Chi-square test for categorical targets
    - mutual_info: Mutual information score
    - f_score: ANOVA F-value
    """
    
    def __init__(
        self,
        n_features: Optional[int] = None,
        threshold: Optional[float] = None,
        filter_method: str = "mutual_info",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self.n_features = n_features
        self.threshold = threshold
        self.filter_method = filter_method
        self._selected_features: List[str] = []
        self._feature_scores: Dict[str, float] = {}
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> "FilterSelector":
        """Fit filter selector."""
        from sklearn.feature_selection import (
            VarianceThreshold,
            SelectKBest,
            chi2,
            f_classif,
            f_regression,
            mutual_info_classif,
            mutual_info_regression,
        )
        
        feature_names = list(X.columns)
        
        if self.filter_method == "variance":
            selector = VarianceThreshold(threshold=self.threshold or 0.01)
            selector.fit(X)
            mask = selector.get_support()
            self._selected_features = [f for f, m in zip(feature_names, mask) if m]
            self._feature_scores = dict(zip(feature_names, selector.variances_))
        
        elif self.filter_method in ["chi2", "f_score", "mutual_info"]:
            if y is None:
                raise FeatureSelectionError("Target variable required for this method")
            
            # Determine if classification or regression
            is_classification = y.nunique() < 10 or y.dtype == "object"
            
            if self.filter_method == "chi2":
                score_func = chi2
                X_non_neg = X - X.min()  # Make non-negative
            elif self.filter_method == "f_score":
                score_func = f_classif if is_classification else f_regression
                X_non_neg = X
            else:  # mutual_info
                score_func = mutual_info_classif if is_classification else mutual_info_regression
                X_non_neg = X
            
            n = self.n_features or min(20, len(feature_names))
            selector = SelectKBest(score_func=score_func, k=n)
            selector.fit(X_non_neg, y)
            
            mask = selector.get_support()
            self._selected_features = [f for f, m in zip(feature_names, mask) if m]
            self._feature_scores = dict(zip(feature_names, selector.scores_))
        
        elif self.filter_method == "correlation":
            # Remove highly correlated features
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            threshold = self.threshold or 0.95
            to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
            
            self._selected_features = [f for f in feature_names if f not in to_drop]
            self._feature_scores = {f: 1.0 for f in self._selected_features}
        
        self._fitted = True
        logger.info(f"Selected {len(self._selected_features)} features using {self.filter_method}")
        return self
    
    def transform(self, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Transform using selected features."""
        return X[self._selected_features]
    
    @property
    def selected_features(self) -> List[str]:
        return self._selected_features
    
    @property
    def feature_scores(self) -> Dict[str, float]:
        return self._feature_scores


class WrapperSelector(BaseProcessor):
    """
    Wrapper-based feature selection.
    
    Methods:
    - RFE: Recursive Feature Elimination
    - forward: Forward selection
    - backward: Backward elimination
    """
    
    def __init__(
        self,
        n_features: Optional[int] = None,
        wrapper_method: str = "rfe",
        estimator: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self.n_features = n_features
        self.wrapper_method = wrapper_method
        self.estimator = estimator
        self._selected_features: List[str] = []
        self._feature_scores: Dict[str, float] = {}
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> "WrapperSelector":
        """Fit wrapper selector."""
        from sklearn.feature_selection import RFE, RFECV
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        if y is None:
            raise FeatureSelectionError("Target variable required for wrapper methods")
        
        feature_names = list(X.columns)
        
        # Create estimator if not provided
        if self.estimator is None:
            is_classification = y.nunique() < 10 or y.dtype == "object"
            if is_classification:
                self.estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                self.estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        
        if self.wrapper_method == "rfe":
            n = self.n_features or max(1, len(feature_names) // 2)
            selector = RFE(self.estimator, n_features_to_select=n, step=1)
            selector.fit(X, y)
            
            mask = selector.get_support()
            self._selected_features = [f for f, m in zip(feature_names, mask) if m]
            self._feature_scores = dict(zip(feature_names, 1 / selector.ranking_))
        
        elif self.wrapper_method == "rfecv":
            selector = RFECV(self.estimator, step=1, cv=5, scoring="accuracy")
            selector.fit(X, y)
            
            mask = selector.get_support()
            self._selected_features = [f for f, m in zip(feature_names, mask) if m]
            self._feature_scores = dict(zip(feature_names, 1 / selector.ranking_))
        
        self._fitted = True
        logger.info(f"Selected {len(self._selected_features)} features using {self.wrapper_method}")
        return self
    
    def transform(self, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Transform using selected features."""
        return X[self._selected_features]
    
    @property
    def selected_features(self) -> List[str]:
        return self._selected_features
    
    @property
    def feature_scores(self) -> Dict[str, float]:
        return self._feature_scores


class EmbeddedSelector(BaseProcessor):
    """
    Embedded feature selection using model coefficients/importances.
    
    Methods:
    - lasso: L1 regularization
    - tree: Tree-based importance
    - permutation: Permutation importance
    """
    
    def __init__(
        self,
        n_features: Optional[int] = None,
        embedded_method: str = "tree",
        estimator: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self.n_features = n_features
        self.embedded_method = embedded_method
        self.estimator = estimator
        self._selected_features: List[str] = []
        self._feature_scores: Dict[str, float] = {}
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> "EmbeddedSelector":
        """Fit embedded selector."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LassoCV, LogisticRegressionCV
        
        if y is None:
            raise FeatureSelectionError("Target variable required for embedded methods")
        
        feature_names = list(X.columns)
        is_classification = y.nunique() < 10 or y.dtype == "object"
        
        if self.embedded_method == "tree":
            if self.estimator is None:
                if is_classification:
                    self.estimator = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    self.estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            
            self.estimator.fit(X, y)
            importances = self.estimator.feature_importances_
            self._feature_scores = dict(zip(feature_names, importances))
        
        elif self.embedded_method == "lasso":
            if is_classification:
                model = LogisticRegressionCV(penalty="l1", solver="saga", cv=5, max_iter=1000)
            else:
                model = LassoCV(cv=5)
            
            model.fit(X, y)
            
            if is_classification:
                importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
            else:
                importances = np.abs(model.coef_)
            
            self._feature_scores = dict(zip(feature_names, importances))
        
        # Select top n features
        sorted_features = sorted(
            self._feature_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        n = self.n_features or len(feature_names)
        self._selected_features = [f for f, _ in sorted_features[:n]]
        
        # Filter out zero-importance features if using lasso
        if self.embedded_method == "lasso":
            self._selected_features = [
                f for f in self._selected_features
                if self._feature_scores[f] > 0
            ]
        
        self._fitted = True
        logger.info(f"Selected {len(self._selected_features)} features using {self.embedded_method}")
        return self
    
    def transform(self, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Transform using selected features."""
        return X[self._selected_features]
    
    @property
    def selected_features(self) -> List[str]:
        return self._selected_features
    
    @property
    def feature_scores(self) -> Dict[str, float]:
        return self._feature_scores
