"""
Feature Transformers for Data Science Master System.

Provides sklearn-compatible transformers:
- Scaling: StandardScaler, MinMaxScaler, RobustScaler
- Encoding: LabelEncoder, OneHotEncoder, TargetEncoder
- Dimensionality reduction: PCA, UMAP adapters

Example:
    >>> scaler = StandardScaler()
    >>> X_scaled = scaler.fit_transform(X)
    >>> 
    >>> encoder = OneHotEncoder()
    >>> X_encoded = encoder.fit_transform(X[["category"]])
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from data_science_master_system.core.base_classes import BaseTransformer
from data_science_master_system.core.exceptions import FeatureError
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)


class FeatureTransformer(BaseTransformer):
    """
    Base class for feature transformers.
    
    Provides sklearn-compatible interface with pandas support.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._feature_names_out: List[str] = []
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Get output feature names."""
        return self._feature_names_out
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get transformer parameters."""
        return {}
    
    def set_params(self, **params: Any) -> "FeatureTransformer":
        """Set transformer parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class StandardScaler(FeatureTransformer):
    """
    Standardize features by removing mean and scaling to unit variance.
    
    Example:
        >>> scaler = StandardScaler()
        >>> X_train_scaled = scaler.fit_transform(X_train)
        >>> X_test_scaled = scaler.transform(X_test)
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
    
    def fit(self, X: Any, y: Optional[Any] = None) -> "StandardScaler":
        """Fit scaler to data."""
        X_arr = np.asarray(X)
        
        if self.with_mean:
            self._mean = np.nanmean(X_arr, axis=0)
        if self.with_std:
            self._std = np.nanstd(X_arr, axis=0)
            self._std[self._std == 0] = 1  # Avoid division by zero
        
        if hasattr(X, "columns"):
            self._feature_names_out = list(X.columns)
        
        self._is_fitted = True
        return self
    
    def transform(self, X: Any) -> Any:
        """Transform data."""
        X_arr = np.asarray(X)
        result = X_arr.copy()
        
        if self.with_mean and self._mean is not None:
            result = result - self._mean
        if self.with_std and self._std is not None:
            result = result / self._std
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, columns=X.columns, index=X.index)
        return result
    
    def inverse_transform(self, X: Any) -> Any:
        """Inverse transform data."""
        X_arr = np.asarray(X)
        result = X_arr.copy()
        
        if self.with_std and self._std is not None:
            result = result * self._std
        if self.with_mean and self._mean is not None:
            result = result + self._mean
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, columns=X.columns, index=X.index)
        return result


class MinMaxScaler(FeatureTransformer):
    """
    Scale features to a given range (default [0, 1]).
    
    Example:
        >>> scaler = MinMaxScaler(feature_range=(0, 1))
        >>> X_scaled = scaler.fit_transform(X)
    """
    
    def __init__(self, feature_range: tuple = (0, 1)) -> None:
        super().__init__()
        self.feature_range = feature_range
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
    
    def fit(self, X: Any, y: Optional[Any] = None) -> "MinMaxScaler":
        """Fit scaler."""
        X_arr = np.asarray(X)
        
        self._min = np.nanmin(X_arr, axis=0)
        self._max = np.nanmax(X_arr, axis=0)
        
        data_range = self._max - self._min
        data_range[data_range == 0] = 1
        
        feature_min, feature_max = self.feature_range
        self._scale = (feature_max - feature_min) / data_range
        
        if hasattr(X, "columns"):
            self._feature_names_out = list(X.columns)
        
        self._is_fitted = True
        return self
    
    def transform(self, X: Any) -> Any:
        """Transform data."""
        X_arr = np.asarray(X)
        feature_min, _ = self.feature_range
        
        result = (X_arr - self._min) * self._scale + feature_min
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, columns=X.columns, index=X.index)
        return result
    
    def inverse_transform(self, X: Any) -> Any:
        """Inverse transform."""
        X_arr = np.asarray(X)
        feature_min, _ = self.feature_range
        
        result = (X_arr - feature_min) / self._scale + self._min
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, columns=X.columns, index=X.index)
        return result


class RobustScaler(FeatureTransformer):
    """
    Scale features using statistics robust to outliers (median and IQR).
    
    Example:
        >>> scaler = RobustScaler()
        >>> X_scaled = scaler.fit_transform(X)
    """
    
    def __init__(
        self,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple = (25.0, 75.0),
    ) -> None:
        super().__init__()
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self._median: Optional[np.ndarray] = None
        self._iqr: Optional[np.ndarray] = None
    
    def fit(self, X: Any, y: Optional[Any] = None) -> "RobustScaler":
        """Fit scaler."""
        X_arr = np.asarray(X)
        
        if self.with_centering:
            self._median = np.nanmedian(X_arr, axis=0)
        
        if self.with_scaling:
            q1, q3 = self.quantile_range
            percentiles = np.nanpercentile(X_arr, [q1, q3], axis=0)
            self._iqr = percentiles[1] - percentiles[0]
            self._iqr[self._iqr == 0] = 1
        
        if hasattr(X, "columns"):
            self._feature_names_out = list(X.columns)
        
        self._is_fitted = True
        return self
    
    def transform(self, X: Any) -> Any:
        """Transform data."""
        X_arr = np.asarray(X)
        result = X_arr.copy()
        
        if self.with_centering and self._median is not None:
            result = result - self._median
        if self.with_scaling and self._iqr is not None:
            result = result / self._iqr
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, columns=X.columns, index=X.index)
        return result


class LabelEncoder(FeatureTransformer):
    """
    Encode categorical labels as integers.
    
    Example:
        >>> encoder = LabelEncoder()
        >>> y_encoded = encoder.fit_transform(y)
        >>> y_original = encoder.inverse_transform(y_encoded)
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._classes: Optional[np.ndarray] = None
        self._class_to_int: Dict[Any, int] = {}
        self._int_to_class: Dict[int, Any] = {}
    
    def fit(self, X: Any, y: Optional[Any] = None) -> "LabelEncoder":
        """Fit encoder."""
        values = np.asarray(X).ravel()
        self._classes = np.unique(values[~pd.isna(values)])
        
        self._class_to_int = {c: i for i, c in enumerate(self._classes)}
        self._int_to_class = {i: c for c, i in self._class_to_int.items()}
        
        self._is_fitted = True
        return self
    
    def transform(self, X: Any) -> Any:
        """Transform labels to integers."""
        values = np.asarray(X).ravel()
        result = np.array([
            self._class_to_int.get(v, -1) if not pd.isna(v) else -1
            for v in values
        ])
        
        if isinstance(X, pd.Series):
            return pd.Series(result, index=X.index, name=X.name)
        return result
    
    def inverse_transform(self, X: Any) -> Any:
        """Inverse transform integers to labels."""
        values = np.asarray(X).ravel()
        result = np.array([
            self._int_to_class.get(int(v), None) if v >= 0 else None
            for v in values
        ])
        
        if isinstance(X, pd.Series):
            return pd.Series(result, index=X.index, name=X.name)
        return result
    
    @property
    def classes_(self) -> np.ndarray:
        """Get unique classes."""
        return self._classes


class OneHotEncoder(FeatureTransformer):
    """
    Encode categorical features as one-hot vectors.
    
    Example:
        >>> encoder = OneHotEncoder(sparse=False)
        >>> X_encoded = encoder.fit_transform(X[["category"]])
    """
    
    def __init__(
        self,
        sparse: bool = False,
        handle_unknown: str = "ignore",
        drop: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.sparse = sparse
        self.handle_unknown = handle_unknown
        self.drop = drop
        self._categories: Dict[str, List[Any]] = {}
    
    def fit(self, X: Any, y: Optional[Any] = None) -> "OneHotEncoder":
        """Fit encoder."""
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                self._categories[col] = X[col].dropna().unique().tolist()
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            for i in range(X_arr.shape[1]):
                self._categories[f"x{i}"] = np.unique(X_arr[:, i]).tolist()
        
        # Build output feature names
        self._feature_names_out = []
        for col, cats in self._categories.items():
            for cat in cats:
                self._feature_names_out.append(f"{col}_{cat}")
        
        self._is_fitted = True
        return self
    
    def transform(self, X: Any) -> Any:
        """Transform to one-hot encoding."""
        if isinstance(X, pd.DataFrame):
            result_dfs = []
            for col in X.columns:
                cats = self._categories.get(col, [])
                for cat in cats:
                    result_dfs.append(
                        pd.DataFrame({f"{col}_{cat}": (X[col] == cat).astype(int)})
                    )
            result = pd.concat(result_dfs, axis=1)
            result.index = X.index
            return result
        else:
            # Use sklearn for array input
            from sklearn.preprocessing import OneHotEncoder as SklearnOHE
            
            encoder = SklearnOHE(sparse=self.sparse, handle_unknown=self.handle_unknown)
            return encoder.fit_transform(X)


class TargetEncoder(FeatureTransformer):
    """
    Encode categorical features using target mean.
    
    Example:
        >>> encoder = TargetEncoder()
        >>> X_encoded = encoder.fit_transform(X[["category"]], y)
    """
    
    def __init__(self, smoothing: float = 1.0) -> None:
        super().__init__()
        self.smoothing = smoothing
        self._encodings: Dict[str, Dict[Any, float]] = {}
        self._global_mean: float = 0.0
    
    def fit(self, X: Any, y: Optional[Any] = None) -> "TargetEncoder":
        """Fit encoder using target."""
        if y is None:
            raise FeatureError("Target variable required for TargetEncoder")
        
        y_arr = np.asarray(y)
        self._global_mean = np.nanmean(y_arr)
        
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                self._encodings[col] = {}
                for cat in X[col].unique():
                    if pd.notna(cat):
                        mask = X[col] == cat
                        n = mask.sum()
                        cat_mean = y_arr[mask].mean()
                        
                        # Smooth encoding
                        smoothed = (n * cat_mean + self.smoothing * self._global_mean) / (n + self.smoothing)
                        self._encodings[col][cat] = smoothed
        
        if hasattr(X, "columns"):
            self._feature_names_out = [f"{c}_encoded" for c in X.columns]
        
        self._is_fitted = True
        return self
    
    def transform(self, X: Any) -> Any:
        """Transform using target encoding."""
        if isinstance(X, pd.DataFrame):
            result = pd.DataFrame(index=X.index)
            for col in X.columns:
                col_encodings = self._encodings.get(col, {})
                result[f"{col}_encoded"] = X[col].map(
                    lambda x: col_encodings.get(x, self._global_mean)
                )
            return result
        else:
            raise FeatureError("TargetEncoder requires DataFrame input")
