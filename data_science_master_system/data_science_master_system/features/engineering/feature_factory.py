"""
Feature Factory for Data Science Master System.

Provides automated feature engineering:
- Automatic feature generation based on data types
- Time-series features (lag, rolling, seasonal)
- Text features (TF-IDF, word embeddings)
- Interaction features
- Polynomial features

Example:
    >>> factory = FeatureFactory()
    >>> features = factory.auto_generate(df, target="label")
    >>> selected = factory.select_best(features, n_features=20)
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from data_science_master_system.core.base_classes import BaseProcessor
from data_science_master_system.core.exceptions import FeatureEngineeringError
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)


class FeatureFactory(BaseProcessor):
    """
    Factory for automated feature engineering.
    
    Automatically generates features based on:
    - Column data types
    - Problem type (classification/regression)
    - Domain knowledge templates
    
    Example:
        >>> factory = FeatureFactory()
        >>> 
        >>> # Auto-generate all features
        >>> features = factory.auto_generate(df, target="label")
        >>> 
        >>> # Generate specific feature types
        >>> numeric_features = factory.generate_numeric_features(df)
        >>> categorical_features = factory.generate_categorical_features(df)
        >>> interaction_features = factory.generate_interactions(df, ["a", "b", "c"])
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize feature factory.
        
        Args:
            config: Feature generation configuration
        """
        super().__init__(config)
        self._feature_stats: Dict[str, Any] = {}
    
    def fit(self, data: pd.DataFrame, **kwargs: Any) -> "FeatureFactory":
        """
        Fit feature factory to data.
        
        Learns statistics needed for feature generation.
        """
        self._feature_stats = {
            "numeric_cols": data.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_cols": data.select_dtypes(include=["object", "category"]).columns.tolist(),
            "datetime_cols": data.select_dtypes(include=["datetime64"]).columns.tolist(),
        }
        self._fitted = True
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Transform data by generating features."""
        return self.auto_generate(data, **kwargs)
    
    def auto_generate(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        max_features: int = 100,
        include_interactions: bool = True,
        include_polynomials: bool = True,
    ) -> pd.DataFrame:
        """
        Automatically generate features based on data characteristics.
        
        Args:
            data: Input DataFrame
            target: Target column name (excluded from features)
            max_features: Maximum number of features to generate
            include_interactions: Generate interaction features
            include_polynomials: Generate polynomial features
            
        Returns:
            DataFrame with generated features
        """
        logger.info("Starting auto feature generation", shape=data.shape)
        
        result = data.copy()
        
        # Exclude target
        if target and target in result.columns:
            target_col = result.pop(target)
        else:
            target_col = None
        
        # Identify column types
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = result.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = result.select_dtypes(include=["datetime64"]).columns.tolist()
        
        # Generate numeric features
        if numeric_cols:
            numeric_features = self.generate_numeric_features(result[numeric_cols])
            result = pd.concat([result, numeric_features], axis=1)
        
        # Generate categorical features
        if categorical_cols:
            categorical_features = self.generate_categorical_features(result[categorical_cols])
            for col, encoded in categorical_features.items():
                result = pd.concat([result, encoded], axis=1)
        
        # Generate datetime features
        if datetime_cols:
            for col in datetime_cols:
                datetime_features = self.generate_datetime_features(result[col], prefix=col)
                result = pd.concat([result, datetime_features], axis=1)
        
        # Generate interaction features
        if include_interactions and len(numeric_cols) >= 2:
            interactions = self.generate_interactions(
                result[numeric_cols[:10]],  # Limit to prevent explosion
                numeric_cols[:10],
            )
            result = pd.concat([result, interactions], axis=1)
        
        # Generate polynomial features
        if include_polynomials and len(numeric_cols) >= 1:
            polynomials = self.generate_polynomial_features(
                result[numeric_cols[:5]],  # Limit to prevent explosion
                degree=2,
            )
            result = pd.concat([result, polynomials], axis=1)
        
        # Add target back
        if target_col is not None:
            result[target] = target_col
        
        # Limit features if needed
        if len(result.columns) > max_features + (1 if target else 0):
            result = self._limit_features(result, max_features, target)
        
        logger.info(
            "Feature generation complete",
            original_features=len(data.columns),
            generated_features=len(result.columns),
        )
        
        return result
    
    def generate_numeric_features(
        self,
        data: pd.DataFrame,
        include_log: bool = True,
        include_sqrt: bool = True,
        include_reciprocal: bool = True,
    ) -> pd.DataFrame:
        """
        Generate features from numeric columns.
        
        Args:
            data: DataFrame with numeric columns
            include_log: Include log transformations
            include_sqrt: Include square root transformations
            include_reciprocal: Include reciprocal transformations
            
        Returns:
            DataFrame with generated features
        """
        result = pd.DataFrame(index=data.index)
        
        for col in data.columns:
            values = data[col]
            
            # Log transform (for positive values)
            if include_log:
                min_val = values.min()
                if min_val > 0:
                    result[f"{col}_log"] = np.log(values)
                elif min_val >= 0:
                    result[f"{col}_log1p"] = np.log1p(values)
            
            # Square root (for non-negative values)
            if include_sqrt and values.min() >= 0:
                result[f"{col}_sqrt"] = np.sqrt(values)
            
            # Reciprocal (avoiding division by zero)
            if include_reciprocal and values.min() > 0:
                result[f"{col}_reciprocal"] = 1 / values
            
            # Binning
            try:
                result[f"{col}_binned"] = pd.qcut(values, q=5, labels=False, duplicates="drop")
            except:
                pass
        
        return result
    
    def generate_categorical_features(
        self,
        data: pd.DataFrame,
        encoding: str = "onehot",
        max_categories: int = 20,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate features from categorical columns.
        
        Args:
            data: DataFrame with categorical columns
            encoding: Encoding method ("onehot", "label", "count")
            max_categories: Max categories for one-hot encoding
            
        Returns:
            Dictionary of column name to encoded DataFrame
        """
        result = {}
        
        for col in data.columns:
            n_unique = data[col].nunique()
            
            if encoding == "onehot" and n_unique <= max_categories:
                # One-hot encoding
                dummies = pd.get_dummies(data[col], prefix=col, dummy_na=True)
                result[col] = dummies
            elif encoding == "label" or n_unique > max_categories:
                # Label encoding
                encoded = data[col].astype("category").cat.codes
                result[col] = pd.DataFrame({f"{col}_encoded": encoded}, index=data.index)
            elif encoding == "count":
                # Count encoding
                counts = data[col].value_counts()
                result[col] = pd.DataFrame(
                    {f"{col}_count": data[col].map(counts)},
                    index=data.index,
                )
        
        return result
    
    def generate_datetime_features(
        self,
        series: pd.Series,
        prefix: str = "",
    ) -> pd.DataFrame:
        """
        Generate features from datetime column.
        
        Args:
            series: Datetime series
            prefix: Prefix for feature names
            
        Returns:
            DataFrame with datetime features
        """
        dt = pd.to_datetime(series)
        prefix = f"{prefix}_" if prefix else ""
        
        result = pd.DataFrame(index=series.index)
        
        # Basic components
        result[f"{prefix}year"] = dt.dt.year
        result[f"{prefix}month"] = dt.dt.month
        result[f"{prefix}day"] = dt.dt.day
        result[f"{prefix}dayofweek"] = dt.dt.dayofweek
        result[f"{prefix}hour"] = dt.dt.hour
        result[f"{prefix}minute"] = dt.dt.minute
        
        # Derived features
        result[f"{prefix}is_weekend"] = dt.dt.dayofweek >= 5
        result[f"{prefix}is_month_start"] = dt.dt.is_month_start
        result[f"{prefix}is_month_end"] = dt.dt.is_month_end
        result[f"{prefix}quarter"] = dt.dt.quarter
        result[f"{prefix}dayofyear"] = dt.dt.dayofyear
        result[f"{prefix}weekofyear"] = dt.dt.isocalendar().week.astype(int)
        
        # Cyclical encoding
        result[f"{prefix}month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
        result[f"{prefix}month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
        result[f"{prefix}hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        result[f"{prefix}hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        
        return result
    
    def generate_interactions(
        self,
        data: pd.DataFrame,
        columns: List[str],
        operations: List[str] = ["multiply", "add", "subtract", "divide"],
    ) -> pd.DataFrame:
        """
        Generate interaction features between columns.
        
        Args:
            data: Input DataFrame
            columns: Columns to generate interactions for
            operations: Interaction operations
            
        Returns:
            DataFrame with interaction features
        """
        result = pd.DataFrame(index=data.index)
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                if "multiply" in operations:
                    result[f"{col1}_x_{col2}"] = data[col1] * data[col2]
                if "add" in operations:
                    result[f"{col1}_+_{col2}"] = data[col1] + data[col2]
                if "subtract" in operations:
                    result[f"{col1}_-_{col2}"] = data[col1] - data[col2]
                if "divide" in operations:
                    # Avoid division by zero
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = data[col1] / data[col2]
                        ratio = ratio.replace([np.inf, -np.inf], np.nan)
                        result[f"{col1}_/_{ col2}"] = ratio
        
        return result
    
    def generate_polynomial_features(
        self,
        data: pd.DataFrame,
        degree: int = 2,
        include_bias: bool = False,
    ) -> pd.DataFrame:
        """
        Generate polynomial features.
        
        Args:
            data: Input DataFrame
            degree: Polynomial degree
            include_bias: Include bias term
            
        Returns:
            DataFrame with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        poly_features = poly.fit_transform(data)
        
        # Get feature names
        feature_names = poly.get_feature_names_out(data.columns)
        
        # Exclude original features
        new_features = [n for n in feature_names if "^" in n or " " in n]
        new_indices = [list(feature_names).index(n) for n in new_features]
        
        result = pd.DataFrame(
            poly_features[:, new_indices],
            columns=new_features,
            index=data.index,
        )
        
        return result
    
    def generate_lag_features(
        self,
        data: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 7, 14, 30],
        group_by: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate lag features for time series.
        
        Args:
            data: Input DataFrame
            columns: Columns to create lags for
            lags: Lag periods
            group_by: Optional grouping column
            
        Returns:
            DataFrame with lag features
        """
        result = pd.DataFrame(index=data.index)
        
        for col in columns:
            for lag in lags:
                if group_by:
                    result[f"{col}_lag_{lag}"] = data.groupby(group_by)[col].shift(lag)
                else:
                    result[f"{col}_lag_{lag}"] = data[col].shift(lag)
        
        return result
    
    def generate_rolling_features(
        self,
        data: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [7, 14, 30],
        aggregations: List[str] = ["mean", "std", "min", "max"],
        group_by: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate rolling window features.
        
        Args:
            data: Input DataFrame
            columns: Columns to create rolling features for
            windows: Window sizes
            aggregations: Aggregation functions
            group_by: Optional grouping column
            
        Returns:
            DataFrame with rolling features
        """
        result = pd.DataFrame(index=data.index)
        
        for col in columns:
            for window in windows:
                if group_by:
                    rolling = data.groupby(group_by)[col].rolling(window)
                else:
                    rolling = data[col].rolling(window)
                
                for agg in aggregations:
                    agg_func = getattr(rolling, agg)
                    values = agg_func()
                    if group_by:
                        values = values.reset_index(level=0, drop=True)
                    result[f"{col}_rolling_{window}_{agg}"] = values
        
        return result
    
    def _limit_features(
        self,
        data: pd.DataFrame,
        max_features: int,
        target: Optional[str],
    ) -> pd.DataFrame:
        """Limit features to max_features using variance-based selection."""
        if target and target in data.columns:
            target_col = data.pop(target)
        else:
            target_col = None
        
        # Calculate variance for numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        variances = numeric_data.var()
        top_features = variances.nlargest(max_features).index.tolist()
        
        # Include non-numeric columns
        non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
        selected = top_features + non_numeric
        
        result = data[selected[:max_features]]
        
        if target_col is not None:
            result[target] = target_col
        
        return result


class AutoFeatureGenerator:
    """
    Automated feature generator using Featuretools-like API.
    
    Example:
        >>> generator = AutoFeatureGenerator()
        >>> features = generator.generate(df, target="label", max_depth=2)
    """
    
    def __init__(
        self,
        max_depth: int = 2,
        aggregation_functions: Optional[List[str]] = None,
        transform_functions: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize auto feature generator.
        
        Args:
            max_depth: Maximum depth for feature generation
            aggregation_functions: Aggregation functions for grouped features
            transform_functions: Transform functions for single features
        """
        self.max_depth = max_depth
        self.aggregation_functions = aggregation_functions or [
            "sum", "mean", "std", "min", "max", "count"
        ]
        self.transform_functions = transform_functions or [
            "log", "sqrt", "square"
        ]
    
    def generate(
        self,
        data: pd.DataFrame,
        target: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate features automatically.
        
        Args:
            data: Input DataFrame
            target: Target column name
            entity_id: Entity identifier column
            
        Returns:
            DataFrame with generated features
        """
        factory = FeatureFactory()
        return factory.auto_generate(data, target=target)
