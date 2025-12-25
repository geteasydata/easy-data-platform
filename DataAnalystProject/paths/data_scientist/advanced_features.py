"""
Advanced Feature Engineering
Automated feature creation, selection, and transformation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import warnings
from scipy import stats

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PolynomialFeatures, PowerTransformer, QuantileTransformer
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    RFE, VarianceThreshold
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineeringResult:
    """Results from feature engineering"""
    original_features: int
    final_features: int
    new_features_created: int
    features_removed: int
    feature_names: List[str]
    transformations_applied: List[str]
    feature_importance: Dict[str, float] = field(default_factory=dict)


class AdvancedFeatureEngineer:
    """
    Advanced Feature Engineering with:
    - Smart feature creation (interactions, polynomials, aggregations)
    - Target encoding with regularization
    - Time-based features
    - Automatic feature selection
    - Dimensionality reduction
    """
    
    def __init__(self, 
                 max_features: int = 100,
                 create_interactions: bool = True,
                 create_polynomials: bool = True,
                 create_aggregations: bool = True,
                 target_encode: bool = True,
                 apply_transformations: bool = True,
                 feature_selection: bool = True,
                 n_clusters: int = 5):
        
        self.max_features = max_features
        self.create_interactions = create_interactions
        self.create_polynomials = create_polynomials
        self.create_aggregations = create_aggregations
        self.target_encode = target_encode
        self.apply_transformations = apply_transformations
        self.feature_selection = feature_selection
        self.n_clusters = n_clusters
        
        self.created_features: List[str] = []
        self.removed_features: List[str] = []
        self.transformations: List[str] = []
        self.target_encodings: Dict[str, Dict] = {}
        self.scalers: Dict[str, Any] = {}
        
    def fit_transform(self, df: pd.DataFrame, target: str = None) -> pd.DataFrame:
        """
        Apply full feature engineering pipeline
        
        Args:
            df: Input DataFrame
            target: Target column name (optional)
        
        Returns:
            Transformed DataFrame with new features
        """
        original_features = len(df.columns)
        result_df = df.copy()
        
        # Separate target
        y = None
        if target and target in result_df.columns:
            y = result_df[target]
            result_df = result_df.drop(columns=[target])
        
        # Get column types
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = result_df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = result_df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # 1. Handle missing values intelligently
        result_df = self._handle_missing_values(result_df, numeric_cols, categorical_cols)
        
        # 2. Create datetime features
        if datetime_cols:
            result_df = self._create_datetime_features(result_df, datetime_cols)
            self.transformations.append('datetime_features')
        
        # 3. Create interaction features
        if self.create_interactions and len(numeric_cols) >= 2:
            result_df = self._create_interaction_features(result_df, numeric_cols[:10])
            self.transformations.append('interactions')
        
        # 4. Create polynomial features
        if self.create_polynomials and len(numeric_cols) >= 2:
            result_df = self._create_polynomial_features(result_df, numeric_cols[:5])
            self.transformations.append('polynomials')
        
        # 5. Create aggregation features
        if self.create_aggregations and categorical_cols and numeric_cols:
            result_df = self._create_aggregation_features(result_df, categorical_cols[:3], numeric_cols[:5])
            self.transformations.append('aggregations')
        
        # 6. Target encoding
        if self.target_encode and y is not None and categorical_cols:
            result_df = self._apply_target_encoding(result_df, categorical_cols, y)
            self.transformations.append('target_encoding')
        
        # 7. Create cluster features
        if len(numeric_cols) >= 3:
            result_df = self._create_cluster_features(result_df, numeric_cols)
            self.transformations.append('clustering')
        
        # 8. Apply transformations (Yeo-Johnson / Quantile)
        if self.apply_transformations:
            result_df = self._apply_advanced_transforms(result_df)
            self.transformations.append('advanced_transforms')
        
        # 9. Feature selection
        if self.feature_selection and y is not None and len(result_df.columns) > self.max_features:
            result_df = self._select_best_features(result_df, y)
            self.transformations.append('feature_selection')
        
        # Add target back
        if y is not None:
            result_df[target] = y.values
        
        logger.info(f"Feature engineering complete: {original_features} -> {len(result_df.columns)} features")
        
        return result_df
    
    def _handle_missing_values(self, df: pd.DataFrame, numeric_cols: List[str], 
                                categorical_cols: List[str]) -> pd.DataFrame:
        """Handle missing values with MICE/Iterative Imputer + KNN"""
        
        # Advanced Imputation for Numeric
        if numeric_cols:
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa
                from sklearn.impute import IterativeImputer
                from sklearn.impute import KNNImputer
                
                # Use Iterative Imputer (MICE equivalent)
                imputer = IterativeImputer(max_iter=10, random_state=42)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            except ImportError:
                # Fallback to simple median
                for col in numeric_cols:
                     df[col] = df[col].fillna(df[col].median())
        
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna('_missing_')
        
        return df
    
    def _create_datetime_features(self, df: pd.DataFrame, datetime_cols: List[str]) -> pd.DataFrame:
        """Extract features from datetime columns"""
        
        for col in datetime_cols:
            try:
                dt = pd.to_datetime(df[col])
                
                # Basic components
                df[f'{col}_year'] = dt.dt.year
                df[f'{col}_month'] = dt.dt.month
                df[f'{col}_day'] = dt.dt.day
                df[f'{col}_dayofweek'] = dt.dt.dayofweek
                df[f'{col}_hour'] = dt.dt.hour
                
                # Cyclical encoding
                df[f'{col}_month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
                df[f'{col}_month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
                df[f'{col}_day_sin'] = np.sin(2 * np.pi * dt.dt.day / 31)
                df[f'{col}_day_cos'] = np.cos(2 * np.pi * dt.dt.day / 31)
                
                # Is weekend
                df[f'{col}_is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
                
                self.created_features.extend([
                    f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek',
                    f'{col}_hour', f'{col}_month_sin', f'{col}_month_cos',
                    f'{col}_day_sin', f'{col}_day_cos', f'{col}_is_weekend'
                ])
                
                # Drop original datetime column
                df = df.drop(columns=[col])
            except:
                pass
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Create interaction features"""
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Multiplication
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                self.created_features.append(f'{col1}_x_{col2}')
                
                # Division (with safety)
                safe_col2 = df[col2].replace(0, np.nan)
                df[f'{col1}_div_{col2}'] = df[col1] / safe_col2
                df[f'{col1}_div_{col2}'] = df[f'{col1}_div_{col2}'].fillna(0)
                self.created_features.append(f'{col1}_div_{col2}')
                
                # Difference
                df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                self.created_features.append(f'{col1}_minus_{col2}')
        
        return df
    
    def _create_polynomial_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Create polynomial features"""
        
        for col in numeric_cols:
            # Square
            df[f'{col}_squared'] = df[col] ** 2
            self.created_features.append(f'{col}_squared')
            
            # Square root (for positive values)
            if (df[col] >= 0).all():
                df[f'{col}_sqrt'] = np.sqrt(df[col])
                self.created_features.append(f'{col}_sqrt')
            
            # Log (for positive values)
            if (df[col] > 0).all():
                df[f'{col}_log'] = np.log1p(df[col])
                self.created_features.append(f'{col}_log')
        
        return df
    
    def _create_aggregation_features(self, df: pd.DataFrame, categorical_cols: List[str], 
                                      numeric_cols: List[str]) -> pd.DataFrame:
        """Create aggregation features (group statistics)"""
        
        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                # Group mean
                group_mean = df.groupby(cat_col)[num_col].transform('mean')
                df[f'{num_col}_by_{cat_col}_mean'] = group_mean
                self.created_features.append(f'{num_col}_by_{cat_col}_mean')
                
                # Group std
                group_std = df.groupby(cat_col)[num_col].transform('std').fillna(0)
                df[f'{num_col}_by_{cat_col}_std'] = group_std
                self.created_features.append(f'{num_col}_by_{cat_col}_std')
                
                # Deviation from group mean
                df[f'{num_col}_dev_from_{cat_col}'] = df[num_col] - group_mean
                self.created_features.append(f'{num_col}_dev_from_{cat_col}')
        
        return df
    
    def _apply_target_encoding(self, df: pd.DataFrame, categorical_cols: List[str], 
                                y: pd.Series) -> pd.DataFrame:
        """Apply target encoding with smoothing"""
        
        global_mean = y.mean()
        
        for col in categorical_cols:
            # Calculate target statistics per category
            stats_df = df.groupby(col).agg({col: 'count'}).rename(columns={col: 'count'})
            stats_df['sum'] = df.groupby(col).apply(lambda x: y.loc[x.index].sum())
            
            # Smoothing parameter
            m = 10
            
            # Calculate smoothed mean
            stats_df['smoothed_mean'] = (stats_df['sum'] + m * global_mean) / (stats_df['count'] + m)
            
            # Map to dataframe
            encoding_map = stats_df['smoothed_mean'].to_dict()
            self.target_encodings[col] = encoding_map
            
            df[f'{col}_target_encoded'] = df[col].map(encoding_map).fillna(global_mean)
            self.created_features.append(f'{col}_target_encoded')
        
        return df
    
    def _create_cluster_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Create cluster-based features"""
        
        try:
            # Scale features for clustering
            X_cluster = df[numeric_cols].copy()
            X_cluster = X_cluster.fillna(X_cluster.median())
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            
            # K-Means clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(X_scaled)
            self.created_features.append('cluster')
            
            # Distance to cluster centers
            distances = kmeans.transform(X_scaled)
            for i in range(self.n_clusters):
                df[f'dist_to_cluster_{i}'] = distances[:, i]
                self.created_features.append(f'dist_to_cluster_{i}')
            
            # Distance to nearest cluster
            df['dist_to_nearest_cluster'] = distances.min(axis=1)
            self.created_features.append('dist_to_nearest_cluster')
        except:
            pass
        
        return df
    
    def _apply_advanced_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Yeo-Johnson and Quantile Transforms"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                skewness = df[col].skew()
                
                # Check for high skewness
                if abs(skewness) > 1:
                    # 1. Try Yeo-Johnson
                    if (df[col] > 0).all():
                        pt = PowerTransformer(method='yeo-johnson')
                        new_col_name = f"{col}_yeo"
                        df[new_col_name] = pt.fit_transform(df[[col]])
                        self.created_features.append(new_col_name)
                    
                    # 2. Try Quantile Transform (Gaussian output)
                    qt = QuantileTransformer(output_distribution='normal', random_state=42)
                    new_col_name = f"{col}_quantile"
                    df[new_col_name] = qt.fit_transform(df[[col]])
                    self.created_features.append(new_col_name)
                    
            except:
                pass
        
        return df
    
    def _select_best_features(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select best features using mutual information"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) <= self.max_features:
            return df
        
        # Determine task type
        is_classification = y.nunique() <= 10 or y.dtype == 'object'
        
        # Calculate mutual information
        try:
            if is_classification:
                mi_scores = mutual_info_classif(df[numeric_cols].fillna(0), y, random_state=42)
            else:
                mi_scores = mutual_info_regression(df[numeric_cols].fillna(0), y, random_state=42)
            
            # Select top features
            feature_scores = dict(zip(numeric_cols, mi_scores))
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in sorted_features[:self.max_features]]
            
            # Keep non-numeric columns
            non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
            
            self.removed_features = [f for f in numeric_cols if f not in top_features]
            
            return df[top_features + non_numeric]
        except:
            return df
    
    def get_summary(self) -> FeatureEngineeringResult:
        """Get summary of feature engineering"""
        
        return FeatureEngineeringResult(
            original_features=0,  # Set by caller
            final_features=0,  # Set by caller
            new_features_created=len(self.created_features),
            features_removed=len(self.removed_features),
            feature_names=self.created_features,
            transformations_applied=self.transformations
        )
