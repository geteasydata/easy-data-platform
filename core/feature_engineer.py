"""
Professional Feature Engineer - Automatic Feature Creation
Creates intelligent features like a 30+ year expert data scientist
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Professional Feature Engineer - creates intelligent features automatically.
    Mimics what a senior data scientist would do.
    """
    
    def __init__(self):
        self.created_features = []
        self.feature_log = []
        self.original_columns = []
        
    def log(self, message: str):
        """Add to feature log."""
        self.feature_log.append(f"✨ {message}")
    
    def engineer_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Automatically create intelligent features.
        
        Features created:
        - Mathematical transformations
        - Statistical aggregations  
        - Date/time features
        - Text features
        - Interaction features
        """
        df = df.copy()
        self.original_columns = list(df.columns)
        self.created_features = []
        self.feature_log = []
        
        # Separate target if exists
        target = None
        if target_col and target_col in df.columns:
            target = df[target_col]
            df = df.drop(columns=[target_col])
        
        # 1. Mathematical Transformations
        df = self._create_math_features(df)
        
        # 2. Statistical Features
        df = self._create_stat_features(df)
        
        # 3. Date Features
        df = self._create_date_features(df)
        
        # 4. Text Features
        df = self._create_text_features(df)
        
        # 5. Interaction Features
        df = self._create_interaction_features(df)
        
        # 6. Ratio Features
        df = self._create_ratio_features(df)
        
        # Add target back
        if target is not None:
            df[target_col] = target.values
        
        self.log(f"Created {len(self.created_features)} new features")
        
        return df
    
    def _create_math_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create mathematical transformation features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols[:10]:  # Limit to avoid explosion
            # Log transformation (for positive values)
            if (df[col] > 0).all():
                new_col = f"{col}_log"
                df[new_col] = np.log1p(df[col])
                self.created_features.append(new_col)
            
            # Square root (for positive values)
            if (df[col] >= 0).all():
                new_col = f"{col}_sqrt"
                df[new_col] = np.sqrt(df[col])
                self.created_features.append(new_col)
            
            # Square
            if df[col].abs().max() < 1e6:  # Avoid overflow
                new_col = f"{col}_squared"
                df[new_col] = df[col] ** 2
                self.created_features.append(new_col)
        
        if len(numeric_cols) > 0:
            self.log(f"Created math features for {min(10, len(numeric_cols))} columns")
        
        return df
    
    def _create_stat_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features across numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Row-wise statistics
            df['row_sum'] = df[numeric_cols].sum(axis=1)
            df['row_mean'] = df[numeric_cols].mean(axis=1)
            df['row_std'] = df[numeric_cols].std(axis=1)
            df['row_min'] = df[numeric_cols].min(axis=1)
            df['row_max'] = df[numeric_cols].max(axis=1)
            df['row_range'] = df['row_max'] - df['row_min']
            
            self.created_features.extend(['row_sum', 'row_mean', 'row_std', 'row_min', 'row_max', 'row_range'])
            self.log("Created row-wise statistical features")
        
        return df
    
    def _create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from date columns."""
        date_features_created = 0
        
        for col in df.columns:
            # Try to detect dates in object columns
            if df[col].dtype == 'object':
                try:
                    dates = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    if dates.notna().sum() / len(dates) > 0.5:
                        # Extract date components
                        df[f'{col}_year'] = dates.dt.year
                        df[f'{col}_month'] = dates.dt.month
                        df[f'{col}_day'] = dates.dt.day
                        df[f'{col}_dayofweek'] = dates.dt.dayofweek
                        df[f'{col}_quarter'] = dates.dt.quarter
                        df[f'{col}_is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
                        
                        self.created_features.extend([
                            f'{col}_year', f'{col}_month', f'{col}_day',
                            f'{col}_dayofweek', f'{col}_quarter', f'{col}_is_weekend'
                        ])
                        date_features_created += 6
                        
                        # Drop original column
                        df = df.drop(columns=[col])
                except:
                    pass
            
            # Handle datetime columns
            elif 'datetime' in str(df[col].dtype):
                dates = df[col]
                df[f'{col}_year'] = dates.dt.year
                df[f'{col}_month'] = dates.dt.month
                df[f'{col}_day'] = dates.dt.day
                df[f'{col}_dayofweek'] = dates.dt.dayofweek
                df[f'{col}_hour'] = dates.dt.hour if hasattr(dates.dt, 'hour') else 0
                
                self.created_features.extend([
                    f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek'
                ])
                date_features_created += 4
                
                df = df.drop(columns=[col])
        
        if date_features_created > 0:
            self.log(f"Created {date_features_created} date features")
        
        return df
    
    def _create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from text columns."""
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        text_features_created = 0
        
        for col in text_cols[:5]:  # Limit
            try:
                # Length
                df[f'{col}_length'] = df[col].astype(str).str.len()
                
                # Word count
                df[f'{col}_words'] = df[col].astype(str).str.split().str.len()
                
                # Has digits
                df[f'{col}_has_digit'] = df[col].astype(str).str.contains(r'\d', regex=True).astype(int)
                
                self.created_features.extend([f'{col}_length', f'{col}_words', f'{col}_has_digit'])
                text_features_created += 3
            except:
                pass
        
        if text_features_created > 0:
            self.log(f"Created {text_features_created} text features")
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Only create for top important columns (limit to avoid explosion)
        cols_to_interact = numeric_cols[:5]
        interactions_created = 0
        
        for i, col1 in enumerate(cols_to_interact):
            for col2 in cols_to_interact[i+1:]:
                # Multiplication
                new_col = f"{col1}_x_{col2}"
                df[new_col] = df[col1] * df[col2]
                self.created_features.append(new_col)
                interactions_created += 1
                
                # Addition
                new_col = f"{col1}_plus_{col2}"
                df[new_col] = df[col1] + df[col2]
                self.created_features.append(new_col)
                interactions_created += 1
        
        if interactions_created > 0:
            self.log(f"Created {interactions_created} interaction features")
        
        return df
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features between numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Only for original columns, not created ones
        original_numeric = [c for c in numeric_cols if c in self.original_columns][:5]
        ratios_created = 0
        
        for i, col1 in enumerate(original_numeric):
            for col2 in original_numeric[i+1:]:
                # Safe division
                if (df[col2] != 0).any():
                    new_col = f"{col1}_div_{col2}"
                    df[new_col] = df[col1] / df[col2].replace(0, np.nan)
                    df[new_col] = df[new_col].fillna(0)
                    self.created_features.append(new_col)
                    ratios_created += 1
        
        if ratios_created > 0:
            self.log(f"Created {ratios_created} ratio features")
        
        return df
    
    def get_created_features(self) -> List[str]:
        """Get list of created features."""
        return self.created_features
    
    def get_log(self) -> List[str]:
        """Get feature engineering log."""
        return self.feature_log


def engineer_features(df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
    """Convenience function for feature engineering."""
    fe = FeatureEngineer()
    return fe.engineer_features(df, target_col)
