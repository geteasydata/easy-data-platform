"""
Imbalanced Data Handler - Professional Class Imbalance Treatment
Handles imbalanced datasets like a senior data scientist
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Try to import imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTETomek
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


class ImbalanceHandler:
    """
    Professional Imbalance Handler - automatically detects and fixes class imbalance.
    Uses best practices for handling imbalanced data.
    """
    
    def __init__(self):
        self.imbalance_ratio = None
        self.is_imbalanced = False
        self.method_used = None
        self.log_messages = []
        
    def log(self, message: str):
        """Add to log."""
        self.log_messages.append(f"⚖️ {message}")
    
    def check_imbalance(self, y: pd.Series, threshold: float = 3.0) -> bool:
        """
        Check if target is imbalanced.
        
        Args:
            y: Target series
            threshold: Ratio threshold for considering imbalanced
            
        Returns:
            True if imbalanced
        """
        if y.dtype not in ['object', 'category'] and y.nunique() > 20:
            # Regression - not applicable
            return False
        
        value_counts = y.value_counts()
        
        if len(value_counts) < 2:
            return False
        
        self.imbalance_ratio = value_counts.max() / value_counts.min()
        self.is_imbalanced = self.imbalance_ratio > threshold
        
        if self.is_imbalanced:
            self.log(f"Data is imbalanced (ratio: {self.imbalance_ratio:.1f})")
            self.log(f"Class distribution: {value_counts.to_dict()}")
        
        return self.is_imbalanced
    
    def balance_data(self, X: pd.DataFrame, y: pd.Series, 
                     method: str = 'auto') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance the dataset using appropriate method.
        
        Args:
            X: Features DataFrame
            y: Target Series
            method: 'smote', 'adasyn', 'undersample', 'oversample', 'auto'
            
        Returns:
            (X_balanced, y_balanced)
        """
        # Check if imbalanced first
        if not self.check_imbalance(y):
            self.log("Data is already balanced, no action needed")
            return X, y
        
        if not HAS_IMBLEARN:
            self.log("imbalanced-learn not installed, using class weights instead")
            return self._manual_oversample(X, y)
        
        # Auto-select method
        if method == 'auto':
            if self.imbalance_ratio > 10:
                method = 'smote_tomek'  # Highly imbalanced
            elif len(X) < 1000:
                method = 'smote'  # Small dataset
            else:
                method = 'random_under'  # Large dataset
        
        # Prepare data
        X_clean = X.fillna(0)
        y_clean = y.copy()
        
        try:
            if method == 'smote':
                sampler = SMOTE(random_state=42)
                self.method_used = 'SMOTE'
            elif method == 'adasyn':
                sampler = ADASYN(random_state=42)
                self.method_used = 'ADASYN'
            elif method == 'borderline':
                sampler = BorderlineSMOTE(random_state=42)
                self.method_used = 'BorderlineSMOTE'
            elif method == 'random_over':
                sampler = RandomOverSampler(random_state=42)
                self.method_used = 'RandomOverSampler'
            elif method == 'random_under':
                sampler = RandomUnderSampler(random_state=42)
                self.method_used = 'RandomUnderSampler'
            elif method == 'smote_tomek':
                sampler = SMOTETomek(random_state=42)
                self.method_used = 'SMOTE + Tomek'
            else:
                sampler = SMOTE(random_state=42)
                self.method_used = 'SMOTE'
            
            X_balanced, y_balanced = sampler.fit_resample(X_clean, y_clean)
            
            # Convert back to DataFrame/Series
            X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
            y_balanced = pd.Series(y_balanced, name=y.name)
            
            self.log(f"Applied {self.method_used}")
            self.log(f"Before: {len(X)} samples → After: {len(X_balanced)} samples")
            
            # Check new distribution
            new_counts = y_balanced.value_counts()
            new_ratio = new_counts.max() / new_counts.min()
            self.log(f"New class distribution: {new_counts.to_dict()} (ratio: {new_ratio:.1f})")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            self.log(f"Balancing failed: {str(e)[:50]}, using fallback")
            return self._manual_oversample(X, y)
    
    def _manual_oversample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Manual random oversampling without imblearn."""
        value_counts = y.value_counts()
        max_count = value_counts.max()
        
        X_balanced_list = []
        y_balanced_list = []
        
        for class_val in value_counts.index:
            class_X = X[y == class_val]
            class_y = y[y == class_val]
            
            n_samples = max_count - len(class_X)
            
            if n_samples > 0:
                # Random oversample
                indices = np.random.choice(class_X.index, size=n_samples, replace=True)
                X_balanced_list.append(class_X)
                X_balanced_list.append(X.loc[indices])
                y_balanced_list.append(class_y)
                y_balanced_list.append(y.loc[indices])
            else:
                X_balanced_list.append(class_X)
                y_balanced_list.append(class_y)
        
        X_balanced = pd.concat(X_balanced_list, ignore_index=True)
        y_balanced = pd.concat(y_balanced_list, ignore_index=True)
        
        self.method_used = 'ManualOversampling'
        self.log(f"Applied manual oversampling: {len(X)} → {len(X_balanced)} samples")
        
        return X_balanced, y_balanced
    
    def get_class_weights(self, y: pd.Series) -> dict:
        """Calculate class weights for model training."""
        value_counts = y.value_counts()
        total = len(y)
        n_classes = len(value_counts)
        
        weights = {}
        for cls, count in value_counts.items():
            weights[cls] = total / (n_classes * count)
        
        self.log(f"Calculated class weights: {weights}")
        return weights
    
    def get_log(self) -> List[str]:
        """Get log messages."""
        return self.log_messages


def balance_data(X: pd.DataFrame, y: pd.Series, 
                method: str = 'auto') -> Tuple[pd.DataFrame, pd.Series]:
    """Convenience function for balancing data."""
    handler = ImbalanceHandler()
    return handler.balance_data(X, y, method)
