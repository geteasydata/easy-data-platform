"""
What-If Analysis Module
Interactive scenarios simulation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from copy import deepcopy

class WhatIfSimulator:
    """
    Simulation engine for What-If scenarios
    
    Features:
    - Feature modification simulation
    - Prediction impact analysis
    - Sensitivity analysis
    """
    
    def __init__(self, model: Any, features: List[str]):
        """
        Args:
            model: Trained ML model (must have predict method)
            features: List of feature names model was trained on
        """
        self.model = model
        self.features = features
        
    def simulate_scenario(self, 
                          df: pd.DataFrame, 
                          changes: Dict[str, Any]) -> pd.DataFrame:
        """
        Simulate scenario by changing feature values
        
        Args:
            df: Base dataframe
            changes: Dictionary of {feature: change_value} or {feature: lambda x: x * 1.1}
            
        Returns:
            DataFrame with original and new predictions
        """
        # Create modified copy
        scenario_df = df.copy()
        
        # Apply changes
        for feature, change in changes.items():
            if feature in scenario_df.columns:
                if callable(change):
                    scenario_df[feature] = scenario_df[feature].apply(change)
                else:
                    scenario_df[feature] = change
                    
        # Predict original
        try:
            orig_preds = self._predict(df)
            new_preds = self._predict(scenario_df)
            
            result = pd.DataFrame({
                'Original Prediction': orig_preds,
                'New Prediction': new_preds,
                'Difference': new_preds - orig_preds,
                'Percent Change': ((new_preds - orig_preds) / (orig_preds + 1e-10)) * 100
            })
            
            return result
        except Exception as e:
            print(f"Simulation failed: {e}")
            return None
            
    def sensitivity_analysis(self, 
                             df: pd.DataFrame, 
                             feature: str, 
                             min_val: float, 
                             max_val: float, 
                             steps: int = 10) -> pd.DataFrame:
        """
        Analyze prediction sensitivity to one feature
        """
        if feature not in df.columns:
            return None
            
        values = np.linspace(min_val, max_val, steps)
        avg_predictions = []
        
        base_row = df.mean().to_frame().T
        
        for val in values:
            temp_df = base_row.copy()
            temp_df[feature] = val
            pred = self._predict(temp_df)[0]
            avg_predictions.append(pred)
            
        return pd.DataFrame({
            feature: values,
            'Prediction': avg_predictions
        })

    def _predict(self, df):
        """Helper to predict safely"""
        # Ensure only model features are passed
        X = df[self.features]
        return self.model.predict(X)
