from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base import BaseLogic

class AnalyticalLogic(BaseLogic):
    """
    Layer 2: Analytical Logic (The Investigator).
    Responsible for Exploratory Data Analysis (EDA), outlier detection, and pattern recognition.
    """
    def __init__(self):
        super().__init__("analytical")

    def execute(self, data: pd.DataFrame, context: Dict[str, Any] = None) -> Dict[str, Any]:
        self.logger.info("AnalyticalLogic: Analyzing data patterns...")
        
        results = {
            "null_analysis": data.isnull().mean().to_dict(),
            "shape": data.shape,
            "numeric_cols": data.select_dtypes(include=np.number).columns.tolist(),
            "categorical_cols": data.select_dtypes(exclude=np.number).columns.tolist()
        }
        
        # Simple Outlier Detection (Z-score > 3)
        outliers = {}
        for col in results["numeric_cols"]:
            if data[col].nunique() > 10:
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers[col] = (z_scores > 3).sum()
        
        results["outlier_counts"] = outliers
        self.logger.info(f"AnalyticalLogic: Found outliers in {len(outliers)} columns.")
        
        return results
