"""
Professional Model Interpreter - Explains Model Decisions
Uses SHAP and other methods for interpretability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class ModelInterpreter:
    """
    Professional Model Interpreter - explains model predictions.
    Makes ML models interpretable for business users.
    """
    
    def __init__(self):
        self.feature_importance = None
        self.shap_values = None
        self.interpretation_log = []
        
    def log(self, message: str):
        """Add to log."""
        self.interpretation_log.append(f"🔍 {message}")
    
    def interpret(self, model, X: pd.DataFrame, y: Optional[pd.Series] = None,
                  feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Interpret model predictions.
        
        Args:
            model: Trained model
            X: Feature data
            y: Target data (optional)
            feature_names: List of feature names
            
        Returns:
            Dictionary with interpretation results
        """
        if feature_names is None:
            feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])]
        
        results = {
            'feature_importance': None,
            'top_features': [],
            'shap_available': HAS_SHAP,
            'interpretation': ''
        }
        
        # Get built-in feature importance
        results['feature_importance'] = self._get_feature_importance(model, feature_names)
        
        if results['feature_importance'] is not None:
            results['top_features'] = results['feature_importance'].head(10)['Feature'].tolist()
        
        # SHAP values if available
        if HAS_SHAP:
            shap_results = self._get_shap_values(model, X, feature_names)
            results.update(shap_results)
        
        # Generate interpretation text
        results['interpretation'] = self._generate_interpretation(results, feature_names)
        
        return results
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """Extract feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).flatten()
            else:
                self.log("Model doesn't have feature importance attribute")
                return None
            
            # Ensure lengths match
            min_len = min(len(feature_names), len(importance))
            
            df = pd.DataFrame({
                'Feature': feature_names[:min_len],
                'Importance': importance[:min_len]
            }).sort_values('Importance', ascending=False).reset_index(drop=True)
            
            # Normalize
            df['Importance_Pct'] = (df['Importance'] / df['Importance'].sum() * 100).round(2)
            
            self.feature_importance = df
            self.log(f"Extracted importance for {len(df)} features")
            
            return df
            
        except Exception as e:
            self.log(f"Failed to get feature importance: {str(e)[:50]}")
            return None
    
    def _get_shap_values(self, model, X: pd.DataFrame, 
                         feature_names: List[str]) -> Dict[str, Any]:
        """Calculate SHAP values for interpretation."""
        results = {}
        
        try:
            # Prepare data
            X_clean = X.fillna(0)
            
            # Use sample for large datasets
            if len(X_clean) > 1000:
                X_sample = X_clean.sample(n=1000, random_state=42)
            else:
                X_sample = X_clean
            
            # Determine explainer type based on model
            model_type = type(model).__name__.lower()
            
            if 'tree' in model_type or 'forest' in model_type or 'xgb' in model_type or 'lgbm' in model_type or 'catboost' in model_type:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            else:
                # Use kernel explainer for other models (slower)
                explainer = shap.KernelExplainer(model.predict, shap.sample(X_sample, 100))
                shap_values = explainer.shap_values(X_sample[:100])
            
            # Handle multi-class
            if isinstance(shap_values, list):
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)
            
            # Calculate mean absolute SHAP
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            shap_importance = pd.DataFrame({
                'Feature': feature_names[:len(mean_shap)],
                'SHAP_Importance': mean_shap
            }).sort_values('SHAP_Importance', ascending=False).reset_index(drop=True)
            
            results['shap_importance'] = shap_importance
            results['shap_values'] = shap_values
            
            self.shap_values = shap_values
            self.log("Calculated SHAP values successfully")
            
        except Exception as e:
            self.log(f"SHAP calculation failed: {str(e)[:50]}")
            results['shap_error'] = str(e)
        
        return results
    
    def _generate_interpretation(self, results: Dict, feature_names: List[str]) -> str:
        """Generate human-readable interpretation."""
        interpretation = []
        
        if results['feature_importance'] is not None:
            top_features = results['feature_importance'].head(5)
            
            interpretation.append("📊 **أهم الميزات المؤثرة في النموذج:**\n")
            
            for i, row in top_features.iterrows():
                feature = row['Feature']
                pct = row.get('Importance_Pct', row['Importance'] * 100)
                interpretation.append(f"{i+1}. **{feature}** - {pct:.1f}%")
            
            interpretation.append("\n")
            
            # Add insight
            top_feature = top_features.iloc[0]['Feature']
            interpretation.append(f"💡 العامل الأهم هو **{top_feature}** ويستحق التركيز عليه.")
        
        if results.get('shap_importance') is not None:
            interpretation.append("\n🔬 **تحليل SHAP المتقدم:**")
            interpretation.append("SHAP يوضح كيف يؤثر كل عامل على التنبؤ بشكل فردي.")
        
        return "\n".join(interpretation)
    
    def get_feature_summary(self, n_top: int = 10, lang: str = 'ar') -> str:
        """Get summary of top features."""
        if self.feature_importance is None:
            return "No feature importance available"
        
        top = self.feature_importance.head(n_top)
        
        if lang == 'ar':
            summary = "**أهم الميزات:**\n"
            for i, row in top.iterrows():
                summary += f"• {row['Feature']}: {row['Importance_Pct']:.1f}%\n"
        else:
            summary = "**Top Features:**\n"
            for i, row in top.iterrows():
                summary += f"• {row['Feature']}: {row['Importance_Pct']:.1f}%\n"
        
        return summary
    
    def explain_prediction(self, model, X_single: pd.DataFrame, 
                          feature_names: List[str]) -> Dict[str, Any]:
        """Explain a single prediction."""
        results = {
            'prediction': None,
            'confidence': None,
            'top_contributors': []
        }
        
        try:
            # Get prediction
            pred = model.predict(X_single)[0]
            results['prediction'] = pred
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_single)[0]
                results['confidence'] = float(max(proba))
            
            # Get SHAP for single prediction
            if HAS_SHAP and hasattr(model, 'feature_importances_'):
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_single)
                
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[0]
                
                # Get top contributors
                contributions = pd.DataFrame({
                    'Feature': feature_names[:len(shap_vals[0])],
                    'Contribution': shap_vals[0]
                }).sort_values('Contribution', key=abs, ascending=False)
                
                results['top_contributors'] = contributions.head(5).to_dict('records')
                
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def get_log(self) -> List[str]:
        """Get interpretation log."""
        return self.interpretation_log


def interpret_model(model, X: pd.DataFrame, 
                   feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convenience function for model interpretation."""
    interpreter = ModelInterpreter()
    return interpreter.interpret(model, X, feature_names=feature_names)
