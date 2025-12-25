"""
Model Explainer Module
SHAP values, feature importance, and model interpretability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Model explanation result"""
    feature_importance: Dict[str, float]
    top_features: List[Tuple[str, float]]
    shap_values: Optional[np.ndarray]
    shap_summary: Optional[Dict]
    partial_dependence: Optional[Dict]
    interaction_effects: Optional[Dict]


class ModelExplainer:
    """
    Model Explanation and Interpretability
    SHAP values, feature importance, and partial dependence
    """
    
    def __init__(self):
        self.explanation: Optional[ExplanationResult] = None
        self.shap_explainer = None
        
    def explain(self, model: Any, X: pd.DataFrame, 
                feature_names: List[str] = None,
                method: str = 'auto') -> ExplanationResult:
        """Generate comprehensive model explanations"""
        if feature_names is None:
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        
        # Get feature importance
        importance = self._get_feature_importance(model, X, feature_names, method)
        
        # Get top features
        top_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        
        # Try SHAP values
        shap_values, shap_summary = self._compute_shap_values(model, X, feature_names)
        
        # Compute partial dependence for top features
        partial_dependence = self._compute_partial_dependence(
            model, X, [f[0] for f in top_features[:5]], feature_names
        )
        
        # Compute interaction effects
        interaction_effects = self._compute_interactions(model, X, feature_names)
        
        self.explanation = ExplanationResult(
            feature_importance=importance,
            top_features=top_features,
            shap_values=shap_values,
            shap_summary=shap_summary,
            partial_dependence=partial_dependence,
            interaction_effects=interaction_effects
        )
        
        logger.info(f"Generated explanations with {len(importance)} features")
        return self.explanation
    
    def _get_feature_importance(self, model: Any, X: pd.DataFrame,
                                  feature_names: List[str],
                                  method: str) -> Dict[str, float]:
        """Extract feature importance from model"""
        importance = {}
        
        # Try tree-based feature importance
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            importance = dict(zip(feature_names, imp.tolist()))
        
        # Try coefficient-based importance (linear models)
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if len(coef.shape) > 1:
                coef = np.mean(np.abs(coef), axis=0)
            importance = dict(zip(feature_names, np.abs(coef).tolist()))
        
        # Fallback: permutation importance
        else:
            try:
                from sklearn.inspection import permutation_importance
                X_array = X.values if hasattr(X, 'values') else X
                result = permutation_importance(model, X_array[:1000], np.zeros(min(1000, len(X_array))), 
                                               n_repeats=5, random_state=42)
                importance = dict(zip(feature_names, result.importances_mean.tolist()))
            except Exception as e:
                logger.warning(f"Permutation importance failed: {e}")
                importance = {name: 0 for name in feature_names}
        
        # Normalize
        total = sum(abs(v) for v in importance.values())
        if total > 0:
            importance = {k: abs(v) / total for k, v in importance.items()}
        
        return importance
    
    def _compute_shap_values(self, model: Any, X: pd.DataFrame,
                              feature_names: List[str]) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Compute SHAP values if available"""
        try:
            import shap
            
            X_array = X.values if hasattr(X, 'values') else X
            X_sample = X_array[:100]  # Use sample for speed
            
            # Select appropriate explainer
            model_type = type(model).__name__.lower()
            
            if 'tree' in model_type or 'forest' in model_type or 'boost' in model_type:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, X_sample[:10])
            
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-class
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Create summary
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            shap_summary = {
                'mean_abs_shap': dict(zip(feature_names, mean_abs_shap.tolist())),
                'top_shap_features': sorted(
                    zip(feature_names, mean_abs_shap.tolist()),
                    key=lambda x: x[1], reverse=True
                )[:10]
            }
            
            self.shap_explainer = explainer
            logger.info("SHAP values computed successfully")
            return shap_values, shap_summary
            
        except ImportError:
            logger.info("SHAP not installed. Skipping SHAP analysis.")
            return None, None
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return None, None
    
    def _compute_partial_dependence(self, model: Any, X: pd.DataFrame,
                                     features: List[str],
                                     feature_names: List[str]) -> Dict[str, Dict]:
        """Compute partial dependence for features"""
        try:
            from sklearn.inspection import partial_dependence
            
            X_array = X.values if hasattr(X, 'values') else X
            pd_results = {}
            
            for feature in features:
                if feature not in feature_names:
                    continue
                    
                feature_idx = feature_names.index(feature)
                
                try:
                    result = partial_dependence(
                        model, X_array, features=[feature_idx],
                        kind='average', grid_resolution=20
                    )
                    
                    pd_results[feature] = {
                        'values': result['grid_values'][0].tolist(),
                        'average': result['average'][0].tolist()
                    }
                except Exception as e:
                    logger.warning(f"PDP failed for {feature}: {e}")
            
            return pd_results
            
        except Exception as e:
            logger.warning(f"Partial dependence computation failed: {e}")
            return {}
    
    def _compute_interactions(self, model: Any, X: pd.DataFrame,
                               feature_names: List[str]) -> Dict[str, float]:
        """Compute feature interaction strengths"""
        interactions = {}
        
        try:
            # Use feature importance to identify potential interactions
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                top_idx = np.argsort(importance)[-5:]  # Top 5 features
                
                # Calculate pairwise correlation as proxy for interaction
                X_array = X.values if hasattr(X, 'values') else X
                for i in range(len(top_idx)):
                    for j in range(i + 1, len(top_idx)):
                        idx1, idx2 = top_idx[i], top_idx[j]
                        if idx1 < len(feature_names) and idx2 < len(feature_names):
                            corr = np.corrcoef(X_array[:, idx1], X_array[:, idx2])[0, 1]
                            interaction_strength = abs(corr) * (importance[idx1] + importance[idx2]) / 2
                            key = f"{feature_names[idx1]} × {feature_names[idx2]}"
                            interactions[key] = float(interaction_strength)
        except Exception as e:
            logger.warning(f"Interaction computation failed: {e}")
        
        return interactions
    
    def explain_prediction(self, model: Any, X_instance: np.ndarray,
                            feature_names: List[str]) -> Dict[str, Any]:
        """Explain a single prediction"""
        explanation = {
            'prediction': None,
            'probability': None,
            'feature_contributions': {},
            'top_positive': [],
            'top_negative': []
        }
        
        # Get prediction
        pred = model.predict(X_instance.reshape(1, -1))[0]
        explanation['prediction'] = pred
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_instance.reshape(1, -1))[0]
            explanation['probability'] = proba.tolist()
        
        # Try SHAP for instance explanation
        if self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(X_instance.reshape(1, -1))
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                shap_values = shap_values.flatten()
                
                contributions = dict(zip(feature_names, shap_values.tolist()))
                explanation['feature_contributions'] = contributions
                
                sorted_contrib = sorted(contributions.items(), key=lambda x: x[1])
                explanation['top_negative'] = sorted_contrib[:3]
                explanation['top_positive'] = sorted_contrib[-3:][::-1]
            except Exception as e:
                logger.warning(f"Instance explanation failed: {e}")
        
        return explanation
    
    def get_importance_df(self) -> pd.DataFrame:
        """Get feature importance as DataFrame"""
        if self.explanation is None:
            return pd.DataFrame()
        
        data = [
            {'Feature': k, 'Importance': v}
            for k, v in self.explanation.feature_importance.items()
        ]
        
        return pd.DataFrame(data).sort_values('Importance', ascending=False)
    
    def generate_explanation_report(self) -> str:
        """Generate text explanation report"""
        if self.explanation is None:
            return "No explanation available. Run explain() first."
        
        report = []
        report.append("=" * 50)
        report.append("MODEL EXPLANATION REPORT")
        report.append("=" * 50)
        
        # Top features
        report.append("\n📊 TOP FEATURES BY IMPORTANCE:")
        report.append("-" * 30)
        for i, (feature, importance) in enumerate(self.explanation.top_features[:10], 1):
            bar = "█" * int(importance * 20)
            report.append(f"{i:2}. {feature:20} {bar} {importance:.3f}")
        
        # SHAP summary
        if self.explanation.shap_summary:
            report.append("\n🔍 SHAP ANALYSIS:")
            report.append("-" * 30)
            for feature, value in self.explanation.shap_summary['top_shap_features'][:5]:
                report.append(f"   {feature}: {value:.4f}")
        
        # Interactions
        if self.explanation.interaction_effects:
            report.append("\n🔗 KEY INTERACTIONS:")
            report.append("-" * 30)
            sorted_int = sorted(self.explanation.interaction_effects.items(), 
                              key=lambda x: x[1], reverse=True)[:5]
            for interaction, strength in sorted_int:
                report.append(f"   {interaction}: {strength:.4f}")
        
        report.append("\n" + "=" * 50)
        
        return "\n".join(report)
