"""
Model Evaluator Module
Comprehensive model evaluation and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Comprehensive evaluation results"""
    task_type: str
    metrics: Dict[str, float]
    confusion_matrix: Optional[np.ndarray]
    classification_report: Optional[str]
    roc_data: Optional[Dict]
    pr_data: Optional[Dict]
    learning_curve_data: Optional[Dict]
    validation_curve_data: Optional[Dict]
    residual_analysis: Optional[Dict]
    cross_validation: Dict[str, float]


class ModelEvaluator:
    """
    Comprehensive Model Evaluator
    Provides detailed evaluation metrics, curves, and diagnostics
    """
    
    def __init__(self, cv_folds: int = 5):
        self.cv_folds = cv_folds
        self.results: Optional[EvaluationResults] = None
        
    def evaluate(self, model: Any, X_train: np.ndarray, X_test: np.ndarray,
                 y_train: np.ndarray, y_test: np.ndarray,
                 task_type: str = 'classification') -> EvaluationResults:
        """Perform comprehensive model evaluation"""
        logger.info(f"Starting comprehensive evaluation for {task_type} task")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        
        # Calculate metrics based on task type
        if task_type == 'classification':
            results = self._evaluate_classification(
                model, X_train, X_test, y_train, y_test, y_pred
            )
        else:
            results = self._evaluate_regression(
                model, X_train, X_test, y_train, y_test, y_pred, y_pred_train
            )
        
        self.results = results
        return results
    
    def _evaluate_classification(self, model, X_train, X_test, 
                                  y_train, y_test, y_pred) -> EvaluationResults:
        """Evaluate classification model"""
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred)
        
        # ROC curve (binary classification)
        roc_data = None
        if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            roc_data = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': auc(fpr, tpr)
            }
            metrics['roc_auc'] = roc_data['auc']
        
        # Precision-Recall curve
        pr_data = None
        if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
            y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
            pr_data = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist(),
                'avg_precision': average_precision_score(y_test, y_proba)
            }
            metrics['avg_precision'] = pr_data['avg_precision']
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='accuracy')
        cv_results = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        # Learning curve
        learning_data = self._compute_learning_curve(model, X_train, y_train, 'accuracy')
        
        return EvaluationResults(
            task_type='classification',
            metrics=metrics,
            confusion_matrix=cm,
            classification_report=report,
            roc_data=roc_data,
            pr_data=pr_data,
            learning_curve_data=learning_data,
            validation_curve_data=None,
            residual_analysis=None,
            cross_validation=cv_results
        )
    
    def _evaluate_regression(self, model, X_train, X_test,
                              y_train, y_test, y_pred, y_pred_train) -> EvaluationResults:
        """Evaluate regression model"""
        # Basic metrics
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_train': r2_score(y_train, y_pred_train),
            'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        }
        
        # Residual analysis
        residuals = y_test - y_pred
        residual_analysis = {
            'residuals': residuals.tolist() if len(residuals) < 1000 else residuals[:1000].tolist(),
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'max_residual': np.max(np.abs(residuals)),
            'predictions': y_pred.tolist() if len(y_pred) < 1000 else y_pred[:1000].tolist(),
            'actuals': y_test.tolist() if len(y_test) < 1000 else y_test[:1000].tolist()
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='r2')
        cv_results = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        # Learning curve
        learning_data = self._compute_learning_curve(model, X_train, y_train, 'r2')
        
        return EvaluationResults(
            task_type='regression',
            metrics=metrics,
            confusion_matrix=None,
            classification_report=None,
            roc_data=None,
            pr_data=None,
            learning_curve_data=learning_data,
            validation_curve_data=None,
            residual_analysis=residual_analysis,
            cross_validation=cv_results
        )
    
    def _compute_learning_curve(self, model, X, y, scoring: str,
                                 train_sizes: np.ndarray = None) -> Dict:
        """Compute learning curve data"""
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        try:
            train_sizes_abs, train_scores, test_scores = learning_curve(
                model, X, y, train_sizes=train_sizes, cv=self.cv_folds,
                scoring=scoring, n_jobs=-1
            )
            
            return {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores_mean': train_scores.mean(axis=1).tolist(),
                'train_scores_std': train_scores.std(axis=1).tolist(),
                'test_scores_mean': test_scores.mean(axis=1).tolist(),
                'test_scores_std': test_scores.std(axis=1).tolist()
            }
        except Exception as e:
            logger.warning(f"Learning curve computation failed: {e}")
            return None
    
    def compare_models(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray,
                       task_type: str = 'classification') -> pd.DataFrame:
        """Compare multiple models"""
        results = []
        scoring = 'accuracy' if task_type == 'classification' else 'r2'
        
        for name, model in models.items():
            try:
                cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=scoring)
                results.append({
                    'Model': name,
                    'CV Mean': round(cv_scores.mean(), 4),
                    'CV Std': round(cv_scores.std(), 4),
                    'CV Min': round(cv_scores.min(), 4),
                    'CV Max': round(cv_scores.max(), 4)
                })
            except Exception as e:
                logger.warning(f"Failed to evaluate {name}: {e}")
        
        return pd.DataFrame(results).sort_values('CV Mean', ascending=False)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation metrics"""
        if self.results is None:
            return {"error": "No evaluation performed"}
        
        summary = {
            'task_type': self.results.task_type,
            'metrics': {k: round(v, 4) for k, v in self.results.metrics.items()},
            'cross_validation': {
                'mean': round(self.results.cross_validation['cv_mean'], 4),
                'std': round(self.results.cross_validation['cv_std'], 4)
            }
        }
        
        if self.results.task_type == 'classification':
            summary['confusion_matrix'] = self.results.confusion_matrix.tolist()
            if self.results.roc_data:
                summary['roc_auc'] = round(self.results.roc_data['auc'], 4)
        else:
            summary['residual_stats'] = {
                'mean': round(self.results.residual_analysis['mean_residual'], 4),
                'std': round(self.results.residual_analysis['std_residual'], 4)
            }
        
        return summary
    
    def check_overfitting(self) -> Dict[str, Any]:
        """Check for signs of overfitting"""
        if self.results is None:
            return {"error": "No evaluation performed"}
        
        if self.results.task_type == 'classification':
            train_score = self.results.metrics.get('accuracy', 0)
            test_score = self.results.cross_validation['cv_mean']
        else:
            train_score = self.results.metrics.get('r2_train', 0)
            test_score = self.results.metrics.get('r2', 0)
        
        gap = train_score - test_score
        
        return {
            'train_score': round(train_score, 4),
            'test_score': round(test_score, 4),
            'gap': round(gap, 4),
            'is_overfitting': gap > 0.1,
            'severity': 'high' if gap > 0.2 else 'moderate' if gap > 0.1 else 'low',
            'recommendation': self._get_overfitting_recommendation(gap)
        }
    
    def _get_overfitting_recommendation(self, gap: float) -> str:
        """Get recommendation based on overfitting gap"""
        if gap > 0.2:
            return "Severe overfitting detected. Consider: reducing model complexity, adding regularization, increasing training data, or using dropout/early stopping."
        elif gap > 0.1:
            return "Moderate overfitting detected. Consider: cross-validation, pruning, or regularization."
        elif gap > 0.05:
            return "Slight overfitting. Monitor with more data or slight regularization."
        else:
            return "Model appears well-fitted. No significant overfitting detected."
