"""
Evaluation Metrics for Data Science Master System.

Provides comprehensive metrics for:
- Classification: accuracy, precision, recall, F1, AUC, etc.
- Regression: MSE, RMSE, MAE, R2, MAPE, etc.
- Ranking: NDCG, MRR, MAP
- Clustering: silhouette, Calinski-Harabasz, Davies-Bouldin

Example:
    >>> evaluator = Evaluator(problem_type="classification")
    >>> metrics = evaluator.evaluate(y_true, y_pred)
    >>> print(metrics)
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from data_science_master_system.core.base_classes import BaseEvaluator
from data_science_master_system.core.exceptions import EvaluationError
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)


class Evaluator(BaseEvaluator):
    """
    Unified model evaluator.
    
    Automatically calculates appropriate metrics based on problem type.
    
    Example:
        >>> evaluator = Evaluator(problem_type="classification")
        >>> 
        >>> # Basic evaluation
        >>> metrics = evaluator.evaluate(y_true, y_pred)
        >>> 
        >>> # With probabilities for classification
        >>> metrics = evaluator.evaluate(y_true, y_pred, y_proba=y_proba)
    """
    
    def __init__(
        self,
        problem_type: str = "classification",
        metrics: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize evaluator.
        
        Args:
            problem_type: "classification", "regression", "clustering", "ranking"
            metrics: Specific metrics to calculate (all if None)
        """
        super().__init__()
        self.problem_type = problem_type.lower()
        self.metrics_list = metrics
        self._history: List[Dict[str, float]] = []
    
    def evaluate(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        y_proba: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_proba: Predicted probabilities (classification)
            sample_weight: Sample weights
            
        Returns:
            Dictionary of metric names to values
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if self.problem_type == "classification":
            metrics = ClassificationMetrics.calculate(
                y_true, y_pred, y_proba, sample_weight
            )
        elif self.problem_type == "regression":
            metrics = RegressionMetrics.calculate(
                y_true, y_pred, sample_weight
            )
        elif self.problem_type == "clustering":
            metrics = ClusteringMetrics.calculate(y_true, y_pred)
        else:
            raise EvaluationError(f"Unknown problem type: {self.problem_type}")
        
        # Filter metrics if specific ones requested
        if self.metrics_list:
            metrics = {k: v for k, v in metrics.items() if k in self.metrics_list}
        
        self._history.append(metrics)
        return metrics
    
    def get_history(self) -> pd.DataFrame:
        """Get evaluation history as DataFrame."""
        return pd.DataFrame(self._history)
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics of evaluation history."""
        if not self._history:
            return {}
        
        df = self.get_history()
        return {
            "mean": df.mean().to_dict(),
            "std": df.std().to_dict(),
            "min": df.min().to_dict(),
            "max": df.max().to_dict(),
        }


class ClassificationMetrics:
    """
    Classification metrics calculator.
    """
    
    @staticmethod
    def calculate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate all classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            sample_weight: Sample weights
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, log_loss,
            matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
            confusion_matrix,
        )
        
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        
        # Multi-class handling
        n_classes = len(np.unique(y_true))
        average = "binary" if n_classes == 2 else "weighted"
        
        metrics["precision"] = precision_score(
            y_true, y_pred, average=average, zero_division=0, sample_weight=sample_weight
        )
        metrics["recall"] = recall_score(
            y_true, y_pred, average=average, zero_division=0, sample_weight=sample_weight
        )
        metrics["f1"] = f1_score(
            y_true, y_pred, average=average, zero_division=0, sample_weight=sample_weight
        )
        
        # Additional metrics
        metrics["mcc"] = matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)
        metrics["kappa"] = cohen_kappa_score(y_true, y_pred, sample_weight=sample_weight)
        
        # Probability-based metrics
        if y_proba is not None:
            try:
                if n_classes == 2:
                    proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    metrics["auc_roc"] = roc_auc_score(y_true, proba, sample_weight=sample_weight)
                    metrics["auc_pr"] = average_precision_score(y_true, proba, sample_weight=sample_weight)
                else:
                    metrics["auc_roc"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted", sample_weight=sample_weight
                    )
                
                metrics["log_loss"] = log_loss(y_true, y_proba, sample_weight=sample_weight)
            except Exception as e:
                logger.warning(f"Could not calculate probability metrics: {e}")
        
        # Confusion matrix derived metrics
        cm = confusion_matrix(y_true, y_pred)
        if n_classes == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return metrics
    
    @staticmethod
    def confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = False,
    ) -> np.ndarray:
        """Get confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        return cm
    
    @staticmethod
    def classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> str:
        """Get classification report."""
        from sklearn.metrics import classification_report
        return classification_report(y_true, y_pred, target_names=target_names)


class RegressionMetrics:
    """
    Regression metrics calculator.
    """
    
    @staticmethod
    def calculate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate all regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            sample_weight: Sample weights
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            mean_absolute_percentage_error, explained_variance_score,
            max_error, median_absolute_error,
        )
        
        metrics = {}
        
        # Basic metrics
        metrics["mse"] = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
        metrics["median_ae"] = median_absolute_error(y_true, y_pred)
        metrics["max_error"] = max_error(y_true, y_pred)
        
        # R2 and explained variance
        metrics["r2"] = r2_score(y_true, y_pred, sample_weight=sample_weight)
        metrics["explained_variance"] = explained_variance_score(y_true, y_pred, sample_weight=sample_weight)
        
        # Percentage error metrics (handle zeros)
        try:
            metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight)
        except:
            # Handle division by zero
            mask = y_true != 0
            if mask.any():
                metrics["mape"] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
            else:
                metrics["mape"] = np.nan
        
        # Symmetric MAPE
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if mask.any():
            metrics["smape"] = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])
        else:
            metrics["smape"] = np.nan
        
        # Root Mean Squared Log Error (for positive values)
        if (y_true > 0).all() and (y_pred > 0).all():
            metrics["rmsle"] = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
        
        return metrics


class ClusteringMetrics:
    """
    Clustering metrics calculator.
    """
    
    @staticmethod
    def calculate(
        X: np.ndarray,
        labels: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate clustering metrics.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            true_labels: True labels (optional, for supervised metrics)
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            silhouette_score, calinski_harabasz_score, davies_bouldin_score,
            adjusted_rand_score, normalized_mutual_info_score, homogeneity_score,
            completeness_score, v_measure_score,
        )
        
        metrics = {}
        
        # Unsupervised metrics (require X)
        if X is not None and len(np.unique(labels)) > 1:
            try:
                metrics["silhouette"] = silhouette_score(X, labels)
                metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
                metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
            except Exception as e:
                logger.warning(f"Could not calculate unsupervised metrics: {e}")
        
        # Supervised metrics (require true labels)
        if true_labels is not None:
            metrics["adjusted_rand"] = adjusted_rand_score(true_labels, labels)
            metrics["nmi"] = normalized_mutual_info_score(true_labels, labels)
            metrics["homogeneity"] = homogeneity_score(true_labels, labels)
            metrics["completeness"] = completeness_score(true_labels, labels)
            metrics["v_measure"] = v_measure_score(true_labels, labels)
        
        return metrics


def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    problem_type: str = "classification",
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Convenience function to calculate metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        problem_type: "classification" or "regression"
        y_proba: Predicted probabilities (classification)
        
    Returns:
        Dictionary of metrics
        
    Example:
        >>> metrics = calculate_metrics(y_true, y_pred, "classification")
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    evaluator = Evaluator(problem_type=problem_type)
    return evaluator.evaluate(y_true, y_pred, y_proba)
