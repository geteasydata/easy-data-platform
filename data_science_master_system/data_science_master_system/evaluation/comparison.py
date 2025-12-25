"""
Model Comparison for Data Science Master System.

Provides tools for comparing multiple models:
- Side-by-side metrics comparison
- Statistical significance tests
- Leaderboard generation
- Cross-validation comparison

Example:
    >>> comparison = ModelComparison()
    >>> comparison.add_model("RF", rf_model)
    >>> comparison.add_model("XGB", xgb_model)
    >>> results = comparison.compare(X_test, y_test)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from data_science_master_system.core.logger import get_logger
from data_science_master_system.evaluation.metrics import Evaluator

logger = get_logger(__name__)


class ModelComparison:
    """
    Compare multiple models on the same dataset.
    
    Example:
        >>> comparison = ModelComparison(problem_type="classification")
        >>> comparison.add_model("RandomForest", rf_model)
        >>> comparison.add_model("XGBoost", xgb_model)
        >>> comparison.add_model("LightGBM", lgb_model)
        >>> 
        >>> results = comparison.compare(X_test, y_test)
        >>> leaderboard = comparison.get_leaderboard(metric="f1")
    """
    
    def __init__(
        self,
        problem_type: str = "classification",
        metrics: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize model comparison.
        
        Args:
            problem_type: "classification" or "regression"
            metrics: Specific metrics to compare
        """
        self.problem_type = problem_type
        self.metrics = metrics
        self._models: Dict[str, Any] = {}
        self._results: Dict[str, Dict[str, float]] = {}
        self._cv_results: Dict[str, Dict[str, Any]] = {}
    
    def add_model(
        self,
        name: str,
        model: Any,
        predictions: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a model for comparison.
        
        Args:
            name: Model name for display
            model: Fitted model or None if providing predictions
            predictions: Pre-computed predictions (optional)
            probabilities: Pre-computed probabilities (optional)
        """
        self._models[name] = {
            "model": model,
            "predictions": predictions,
            "probabilities": probabilities,
        }
        logger.info(f"Added model: {name}")
    
    def compare(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """
        Compare all models on test data.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            DataFrame with metrics for each model
        """
        evaluator = Evaluator(problem_type=self.problem_type, metrics=self.metrics)
        
        for name, info in self._models.items():
            model = info["model"]
            predictions = info.get("predictions")
            probabilities = info.get("probabilities")
            
            # Get predictions if not provided
            if predictions is None and model is not None:
                predictions = model.predict(X)
            
            # Get probabilities if not provided
            if probabilities is None and model is not None:
                if hasattr(model, "predict_proba"):
                    try:
                        probabilities = model.predict_proba(X)
                    except:
                        pass
            
            # Calculate metrics
            if predictions is not None:
                metrics = evaluator.evaluate(y, predictions, probabilities)
                self._results[name] = metrics
        
        return self.get_results()
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compare models using cross-validation.
        
        Args:
            X: Features
            y: Target
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            DataFrame with CV results
        """
        from sklearn.model_selection import cross_val_score
        
        if scoring is None:
            scoring = "accuracy" if self.problem_type == "classification" else "neg_mean_squared_error"
        
        for name, info in self._models.items():
            model = info["model"]
            if model is None:
                continue
            
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                self._cv_results[name] = {
                    "mean": scores.mean(),
                    "std": scores.std(),
                    "scores": scores.tolist(),
                }
            except Exception as e:
                logger.warning(f"CV failed for {name}: {e}")
        
        return self.get_cv_results()
    
    def get_results(self) -> pd.DataFrame:
        """Get comparison results as DataFrame."""
        if not self._results:
            return pd.DataFrame()
        
        return pd.DataFrame(self._results).T.reset_index().rename(columns={"index": "model"})
    
    def get_cv_results(self) -> pd.DataFrame:
        """Get cross-validation results as DataFrame."""
        if not self._cv_results:
            return pd.DataFrame()
        
        data = []
        for name, result in self._cv_results.items():
            data.append({
                "model": name,
                "mean_score": result["mean"],
                "std_score": result["std"],
            })
        
        return pd.DataFrame(data)
    
    def get_leaderboard(
        self,
        metric: str = "f1",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Get sorted leaderboard.
        
        Args:
            metric: Metric to sort by
            ascending: Sort order
            
        Returns:
            Sorted DataFrame
        """
        results = self.get_results()
        if results.empty or metric not in results.columns:
            return results
        
        return results.sort_values(metric, ascending=ascending).reset_index(drop=True)
    
    def best_model(self, metric: str = "f1") -> Tuple[str, Any]:
        """
        Get the best model by metric.
        
        Args:
            metric: Metric to use for ranking
            
        Returns:
            Tuple of (model_name, model)
        """
        leaderboard = self.get_leaderboard(metric=metric)
        if leaderboard.empty:
            return None, None
        
        best_name = leaderboard.iloc[0]["model"]
        return best_name, self._models[best_name]["model"]


class StatisticalTests:
    """
    Statistical tests for model comparison.
    
    Example:
        >>> tests = StatisticalTests()
        >>> p_value = tests.mcnemar_test(y_true, pred_a, pred_b)
        >>> is_significant = tests.is_significant(p_value)
    """
    
    @staticmethod
    def paired_ttest(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Paired t-test for comparing CV scores.
        
        Args:
            scores_a: CV scores for model A
            scores_b: CV scores for model B
            
        Returns:
            Tuple of (t-statistic, p-value)
        """
        from scipy import stats
        return stats.ttest_rel(scores_a, scores_b)
    
    @staticmethod
    def wilcoxon_test(
        scores_a: np.ndarray,
        scores_b: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Wilcoxon signed-rank test (non-parametric).
        
        Args:
            scores_a: CV scores for model A
            scores_b: CV scores for model B
            
        Returns:
            Tuple of (statistic, p-value)
        """
        from scipy import stats
        result = stats.wilcoxon(scores_a, scores_b)
        return result.statistic, result.pvalue
    
    @staticmethod
    def mcnemar_test(
        y_true: np.ndarray,
        pred_a: np.ndarray,
        pred_b: np.ndarray,
    ) -> float:
        """
        McNemar's test for comparing classifiers.
        
        Args:
            y_true: True labels
            pred_a: Predictions from model A
            pred_b: Predictions from model B
            
        Returns:
            p-value
        """
        from scipy import stats
        
        # Build contingency table
        correct_a = (pred_a == y_true)
        correct_b = (pred_b == y_true)
        
        # Count disagreements
        b = np.sum(correct_a & ~correct_b)  # A correct, B wrong
        c = np.sum(~correct_a & correct_b)  # A wrong, B correct
        
        # McNemar's test
        if b + c == 0:
            return 1.0
        
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = stats.chi2.sf(chi2, df=1)
        
        return p_value
    
    @staticmethod
    def bootstrap_comparison(
        y_true: np.ndarray,
        pred_a: np.ndarray,
        pred_b: np.ndarray,
        metric_func: callable,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Dict[str, float]:
        """
        Bootstrap comparison of two models.
        
        Args:
            y_true: True labels
            pred_a: Predictions from model A
            pred_b: Predictions from model B
            metric_func: Metric function (y_true, y_pred) -> float
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Dictionary with difference statistics
        """
        n = len(y_true)
        differences = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            
            score_a = metric_func(y_true[indices], pred_a[indices])
            score_b = metric_func(y_true[indices], pred_b[indices])
            
            differences.append(score_a - score_b)
        
        differences = np.array(differences)
        
        alpha = 1 - confidence
        lower = np.percentile(differences, 100 * alpha / 2)
        upper = np.percentile(differences, 100 * (1 - alpha / 2))
        
        return {
            "mean_difference": differences.mean(),
            "std_difference": differences.std(),
            "ci_lower": lower,
            "ci_upper": upper,
            "significant": not (lower <= 0 <= upper),
        }
    
    @staticmethod
    def is_significant(p_value: float, alpha: float = 0.05) -> bool:
        """Check if p-value is significant."""
        return p_value < alpha
