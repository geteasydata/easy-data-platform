"""
Traditional ML Models for Data Science Master System.

Provides unified wrappers for:
- Scikit-learn models
- XGBoost, LightGBM, CatBoost
- Ensemble methods

Example:
    >>> model = ClassificationModel("xgboost", n_estimators=100)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> probabilities = model.predict_proba(X_test)
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from data_science_master_system.core.base_classes import BaseModel, ModelState
from data_science_master_system.core.exceptions import ModelError, ModelTrainingError, ModelPredictionError
from data_science_master_system.core.logger import get_logger
from data_science_master_system.models.model_factory import ModelFactory

logger = get_logger(__name__)


class TraditionalMLModel(BaseModel):
    """
    Wrapper for traditional ML models.
    
    Provides unified interface for:
    - Training and prediction
    - Cross-validation
    - Feature importance
    - Model persistence
    
    Example:
        >>> model = TraditionalMLModel("random_forest", problem_type="classification")
        >>> model.fit(X_train, y_train)
        >>> 
        >>> # Get feature importance
        >>> importance = model.feature_importance()
        >>> 
        >>> # Cross-validate
        >>> scores = model.cross_validate(X, y, cv=5)
    """
    
    def __init__(
        self,
        model_name: str = "random_forest",
        problem_type: str = "classification",
        config: Optional[Dict[str, Any]] = None,
        **model_params: Any,
    ) -> None:
        """
        Initialize traditional ML model.
        
        Args:
            model_name: Name of the model
            problem_type: "classification" or "regression"
            config: Additional configuration
            **model_params: Model hyperparameters
        """
        super().__init__(config)
        self.model_name = model_name
        self.problem_type = problem_type
        self.model_params = model_params
        self._model = None
        self._feature_names: List[str] = []
        self._classes: Optional[np.ndarray] = None
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs: Any,
    ) -> "TraditionalMLModel":
        """
        Fit model to data.
        
        Args:
            X: Feature matrix
            y: Target variable
            **kwargs: Additional fit parameters
            
        Returns:
            Self
        """
        logger.info(f"Training {self.model_name}", shape=X.shape if hasattr(X, "shape") else len(X))
        
        try:
            # Create model
            self._model = ModelFactory.create(
                self.model_name,
                self.problem_type,
                **self.model_params,
            )
            
            # Store feature names
            if isinstance(X, pd.DataFrame):
                self._feature_names = list(X.columns)
            
            # Fit model
            self._model.fit(X, y, **kwargs)
            
            # Store classes for classification
            if self.problem_type == "classification" and hasattr(self._model, "classes_"):
                self._classes = self._model.classes_
            
            self.state = ModelState.FITTED
            self.notify("training_complete", {"model": self.model_name})
            
            logger.info(f"Training complete", model=self.model_name)
            return self
            
        except Exception as e:
            raise ModelTrainingError(
                f"Training failed for {self.model_name}",
                context={"error": str(e)},
            )
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ModelPredictionError("Model not fitted. Call fit() first.")
        
        try:
            return self._model.predict(X, **kwargs)
        except Exception as e:
            raise ModelPredictionError(
                "Prediction failed",
                context={"error": str(e)},
            )
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ModelPredictionError("Model not fitted. Call fit() first.")
        
        if self.problem_type != "classification":
            raise ModelPredictionError("predict_proba only available for classification")
        
        if not hasattr(self._model, "predict_proba"):
            raise ModelPredictionError(f"{self.model_name} does not support predict_proba")
        
        return self._model.predict_proba(X, **kwargs)
    
    def feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            top_n: Return only top N features
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ModelError("Model not fitted. Call fit() first.")
        
        # Try different importance attributes
        importance = None
        
        if hasattr(self._model, "feature_importances_"):
            importance = self._model.feature_importances_
        elif hasattr(self._model, "coef_"):
            coef = self._model.coef_
            importance = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            raise ModelError(f"{self.model_name} does not provide feature importance")
        
        # Create DataFrame
        if self._feature_names:
            names = self._feature_names
        else:
            names = [f"feature_{i}" for i in range(len(importance))]
        
        df = pd.DataFrame({
            "feature": names,
            "importance": importance,
        })
        
        df = df.sort_values("importance", ascending=False)
        
        if top_n:
            df = df.head(top_n)
        
        return df.reset_index(drop=True)
    
    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = 5,
        scoring: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Dict with train and test scores
        """
        from sklearn.model_selection import cross_validate as sklearn_cv
        
        if scoring is None:
            scoring = "accuracy" if self.problem_type == "classification" else "neg_mean_squared_error"
        
        model = ModelFactory.create(
            self.model_name,
            self.problem_type,
            **self.model_params,
        )
        
        results = sklearn_cv(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )
        
        return {
            "test_score_mean": results["test_score"].mean(),
            "test_score_std": results["test_score"].std(),
            "train_score_mean": results["train_score"].mean(),
            "train_score_std": results["train_score"].std(),
            "fit_time_mean": results["fit_time"].mean(),
            "all_test_scores": results["test_score"].tolist(),
        }
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        if not self.is_fitted:
            raise ModelError("Cannot save unfitted model")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "model": self._model,
            "model_name": self.model_name,
            "problem_type": self.problem_type,
            "model_params": self.model_params,
            "feature_names": self._feature_names,
            "classes": self._classes,
            "config": self.config,
        }
        
        joblib.dump(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "TraditionalMLModel":
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
            
        Returns:
            Loaded model
        """
        save_dict = joblib.load(path)
        
        instance = cls(
            model_name=save_dict["model_name"],
            problem_type=save_dict["problem_type"],
            config=save_dict.get("config"),
            **save_dict.get("model_params", {}),
        )
        
        instance._model = save_dict["model"]
        instance._feature_names = save_dict.get("feature_names", [])
        instance._classes = save_dict.get("classes")
        instance._state = ModelState.FITTED
        
        logger.info(f"Model loaded from {path}")
        return instance
    
    @property
    def classes_(self) -> Optional[np.ndarray]:
        """Get class labels for classification."""
        return self._classes
    
    @property
    def underlying_model(self) -> Any:
        """Get the underlying sklearn-like model."""
        return self._model


class ClassificationModel(TraditionalMLModel):
    """
    Classification-specific model wrapper.
    
    Example:
        >>> model = ClassificationModel("xgboost")
        >>> model.fit(X_train, y_train)
        >>> proba = model.predict_proba(X_test)
    """
    
    def __init__(
        self,
        model_name: str = "random_forest",
        config: Optional[Dict[str, Any]] = None,
        **model_params: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            problem_type="classification",
            config=config,
            **model_params,
        )
    
    def predict_class(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Predict class labels with custom threshold.
        
        Args:
            X: Feature matrix
            threshold: Classification threshold (for binary)
            
        Returns:
            Class labels
        """
        if self._classes is not None and len(self._classes) == 2:
            proba = self.predict_proba(X)
            return (proba[:, 1] >= threshold).astype(int)
        return self.predict(X)


class RegressionModel(TraditionalMLModel):
    """
    Regression-specific model wrapper.
    
    Example:
        >>> model = RegressionModel("gradient_boosting")
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        model_name: str = "random_forest",
        config: Optional[Dict[str, Any]] = None,
        **model_params: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            problem_type="regression",
            config=config,
            **model_params,
        )
    
    def predict_interval(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        alpha: float = 0.05,
    ) -> Dict[str, np.ndarray]:
        """
        Predict with confidence interval (if supported).
        
        Args:
            X: Feature matrix
            alpha: Significance level
            
        Returns:
            Dict with predictions and intervals
        """
        predictions = self.predict(X)
        
        # For models without native interval support,
        # return point predictions only
        return {
            "predictions": predictions,
            "lower": predictions,  # Placeholder
            "upper": predictions,  # Placeholder
        }
