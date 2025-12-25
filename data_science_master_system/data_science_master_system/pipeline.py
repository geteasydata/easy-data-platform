"""
Main Pipeline for Data Science Master System.

Provides end-to-end ML pipeline:
- Auto-detection of problem type
- Automatic preprocessing
- Model training and evaluation
- Feature importance analysis

Example:
    >>> from data_science_master_system import Pipeline
    >>> 
    >>> pipeline = Pipeline.auto_detect(df, target="label")
    >>> pipeline.fit()
    >>> predictions = pipeline.predict(new_data)
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from data_science_master_system.core.base_classes import BasePipeline, PipelineStep, ProblemType
from data_science_master_system.core.exceptions import PipelineError
from data_science_master_system.core.logger import get_logger
from data_science_master_system.features.transformation.transformers import StandardScaler, OneHotEncoder
from data_science_master_system.models.model_factory import ModelFactory, AutoModelSelector, detect_problem_type
from data_science_master_system.models.traditional.traditional_ml import TraditionalMLModel

logger = get_logger(__name__)


class Pipeline(BasePipeline):
    """
    End-to-end ML pipeline with automatic configuration.
    
    Features:
    - Auto-detect problem type
    - Automatic preprocessing (scaling, encoding)
    - Model selection and training
    - Cross-validation and evaluation
    - Feature importance analysis
    
    Example:
        >>> # Quick start
        >>> pipeline = Pipeline.auto_detect(df, target="label")
        >>> pipeline.fit()
        >>> predictions = pipeline.predict(new_data)
        >>> 
        >>> # Custom configuration
        >>> pipeline = Pipeline(
        ...     problem_type="classification",
        ...     model_name="xgboost",
        ...     auto_preprocess=True,
        ... )
        >>> pipeline.fit(X_train, y_train)
    """
    
    def __init__(
        self,
        problem_type: Optional[str] = None,
        model_name: Optional[str] = None,
        auto_preprocess: bool = True,
        auto_select_model: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize pipeline.
        
        Args:
            problem_type: "classification" or "regression" (auto-detected if None)
            model_name: Model to use (e.g., "xgboost", "random_forest")
            auto_preprocess: Automatically preprocess data
            auto_select_model: Use AutoModelSelector to find best model
            config: Additional configuration
        """
        super().__init__(config=config)
        self.problem_type = problem_type
        self.model_name = model_name or "random_forest"
        self.auto_preprocess = auto_preprocess
        self.auto_select_model = auto_select_model
        
        self._model = None
        self._preprocessor = None
        self._feature_names: List[str] = []
        self._target_name: Optional[str] = None
        self._X: Optional[pd.DataFrame] = None
        self._y: Optional[pd.Series] = None
    
    @classmethod
    def auto_detect(
        cls,
        data: pd.DataFrame,
        target: str,
        **kwargs: Any,
    ) -> "Pipeline":
        """
        Create pipeline with automatic configuration.
        
        Args:
            data: Full dataset including target
            target: Target column name
            **kwargs: Additional configuration
            
        Returns:
            Configured Pipeline
        """
        if target not in data.columns:
            raise PipelineError(f"Target column '{target}' not found in data")
        
        y = data[target]
        X = data.drop(columns=[target])
        
        # Detect problem type
        problem_type = detect_problem_type(y)
        
        pipeline = cls(
            problem_type=problem_type.name.lower(),
            **kwargs,
        )
        
        pipeline._X = X
        pipeline._y = y
        pipeline._target_name = target
        pipeline._feature_names = list(X.columns)
        
        logger.info(
            f"Pipeline auto-configured",
            problem_type=problem_type.name,
            features=len(X.columns),
            samples=len(X),
        )
        
        return pipeline
    
    def fit(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> "Pipeline":
        """
        Fit the pipeline.
        
        Args:
            X: Feature matrix (optional if auto_detect was used)
            y: Target variable (optional if auto_detect was used)
            **kwargs: Additional fit parameters
            
        Returns:
            Self
        """
        # Use stored data if not provided
        if X is None and self._X is not None:
            X = self._X
        if y is None and self._y is not None:
            y = self._y
        
        if X is None or y is None:
            raise PipelineError("Must provide X and y, or use auto_detect()")
        
        logger.info(f"Fitting pipeline", shape=X.shape)
        
        # Detect problem type if not set
        if self.problem_type is None:
            self.problem_type = detect_problem_type(y).name.lower()
        
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
        
        # Preprocess if enabled
        if self.auto_preprocess:
            X = self._preprocess_fit_transform(X)
        
        # Select model if auto_select enabled
        if self.auto_select_model:
            selector = AutoModelSelector(problem_type=self.problem_type)
            self._model = selector.select(X, y)
        else:
            self._model = ModelFactory.create(self.model_name, self.problem_type)
        
        # Fit model
        self._model.fit(X, y, **kwargs)
        
        self._fitted = True
        self.notify("fit_complete", {"samples": len(X)})
        
        logger.info(f"Pipeline fitting complete")
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self._fitted:
            raise PipelineError("Pipeline not fitted. Call fit() first.")
        
        if self.auto_preprocess and self._preprocessor:
            X = self._preprocess_transform(X)
        
        return self._model.predict(X, **kwargs)
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Predict class probabilities (classification only).
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        if self.problem_type != "classification":
            raise PipelineError("predict_proba only available for classification")
        
        if not self._fitted:
            raise PipelineError("Pipeline not fitted. Call fit() first.")
        
        if self.auto_preprocess and self._preprocessor:
            X = self._preprocess_transform(X)
        
        return self._model.predict_proba(X, **kwargs)
    
    def transform(self, X: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        if self.auto_preprocess and self._preprocessor:
            return self._preprocess_transform(X)
        return X
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate pipeline on test data.
        
        Args:
            X: Feature matrix
            y: True labels
            metrics: List of metrics to compute
            
        Returns:
            Dict of metric names to values
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_squared_error, mean_absolute_error, r2_score,
        )
        
        predictions = self.predict(X)
        results = {}
        
        if self.problem_type == "classification":
            results["accuracy"] = accuracy_score(y, predictions)
            results["precision"] = precision_score(y, predictions, average="weighted", zero_division=0)
            results["recall"] = recall_score(y, predictions, average="weighted", zero_division=0)
            results["f1"] = f1_score(y, predictions, average="weighted", zero_division=0)
        else:
            results["mse"] = mean_squared_error(y, predictions)
            results["rmse"] = np.sqrt(mean_squared_error(y, predictions))
            results["mae"] = mean_absolute_error(y, predictions)
            results["r2"] = r2_score(y, predictions)
        
        return results
    
    def cross_validate(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        cv: int = 5,
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv: Number of folds
            
        Returns:
            Cross-validation results
        """
        X = X if X is not None else self._X
        y = y if y is not None else self._y
        
        if X is None or y is None:
            raise PipelineError("Data required for cross-validation")
        
        from sklearn.model_selection import cross_val_score
        
        model = ModelFactory.create(self.model_name, self.problem_type)
        scoring = "accuracy" if self.problem_type == "classification" else "neg_mean_squared_error"
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "scores": scores.tolist(),
        }
    
    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from fitted model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not self._fitted:
            raise PipelineError("Pipeline not fitted")
        
        if hasattr(self._model, "feature_importances_"):
            importance = self._model.feature_importances_
        elif hasattr(self._model, "coef_"):
            importance = np.abs(self._model.coef_).flatten()
        else:
            raise PipelineError("Model does not provide feature importance")
        
        df = pd.DataFrame({
            "feature": self._feature_names[:len(importance)],
            "importance": importance,
        })
        
        return df.nlargest(top_n, "importance").reset_index(drop=True)
    
    def _preprocess_fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor and transform data."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler as SkScaler, OneHotEncoder as SkOHE
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline as SkPipeline
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        
        transformers = []
        
        if numeric_cols:
            numeric_transformer = SkPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", SkScaler()),
            ])
            transformers.append(("num", numeric_transformer, numeric_cols))
        
        if categorical_cols:
            categorical_transformer = SkPipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", SkOHE(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", categorical_transformer, categorical_cols))
        
        if not transformers:
            return X
        
        self._preprocessor = ColumnTransformer(transformers, remainder="passthrough")
        X_transformed = self._preprocessor.fit_transform(X)
        
        # Get feature names
        feature_names = []
        if numeric_cols:
            feature_names.extend(numeric_cols)
        if categorical_cols:
            encoder = self._preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_features = encoder.get_feature_names_out(categorical_cols)
            feature_names.extend(cat_features)
        
        self._feature_names = list(feature_names)
        
        return pd.DataFrame(X_transformed, columns=self._feature_names[:X_transformed.shape[1]], index=X.index)
    
    def _preprocess_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        if self._preprocessor is None:
            return X
        
        X_transformed = self._preprocessor.transform(X)
        return pd.DataFrame(X_transformed, columns=self._feature_names[:X_transformed.shape[1]], index=X.index)
    
    def save(self, path: str) -> None:
        """Save pipeline to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "model": self._model,
            "preprocessor": self._preprocessor,
            "problem_type": self.problem_type,
            "model_name": self.model_name,
            "feature_names": self._feature_names,
            "config": self.config,
        }
        
        joblib.dump(save_dict, path)
        logger.info(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "Pipeline":
        """Load pipeline from disk."""
        save_dict = joblib.load(path)
        
        pipeline = cls(
            problem_type=save_dict["problem_type"],
            model_name=save_dict["model_name"],
            config=save_dict.get("config"),
        )
        
        pipeline._model = save_dict["model"]
        pipeline._preprocessor = save_dict.get("preprocessor")
        pipeline._feature_names = save_dict.get("feature_names", [])
        pipeline._fitted = True
        
        logger.info(f"Pipeline loaded from {path}")
        return pipeline
