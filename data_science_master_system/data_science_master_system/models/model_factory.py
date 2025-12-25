"""
Model Factory for Data Science Master System.

Provides unified model creation and auto-selection:
- Auto-detect problem type
- Select best model based on data characteristics
- Handle all ML frameworks uniformly

Example:
    >>> factory = ModelFactory()
    >>> model = factory.create("xgboost", problem_type="classification")
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
"""

from typing import Any, Dict, List, Optional, Type, Union
import numpy as np
import pandas as pd

from data_science_master_system.core.base_classes import BaseModel, BaseFactory, ProblemType
from data_science_master_system.core.exceptions import ModelError
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)


class ModelFactory(BaseFactory):
    """
    Factory for creating ML models.
    
    Supports all major ML frameworks with a unified interface.
    
    Example:
        >>> factory = ModelFactory()
        >>> 
        >>> # Create specific model
        >>> model = factory.create("random_forest", n_estimators=100)
        >>> 
        >>> # Auto-select best model
        >>> model = factory.auto_select(X, y)
    """
    
    # Registry of available models
    _registry: Dict[str, Dict[str, Any]] = {
        # Sklearn models
        "random_forest": {
            "classification": "sklearn.ensemble.RandomForestClassifier",
            "regression": "sklearn.ensemble.RandomForestRegressor",
        },
        "gradient_boosting": {
            "classification": "sklearn.ensemble.GradientBoostingClassifier",
            "regression": "sklearn.ensemble.GradientBoostingRegressor",
        },
        "logistic_regression": {
            "classification": "sklearn.linear_model.LogisticRegression",
        },
        "linear_regression": {
            "regression": "sklearn.linear_model.LinearRegression",
        },
        "ridge": {
            "regression": "sklearn.linear_model.Ridge",
        },
        "lasso": {
            "regression": "sklearn.linear_model.Lasso",
        },
        "svm": {
            "classification": "sklearn.svm.SVC",
            "regression": "sklearn.svm.SVR",
        },
        "knn": {
            "classification": "sklearn.neighbors.KNeighborsClassifier",
            "regression": "sklearn.neighbors.KNeighborsRegressor",
        },
        "decision_tree": {
            "classification": "sklearn.tree.DecisionTreeClassifier",
            "regression": "sklearn.tree.DecisionTreeRegressor",
        },
        "naive_bayes": {
            "classification": "sklearn.naive_bayes.GaussianNB",
        },
        # Boosting models
        "xgboost": {
            "classification": "xgboost.XGBClassifier",
            "regression": "xgboost.XGBRegressor",
        },
        "lightgbm": {
            "classification": "lightgbm.LGBMClassifier",
            "regression": "lightgbm.LGBMRegressor",
        },
        "catboost": {
            "classification": "catboost.CatBoostClassifier",
            "regression": "catboost.CatBoostRegressor",
        },
        # Clustering
        "kmeans": {
            "clustering": "sklearn.cluster.KMeans",
        },
        "dbscan": {
            "clustering": "sklearn.cluster.DBSCAN",
        },
    }
    
    @classmethod
    def create(
        cls,
        name: str,
        problem_type: str = "classification",
        **kwargs: Any,
    ) -> Any:
        """
        Create a model by name.
        
        Args:
            name: Model name (e.g., "xgboost", "random_forest")
            problem_type: "classification", "regression", or "clustering"
            **kwargs: Model hyperparameters
            
        Returns:
            Model instance
        """
        name = name.lower().replace("-", "_").replace(" ", "_")
        
        if name not in cls._registry:
            raise ModelError(
                f"Unknown model: {name}",
                context={"available": list(cls._registry.keys())},
            )
        
        model_paths = cls._registry[name]
        if problem_type not in model_paths:
            available_types = list(model_paths.keys())
            raise ModelError(
                f"Model {name} does not support {problem_type}",
                context={"available_types": available_types},
            )
        
        model_path = model_paths[problem_type]
        
        try:
            # Import and create model
            module_path, class_name = model_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            
            # Set sensible defaults
            if name == "xgboost":
                kwargs.setdefault("use_label_encoder", False)
                kwargs.setdefault("eval_metric", "logloss" if problem_type == "classification" else "rmse")
            elif name == "catboost":
                kwargs.setdefault("verbose", False)
            elif name == "lightgbm":
                kwargs.setdefault("verbose", -1)
            
            model = model_class(**kwargs)
            logger.info(f"Created model", model=name, problem_type=problem_type)
            return model
            
        except ImportError as e:
            raise ModelError(
                f"Failed to import model {name}. Install required package.",
                context={"error": str(e), "model_path": model_path},
            )
    
    @classmethod
    def list_available(cls, problem_type: Optional[str] = None) -> List[str]:
        """List available models."""
        if problem_type:
            return [
                name for name, paths in cls._registry.items()
                if problem_type in paths
            ]
        return list(cls._registry.keys())
    
    @classmethod
    def register(cls, name: str, model_paths: Dict[str, str]) -> None:
        """
        Register a new model.
        
        Args:
            name: Model name
            model_paths: Dict mapping problem_type to import path
        """
        cls._registry[name] = model_paths
        logger.info(f"Registered model: {name}")


class AutoModelSelector:
    """
    Automatically select the best model for a dataset.
    
    Uses meta-learning and cross-validation to find optimal model.
    
    Example:
        >>> selector = AutoModelSelector()
        >>> best_model = selector.select(X, y)
        >>> best_model.fit(X, y)
    """
    
    def __init__(
        self,
        problem_type: Optional[str] = None,
        cv: int = 5,
        scoring: Optional[str] = None,
        models_to_try: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize auto model selector.
        
        Args:
            problem_type: "classification" or "regression" (auto-detected if None)
            cv: Cross-validation folds
            scoring: Scoring metric
            models_to_try: List of model names to evaluate
        """
        self.problem_type = problem_type
        self.cv = cv
        self.scoring = scoring
        self.models_to_try = models_to_try
        self._results: Dict[str, Dict[str, float]] = {}
    
    def select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_limit: Optional[int] = None,
    ) -> Any:
        """
        Select the best model.
        
        Args:
            X: Feature matrix
            y: Target variable
            time_limit: Max time in seconds (optional)
            
        Returns:
            Best model (unfitted)
        """
        from sklearn.model_selection import cross_val_score
        
        # Auto-detect problem type
        if self.problem_type is None:
            if y.nunique() <= 10 or y.dtype == "object":
                self.problem_type = "classification"
            else:
                self.problem_type = "regression"
        
        # Set default scoring
        if self.scoring is None:
            self.scoring = "accuracy" if self.problem_type == "classification" else "neg_mean_squared_error"
        
        # Models to try
        if self.models_to_try is None:
            self.models_to_try = [
                "random_forest",
                "gradient_boosting",
                "xgboost",
                "lightgbm",
            ]
        
        logger.info(
            f"Auto-selecting model",
            problem_type=self.problem_type,
            models=self.models_to_try,
        )
        
        best_score = -np.inf
        best_model_name = None
        
        for model_name in self.models_to_try:
            try:
                model = ModelFactory.create(model_name, self.problem_type)
                
                scores = cross_val_score(
                    model, X, y,
                    cv=self.cv,
                    scoring=self.scoring,
                    n_jobs=-1,
                )
                
                mean_score = scores.mean()
                std_score = scores.std()
                
                self._results[model_name] = {
                    "mean_score": mean_score,
                    "std_score": std_score,
                }
                
                logger.info(
                    f"Evaluated {model_name}",
                    mean_score=round(mean_score, 4),
                    std_score=round(std_score, 4),
                )
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_name = model_name
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate {model_name}: {e}")
                continue
        
        if best_model_name is None:
            raise ModelError("No models could be evaluated successfully")
        
        logger.info(f"Best model: {best_model_name} with score {best_score:.4f}")
        
        # Return fresh instance of best model
        return ModelFactory.create(best_model_name, self.problem_type)
    
    @property
    def results(self) -> Dict[str, Dict[str, float]]:
        """Get evaluation results for all models."""
        return self._results
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get sorted leaderboard of model scores."""
        if not self._results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self._results).T
        df.index.name = "model"
        return df.sort_values("mean_score", ascending=False).reset_index()


def detect_problem_type(y: pd.Series) -> ProblemType:
    """
    Auto-detect the problem type from target variable.
    
    Args:
        y: Target variable
        
    Returns:
        ProblemType enum value
    """
    if y.dtype == "object" or y.dtype.name == "category":
        return ProblemType.CLASSIFICATION
    
    n_unique = y.nunique()
    n_total = len(y)
    
    if n_unique <= 2:
        return ProblemType.CLASSIFICATION
    elif n_unique / n_total < 0.05 and n_unique <= 20:
        return ProblemType.CLASSIFICATION
    else:
        return ProblemType.REGRESSION
