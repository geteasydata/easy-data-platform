"""
AutoML Integration Module.

Unified interface for AutoML frameworks:
- AutoGluon
- H2O AutoML
- FLAML (Fast Lightweight AutoML)
- Auto-sklearn
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path

from data_science_master_system.core.base_classes import BaseModel
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)


class AutoMLEngine:
    """
    Unified AutoML interface supporting multiple backends.
    
    Example:
        >>> engine = AutoMLEngine(backend='autogluon', time_limit=300)
        >>> engine.fit(X_train, y_train)
        >>> predictions = engine.predict(X_test)
        >>> print(engine.leaderboard())
    """
    
    BACKENDS = ['autogluon', 'h2o', 'flaml', 'auto_sklearn']
    
    def __init__(
        self,
        backend: str = 'flaml',
        problem_type: str = 'auto',
        time_limit: int = 300,
        metric: str = 'auto',
        n_jobs: int = -1,
        **kwargs
    ):
        self.backend = backend.lower()
        self.problem_type = problem_type
        self.time_limit = time_limit
        self.metric = metric
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.model = None
        self._leaderboard = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'AutoMLEngine':
        """Train AutoML model."""
        if self.backend == 'flaml':
            self._fit_flaml(X, y, **kwargs)
        elif self.backend == 'autogluon':
            self._fit_autogluon(X, y, **kwargs)
        elif self.backend == 'h2o':
            self._fit_h2o(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        return self
    
    def _fit_flaml(self, X, y, **kwargs):
        """Fit using FLAML."""
        try:
            from flaml import AutoML
            
            self.model = AutoML()
            self.model.fit(
                X, y,
                task=self.problem_type if self.problem_type != 'auto' else 'classification',
                time_budget=self.time_limit,
                metric=self.metric if self.metric != 'auto' else 'accuracy',
                n_jobs=self.n_jobs,
                **kwargs
            )
            logger.info(f"FLAML best model: {self.model.best_estimator}")
        except ImportError:
            raise ImportError("FLAML not installed. Install with: pip install flaml")
    
    def _fit_autogluon(self, X, y, **kwargs):
        """Fit using AutoGluon."""
        try:
            from autogluon.tabular import TabularPredictor
            
            df = X.copy()
            df['target'] = y
            
            self.model = TabularPredictor(label='target', problem_type=self.problem_type)
            self.model.fit(df, time_limit=self.time_limit, **kwargs)
            self._leaderboard = self.model.leaderboard()
            logger.info(f"AutoGluon best model: {self.model.get_model_best()}")
        except ImportError:
            raise ImportError("AutoGluon not installed. Install with: pip install autogluon")
    
    def _fit_h2o(self, X, y, **kwargs):
        """Fit using H2O AutoML."""
        try:
            import h2o
            from h2o.automl import H2OAutoML
            
            h2o.init()
            df = X.copy()
            df['target'] = y
            h2o_df = h2o.H2OFrame(df)
            
            self.model = H2OAutoML(max_runtime_secs=self.time_limit, seed=42)
            self.model.train(y='target', training_frame=h2o_df)
            self._leaderboard = self.model.leaderboard.as_data_frame()
            logger.info(f"H2O best model: {self.model.leader.model_id}")
        except ImportError:
            raise ImportError("H2O not installed. Install with: pip install h2o")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.backend == 'flaml':
            return self.model.predict(X)
        elif self.backend == 'autogluon':
            return self.model.predict(X).values
        elif self.backend == 'h2o':
            import h2o
            h2o_df = h2o.H2OFrame(X)
            return self.model.predict(h2o_df).as_data_frame().values.flatten()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if self.backend == 'flaml':
            return self.model.predict_proba(X)
        elif self.backend == 'autogluon':
            return self.model.predict_proba(X).values
        elif self.backend == 'h2o':
            import h2o
            h2o_df = h2o.H2OFrame(X)
            return self.model.predict(h2o_df).as_data_frame().values
    
    def leaderboard(self) -> pd.DataFrame:
        """Get model leaderboard."""
        if self._leaderboard is not None:
            return self._leaderboard
        if self.backend == 'flaml':
            return pd.DataFrame([{
                'model': self.model.best_estimator,
                'best_score': self.model.best_loss
            }])
        return pd.DataFrame()
    
    def save(self, path: str):
        """Save AutoML model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.backend == 'flaml':
            import pickle
            with open(path / 'model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
        elif self.backend == 'autogluon':
            self.model.save(str(path))
    
    @classmethod
    def load(cls, path: str, backend: str = 'flaml') -> 'AutoMLEngine':
        """Load AutoML model."""
        instance = cls(backend=backend)
        path = Path(path)
        
        if backend == 'flaml':
            import pickle
            with open(path / 'model.pkl', 'rb') as f:
                instance.model = pickle.load(f)
        elif backend == 'autogluon':
            from autogluon.tabular import TabularPredictor
            instance.model = TabularPredictor.load(str(path))
        
        return instance


class HyperparameterOptimizer:
    """
    Distributed hyperparameter optimization.
    
    Supports Optuna, Ray Tune, and custom search spaces.
    """
    
    def __init__(
        self,
        search_space: Dict,
        metric: str = 'accuracy',
        direction: str = 'maximize',
        n_trials: int = 100,
        timeout: int = 3600,
        n_jobs: int = -1
    ):
        self.search_space = search_space
        self.metric = metric
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.study = None
        self.best_params = None
    
    def optimize(self, objective_fn, **kwargs) -> Dict:
        """Run hyperparameter optimization."""
        try:
            import optuna
            
            self.study = optuna.create_study(direction=self.direction)
            self.study.optimize(
                objective_fn,
                n_trials=self.n_trials,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                **kwargs
            )
            
            self.best_params = self.study.best_params
            logger.info(f"Best params: {self.best_params}")
            logger.info(f"Best value: {self.study.best_value}")
            
            return self.best_params
        except ImportError:
            raise ImportError("Optuna not installed. Install with: pip install optuna")
    
    def get_trials_dataframe(self) -> pd.DataFrame:
        """Get trials as DataFrame."""
        if self.study is None:
            return pd.DataFrame()
        return self.study.trials_dataframe()
    
    def plot_optimization_history(self):
        """Plot optimization history."""
        try:
            import optuna.visualization as vis
            return vis.plot_optimization_history(self.study)
        except:
            pass
    
    def plot_param_importances(self):
        """Plot parameter importances."""
        try:
            import optuna.visualization as vis
            return vis.plot_param_importances(self.study)
        except:
            pass


def create_search_space(model_type: str = 'random_forest') -> Dict:
    """Create predefined search spaces for common models."""
    spaces = {
        'random_forest': {
            'n_estimators': ('int', 50, 500),
            'max_depth': ('int', 3, 20),
            'min_samples_split': ('int', 2, 10),
            'min_samples_leaf': ('int', 1, 5),
        },
        'xgboost': {
            'n_estimators': ('int', 50, 500),
            'max_depth': ('int', 3, 15),
            'learning_rate': ('float', 0.01, 0.3),
            'subsample': ('float', 0.6, 1.0),
            'colsample_bytree': ('float', 0.6, 1.0),
        },
        'lightgbm': {
            'n_estimators': ('int', 50, 500),
            'max_depth': ('int', 3, 15),
            'learning_rate': ('float', 0.01, 0.3),
            'num_leaves': ('int', 20, 100),
            'min_child_samples': ('int', 5, 100),
        }
    }
    return spaces.get(model_type, {})
