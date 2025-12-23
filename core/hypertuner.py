"""
Professional Hyperparameter Tuner - Intelligent Parameter Optimization
Uses Optuna for smart hyperparameter search
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class HyperTuner:
    """
    Professional Hyperparameter Tuner using Optuna.
    Optimizes model hyperparameters like a senior ML engineer.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_params = {}
        self.best_score = 0
        self.tuning_log = []
        self.n_trials_completed = 0
        
    def log(self, message: str):
        """Add to tuning log."""
        self.tuning_log.append(f"🔧 {message}")
    
    def tune(self, model_name: str, X: pd.DataFrame, y: pd.Series,
             problem_type: str = 'classification',
             n_trials: int = 50,
             cv_folds: int = 5,
             timeout: int = 300) -> Dict[str, Any]:
        """
        Tune hyperparameters for a specific model.
        
        Args:
            model_name: Name of the model to tune
            X: Features
            y: Target
            problem_type: 'classification' or 'regression'
            n_trials: Number of Optuna trials
            cv_folds: Cross-validation folds
            timeout: Maximum time in seconds
            
        Returns:
            Best parameters dictionary
        """
        if not HAS_OPTUNA:
            self.log("Optuna not installed, using default parameters")
            return self._get_default_params(model_name)
        
        # Prepare data
        X_clean = X.fillna(0).values
        y_clean = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0).values
        
        # Get objective function for this model
        objective = self._get_objective(model_name, X_clean, y_clean, problem_type, cv_folds)
        
        if objective is None:
            self.log(f"No tuning available for {model_name}")
            return {}
        
        try:
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.random_state)
            )
            
            # Optimize
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=False
            )
            
            self.best_params = study.best_params
            self.best_score = study.best_value
            self.n_trials_completed = len(study.trials)
            
            self.log(f"Tuned {model_name}: best score = {self.best_score:.4f}")
            self.log(f"Best params: {self.best_params}")
            
            return self.best_params
            
        except Exception as e:
            self.log(f"Tuning failed: {str(e)[:50]}")
            return self._get_default_params(model_name)
    
    def _get_objective(self, model_name: str, X: np.ndarray, y: np.ndarray,
                       problem_type: str, cv_folds: int) -> Optional[Callable]:
        """Get Optuna objective function for model."""
        
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        
        if problem_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            scoring = 'r2'
        
        def rf_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),  # Increased max
                'max_depth': trial.suggest_int('max_depth', 3, 30),          # Increased max
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]), # Added max_features
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            if problem_type == 'classification':
                model = RandomForestClassifier(**params)
            else:
                model = RandomForestRegressor(**params)
            
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            return scores.mean()
        
        def gb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True), # Wider range
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),                     # Added subsample
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'random_state': self.random_state,
            }
            
            if problem_type == 'classification':
                model = GradientBoostingClassifier(**params)
            else:
                model = GradientBoostingRegressor(**params)
            
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            return scores.mean()
        
        def lr_objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'max_iter': 1000,
                'random_state': self.random_state,
            }
            model = LogisticRegression(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            return scores.mean()
        
        def ridge_objective(trial):
            params = {
                'alpha': trial.suggest_float('alpha', 0.001, 100, log=True),
                'random_state': self.random_state,
            }
            model = Ridge(**params)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            return scores.mean()
        
        objectives = {
            'Random Forest': rf_objective,
            'Gradient Boosting': gb_objective,
            'Logistic Regression': lr_objective,
            'Ridge': ridge_objective,
        }
        
        # XGBoost
        try:
            from xgboost import XGBClassifier, XGBRegressor
            
            def xgb_objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'random_state': self.random_state,
                    'verbosity': 0,
                    'use_label_encoder': False,
                }
                
                if problem_type == 'classification':
                    params['eval_metric'] = 'logloss'
                    model = XGBClassifier(**params)
                else:
                    model = XGBRegressor(**params)
                
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                return scores.mean()
            
            objectives['XGBoost'] = xgb_objective
        except ImportError:
            pass
        
        # LightGBM
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            
            def lgbm_objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': self.random_state,
                    'verbose': -1,
                }
                
                if problem_type == 'classification':
                    model = LGBMClassifier(**params)
                else:
                    model = LGBMRegressor(**params)
                
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                return scores.mean()
            
            objectives['LightGBM'] = lgbm_objective
        except ImportError:
            pass
        
        return objectives.get(model_name)
    
    def _get_default_params(self, model_name: str) -> Dict[str, Any]:
        """Get default parameters when tuning is not available."""
        defaults = {
            'Random Forest': {'n_estimators': 100, 'max_depth': 10},
            'Gradient Boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
            'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
            'LightGBM': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
            'Logistic Regression': {'C': 1.0, 'max_iter': 1000},
            'Ridge': {'alpha': 1.0},
        }
        return defaults.get(model_name, {})
    
    def quick_tune(self, X: pd.DataFrame, y: pd.Series, 
                   problem_type: str = 'classification') -> Dict[str, Dict]:
        """
        Quick tune for all available models.
        
        Returns:
            Dictionary of {model_name: best_params}
        """
        models = ['Random Forest', 'Gradient Boosting']
        
        try:
            from xgboost import XGBClassifier
            models.append('XGBoost')
        except:
            pass
        
        try:
            from lightgbm import LGBMClassifier
            models.append('LightGBM')
        except:
            pass
        
        results = {}
        for model_name in models:
            self.log(f"Quick tuning {model_name}...")
            params = self.tune(model_name, X, y, problem_type, n_trials=20, timeout=60)
            results[model_name] = params
        
        return results
    
    def get_log(self) -> list:
        """Get tuning log."""
        return self.tuning_log


def tune_model(model_name: str, X: pd.DataFrame, y: pd.Series,
               problem_type: str = 'classification') -> Dict[str, Any]:
    """Convenience function for tuning."""
    tuner = HyperTuner()
    return tuner.tune(model_name, X, y, problem_type)
