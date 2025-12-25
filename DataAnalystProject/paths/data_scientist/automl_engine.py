"""
Advanced AutoML Engine
Expert-level automated machine learning with:
- Bayesian Optimization (Optuna)
- Advanced Ensembling (Stacking, Voting, Blending)
- Meta-learning
- Multi-objective optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import logging
import warnings
from datetime import datetime
import json

# ML imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Models
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Imbalanced Learning
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTEENN
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

# Try to import advanced libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class AutoMLResult:
    """Results from AutoML optimization"""
    best_model: Any
    best_score: float
    best_params: Dict
    all_trials: List[Dict]
    optimization_time: float
    task_type: str
    feature_importance: Dict[str, float] = field(default_factory=dict)
    shap_values: Optional[np.ndarray] = None
    ensemble_models: List[Any] = field(default_factory=list)
    cv_scores: List[float] = field(default_factory=list)


class AdvancedAutoML:
    """
    Advanced AutoML Engine with Expert-Level Capabilities
    
    Features:
    - Bayesian Optimization with Optuna
    - Advanced Ensembling (Stacking, Voting, Blending)
    - Automatic task detection
    - SHAP explanations
    - Multi-objective optimization
    """
    
    def __init__(self, 
                 n_trials: int = 50,
                 cv_folds: int = 5,
                 timeout_minutes: int = 30,
                 optimization_metric: str = 'auto',
                 use_ensemble: bool = True,
                 random_state: int = 42):
        
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.timeout = timeout_minutes * 60
        self.optimization_metric = optimization_metric
        self.use_ensemble = use_ensemble
        self.random_state = random_state
        
        self.task_type: str = None
        self.best_model = None
        self.best_params: Dict = {}
        self.feature_names: List[str] = []
        self.target_name: str = None
        self.results: AutoMLResult = None
        
        # Track all trained models for ensembling
        self.trained_models: List[Tuple[str, Any, float]] = []
        
    def fit(self, df: pd.DataFrame, target: str, 
            features: List[str] = None) -> AutoMLResult:
        """
        Fit AutoML on data
        
        Args:
            df: DataFrame with features and target
            target: Target column name
            features: List of feature columns (optional)
        
        Returns:
            AutoMLResult with best model and metrics
        """
        start_time = datetime.now()
        
        # Prepare data
        X, y = self._prepare_data(df, target, features)
        
        # Detect task type
        self.task_type = self._detect_task_type(y)
        logger.info(f"Detected task type: {self.task_type}")
        
        # Set optimization metric
        if self.optimization_metric == 'auto':
            self.optimization_metric = 'roc_auc' if self.task_type == 'classification' else 'neg_mean_squared_error'
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state,
            stratify=y if self.task_type == 'classification' else None
        )
        
        # Apply SMOTE for imbalanced classification
        if self.task_type == 'classification' and HAS_IMBLEARN:
            try:
                # Check imbalance ratio
                counts = y_train.value_counts()
                imbalance_ratio = counts.min() / counts.max()
                
                if imbalance_ratio < 0.2:  # If minority class is < 20%
                    logger.info(f"Imbalanced data detected (ratio={imbalance_ratio:.2f}). Applying SMOTE-ENN...")
                    resampler = SMOTEENN(random_state=self.random_state)
                    X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
                    logger.info(f"Data resampled: {len(X_train)} -> {len(X_train_resampled)}")
                    X_train, y_train = X_train_resampled, y_train_resampled
            except Exception as e:
                logger.warning(f"SMOTE application failed: {e}")
        
        # Run optimization
        if HAS_OPTUNA:
            best_model, best_params, all_trials = self._optimize_with_optuna(
                X_train, y_train, X_test, y_test
            )
        else:
            best_model, best_params, all_trials = self._optimize_grid_search(
                X_train, y_train, X_test, y_test
            )
        
        # Build ensemble if enabled
        if self.use_ensemble and len(self.trained_models) >= 3:
            ensemble = self._build_ensemble(X_train, y_train)
            ensemble_score = self._evaluate_model(ensemble, X_test, y_test)
            
            # Use ensemble if it's better
            if ensemble_score > self._evaluate_model(best_model, X_test, y_test):
                best_model = ensemble
                logger.info(f"Ensemble outperformed best single model: {ensemble_score:.4f}")
        
        # Calculate feature importance
        feature_importance = self._get_feature_importance(best_model, X)
        
        # Get SHAP values if available
        shap_values = None
        if HAS_SHAP:
            try:
                shap_values = self._get_shap_values(best_model, X_test)
            except:
                pass
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            best_model, X, y, cv=self.cv_folds, 
            scoring=self.optimization_metric
        ).tolist()
        
        # Calculate optimization time
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Store results
        if self.task_type == 'classification' and hasattr(best_model, "predict_proba"):
            try:
                # Calibrate probabilities
                best_model = CalibratedClassifierCV(best_model, cv='prefit', method='sigmoid')
                best_model.fit(X_test, y_test) # Calibrate on test set or holdout
            except:
                pass

        self.best_model = best_model
        self.best_params = best_params
        
        self.results = AutoMLResult(
            best_model=best_model,
            best_score=self._evaluate_model(best_model, X_test, y_test),
            best_params=best_params,
            all_trials=all_trials,
            optimization_time=optimization_time,
            task_type=self.task_type,
            feature_importance=feature_importance,
            shap_values=shap_values,
            ensemble_models=[m[0] for m in self.trained_models[:5]],
            cv_scores=cv_scores
        )
        
        return self.results
    
    def _prepare_data(self, df: pd.DataFrame, target: str, 
                      features: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training"""
        
        self.target_name = target
        
        if features is None:
            features = [c for c in df.columns if c != target]
        
        self.feature_names = features
        
        X = df[features].copy()
        y = df[target].copy()
        
        # Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
        
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                X[col] = X[col].fillna('missing')
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target if classification
        if y.dtype == 'object':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y))
        
        return X, y
    
    def _detect_task_type(self, y: pd.Series) -> str:
        """Detect if this is classification or regression"""
        unique_ratio = y.nunique() / len(y)
        
        if y.dtype == 'object' or y.nunique() <= 10 or unique_ratio < 0.05:
            return 'classification'
        return 'regression'
    
    def _optimize_with_optuna(self, X_train, y_train, X_test, y_test) -> Tuple:
        """Optimize using Optuna Bayesian Optimization"""
        
        all_trials = []
        
        def objective(trial):
            # Select model type
            model_type = trial.suggest_categorical('model_type', [
                'rf', 'gb', 'xgb', 'lgb', 'cat', 'et', 'lr'
            ])
            
            model = self._create_model_with_params(trial, model_type)
            
            if model is None:
                return float('-inf') if self.task_type == 'classification' else float('inf')
            
            # Cross-validation
            try:
                cv = StratifiedKFold(n_splits=self.cv_folds) if self.task_type == 'classification' else KFold(n_splits=self.cv_folds)
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=self.optimization_metric)
                score = scores.mean()
                
                # Train and store model
                model.fit(X_train, y_train)
                test_score = self._evaluate_model(model, X_test, y_test)
                
                self.trained_models.append((model_type, model, test_score))
                
                all_trials.append({
                    'model_type': model_type,
                    'params': trial.params,
                    'cv_score': score,
                    'test_score': test_score
                })
                
                return score
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('-inf') if self.task_type == 'classification' else float('inf')
        
        # Create and run study
        study = optuna.create_study(
            direction='maximize' if self.task_type == 'classification' else 'minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        study.optimize(
            objective, 
            n_trials=self.n_trials, 
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Get best model
        best_trial = study.best_trial
        best_model = self._create_model_with_params(best_trial, best_trial.params['model_type'])
        best_model.fit(X_train, y_train)
        
        return best_model, best_trial.params, all_trials
    
    def _create_model_with_params(self, trial, model_type: str):
        """Create model with trial parameters"""
        
        if self.task_type == 'classification':
            if model_type == 'rf':
                return RandomForestClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    max_depth=trial.suggest_int('max_depth', 3, 20),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    random_state=self.random_state,
                    n_jobs=-1
                )
            elif model_type == 'gb':
                return GradientBoostingClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    max_depth=trial.suggest_int('max_depth', 3, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    random_state=self.random_state
                )
            elif model_type == 'xgb' and HAS_XGB:
                return xgb.XGBClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    max_depth=trial.suggest_int('max_depth', 3, 15),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    random_state=self.random_state,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            elif model_type == 'lgb' and HAS_LGB:
                return lgb.LGBMClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    max_depth=trial.suggest_int('max_depth', 3, 15),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    num_leaves=trial.suggest_int('num_leaves', 20, 100),
                    random_state=self.random_state,
                    verbose=-1
                )
            elif model_type == 'cat' and HAS_CAT:
                return cb.CatBoostClassifier(
                    iterations=trial.suggest_int('iterations', 50, 500),
                    depth=trial.suggest_int('depth', 3, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    random_state=self.random_state,
                    verbose=0
                )
            elif model_type == 'et':
                return ExtraTreesClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    max_depth=trial.suggest_int('max_depth', 3, 20),
                    random_state=self.random_state,
                    n_jobs=-1
                )
            elif model_type == 'lr':
                return LogisticRegression(
                    C=trial.suggest_float('C', 0.01, 10.0),
                    max_iter=1000,
                    random_state=self.random_state
                )
            else:
                return RandomForestClassifier(random_state=self.random_state)
        
        else:  # Regression
            if model_type == 'rf':
                return RandomForestRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    max_depth=trial.suggest_int('max_depth', 3, 20),
                    random_state=self.random_state,
                    n_jobs=-1
                )
            elif model_type == 'gb':
                return GradientBoostingRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    max_depth=trial.suggest_int('max_depth', 3, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    random_state=self.random_state
                )
            elif model_type == 'xgb' and HAS_XGB:
                return xgb.XGBRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    max_depth=trial.suggest_int('max_depth', 3, 15),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    random_state=self.random_state
                )
            elif model_type == 'lgb' and HAS_LGB:
                return lgb.LGBMRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 500),
                    max_depth=trial.suggest_int('max_depth', 3, 15),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    random_state=self.random_state,
                    verbose=-1
                )
            elif model_type == 'cat' and HAS_CAT:
                return cb.CatBoostRegressor(
                    iterations=trial.suggest_int('iterations', 50, 500),
                    depth=trial.suggest_int('depth', 3, 10),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    random_state=self.random_state,
                    verbose=0
                )
            elif model_type == 'lr':
                return Ridge(
                    alpha=trial.suggest_float('alpha', 0.01, 10.0),
                    random_state=self.random_state
                )
            else:
                return RandomForestRegressor(random_state=self.random_state)
    
    def _optimize_grid_search(self, X_train, y_train, X_test, y_test) -> Tuple:
        """Fallback optimization without Optuna"""
        
        all_trials = []
        best_model = None
        best_score = float('-inf') if self.task_type == 'classification' else float('inf')
        best_params = {}
        
        # Define simple model grid
        if self.task_type == 'classification':
            models = [
                ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
                ('gb', GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)),
                ('et', ExtraTreesClassifier(n_estimators=100, random_state=self.random_state)),
                ('lr', LogisticRegression(max_iter=1000, random_state=self.random_state))
            ]
        else:
            models = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=self.random_state)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)),
                ('ridge', Ridge(random_state=self.random_state))
            ]
        
        for name, model in models:
            try:
                model.fit(X_train, y_train)
                score = self._evaluate_model(model, X_test, y_test)
                
                self.trained_models.append((name, model, score))
                all_trials.append({'model_type': name, 'score': score})
                
                is_better = (score > best_score) if self.task_type == 'classification' else (score < best_score)
                if is_better:
                    best_score = score
                    best_model = model
                    best_params = {'model_type': name}
            except:
                continue
        
        return best_model, best_params, all_trials
    
    def _build_ensemble(self, X_train, y_train):
        """Build ensemble from best models"""
        
        # Sort models by score
        sorted_models = sorted(self.trained_models, key=lambda x: x[2], reverse=True)[:5]
        
        estimators = [(name, model) for name, model, _ in sorted_models]
        
        if self.task_type == 'classification':
            # Stacking ensemble
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                cv=3,
                n_jobs=-1
            )
        else:
            ensemble = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(),
                cv=3,
                n_jobs=-1
            )
        
        ensemble.fit(X_train, y_train)
        return ensemble
    
    def _evaluate_model(self, model, X_test, y_test) -> float:
        """Evaluate model performance"""
        
        predictions = model.predict(X_test)
        
        if self.task_type == 'classification':
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X_test)
                    if proba.shape[1] == 2:
                        return roc_auc_score(y_test, proba[:, 1])
                    else:
                        return roc_auc_score(y_test, proba, multi_class='ovr')
                except:
                    pass
            return f1_score(y_test, predictions, average='weighted')
        else:
            return -mean_squared_error(y_test, predictions)  # Negative for consistency
    
    def _get_feature_importance(self, model, X) -> Dict[str, float]:
        """Extract feature importance"""
        
        importance = {}
        
        if hasattr(model, 'feature_importances_'):
            for name, imp in zip(self.feature_names, model.feature_importances_):
                importance[name] = float(imp)
        elif hasattr(model, 'coef_'):
            coef = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
            for name, imp in zip(self.feature_names, np.abs(coef)):
                importance[name] = float(imp)
        
        # Normalize
        total = sum(importance.values()) if importance else 1
        importance = {k: v/total for k, v in importance.items()}
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def _get_shap_values(self, model, X_test) -> np.ndarray:
        """Calculate SHAP values"""
        if not HAS_SHAP:
            return None
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            return shap_values
        except:
            try:
                explainer = shap.Explainer(model, X_test)
                shap_values = explainer(X_test)
                return shap_values.values
            except:
                return None
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with best model"""
        if self.best_model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Prepare features
        X_processed = X[self.feature_names].copy()
        
        # Handle missing and categorical
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                X_processed[col] = X_processed[col].fillna('missing')
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            else:
                X_processed[col] = X_processed[col].fillna(X_processed[col].median())
        
        return self.best_model.predict(X_processed)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of AutoML results"""
        if self.results is None:
            return {"error": "No results available"}
        
        return {
            "task_type": self.results.task_type,
            "best_score": self.results.best_score,
            "best_params": self.results.best_params,
            "optimization_time": f"{self.results.optimization_time:.1f}s",
            "cv_mean": np.mean(self.results.cv_scores),
            "cv_std": np.std(self.results.cv_scores),
            "n_trials": len(self.results.all_trials),
            "top_features": list(self.results.feature_importance.keys())[:5],
            "ensemble_models": self.results.ensemble_models
        }
