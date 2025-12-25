"""
Machine Learning Pipeline Module
Advanced ML with automatic model selection, training, and evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Results from a single model"""
    name: str
    model: Any
    train_score: float
    test_score: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None


@dataclass
class PipelineResults:
    """Complete pipeline results"""
    task_type: str  # classification or regression
    target_column: str
    best_model: ModelResult
    all_models: List[ModelResult]
    feature_names: List[str]
    preprocessing_steps: List[str]
    X_train_shape: Tuple[int, int]
    X_test_shape: Tuple[int, int]


class MLPipeline:
    """
    Advanced Machine Learning Pipeline
    Automatic model selection, hyperparameter tuning, and evaluation
    """
    
    def __init__(self, 
                 task_type: str = "auto",
                 n_cv_folds: int = 5,
                 test_size: float = 0.2,
                 random_state: int = 42):
        self.task_type = task_type
        self.n_cv_folds = n_cv_folds
        self.test_size = test_size
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.results: Optional[PipelineResults] = None
        self.best_model = None
        
        # Model configurations
        self.classification_models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "SVM": SVC(probability=True, random_state=random_state),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=random_state),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=random_state),
            "Naive Bayes": GaussianNB()
        }
        
        self.regression_models = {
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=random_state),
            "Lasso": Lasso(random_state=random_state),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state),
            "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=random_state),
            "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=random_state)
        }
    
    def _detect_task_type(self, y: pd.Series) -> str:
        """Automatically detect if classification or regression"""
        unique_ratio = y.nunique() / len(y)
        
        if y.dtype == 'object' or y.dtype.name == 'category':
            return 'classification'
        elif unique_ratio < 0.05 or y.nunique() <= 10:
            return 'classification'
        else:
            return 'regression'
    
    def _preprocess_features(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocess features for modeling"""
        X_processed = X.copy()
        preprocessing_steps = []
        
        # Handle missing values
        for col in X_processed.columns:
            if X_processed[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(X_processed[col]):
                    X_processed[col] = X_processed[col].fillna(X_processed[col].median())
                    preprocessing_steps.append(f"Imputed {col} with median")
                else:
                    X_processed[col] = X_processed[col].fillna(X_processed[col].mode().iloc[0])
                    preprocessing_steps.append(f"Imputed {col} with mode")
        
        # Encode categorical variables
        for col in X_processed.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            self.label_encoders[col] = le
            preprocessing_steps.append(f"Label encoded {col}")
        
        # Scale numeric features
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numeric_cols] = self.scaler.fit_transform(X_processed[numeric_cols])
        preprocessing_steps.append("Scaled numeric features")
        
        return X_processed.values, preprocessing_steps
    
    def train(self, df: pd.DataFrame, target_column: str, 
              feature_columns: List[str] = None) -> PipelineResults:
        """Train multiple models and select the best one"""
        logger.info(f"Starting ML Pipeline for target: {target_column}")
        
        # Prepare features and target
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Detect task type
        if self.task_type == "auto":
            self.task_type = self._detect_task_type(y)
        logger.info(f"Task type: {self.task_type}")
        
        # Encode target for classification
        if self.task_type == 'classification' and y.dtype == 'object':
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y)
        
        # Preprocess features
        X_processed, preprocessing_steps = self._preprocess_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y if self.task_type == 'classification' else None
        )
        
        # Select models based on task type
        models = self.classification_models if self.task_type == 'classification' else self.regression_models
        
        # Train and evaluate all models
        all_results = []
        for name, model in models.items():
            try:
                result = self._train_single_model(
                    name, model, X_train, X_test, y_train, y_test, feature_columns
                )
                all_results.append(result)
                logger.info(f"{name}: CV Score = {result.cv_mean:.4f} (+/- {result.cv_std:.4f})")
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
        
        # Sort by CV score and select best
        all_results.sort(key=lambda x: x.cv_mean, reverse=True)
        best_result = all_results[0]
        self.best_model = best_result.model
        
        # Store results
        self.results = PipelineResults(
            task_type=self.task_type,
            target_column=target_column,
            best_model=best_result,
            all_models=all_results,
            feature_names=feature_columns,
            preprocessing_steps=preprocessing_steps,
            X_train_shape=X_train.shape,
            X_test_shape=X_test.shape
        )
        
        logger.info(f"Best model: {best_result.name} with CV score: {best_result.cv_mean:.4f}")
        return self.results
    
    def _train_single_model(self, name: str, model: Any,
                            X_train: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, y_test: np.ndarray,
                            feature_names: List[str]) -> ModelResult:
        """Train and evaluate a single model"""
        # Fit model
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        predictions = y_pred_test
        
        # Get probabilities for classification
        probabilities = None
        if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)
        
        # Cross-validation
        scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        cv = StratifiedKFold(n_splits=self.n_cv_folds) if self.task_type == 'classification' else self.n_cv_folds
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        
        # Calculate metrics
        if self.task_type == 'classification':
            train_score = accuracy_score(y_train, y_pred_train)
            test_score = accuracy_score(y_test, y_pred_test)
            metrics = {
                'accuracy': test_score,
                'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
            }
            if probabilities is not None and len(np.unique(y_test)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_test, probabilities[:, 1])
        else:
            train_score = r2_score(y_train, y_pred_train)
            test_score = r2_score(y_test, y_pred_test)
            metrics = {
                'r2': test_score,
                'mse': mean_squared_error(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test)
            }
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = dict(zip(feature_names, importance.tolist()))
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if len(coef.shape) > 1:
                coef = np.mean(np.abs(coef), axis=0)
            feature_importance = dict(zip(feature_names, np.abs(coef).tolist()))
        
        return ModelResult(
            name=name,
            model=model,
            train_score=train_score,
            test_score=test_score,
            cv_scores=cv_scores.tolist(),
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            metrics=metrics,
            feature_importance=feature_importance,
            predictions=predictions,
            probabilities=probabilities
        )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the best model"""
        if self.best_model is None:
            raise ValueError("No model trained. Call train() first.")
        
        X_processed, _ = self._preprocess_features(X)
        return self.best_model.predict(X_processed)
    
    def hyperparameter_tune(self, X: pd.DataFrame, y: pd.Series,
                            model_name: str, param_grid: Dict) -> Dict[str, Any]:
        """Perform hyperparameter tuning for a specific model"""
        if self.task_type == 'classification':
            base_model = self.classification_models.get(model_name)
        else:
            base_model = self.regression_models.get(model_name)
        
        if base_model is None:
            raise ValueError(f"Model {model_name} not found")
        
        X_processed, _ = self._preprocess_features(X)
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=self.n_cv_folds,
            scoring='accuracy' if self.task_type == 'classification' else 'r2',
            n_jobs=-1
        )
        grid_search.fit(X_processed, y)
        
        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": pd.DataFrame(grid_search.cv_results_)
        }
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all trained models"""
        if self.results is None:
            return pd.DataFrame()
        
        data = []
        for result in self.results.all_models:
            row = {
                "Model": result.name,
                "CV Mean": round(result.cv_mean, 4),
                "CV Std": round(result.cv_std, 4),
                "Train Score": round(result.train_score, 4),
                "Test Score": round(result.test_score, 4)
            }
            row.update({k: round(v, 4) for k, v in result.metrics.items()})
            data.append(row)
        
        return pd.DataFrame(data)
