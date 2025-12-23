"""
Professional AutoML Engine - Enterprise-Grade Automatic Machine Learning
Integrates all professional modules for end-to-end ML pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import professional modules
from core.data_profiler import DataProfiler
from core.data_cleaner import ExpertDataCleaner
from core.feature_engineer import FeatureEngineer
from core.feature_selector import FeatureSelector
from core.imbalance_handler import ImbalanceHandler
from core.hypertuner import HyperTuner
from core.model_interpreter import ModelInterpreter

# Import AI understanding
try:
    from core.ai_understanding import AIDataExpert
    HAS_AI_UNDERSTANDING = True
except ImportError:
    HAS_AI_UNDERSTANDING = False

# Try to import advanced libraries
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class AutoML:
    """
    Professional AutoML Engine - Enterprise-Grade.
    Combines all professional modules for comprehensive ML pipeline.
    Works like a team of senior data scientists.
    """
    
    def __init__(self, random_state: int = 42, 
                 enable_feature_engineering: bool = True,
                 enable_hypertuning: bool = True,
                 enable_balancing: bool = True,
                 optimization_rounds: int = 3):
        self.random_state = random_state
        self.enable_feature_engineering = enable_feature_engineering
        self.enable_hypertuning = enable_hypertuning
        self.enable_balancing = enable_balancing
        self.optimization_rounds = optimization_rounds
        
        # Core attributes
        self.model = None
        self.best_model_name = None
        self.problem_type = None
        self.target_col = None
        self.feature_names = []
        self.label_encoders = {}
        self.scaler = None
        self.metrics = {}
        self.feature_importance = None
        
        # Professional modules
        self.profiler = DataProfiler()
        self.cleaner = ExpertDataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector()
        self.imbalance_handler = ImbalanceHandler()
        self.hypertuner = HyperTuner(random_state=random_state)
        self.interpreter = ModelInterpreter()
        
        # Logs
        self.pipeline_log = []
        self.data_profile = None
        
    def log(self, message: str):
        """Add to pipeline log."""
        self.pipeline_log.append(message)
    
    def detect_problem_type(self, y: pd.Series) -> str:
        """Intelligently detect problem type."""
        if y.dtype == 'object' or str(y.dtype) == 'category':
            return 'classification'
        
        n_unique = y.nunique()
        if n_unique <= 20:
            return 'classification'
        
        if n_unique / len(y) < 0.05:
            return 'classification'
        
        return 'regression'
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Deep data profiling."""
        self.data_profile = self.profiler.profile_dataset(df)
        return self.data_profile
    
    def get_models(self, problem_type: str) -> Dict[str, Any]:
        """Get all available models."""
        if problem_type == 'classification':
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
                'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'Decision Tree': DecisionTreeClassifier(random_state=self.random_state, max_depth=10),
            }
            
            if HAS_XGBOOST:
                models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=self.random_state, 
                                                   use_label_encoder=False, eval_metric='logloss', verbosity=0)
            if HAS_LIGHTGBM:
                models['LightGBM'] = LGBMClassifier(n_estimators=100, random_state=self.random_state, verbose=-1)
            if HAS_CATBOOST:
                models['CatBoost'] = CatBoostClassifier(n_estimators=100, random_state=self.random_state, verbose=0)
                
        else:  # regression
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(random_state=self.random_state),
                'Decision Tree': DecisionTreeRegressor(random_state=self.random_state, max_depth=10),
            }
            
            if HAS_XGBOOST:
                models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=self.random_state, verbosity=0)
            if HAS_LIGHTGBM:
                models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)
            if HAS_CATBOOST:
                models['CatBoost'] = CatBoostRegressor(n_estimators=100, random_state=self.random_state, verbose=0)
        
        return models
    
    def train(self, df: pd.DataFrame, target_col: str, 
              test_size: float = 0.2) -> Dict[str, Any]:
        """
        Complete training pipeline - professional grade.
        
        Pipeline:
        1. Data Profiling
        2. Data Cleaning
        3. Feature Engineering
        4. Feature Selection
        5. Imbalance Handling
        6. Model Training
        7. Hyperparameter Tuning (optional)
        8. Model Interpretation
        """
        self.target_col = target_col
        self.pipeline_log = []
        
        # ===== AUTO-SAMPLING FOR LARGE DATASETS =====
        MAX_ROWS = 100000  # Maximum rows to process in memory safely
        original_size = len(df)
        
        if len(df) > MAX_ROWS:
            self.log(f"⚠️ Dataset is large ({original_size:,} rows)")
            self.log(f"📉 Auto-sampling to {MAX_ROWS:,} rows for training")
            df = df.sample(n=MAX_ROWS, random_state=self.random_state)
            self.log(f"   Original: {original_size:,} → Sampled: {len(df):,}")
        
        # ===== STEP 1: Data Profiling =====
        self.log("📊 Step 1: Data Profiling")
        self.data_profile = self.profiler.profile_dataset(df, target_col)
        self.log(f"   Quality Score: {self.data_profile['quality_score']}/100")
        
        # ===== STEP 2: Data Cleaning =====
        self.log("🧹 Step 2: Data Cleaning")
        X, y = self.cleaner.clean(df, target_col)
        self.label_encoders = self.cleaner.label_encoders
        self.pipeline_log.extend(self.cleaner.cleaning_log)
        
        if len(X) == 0 or len(X.columns) == 0:
            raise ValueError("No data remaining after cleaning")
        
        # ===== STEP 3: Feature Engineering =====
        if self.enable_feature_engineering:
            self.log("⚙️ Step 3: Feature Engineering")
            X = self.feature_engineer.engineer_features(X)
            self.pipeline_log.extend(self.feature_engineer.feature_log)
        
        # ===== STEP 4: Detect Problem Type =====
        self.problem_type = self.detect_problem_type(y)
        self.log(f"📋 Problem Type: {self.problem_type}")
        
        # ===== STEP 5: Feature Selection =====
        self.log("🎯 Step 5: Feature Selection")
        X, selected_features = self.feature_selector.select_features(
            X, y, self.problem_type, n_features=min(50, len(X.columns))
        )
        self.pipeline_log.extend(self.feature_selector.selection_log)
        
        self.feature_names = list(X.columns)
        
        # ===== STEP 6: Imbalance Handling =====
        if self.enable_balancing and self.problem_type == 'classification':
            self.log("⚖️ Step 6: Imbalance Check")
            if self.imbalance_handler.check_imbalance(y):
                X, y = self.imbalance_handler.balance_data(X, y)
                self.pipeline_log.extend(self.imbalance_handler.log_messages)
        
        # ===== STEP 7: Scale Features =====
        self.log("📏 Step 7: Feature Scaling")
        self.scaler = StandardScaler()
        try:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X.fillna(0)), 
                columns=X.columns
            )
        except:
            X_scaled = X.fillna(0)
        
        # ===== STEP 8: Split Data =====
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.random_state
        )
        
        # ===== STEP 9: Model Training =====
        self.log("🤖 Step 8: Model Training")
        models = self.get_models(self.problem_type)
        results = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if self.problem_type == 'classification':
                    score = accuracy_score(y_test, y_pred)
                else:
                    score = max(0, r2_score(y_test, y_pred))
                
                results[name] = {
                    'model': model,
                    'score': score,
                    'predictions': y_pred
                }
                self.log(f"   ✓ {name}: {score:.4f}")
            except Exception as e:
                self.log(f"   ✗ {name}: Failed ({str(e)[:30]})")
                continue
        
        if not results:
            # Fallback
            self.log("⚠️ Using fallback model")
            if self.problem_type == 'classification':
                fallback = DecisionTreeClassifier(random_state=self.random_state, max_depth=5)
            else:
                fallback = DecisionTreeRegressor(random_state=self.random_state, max_depth=5)
            
            fallback.fit(X_train, y_train)
            y_pred = fallback.predict(X_test)
            score = accuracy_score(y_test, y_pred) if self.problem_type == 'classification' else max(0, r2_score(y_test, y_pred))
            
            results['Fallback'] = {'model': fallback, 'score': score, 'predictions': y_pred}
        
        # ===== STEP 10: Select Best Model =====
        best_name = max(results, key=lambda x: results[x]['score'])
        self.model = results[best_name]['model']
        self.best_model_name = best_name
        
        self.log(f"🏆 Best Model: {best_name} (Score: {results[best_name]['score']:.4f})")
        
        # ===== STEP 11: Hyperparameter Tuning & Optimization =====
        if self.enable_hypertuning:
            self.log("🔧 Step 11: Hyperparameter Tuning for Higher Accuracy")
            original_score = results[best_name]['score']
            
            # Try optimization rounds
            for round_num in range(self.optimization_rounds):
                self.log(f"   🔄 Optimization Round {round_num + 1}/{self.optimization_rounds}")
                
                best_params = self.hypertuner.tune(
                    best_name, X_train, y_train, self.problem_type, 
                    n_trials=30 + (round_num * 10),  # More trials each round
                    timeout=120
                )
                
                if best_params:
                    # Get the model class and create with tuned params
                    tuned_model = self._create_tuned_model(best_name, best_params)
                    if tuned_model is not None:
                        tuned_model.fit(X_train, y_train)
                        y_pred_tuned = tuned_model.predict(X_test)
                        
                        if self.problem_type == 'classification':
                            tuned_score = accuracy_score(y_test, y_pred_tuned)
                        else:
                            tuned_score = max(0, r2_score(y_test, y_pred_tuned))
                        
                        # Keep if better
                        if tuned_score > results[best_name]['score']:
                            self.model = tuned_model
                            results[best_name]['model'] = tuned_model
                            results[best_name]['score'] = tuned_score
                            results[best_name]['predictions'] = y_pred_tuned
                            self.log(f"   ✅ Improved: {original_score:.4f} → {tuned_score:.4f}")
                        else:
                            self.log(f"   ➡️ No improvement in round {round_num + 1}")
            
            final_score = results[best_name]['score']
            if final_score > original_score:
                self.log(f"🎯 Total Improvement: {original_score:.4f} → {final_score:.4f} (+{(final_score-original_score)*100:.2f}%)")
            
            self.pipeline_log.extend(self.hypertuner.tuning_log)

        # ===== STEP 12: Advanced Ensemble (Voting) =====
        try:
            self.log("🤝 Step 12: creating Voting Ensemble (Top 3 Models)")
            
            # Sort models by score
            sorted_models = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
            top_3 = sorted_models[:3]
            
            if len(top_3) >= 2:
                estimators = [(name, data['model']) for name, data in top_3]
                
                if self.problem_type == 'classification':
                    from sklearn.ensemble import VotingClassifier
                    ensemble = VotingClassifier(estimators=estimators, voting='soft')
                else:
                    from sklearn.ensemble import VotingRegressor
                    ensemble = VotingRegressor(estimators=estimators)
                
                ensemble.fit(X_train, y_train)
                y_pred_ens = ensemble.predict(X_test)
                
                if self.problem_type == 'classification':
                    ens_score = accuracy_score(y_test, y_pred_ens)
                else:
                    ens_score = max(0, r2_score(y_test, y_pred_ens))
                
                self.log(f"   👥 Ensemble Score: {ens_score:.4f}")
                
                # Use ensemble if better
                if ens_score > results[best_name]['score']:
                    self.model = ensemble
                    self.best_model_name = "Voting Ensemble"
                    results['Voting Ensemble'] = {
                        'model': ensemble,
                        'score': ens_score,
                        'predictions': y_pred_ens
                    }
                    best_name = 'Voting Ensemble'  # Update best name for metrics calculation
                    self.log(f"   🏆 Voting Ensemble is the new champion!")
        except Exception as e:
            self.log(f"   ⚠️ Ensemble failed: {e}")
        
        # ===== STEP 12: Calculate Final Metrics =====
        y_pred = results[best_name]['predictions']
        
        if self.problem_type == 'classification':
            self.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            }
        else:
            self.metrics = {
                'r2': max(0, r2_score(y_test, y_pred)),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
            }
        
        # ===== STEP 13: Model Interpretation =====
        self.log("🔍 Step 10: Model Interpretation")
        interpretation = self.interpreter.interpret(self.model, X_test, feature_names=self.feature_names)
        self.feature_importance = interpretation.get('feature_importance')
        
        if self.feature_importance is None:
            self.feature_importance = self._fallback_importance()
        
        self.log("✅ Training Complete!")
        
        return {
            'problem_type': self.problem_type,
            'best_model': self.best_model_name,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'cleaning_steps': self.pipeline_log,
            'all_models': {k: v['score'] for k, v in results.items()},
            'data_profile': self.data_profile,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def _create_tuned_model(self, model_name: str, params: Dict[str, Any]):
        """Create a model instance with tuned hyperparameters."""
        params = params.copy()
        params['random_state'] = self.random_state
        
        try:
            if model_name == 'Random Forest':
                if self.problem_type == 'classification':
                    return RandomForestClassifier(**params, n_jobs=-1)
                else:
                    return RandomForestRegressor(**params, n_jobs=-1)
            
            elif model_name == 'Gradient Boosting':
                if self.problem_type == 'classification':
                    return GradientBoostingClassifier(**params)
                else:
                    return GradientBoostingRegressor(**params)
            
            elif model_name == 'XGBoost' and HAS_XGBOOST:
                from xgboost import XGBClassifier, XGBRegressor
                params['verbosity'] = 0
                params['use_label_encoder'] = False
                if self.problem_type == 'classification':
                    params['eval_metric'] = 'logloss'
                    return XGBClassifier(**params)
                else:
                    return XGBRegressor(**params)
            
            elif model_name == 'LightGBM' and HAS_LIGHTGBM:
                from lightgbm import LGBMClassifier, LGBMRegressor
                params['verbose'] = -1
                if self.problem_type == 'classification':
                    return LGBMClassifier(**params)
                else:
                    return LGBMRegressor(**params)
            
            elif model_name == 'CatBoost' and HAS_CATBOOST:
                from catboost import CatBoostClassifier, CatBoostRegressor
                params['verbose'] = 0
                if self.problem_type == 'classification':
                    return CatBoostClassifier(**params)
                else:
                    return CatBoostRegressor(**params)
            
            elif model_name == 'Logistic Regression':
                return LogisticRegression(**params, max_iter=1000)
            
            elif model_name == 'Ridge':
                return Ridge(**params)
        
        except Exception as e:
            self.log(f"   ⚠️ Could not create tuned model: {e}")
            return None
        
        return None
    
    def _fallback_importance(self) -> pd.DataFrame:
        """Create fallback feature importance."""
        return pd.DataFrame({
            'Feature': self.feature_names[:10],
            'Importance': np.ones(min(10, len(self.feature_names))) / min(10, len(self.feature_names))
        })
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create a copy to avoid modifying original
        X = X.copy()
        
        # Remove target column if present
        if self.target_col and self.target_col in X.columns:
            X = X.drop(columns=[self.target_col])
        
        # Apply same cleaning as training
        # 1. Handle categorical columns with label encoders
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if col in self.label_encoders:
                # Use the same encoder from training
                le = self.label_encoders[col]
                try:
                    # Handle unseen labels by setting to -1
                    X[col] = X[col].fillna('Unknown')
                    X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                except:
                    X[col] = 0
            else:
                # New categorical column - simple label encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X[col] = X[col].fillna('Unknown').astype(str)
                try:
                    X[col] = le.fit_transform(X[col])
                except:
                    X[col] = 0
        
        # 2. Fill missing numeric values
        X = X.fillna(0)
        
        # 3. Ensure all required columns exist
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        
        # 4. Select only the features used in training
        X = X[self.feature_names]
        
        # 5. Apply scaling
        if self.scaler:
            try:
                X = pd.DataFrame(self.scaler.transform(X), columns=self.feature_names)
            except:
                pass
        
        return self.model.predict(X)
    
    def save_model(self, path):
        """Save model and preprocessors."""
        save_dict = {
            'model': self.model,
            'best_model_name': self.best_model_name,
            'problem_type': self.problem_type,
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'metrics': self.metrics,
        }
        joblib.dump(save_dict, path)
    
    @classmethod
    def load_model(cls, path: str) -> 'AutoML':
        """Load a saved model."""
        save_dict = joblib.load(path)
        instance = cls()
        for key, value in save_dict.items():
            setattr(instance, key, value)
        return instance
