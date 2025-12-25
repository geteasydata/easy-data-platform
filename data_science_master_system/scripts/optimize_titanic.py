import sys
import os
import pandas as pd
import numpy as np
import optuna
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Added {project_root} to sys.path")

from data_science_master_system.features.engineering.titanic_features import TitanicFeatureGenerator
from data_science_master_system.core.logger import get_logger

logger = get_logger(__name__)

def load_data():
    """Load Titantic data from local file or seaborn."""
    # Try local paths first
    possible_paths = [
        "data/titanic/train.csv",
        "data/train.csv",
        "../data/train.csv",
        "train.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Loading data from {path}")
            return pd.read_csv(path)
            
    # Fallback to seaborn
    try:
        import seaborn as sns
        logger.info("Local file not found. Loading from seaborn 'titanic' dataset.")
        df = sns.load_dataset('titanic')
        # Seaborn dataset needs some adjustment to match Kaggle format
        # It already has 'survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone'
        # We need to reshape it to match what TitanicFeatureGenerator expects (PascalCase cols)
        df = df.rename(columns={
            'survived': 'Survived',
            'pclass': 'Pclass',
            'sex': 'Sex',
            'age': 'Age',
            'sibsp': 'SibSp',
            'parch': 'Parch',
            'fare': 'Fare',
            'embarked': 'Embarked'
        })
        # Mock Name and Cabin/Ticket for compatibility
        df['Name'] = df.apply(lambda x: f"Person, {x['who']}. Mock", axis=1) # Rudimentary mock
        df['Ticket'] = 'MOCK'
        df['Cabin'] = df['deck'].astype(str) # Map deck back to Cabin roughly
        
        # Keep only standard columns
        standard_cols = ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        df = df[standard_cols]
        
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None

def objective(trial, X, y):
    """Optuna objective function."""
    
    classifier_name = trial.suggest_categorical("classifier", ["XGBoost", "LightGBM", "RandomForest"])
    
    if classifier_name == "XGBoost":
        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "booster": "gbtree",
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }
        model = xgb.XGBClassifier(**param)
        
    elif classifier_name == "LightGBM":
        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        model = lgb.LGBMClassifier(**param)
        
    else: # RandomForest
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
        model = RandomForestClassifier(**param)
    
    # Cross validations
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    
    return scores.mean()

def main():
    logger.info("Starting Titanic Optimization...")
    
    # 1. Load Data
    raw_df = load_data()
    if raw_df is None:
        logger.error("Could not load dataset. Please ensure 'train.csv' is in 'data/' or 'data/titanic/'.")
        return
    
    logger.info(f"Loaded data shape: {raw_df.shape}")
    
    # 2. Preprocess
    # Identify target
    target_col = "Survived"
    if target_col not in raw_df.columns:
         logger.error("Target 'Survived' not found in dataset.")
         return
         
    y = raw_df[target_col]
    X_raw = raw_df.drop(columns=[target_col])
    
    # Feature Engineering
    logger.info("Generating features...")
    feature_gen = TitanicFeatureGenerator()
    feature_gen.fit(raw_df) # Use full df for stats
    X = feature_gen.transform(X_raw)
    
    logger.info(f"Transformed data shape: {X.shape}")
    logger.info(f"Features: {list(X.columns)}")
    
    # 3. Optimization
    logger.info("Starting Optuna optimization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50) # 50 trials
    
    # 4. Results
    logger.info("Optimization complete.")
    logger.info(f"Best trial: {study.best_trial.value}")
    logger.info(f"Best params: {study.best_trial.params}")
    
    # Save best model
    # (Re-train on full data)
    best_params = study.best_params.copy()
    classifier_name = best_params.pop("classifier")
    
    if classifier_name == "XGBoost":
        model = xgb.XGBClassifier(**best_params)
    elif classifier_name == "LightGBM":
        model = lgb.LGBMClassifier(**best_params)
    else:
        model = RandomForestClassifier(**best_params)
        
    model.fit(X, y)
    
    # Save artifacts
    os.makedirs("models/titanic", exist_ok=True)
    joblib.dump(model, "models/titanic/best_model.pkl")
    joblib.dump(feature_gen, "models/titanic/feature_gen.pkl")
    logger.info("Best model saved to 'models/titanic/'")

if __name__ == "__main__":
    main()
