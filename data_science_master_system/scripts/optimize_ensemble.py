import sys
import os
import pandas as pd
import numpy as np
import optuna
import logging
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_science_master_system.core.logger import get_logger
from data_science_master_system.features.engineering.titanic_features import TitanicFeatureGenerator

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger(__name__)

# Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
    """Load real Titanic data"""
    train_path = os.path.join(project_root, 'data', 'train.csv')
    test_path = os.path.join(project_root, 'data', 'test.csv')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File not found: {train_path}")
    return pd.read_csv(train_path), pd.read_csv(test_path)

def objective(trial, X, y):
    """Optuna objective to tune 3 models simultaneously for Ensemble"""
    
    # 1. XGBoost Params
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.3),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample', 0.6, 1.0),
        'random_state': 42,
        'n_jobs': 1,
        'verbosity': 0
    }

    # 2. LightGBM Params
    lgbm_params = {
        'n_estimators': trial.suggest_int('lgbm_n_estimators', 50, 300),
        'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('lgbm_lr', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('lgbm_feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('lgbm_bagging_fraction', 0.6, 1.0),
        'bagging_freq': 1,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': 1
    }

    # 3. RandomForest Params
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
        'max_depth': trial.suggest_int('rf_max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('rf_min_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('rf_min_leaf', 1, 5),
        'random_state': 42,
        'n_jobs': 1
    }

    # Instantiate Models
    clf1 = XGBClassifier(**xgb_params)
    clf2 = LGBMClassifier(**lgbm_params)
    clf3 = RandomForestClassifier(**rf_params)

    # Ensemble (Soft Voting)
    # Note: We are not tuning weights here to save time, assuming equal contribution 
    # of optimized models is a strong baseline.
    eclf = VotingClassifier(
        estimators=[('xgb', clf1), ('lgbm', clf2), ('rf', clf3)],
        voting='soft'
    )

    # Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(eclf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    
    return scores.mean()

def main():
    try:
        logger.info("Starting Smart Ensemble Optimization...")
        
        # Load & Prepare
        train_df, test_df = load_data()
        
        fg = TitanicFeatureGenerator()
        fg.fit(train_df)
        X = fg.transform(train_df)
        if 'Survived' in X.columns:
            X = X.drop(columns=['Survived'])
        y = train_df['Survived']
        
        X_test = fg.transform(test_df)
        if 'Survived' in X_test.columns:
            X_test = X_test.drop(columns=['Survived'])

        # Optimize
        logger.info("Tuning Ensemble (XGB+LGBM+RF)...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X, y), n_trials=30) # 30 trials for speed
        
        logger.info(f"Best CV Accuracy: {study.best_value:.4f}")
        logger.info("Training Final Ensemble with Best Params...")

        # Re-create best models
        bp = study.best_params
        
        xgb_final = XGBClassifier(
            n_estimators=bp['xgb_n_estimators'], max_depth=bp['xgb_max_depth'],
            learning_rate=bp['xgb_lr'], subsample=bp['xgb_subsample'],
            colsample_bytree=bp['xgb_colsample'], random_state=42, n_jobs=-1
        )
        lgbm_final = LGBMClassifier(
            n_estimators=bp['lgbm_n_estimators'], num_leaves=bp['lgbm_num_leaves'],
            learning_rate=bp['lgbm_lr'], feature_fraction=bp['lgbm_feature_fraction'],
            bagging_fraction=bp['lgbm_bagging_fraction'], bagging_freq=1,
            random_state=42, verbose=-1, n_jobs=-1
        )
        rf_final = RandomForestClassifier(
            n_estimators=bp['rf_n_estimators'], max_depth=bp['rf_max_depth'],
            min_samples_split=bp['rf_min_split'], min_samples_leaf=bp['rf_min_leaf'],
            random_state=42, n_jobs=-1
        )

        final_ensemble = VotingClassifier(
            estimators=[('xgb', xgb_final), ('lgbm', lgbm_final), ('rf', rf_final)],
            voting='soft'
        )
        
        final_ensemble.fit(X, y)
        
        # Predict & Save
        preds = final_ensemble.predict(X_test)
        
        sub = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': preds})
        out_file = os.path.join(project_root, 'submission_ensemble.csv')
        sub.to_csv(out_file, index=False)
        
        logger.info(f"Ensemble Submission Saved: {out_file}")
        logger.info("Process Complete.")

    except Exception as e:
        logger.error(f"Ensemble failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
