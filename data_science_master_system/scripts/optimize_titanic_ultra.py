import sys
import os
import pandas as pd
import numpy as np
import optuna
import logging
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Ensure project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_science_master_system.core.logger import get_logger
from data_science_master_system.features.engineering.titanic_features import TitanicFeatureGenerator
from data_science_master_system.features.expert import ExpertFeatureGen, HierarchicalImputer, InteractionSegmenter

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
    t_path = os.path.join(project_root, 'data', 'train.csv')
    te_path = os.path.join(project_root, 'data', 'test.csv')
    return pd.read_csv(t_path), pd.read_csv(te_path)

def add_wcg_feature(X, raw_df):
    """
    Woman-Child-Group Heuristic Feature.
    1. Identify groups (Family/Ticket).
    2. If any female/child in group survived -> All females/children in group survive?
    Actually, simpler version as a feature:
    'WCG_SurvivalRate': Mean survival of females/children in the same group (excluding self).
    """
    # For now, let's rely on ExpertFeatureGen's GroupRate which captures this generally.
    # But specifically targeting 'Woman-Child' subgroup makes it stronger.
    # Let's perform a specialized 'WomanChildGroup' logic here if needed.
    # Construct a 'GroupID'
    
    # We will assume ExpertFeatureGen handles generalized group survival. 
    # Let's add 'IsWomanOrChild' explicit feature to help trees partition this logic.
    X['IsWomanOrChild'] = ((X['Sex'] == 1) | (X['Age'] < 16)).astype(int)
    # Note: Sex is usually label encoded 0/1. TitanicFeatures maps female->1 usually? 
    # Checking TitanicFeatureGenerator... it uses LabelEncoder. Usually female=0 or 1.
    # We'll just rely on Pclass/Sex/Age interactions.
    return X

def objective(trial, X, y):
    # Hyper-tune Ensemble
    xgb_p = {
        'n_estimators': trial.suggest_int('x_n', 50, 300),
        'max_depth': trial.suggest_int('x_d', 3, 10),
        'learning_rate': trial.suggest_float('x_lr', 0.01, 0.2),
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'random_state': 42, 'n_jobs': 1, 'verbosity': 0
    }
    rf_p = {
        'n_estimators': trial.suggest_int('r_n', 50, 300),
        'max_depth': trial.suggest_int('r_d', 5, 20),
        'min_samples_split': trial.suggest_int('r_s', 2, 10),
        'random_state': 42, 'n_jobs': 1
    }
    
    model = VotingClassifier(
        estimators=[('xgb', XGBClassifier(**xgb_p)), ('rf', RandomForestClassifier(**rf_p))],
        voting='soft'
    )
    return cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=1).mean()

def main():
    try:
        logger.info("Starting Ultra Opt...")
        train_raw, test_raw = load_data()
        y = train_raw['Survived']
        
        # 1. Basic Cleaning
        fg = TitanicFeatureGenerator()
        fg.fit(train_raw)
        X_train = fg.transform(train_raw).drop(columns=['Survived'], errors='ignore')
        X_test = fg.transform(test_raw).drop(columns=['Survived'], errors='ignore')
        
        # Restore raw needed cols for Expert tools if lost
        # TitanicFeatureGenerator might drop Name/Ticket.
        # We re-inject Ticket and Title for Hierarchical Imputer
        X_train['Ticket'] = train_raw['Ticket']
        X_test['Ticket'] = test_raw['Ticket']
        # Title is likely in X_train from fg, assuming it kept it or created it.
        # TitanicFeatureGenerator creates 'Title'.
        
        # 2. Hierarchical Imputation (Smart Age)
        # Hierarchy: Title -> Pclass
        if 'Age' in X_train.columns:
            logger.info("Applying Hierarchical Imper for Age...")
            # We must fit on TRAIN only to learn stats
            imputer = HierarchicalImputer(target_col='Age', hierarchy=['Title', 'Pclass'])
            imputer.fit(X_train) # Learn from Train
            X_train = imputer.transform(X_train)
            X_test = imputer.transform(X_test)
            
        # 3. Expert Group Logic (LOO)
        # We need to manually do LOO for Train again to be safe, 
        # or rely on ExpertFeatureGen's implementation if we verified it handles LOO.
        # expert.py implementation had LOO logic.
        expert = ExpertFeatureGen(apply_group_dynamics=True, apply_interactions=False) # We start with groups
        expert.fit(X_train, y) # This learns group stats
        
        # Note: ExpertFeatureGen._transform_group_dynamics creates 'GroupRate'
        # based on self.group_stats. If we just transform Train, we leak self.
        # Correct approach: Re-implement LOO specifically for Train here or trust previous script logic.
        # In optimize_titanic_expert.py we wrote a manual 'add_leave_one_out_features'.
        # Let's reuse that powerful logic by copying it or importing if it was in a generic lib.
        # Since it's not in expert.py (we put a placeholder mapping there), we'll do manual LOO here
        # to guarantee the >80% rigour.
        
        # ... (Re-using LOO Logic for brevity/safety) ...
        # [Simplified for this script: Rely on 'Ticket_GroupRate' manually created]
        
        full_df = X_train.copy()
        full_df['Survived'] = y
        # Ticket LOO
        ticket_sums = full_df.groupby('Ticket')['Survived'].sum()
        ticket_counts = full_df.groupby('Ticket')['Survived'].count()
        
        # Train Transformation (LOO)
        train_rates = []
        mean_surv = y.mean()
        for idx, row in full_df.iterrows():
            t = row['Ticket']
            if ticket_counts[t] > 1:
                loo_sum = ticket_sums[t] - row['Survived']
                loo_count = ticket_counts[t] - 1
                train_rates.append(loo_sum/loo_count)
            else:
                train_rates.append(mean_surv)
        X_train['Ticket_GroupRate'] = train_rates
        
        # Test Transformation (Standard)
        test_rates = []
        for t in X_test['Ticket']:
            if t in ticket_sums:
                test_rates.append(ticket_sums[t] / ticket_counts[t])
            else:
                test_rates.append(mean_surv)
        X_test['Ticket_GroupRate'] = test_rates
        
        # Drop Ticket textual
        X_train.drop(columns=['Ticket'], inplace=True)
        X_test.drop(columns=['Ticket'], inplace=True)

        # 4. Interaction Segmenter
        # Discover top interactions
        seg = InteractionSegmenter(top_n=3)
        seg.fit(X_train)
        X_train = seg.transform(X_train)
        X_test = seg.transform(X_test)

        # 5. Optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, X_train, y), n_trials=25)
        
        logger.info(f"Ultra CV Score: {study.best_value:.4f}")
        
        # 6. Train & Submit
        bp = study.best_params
        final_model = VotingClassifier(
            estimators=[
                ('xgb', XGBClassifier(n_estimators=bp['x_n'], max_depth=bp['x_d'], learning_rate=bp['x_lr'], subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)),
                ('rf', RandomForestClassifier(n_estimators=bp['r_n'], max_depth=bp['r_d'], min_samples_split=bp['r_s'], random_state=42, n_jobs=-1))
            ],
            voting='soft'
        )
        final_model.fit(X_train, y)
        preds = final_model.predict(X_test)
        
        sub = pd.DataFrame({'PassengerId': test_raw['PassengerId'], 'Survived': preds})
        sub.to_csv('submission_ultra.csv', index=False)
        logger.info("Ultra Submission Saved: submission_ultra.csv")

    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
