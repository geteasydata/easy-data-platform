import sys
import os
import pandas as pd
import numpy as np
import optuna
import logging
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_science_master_system.core.logger import get_logger
from data_science_master_system.features.engineering.titanic_features import TitanicFeatureGenerator
from data_science_master_system.features.expert import ExpertFeatureGen

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
    train_path = os.path.join(project_root, 'data', 'train.csv')
    test_path = os.path.join(project_root, 'data', 'test.csv')
    return pd.read_csv(train_path), pd.read_csv(test_path)

def add_leave_one_out_features(X_train, y_train, X_test):
    """
    Manually implements rigorous Leave-One-Out Group Survival Rates.
    This is the 'Expert Human Logic' part to avoid leakage.
    
    Groups to target: 'Ticket', 'FamilySize' (if combined with name to make unique family ID)
    """
    logger.info("Applying Expert Leave-One-Out Group Logic...")
    
    # Combined df for easier processing of groups (but carefully split later)
    # Actually, proper LOO is done on Train only. Test uses Train stats.
    
    # 1. Ticket Survival Rate
    # On Train: For each row, rate = (Sum(TicketGroup) - Me) / (Count(TicketGroup) - 1)
    # On Test: rate = (Sum(Train_TicketGroup)) / (Count(Train_TicketGroup))
    
    # Helper for LOO
    def get_loo_rate(subset_df, group_col, target_col):
        # Calculate global sums per group
        group_sums = subset_df.groupby(group_col)[target_col].sum()
        group_counts = subset_df.groupby(group_col)[target_col].count()
        
        # Vectorized LOO
        # Rate = (GroupSum - Target) / (GroupCount - 1)
        # Handle singleton groups (Count=1) -> -1 or global mean
        
        rates = []
        global_mean = subset_df[target_col].mean()
        
        for idx, row in subset_df.iterrows():
            g_val = row[group_col]
            t_val = row[target_col]
            
            g_sum = group_sums.get(g_val, 0)
            g_count = group_counts.get(g_val, 0)
            
            if g_count > 1:
                loo_sum = g_sum - t_val
                loo_count = g_count - 1
                rates.append(loo_sum / loo_count)
            else:
                rates.append(global_mean) # Fallback to mean (or -1) 
                
        return rates

    # Add to Train
    X_train_full = X_train.copy()
    X_train_full['Survived'] = y_train
    
    # Ticket Logic
    X_train['Ticket_GroupRate'] = get_loo_rate(X_train_full, 'Ticket', 'Survived')
    
    # Add to Test (Standard Mapping)
    # Map from Train stats
    ticket_stats = X_train_full.groupby('Ticket')['Survived'].agg(['sum', 'count'])
    global_mean = y_train.mean()
    
    test_rates = []
    for t_val in X_test['Ticket']:
        if t_val in ticket_stats.index:
            s, c = ticket_stats.loc[t_val]
            test_rates.append(s/c)
        else:
            test_rates.append(global_mean)
    X_test['Ticket_GroupRate'] = test_rates
    
    return X_train, X_test

def objective(trial, X, y):
    # Reduced search space for speed, focusing on power
    xgb_params = {
        'n_estimators': trial.suggest_int('x_n', 50, 200),
        'max_depth': trial.suggest_int('x_d', 3, 8),
        'learning_rate': trial.suggest_float('x_lr', 0.05, 0.3),
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42, 'n_jobs': 1, 'verbosity': 0
    }
    rf_params = {
        'n_estimators': trial.suggest_int('r_n', 50, 200),
        'max_depth': trial.suggest_int('r_d', 5, 15),
        'random_state': 42, 'n_jobs': 1
    }
    
    eclf = VotingClassifier(
        estimators=[
            ('xgb', XGBClassifier(**xgb_params)),
            ('rf', RandomForestClassifier(**rf_params))
        ],
        voting='soft'
    )
    
    return cross_val_score(eclf, X, y, cv=5, scoring='accuracy', n_jobs=1).mean()

def main():
    try:
        logger.info("Starting Expert Optimization...")
        
        # 1. Load
        train_df, test_df = load_data()
        y = train_df['Survived']
        
        # 2. Basic Feature Gen
        fg = TitanicFeatureGenerator()
        fg.fit(train_df)
        X_train = fg.transform(train_df).drop(columns=['Survived'], errors='ignore')
        X_test = fg.transform(test_df).drop(columns=['Survived'], errors='ignore')
        
        # 3. EXPERT LOGIC (Manual Injection for now to ensure correctness)
        # Using the Ticket column from original DFs (needs to be preserved or re-merged if dropped)
        # TitanicFeatureGenerator might drop Ticket. Let's check.
        # It creates 'Ticket_Freq' but might drop 'Ticket'.
        # We re-inject Ticket for the logic if needed, or use 'Ticket_Freq' proxy?
        # Better to re-extract Ticket from raw df logic or modify TitanicFeatures to keep it.
        # For safety, let's just use raw 'Ticket' column from loaded dfs.
        
        X_train['Ticket'] = train_df['Ticket']
        X_test['Ticket'] = test_df['Ticket']
        
        X_train, X_test = add_leave_one_out_features(X_train, y, X_test)
        
        # Drop raw Ticket now (it's a string)
        X_train = X_train.drop(columns=['Ticket'])
        X_test = X_test.drop(columns=['Ticket'])
        
        # 4. General Expert Logic (Semantic/Interactions) from Library
        expert = ExpertFeatureGen(apply_group_dynamics=False) # We did direct LOO above
        expert.fit(X_train, y)
        X_train = expert.transform(X_train)
        X_test = expert.transform(X_test)
        
        # 5. Optimize
        logger.info(f"Features in Use: {X_train.columns.tolist()}")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, X_train, y), n_trials=20)
        
        logger.info(f"Expert CV Score: {study.best_value:.4f}")
        
        # 6. Final Train & Submit
        bp = study.best_params
        final_model = VotingClassifier(
            estimators=[
                ('xgb', XGBClassifier(n_estimators=bp['x_n'], max_depth=bp['x_d'], learning_rate=bp['x_lr'], subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)),
                ('rf', RandomForestClassifier(n_estimators=bp['r_n'], max_depth=bp['r_d'], random_state=42, n_jobs=-1))
            ],
            voting='soft'
        )
        final_model.fit(X_train, y)
        preds = final_model.predict(X_test)
        
        sub = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': preds})
        sub.to_csv('submission_expert.csv', index=False)
        logger.info("Expert Submission Saved: submission_expert.csv")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
