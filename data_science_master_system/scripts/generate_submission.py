import sys
import os
import pandas as pd
import numpy as np
import logging
from xgboost import XGBClassifier

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_science_master_system.core.logger import get_logger
from data_science_master_system.features.engineering.titanic_features import TitanicFeatureGenerator

logger = get_logger(__name__)

def load_data():
    """Load data from real CSV files"""
    try:
        train_path = os.path.join(project_root, 'data', 'train.csv')
        test_path = os.path.join(project_root, 'data', 'test.csv')
        
        logger.info(f"Loading data from: {train_path} and {test_path}")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
             # Fallback to download if still missing (safety net)
             raise FileNotFoundError("Data files not found. They should have been downloaded.")
             
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        return train_df, test_df
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def main():
    try:
        logger.info("Starting Submission Generation...")
        
        # 1. Load Data
        train_df, test_df = load_data()
        logger.info(f"Loaded Train: {train_df.shape}, Test: {test_df.shape}")
        
        # 2. Feature Engineering
        fg = TitanicFeatureGenerator()
        fg.fit(train_df)
        
        X_train = fg.transform(train_df)
        if 'Survived' in X_train.columns:
            X_train = X_train.drop(columns=['Survived'])
            
        y_train = train_df['Survived']
        X_test = fg.transform(test_df)
        
        # 3. Model Training
        # Classifier: XGBoost
        best_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'enable_categorical': True,
            'random_state': 42
        }
        
        logger.info(f"Training XGBoost with params: {best_params}")
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        
        # 4. Prediction
        logger.info("Generating predictions...")
        predictions = model.predict(X_test)
        
        # 5. Create Submission File
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'],
            'Survived': predictions
        })
        
        output_path = os.path.join(project_root, 'submission.csv')
        submission.to_csv(output_path, index=False)
        logger.info(f"Submission saved to: {output_path}")
        logger.info(f"Submission contains {len(submission)} rows (should be 418).")
        
    except Exception as e:
        logger.error(f"Submission generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
