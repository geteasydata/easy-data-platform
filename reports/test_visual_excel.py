
import pandas as pd
import numpy as np
import os
import sys

# Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reports.excel_output import create_excel_report

def create_dummy_data():
    # 1. Main DataFrame
    np.random.seed(42)
    rows = 200
    df = pd.DataFrame({
        'CustomerID': [f'CUST-{i:04d}' for i in range(rows)],
        'Age': np.random.randint(18, 70, size=rows),
        'Income': np.random.normal(50000, 15000, size=rows),
        'Score': np.random.randint(1, 100, size=rows),
        'Category': np.random.choice(['Gold', 'Silver', 'Bronze'], size=rows),
        'Churn': np.random.choice([0, 1], size=rows, p=[0.8, 0.2])
    })
    
    # 2. Assertions/Metrics
    metrics = {
        'accuracy': 0.875,
        'f1_score': 0.82,
        'roc_auc': 0.91
    }
    
    # 3. Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': ['Income', 'Age', 'Score', 'Category'],
        'Importance': [0.45, 0.25, 0.20, 0.10]
    })
    
    # 4. Insights
    insights = """
    - High income customers show 40% less churn probability.
    - Bronze category needs immediate attention.
    - Age is a stabilizing factor after 45 years old.
    """
    
    return df, metrics, feature_importance, insights

def main():
    print("Generating Dummy Data...")
    df, metrics, fi, insights = create_dummy_data()
    
    print("Creating Excel Dashboard...")
    # Generate for English
    excel_bytes_en = create_excel_report(
        df_clean=df,
        df_predictions=df, # Just passing same df for test
        feature_importance=fi,
        metrics=metrics,
        insights=insights,
        lang='en'
    )
    
    # Generate for Arabic
    excel_bytes_ar = create_excel_report(
        df_clean=df,
        df_predictions=df,
        feature_importance=fi,
        metrics=metrics,
        insights="هذا نص تجريبي للرؤى باللغة العربية",
        lang='ar'
    )
    
    # Save files
    with open('test_dashboard_EN.xlsx', 'wb') as f:
        f.write(excel_bytes_en)
    
    with open('test_dashboard_AR.xlsx', 'wb') as f:
        f.write(excel_bytes_ar)
        
    print(f"✅ Created 'test_dashboard_EN.xlsx' ({len(excel_bytes_en)/1024:.1f} KB)")
    print(f"✅ Created 'test_dashboard_AR.xlsx' ({len(excel_bytes_ar)/1024:.1f} KB)")

if __name__ == "__main__":
    main()
