"""
Airflow DAG Template for ML Pipelines.

Complete ML pipeline with data processing, training, and deployment.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# Default arguments
default_args = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='End-to-end ML training pipeline',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'training', 'production'],
)


def extract_data(**context):
    """Extract data from source."""
    import pandas as pd
    
    # Example: Load from database or S3
    df = pd.read_csv('s3://bucket/data/raw_data.csv')
    
    # Save to staging
    df.to_parquet('/tmp/staging/raw_data.parquet')
    
    context['ti'].xcom_push(key='row_count', value=len(df))
    return len(df)


def validate_data(**context):
    """Validate data quality."""
    import pandas as pd
    
    df = pd.read_parquet('/tmp/staging/raw_data.parquet')
    
    # Quality checks
    checks = {
        'no_nulls': df.isnull().sum().sum() == 0,
        'min_rows': len(df) >= 1000,
        'no_duplicates': df.duplicated().sum() == 0,
    }
    
    if not all(checks.values()):
        raise ValueError(f"Data validation failed: {checks}")
    
    return checks


def feature_engineering(**context):
    """Create features for model training."""
    import pandas as pd
    
    df = pd.read_parquet('/tmp/staging/raw_data.parquet')
    
    # Feature transformations
    # Add your feature engineering logic here
    
    df.to_parquet('/tmp/staging/features.parquet')
    return df.shape


def train_model(**context):
    """Train ML model."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    df = pd.read_parquet('/tmp/staging/features.parquet')
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    
    # Save model
    joblib.dump(model, '/tmp/models/model.joblib')
    
    context['ti'].xcom_push(key='accuracy', value=accuracy)
    return accuracy


def evaluate_model(**context):
    """Evaluate model and decide on deployment."""
    accuracy = context['ti'].xcom_pull(key='accuracy', task_ids='train_model')
    
    # Deployment threshold
    if accuracy >= 0.85:
        return 'deploy'
    else:
        return 'skip_deploy'


def deploy_model(**context):
    """Deploy model to production."""
    import shutil
    
    # Copy to production path
    shutil.copy('/tmp/models/model.joblib', '/production/models/model.joblib')
    
    # Trigger model server restart
    # requests.post('http://model-server/reload')
    
    return 'deployed'


# Task definitions
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

feature_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

notify_task = BashOperator(
    task_id='notify_completion',
    bash_command='echo "Pipeline completed at $(date)"',
    dag=dag,
)

# DAG dependencies
extract_task >> validate_task >> feature_task >> train_task >> evaluate_task >> deploy_task >> notify_task
