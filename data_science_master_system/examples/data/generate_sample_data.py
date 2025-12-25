"""
Generate comprehensive sample datasets for all notebooks.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

np.random.seed(42)

def generate_all_datasets():
    """Generate all sample datasets."""
    
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / 'csv'
    json_dir = output_dir / 'json'
    csv_dir.mkdir(exist_ok=True)
    json_dir.mkdir(exist_ok=True)
    
    # 1. Customer Churn Dataset (Classification)
    n = 1000
    churn_df = pd.DataFrame({
        'customer_id': [f'CUST_{i:05d}' for i in range(n)],
        'tenure_months': np.random.randint(1, 72, n),
        'monthly_charges': np.random.uniform(20, 100, n).round(2),
        'total_charges': np.random.uniform(100, 5000, n).round(2),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
        'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check'], n),
        'num_support_tickets': np.random.poisson(2, n),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n),
        'online_security': np.random.choice(['Yes', 'No'], n),
        'churn': np.random.choice([0, 1], n, p=[0.73, 0.27])
    })
    churn_df.to_csv(csv_dir / 'customer_churn.csv', index=False)
    
    # 2. House Prices Dataset (Regression)
    n = 500
    house_df = pd.DataFrame({
        'house_id': [f'H_{i:04d}' for i in range(n)],
        'sqft_living': np.random.randint(800, 5000, n),
        'bedrooms': np.random.randint(1, 6, n),
        'bathrooms': np.random.uniform(1, 4, n).round(1),
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n),
        'waterfront': np.random.choice([0, 1], n, p=[0.95, 0.05]),
        'view': np.random.randint(0, 5, n),
        'condition': np.random.randint(1, 6, n),
        'grade': np.random.randint(4, 13, n),
        'year_built': np.random.randint(1920, 2024, n),
        'year_renovated': np.random.choice([0] * 80 + list(range(1990, 2024)), n),
        'price': np.random.uniform(100000, 1500000, n).round(0).astype(int)
    })
    house_df.to_csv(csv_dir / 'house_prices.csv', index=False)
    
    # 3. Sales Time Series Dataset
    dates = pd.date_range('2022-01-01', periods=730, freq='D')
    categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Sports']
    sales_records = []
    
    for cat in categories:
        trend = np.linspace(100, 150 if cat == 'Electronics' else 120, len(dates))
        seasonality = 30 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        weekly = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        noise = np.random.normal(0, 10, len(dates))
        sales = trend + seasonality + weekly + noise
        
        for i, date in enumerate(dates):
            sales_records.append({
                'date': date.strftime('%Y-%m-%d'),
                'product_category': cat,
                'sales': max(0, sales[i]),
                'units': max(0, int(sales[i] / 10)),
                'day_of_week': date.strftime('%A'),
                'is_weekend': 1 if date.weekday() >= 5 else 0
            })
    
    sales_df = pd.DataFrame(sales_records)
    sales_df.to_csv(csv_dir / 'sales_timeseries.csv', index=False)
    
    # 4. Product Reviews Dataset (NLP)
    n = 500
    positive = ['great product', 'excellent quality', 'love it', 'highly recommend', 'best purchase', 'works perfectly']
    negative = ['terrible quality', 'waste of money', 'does not work', 'very disappointed', 'returned immediately', 'avoid']
    neutral = ['okay product', 'nothing special', 'average quality', 'does the job', 'as expected', 'fair price']
    
    reviews = []
    for i in range(n):
        if i < n * 0.4:
            sentiment = 'positive'
            text = np.random.choice(positive) + '. ' + np.random.choice(positive)
        elif i < n * 0.7:
            sentiment = 'negative'
            text = np.random.choice(negative) + '. ' + np.random.choice(negative)
        else:
            sentiment = 'neutral'
            text = np.random.choice(neutral) + '. ' + np.random.choice(neutral)
        
        reviews.append({
            'review_id': f'R_{i:05d}',
            'product_id': f'P_{np.random.randint(1, 100):03d}',
            'text': text,
            'rating': np.random.choice([4, 5]) if sentiment == 'positive' else (np.random.choice([1, 2]) if sentiment == 'negative' else 3),
            'sentiment': sentiment,
            'date': (datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d')
        })
    
    reviews_df = pd.DataFrame(reviews)
    reviews_df.to_csv(csv_dir / 'product_reviews.csv', index=False)
    
    # 5. IoT Sensor Data (Anomaly Detection)
    n = 2000
    t = np.arange(n)
    temp = 22 + 3 * np.sin(2 * np.pi * t / 100) + np.random.normal(0, 0.5, n)
    humidity = 45 + 10 * np.cos(2 * np.pi * t / 150) + np.random.normal(0, 2, n)
    pressure = 1013 + np.random.normal(0, 3, n)
    
    # Inject anomalies
    anomaly_idx = np.random.choice(n, 50, replace=False)
    temp[anomaly_idx] += np.random.choice([-10, 10], 50)
    
    iot_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='1min'),
        'sensor_id': 'SENSOR_001',
        'temperature': temp.round(2),
        'humidity': humidity.round(2),
        'pressure': pressure.round(2),
        'is_anomaly': [1 if i in anomaly_idx else 0 for i in range(n)]
    })
    iot_df.to_csv(csv_dir / 'iot_sensor_data.csv', index=False)
    
    # 6. User-Item Interactions (Recommender Systems)
    n_users, n_items = 500, 200
    n_interactions = 10000
    
    interactions = pd.DataFrame({
        'user_id': np.random.randint(0, n_users, n_interactions),
        'item_id': np.random.randint(0, n_items, n_interactions),
        'rating': np.random.randint(1, 6, n_interactions),
        'timestamp': pd.date_range('2024-01-01', periods=n_interactions, freq='1min')
    }).drop_duplicates(['user_id', 'item_id'])
    interactions.to_csv(csv_dir / 'user_item_interactions.csv', index=False)
    
    # 7. Financial Data (Advanced Time Series)
    n = 500
    prices = [100]
    for _ in range(n - 1):
        change = np.random.normal(0, 2)
        prices.append(max(10, prices[-1] + change))
    
    financial_df = pd.DataFrame({
        'date': pd.date_range('2022-01-01', periods=n, freq='B'),
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'close': [p + np.random.uniform(-1, 1) for p in prices],
        'volume': np.random.randint(100000, 10000000, n)
    })
    financial_df.to_csv(csv_dir / 'financial_timeseries.csv', index=False)
    
    print(f"✅ Generated 7 datasets in {csv_dir}")
    return {
        'customer_churn': len(churn_df),
        'house_prices': len(house_df),
        'sales_timeseries': len(sales_df),
        'product_reviews': len(reviews_df),
        'iot_sensor': len(iot_df),
        'user_item': len(interactions),
        'financial': len(financial_df)
    }

if __name__ == '__main__':
    stats = generate_all_datasets()
    print("\n📊 Dataset Statistics:")
    for name, count in stats.items():
        print(f"  {name}: {count} records")
