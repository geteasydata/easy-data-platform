# 📊 Sample Data

Sample datasets for all notebooks.

## Folder Structure

```
data/
├── sample_tabular/      # CSV, Parquet for ML
├── sample_images/       # Images for CV notebooks
├── sample_text/         # Text data for NLP
├── sample_time_series/  # Time series data
└── generate_sample_data.py  # Generator script
```

## Datasets Included

| Dataset | Type | Rows | Use Case |
|---------|------|------|----------|
| customer_churn.csv | Tabular | 1000 | Classification |
| house_prices.csv | Tabular | 500 | Regression |
| sales_timeseries.csv | Time Series | 3650 | Forecasting |
| product_reviews.csv | Text | 500 | Sentiment |
| iot_sensor_data.csv | Time Series | 2000 | Anomaly |
| user_item_interactions.csv | Tabular | 10000 | Recommenders |
| financial_timeseries.csv | Time Series | 500 | Forecasting |

## Generate Data

```bash
python generate_sample_data.py
```

## Data Sources

All datasets are synthetically generated for educational purposes.
For production, use real datasets from:
- Kaggle
- UCI ML Repository
- HuggingFace Datasets
