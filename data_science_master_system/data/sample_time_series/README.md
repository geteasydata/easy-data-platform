# Sample Time Series Data

Time series data for forecasting and anomaly detection.

## Files

| File | Rows | Frequency | Task |
|------|------|-----------|------|
| sales_timeseries.csv | 3650 | Daily | Forecasting |
| iot_sensor_data.csv | 2000 | Minute | Anomaly detection |
| financial_timeseries.csv | 500 | Business day | Forecasting |

## Load Data

```python
import pandas as pd
df = pd.read_csv('sales_timeseries.csv', parse_dates=['date'])
```
