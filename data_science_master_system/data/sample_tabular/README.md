# Sample Tabular Data

CSV and Parquet files for tabular ML.

## Files

| File | Rows | Columns | Task |
|------|------|---------|------|
| customer_churn.csv | 1000 | 10 | Binary classification |
| house_prices.csv | 500 | 12 | Regression |
| employee_data.csv | 300 | 8 | Classification |
| user_item_interactions.csv | 10000 | 4 | Recommendations |

## Load Data

```python
import pandas as pd
df = pd.read_csv('customer_churn.csv')
```
