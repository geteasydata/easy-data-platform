# 📋 Time Series Forecasting Cheatsheet

## 📌 Key Concepts
- **Trend**: Long-term direction
- **Seasonality**: Regular repeating patterns
- **Stationarity**: Constant mean/variance over time
- **Lag Features**: Past values as predictors
- **Horizon**: How far ahead to forecast

## 🛠️ Essential Code

### Prophet
```python
from prophet import Prophet

model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(df)  # df must have 'ds' and 'y' columns

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
model.plot(forecast)
```

### ARIMA
```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(data, order=(p, d, q))  # p=AR, d=diff, q=MA
fitted = model.fit()
forecast = fitted.forecast(steps=10)
```

### Deep Learning (Darts)
```python
from darts import TimeSeries
from darts.models import NBEATSModel

series = TimeSeries.from_dataframe(df, 'date', 'value')
model = NBEATSModel(input_chunk_length=30, output_chunk_length=7)
model.fit(series)
forecast = model.predict(n=7)
```

## 📊 Model Comparison
| Model | Type | Multivariate | Scalability |
|-------|------|--------------|-------------|
| ARIMA | Statistical | No | Low |
| Prophet | Statistical | No | Medium |
| N-BEATS | Deep Learning | Yes | High |
| TFT | Deep Learning | Yes | High |

## 📐 Key Formulas
```
MAPE = mean(|actual - predicted| / |actual|) × 100%
RMSE = sqrt(mean((actual - predicted)²))
MAE = mean(|actual - predicted|)
```

## ⚠️ Common Pitfalls
| Problem | Solution |
|---------|----------|
| Non-stationary data | Differencing, log transform |
| Missing values | Interpolation, forward fill |
| Seasonality ignored | Add seasonal components |
| Overfitting | Cross-validation with time splits |

## 🚀 Production Tips
- Update models with new data regularly
- Monitor forecast accuracy over time
- Use confidence intervals
- Ensemble multiple models
