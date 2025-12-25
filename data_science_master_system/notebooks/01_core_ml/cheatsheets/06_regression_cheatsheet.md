# 📋 Regression Models Cheatsheet (Notebook 06)

## 📌 Key Algorithms

| Model | Use When | Pros | Cons |
|-------|----------|------|------|
| Linear Regression | Linear relationship | Fast, interpretable | Assumes linearity |
| Ridge (L2) | Multicollinearity | Handles correlated features | Less sparse |
| Lasso (L1) | Feature selection | Sparse solutions | Unstable with correlated |
| ElasticNet | Best of both | Balanced | Two hyperparameters |
| Random Forest | Non-linear | No scaling needed | Slow inference |
| XGBoost | Tabular data | State-of-art | Tuning complexity |

## 🛠️ Essential Code

### Linear Regression
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Regularized Regression
```python
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
```

### XGBoost
```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6
)
model.fit(X_train, y_train)
```

## 📐 Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| MSE | mean((y - ŷ)²) | Penalize large errors |
| RMSE | √MSE | Same scale as target |
| MAE | mean(|y - ŷ|) | Robust to outliers |
| R² | 1 - SS_res/SS_tot | Explained variance |
| MAPE | mean(|y - ŷ|/y) × 100 | Percentage error |

## ⚠️ Common Pitfalls
- Not scaling features for regularized models
- Using R² for time series (use MAPE)
- Ignoring residual analysis
- Extrapolating beyond training range

## 🚀 Tuning Tips
```python
# Ridge alpha tuning
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(alphas=[0.1, 1, 10, 100])
```
