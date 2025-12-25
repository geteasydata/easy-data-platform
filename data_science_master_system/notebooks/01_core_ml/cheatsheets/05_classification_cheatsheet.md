# 📋 Classification Models Cheatsheet

## Quick Reference

### Model Selection
| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| Logistic Regression | Binary, interpretable | Fast, probabilities | Linear only |
| Decision Tree | Interpretable | No scaling | Overfits |
| Random Forest | General purpose | Robust | Slow inference |
| XGBoost/LightGBM | Competitions | Best accuracy | Complex tuning |
| SVM | Small datasets | Works well | Slow on large data |
| KNN | Simple problems | No training | Slow inference |

### Key Code

```python
# Standard workflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)
print(classification_report(y_test, model.predict(X_test_scaled)))
```

### Metrics
- **Accuracy**: Overall correctness (balanced classes only)
- **Precision**: TP / (TP + FP) - minimize false positives
- **Recall**: TP / (TP + FN) - minimize false negatives
- **F1**: Harmonic mean of precision & recall

### Common Pitfalls
1. Not scaling features for SVM/KNN
2. Using accuracy on imbalanced data
3. Data leakage from preprocessing before split
4. Overfitting with too many tree depth

### Hyperparameters to Tune
- **Random Forest**: n_estimators, max_depth, min_samples_split
- **XGBoost**: learning_rate, max_depth, n_estimators
- **SVM**: C, gamma, kernel
