# 📋 Model Evaluation Cheatsheet (Notebook 10)

## 📌 Classification Metrics

| Metric | Formula | Use When |
|--------|---------|----------|
| Accuracy | (TP+TN)/(All) | Balanced classes |
| Precision | TP/(TP+FP) | False positives costly |
| Recall | TP/(TP+FN) | False negatives costly |
| F1 | 2×(P×R)/(P+R) | Imbalanced classes |
| AUC-ROC | Area under curve | Compare models |

## 🛠️ Essential Code

### Classification Report
```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)
```

### ROC Curve
```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

## 📐 Regression Metrics
| Metric | Best Value | Interpretation |
|--------|------------|----------------|
| MSE | 0 | Average squared error |
| RMSE | 0 | Same units as target |
| MAE | 0 | Average absolute error |
| R² | 1 | Explained variance |

## ⚠️ Common Mistakes
- Using accuracy for imbalanced data
- Not using stratified CV
- Data leakage in preprocessing
- Overfitting to validation set

## 🚀 Best Practices
- Always report confidence intervals
- Use appropriate metric for business goal
- Plot learning curves to detect overfitting
- Bootstrap for small datasets
