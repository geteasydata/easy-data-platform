# 📋 Hyperparameter Tuning Cheatsheet (Notebook 09)

## 📌 Methods

| Method | Pros | Cons |
|--------|------|------|
| Grid Search | Exhaustive | Slow |
| Random Search | Fast | May miss optimal |
| Bayesian (Optuna) | Intelligent | Complex setup |
| Hyperband | Efficient | For neural nets |

## 🛠️ Essential Code

### Grid Search
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7]
}

grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print(grid.best_params_)
```

### Optuna
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    }
    model = XGBClassifier(**params)
    return cross_val_score(model, X, y, cv=3).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

## 📐 Common Hyperparameters

### Tree Models (XGBoost, RF)
| Param | Range | Impact |
|-------|-------|--------|
| n_estimators | 100-1000 | More = slower |
| max_depth | 3-10 | Higher = overfit |
| learning_rate | 0.01-0.3 | Lower = more trees |
| min_child_weight | 1-10 | Higher = conservative |

### Neural Networks
| Param | Range | Impact |
|-------|-------|--------|
| learning_rate | 1e-5 to 1e-2 | Critical |
| batch_size | 16-128 | Memory/speed |
| dropout | 0.1-0.5 | Regularization |
| hidden_units | 32-512 | Capacity |

## ⚠️ Tips
- Start with random search
- Use log scale for learning rate
- Early stopping to save time
- Nested CV for unbiased estimate
