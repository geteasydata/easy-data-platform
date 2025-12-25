# 📋 Getting Started Cheatsheet

## 📌 Core Python + Data Science

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install packages
pip install numpy pandas scikit-learn matplotlib jupyter
```

### NumPy Basics
```python
import numpy as np

arr = np.array([1, 2, 3])
arr.shape, arr.dtype
arr.mean(), arr.std(), arr.sum()
np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)  # 5 points from 0 to 1
```

### Pandas Basics
```python
import pandas as pd

df = pd.read_csv('data.csv')
df.head()      # First 5 rows
df.info()      # Column types, nulls
df.describe()  # Statistics
df['col'].value_counts()  # Frequency

# Filtering
df[df['age'] > 30]
df.query('age > 30 and city == "NYC"')

# Grouping
df.groupby('category')['value'].mean()
```

### Matplotlib Basics
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Line')
plt.scatter(x, y, label='Points')
plt.bar(categories, values)
plt.xlabel('X'), plt.ylabel('Y')
plt.title('Title')
plt.legend()
plt.savefig('plot.png')
plt.show()
```

## 📊 ML Workflow
```
1. Load Data      → pd.read_csv()
2. Explore        → df.info(), df.describe()
3. Clean          → df.dropna(), df.fillna()
4. Split          → train_test_split(X, y)
5. Scale          → StandardScaler()
6. Train          → model.fit(X_train, y_train)
7. Evaluate       → model.score(), metrics
8. Save           → joblib.dump(model, 'model.pkl')
```

## ⚠️ Common Pitfalls
| Problem | Solution |
|---------|----------|
| Import error | pip install package |
| File not found | Use absolute paths |
| Memory error | Read data in chunks |
| Slow code | Use vectorized operations |

## 🚀 Quick Commands
```python
# Check GPU
import torch
print(torch.cuda.is_available())

# Memory usage
df.memory_usage(deep=True).sum() / 1e6  # MB

# Progress bar
from tqdm import tqdm
for i in tqdm(range(1000)):
    pass
```
