# 📋 Feature Engineering Cheatsheet

## Quick Reference

### Numeric Features
```python
# Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
StandardScaler()  # Mean=0, Std=1
MinMaxScaler()    # Range [0, 1]

# Transformations
np.log1p(df['skewed_column'])  # Log for right-skewed
np.sqrt(df['column'])          # Square root
```

### Categorical Features
```python
# One-Hot Encoding
pd.get_dummies(df['category'], prefix='cat')

# Label Encoding (ordinal)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['encoded'] = le.fit_transform(df['category'])

# Target Encoding (high cardinality)
mean_target = df.groupby('category')['target'].mean()
df['cat_encoded'] = df['category'].map(mean_target)
```

### Datetime Features
```python
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter
df['days_since'] = (pd.Timestamp.now() - df['date']).dt.days
```

### Text Features
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000)
text_features = tfidf.fit_transform(df['text'])
```

### Missing Values
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')  # or 'mean', 'most_frequent'
df_imputed = imputer.fit_transform(df)

# Create missing indicator
df['col_missing'] = df['col'].isna().astype(int)
```

### Feature Interactions
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X)
```

### Feature Selection
```python
# Correlation filtering
corr = df.corr().abs()
high_corr = corr[corr > 0.9]

# Feature importance
importances = model.feature_importances_
```

### Common Pitfalls
1. Fit scaler on training data only
2. Don't use target info before split
3. Handle missing before encoding
4. Watch for multicollinearity
