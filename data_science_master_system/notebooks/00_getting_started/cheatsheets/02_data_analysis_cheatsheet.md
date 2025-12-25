# 📋 Data Analysis Cheatsheet (Notebooks 02-03)

## 📌 Pandas Quick Reference

### Loading Data
```python
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
df = pd.read_json('file.json')
df = pd.read_sql(query, connection)
```

### Viewing Data
```python
df.head(10)        # First 10 rows
df.tail(5)         # Last 5 rows
df.sample(5)       # Random 5 rows
df.shape           # (rows, cols)
df.info()          # Types, nulls
df.describe()      # Statistics
```

### Selection
```python
df['col']                    # Single column
df[['col1', 'col2']]         # Multiple columns
df.loc[0]                    # Row by label
df.iloc[0]                   # Row by position
df.loc[df['col'] > 5]        # Filter rows
df.query('col > 5')          # Query syntax
```

### Cleaning
```python
df.dropna()                  # Drop null rows
df.fillna(0)                 # Fill nulls with 0
df.fillna(df.mean())         # Fill with mean
df.drop_duplicates()         # Remove duplicates
df.rename(columns={'old': 'new'})
```

### Aggregation
```python
df.groupby('cat')['val'].mean()
df.groupby(['cat1', 'cat2']).agg({'val': ['mean', 'sum']})
df.pivot_table(values='val', index='cat', aggfunc='mean')
```

## 📊 Visualization Quick Reference

### Matplotlib
```python
plt.figure(figsize=(10, 6))
plt.plot(x, y)           # Line
plt.scatter(x, y)        # Scatter
plt.bar(x, y)            # Bar
plt.hist(x, bins=20)     # Histogram
plt.savefig('plot.png')
```

### Seaborn
```python
sns.histplot(df, x='col', hue='cat')
sns.boxplot(df, x='cat', y='val')
sns.heatmap(df.corr(), annot=True)
sns.pairplot(df, hue='target')
```

## ⚠️ Common Pitfalls
| Issue | Solution |
|-------|----------|
| SettingWithCopyWarning | Use .loc[] or .copy() |
| Memory error | Read in chunks |
| Mixed types | Specify dtype |

## 🚀 Performance Tips
- Use `category` dtype for low-cardinality strings
- Use `pd.eval()` for complex expressions
- Use `swifter` or `modin` for parallelism
