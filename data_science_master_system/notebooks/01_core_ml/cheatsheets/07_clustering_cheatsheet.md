# 📋 Clustering Cheatsheet (Notebook 07)

## 📌 Algorithms

| Algorithm | Complexity | Best For | Limitations |
|-----------|------------|----------|-------------|
| K-Means | O(n·k·i) | Spherical clusters | Must specify k |
| DBSCAN | O(n²) | Arbitrary shapes | Density sensitivity |
| Hierarchical | O(n³) | Dendrograms | Memory intensive |
| Gaussian Mixture | O(n·k³·i) | Soft clustering | Assumes Gaussian |

## 🛠️ Essential Code

### K-Means
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
```

### DBSCAN
```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)
# -1 indicates noise points
```

### Finding Optimal K
```python
# Elbow method
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k)
    km.fit(X)
    inertias.append(km.inertia_)

# Silhouette score
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)
```

## 📐 Metrics
| Metric | Range | Best |
|--------|-------|------|
| Silhouette | [-1, 1] | Higher |
| Davies-Bouldin | [0, ∞) | Lower |
| Calinski-Harabasz | [0, ∞) | Higher |

## ⚠️ Common Pitfalls
- Not scaling features
- Random initialization (use k-means++)
- Wrong distance metric for data type
- Ignoring noise in DBSCAN

## 🚀 Best Practices
- Always visualize with PCA/t-SNE
- Try multiple algorithms
- Scale data before clustering
- Validate with domain knowledge
