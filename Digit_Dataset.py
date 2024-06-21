import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
data = load_digits()
X = data.data
y = data.target
# Applying PCA to reduce dimensionality to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='green', marker='o', s=50, alpha=0.8)
plt.title('Before Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
#KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)
# Calculate Adjusted Rand Index to measure clustering accuracy
ari = adjusted_rand_score(y, clusters)
print(f"Adjusted Rand Index: {ari}")
# after clustering
plt.subplot(1, 2, 2)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X',
label='Centroids')
plt.title('K-means Clustering on Digits Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.legend()
plt.tight_layout()
plt.show()