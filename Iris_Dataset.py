import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
iris = load_iris()
X = iris.data
y = iris.target
print(type(X))
print(type(y))
plt.figure(figsize=(12, 6))
#Before Clustering
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='gray', marker='o')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Before Clustering')
# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_
#After Clustering
plt.subplot(1, 2, 2)
scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('After Clustering')
handles = scatter.legend_elements()[0]
plt.legend(handles=handles,labels=['Cluster 1', 'Cluster 2', 'Cluster 3'], loc='upper right')
plt.tight_layout() 
plt.show()