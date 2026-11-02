import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv("Iris.csv")
print(df.head())

# 2. Extract numeric features (Drop 'Id' and 'Species' columns if present)
X = df.drop(['Id', 'Species'], axis=1, errors='ignore')

# Optional Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Determine optimal number of clusters using the Elbow Method
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_) # inertia_ gives the WCSS value

# Plot the Elbow Graph
plt.figure(figsize=(8, 4))
plt.plot(k_range, wcss, marker='o', linestyle='--', color='b')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.xticks(k_range)
plt.grid(True)
plt.show()

kmeans_optimal = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans_optimal.fit_predict(X_scaled)

df['Cluster'] = y_kmeans
print("\nCluster distribution:\n", df['Cluster'].value_counts())

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[y_kmeans == 0, 0], X_scaled[y_kmeans == 0, 1], s=70, c='red', label='Cluster 1')
plt.scatter(X_scaled[y_kmeans == 1, 0], X_scaled[y_kmeans == 1, 1], s=70, c='blue', label='Cluster 2')
plt.scatter(X_scaled[y_kmeans == 2, 0], X_scaled[y_kmeans == 2, 1], s=70, c='green', label='Cluster 3')

plt.scatter(
    kmeans_optimal.cluster_centers_[:, 0], 
    kmeans_optimal.cluster_centers_[:, 1], 
    s=200, c='yellow', edgecolor='black', marker='*', label='Centroids'
)
plt.title('K-Means Clusters on Iris Dataset (K=3)')
plt.xlabel('Sepal Length (Scaled)')
plt.ylabel('Sepal Width (Scaled)')
plt.legend()
plt.grid(True)
plt.show()