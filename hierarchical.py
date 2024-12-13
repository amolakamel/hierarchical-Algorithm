import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('student_performance.csv')

# Extract the features for clustering
features = df[['MathScore', 'ScienceScore', 'ReadingScore']]

# Standardize the features to make them comparable
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Perform hierarchical clustering using Ward's method
linked = linkage(features_scaled, method='ward')

# Plot the dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linked,
           orientation='top',
           labels=df['StudentID'].values,
           distance_sort='descending',
           show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram - Student Performance")
plt.xlabel("Student ID")
plt.ylabel("Euclidean Distance")
plt.show()

# Let's assume the optimal number of clusters from the dendrogram is 3
optimal_k = 3

# Apply Agglomerative Clustering
agg_clust = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
df['Cluster'] = agg_clust.fit_predict(features_scaled)

# Display the clusters
print("Clustered Dataset with Agglomerative Hierarchical Clustering:")
print(df[['StudentID', 'MathScore', 'ScienceScore', 'ReadingScore', 'Cluster']])

# Optionally, visualize the clusters (2D or 3D visualization, depending on features)
plt.figure(figsize=(8, 6))
plt.scatter(df['MathScore'], df['ScienceScore'], c=df['Cluster'], cmap='viridis')
plt.title('Agglomerative Hierarchical Clustering - Student Performance')
plt.xlabel('Math Score')
plt.ylabel('Science Score')
plt.colorbar(label='Cluster')
plt.show()
