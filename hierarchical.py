import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('student_performance.csv')

# Extract the features for clustering
features = df[['MathScore', 'ScienceScore', 'ReadingScore']]

# Perform hierarchical clustering using Ward's method
linked = linkage(features, method='ward')

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