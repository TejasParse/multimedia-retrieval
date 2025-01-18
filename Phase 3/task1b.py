import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

# TODO1: Update the feature set locations
file_choice = {
    0: "E:\\Coding\\MultimediaWebDatabases\\Phase 3\\Dataset\\avgpool_kmeans(300).csv",
    1: "E:\\Coding\\MultimediaWebDatabases\\Phase 3\\Dataset\\col_hist_svd(300).csv",
    2: "E:\\Coding\\MultimediaWebDatabases\\Phase 3\\Dataset\\layer4_pca(300).csv"
}

# TODO2: Update the s values of the features selected
feature_count = {
    0: 300,
    1: 300,
    2: 300,
}

print("0: Avgpool + KMeans + 300")
print("1: Col Hist + SVD + 300")
print("2: Layer4 + PCA + 300")

model_input = int(input("Enter your choice: "))
if(model_input < 0 or model_input > 2):
    print("Invalid Choice")
    exit()

# Load CSV Data
file_path = file_choice[model_input]
data = pd.read_csv(file_path)

# Filter Data: Keep only rows where "category" is "target_videos" and "label" is not empty/NaN
filtered_data = data[(data['category'] == "target_videos") & (data['labels'].notna())]

n_features = feature_count[model_input]  # Adjust this to the number of features

# Extract Features and Labels
features = filtered_data.iloc[:, -n_features:].values
labels = filtered_data['labels'].values

# Parameters
c_clusters = int(input("\nEnter number of clusters: "))
if(c_clusters<=0):
    print("Invalid Choice")
    exit()

max_iterations = 100

# Calculate Label Representatives (Mean of Features for Each Label)
label_groups = defaultdict(list)

for label, feature in zip(labels, features):
    label_groups[label].append(feature)


label_representatives = {label: np.mean(features, axis=0) for label, features in label_groups.items()}

# Convert Label Representatives to a NumPy Array
rep_labels = np.array(list(label_representatives.keys()))
rep_features = np.array(list(label_representatives.values()))

# Normalize Features
rep_features = (rep_features - np.mean(rep_features, axis=0)) / np.std(rep_features, axis=0)

# print(rep_labels)
# print(rep_features)

# print(rep_labels.shape)
# print(rep_features.shape)

# Implement K-Means Clustering
def k_means_clustering(data, c, max_iter=100, metric="euclidean",):
    # Randomly initialize centroids
    np.random.seed(21)
    centroids = data[np.random.choice(data.shape[0], c, replace=False)]
    i=0
    # print(centroids)
    for _ in range(max_iter):
        i+=1
        # print("Iteraton: ", i)
        # Assign each point to the nearest centroid
        distances = cdist(data, centroids, metric=metric)
        # distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([data[clusters == k].mean(axis=0) for k in range(c)])
        
        # Stop if centroids don't change
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids


# k_means_clustering(features, c_clusters, max_iterations)

# Run K-Means
clusters, centroids = k_means_clustering(rep_features, c_clusters, metric="cosine", max_iter=max_iterations)

# Summarize Clusters by Labels
def summarize_clusters(clusters, labels, c):
    cluster_summary = {}
    for k in range(c):
        cluster_labels = labels[clusters == k]
        label_counts = Counter(cluster_labels)
        cluster_summary[k] = label_counts.most_common(5)  # Top 5 labels per cluster
    return cluster_summary

# Print Cluster Assignments
print("Cluster Assignments for Labels:")
for cluster_id in range(c_clusters):
    cluster_labels = rep_labels[clusters == cluster_id]
    print(f"Cluster {cluster_id + 1}: {cluster_labels}")

# Visualize Clusters in MDS Space
mds = MDS(n_components=2, random_state=21)
rep_features_2d = mds.fit_transform(rep_features)

plt.figure(figsize=(10, 8))
for cluster_id in range(c_clusters):
    cluster_points = rep_features_2d[clusters == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id + 1}')
plt.title("K-Means Clustering of Label Representatives in MDS Space")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.legend()
plt.show()