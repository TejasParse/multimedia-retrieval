import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from sklearn.manifold import MDS
import pandas as pd
import random
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
file_path = file_choice[model_input]  # Replace with your CSV file path
n_features = feature_count[model_input]   # Adjust this to the number of features 
data = pd.read_csv(file_path)

# Filter Data: Keep only rows where "category" is "target_videos" and "label" is not empty/NaN
filtered_data = data[(data['category'] == "target_videos") & (data['labels'].notna())]


# Extract Features and Labels
features = filtered_data.iloc[:, -n_features:].values
labels = filtered_data['labels'].values
videoIds = filtered_data['videoId'].values

grouped_features = {}
unique_labels = np.unique(labels)
for label in unique_labels:
    grouped_features[label] = {
        'features': features[labels == label],
        'videoIds': videoIds[labels == label]
    }

def generate_adjacency_matrix(features, percentile=50):

    n = features.shape[0]

    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(features[i] - features[j])
    
    distance_array = distance_matrix[np.triu_indices(n, k=1)]
    
    threshold_distance = np.percentile(distance_array, percentile)
    
    adjacency_matrix = (distance_matrix < threshold_distance).astype(int)
    
    return adjacency_matrix

def visualize_clusters(features, cluster_labels, label):

    mds = MDS(n_components=2)
    features_2d = mds.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels, cmap='viridis', s=50)
    plt.colorbar(scatter, label="Cluster Label")
    plt.title(f"2D Visualization of Clusters for {label}")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.grid(True)
    plt.show()

import os
import pandas as pd

def scan_video_files(root_dir):

    video_paths = {}
    
    for subdir in ['target_videos', 'non_target_videos']:
        subdir_path = os.path.join(root_dir, subdir)
        
        for label_folder in os.listdir(subdir_path):
            label_folder_path = os.path.join(subdir_path, label_folder)
            
            if os.path.isdir(label_folder_path):
                for video_file in os.listdir(label_folder_path):
                    video_path = os.path.join(label_folder_path, video_file)
                    video_paths[video_file] = video_path
    
    return video_paths

def get_video_filename_mapping(mapping_file='E:\\Coding\\MultimediaWebDatabases\\Phase 3\\VideoID_Mapping.csv'):

    df = pd.read_csv(mapping_file)
    video_id_to_filename = dict(zip(df['VideoID'], df['Filename']))
    return video_id_to_filename

def map_video_id_to_video_path(root_dir, video_id_to_filename):

    video_paths = scan_video_files(root_dir)
    
    # Map VideoID to videoPath
    video_id_to_path = {}
    for video_id, filename in video_id_to_filename.items():
        if filename in video_paths:
            video_id_to_path[video_id] = video_paths[filename]
    
    return video_id_to_path

# TODO3: Update the location to dataset
root_dir = 'E:\\Coding\\MultimediaWebDatabases\\Phase 3\\Assets\\hmdb51_org'
video_paths = scan_video_files(root_dir)
# TODO4: Update the location to VideoID_Mapping.csv file
video_id_to_filename = get_video_filename_mapping('E:\\Coding\\MultimediaWebDatabases\\Phase 3\\VideoID_Mapping.csv')  

# Used to Map Video ID to Video Path
video_id_to_video_path = map_video_id_to_video_path(root_dir, video_id_to_filename)

import cv2
import numpy as np

def get_middle_frame(video_path):

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    middle_frame_index = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    
    ret, frame = cap.read()
    
    cap.release()
    
    if not ret:
        print("Error: Could not read the middle frame")
        return None
    
    return frame

def get_frames_from_video_ids(video_id_array, video_id_to_video_path):

    frames = []
    
    for video_id in video_id_array:
        video_path = video_id_to_video_path.get(video_id, None)
        
        if video_path is None:
            print(f"Error: Video path not found for VideoID {video_id}")
            continue
        
        frame = get_middle_frame(video_path)
        
        if frame is not None:
            frames.append(frame)
    
    return frames

import matplotlib.pyplot as plt
import numpy as np

def plot_images_in_grid(images, cluster_labels, n_clusters):

    # print(len(images), "Count of Images")
    # print(images[0].shape, "Shape of Image")
    # print(cluster_labels, "Clusters")

    for cluster_id in range(n_clusters):
        # cluster_images = images[cluster_labels == cluster_id]
        cluster_images = [image for image, label in zip(images, cluster_labels) if label == cluster_id]
        if(len(cluster_images)!=0):
            fig, axes = plt.subplots(1, len(cluster_images), figsize=(12, 4))

            # If there is only one image, axes will not be an array, so make it iterable
            if len(cluster_images) == 1:
                axes = [axes]

            fig.suptitle(f'Cluster {cluster_id}')
            
            for idx, image in enumerate(cluster_images):
                axes[idx].imshow(image, cmap='gray') 
                axes[idx].axis('off')
            
            plt.show()


def spectral_clustering_scratch(adj_matrix, num_clusters):

    # Step 1: Compute the Degree Matrix (D)
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))  # Degree matrix (diagonal)
    
    # Step 2: Compute the Laplacian Matrix (L)
    laplacian_matrix = degree_matrix - adj_matrix  # Unnormalized Laplacian matrix

    # Step 3: Compute the Eigenvalues and Eigenvectors of the Laplacian matrix
    # Use eigsh for large sparse matrices to compute the smallest eigenvalues
    eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=num_clusters, which='SM')  # Smallest eigenvalues
    # print(eigenvectors.shape, "Vectors", adj_matrix.shape)
    # Step 4: Partition the graph using the eigenvectors
    # Use the eigenvectors as a new feature space. For two clusters, we can use the Fiedler vector (second smallest eigenvector).
    # For more clusters, we partition based on multiple eigenvectors.
    
    # Select the eigenvectors corresponding to the smallest eigenvalues
    selected_eigenvectors = eigenvectors[:, 1:num_clusters]  # Exclude the first eigenvector (corresponding to zero eigenvalue)
    # print(selected_eigenvectors.shape, "Selected Eigenvectors")
    # Step 5: Partition nodes based on the eigenvectors' values (e.g., sign of eigenvector entries)
    cluster_labels = np.zeros(adj_matrix.shape[0], dtype=int)  # Initialize cluster labels
    
    for i in range(num_clusters-1):
        # The values of the eigenvector determine the cluster assignment
        cluster_labels[selected_eigenvectors[:, i] > 0] = i  # Assign based on the sign of the eigenvector's values
    
    return cluster_labels

import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

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

def spectral_clustering_with_kmeans(adj_matrix, num_clusters):
    # Step 1: Compute the Degree Matrix (D)
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))  # Degree matrix (diagonal)
    
    # Step 2: Compute the Laplacian Matrix (L)
    laplacian_matrix = degree_matrix - adj_matrix  # Unnormalized Laplacian matrix

    # Step 3: Compute the Eigenvalues and Eigenvectors of the Laplacian matrix
    # We use 'eigsh' to find the smallest eigenvalues/eigenvectors for large sparse matrices
    eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=num_clusters, which='SM')  # Smallest eigenvalues

    # Step 4: Select the eigenvectors corresponding to the smallest eigenvalues
    # Skip the first eigenvector (which corresponds to the zero eigenvalue)
    selected_eigenvectors = eigenvectors[:, 1:num_clusters]  # Select eigenvectors starting from the second smallest eigenvalue

    # Step 5: Use KMeans to cluster the nodes based on the selected eigenvectors

    # kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    # cluster_labels = kmeans.fit_predict(selected_eigenvectors)  # Clustering based on eigenvectors

    cluster_labels, centroids = k_means_clustering(selected_eigenvectors, num_clusters, 100, metric = "euclidean")
    # print(cluster_labels)

    return cluster_labels


cluster_count = int(input("Enter number of clusters: "))

def allLabels(grouped_features):
    for label, data1 in grouped_features.items():

        feature_array = data1['features']
        video_id_array = data1['videoIds']
        # print(data)
        print(f"Label: {label}")
        # print(f"Feature Array Shape: {feature_array.shape}")
        
        adj_matrix = generate_adjacency_matrix(feature_array, percentile=1)
        adj_matrix = adj_matrix.astype(np.float64)
        # cluster_labels = spectral_clustering_2_clusters(adj_matrix, num_clusters=cluster_count)  # Adjust n_clusters as needed
        cluster_labels = spectral_clustering_with_kmeans(adj_matrix, num_clusters=cluster_count)  # Adjust n_clusters as needed
        # cluster_labels = kmeans_clustering(feature_array, n_clusters=cluster_count)  # Adjust n_clusters as needed
        
        # print(f"Adjacency Matrix:\n{adj_matrix}")
        # print(f"Cluster Labels: {cluster_labels}")
        # print(video_id_array)
        
        # Visualize clusters in 2D MDS space
        visualize_clusters(feature_array, cluster_labels, label)
        
        # # Generate video thumbnails for each cluster
        frames = get_frames_from_video_ids(video_id_array, video_id_to_video_path)
        # print(frames[0].shape)
        # print(len(frames))
        plot_images_in_grid(frames, cluster_labels, n_clusters=cluster_count)

        
        print("-" * 40)

def singleLabel(label, grouped_features):

    data1 = grouped_features[label]

    feature_array = data1['features']
    video_id_array = data1['videoIds']
    # print(data)
    print(f"Label: {label}")
    # print(f"Feature Array Shape: {feature_array.shape}")
    
    adj_matrix = generate_adjacency_matrix(feature_array, percentile=1)
    adj_matrix = adj_matrix.astype(np.float64)
    # cluster_labels = spectral_clustering_2_clusters(adj_matrix, num_clusters=cluster_count)  # Adjust n_clusters as needed
    cluster_labels = spectral_clustering_with_kmeans(adj_matrix, num_clusters=cluster_count)  # Adjust n_clusters as needed
    # cluster_labels = spectral_clustering_scratch(adj_matrix, num_clusters=cluster_count+1)  # Adjust n_clusters as needed
    # cluster_labels = kmeans_clustering(feature_array, n_clusters=cluster_count)  # Adjust n_clusters as needed
    
    # print(f"Adjacency Matrix:\n{adj_matrix}")
    # print(f"Cluster Labels: {cluster_labels}")
    # print(video_id_array)
    
    # Visualize clusters in 2D MDS space
    visualize_clusters(feature_array, cluster_labels, label)
    
    # # Generate video thumbnails for each cluster
    frames = get_frames_from_video_ids(video_id_array, video_id_to_video_path)
    # print(frames[0].shape)
    # print(len(frames))
    plot_images_in_grid(frames, cluster_labels, n_clusters=cluster_count)
    
    print("-" * 40)

print("\nPick One")
print("1: For all labels")
print("2: For single label")

input_dtype = int(input("Enter choice: "))

if input_dtype == 1:
    allLabels(grouped_features)
elif input_dtype == 2:
    label_input =  input("Enter the target videos label: ")
    singleLabel(label_input, grouped_features)