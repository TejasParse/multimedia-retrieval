# Function to visualize Video for Task 2b and 2c

import cv2
import csv
import numpy as np
import os

def visualize_video(video_file):
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Function to load hog representatives for task 2b

def load_all_cluster_representatives_hog():

    cluster_representatives = {}
    
    sigma2_values = [4, 8, 16, 32, 64, 128]
    tau2_values = [2, 4]
    
    pair_index = 0
    for sigma2 in sigma2_values:
        for tau2 in tau2_values:
            centroids_file = f'E:\Coding\MultimediaWebDatabases\HoG\pair_{sigma2}_{tau2}_HoG.csv'
            centroids = np.loadtxt(centroids_file, delimiter=',')
            cluster_representatives[(sigma2, tau2)] = centroids
            pair_index += 1
    
    return cluster_representatives

# Function to load hof representatives for task 2c

def load_all_cluster_representatives_hof():

    cluster_representatives = {}
    
    sigma2_values = [4, 8, 16, 32, 64, 128]
    tau2_values = [2, 4]
    
    pair_index = 0
    for sigma2 in sigma2_values:
        for tau2 in tau2_values:
            centroids_file = f'E:\Coding\MultimediaWebDatabases\HoF\pair_{sigma2}_{tau2}_HoF.csv'
            centroids = np.loadtxt(centroids_file, delimiter=',')
            cluster_representatives[(sigma2, tau2)] = centroids
            pair_index += 1
    
    return cluster_representatives

# Function to return STIP features as a simple array from file for task 2b and 2c

import csv
import numpy as np

def extract_features(file_name):

    data_array = []
    with open(file_name, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            data_array.append([float(x) for x in row])
    
    data_array = np.array(data_array)
    
    return data_array

# Function to give (sigma2, tau2) value to efficiently get cluster representatives because of how we are saving it

def get_sigma2_tau2_pair(row):

    sigma2 = row[4]
    tau2 = row[5]
    
    return (sigma2, tau2)

# Assigning a cluster representative based on distance

from scipy.spatial.distance import cdist

def assign_row_to_cluster(features_row, cluster_representatives, sigma2_tau2_pair):

    centroids = cluster_representatives[sigma2_tau2_pair]
    
    distances = cdist([features_row], centroids, 'euclidean')
    
    closest_cluster = np.argmin(distances)
    
    return closest_cluster

# Using hist_data.
# Array of [(sigma2, tau2), index]
# Create 12 histograms

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def create_histogram_for_pairs(hist_data):

    histograms = defaultdict(list)
    
    for (sigma2_tau2, index) in hist_data:
        histograms[sigma2_tau2].append(index)

    for sigma2_tau2, indices in histograms.items():

        histogram, _ = np.histogram(indices, bins=np.arange(41), density=False)
        
        plt.bar(range(40), histogram)
        plt.xlabel('Cluster Index')
        plt.ylabel('Number of Assignments')
        plt.title(f'Histogram for (sigma2={sigma2_tau2[0]}, tau2={sigma2_tau2[1]})')
        plt.show()

# Create 480 concatenated for each vector
def create_and_concatenate_histograms(hist_data):

    expected_pairs = [
        (4, 2), (4, 4), (8, 2), (8, 4), (16, 2), (16, 4), 
        (32, 2), (32, 4), (64, 2), (64, 4), (128, 2), (128, 4)
    ]
    
    grouped_indices = {pair: [] for pair in expected_pairs}
    
    for (sigma2_tau2, index) in hist_data:
        grouped_indices[sigma2_tau2].append(index)
    
    histograms = []
    
    for sigma2_tau2, indices in grouped_indices.items():

        histogram, _ = np.histogram(indices, bins=np.arange(41), density=False)
        histograms.append(histogram)
    
    if len(histograms) != 12:
        raise ValueError(f"Expected 12 histograms, but found {len(histograms)}.")
    
    concatenated_vector = np.hstack(histograms)
    
    return concatenated_vector

def Task2b(video_path, video_stip_path):

    # visualize_video(video_path)

    cluster_representatives = load_all_cluster_representatives_hog()

    stip_features = extract_features(video_stip_path)

    hist_data = []

    for row in stip_features:

        ind1 = assign_row_to_cluster(row[7:79], cluster_representatives, get_sigma2_tau2_pair(row))
        hist_data.append([get_sigma2_tau2_pair(row), ind1])
    
    create_histogram_for_pairs(hist_data)

    bog_hog_480 = create_and_concatenate_histograms(hist_data)
    # print("Concatenated 480-dimensional vector:", bog_hog_480)
    # print("Shape of the concatenated vector:", bog_hog_480.shape)

    return bog_hog_480

def Task2c(video_path, video_stip_path):
    
    # Example usage
    # visualize_video(video_path)

    # Example usage
    cluster_representatives = load_all_cluster_representatives_hof()

    stip_features = extract_features(video_stip_path)

    hist_data = []

    for row in stip_features:

        ind1 = assign_row_to_cluster(row[79:], cluster_representatives, get_sigma2_tau2_pair(row))
        hist_data.append([get_sigma2_tau2_pair(row), ind1])
    

    # print(hist_data)
    create_histogram_for_pairs(hist_data)

    # Concatenate the 12 histograms into a 480-dimensional vector
    bog_hof_480 = create_and_concatenate_histograms(hist_data)
    # print("Concatenated 480-dimensional vector:", bog_hof_480)
    # print("Shape of the concatenated vector:", bog_hof_480.shape)

    return bog_hof_480


given_video_name = input("Enter Video Path: ") 
# "E:\\Coding\\MultimediaWebDatabases\\Assets\\hmdb51_org\\target_videos\\drink\\CastAway2_drink_u_cm_np1_le_goo_8.avi"

stips_folder = "E:\Coding\MultimediaWebDatabases\Assets\hmdb51_org_stips_filtered"
action_subfolder = os.path.basename(os.path.dirname(given_video_name))  
video_name = os.path.basename(given_video_name)
stip_file_name = f"{video_name}.csv"
video_stip_path = os.path.join(stips_folder, action_subfolder, stip_file_name)

print("\nModel Menu\n0: HoG\n1: HoF")
given_index = input("Enter Model: ")

if(given_index == "0"):
    feature_2b = Task2b(given_video_name, video_stip_path)
    print(feature_2b)
elif(given_index == "1"):
    feature_2c = Task2c(given_video_name, video_stip_path)
    print(feature_2c)