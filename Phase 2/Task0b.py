import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import cv2
import os
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Updated function to load features from CSV instead of folder
def load_features_from_csv(csv_file_path):
    data = pd.read_csv(csv_file_path)
    # Extract video names and feature vectors into a dictionary
    features = {}
    even_target_videos = data[(data['videoId'].astype(int) % 2 == 0) & (data['category'] == 'target_videos')]
    # print(even_target_videos)
    for index, row in even_target_videos.iterrows():
        video_name = row['videoName']
        features[video_name] = row[4:].values.astype(float)  # Extract feature vectors starting from column 5
    return features

# Function to extract query video features from CSV
def get_query_video_features(csv_file_path, query_video_name):
    data = pd.read_csv(csv_file_path)
    query_row = data[data['videoName'] == query_video_name]
    if query_row.empty:
        print(f"Query video '{query_video_name}' not found.")
        return None
    query_features = query_row.iloc[0, 4:].values.astype(float)
    return query_features

def video_visualisation(video_name, video_folder_path):
    video_path = None
    # Loop through the folder and subfolders to find the video file
    for root, dirs, files in os.walk(video_folder_path):
        for file in files:
            if video_name in file:
                video_path = os.path.join(root, file)
                print(f"Found video: {video_path}")
                break

    if video_path:
        # Open the video file and display frames using OpenCV
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened() == False:
            print("Error: Can't open video")
            return
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Video', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"Error: Video '{video_name}' not found in the specified folder.")

# Function to compute similarity or distance scores
def compute_distance(query_features, all_features, metric='euclidean'):
    scores = {}
    for video_name, features in all_features.items():
        if metric == 'cosine':
            similarity = cosine_similarity(query_features.reshape(1, -1), features.reshape(1, -1)).mean()
            score = 1 - similarity  # Convert similarity to distance
        elif metric == 'euclidean':
            distances = np.array([euclidean(query_features, feature) for feature in features.reshape(1, -1)])
            score = distances.mean()
        else:
            raise ValueError("Unsupported metric. Please use 'cosine' or 'euclidean' metrics.")
        scores[video_name] = score
    return scores

# Function to visualize the most similar videos
def visualize_similar_videos(similar_videos):
    # Sorting based on score or distance
    sorted_videos = sorted(similar_videos.items(), key=lambda item: item[1])
    video_names = [item[0] for item in sorted_videos]
    scores = [item[1] for item in sorted_videos]
    print(f"Video Names: {video_names}")
    print(f"Scores: {scores}")
    
    plt.figure(figsize=(10, 6))
    plt.barh(video_names[:], scores, color='skyblue')
    plt.xlabel('Distance')
    plt.title('Top Similar Videos')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Function to find video name based on video ID
def get_video_name_from_id(csv_file_path, video_id):
    data = pd.read_csv(csv_file_path)
    query_row = data[data['videoId'] == int(video_id)]
    if query_row.empty:
        print(f"Video ID '{video_id}' not found.")
        return None
    return query_row.iloc[0]['videoName']

# Main function to find and visualize the most similar videos using CSV data
def m_similar_videos_csv(query_video, m, csv_file_path, video_folder_path, metric='euclidean', query_by_name=True):
    # Load all video features from the CSV
    all_features = load_features_from_csv(csv_file_path)
    
    # Extract query video features based on name or ID
    if query_by_name:
        query_features = get_query_video_features(csv_file_path, query_video)
    else:
        video_name = get_video_name_from_id(csv_file_path, query_video)
        if video_name is None:
            return
        query_features = get_query_video_features(csv_file_path, video_name)
    
    if query_features is None:
        return
    
    # print(query_features, "Query Features")
    
    # Compute similarity scores
    similarity_scores = compute_distance(query_features, all_features, metric)

    # print(similarity_scores, "Similarity Scores")
    
    # Get top m similar videos
    top_m_videos = dict(sorted(similarity_scores.items(), key=lambda item: item[1])[:m])
    
    # print(top_m_videos, "Top K Videos")
    # Visualize the top m similar videos
    visualize_similar_videos(top_m_videos)

    print(f"\nTop {m} similar videos for query video or query video_id: {query_video}")
    for sample, score in top_m_videos.items():
        print(f"\nVideo: {sample}, Distance: {score}")
        # Visualize each of the matched videos
        video_visualisation(sample, video_folder_path)

# Example usage with the provided CSV file
model = input("Enter the name of the model from ['avgpool','col_hist','layer3','layer4','hof','hog'] : ").strip() + "_data"

# Ask user whether to input video by name or by ID
query_choice = input("Do you want to query by Video Name (n) or Video ID (i)? Enter 'n' or 'i': ").strip().lower()

if query_choice == 'n':
    query_by_name = True
    query_video = input("Enter the video name: ").strip()
elif query_choice == 'i':
    query_by_name = False
    query_video = input("Enter the video ID: ").strip()
else:
    print("Invalid choice. Please run the program again.")
   

# Ask for the number of similar videos
m = int(input("Enter the number of similar videos to retrieve: "))
# query_video_name = "MAF_Tenshin_Ryu_draw_sword_f_cm_np1_fr_med_3.avi"  # Example query video name from the CSV
# TODO1: Update the Task0 output file location. The folder should contain the following 6 files with strictly same names
# avgpool_data.csv
# col_hist_data.csv
# hof_data.csv
# hog_data.csv
# layer3_data.csv
# layer4_data.csv

csv_file_path = f'E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\{model}.csv'  # Replace with your actual CSV file path
# m_similar_videos_csv(query_video_name, m, csv_file_path, metric='euclidean')

# TODO2: Update the location to the videos database. 
video_folder_path="E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Assets\\hmdb51_org"
m_similar_videos_csv(query_video, m, csv_file_path, video_folder_path, metric='euclidean', query_by_name=query_by_name)
