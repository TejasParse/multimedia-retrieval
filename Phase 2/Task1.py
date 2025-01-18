import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import csv
import warnings
warnings.filterwarnings("ignore")

# TODO1: Change the feature file locations
feature_space_map = {
    'R3D18-Layer3-512': 'E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\layer3_data.csv',
    'R3D18-Layer4-512': 'E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\layer4_data.csv',
    'R3D18-AvgPool-512': 'E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\avgpool_data.csv',
    'BOF-HOF-480': 'E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\hof_data.csv',
    'BOF-HOG-480': 'E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\hog_data.csv',
    'col - hist': "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\col_hist_data.csv"
}

def load_csv_data(feature_space):
    """
    Load the CSV file corresponding to the feature space.
    """
    csv_file = feature_space_map.get(feature_space)
    if csv_file:
        return pd.read_csv(csv_file)
    else:
        print(f"Feature space {feature_space} not found.")
        return None

def get_video_features(df, videoID):
    """
    Extract the feature vector for a given videoID from the dataframe.
    """
    row = df[df['videoId'] == videoID]
    # print(row)
    if row.empty:
        print(f"VideoID {videoID} not found in the dataset.")
        return None, None
    features = row.iloc[0, 4:].values  # Get the feature columns
    label = row['label'].values[0]  # Get the label
    return features, label

def find_similarities(query_videoID, feature_space):
    df = load_csv_data(feature_space)
    if df is None:
        return
    
    query_features, query_label = get_video_features(df, query_videoID)
    if query_features is None:
        return
    
    # Filter for even-numbered target videos
    target_videos = df[(df['category'] == 'target_videos') & (df['videoId'] % 2 == 0)]
    # print(target_videos)
    similarities_with_labels = []
    
    # Compare the query video with each even-numbered target video
    for _, row in target_videos.iterrows():
        target_features = row.iloc[4:].values
        similarity = cosine_similarity(query_features.reshape(1, -1), target_features.reshape(1, -1))[0][0]
        similarities_with_labels.append((row['label'], similarity))
    
    # Sort similarities in descending order
    similarities_with_labels.sort(key=lambda x: x[1], reverse=True)
    return similarities_with_labels
    # Get top l results
def group_similarities(similarities_with_labels):
    # Dictionary to store total similarity and count for each label
    label_stats = defaultdict(lambda: {'similarity_sum': 0.0, 'count': 0})

    # Iterate over the similarities_with_labels list
    for label, similarity in similarities_with_labels:
        label_stats[label]['similarity_sum'] += similarity
        label_stats[label]['count'] += 1

    # Convert the dictionary into a list and calculate average similarity for each label
    groupedList = [(label, stats['similarity_sum'] / stats['count'], stats['count']) for label, stats in label_stats.items()]

    # Sort the groupedList by average similarity in descending order
    sorted_groupedList = sorted(groupedList, key=lambda x: x[1], reverse=True)

    return sorted_groupedList

def findIdFromName(videoName):
    # TODO2: Change the VideoId Mapping CSV File location
    file = 'E:\\Coding\\MultimediaWebDatabases\\Phase 2\\VideoID_Mapping.csv'
    with open(file,'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Filename'] == videoName:
                return int(row['VideoID'])

def menu():
    # Display menu options
    print("\nSelect a Feature Model:")
    print("1. R3D18-Layer3-512")
    print("2. R3D18-Layer4-512")
    print("3. R3D18-AvgPool-512")
    print("4. col - hist")
    print("5. BOF-HOF-480")
    print("6. BOF-HOG-480")
    print("7. Exit")

    choice = input("Enter a number (1-6): ")

    if choice == '1':
        print("You selected: R3D18-Layer3-512")
        selected_model = "R3D18-Layer3-512"

    elif choice == '2':
        print("You selected: R3D18-Layer4-512")
        selected_model = "R3D18-Layer4-512"

    elif choice == '3':
        print("You selected: R3D18-AvgPool-512")
        selected_model = "R3D18-AvgPool-512"

    elif choice == '4':
        print("You selected: col - hist")
        selected_model = "col - hist"

    elif choice == '5':
        print("You selected: BOF-HOF-480")
        selected_model = "BOF-HOF-480"
    elif choice == '6':
        print("You selected: BOF-HOG-480")
        selected_model = "BOF-HOG-480"
    else:
        print("Invalid choice. Please select a number between 1 and 6.")
    return selected_model

query = input("Enter the video ID or Video name : ")

featureSpace = menu()
l = input("Enter l : ")
if '.avi' in query:
    query = findIdFromName(query)
query = int(query)

similarities = find_similarities(query, featureSpace)
groupedList = group_similarities(similarities)

for label,score,_ in groupedList[:int(l)]:
    print(f"label : {label} |  Score : {score}")