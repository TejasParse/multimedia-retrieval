import numpy as np
import pandas as pd
import cv2
import os

def video_visualisation(video_id, video_folder_path,mapping_csv_file_path, rank=None,  frame_size=(800, 600)):
    mapping_data = pd.read_csv(mapping_csv_file_path, low_memory=False)
    # Find the video name using the mapping file
    mapping_row = mapping_data[mapping_data['VideoID'] == int(video_id)]
    if mapping_row.empty:
        print(f"Video ID '{video_id}' not found in the mapping file.")
        return None
    video_name = mapping_row.iloc[0]['Filename']
    video_path = None
    # Loop through the folder and subfolders to find the video file
    for root, dirs, files in os.walk(video_folder_path):
        for file in files:
            if video_name in file:
                video_path = os.path.join(root, file)
                break

    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Can't open video.")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_index = frame_count // 2

        # Set the video to the middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, frame_size)
            # Overlay the video name on the frame
            text = f"Video: {video_name}"
            if rank is not None:
                text += f" | Rank: {rank}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (255, 255, 255)  
            thickness = 2
            position = (10, 30)  
            cv2.putText(frame, text, position, font, font_scale, font_color, thickness)

            # Display the frame as a thumbnail
            cv2.imshow('Thumbnail', frame)
            cv2.waitKey(5000)  # Wait 
            cv2.destroyAllWindows()

        cap.release()
    else:
        print(f"Error: Video '{video_name}' not found in the specified folder.")


# Initialize LSH structure with random planes and hash tables
def initialize_lsh(L, h, vector_dim):
    hash_tables = [{} for _ in range(L)]
    random_planes = [
        [np.random.randn(vector_dim) for _ in range(h)]
        for _ in range(L)
    ]
    return hash_tables, random_planes

# Hash a vector for a given layer
def hash_vector(vector, layer, random_planes):
    return tuple(int(np.dot(vector, plane) > 0) for plane in random_planes[layer])

def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0  # Handle cases where vectors are all zeros
    return dot_product / (norm_a * norm_b)

# Add a video to the LSH structure
def add_video(hash_tables, random_planes, L, video_id, vector):
    for layer in range(L):
        hash_value = hash_vector(vector, layer, random_planes)
        if hash_value not in hash_tables[layer]:
            hash_tables[layer][hash_value] = []
        hash_tables[layer][hash_value].append((video_id, vector))

# Query similar videos based on the LSH structure
def query_lsh(hash_tables, random_planes, L, query_vector, query_video_id, num_results):
    candidates = {}  # To store unique candidates
    overall_candidates = 0  # Counter for all candidate vectors considered

    for layer in range(L):
        hash_value = hash_vector(query_vector, layer, random_planes)
        if hash_value in hash_tables[layer]:
            for video_id, vector in hash_tables[layer][hash_value]:
                overall_candidates += 1  # Count every candidate considered
                if video_id != query_video_id:  # Exclude the query video itself
                    candidates[video_id] = vector  # Use a dictionary to store unique candidates
    
    # Sort the unique candidates by cosine similarity
    sorted_candidates = sorted(
        candidates.items(),
        key=lambda item: cosine_similarity(query_vector, item[1]),  # Sort by cosine similarity
        reverse=True  # Higher similarity should come first
    )
    
    # Retrieve the top `num_results`
    return sorted_candidates[:num_results], len(candidates), overall_candidates

# Load video features from a CSV file
def load_features_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path,low_memory=False)
    # Ensure all numeric columns are processed properly
    def convert_to_float(value):
        if isinstance(value, str) and 'j' in value:  
            value = complex(value.replace(" ", "").replace("\n", ""))
            return value.real  # Use only the real part
        try:
            return float(value)
        except ValueError:
            return 0.0  
    # Process all features from column 3 onwards
    video_features = {
        row['videoId']: [convert_to_float(val) for val in row.iloc[3:].values]
        for _, row in df.iterrows()
    }
    return video_features

# Search for similar videos given a query video ID
def search_similar_videos(video_features, hash_tables, random_planes, L,video_folder_path, csv_file_path, mapping_csv_file_path, query_video_id, num_results):
    query_vector = video_features.get(query_video_id)
    if query_vector is None:
        print("Video not found.")
        return []
    results, unique_candidates, overall_candidates = query_lsh(hash_tables, random_planes, L, query_vector, query_video_id, num_results)

    # Print statistics
    print(f"Number of unique candidates considered: {unique_candidates}")
    print(f"Number of overall candidates considered: {overall_candidates}")

    # Running the query video
    video_visualisation(query_video_id,video_folder_path,mapping_csv_file_path)
    # Running the similar ones
    print(f"Top {num_results} similar videos for Video ID {query_video_id}:")
    for rank, (video_id, vector) in enumerate(results, start=1):
        similarity = cosine_similarity(query_vector, vector)
        distance = 1 - similarity
        print(f"Rank {rank}: Video ID {video_id}, Distance: {distance}")
        video_visualisation(video_id,video_folder_path,mapping_csv_file_path, rank=rank)
    return results

# TODO1: Update the location to latent models
file_path = 'E:\\Coding\\MultimediaWebDatabases\\Phase 3\\Dataset\\'  #Path_change
# TODO2: Update the location to dataset
video_folder_path="E:\\Coding\\MultimediaWebDatabases\\Phase 3\\Assets\\hmdb51_org"                              #Path_change
# TODO3: Update the location to VideoID_Mapping.csv file
mapping_csv_file_path="E:\\Coding\\MultimediaWebDatabases\\Phase 3\\VideoID_Mapping.csv"    #Path_change

Model_dict={1: "avgpool_kmeans(300).csv", 2 :"col_hist_svd(300).csv",3:"layer4_pca(300).csv"}
Model=int(input("Select model number from the below options \n 1. Average_pool + k_means + s=300 \n 2. Col_hist + svd + s=300 \n 3. Layer4 + pca + s=300 \n  "))
csv_file_path= file_path+Model_dict[Model]
video_features = load_features_from_csv(csv_file_path)

# Initialize LSH parameters
dimension = len(next(iter(video_features.values())))  
L = int(input("Please enter # of layers :  "))  #num_layers
h = int(input("Please enter # of hashes per layer :  "))  #num_hashes_per_layer
v= int(input("Enter the VideoID :  "))
t=int(input("Enter the number of similar videos need to retrieve :  "))
# Initialize LSH structure
hash_tables, random_planes = initialize_lsh(L, h, dimension)

for video_id, vector in video_features.items():
    add_video(hash_tables, random_planes, L, video_id, vector)

# Search for similar videos
search_similar_videos(video_features, hash_tables, random_planes, L, video_folder_path, csv_file_path, mapping_csv_file_path,query_video_id=v, num_results=t)