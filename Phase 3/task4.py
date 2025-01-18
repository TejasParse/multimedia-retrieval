import numpy as np
import pandas as pd
import cv2
import os
import warnings
warnings.filterwarnings("ignore")
def calculate_majority_label(labels):
    return max(set(labels), key=labels.count)

def split_dataset(features, labels, feature_index, threshold):
    left_features, left_labels, right_features, right_labels = [], [], [], []
    for i, feature in enumerate(features):
        if feature[feature_index] <= threshold:
            left_features.append(feature)
            left_labels.append(labels[i])
        else:
            right_features.append(feature)
            right_labels.append(labels[i])
    return left_features, left_labels, right_features, right_labels

def calculate_gini(labels):
    """Calculate Gini impurity for a set of labels."""
    total = len(labels)
    if total == 0:
        return 0
    counts = np.bincount(labels)  # Count occurrences of each label
    probabilities = counts / total
    gini = 1 - np.sum(probabilities ** 2)
    return gini

def find_best_split(features, labels):
    """Find the best feature and threshold to split the dataset."""
    num_features = len(features[0])
    best_feature = None
    best_threshold = None
    best_gini = float('inf')  # Initialize with a very large value

    for feature_index in range(num_features):
        # Extract all values for the current feature
        feature_values = [f[feature_index] for f in features]
        thresholds = np.unique(feature_values)  # Unique values as potential thresholds

        for threshold in thresholds:
            # Split the dataset
            left_labels = [labels[i] for i in range(len(features)) if features[i][feature_index] <= threshold]
            right_labels = [labels[i] for i in range(len(features)) if features[i][feature_index] > threshold]

            # Calculate Gini impurity for the split
            left_gini = calculate_gini(left_labels)
            right_gini = calculate_gini(right_labels)
            weighted_gini = (len(left_labels) * left_gini + len(right_labels) * right_gini) / len(labels)

            # Update the best split if this one is better
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold

def build_simple_tree(features, labels, max_depth=5):
    if len(set(labels)) == 1 or max_depth == 0:
        return calculate_majority_label(labels)

    # Find the best feature and threshold
    best_feature, best_threshold = find_best_split(features, labels)
    if best_feature is None:  # If no valid split found
        return calculate_majority_label(labels)

    # Split the dataset
    left_features, left_labels, right_features, right_labels = split_dataset(features, labels, best_feature, best_threshold)
    if not left_features or not right_features:
        return calculate_majority_label(labels)

    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left": build_simple_tree(left_features, left_labels, max_depth - 1),
        "right": build_simple_tree(right_features, right_labels, max_depth - 1),
    }

def predict_with_tree(tree, feature):
    if not isinstance(tree, dict):
        return tree
    if feature[tree["feature"]] <= tree["threshold"]:
        return predict_with_tree(tree["left"], feature)
    else:
        return predict_with_tree(tree["right"], feature)
def decision_tree_feedback(video_features, relevant, irrelevant, num_results, video_folder_path, mapping_csv_file_path):
    features = []
    labels = []
    for video_id in relevant:
        features.append(video_features[video_id])
        labels.append(1)
    for video_id in irrelevant:
        features.append(video_features[video_id])
        labels.append(0)
    if len(features) < 2:
        print("Not enough data for training.")
        return []
    tree = build_simple_tree(features, labels)
    scores = {video_id: predict_with_tree(tree, feature) for video_id, feature in video_features.items()}
    ranked_videos = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_results]
    # print("len : ", scores)
    print(f"Top {num_results} videos after feedback:")
    for rank, (video_id, score) in enumerate(ranked_videos, start=1):
        print(f"Rank {rank}: Video ID {video_id}, Relevance Score: {score}")
        video_visualisation(video_id, video_folder_path, mapping_csv_file_path, rank=rank)
    return ranked_videos


def knn_with_cosine_features(dataset, query, k, m):
    distances = []
    # Compute Euclidean distance between query and all dataset features
    for features, video_id in dataset:
        distance = 1 - cosine_similarity(features, query)  # Use library-defined Euclidean distance
        # print(f"distance btween {video_id} and query : {4900} is : ",distance)
        distances.append((distance, features, video_id))

    # Sort by distance and select k nearest neighbors
    distances.sort(key=lambda x: x[0])
    k_neighbors = distances[:k]
    x = []
    for dist, _, id in k_neighbors:
        # print(f"{id} : {dist}")
        x.append((dist,id))
    return x[:m]

def updateQuery(queryFeatures, relevant, irrelevant, m, csv_file_path):
    alpha = 1 / m
    beta = alpha
    data = pd.read_csv(csv_file_path)
    
    # Ensure feature columns are numeric
    feature_columns = data.iloc[:, 3:].apply(pd.to_numeric, errors='coerce')
    
    # Handle missing or invalid values
    feature_columns = feature_columns.fillna(0)
    data.iloc[:, 3:] = feature_columns

    # Initialize sums for relevant and irrelevant features
    relevantSum = np.zeros_like(queryFeatures, dtype=np.float64)
    irrelevantSum = np.zeros_like(queryFeatures, dtype=np.float64)

    # Sum relevant features
    for relId in relevant:
        relFeatures = data[data['videoId'] == relId].iloc[:, 3:].values.flatten()
        relevantSum += relFeatures.astype(np.float64)

    # Sum irrelevant features
    for irrelId in irrelevant:
        irrelFeatures = data[data['videoId'] == irrelId].iloc[:, 3:].values.flatten()
        irrelevantSum += irrelFeatures.astype(np.float64)

    # Update query features
    newQuery = queryFeatures + alpha * relevantSum - beta * irrelevantSum
    return newQuery
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
def query_lsh(hash_tables, random_planes, L, query_vector, num_results):
    candidates = {}  # To store unique candidates
    overall_candidates = 0  # Counter for all candidate vectors considered

    for layer in range(L):
        hash_value = hash_vector(query_vector, layer, random_planes)
        if hash_value in hash_tables[layer]:
            for video_id, vector in hash_tables[layer][hash_value]:
                overall_candidates += 1  # Count every candidate considered
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
def search_similar_videos(video_features, hash_tables, random_planes, L,video_folder_path, csv_file_path, mapping_csv_file_path, query_vector, num_results):
    # query_vector = video_features.get(query_video_id)
    if query_vector is None:
        print("Video not found.")
        return []
    results, unique_candidates, overall_candidates = query_lsh(hash_tables, random_planes, L, query_vector, num_results)

    # Print statistics
    print(f"Number of unique candidates considered: {unique_candidates}")
    print(f"Number of overall candidates considered: {overall_candidates}")

    # Running the similar ones
    print(f"Top {num_results} similar videos for Video ID :")
    for rank, (video_id, vector) in enumerate(results, start=1):
        similarity = cosine_similarity(query_vector, vector)
        distance = 1 - similarity
        print(f"Rank {rank}: Video ID {video_id}, Distance: {distance}")
        video_visualisation(video_id,video_folder_path,mapping_csv_file_path, rank=rank)
    return results

# Main - Initialization and usage
# TODO1: Change path to Latent Models
file_path = 'E:\\Coding\\MultimediaWebDatabases\\Phase 3\\Dataset\\'  #Path_change
# TODO2: Change path to videos dataset
video_folder_path=r"E:\\Coding\\MultimediaWebDatabases\\Phase 3\\Assets\\hmdb51_org"                            #Path_change
# TODO3: Change path to VideoId Mapping
mapping_csv_file_path="E:\\Coding\\MultimediaWebDatabases\\Phase 3\\VideoID_Mapping.csv"

Model_dict={1: "avgpool_kmeans(300).csv", 2 :"col_hist_svd(300).csv",3:"layer4_pca(300).csv"}
Model=int(input("Select model number from the below options \n 1. Average_pool + k_means + s=300 \n 2. Col_hist + svd + s=300 \n 3. Layer4 + pca + s=300 \n  "))
csv_file_path= file_path+Model_dict[Model]
video_features = load_features_from_csv(csv_file_path)

# Initialize LSH parameters
dimension = len(next(iter(video_features.values())))  
L = int(input("Please enter # of layers :  "))  #num_layers
h = int(input("Please enter # of hashes per layer :  "))  #num_hashes_per_layer
v= int(input("Enter the VideoID :  "))
query_features = video_features.get(v)
t=int(input("Enter the number of similar videos need to retrieve :  "))
# Initialize LSH structure
hash_tables, random_planes = initialize_lsh(L, h, dimension)

    # Add videos to the LSH index
for video_id, vector in video_features.items():
    add_video(hash_tables, random_planes, L, video_id, vector)

# Search for similar videos
result = search_similar_videos(video_features, hash_tables, random_planes, L, video_folder_path, csv_file_path, mapping_csv_file_path,query_vector=query_features, num_results=t)

l = list()
for rank, (video_id, vector) in enumerate(result, start=1):
        similarity = cosine_similarity(query_features, vector)
        distance = 1 - similarity
        # print(f"Rank {rank}: Video ID {video_id}, Distance: {distance}")
        l.append((rank,video_id))
print("Enter 1 for like and 0 for dislike against each video ID")

feedback = []
for rank,(video_id,_) in enumerate(result,start=1):
        choice = int(input(f"rank : {rank} , videoID : {video_id} , feedback : "))
        feedback.append(choice)
relevant = []
irrelevant = []

for i,x in enumerate(feedback):
    if(x == 1):
        relevant.append(l[i][1])
    if(x == 0):
        irrelevant.append(l[i][1])
print("relevant : ", relevant)
print("irrelevant : ", irrelevant)


data = pd.read_csv(csv_file_path)

feature_columns = data.iloc[:, 3:]
real_features = feature_columns.applymap(lambda x: np.real(complex(x)))
data.iloc[:, 3:] = real_features

target = data[(data["category"]=="target_videos") & (data["videoId"].astype(int)%2 == 0)]
features = target.iloc[:, 3:].values
IDs = target["videoId"].values
knn_data = [(list(features[i]), IDs[i]) for i in range(len(features))]
val = int(input("\n 1 : KNN Based \n 2 : Decision Tree based"))
if(val == 1):
    newQueryFeatures = updateQuery(query_features,relevant,irrelevant,t,csv_file_path)
    print("Enter value of k for knn : ")
    k = int(input())
    newResult = knn_with_cosine_features(knn_data,newQueryFeatures,k,t)
    # print(newResult)
    print("-------New Results---------")
    for rank,(dist,id) in enumerate(newResult,start=1):
        print(f"rank : {rank} , videoID : {id} , distance : {dist}")
        video_visualisation(id,video_folder_path,mapping_csv_file_path,rank)
if(val == 2):
    decision_tree_feedback(video_features, relevant, irrelevant, t, video_folder_path, mapping_csv_file_path)
