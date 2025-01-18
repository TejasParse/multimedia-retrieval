from scipy.spatial.distance import cdist
import os
import numpy as np
import csv
import torch
import cv2
import torchvision.models.video as models
import torchvision.transforms as transforms

import numpy as np
import torch.nn as nn
import pandas as pd

# Task 2

def Task2(file_path,model_name):
    
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
    
    def extract_features(file_name):
        data_array = []
        with open(file_name, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                data_array.append([float(x) for x in row])
        data_array = np.array(data_array)       
        return data_array

    def get_sigma2_tau2_pair(row):
        sigma2 = row[4]
        tau2 = row[5]   
        return (sigma2, tau2)

    def assign_row_to_cluster(features_row, cluster_representatives, sigma2_tau2_pair):
        centroids = cluster_representatives[sigma2_tau2_pair] 
        distances = cdist([features_row], centroids, 'euclidean') 
        closest_cluster = np.argmin(distances) 
        return closest_cluster


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

    #visualise the video
    cap = cv2.VideoCapture(file_path)
    if(cap.isOpened() == False):
        print("error... cant open video")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: break
    cap.release()
    cv2.destroyAllWindows()

    stips_folder = "E:\Coding\MultimediaWebDatabases\Assets\hmdb51_org_stips_filtered"
    action_subfolder = os.path.basename(os.path.dirname(file_path))  # Get the subfolder (action) name
    video_name = os.path.basename(file_path)  # Get the video filename (e.g., videoname.avi)
    # Create the corresponding STIP file name by appending '.csv' to the video filename
    stip_file_name = f"{video_name}.csv"
    # Construct the full path to the STIP file
    video_stip_path = os.path.join(stips_folder, action_subfolder, stip_file_name)
    
    def Task2b():
        cluster_representatives = load_all_cluster_representatives_hog()
        stip_features = extract_features(video_stip_path)
        hist_data = []
        for row in stip_features:
            ind1 = assign_row_to_cluster(row[7:79], cluster_representatives, get_sigma2_tau2_pair(row))
            hist_data.append([get_sigma2_tau2_pair(row), ind1])
        # create_histogram_for_pairs(hist_data)
        bog_hog_480 = create_and_concatenate_histograms(hist_data)
        # print("Concatenated 480-dimensional vector:", bog_hog_480)
        # print("Shape of the concatenated vector:", bog_hog_480.shape)
        return bog_hog_480
    
    def Task2c():
        cluster_representatives = load_all_cluster_representatives_hof()
        stip_features = extract_features(video_stip_path)
        hist_data = []
        for row in stip_features:
            ind1 = assign_row_to_cluster(row[79:], cluster_representatives, get_sigma2_tau2_pair(row))
            hist_data.append([get_sigma2_tau2_pair(row), ind1])
        # Concatenate the 12 histograms into a 480-dimensional vector
        bog_hof_480 = create_and_concatenate_histograms(hist_data)
        # print("Concatenated 480-dimensional vector:", bog_hof_480)
        # print("Shape of the concatenated vector:", bog_hof_480.shape)
        return bog_hof_480

    if model_name == "hog":
        final = Task2b()
        return final
    elif model_name =="hof":
        final = Task2c()
        return final

def load_vectors(all_hogs_folder):
    # List all the CSV files in the directory
    csv_files = [file for file in os.listdir(all_hogs_folder) if file.endswith('.csv')]
    
    hog_vectors = []
    file_names = []

    # Loop through each CSV file and load the vector
    for csv_file in csv_files:
        file_path = os.path.join(all_hogs_folder, csv_file)
        df = pd.read_csv(file_path, header=None)
        hog_vectors.append(df.values.flatten())  # Flatten to ensure it's 1D
        file_names.append(csv_file)  # Save the file name without the path

    # Convert the list of vectors into an n x 480 NumPy array
    hog_array = np.array(hog_vectors)

    return hog_array, file_names

def model_index_3(video_file):
    # Assume Task2 returns the 1x480 vector for the given video
    video_hog = Task2(video_file, "hog")  # Shape: (1, 480)

    all_hogs = "E:\\Coding\\MultimediaWebDatabases\\Outputs\\Task2\\hog"

    # Load the HOG vectors and corresponding filenames
    data, file_names = load_vectors(all_hogs)  # data shape: (732, 480)

    # Compute Cosine distances between video_hog and all HOG vectors in data using cdist
    distances = cdist(video_hog.reshape(1, -1), data, metric='cosine').flatten()

    # Get the indices of the 10 smallest distances
    top_10_indices = np.argsort(distances)[:10]

    # Retrieve the corresponding filenames
    top_10_similar_files = [file_names[i].replace('.csv', '.avi') for i in top_10_indices]
    top_10_distances = [distances[i] for i in top_10_indices]

    top_10_results = [[file_name, distance] for file_name, distance in zip(top_10_similar_files, top_10_distances)]

    return top_10_results

def model_index_4(video_file):
    # Assume Task2 returns the 1x480 vector for the given video
    video_hog = Task2(video_file, "hof")  # Shape: (1, 480)

    all_hofs = "E:\\Coding\\MultimediaWebDatabases\\Outputs\\Task2\\hof"

    # Load the HOF vectors and corresponding filenames
    data, file_names = load_vectors(all_hofs)  # data shape: (732, 480)

    # Compute Cosine distances between video_hog and all HOG vectors in data using cdist
    distances = cdist(video_hog.reshape(1, -1), data, metric='cosine').flatten()

    # Get the indices of the 10 smallest distances
    top_10_indices = np.argsort(distances)[:10]

    # Retrieve the corresponding filenames
    top_10_similar_files = [file_names[i].replace('.csv', '.avi') for i in top_10_indices]
    top_10_distances = [distances[i] for i in top_10_indices]

    top_10_results = [[file_name, distance] for file_name, distance in zip(top_10_similar_files, top_10_distances)]

    return top_10_results

# Task 1

def task1(file_path,model_name):
    
    #visualise the video
    cap = cv2.VideoCapture(file_path)
    if(cap.isOpened() == False):
        print("error... cant open video")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: break
    cap.release()
    cv2.destroyAllWindows()


    def getFrames(file_path):
        '''
        Reads the video frame by frame and returns a list of frames
        '''
        video = cv2.VideoCapture(file_path)
        frames = []
        while True:
            ret, frame = video.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        video.release()
        return frames

    def processFrames(frames,t):
        '''
        Processes and transforms the frames based on the transformation (t) provided
        '''
        processed = []
        for frame in frames:
            frame = t(frame)
            processed.append(frame)
        return torch.stack(processed)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112,112)),
        transforms.ToTensor()  
    ])

    frames = getFrames(file_path)
    pro = processFrames(frames,transform)
    pro = pro.unsqueeze(0) # Add batch dimension , shape becomes [N,D,C,H,W] , N-batch, D-Depth, C-channels, Height(H) and Width(W) of frame
    pro1 = torch.movedim(pro,1,2) # convert to shape [N,C,D,H,W] which is the required input shape in r3d 
    
    def hook(module,input,output):
        global features
        features = output

    def initialize_model(model_name):
        '''
        Add hooks to the layer based on the model name provided
        '''
        global features
        model = models.r3d_18(pretrained=True)
        model.eval()
        if model_name == "layer3":
            h1 = model.layer3.register_forward_hook(hook)
        elif model_name == "layer4":
            h1 = model.layer4.register_forward_hook(hook)
        elif model_name == "avgpool":
            h1 = model.avgpool.register_forward_hook(hook)
        # pro1 = pro[0][None,:,:,:]
        out = model(pro1)
        h1.remove()

    initialize_model(model_name)

    if(features.shape[1] == 256): 
        # if model is layer 3, use a convolutional layer to bring channels to 512 then perform mean to get 512 dimensional tensor
        torch.manual_seed(0) # setting seed so that linear layer is initialised with same weights each time 
        avg_features = torch.mean(features,dim=(3,4))
        squeezed = torch.squeeze(avg_features) #remove batch dimension
        in_tensor = torch.flatten(squeezed) #collapse into a single dimension 
        myLayer = nn.Linear(in_features=256*features.shape[2],out_features=512) # define a linear layer 
        final_tensor = myLayer(torch.squeeze(in_tensor)) #remove batch dimension with squeeze and then apply linear transformation
        # print("in l3")
    elif(features.shape[1] == 512 and features.shape[2] != 1): # case of layer 4
        # average the tensor on dimension 2,3,4 to get 512 dimensional tensor
        final_tensor = torch.squeeze(torch.mean(features,dim=(2,3,4)))
        # print("in l4")
    else: #case of avgpool
        #this layer will already give output as 512 dimensional tensor
        final_tensor = torch.squeeze(features)
        # print("in avg")
    return final_tensor

def compute_similarity(features1, features2, metric='cosine'):
    # Ensure the input is 2D by reshaping if necessary
    if len(features1.shape) == 1:
        features1 = features1.reshape(1, -1)
    if len(features2.shape) == 1:
        features2 = features2.reshape(1, -1)
    return cdist(features1, features2, metric=metric)

# Function to load the features from the saved .pt file
def load_features_from_pt(file_path):
    # Load the PyTorch tensor
    tensor = torch.load(file_path)
    # Detach from the computation graph, move to CPU, and convert to numpy
    return tensor.detach().cpu().numpy()

# Function to get the 10 most similar videos for a given model
def get_most_similar_videos(input_video_features, model_name, metric='cosine', top_n=10):
    similarity_scores = []
    
    # Directory containing the feature vectors for the specified model
    model_dir = os.path.join("E:\Coding\MultimediaWebDatabases\Outputs\Task1", model_name)
    
    # Compare input video features with each stored video feature in the model directory
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".pt"):  # Make sure you're comparing only feature files
                video_feature_filepath = os.path.join(root, file)
                target_features = load_features_from_pt(video_feature_filepath)
                similarity = compute_similarity(input_video_features.detach().cpu().numpy(), target_features, metric)
                video_filepath = file.replace('.pt', '.avi')  # Assuming the video file name matches the feature name
                # similarity_scores.append((video_filepath, avg_similarity))
                similarity_scores.append((video_filepath, similarity))
    # Sort videos by similarity score
    similarity_scores.sort(key=lambda x: x[1])  # Ascending order for distances (lower distance = more similar)
    # Return the top N most similar videos
    return similarity_scores[1:top_n+1]

def find_similar_videos(input_video_filepath, model_name, metric='cosine'):
    input_video_features = task1(input_video_filepath,model_name)
    # Find the top 10 similar videos for the specified model
    top_similar_videos = get_most_similar_videos(input_video_features, model_name, metric)
    return top_similar_videos

def model_index_0(path):
    model_name = "layer3"
    similar_videos = find_similar_videos(path, model_name)
    return similar_videos
def model_index_1(path):
    model_name = "layer4"
    similar_videos = find_similar_videos(path, model_name)
    return similar_videos
def model_index_2(path):
    model_name = "avgpool"
    similar_videos = find_similar_videos(path, model_name)
    return similar_videos


# Task 3

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

def n_sim_videos(inp_vid_feature, feature_folder, top_n=10):
    sim_scores = []
    feature_vector_of_inp = np.load(inp_vid_feature)
     
    for files in os.listdir(feature_folder):
        if files != inp_vid_feature:
            feature_vector_of_other_vids = np.load(os.path.join(feature_folder,files))
            similarity = cosine_similarity([feature_vector_of_inp], [feature_vector_of_other_vids])
            sim_scores.append((files,similarity))
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    return sim_scores[:top_n]

def model_index_5(video_path):
    feature_folder = "E:\Coding\MultimediaWebDatabases\Outputs\Task3\Feature_Vector"
    visualize_videos([video_path])

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    formatted_video_name = video_name.replace(' ', '_')
    # print(formatted_video_name, "Yo")
    inp = os.path.join(feature_folder, f"{formatted_video_name}_features.npy")

    # inp = "Outputs/Task3/Feature_Vector/#437_How_To_Ride_A_Bike_ride_bike_f_cm_np1_ba_med_0_features.npy"
    top = n_sim_videos(inp_vid_feature=inp,feature_folder=feature_folder)
    return top

# Helper Function

def print_results(results):
    ind1 = 1
    # print(results, "Yo What is happening")
    for name, score in results:
        print(ind1, name, score)
        ind1+=1


# video_path = "E:\\Coding\\MultimediaWebDatabases\\Assets\\hmdb51_org\\target_videos\\cartwheel\\Bodenturnen_2004_cartwheel_f_cm_np1_le_med_0.avi"
# video_path = "E:\\Coding\\MultimediaWebDatabases\\Assets\\hmdb51_org\\target_videos\\sword_exercise\\Blade_Of_Fury_-_Scene_1_sword_exercise_f_cm_np1_ri_med_3.avi"
# video_path = "E:\\Coding\\MultimediaWebDatabases\\Assets\\hmdb51_org\\target_videos\\sword\\AHF_longsword_against_Rapier_and_Dagger_Fight_sword_f_cm_np2_ri_bad_0.avi"
# video_path = "E:\\Coding\\MultimediaWebDatabases\\Assets\\hmdb51_org\\target_videos\\drink\\CastAway2_drink_u_cm_np1_le_goo_8.avi"

# 3: HoG, 4: HoF
# model_index = 3

# results = model_index_0(video_path)
# # print(results, "Kya hora hai")
# print_results(results)

def visualize_videos(video_files):
    for video_file in video_files:
        # Open the video file
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}")
            continue

        # print(f"Playing video: {video_file}")
        
        # Loop through each frame of the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display the frame
            cv2.imshow('Video', frame)

            # Press 'q' to quit the video visualization
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Release the video capture object and close the window
        cap.release()
        cv2.destroyAllWindows()

def getFilesPathTask1(results):
    # Extract filenames and remove "_features"
    file_names = [i[0].replace('_features', '') for i in results]
    search_directory = "E:\\Coding\\MultimediaWebDatabases\\Assets\\hmdb51_org\\target_videos"
    
    video_paths = []

    for file_name in file_names:

        for root, dirs, files in os.walk(search_directory):
            for file in files:
                if file_name in file and file.endswith(('.avi')): 
                    video_paths.append(os.path.join(root, file))
                    break  

    # print(video_paths, "Absolute Paths of Matching Videos")
    return video_paths

def getFilesPathTask2(results):

    file_names = [i[0] for i in results]
    search_directory = "E:\\Coding\\MultimediaWebDatabases\\Assets\\hmdb51_org\\target_videos"
    
    video_paths = []

    for file_name in file_names:

        for root, dirs, files in os.walk(search_directory):
            for file in files:
                if file_name in file and file.endswith(('.avi')): 
                    video_paths.append(os.path.join(root, file))
                    break  

    # print(video_paths, "Absolute Paths of Matching Videos")
    return video_paths
    
def getFilesPathTask3(results):

    file_names = [i[0].replace('_features.npy', '.avi') for i in results]
    search_directory = "E:\\Coding\\MultimediaWebDatabases\\Assets\\hmdb51_org\\target_videos"
    
    video_paths = []

    for file_name in file_names:

        for root, dirs, files in os.walk(search_directory):
            for file in files:
                if file_name in file and file.endswith(('.avi')): 
                    video_paths.append(os.path.join(root, file))
                    break  

    # print(video_paths, "Absolute Paths of Matching Videos")
    return video_paths

def is_video_file(file_path):
    # Check if the file has a video extension
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    return os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in video_extensions

while True:

    print("\n")

    user_input = input("Enter video file path or enter 'q' to quit: ").strip()
    if user_input.lower() == 'q':
        print("Quitting the program.")
        break

    # Check if the provided file path is valid and is a video file
    if not is_video_file(user_input):
        print("Invalid video file path. Please try again.")
        continue

    # Display menu options
    print("\nSelect an option from the menu:")
    print("0: layer3")
    print("1: layer4")
    print("2: avg-pool")
    print("3: HoG")
    print("4: HoF")
    print("5: Col-Hist")
    
    try:
        selected_index = int(input("Enter your choice (0-5): ").strip())
        if selected_index not in range(6):
            print("Invalid selection. Please choose a number between 0 and 5.")
            continue
    except ValueError:
        print("Invalid input. Please enter a number between 0 and 5.")
        continue
    # Call the corresponding model_index function based on the selection
    function_map = {
        0: model_index_0,
        1: model_index_1,
        2: model_index_2,
        3: model_index_3,
        4: model_index_4,
        5: model_index_5
    }
    
    # Execute the corresponding function and get the results
    results = function_map[selected_index](user_input)

    print("\nResults: ")
    print_results(results)

    if(selected_index <= 2):
        top_file_paths = getFilesPathTask1(results)
        print("\nVisualizing Top 10 Similar Videos\n")
        visualize_videos(top_file_paths)
    elif(selected_index >=3 and selected_index <= 4):
        top_file_paths = getFilesPathTask2(results)
        print("\nVisualizing Top 10 Similar Videos\n")
        visualize_videos(top_file_paths)
    elif(selected_index >= 5):
        top_file_paths = getFilesPathTask3(results)
        print("\nVisualizing Top 10 Similar Videos\n")
        visualize_videos(top_file_paths)

    

    