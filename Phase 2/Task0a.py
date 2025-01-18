import torch
import torchvision.models.video as models
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
import numpy as np
import os
import warnings
import csv
import time
from collections import defaultdict
from scipy.spatial.distance import cdist, euclidean
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy.spatial import distance

######## Model 1 ############################

def task1(file_path,model_name):
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
        if model_name == "R3D18-Layer3-512":
            h1 = model.layer3.register_forward_hook(hook)
        elif model_name == "R3D18-Layer4-512":
            h1 = model.layer4.register_forward_hook(hook)
        elif model_name == "R3D18-AvgPool-512":
            h1 = model.avgpool.register_forward_hook(hook)
        # pro1 = pro[0][None,:,:,:]
        out = model(pro1)
        h1.remove()

    initialize_model(model_name)

    if(features.shape[1] == 256): 
    # if model is layer 3, average spatial dimensions and then flatten the tensor followed by a linear transformation to get 512 dimensional tensor
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



    # query_path = '/Users/tanishquezaware/Documents/MS/Sem 1/CSE 515 Multimedia/Project/target_videos/'+category+'/'+Query_Video_Name +".avi"
    # # query_path = '/Users/tanishquezaware/Documents/MS/Sem 1/CSE 515 Multimedia/Project/target_videos/cartwheel/Bodenturnen_2004_cartwheel_f_cm_np1_le_med_0.avi'
    # if Model=="Model1a":
    #     model = "layer3"
    # if Model=="Model1b":
    #     model = "layer4"
    # if Model=="Model1c":
    #     model = "avgpool"
    # os.makedirs("/Users/tanishquezaware/Documents/MS/Sem 1/CSE 515 Multimedia/Project/Input_features/"+category+"/"+Query_Video_Name+"/"+Model, exist_ok=True)
    # feature_path="/Users/tanishquezaware/Documents/MS/Sem 1/CSE 515 Multimedia/Project/Input_features/"+category+"/"+Query_Video_Name+"/"+Model+"/" +Query_Video_Name+".pt"
    # task1(query_path,model)
    # torch.save(task1(query_path,model), feature_path)

############ Model2 ###########################


def task2(file_path,model_name):
    def load_all_cluster_representatives_hog():
        cluster_representatives = {}
        sigma2_values = [4, 8, 16, 32, 64, 128]
        tau2_values = [2, 4]
        pair_index = 0
        for sigma2 in sigma2_values:
            for tau2 in tau2_values:
                # TODO1: Change HoG Cluster Centers File path
                centroids_file = f'E:\\Coding\\MultimediaWebDatabases\Phase 2\\HoG\\pair_{sigma2}_{tau2}_HoG.csv'
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
                # TODO2: Change HoF Cluster Centers File path
                centroids_file = f'E:\\Coding\\MultimediaWebDatabases\\Phase 2\\HoF\\pair_{sigma2}_{tau2}_HoF.csv'
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
    
    def PreProcessTask2():
        folder_path = "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Assets\\hmdb51_org_stips_filtered"

        # Initialize an empty list to store folder name, file name, and file path
        file_data = []

        # Walk through the directory
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Get the folder name (last part of the root path)
                folder_name = os.path.basename(root)
                # Get the full path of the file
                full_path = os.path.join(root, file)
                
                # Append folder name, file name, and full path as a row to the list
                file_data.append([folder_name, file, full_path])

    
        output_base_path = '/Users/tanishquezaware/Documents/MS/Sem 1/CSE 515 Multimedia/Project/Phase 2/hmdb51_org_stips_filtered'
        for i in file_data:
            indi_data = []
            with open(i[2], 'r') as file:
                print(f"Reading from: {i[1]}")
                for line in file:
                
                    indiLine = line.strip().split("\t")
                    
                    if len(indiLine) == 1:
                        continue

                    indi_data.append([float(x) for x in indiLine])

            # Sort by detector-confidence (index 6) in reverse order
            indi_data.sort(key=lambda x: x[6], reverse=True)

            # Check if we have fewer than 400 points, log it
            if len(indi_data) < 400:
                print(f"Less than 400 points for {i[1]}")

            # Take the top 400 points
            indi_data = indi_data[:400]

            # Determine the output folder and file path
            folder_name = i[0]  # Assuming i[0] is the folder name
            file_name = i[1].replace('.txt', '.csv')    # Assuming i[1] is the file name

            output_folder_path = os.path.join(output_base_path, folder_name)
            output_file_path = os.path.join(output_folder_path, file_name)

            # Create folder if it does not exist
            os.makedirs(output_folder_path, exist_ok=True)

            # Write the filtered data to the new file in CSV format
            with open(output_file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerows(indi_data)

    ########## Uncomment below function only when to update the filtered stips  ############
    # PreProcessTask2() 
    # TODO3: Change stips_filtered folder
    stips_folder = 'E:\\Coding\MultimediaWebDatabases\\Phase 2\\Assets\\hmdb51_org_stips_filtered'
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

    if model_name == "BOF-HOG-480":
        final = Task2b()
        return final
    elif model_name =="BOF-HOF-480":
        final = Task2c()
        return final

    # video_path = "/Users/tanishquezaware/Documents/MS/Sem 1/CSE 515 Multimedia/Project/target_videos/"+category+"/" +Query_Video_Name+".avi"


    # if Model=="Model2b":
    #     video_hog = Task2(video_path, "hog") 
    #     print(video_hog)
    #     os.makedirs("/Users/tanishquezaware/Documents/MS/Sem 1/CSE 515 Multimedia/Project/Input_features/"+category+"/" +Query_Video_Name+"/"+Model, exist_ok=True)
    #     np.save("/Users/tanishquezaware/Documents/MS/Sem 1/CSE 515 Multimedia/Project/Input_features/"+category+"/" +Query_Video_Name+"/"+Model+"/"+Query_Video_Name,video_hog)
        
    # if Model=="Model2c":
    #     video_hof = Task2(video_path, "hof") 
    #     print(video_hof)
    #     os.makedirs("/Users/tanishquezaware/Documents/MS/Sem 1/CSE 515 Multimedia/Project/Input_features/"+category+"/" +Query_Video_Name+"/"+Model, exist_ok=True)
    #     np.save("/Users/tanishquezaware/Documents/MS/Sem 1/CSE 515 Multimedia/Project/Input_features/"+category+"/" +Query_Video_Name+"/"+Model+"/"+Query_Video_Name,video_hof)
    
predefined_colors = np.array([
    [5, 127, 128], 
    [159, 127, 126],  
    [75, 128, 139],
    [189, 123, 167],
    [129, 126, 132],
    [51, 128, 134],
    [27, 127, 132],
    [192, 126, 129],
    [236, 125, 132],     
    [148, 123, 159],
    [107, 130, 150],
    [99, 127, 126],
])

def task3_frameextract(path, r=4, n=12, frames_folder='Frames', feature_folder='Features'):
    hist_dir = "Outputs_trial/Task3_trial/Histograms_Framewise_trial"
    os.makedirs(frames_folder, exist_ok=True) # Folder for saving frames
    os.makedirs(frames_folder, exist_ok=True) # Folder for saving frames
    os.makedirs(feature_folder, exist_ok=True)  # Folder for saving features

    base_name = os.path.basename(path).split('.')[0] # This is to get the video name and not path
    cam = cv2.VideoCapture(path)
    
    # Here we are counting total frames in the video
    total_frames=int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(total_frames)
    frameno = 0
    while(True):
        ret,frame = cam.read()
        if ret:
            frameno=frameno+1  # Counter for frames
            if(frameno==1):  # If and else condition for reaching first,middle and last frame and saving them
                frame_name = os.path.join(frames_folder, f'{base_name}_frame_1.jpg')  # Save with base name
                # print("First frame: "+frame_name)
                cv2.imwrite(frame_name,frame)
            elif (frameno==int(total_frames/2)):
                frame_name = os.path.join(frames_folder, f'{base_name}_frame_2.jpg')  # Save with base name
                # print("Middle frame: "+frame_name)
                cv2.imwrite(frame_name,frame)
            elif (frameno==int(total_frames)-1):
                frame_name = os.path.join(frames_folder, f'{base_name}_frame_3.jpg')  # Save with base name
                # print("Last frame: "+frame_name)
                cv2.imwrite(frame_name,frame)
                # Will skip rest of the frames
        else:
            break # If no more frames are found in video we exit
    
    cam.release()
    cv2.destroyAllWindows()

def task3(path, r=4, n=12, frames_folder='Frames', feature_folder='Features'):

    task3_frameextract(path,4,frames_folder='Frames',feature_folder=feature_folder)

    hist_dir = "Outputs_trial/Task3_trial/Histograms_Framewise_trial"
    os.makedirs(frames_folder, exist_ok=True) # Folder for saving frames
    os.makedirs(frames_folder, exist_ok=True) # Folder for saving frames
    os.makedirs(feature_folder, exist_ok=True)  # Folder for saving features
    base_name = os.path.basename(path).split('.')[0] # This is to get the video name and not path
    histos = []
    video_feature_vector = [] # This is a list for feature vectors of concatenated histograms
    
    for frame_index in range(1, 4):  # first loop to go through first,middle and last frame
        image_name = os.path.join(frames_folder, f'{base_name}_frame_{frame_index}.jpg')
        img = cv2.imread(image_name)
        if img is None:
            print(f"Error reading {image_name}. File may not exist or is corrupted.")
            continue  # Skip to the next frame
        
        im_h, im_w, channels = img.shape  # Reading the image and extracting its height and width
        
        # Feature vector for the current frame
        frame_feature_vector = []
        
        for i in range(1,r+1): # This loop is for going from top to bottom of image
            for j in range(1,r+1):  # This loop is for going from left to right of image
                tile = img[(im_h//r)*(i-1):(im_h//r)*i,(im_w//r)*(j-1):(im_w//r)*j]# This is cutting
                # out cells based on dimensions of the image
                rgb_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2Lab) # Converting cells/tiles from bgr to rgb
                
                pixels = rgb_tile.reshape((-1, 3))
                dists = distance.cdist(pixels, predefined_colors, 'euclidean')
                labels = np.argmin(dists, axis=1)
                n_colors = 12
                hist, _ = np.histogram(labels, bins=np.arange(0, n_colors+1))
                # print(hist)
                # # Concatenate histograms for all three channels into a single vector for this tile
                
                # plt.figure(figsize=(8, 6))
                # plt.bar(range(n_colors), hist, color=np.array(predefined_colors) / 255, tick_label=np.arange(n_colors),edgecolor="black")
                # plt.title('Predefined Color Histogram')
                # plt.xlabel('Color Index')
                # plt.ylabel('Pixel Count')
                # plt.show()
                # Append this tile's histogram to the frame feature vector
                frame_feature_vector.extend(hist)
                
        
        # Append this frame's feature vector to the video feature vector
        # print(frame_feature_vector)
        video_feature_vector.extend(frame_feature_vector)
        
        # following is to save the histogram based on details like timestamp and video it belongs to
#         timestamp = int(time.time())  # here I used timestamp because if the name of histograms are same than it will
#         # overwrite the saved histogram so with timestamp it gives unique name to each saved file
#         hist_name = f'{base_name}_histogram_frame_{frame_index}_{timestamp}.png'
#         hist_path = os.path.join(hist_dir, hist_name)
#         plt.savefig(hist_path)  # Close the figure after saving
        
#         histos.append(hist_path)
    # print(video_feature_vector)
    video_feature_vector = np.array(video_feature_vector) # Here we convert the video feature vector to np array
    # print(video_feature_vector)
    # following is to save the feature vector
    feature_file = os.path.join(feature_folder,'col_hist.npy')
    np.save(feature_file,video_feature_vector)
    
    # print(f"Saved feature vector for video '{base_name}' to {feature_file}")
    
    return video_feature_vector, histos

# Define the paths to the two folders
# TODO4: Change directory of dataset
non_target_videos = "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Assets\\hmdb51_org\\non_target_videos"      ##### Path_Change ####
target_videos = "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Assets\\hmdb51_org\\target_videos"              ##### Path_Change ####

# Initialize an empty list to store video filenames
video_files = []

# Walk through the first folder and collect video file paths
for root, dirs, files in os.walk(non_target_videos):
    for file in files:
        if file.endswith(('.avi')): 
            video_files.append(os.path.join(root, file))

# Walk through the second folder and collect video file paths
for root, dirs, files in os.walk(target_videos):
    for file in files:
        if file.endswith(('.avi')):  
            video_files.append(os.path.join(root, file))

# Dictionary to store the videoID to filename mapping
video_id_map = {}

# Assign unique videoID by incrementing counter
for video_id, video_file in enumerate(video_files):
    video_id_map[video_id] = video_file

# TODO5: Change the storage location of VideoId Mapping 
with open('E:\\Coding\\MultimediaWebDatabases\\Phase 2\\VideoID_Mapping1.csv', mode='w', newline='') as file:         ##### Path_Change ####
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(["VideoID", "Filename"])
    
    # Write the data
    for key, value in video_id_map.items():
        writer.writerow([key, os.path.basename(value)])

feature_naming={"Model1a":"R3D18-Layer3-512","Model1b":"R3D18-Layer4-512","Model1c":"R3D18-AvgPool-512",
                "Model2b":"BOF-HOG-480", "Model2c":"BOF-HOF-480","Model3":"COL-HIST"}

folder_count = 0
max_folders = 40

for video_id,video_file in video_id_map.items():
           
    print("Video ID: ", video_id)

    # TODO6: Change file storage locations
    if "/target_videos/" in video_file:
        subfolder = 'Even' if video_id % 2 == 0 else 'Odd'
        folder_path = f'E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Output1\\target_videos\\{subfolder}\\videoID_{video_id}'      ##### Path_Change ####
    else:
        folder_path = f'E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Output1\\non_target_videos\\videoID_{video_id}'             ##### Path_Change ####

    os.makedirs(folder_path, exist_ok=True)
    folder_count+=1
    for Model_name,feature_model in feature_naming.items():
        feature_file = os.path.join(folder_path, os.path.basename(feature_model))
        if "Model1" in Model_name:
            torch.save(task1(video_file,feature_model), feature_file)
        if "Model2" in Model_name:
            np.save(feature_file,task2(video_file,feature_model))
        if "Model3" in Model_name:
            frames_path=feature_file+"/Frames/"
            # print(frames_path)
            task3(video_file, 4, 12, frames_path, feature_file)

    metadata_path = os.path.join(folder_path, 'metadata.txt')
    with open(metadata_path, 'w') as metadata_file:
        metadata_file.write(f"Filename: {os.path.basename(video_file)}\n")
        category_label= os.path.basename(os.path.dirname(video_file))
        if "/target_videos/" in video_file and subfolder == 'Even' :
            metadata_file.write(f"Category Label: {category_label}\n")

import os
import numpy as np

# TODO7: Change to location where the over code saved the output
non_target_path = "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Output\\non_target_videos"
target_path = "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Output\\target_videos"
# TODO8: Change the Final Task 0a Output
Final_Task0a_Output_Folder = "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0_Test"

if not os.path.exists(Final_Task0a_Output_Folder):
    os.makedirs(Final_Task0a_Output_Folder)

def process_hog_hof(non_target_path, target_path, output_folder):
    # Define the base path
    base_path = non_target_path

    # Arrays to store HOG and HOF data
    hog_data_list = []
    hof_data_list = []

    # Traverse through the directories
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Extract videoId from the folder name (assuming format: "videoID_{id}")
            videoId = folder_name.split("_")[-1]
            
            # Define the paths for HOG, HOF, and metadata files
            hog_file = os.path.join(folder_path, "BOF-HOG-480.npy")
            hof_file = os.path.join(folder_path, "BOF-HOF-480.npy")
            metadata_file = os.path.join(folder_path, "metadata.txt")
            
            # Check if both HOG and HOF files exist
            if os.path.exists(hog_file) and os.path.exists(hof_file):
                # Load HOG and HOF data
                hog_data = np.load(hog_file)
                hof_data = np.load(hof_file)
                
                # Initialize variables for videoName and category
                videoName = ""
                category = ""
                
                # Read the metadata.txt file if it exists
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            # Extract videoName from the first line
                            videoName = lines[0].strip().replace("Filename: ", "")
                        if len(lines) > 1:
                            # Extract category from the second line (optional)
                            category = lines[1].strip().replace("Category Label: ", "")
                
                # If category is not present, use an empty string
                if not category:
                    category = ""
                
                # Append the HOG data [videoId, videoName, category, "non_target_videos", hog_array...]
                hog_entry = [videoId, videoName, category, "non_target_videos"] + hog_data.tolist()
                hog_data_list.append(hog_entry)
                
                # Append the HOF data [videoId, videoName, category, "non_target_videos", hof_array...]
                hof_entry = [videoId, videoName, category, "non_target_videos"] + hof_data.tolist()
                hof_data_list.append(hof_entry)

    # Now `hog_data_list` contains all HOG data and `hof_data_list` contains all HOF data
    print(f"Total HOG videos processed: {len(hog_data_list)}")
    print(f"Total HOF videos processed: {len(hof_data_list)}")

    # Define the base path
    base_path = f"{target_path}\\Even"

    # Traverse through the directories
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Extract videoId from the folder name (assuming format: "videoID_{id}")
            videoId = folder_name.split("_")[-1]
            
            # Define the paths for HOG, HOF, and metadata files
            hog_file = os.path.join(folder_path, "BOF-HOG-480.npy")
            hof_file = os.path.join(folder_path, "BOF-HOF-480.npy")
            metadata_file = os.path.join(folder_path, "metadata.txt")
            
            # Check if both HOG and HOF files exist
            if os.path.exists(hog_file) and os.path.exists(hof_file):
                # Load HOG and HOF data
                hog_data = np.load(hog_file)
                hof_data = np.load(hof_file)
                
                # Initialize variables for videoName and category
                videoName = ""
                category = ""
                
                # Read the metadata.txt file if it exists
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            # Extract videoName from the first line
                            videoName = lines[0].strip().replace("Filename: ", "")
                        if len(lines) > 1:
                            # Extract category from the second line (optional)
                            category = lines[1].strip().replace("Category Label: ", "")
                
                # If category is not present, use an empty string
                if not category:
                    category = ""
                
                # Append the HOG data [videoId, videoName, category, "non_target_videos", hog_array...]
                hog_entry = [videoId, videoName, category, "target_videos"] + hog_data.tolist()
                hog_data_list.append(hog_entry)
                
                # Append the HOF data [videoId, videoName, category, "non_target_videos", hof_array...]
                hof_entry = [videoId, videoName, category, "target_videos"] + hof_data.tolist()
                hof_data_list.append(hof_entry)

    # Now `hog_data_list` contains all HOG data and `hof_data_list` contains all HOF data
    print(f"Total HOG videos processed: {len(hog_data_list)}")
    print(f"Total HOF videos processed: {len(hof_data_list)}")

    # Define the base path
    base_path = f"{target_path}\\Odd"

    # Traverse through the directories
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Extract videoId from the folder name (assuming format: "videoID_{id}")
            videoId = folder_name.split("_")[-1]
            
            # Define the paths for HOG, HOF, and metadata files
            hog_file = os.path.join(folder_path, "BOF-HOG-480.npy")
            hof_file = os.path.join(folder_path, "BOF-HOF-480.npy")
            metadata_file = os.path.join(folder_path, "metadata.txt")
            
            # Check if both HOG and HOF files exist
            if os.path.exists(hog_file) and os.path.exists(hof_file):
                # Load HOG and HOF data
                hog_data = np.load(hog_file)
                hof_data = np.load(hof_file)
                
                # Initialize variables for videoName and category
                videoName = ""
                category = ""
                
                # Read the metadata.txt file if it exists
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            # Extract videoName from the first line
                            videoName = lines[0].strip().replace("Filename: ", "")
                        if len(lines) > 1:
                            # Extract category from the second line (optional)
                            category = lines[1].strip().replace("Category Label: ", "")
                
                # If category is not present, use an empty string
                if not category:
                    category = ""
                
                # Append the HOG data [videoId, videoName, category, "non_target_videos", hog_array...]
                hog_entry = [videoId, videoName, category, "target_videos"] + hog_data.tolist()
                hog_data_list.append(hog_entry)
                
                # Append the HOF data [videoId, videoName, category, "non_target_videos", hof_array...]
                hof_entry = [videoId, videoName, category, "target_videos"] + hof_data.tolist()
                hof_data_list.append(hof_entry)

    # Now `hog_data_list` contains all HOG data and `hof_data_list` contains all HOF data
    print(f"Total HOG videos processed: {len(hog_data_list)}")
    print(f"Total HOF videos processed: {len(hof_data_list)}")

    
    # Define the headers
    headers = ['videoId', 'videoName', 'label', 'category'] + [f'feature_{i}' for i in range(480)]  # 480 HOG features
    hof_headers = ['videoId', 'videoName', 'label', 'category'] + [f'feature_{i}' for i in range(480)]  # 480 HOF features

    # Define the file paths
    hog_csv_path = f"{output_folder}\\hog_data.csv"
    hof_csv_path = f"{output_folder}\\hof_data.csv"

    # Save HOG data with headers
    hog_df = pd.DataFrame(hog_data_list)
    hog_df.to_csv(hog_csv_path, index=False, header=headers)

    # Save HOF data with headers
    hof_df = pd.DataFrame(hof_data_list)
    hof_df.to_csv(hof_csv_path, index=False, header=hof_headers)

process_hog_hof(non_target_path, target_path, Final_Task0a_Output_Folder)

def process_dnn_features(non_target_path, target_path, output_folder):
    # Define the base path
    base_path = non_target_path

    # Arrays to store data from the three models
    model1_data_list = [] # Avg
    model2_data_list = [] # Layer3
    model3_data_list = [] # Layer4

    # Traverse through the directories
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Extract videoId from the folder name (assuming format: "videoID_{id}")
            videoId = folder_name.split("_")[-1]
            
            # Define the paths for the models and metadata files
            model1_file = os.path.join(folder_path, "R3D18-AvgPool-512")
            model2_file = os.path.join(folder_path, "R3D18-Layer3-512")
            model3_file = os.path.join(folder_path, "R3D18-Layer4-512")
            metadata_file = os.path.join(folder_path, "metadata.txt")
            
            # Check if all model files exist
            if os.path.exists(model1_file) and os.path.exists(model2_file) and os.path.exists(model3_file):
                # Load model data
                model1_data = torch.load(model1_file)
                model2_data = torch.load(model2_file)
                model3_data = torch.load(model3_file)
                
                # Initialize variables for videoName and category
                videoName = ""
                category = ""
                
                # Read the metadata.txt file if it exists
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            # Extract videoName from the first line
                            videoName = lines[0].strip().replace("Filename: ", "")
                        if len(lines) > 1:
                            # Extract category from the second line (optional)
                            category = lines[1].strip().replace("Category Label: ", "")
                
                # If category is not present, use an empty string
                if not category:
                    category = ""
                
                # Append the model1 data [videoId, videoName, category, "non_target_videos", model1_data...]
                model1_entry = [videoId, videoName, category, "non_target_videos"] + model1_data.tolist()
                model1_data_list.append(model1_entry)
                
                # Append the model2 data [videoId, videoName, category, "non_target_videos", model2_data...]
                model2_entry = [videoId, videoName, category, "non_target_videos"] + model2_data.tolist()
                model2_data_list.append(model2_entry)
                
                # Append the model3 data [videoId, videoName, category, "non_target_videos", model3_data...]
                model3_entry = [videoId, videoName, category, "non_target_videos"] + model3_data.tolist()
                model3_data_list.append(model3_entry)

    # Now `model1_data_list`, `model2_data_list`, and `model3_data_list` contain the data from each respective model
    print(f"Total videos processed for Avg: {len(model1_data_list)}")
    print(f"Total videos processed for Layer3: {len(model2_data_list)}")
    print(f"Total videos processed for Layer4: {len(model3_data_list)}")

    base_path = f"{target_path}\\Even"

    # Traverse through the directories
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Extract videoId from the folder name (assuming format: "videoID_{id}")
            videoId = folder_name.split("_")[-1]
            
            # Define the paths for the models and metadata files
            model1_file = os.path.join(folder_path, "R3D18-AvgPool-512")
            model2_file = os.path.join(folder_path, "R3D18-Layer3-512")
            model3_file = os.path.join(folder_path, "R3D18-Layer4-512")
            metadata_file = os.path.join(folder_path, "metadata.txt")
            
            # Check if all model files exist
            if os.path.exists(model1_file) and os.path.exists(model2_file) and os.path.exists(model3_file):
                # Load model data
                model1_data = torch.load(model1_file)
                model2_data = torch.load(model2_file)
                model3_data = torch.load(model3_file)
                
                # Initialize variables for videoName and category
                videoName = ""
                category = ""
                
                # Read the metadata.txt file if it exists
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            # Extract videoName from the first line
                            videoName = lines[0].strip().replace("Filename: ", "")
                        if len(lines) > 1:
                            # Extract category from the second line (optional)
                            category = lines[1].strip().replace("Category Label: ", "")
                
                # If category is not present, use an empty string
                if not category:
                    category = ""
                
                # Append the model1 data [videoId, videoName, category, "non_target_videos", model1_data...]
                model1_entry = [videoId, videoName, category, "target_videos"] + model1_data.tolist()
                model1_data_list.append(model1_entry)
                
                # Append the model2 data [videoId, videoName, category, "non_target_videos", model2_data...]
                model2_entry = [videoId, videoName, category, "target_videos"] + model2_data.tolist()
                model2_data_list.append(model2_entry)
                
                # Append the model3 data [videoId, videoName, category, "non_target_videos", model3_data...]
                model3_entry = [videoId, videoName, category, "target_videos"] + model3_data.tolist()
                model3_data_list.append(model3_entry)

    # Now `model1_data_list`, `model2_data_list`, and `model3_data_list` contain the data from each respective model
    print(f"Total videos processed for Avg: {len(model1_data_list)}")
    print(f"Total videos processed for Layer3: {len(model2_data_list)}")
    print(f"Total videos processed for Layer4: {len(model3_data_list)}")

    # Define the base path
    base_path = f"{target_path}\\Odd"

    # Traverse through the directories
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Extract videoId from the folder name (assuming format: "videoID_{id}")
            videoId = folder_name.split("_")[-1]
            
            # Define the paths for the models and metadata files
            model1_file = os.path.join(folder_path, "R3D18-AvgPool-512")
            model2_file = os.path.join(folder_path, "R3D18-Layer3-512")
            model3_file = os.path.join(folder_path, "R3D18-Layer4-512")
            metadata_file = os.path.join(folder_path, "metadata.txt")
            
            # Check if all model files exist
            if os.path.exists(model1_file) and os.path.exists(model2_file) and os.path.exists(model3_file):
                # Load model data
                model1_data = torch.load(model1_file)
                model2_data = torch.load(model2_file)
                model3_data = torch.load(model3_file)
                
                # Initialize variables for videoName and category
                videoName = ""
                category = ""
                
                # Read the metadata.txt file if it exists
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            # Extract videoName from the first line
                            videoName = lines[0].strip().replace("Filename: ", "")
                        if len(lines) > 1:
                            # Extract category from the second line (optional)
                            category = lines[1].strip().replace("Category Label: ", "")
                
                # If category is not present, use an empty string
                if not category:
                    category = ""
                
                # Append the model1 data [videoId, videoName, category, "non_target_videos", model1_data...]
                model1_entry = [videoId, videoName, category, "target_videos"] + model1_data.tolist()
                model1_data_list.append(model1_entry)
                
                # Append the model2 data [videoId, videoName, category, "non_target_videos", model2_data...]
                model2_entry = [videoId, videoName, category, "target_videos"] + model2_data.tolist()
                model2_data_list.append(model2_entry)
                
                # Append the model3 data [videoId, videoName, category, "non_target_videos", model3_data...]
                model3_entry = [videoId, videoName, category, "target_videos"] + model3_data.tolist()
                model3_data_list.append(model3_entry)

    # Now `model1_data_list`, `model2_data_list`, and `model3_data_list` contain the data from each respective model
    print(f"Total videos processed for Avg: {len(model1_data_list)}")
    print(f"Total videos processed for Layer3: {len(model2_data_list)}")
    print(f"Total videos processed for Layer4: {len(model3_data_list)}")


    # Define the headers for each model (assuming 512 features in each model)
    headers = ['videoId', 'videoName', 'label', 'category'] + [f'feature_{i}' for i in range(512)]

    # Define the paths to save the CSV files
    model1_csv_path = f"{output_folder}\\avgpool_data.csv"
    model2_csv_path = f"{output_folder}\\layer3_data.csv"
    model3_csv_path = f"{output_folder}\\layer4_data.csv"

    # Convert the model data lists into pandas DataFrames and save as CSV
    # Save Model 1 data
    model1_df = pd.DataFrame(model1_data_list)
    model1_df.to_csv(model1_csv_path, index=False, header=headers)

    # Save Model 2 data
    model2_df = pd.DataFrame(model2_data_list)
    model2_df.to_csv(model2_csv_path, index=False, header=headers)

    # Save Model 3 data
    model3_df = pd.DataFrame(model3_data_list)
    model3_df.to_csv(model3_csv_path, index=False, header=headers)

    print("Model data saved as CSV files for AvgPool, Layer3, and Layer4.")

process_dnn_features(non_target_path, target_path, Final_Task0a_Output_Folder)

def process_col_hist(non_target_path, target_path, output_folder):

    base_path = non_target_path

    # Arrays to store HOG and HOF data
    col_hist_data = []

    # Traverse through the directories
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Extract videoId from the folder name (assuming format: "videoID_{id}")
            videoId = folder_name.split("_")[-1]
            
            # Define the paths for HOG, HOF, and metadata files
            col_hist = os.path.join(folder_path, "COL-HIST/col_hist.npy")
            metadata_file = os.path.join(folder_path, "metadata.txt")
            
            # Check if both HOG and HOF files exist
            if os.path.exists(col_hist):
                # Load HOG and HOF data
                hist_data = np.load(col_hist)

                
                # Initialize variables for videoName and category
                videoName = ""
                category = ""
                
                # Read the metadata.txt file if it exists
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            # Extract videoName from the first line
                            videoName = lines[0].strip().replace("Filename: ", "")
                        if len(lines) > 1:
                            # Extract category from the second line (optional)
                            category = lines[1].strip().replace("Category Label: ", "")
                
                # If category is not present, use an empty string
                if not category:
                    category = ""
                
                # Append the HOG data [videoId, videoName, category, "non_target_videos", hog_array...]
                hist_entry = [videoId, videoName, category, "non_target_videos"] + hist_data.tolist()
                col_hist_data.append(hist_entry)
            

    # Now `hog_data_list` contains all HOG data and `hof_data_list` contains all HOF data
    print(f"Total Col Hist videos processed: {len(col_hist_data)}")

    # Define the base path
    base_path = f"{target_path}\\Even"

    # Traverse through the directories
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Extract videoId from the folder name (assuming format: "videoID_{id}")
            videoId = folder_name.split("_")[-1]
            
            # Define the paths for HOG, HOF, and metadata files
            col_hist = os.path.join(folder_path, "COL-HIST/col_hist.npy")
            metadata_file = os.path.join(folder_path, "metadata.txt")
            
            # Check if both HOG and HOF files exist
            if os.path.exists(col_hist):
                # Load HOG and HOF data
                hist_data = np.load(col_hist)

                
                # Initialize variables for videoName and category
                videoName = ""
                category = ""
                
                # Read the metadata.txt file if it exists
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            # Extract videoName from the first line
                            videoName = lines[0].strip().replace("Filename: ", "")
                        if len(lines) > 1:
                            # Extract category from the second line (optional)
                            category = lines[1].strip().replace("Category Label: ", "")
                
                # If category is not present, use an empty string
                if not category:
                    category = ""
                
                # Append the HOG data [videoId, videoName, category, "non_target_videos", hog_array...]
                hist_entry = [videoId, videoName, category, "target_videos"] + hist_data.tolist()
                col_hist_data.append(hist_entry)
            

    # Now `hog_data_list` contains all HOG data and `hof_data_list` contains all HOF data
    print(f"Total Col Hist videos processed: {len(col_hist_data)}")


    base_path = f"{target_path}\\Odd"

    # Traverse through the directories
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Extract videoId from the folder name (assuming format: "videoID_{id}")
            videoId = folder_name.split("_")[-1]
            
            # Define the paths for HOG, HOF, and metadata files
            col_hist = os.path.join(folder_path, "COL-HIST/col_hist.npy")
            metadata_file = os.path.join(folder_path, "metadata.txt")
            
            # Check if both HOG and HOF files exist
            if os.path.exists(col_hist):
                # Load HOG and HOF data
                hist_data = np.load(col_hist)

                
                # Initialize variables for videoName and category
                videoName = ""
                category = ""
                
                # Read the metadata.txt file if it exists
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            # Extract videoName from the first line
                            videoName = lines[0].strip().replace("Filename: ", "")
                        if len(lines) > 1:
                            # Extract category from the second line (optional)
                            category = lines[1].strip().replace("Category Label: ", "")
                
                # If category is not present, use an empty string
                if not category:
                    category = ""
                
                # Append the HOG data [videoId, videoName, category, "non_target_videos", hog_array...]
                hist_entry = [videoId, videoName, category, "target_videos"] + hist_data.tolist()
                col_hist_data.append(hist_entry)
            

    # Now `hog_data_list` contains all HOG data and `hof_data_list` contains all HOF data
    print(f"Total Col Hist videos processed: {len(col_hist_data)}")


    # Define the headers for each model (assuming 512 features in each model)
    headers = ['videoId', 'videoName', 'label', 'category'] + [f'feature_{i}' for i in range(576)]

    # Define the paths to save the CSV files
    model1_csv_path = f"{output_folder}\\col_hist_data.csv"

    # Convert the model data lists into pandas DataFrames and save as CSV
    # Save Model 1 data
    model1_df = pd.DataFrame(col_hist_data)
    model1_df.to_csv(model1_csv_path, index=False, header=headers)

    print("Model data saved as CSV files for COL HIST")

process_col_hist(non_target_path, target_path, Final_Task0a_Output_Folder)