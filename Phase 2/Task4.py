import os
import numpy as np
import pandas as pd
import csv
import cv2
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from collections import defaultdict
import torch 
import warnings
warnings.filterwarnings("ignore")

label = input("Enter the label: ")
method_mapping = {
    1: "lda",
    2: "pca",
    3: "svd",
    4: "k_means"
}

model_mapping = {
    1: "avgpool",
    2: "layer3",
    3: "layer4",
    4: "hog",
    5: "hof",
    6: "col_hist"
}
print(model_mapping)
model = int(input("Enter model from task 2 outputs: ")) # User Input
print(method_mapping)
method = int(input("Enter method from task 2 outputs: ")) # User Input
s = int(input("Enter s from task 2 outputs: ")) # User Input
m = int(input("Enter m: "))

# Getting videos from given label even if it is non-target -----------------------------
def getLabelVideos(label):
    # TODO1: Update location to dataset
    non = "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Assets\\hmdb51_org\\non_target_videos"
    label_folder = os.path.join(non, label)
    all_vids = []
    if not os.path.exists(label_folder):
        # TODO1: Update location to dataset
        tar = "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Assets\\hmdb51_org\\target_videos"
        label_folder = os.path.join(tar, label)
        if not os.path.exists(label_folder):
            print(f"No folder found for label '{label}'")
            return []
        # print(os.listdir(label_folder))
        if(len(os.listdir(label_folder)) == 1):
            label_folder = os.path.join(label_folder,label)
        video_names = [f for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]
        for i in video_names:
            vidid = findIdFromName(i)
            all_vids.append(vidid)
        return all_vids
    # print(os.listdir(label_folder))
    if(len(os.listdir(label_folder)) == 1):
        label_folder = os.path.join(label_folder,label)
    video_names = [f for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]
    for i in video_names:
        vidid = findIdFromName(i)
        all_vids.append(vidid)
    return all_vids

def findIdFromName(videoName):
    # TODO2: Change location to VideoID_Mapping
    file = "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\VideoID_Mapping.csv"
    with open(file,'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Filename'] == videoName:
                return int(row['VideoID'])
            
# Getting videos from given label even if it is non-target -----------------------------

# def find_vids_in_label(label, s, model, method):
#     latent_space_method = method_mapping[method]
#     latent_space_model = model_mapping[model]
#     #Path_Changes-------------
#     label_mapping = pd.read_csv(f"C:\\Users\\ompat\\Desktop\\everything else\\Multimedia and WebDatabases\\Phase 2\\Latent Semantics\\Task2\\{latent_space_model}\\{s}\\{latent_space_method}_reduced_all.csv")
#     all_vids = []
#     for _, row in label_mapping.iterrows():
#         if row['labels']==label:
#             all_vids.append(row['videoId'])
#     return all_vids

def relevance_score(label, s, model, method, m):
    latent_space_method = method_mapping[method]
    latent_space_model = model_mapping[model]
    # TODO3: Change location to Task2 Latent Semantic output folder
    label_mapping = pd.read_csv(f"E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Latent Semantics\\Task2\\{latent_space_model}\\{s}\\{latent_space_method}_reduced_all.csv")
    # TODO2: Change location to VideoID_Mapping
    videoid_namemapping = pd.read_csv("E:\\Coding\\MultimediaWebDatabases\\Phase 2\\VideoID_Mapping.csv")
    target_vids = label_mapping[label_mapping['category']=='target_videos']
    vids_in_label = getLabelVideos(label)

    relevant_vids = []

    # print("Entering loop?")

    for _, rows in target_vids.iterrows():
        target_id = rows['videoId']
        target_feature = rows[3:].values.reshape(1,-1)
        similarity_per_target_video = []

        for vids in vids_in_label:
            vid_row = label_mapping[label_mapping['videoId']==vids]
            vid_feature = vid_row.iloc[0,3:].values.reshape(1,-1)

            simi = cosine_similarity(target_feature, vid_feature)[0][0]
            similarity_per_target_video.append(simi)

            vid_name = videoid_namemapping[videoid_namemapping['VideoID']==int(target_id)]
            vidn = vid_name['Filename'].values[0]

            avg_sim = np.mean(similarity_per_target_video)
            relevant_vids.append((target_id,vidn,avg_sim))
    
    relevant_vids.sort(key=itemgetter(2),reverse=True)
    return relevant_vids[:m]

# For My Reference ---- Following function gives all the videoIds of videos under a specific input given label, use this to 
# compare all this videos to all videos under target category and find the most m most relevant target videos 

m_most_relevant_vids = relevance_score(label, s, model, method, m)
for vid in m_most_relevant_vids:
    print(f"VideoID: {vid[0]}, Filename: {vid[1]}, Relevance Score: {vid[2]}")

# Changes to make
# order in which input is taken -- done
# take non target video labels as input as well