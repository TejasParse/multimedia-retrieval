import os
import csv
import numpy as np
import pandas as pd
import cv2
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from collections import defaultdict
import torch 
import warnings
warnings.filterwarnings("ignore")

def findIdFromName(videoName):
    # TODO1: Change the VideoID_Mapping file location
    file = "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\VideoID_Mapping.csv"
    with open(file,'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Filename'] == videoName:
                return int(row['VideoID'])

def get_matrices(method, s, model):
    # TODO2: Change the folder where Task2 Latent Semantics are stored
    directory = f"E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Latent Semantics\\Task2\\{model}\\{s}"

    reduced = os.path.join(directory, f'{method}_reduced_all.csv')

    mat1 = pd.read_csv(reduced).iloc[:,3:].to_numpy()
    mat2 = pd.read_csv(reduced)['videoId'].to_numpy()

    return mat1, mat2

def clamp_similarity(value):  # Used this function to ignore floating point precision error
    return max(min(value, 1.0), -1.0)

def load_the_features_of_video(the_model,video_id):  # the_model corresponds to csv file of each feature space
    data = pd.read_csv(the_model)
    particular_video = data[data['videoId']==video_id]  # find the particular video based on ID
    features_of_that_video = particular_video.iloc[0,4:].to_numpy() # features start from 5th column till end
    return features_of_that_video

def find_m_similar_videos_in_latent(model, method, s, chosen_video, m):

    # TODO1: Change the VideoID_Mapping file location
    mapping = pd.read_csv(r"E:\\Coding\\MultimediaWebDatabases\\Phase 2\\VideoID_Mapping.csv")
    id2name_map = dict(zip(mapping['VideoID'],mapping['Filename']))

    # TODO4: Update the location to Task 0 output, avgpool_data.csv file
    temp = pd.read_csv("E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\avgpool_data.csv") 
    target_vid_ids = set(temp[temp['category']=='target_videos']['videoId'].to_numpy())

    reduced, reduced_videoId = get_matrices(method=method_mapping[method], s=s, model=model_mapping[model])

    chosen_video_index = np.where(reduced_videoId == chosen_video)[0][0]
    chosen_video_latents = reduced[chosen_video_index].reshape(1,-1)
    similarity_scores = []

    for i,video_id in enumerate(reduced_videoId):
        if video_id in target_vid_ids:
            other_video_latents = reduced[i].reshape(1,-1)
            dist_btw_both = cosine_similarity(chosen_video_latents,other_video_latents)[0][0]
            dist_btw_both = clamp_similarity(dist_btw_both)

            get_name = id2name_map.get(video_id, "Unknown")

            similarity_scores.append((video_id, get_name,dist_btw_both))  # figure out a way to add names

    similarity_scores.sort(key=itemgetter(2),reverse=True)

    return similarity_scores[:m] 

def find_m_similar_videos_in_feature(the_model, chosen_video, m):
    
    # loading the selected video features and rest of video features
    features_of_chosen = load_the_features_of_video(the_model, chosen_video)
    full_data = pd.read_csv(the_model)
    full_data = full_data[full_data['category']=='target_videos']
    
    similarity_scores = []
    
    for _, row in full_data.iterrows():
        second_video = row['videoId']
        feature_of_second_video = row[4:].to_numpy()
        
        dist_btw_both = cosine_similarity(features_of_chosen.reshape(1,-1), feature_of_second_video.reshape(1,-1))[0][0]
        dist_btw_both = clamp_similarity(dist_btw_both)
        
        similarity_scores.append((second_video,row['videoName'],dist_btw_both))
        
    similarity_scores.sort(key=itemgetter(2),reverse=True)
    
    return similarity_scores[:m]

def play_m_videos(list_of_m_videos,m):
    index = 0
    while True:
        video_name = list_of_m_videos[index][1]
        # TODO3: Change the path to the dataset
        dataset = r"E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Assets\\hmdb51_org"
        for root, dirs, files in os.walk(dataset):
            for file in files:
                if(video_name==file):
                    video_path = os.path.join(root,file)
                    break
        
        print("playing video: ")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Missing Video here!")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            title = f"Rank: {index+1}, ID: {list_of_m_videos[index][0]}"
            cv2.putText(frame, title, (10,30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('Video', frame)
            key = cv2.waitKey(25) & 0xFFF
            if key == ord('s'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('d'):
                index = (index+1)%m
                break
            elif key == ord('a'):
                index = (index-1)%m
                break   

video = input("Enter video ID or name: ")
if video.isdigit():
    videoid = int(video)
else:
    videoid = findIdFromName(video)
#Path_Changes-------------
path = f"E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0"
zero_one = int(input("Enter:\n0: Feature Model\n1: Latent Semantics\n"))
m = int(input("Enter m: "))
if(zero_one==0):
    featurespace = str(input("Enter feature model generated from task 0 ['avgpool', 'layer3', 'layer4', 'hog', 'hof', 'col_hist']: ")) + "_data.csv"
    trial = find_m_similar_videos_in_feature(path+"\\"+featurespace,videoid,m)
elif(zero_one==1):
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
    trial = find_m_similar_videos_in_latent(model, method, s, videoid, m)

print(trial)
play_m_videos(trial,m)


# Changes to make
# order in which input is taken
# take filename as input