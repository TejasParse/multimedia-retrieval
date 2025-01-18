import os
import numpy as np
import pandas as pd

file_choice = {
    0: "E:\\Coding\\MultimediaWebDatabases\\Phase 3\\Dataset\\avgpool_kmeans(300).csv",
    1: "E:\\Coding\\MultimediaWebDatabases\\Phase 3\\Dataset\\col_hist_svd(300).csv",
    2: "E:\\Coding\\MultimediaWebDatabases\\Phase 3\\Dataset\\layer4_pca(300).csv"
}

feature_count = {
    0: 300,
    1: 300,
    2: 300,
}

# print("0: Avgpool + KMeans + 300")
# print("1: Col Hist + SVD + 300")
# print("2: Layer4 + PCA + 300")

# model_input = int(input("Enter your choice: "))
# if(model_input < 0 or model_input > 2):
#     print("Invalid Choice")
#     exit()

# Path Changes ---------------------------------
data_path = "E:\\Coding\\MultimediaWebDatabases\\Phase 3\\Assets\\hmdb51_org\\target_videos"
latent_model = "E:\\Coding\\MultimediaWebDatabases\\Phase 3\\Dataset\\layer4_pca(300).csv"
id_name_mapping = "E:\\Coding\\MultimediaWebDatabases\\Phase 3\\VideoID_Mapping.csv"

# Load the files -------------------------------
features_csv = pd.read_csv(latent_model)
features = features_csv.drop(columns=["labels", "category"])
id_name_mapping = pd.read_csv(id_name_mapping)

# Dictionary for locating videos with name ------
id_features = {row['videoId']: row.iloc[1:].values for _, row in features.iterrows()}
id_name = {row['VideoID']: row['Filename'] for _, row in id_name_mapping.iterrows()}

label_dimensionality = {}

for label in os.listdir(data_path):
    # Following is for handling data and obtaining features of each video under each label ---------------
    label_path = os.path.join(data_path, label)
    if os.path.isdir(label_path):
        label_features = []
        for video_file in os.listdir(label_path):
            video_name = video_file
            video_id = next((k for k, v in id_name.items() if v==video_name), None)
            if video_id and video_id in id_features:
                label_features.append(id_features[video_id])
        if not label_features:
            print(f"Warning: No valid videos found for label: {label}")
        else:
            feat_mat = np.array(label_features)

            covar_mat = np.cov(feat_mat,rowvar=False)
            eigenvalues, _ = np.linalg.eig(covar_mat)
            sorted_eigenv = np.sort(eigenvalues)[::-1]
            cumulative_eig = np.cumsum(sorted_eigenv)/np.sum(sorted_eigenv)

            inherent_dimenionality = np.argmax(cumulative_eig>=0.95)+1
            label_dimensionality[label] = inherent_dimenionality

print("\nInherent dimensionality of each label:\n")
for label, dim in label_dimensionality.items():
    print(f"{label}: {dim}")