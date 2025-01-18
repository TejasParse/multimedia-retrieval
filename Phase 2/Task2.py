import numpy as np
import os
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import pickle
from sklearn.preprocessing import MinMaxScaler

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

def PCA(features, feature_count=20):

    X_meaned = features - np.mean(features, axis=0)

    cov_matrix = np.cov(X_meaned, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:, sorted_index]

    core = np.diag(sorted_eigenvalues[: feature_count])

    # X_pca = np.dot(X_meaned, sorted_eigenvectors)

    # X_pca_reduced = X_pca[:, :feature_count]

    # X_test = np.dot(X_meaned, sorted_eigenvectors[:, :feature_count])

    # print(X_test, X_pca_reduced)
    
    return sorted_eigenvectors[:,:feature_count], core

def get_reduced_PCA(factor_matrix, core_matrix, X):

    X_meaned = X - np.mean(X, axis=0)
    # print(X_meaned.shape, factor_matrix.shape)
    X_pca = np.dot(X_meaned, factor_matrix)

    return X_pca

# How to use
# factor, core = PCA(random_features, feature_count=202)
# print(factor.shape, core)
# get_reduced_PCA(factor, core, random_features)

def SVD(features, n_components):
    
    features_meaned = features - np.mean(features, axis=0)
    
    covariance_matrix = np.dot(features_meaned.T, features_meaned)

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    sorted_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_idx]
    sorted_eigenvectors = eigenvectors[:, sorted_idx]

    V_reduced = sorted_eigenvectors[:, :n_components]

    Sigma_reduced = np.sqrt(sorted_eigenvalues[:n_components])

    U_reduced = np.dot(features_meaned, V_reduced) / Sigma_reduced

    Sigma_matrix = np.diag(Sigma_reduced)
    
    return Sigma_matrix, U_reduced, V_reduced

def get_reduced_SVD(U_reduced, Sigma_matrix):

    reduced_data = np.dot(U_reduced, Sigma_matrix)
    
    return reduced_data

def LDA(features, n_topics, all_data):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    
    lda_transformed = lda.fit_transform(features)

    lda_all = lda.transform(all_data)
    
    return lda_transformed, lda_all

model_mapping = {
    1: "avgpool",
    2: "layer3",
    3: "layer4",
    4: "hog",
    5: "hof",
    6: "col_hist"
}

method_mapping = {
    1: "lda",
    2: "pca",
    3: "svd",
    4: "k_means"
}


# 0: AvgPool
# 1: Layer3
# 2: Layer4
# 3: HoG
# 4: HoF
# 5: COL-HIST

# TODO1: Change File Location
filesLocation = {
    1: "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\avgpool_data.csv",
    2: "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\layer3_data.csv",
    3: "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\layer4_data.csv",
    4: "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\hog_data.csv",
    5: "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\hof_data.csv",
    6: "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\col_hist_data.csv"
}

import pandas as pd

def pre_process_data(file_name, model_type):

    data = pd.read_csv(file_name)
    
    filtered_data = data[(data['category'] == 'target_videos') & (data['videoId'].astype(int) % 2 == 0)]
    
    if model_type in [1, 2, 3]:
        feature_cols = [f'feature_{i}' for i in range(512)]
    elif model_type in [5, 4]:
        feature_cols = [f'feature_{i}' for i in range(480)]
    elif model_type == 6:
        feature_cols = [f'feature_{i}' for i in range(576)]
    else:
        raise ValueError("Invalid model_type. It should be in the range 0-5.")
    
    features = filtered_data[feature_cols]
    video_ids = filtered_data['videoId'].values
    labels = filtered_data['label'].values
    category = filtered_data['category'].values
    # print(video_ids.shape, features.shape)
    
    return features, video_ids, labels, category

def process_complete_data(file_name, model_type):

    data = pd.read_csv(file_name)
    
    filtered_data = data
    
    if model_type in [1, 2, 3]:
        feature_cols = [f'feature_{i}' for i in range(512)]
    elif model_type in [5, 4]:
        feature_cols = [f'feature_{i}' for i in range(480)]
    elif model_type == 6:
        feature_cols = [f'feature_{i}' for i in range(576)]
    else:
        raise ValueError("Invalid model_type. It should be in the range 0-5.")
    
    features = filtered_data[feature_cols]
    video_ids = filtered_data['videoId'].values
    labels = filtered_data['label'].values
    category = filtered_data['category'].values
    # print(video_ids.shape, features.shape)
    
    return features, video_ids, labels, category


def save_matrix(data, model, s, matrix_name, method):

    # TODO2: Change File Location to save the matrices
    directory = f"E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Latent Semantics\\Task2\\{model_mapping[model]}\\{s}"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(directory, f"{method_mapping[method]}_{matrix_name}.npy")
    
    np.save(file_path, data)

def save_csv(reduced_features, videoId, labels, category, model, s, matrix_name, method):
    df_basic = pd.DataFrame({
        'videoId': videoId,
        'labels': labels,
        'category': category
    })  
    # print(reduced_features.shape, "Idhar")

    # Convert reduced_features to a DataFrame with appropriate column names
    df_features = pd.DataFrame(reduced_features, columns=[f'feature_{i+1}' for i in range(reduced_features.shape[1])])

    # Step 2: Concatenate the basic information (videoId, labels, category) with the reduced features
    df_final = pd.concat([df_basic, df_features], axis=1)

    directory = f"E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Latent Semantics\\Task2\\{model_mapping[model]}\\{s}"
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(directory, f"{method_mapping[method]}_{matrix_name}.csv")

    # Step 3: Save the DataFrame to a CSV file
    df_final.to_csv(file_path, index=False)

def print_features(reduced_feature, videoIds):

    n_latent_semantics = reduced_feature.shape[1]


    with open("task_2_output.txt", "w") as f:

        for i in range(n_latent_semantics):

            latent_semantic_weights = reduced_feature[:, i]
            
            video_weight_pairs = list(zip(videoIds, latent_semantic_weights))
            
            sorted_video_weight_pairs = sorted(video_weight_pairs, key=lambda x: x[1], reverse=True)
            
            f.write(f"\nLatent Semantic {i + 1} (ordered by weights):\n")
            for videoId, weight in sorted_video_weight_pairs:
                f.write(f"Video ID: {videoId}, Weight: {weight:.4f}\n")


def generate(model, s, method):
    
    data, video_ids, labels, category = pre_process_data(file_name=filesLocation[model], model_type=model)
    all_data, all_video_ids, all_labels, all_category = process_complete_data(file_name=filesLocation[model], model_type=model)
    # print(labels.shape, all_labels.shape)
    # print("1: LDA")
    # print("2: PCA")
    # print("3: SVD")
    # print("4: K-Means\n")

    if(method == 2):

        factor, core = PCA(features=data, feature_count=s)

        # Even Target
        x_reduced = get_reduced_PCA(factor_matrix=factor, core_matrix=core, X=data)
        print_features(x_reduced, video_ids)
        # save_matrix(x_reduced, model, s, matrix_name="reduced", method=method)
        # save_matrix(video_ids, model, s, matrix_name="reduced_videoId", method=method)

        save_csv(reduced_features=x_reduced, videoId=video_ids, labels=labels, category=category, model=model, s=s, matrix_name="reduced", method=method)

        # Caching all
        x_reduced = get_reduced_PCA(factor_matrix=factor, core_matrix=core, X=all_data)

        save_matrix(core, model, s, matrix_name="core", method=method)
        save_matrix(factor, model, s, matrix_name="factor", method=method)
        # save_matrix(x_reduced, model, s, matrix_name="reduced_all", method=method)
        # save_matrix(all_video_ids, model, s, matrix_name="reduced_videoId_all", method=method)

        save_csv(reduced_features=x_reduced, videoId=all_video_ids, labels=all_labels, category=all_category, model=model, s=s, matrix_name="reduced_all", method=method)


    elif(method == 3):

        core, u, v = SVD(features=data, n_components=s)

        # Even Target
        x_reduced = get_reduced_SVD(U_reduced=u, Sigma_matrix=core)
        # save_matrix(x_reduced, model, s, matrix_name="reduced", method=method)
        # save_matrix(video_ids, model, s, matrix_name="reduced_videoId", method=method)
        print_features(x_reduced, video_ids)
        save_csv(reduced_features=x_reduced, videoId=video_ids, labels=labels, category=category, model=model, s=s, matrix_name="reduced", method=method)

        # All videos
        x_reduced = np.dot(all_data, v)

        save_matrix(core, model, s, matrix_name='core', method=method)
        save_matrix(u, model, s, matrix_name='u', method=method)
        save_matrix(v, model, s, matrix_name='v', method=method)
        # save_matrix(x_reduced, model, s, matrix_name="reduced_all", method=method)
        # save_matrix(all_video_ids, model, s, matrix_name="reduced_videoId_all", method=method)

        save_csv(reduced_features=x_reduced, videoId=all_video_ids, labels=all_labels, category=all_category, model=model, s=s, matrix_name="reduced_all", method=method)

    elif(method == 1):

        if(model>=1 and model<=3):
            print("Min Max Scaler")
            max_value = np.max(data)
            scaler1 = MinMaxScaler(feature_range=(0, max_value))
            data = scaler1.fit_transform(data)

            max_value = np.max(all_data)
            scaler2 = MinMaxScaler(feature_range=(0, max_value))
            all_data = scaler2.fit_transform(all_data)
 

        x_reduced, x_all_reduced = LDA(features=data, n_topics=s, all_data=all_data)
    
        print_features(x_reduced, video_ids)

        save_csv(reduced_features=x_reduced, videoId=video_ids, labels=labels, category=category, model=model, s=s, matrix_name="reduced", method=method)

        # save_matrix(x_reduced, model, s, matrix_name="reduced", method=method)
        # save_matrix(video_ids, model, s, matrix_name="reduced_videoId", method=method)

        save_csv(reduced_features=x_all_reduced, videoId=all_video_ids, labels=all_labels, category=all_category, model=model, s=s, matrix_name="reduced_all", method=method)

        # save_matrix(x_all_reduced, model, s, matrix_name="reduced_all", method=method)
        # save_matrix(all_video_ids, model, s, matrix_name="reduced_videoId_all", method=method)

    elif(method == 4):
        
        kmeans_model = KMeans(n_clusters=s, random_state=1)
        x_reduced = kmeans_model.fit_transform(data)
        print_features(x_reduced, video_ids)
        save_csv(reduced_features=x_reduced, videoId=video_ids, labels=labels, category=category, model=model, s=s, matrix_name="reduced", method=method)
        # save_matrix(x_reduced, model, s, matrix_name="reduced", method=method)
        # save_matrix(video_ids, model, s, matrix_name="reduced_videoId", method=method)

        directory = f"E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Latent Semantics\\Task2\\{model_mapping[model]}\\{s}"
    
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        file_path = os.path.join(directory, f"{method_mapping[method]}_weights.pkl")

        with open(file_path, 'wb') as file:
            pickle.dump(kmeans_model, file)

        x_reduced = kmeans_model.transform(all_data)

        # save_matrix(x_reduced, model, s, matrix_name="reduced_all", method=method)
        # save_matrix(all_video_ids, model, s, matrix_name="reduced_videoId_all", method=method)
        save_csv(reduced_features=x_reduced, videoId=all_video_ids, labels=all_labels, category=all_category, model=model, s=s, matrix_name="reduced_all", method=method)
     

def inputFunc():

    print("\nTask 2\n\n")
    print("Pick a Feature Model from the menu")
    print("1: AvgPool")
    print("2: Layer3")
    print("3: Layer4")
    print("4: HOG")
    print("5: HOF")
    print("6: COL-HIST\n")
    model = input("Pick a feature model: ")
    model = int(model)

    print("Now Pick the top-s features to be reduced to: ")
    s = input("Input: ")
    s = int(s)

    print("\nPick a Dimensionality Reduction Technique from the menu: ")
    print("1: LDA")
    print("2: PCA")
    print("3: SVD")
    print("4: K-Means\n")
    method = input("Pick a technique: ")
    method = int(method)
    return model, s, method

import random

def generate_model_parameters():

    model = random.randint(1, 6)
    
    s = random.randint(100, 400)
    
    method = random.randint(2, 4)
    
    return model, s, method

# User Input
model, s, method = inputFunc()
generate(model, s, method)

# Generate Randomly
# for i in range(100):

#     model, s, method = generate_model_parameters()
#     print(model_mapping[model], s, method_mapping[method])

#     generate(model, s, method)