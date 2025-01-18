import numpy as np
import pandas as pd
import os
import warnings
import pickle
from collections import defaultdict
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
from sklearn.metrics.pairwise import cosine_similarity

# TODO1: Change the main directory
main_dir = "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Latent Semantics\\Task5"
if not os.path.exists(main_dir):
    os.makedirs(main_dir)
#
# TODO2: Change the Latent Semantics File
latent_directory = "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Latent Semantics"
# TODO3: Change the location to database
non_target_videos = "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Assets\\hmdb51_org\\non_target_videos"
target_videos = "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Assets\\hmdb51_org\\target_videos"

# TODO3: Change Task0a output locations
filesLocation = {
    "avgpool_data.csv": "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\avgpool_data.csv", #change path to your liking in the second field
    "layer3_data.csv": "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\layer3_data.csv",
    "layer4_data.csv": "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\layer4_data.csv",
    "hog_data.csv": "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\hog_data.csv",
    "hof_data.csv": "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\hof_data.csv",
    "col_hist_data.csv": "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\col_hist_data.csv"
}

def load_feature_vectors(file_path):
    df = pd.read_csv(filesLocation[file_path])
    return df
def getLabelVideos(label):
        # TODO2: Change the location to database
        non = non_target_videos
        label_folder = os.path.join(non, label)
        if not os.path.exists(label_folder):
            # TODO2: Change the location to database
            tar = target_videos
            label_folder = os.path.join(tar, label)
            if not os.path.exists(label_folder):
                print(f"No folder found for label '{label}'")
                return []
            # print(os.listdir(label_folder))
            if(len(os.listdir(label_folder)) == 1):
                label_folder = os.path.join(label_folder,label)
            video_names = [f for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]
            return video_names
        # print(os.listdir(label_folder))
        if(len(os.listdir(label_folder)) == 1):
            label_folder = os.path.join(label_folder,label)
        video_names = [f for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]
        return video_names
def getLabelRepresentative(labelVideos,model):
    df = pd.read_csv(filesLocation[model])
    df_filtered = df[df['videoName'].isin(labelVideos)]
    feature_columns = [col for col in df_filtered.columns if col.startswith('feature') or col.startswith('hof') or col.startswith('hog')]
    average_feature = df_filtered[feature_columns].mean().values
    return average_feature

def PCA(features, feature_count=20):

    X_meaned = features - np.mean(features, axis=0)

    cov_matrix = np.cov(X_meaned, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:, sorted_index]

    core = np.diag(sorted_eigenvalues[: feature_count])
    
    return sorted_eigenvectors[:,:feature_count], core

def get_reduced_PCA(factor_matrix, core_matrix, X):

    X_meaned = X - np.mean(X, axis=0)
    # print(X_meaned.shape, factor_matrix.shape)
    X_pca = np.dot(X_meaned, factor_matrix)

    return X_pca

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
def task2(model,s,method,label,l):

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

    # TODO3: Change Task0a output locations
    filesLocation1 = {
        1: "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\avgpool_data.csv", 
        2: "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\layer3_data.csv",
        3: "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\layer4_data.csv",
        4: "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\hog_data.csv",
        5: "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\hof_data.csv",
        6: "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\col_hist_data.csv"
    }


    def pre_process_data(file_name, model_type):

        data = pd.read_csv(file_name)
        
        filtered_data = data[(data['category'] == 'target_videos') & (data['videoId'].astype(int) % 2 == 0)]
        
        if model_type in [1, 2, 3]:
            feature_cols = [f'feature_{i}' for i in range(512)]
        elif model_type == 4:
            feature_cols = [f'hog_{i}' for i in range(480)]
        elif model_type == 5:
            feature_cols = [f'hof_{i}' for i in range(480)]
        elif model_type == 6:
            feature_cols = [f'feature_{i}' for i in range(576)]
        else:
            raise ValueError("Invalid model_type. It should be in the range 0-5.")
        
        # print("\n\n\n\n ",filtered_data)
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
        elif model_type == 4:
            feature_cols = [f'hog_{i}' for i in range(480)]
        elif model_type == 5:
            feature_cols = [f'hof_{i}' for i in range(480)]
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
        directory = f"{latent_directory}\\Task2\\{model_mapping[model]}\\{s}"
        
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

        directory = f"{latent_directory}\\Task2\\{model_mapping[model]}\\{s}"
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        file_path = os.path.join(directory, f"{method_mapping[method]}_{matrix_name}.csv")

        # Step 3: Save the DataFrame to a CSV file
        df_final.to_csv(file_path, index=False)


        
    def generate(model, s, method):
        
        data, video_ids, labels, category = pre_process_data(file_name=filesLocation1[model], model_type=model)
        all_data, all_video_ids, all_labels, all_category = process_complete_data(file_name=filesLocation1[model], model_type=model)
        # print(labels.shape, all_labels.shape)
        # print("1: LDA")
        # print("2: PCA")
        # print("3: SVD")
        # print("4: K-Means\n")

        if(method == 2):

            factor, core = PCA(features=data, feature_count=s)

            # Even Target
            x_reduced = get_reduced_PCA(factor_matrix=factor, core_matrix=core, X=data)
            # print_features(x_reduced, video_ids)
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
            # TODO: For LDA


            lda = LatentDirichletAllocation(n_components=s, random_state=42) 
            print("here1") 
            x_reduced = lda.fit_transform(data)
            print("here2")
            x_all_reduced = lda.transform(all_data)    
            directory = f"{latent_directory}\\Task2\\{model_mapping[model]}\\{s}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_path = os.path.join(directory, "lda_weights.pkl")
            with open(file_path, 'wb') as file:
                pickle.dump(lda, file)

            save_csv(reduced_features=x_reduced, videoId=video_ids, labels=labels, category=category, model=model, s=s, matrix_name="reduced", method=method)
            save_csv(reduced_features=x_all_reduced, videoId=all_video_ids, labels=all_labels, category=all_category, model=model, s=s, matrix_name="reduced_all", method=method)

        elif(method == 4):
            
            kmeans_model = KMeans(n_clusters=s, random_state=1)
            x_reduced = kmeans_model.fit_transform(data)
            save_csv(reduced_features=x_reduced, videoId=video_ids, labels=labels, category=category, model=model, s=s, matrix_name="reduced", method=method)
            directory = f"{latent_directory}\\Task2\\{model_mapping[model]}\\{s}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            file_path = os.path.join(directory, f"{method_mapping[method]}_weights.pkl")

            with open(file_path, 'wb') as file:
                pickle.dump(kmeans_model, file)

            x_reduced = kmeans_model.transform(all_data)

            save_csv(reduced_features=x_reduced, videoId=all_video_ids, labels=all_labels, category=all_category, model=model, s=s, matrix_name="reduced_all", method=method)
        
    generate(model,s,method)
    def getGenerated(model,s,method):
        #paath
        evenPath = f"{latent_directory}\\Task2\\{model_mapping[model]}\\{s}\\{method_mapping[method]}_reduced.csv"
        allPath = f"{latent_directory}\\Task2\\{model_mapping[model]}\\{s}\\{method_mapping[method]}_reduced_all.csv"
        even = pd.read_csv(evenPath)
        all = pd.read_csv(allPath)
        return even , all
    even , all = getGenerated(model,s,method)
    def getLabelVideos(label):
        non = non_target_videos
        label_folder = os.path.join(non, label)
        if not os.path.exists(label_folder):
            tar = target_videos
            label_folder = os.path.join(tar, label)
            if not os.path.exists(label_folder):
                print(f"No folder found for label '{label}'")
                return []
            # print(os.listdir(label_folder))
            if(len(os.listdir(label_folder)) == 1):
                label_folder = os.path.join(label_folder,label)
            video_names = [f for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]
            return video_names
        # print(os.listdir(label_folder))
        if(len(os.listdir(label_folder)) == 1):
            label_folder = os.path.join(label_folder,label)
        video_names = [f for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]
        return video_names
    labelVideos = getLabelVideos(label)
    def getLabelRepresentative(labelVideos,model):
        df = pd.read_csv(filesLocation[selected_model])
        df_filtered = df[df['videoName'].isin(labelVideos)]
        feature_columns = [col for col in df_filtered.columns if col.startswith('feature') or col.startswith('hof') or col.startswith('hog')]
        average_feature = df_filtered[feature_columns].mean().values
        return average_feature
    average = getLabelRepresentative(labelVideos,model)
    def transform(average,model,s,method):
        if method == 1: #lda
            with open(f'{latent_directory}\\Task2\\{model_mapping[model]}\\{s}\\lda_weights.pkl', 'rb') as file:
                loaded_lda = pickle.load(file)
            transformed = loaded_lda.transform(average.reshape(1,-1))
        elif method == 2: #pca
            factor = np.load(f'{latent_directory}\\Task2\\{model_mapping[model]}\\{s}\\pca_factor.npy')
            transformed = np.dot(average,factor)
            transformed = transformed.reshape(1,-1)
        elif method == 3: #svd
            v = np.load(f'{latent_directory}\\Task2\\{model_mapping[model]}\\{s}\\svd_v.npy')
            transformed = np.dot(average,v)
            transformed = transformed.reshape(1,-1)
        elif method == 4: #kmeans
            with open(f'{latent_directory}\\Task2\\{model_mapping[model]}\\{s}\\k_means_weights.pkl', 'rb') as file:
                loaded_kmeans_model = pickle.load(file)
            transformed = loaded_kmeans_model.transform(average.reshape(1,-1))
        return transformed
    transformed  = transform(average,model,s,method)
    def findSimilarity(target_videos,query_features,l):
        #query_features should be reshaped (1,-1)
        similarities_with_labels = []
        for _, row in target_videos.iterrows():
            target_features = row.iloc[3:].values
            similarity = cosine_similarity(query_features, target_features.reshape(1, -1))[0][0]
            similarities_with_labels.append((row['labels'], similarity))
        label_stats = defaultdict(lambda: {'similarity_sum': 0.0, 'count': 0})
        # Iterate over the similarities_with_labels list
        for label, similarity in similarities_with_labels:
            label_stats[label]['similarity_sum'] += similarity
            label_stats[label]['count'] += 1
        # Convert the dictionary into a list and calculate average similarity for each label
        groupedList = [(label, stats['similarity_sum'] / stats['count'], stats['count']) for label, stats in label_stats.items()]

        # Sort the groupedList by average similarity in descending order
        sorted_groupedList = sorted(groupedList, key=lambda x: x[1], reverse=True)
        print(f"printing top {l} similar labels : ")
        for label,score,_ in sorted_groupedList[:int(l)]:
            print(f"label : {label} |  Score : {score}")
    findSimilarity(even,transformed,l)

def createLatent(selected_model):
    if(selected_model == "layer3_data.csv"): name = "layer3"
    if(selected_model == "layer4_data.csv"): name = "layer4"
    if(selected_model == "avgpool_data.csv"): name = "avgpool"
    if(selected_model == "hog_data.csv"): name = "hog"
    if(selected_model == "hof_data.csv"): name = "hof"
    if(selected_model == "col_hist_data.csv"): name = "col_hist"
    def createSimilarityMatrix(selected_model):
        data = pd.read_csv(filesLocation[selected_model])

        even_videos = data[data['videoId'] % 2 == 0]

        even_target_videos = even_videos[even_videos['category'] == 'target_videos']
        numeric_columns = even_target_videos.columns[even_target_videos.columns.str.contains('feature|hof|hog')]

        label_features = even_target_videos.groupby('label')[numeric_columns].mean()

        similarity_matrix = cosine_similarity(label_features)
        labels = label_features.index
        similarity_df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
        similarity_df.to_csv(os.path.join(main_dir,'label_label_similarity_matrix'+'_'+selected_model))
        return labels,similarity_df
    
    labels,similarity_df = createSimilarityMatrix(selected_model)
    print("Enter the dimensionality reduction technique you want to use from the following:")
    print("1: PCA")
    print("2: SVD")
    print("3: LDA")
    print("4: KMeans")
    val = int(input())
    model = ""
    s = int(input("Enter the number of components (s): "))

    if val == 1:
        factor,core = PCA(similarity_df.values,s)
        reduced_matrix = get_reduced_PCA(factor,core,similarity_df.values)
        reduced_matrix = pd.DataFrame(reduced_matrix,index=labels)  # Convert to DataFrame
        model = "PCA"
        np.savetxt(os.path.join(main_dir, f"latent_semantics_{model}_{s}_factor"+'_'+selected_model), factor, delimiter=',')
        output_filename = os.path.join(main_dir, f"latent_semantics_{model}_{s}_components"+'_'+selected_model)
        reduced_matrix.to_csv(output_filename)
    elif val == 2:
        core, u, v  = SVD(similarity_df.values, s)
        reduced_matrix = get_reduced_SVD(u, core)
        reduced_matrix = pd.DataFrame(reduced_matrix,index=labels)  # Convert to DataFrame
        model = "SVD"
        np.savetxt(os.path.join(main_dir, f"latent_semantics_{model}_{s}_U"+'_'+selected_model), u, delimiter=',')
        np.savetxt(os.path.join(main_dir, f"latent_semantics_{model}_{s}_core"+'_'+selected_model), core, delimiter=',')
        np.savetxt(os.path.join(main_dir, f"latent_semantics_{model}_{s}_V"+'_'+selected_model), v, delimiter=',')
        output_filename = os.path.join(main_dir, f"latent_semantics_{model}_{s}_components"+'_'+selected_model)
        reduced_matrix.to_csv(output_filename)
    elif val == 3:
        lda = LatentDirichletAllocation(n_components=s, random_state=42) 
        print("here1")
        max_value = np.max(similarity_df)
        scaler = MinMaxScaler(feature_range=(0,max_value))
        similarity_df = scaler.fit_transform(similarity_df) 
        reduced_matrix = lda.fit_transform(similarity_df,s)
        reduced_matrix = pd.DataFrame(reduced_matrix, index=labels)
        directory = f"{latent_directory}\\Task5"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"lda_weights_{s}_{name}.pkl")
        with open(file_path, 'wb') as file:
            pickle.dump(lda, file)
        model = "LDA"
        output_filename = os.path.join(main_dir, f"latent_semantics_{model}_{s}_components"+'_'+selected_model)
        reduced_matrix.to_csv(output_filename)
    elif val == 4:
        kmeans_model = KMeans(n_clusters=s,random_state=1)
        reduced_matrix = kmeans_model.fit_transform(similarity_df)
        reduced_matrix = pd.DataFrame(reduced_matrix, index=labels)
        model = "KMeans"
        directory = f"{latent_directory}\\Task5"
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"kmeans_weights_{s}_{name}.pkl")
        with open(file_path, 'wb') as file:
            pickle.dump(kmeans_model, file)
        output_filename = os.path.join(main_dir, f"latent_semantics_{model}_{s}_components"+'_'+selected_model)
        reduced_matrix.to_csv(output_filename)

    return model,s

def task5(selected_model,l,label):
    # Function to load latent semantics based on selected model and components
    def get_latent_semantics(model, s):
        # TODO_changepath
        
        # Construct the filename based on the model and s value
        filename = f"latent_semantics_{model}_{s}_components_{selected_model}"
        
        # Create the full path
        file_path = os.path.join(main_dir, filename)
        
        # Check if the file exists
        if os.path.exists(file_path):
            reduced_matrix = pd.read_csv(file_path, index_col=0)  
            print(f"Loaded latent semantics from {file_path}")
            return reduced_matrix
        else:
            # If the file does not exist, raise an error
            raise FileNotFoundError(f"File {filename} not found in {main_dir}")
    def transform(average,selected_model,s,method):
        if(selected_model == "layer3_data.csv"): name = "layer3"
        if(selected_model == "layer4_data.csv"): name = "layer4"
        if(selected_model == "avgpool_data.csv"): name = "avgpool"
        if(selected_model == "hog_data.csv"): name = "hog"
        if(selected_model == "hof_data.csv"): name = "hof"
        if(selected_model == "col_hist_data.csv"): name = "col_hist"
        if method == "LDA": #lda
            with open(f'{latent_directory}\\Task5\\lda_weights_{s}_{name}.pkl', 'rb') as file:
                loaded_lda = pickle.load(file)
            transformed = loaded_lda.transform(average.reshape(1,-1))
        elif method == "PCA": #pca
            factor = pd.read_csv(f'{latent_directory}\\Task5\\latent_semantics_PCA_{s}_factor_{selected_model}',header=None)
            transformed = np.dot(average,factor.values)
            transformed = transformed.reshape(1,-1)
        elif method == "SVD": #svd
            v = pd.read_csv(f'{latent_directory}\\Task5\\latent_semantics_SVD_{s}_V_{selected_model}')
            transformed = np.dot(average,v.values)
            transformed = transformed.reshape(1,-1)
        elif method == "KMeans": #kmeans
            with open(f'{latent_directory}\\Task5\\kmeans_weights_{s}_{name}.pkl', 'rb') as file:
                loaded_kmeans_model = pickle.load(file)
            transformed = loaded_kmeans_model.transform(average.reshape(1,-1))
        return transformed

    # Function to compute cosine similarity between a given label and other target labels
    def find_most_similar_labels(label, latent_matrix, l,s,method):
        if label not in latent_matrix.index:
            data = pd.read_csv(filesLocation[selected_model])
            even_videos = data[data['videoId'] % 2 == 0]
            even_target_videos = even_videos[even_videos['category'] == 'target_videos']
            numeric_columns = even_target_videos.columns[even_target_videos.columns.str.contains('feature|hof|hog')]
            label_features = even_target_videos.groupby('label')[numeric_columns].mean()
            videoNames = getLabelVideos('drink')
            average = getLabelRepresentative(videoNames,selected_model)
            query = average.reshape(1,-1)
            similarity_vector = cosine_similarity(query, label_features)
            transformed  = transform(similarity_vector,selected_model,s,method)
            label_vector = transformed
        # Extract the vector for the given label
        else:
            label_vector = latent_matrix.loc[label].values.reshape(1, -1)
        
        # Compute cosine similarities between the label vector and all other label vectors
        similarity_scores = cosine_similarity(label_vector, latent_matrix.values).flatten()
        
        similarity_df = pd.DataFrame({'label': latent_matrix.index, 'similarity': similarity_scores})
        
        # Sort by similarity scores in descending order and exclude the original label itself
        most_similar = similarity_df.sort_values(by='similarity', ascending=False)
        
        # Return the top l most similar labels along with their scores
        return most_similar.head(int(l))
    model,s = createLatent(selected_model)
    # Load the selected latent semantics matrix
    latent_matrix = get_latent_semantics(model, s)

    # Find the most similar labels for the given label
    try:
        similar_labels = find_most_similar_labels(label, latent_matrix, l,s,model)
        print(f"\nTop {l} most similar target video labels to '{label}':")
        for idx, row in similar_labels.iterrows():
            print(f"{row['label']} - Similarity score: {row['similarity']:.4f}")
    except ValueError as e:
        print(e)


def feature_menu():
    print("\n\nEnter feature model to use : ")
    print("1. R3D18-Layer3-512")
    print("2. R3D18-Layer4-512")
    print("3. R3D18-AvgPool-512")
    print("4. col - hist")
    print("5. BOF-HOF-480")
    print("6. BOF-HOG-480")
    print("7. Exit")
    c = input("Enter a number (1-6): ")

    if c == '1':
        print("You selected: R3D18-Layer3-512")
        selected_model = "layer3_data.csv"

    elif c == '2':
        print("You selected: R3D18-Layer4-512")
        selected_model = "layer4_data.csv"

    elif c == '3':
        print("You selected: R3D18-AvgPool-512")
        selected_model = "avgpool_data.csv"

    elif c == '4':
        print("You selected: col - hist")
        selected_model = "col_hist_data.csv"

    elif c == '5':
        print("You selected: BOF-HOF-480")
        selected_model = "hof_data.csv"
    elif c == '6':
        print("You selected: BOF-HOG-480")
        selected_model = "hog_data.csv"
    else:
        print("Invalid choice. Please select a number between 1 and 6.")
    return selected_model

# Function to compute cosine similarity and return the top similar labels
def find_similar_labels(features, query_vector, top_n=5):
    # Compute cosine similarity between the query vector and all features
    similarity = cosine_similarity([query_vector], features)
    
    # Get the top N most similar items
    top_indices = np.argsort(similarity[0])[::-1][:top_n]
    return top_indices, similarity[0][top_indices]

label = input("Enter a label : ")
l = input("Enter l : ")
selected_model = feature_menu()
choice = int(input("\n\nWhat would you like to use ? \n 1 : Feature Space (from task 0) \n 2 : Latent Semantics\n"))
if(choice == 1):
    df = load_feature_vectors(selected_model)
    labelVideos = getLabelVideos(label)
    query_features = getLabelRepresentative(labelVideos,selected_model)
    # query_features = df[df['label'] == label].iloc[:, 4:].values[0]  # Assuming features start from 5th column
    
    target_videos = df[(df['category'] == 'target_videos') & (df['videoId'] % 2 == 0)]
    # Find similar labels based on the feature vectors
    similarities_with_labels = []
    for _, row in target_videos.iterrows():
        target_features = row.iloc[4:].values
        similarity = cosine_similarity(query_features.reshape(1, -1), target_features.reshape(1, -1))[0][0]
        similarities_with_labels.append((row['label'], similarity))
    label_stats = defaultdict(lambda: {'similarity_sum': 0.0, 'count': 0})

    # Iterate over the similarities_with_labels list
    for label, similarity in similarities_with_labels:
        label_stats[label]['similarity_sum'] += similarity
        label_stats[label]['count'] += 1

    # Convert the dictionary into a list and calculate average similarity for each label
    groupedList = [(label, stats['similarity_sum'] / stats['count'], stats['count']) for label, stats in label_stats.items()]

    # Sort the groupedList by average similarity in descending order
    sorted_groupedList = sorted(groupedList, key=lambda x: x[1], reverse=True)
    
    print(f"printing top {l} similar labels : ")
    for label,score,_ in sorted_groupedList[:int(l)]:
        print(f"label : {label} |  Score : {score}")
elif choice == 2:
    print("\nWhich Latent Semantics would you like to use ? ")
    print("1 : From Task 2")
    print("2 : From Task 5")
    ch = int(input())
    if(ch == 1):
        s = int(input("Enter s : "))
        print("Enter Dimensionality reduction methhod")
        print("1 : LDA")
        print("2 : PCA")
        print("3 : SVD")
        print("4 : Kmeans")
        method = int(input("Select any one : "))
        if(selected_model == "layer3_data.csv"): model = 2
        if(selected_model == "layer4_data.csv"): model = 3
        if(selected_model == "avgpool_data.csv"): model = 1
        if(selected_model == "hog_data.csv"): model = 4
        if(selected_model == "hof_data.csv"): model = 5
        if(selected_model == "col_hist_data.csv"): model = 6
        task2(model,s,method,label,l)
    elif(ch == 2):  
        task5(selected_model,l,label)
        

    #find similar labels using selected latent semantics

