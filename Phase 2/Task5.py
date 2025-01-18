import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import os
import pickle
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')
# CREATING LABEL-LABEL SIMILARITY MATRIX AND SAVING
# TODO2: Change the output directory where the factor and core matrices will be saved
main_dir = "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Latent Semantics\\Task5"  
if not os.path.exists(main_dir):
    os.makedirs(main_dir)

# TODO1: Change file location to each of the 6 Task0a output files invidually
filesLocation = {
    "avgpool_data.csv": "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\avgpool_data.csv", #change path to your liking in the second field
    "layer3_data.csv": "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\layer3_data.csv",
    "layer4_data.csv": "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\layer4_data.csv",
    "hog_data.csv": "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\hog_data.csv",
    "hof_data.csv": "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\hof_data.csv",
    "col_hist_data.csv": "E:\\Coding\\MultimediaWebDatabases\\Phase 2\\Task0\\col_hist_data.csv"
}

print("Enter feature model to use : ")
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
    selected_model = "layer3_data.csv"

elif choice == '2':
    print("You selected: R3D18-Layer4-512")
    selected_model = "layer4_data.csv"

elif choice == '3':
    print("You selected: R3D18-AvgPool-512")
    selected_model = "avgpool_data.csv"

elif choice == '4':
    print("You selected: col - hist")
    selected_model = "col_hist_data.csv"

elif choice == '5':
    print("You selected: BOF-HOF-480")
    selected_model = "hof_data.csv"
elif choice == '6':
    print("You selected: BOF-HOG-480")
    selected_model = "hog_data.csv"
else:
    print("Invalid choice. Please select a number between 1 and 6.")
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
    print("Label-label similarity matrix saved to : ",os.path.join(main_dir,'label_label_similarity_matrix.csv'))
    return labels,similarity_df

labels,similarity_df = createSimilarityMatrix(selected_model)



# FUNCTIONS FOR DIMENSIONALITY REDUCTION

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


# USER INPUT AND SAVING LATENT SEMANTICS
print("Enter the dimensionality reduction technique you want to use from the following:")
print("1: PCA")
print("2: SVD")
print("3: LDA")
print("4: KMeans")
val = int(input())

s = int(input("Enter the number of components (s): "))

if val == 1:
    factor,core = PCA(similarity_df.values,s)
    reduced_matrix = get_reduced_PCA(factor,core,similarity_df.values)
    reduced_matrix = pd.DataFrame(reduced_matrix,index=labels)  # Convert to DataFrame
    model = "PCA"
    np.savetxt(os.path.join(main_dir, f"latent_semantics_{model}_{s}_factor"+'_'+selected_model), factor, delimiter=',')
elif val == 2:
    core, u, v  = SVD(similarity_df.values, s)
    reduced_matrix = get_reduced_SVD(u, core)
    reduced_matrix = pd.DataFrame(reduced_matrix,index=labels)  # Convert to DataFrame
    model = "SVD"
    np.savetxt(os.path.join(main_dir, f"latent_semantics_{model}_{s}_U"+'_'+selected_model), u, delimiter=',')
    np.savetxt(os.path.join(main_dir, f"latent_semantics_{model}_{s}_core"+'_'+selected_model), core, delimiter=',')
    np.savetxt(os.path.join(main_dir, f"latent_semantics_{model}_{s}_V"+'_'+selected_model), v, delimiter=',')
elif val == 3:
    lda = LatentDirichletAllocation(n_components=s, random_state=42) 
    print("here1")
    max_value = np.max(similarity_df)
    scaler = MinMaxScaler(feature_range=(0,max_value))
    similarity_df = scaler.fit_transform(similarity_df) 
    reduced_matrix = lda.fit_transform(similarity_df,s) 
    reduced_matrix = pd.DataFrame(reduced_matrix, index=labels)
    directory = main_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f"lda_weights_{s}_{name}.pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(lda, file)
    model = "LDA"
elif val == 4:
    kmeans_model = KMeans(n_clusters=s,random_state=1)
    reduced_matrix = kmeans_model.fit_transform(similarity_df)
    reduced_matrix = pd.DataFrame(reduced_matrix, index=labels)
    model = "KMeans"
    directory = main_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f"kmeans_weights_{s}_{name}.pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(kmeans_model, file)
    model = "KMeans"

output_filename = os.path.join(main_dir, f"latent_semantics_{model}_{s}_components"+'_'+selected_model)
reduced_matrix.to_csv(output_filename)
print(f"Latent semantics saved to {output_filename}")


# PRINTING LABEL-WEIGHT PAIRS

print("\nLabel-Weight Pairs (sorted by weight for each latent semantic):")

for component in range(s):
    print(f"\nLatent Semantic {component + 1}:")
    sorted_labels = reduced_matrix.iloc[:, component].sort_values(ascending=False)
    for label, weight in sorted_labels.items():
        print(f"{label} - {weight:.4f}")


# d = getFromTask5("SVD",6,"layer4_data")
# print(d)
