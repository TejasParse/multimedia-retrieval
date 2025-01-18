import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import warnings
warnings.filterwarnings("ignore")

# TODO1: Update the location to target dataset videos
targetFolder = r"E:\\Coding\\MultimediaWebDatabases\\Phase 3\\Assets\\hmdb51_org\\target_videos"
# TODO2: Update the location to VideoID_Mapping.csv file
videoMap = r"E:\\Coding\\MultimediaWebDatabases\\Phase 3\\VideoID_Mapping.csv"
latentModel = ""
print("Enter The latent Model you want to use : ")
print(" 1 : Avgpool with Kmeans (300 dims) : ")
print(" 2 : Layer4 with PCA (300 dims) : ")
print(" 3 : Color Histogram with SVD (300 dims) \n")
modelChoice = int(input())
if(modelChoice == 1):   latentModel = "avgpool_kmeans(300).csv"
elif(modelChoice == 2):     latentModel = "layer4_pca(300).csv"
elif(modelChoice == 3):     latentModel = "col_hist_svd(300).csv"

print("Which classifier would you like to use : ")
print(" 1 : KNN")
print(" 2 : SVM ")
classifierChoice = int(input(" Enter your choice : "))

queryId = int(input("\nEnter query video ID : "))

m = int(input("\n Enter m : "))

# TODO3: Update the location to latent model files
data = pd.read_csv("E:\\Coding\\MultimediaWebDatabases\\Phase 3\\Dataset\\" + latentModel)

feature_columns = data.iloc[:, 3:]
real_features = feature_columns.applymap(lambda x: np.real(complex(x)))
data.iloc[:, 3:] = real_features

target = data[(data["category"]=="target_videos") & (data["videoId"].astype(int)%2 == 0)]
features = target.iloc[:, 3:].values
labels = target["labels"].values
knn_data = [(list(features[i]), labels[i]) for i in range(len(features))]
# query = 4765
# m = 5
# k=25



def cosine_distance(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return 1 - (dot_product / (norm1 * norm2 + 1e-10))  # Add small epsilon to avoid division by zero

def knn_with_cosine(dataset, query, k, m):
    distances = []
    for features, label in dataset:
        distance = cosine_distance(features, query)
        distances.append((distance, label))

    distances.sort(key=lambda x: x[0])  # Sort by distance
    # print("distances : ",distances)
    k_neighbors = distances[:k]  # Select k nearest neighbors
    labels = [label for _, label in k_neighbors]
    label_counts = Counter(labels)
    return [(label,count) for label, count in label_counts.most_common(m)]
def getQueryFeatures(videoId):
    row = data[data["videoId"] == videoId]
    return row.iloc[:,3:].values.flatten()
def accuracyClassifier(result,query,m):
    trueLabel = getLabel(query)
    rank = m+1
    print("\n-------------------------------------------------")
    print("\ntrue label : ",trueLabel)
    for i , (label,vote) in enumerate(result):
        # print(label , vote)
        if(label == trueLabel):
            rank = i+1
    # print(rank)
    accuracy = (m - rank + 1)/m
    return accuracy
def getLabel(videoId):
    # Load the VideoID_Mapping CSV
    video_map_df = pd.read_csv(videoMap)
    
    # Find the videoName corresponding to the given videoId
    video_row = video_map_df[video_map_df["VideoID"] == videoId]
    if video_row.empty:
        return f"Video ID {videoId} not found in mapping."
    
    video_name = video_row.iloc[0]["Filename"]
    
    # Search for the video name in the target folder
    for root, dirs, files in os.walk(targetFolder):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            if video_name in os.listdir(folder_path):  # Check if video name exists in the folder
                return folder  # Return the name of the folder as the label
    
    return f"Video name {video_name} not found in target videos."


# -------------------SVM---------------------

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

# Load the model from a file
def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filename}")
    return model

class MultiClassSVM:
    def __init__(self, learning_rate=0.00001, lambda_param=0.001, n_iters=4000, early_stopping_patience=10, early_stopping_threshold=0.001):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.models = {}
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold

    def fit(self, X, y):
        unique_classes = np.unique(y)
        i = 0
        for cls in unique_classes:
            # Convert labels to binary for the current class
            y_binary = np.where(y == cls, 1, -1)
            
            # Train a binary SVM for this class
            model = BinarySVM(self.lr, self.lambda_param, self.n_iters, self.early_stopping_patience, self.early_stopping_threshold)
            model.fit(X, y_binary)
            print("fitted : ", i)
            i += 1
            self.models[cls] = model
        print("----------------------COMPLETED--------------------")

    def predict(self, X,m=1):
        # Evaluate each binary SVM on the data
        predictions = {}
        for cls, model in self.models.items():
            predictions[cls] = model.decision_function(X)  # Use decision score
        # print("predictions : ", predictions)
        # Assign the class with the highest score
        predictions = np.array(list(predictions.values())).T
        # print("predictions : ", predictions)
        top_m_indices = np.argsort(predictions, axis=1)[:, -m:][:, ::-1]  # Indices of top m scores
        top_m_scores = np.sort(predictions, axis=1)[:, -m:][:, ::-1]      # Top m scores
        return top_m_indices,top_m_scores


class BinarySVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=2000, early_stopping_patience=10, early_stopping_threshold=0.001):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = 0
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold

    def compute_loss(self, X, y):
        """
        Compute hinge loss and regularization term.
        """
        n_samples = X.shape[0]
        margins = 1 - y * (np.dot(X, self.w) + self.b)
        margins[margins < 0] = 0  # hinge loss is 0 if margin >= 1
        loss = 0.5 * self.lambda_param * np.dot(self.w, self.w) + np.mean(margins)
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                    # self.b -= self.lr * (self.lambda_param * self.b)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

            # Calculate the loss at the end of each epoch
            current_loss = self.compute_loss(X, y)
            # print(f"Epoch {epoch}, Loss: {current_loss}")

            # Check if early stopping condition is met
            if abs(best_loss - current_loss) < self.early_stopping_threshold:
                patience_counter += 1
            else:
                patience_counter = 0

            # If no improvement in 'early_stopping_patience' epochs, stop training
            if patience_counter >= self.early_stopping_patience:
                print("Early stopping triggered!")
                break

            # Update the best_loss to the current loss
            best_loss = current_loss

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

def standardize_query(query, scaler_params):
    mean = scaler_params['mean']
    scale = scaler_params['scale']
    return (query - mean) / scale


if(classifierChoice == 1):
    k = int(input("\n Enter k : "))
    knnResult = knn_with_cosine(knn_data,getQueryFeatures(queryId),k,m)
    # acc =accuracyClassifier(knnResult,queryId,m)
    print(f"Top {m} labels are : \n")
    for label, score in knnResult:
        print(f"{label} : {score}")
    # print("accuracy : ",acc)
    category = data[data["videoId"] == queryId]["category"].values[0]
    if(category == 'target_videos' and queryId%2==1):
        acc =accuracyClassifier(knnResult,queryId,m)
        print("Classifier accuracy : ",acc)
    

if(classifierChoice == 2):
    model = latentModel.split(".")[0]
    label_encoder = LabelEncoder()
    encodedLabels = label_encoder.fit_transform(labels)
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    scaler_params = {'mean': scaler.mean_, 'scale': scaler.scale_}
    np.save(f'scaler_params_{model}.npy', scaler_params)
    scaler_params = np.load(f'scaler_params_{model}.npy', allow_pickle=True).item()
    X_multi = normalized_features[:]
    y_multi = encodedLabels[:]
    # svm_multi = MultiClassSVM()  # train
    # svm_multi.fit(X_multi,y_multi)  # train
    # save_model(svm_multi, f"svm_{model}.pkl") # save trained
    loaded_model = load_model(f"svm_{model}.pkl")
    query_features = getQueryFeatures(queryId)
    query_normalized = standardize_query(query_features, scaler_params)

    top_m_labels, top_m_scores = loaded_model.predict(query_normalized.reshape(1, -1), m=m)
    decoded_labels = label_encoder.inverse_transform(top_m_labels[0])

    print("Top", m, "labels : ")
    svmResult = list()
    for i,label in enumerate(decoded_labels):
        print(f"{label} : {top_m_scores[0][i]}")
        svmResult.append((label,top_m_scores[0][i]))
    category = data[data["videoId"] == queryId]["category"].values[0]
    if(category == 'target_videos' and queryId%2==1):
        acc =accuracyClassifier(svmResult,queryId,m)
        print("Classifier accuracy : ",acc)

