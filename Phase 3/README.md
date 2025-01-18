# Environment Setup
Use the below common to create the environment in Sol or locally
```
mamba create -n MWDEnv3 -c conda-forge -c pytorch python=3.10 pytorch torchvision opencv scipy scikit-learn numpy=1.23.2 pandas
```
<i>Replace mamba with conda, if run locally</i>  

# Task 0  
## How to Run
### Prequisites
1. HMDB51 Videos Dataset with the following folder structure  
<pre>
- hmdb51_org  
    - target_videos
        - catch
        - climb
        - ....
    - non_target_videos
        - dive
        - kiss
        - ....
</pre> 
2. Latent Models from Phase 2
3. VideoID_Mapping.csv file

### Mandatory Code Changes
None

### Running the Code
Run the python code using the command
```
python task0.py
```
## Expected Input  
None

## Expected Output  
Each label followed by it's inherent dimensionality

# Task 1a  
## How to Run
### Prequisites
1. HMDB51 Videos Dataset with the following folder structure  
<pre>
- hmdb51_org  
    - target_videos
        - catch
        - climb
        - ....
    - non_target_videos
        - dive
        - kiss
        - ....
</pre> 
2. 3 Latent Models from Phase 2
3. VideoID_Mapping.csv file

### Mandatory Code Changes
File locations to be changed (Search for "TODOn" in task1a.py and follow below instructions for 1 to n)
- <b>TODO1</b>: Update the feature set locations
- <b>TODO2</b>: Update the s values of the features selected
- <b>TODO3</b>: Update the location to dataset
- <b>TODO4</b>: Update the location to VideoID_Mapping.csv file

### Running the Code
Run the python code using the command
```
python task1a.py
```
## Expected Input  
1. Pick Latent Model
2. Number of Clusters
3. Pick if clustering for all or single label
4. If chosen single label, enter the label

## Expected Output  
Clusters visualized as:
1. as groups of thumbnails.
2. as differently colored point clouds in a 2-dimensional MDS space 

# Task 1b  
## How to Run
### Prequisites
1. HMDB51 Videos Dataset with the following folder structure  
<pre>
- hmdb51_org  
    - target_videos
        - catch
        - climb
        - ....
    - non_target_videos
        - dive
        - kiss
        - ....
</pre> 
2. 3 Latent Models from Phase 2
3. VideoID_Mapping.csv file

### Mandatory Code Changes
File locations to be changed (Search for "TODOn" in task1b.py and follow below instructions for 1 to n)
- <b>TODO1</b>: Update the feature set locations
- <b>TODO2</b>: Update the s values of the features selected

### Running the Code
Run the python code using the command
```
python task1b.py
```
## Expected Input  
1. Pick Latent Model
2. Number of Clusters

## Expected Output  
Clusters visualized as:
1. as groups of labels.
2. as differently colored point clouds in a 2-dimensional MDS space 

# Task 2 
## How to Run
### Prequisites
1. HMDB51 Videos Dataset with the following folder structure  
<pre>
- hmdb51_org  
    - target_videos
        - catch
        - climb
        - ....
    - non_target_videos
        - dive
        - kiss
        - ....
</pre> 
2. 3 Latent Models from Phase 2  
3. VideoID_Mapping.csv file

### Mandatory Code Changes
File locations to be changed (Search for "TODOn" in task2.py and follow below instructions for 1 to n)
- <b>TODO1</b>: Update the location to target dataset videos
- <b>TODO2</b>: Update the location to VideoID_Mapping.csv file
- <b>TODO3</b>: Update the location to latent model files

### Running the Code
Run the python code using the command
```
python task2.py
```
## Expected Input  
1. Pick Latent Model
2. Pick among KNN and SVM Classifier
3. Query Video ID
4. 'm' most likely labels
5. k value for KNN

## Expected Output  
1. Most likely 'm' labels along with their scores for the given inputs

# Task 3 
## How to Run
### Prequisites
1. HMDB51 Videos Dataset with the following folder structure  
<pre>
- hmdb51_org  
    - target_videos
        - catch
        - climb
        - ....
    - non_target_videos
        - dive
        - kiss
        - ....
</pre> 
2. 3 Latent Models from Phase 2  
3. VideoID_Mapping.csv file

### Mandatory Code Changes
File locations to be changed (Search for "TODOn" in task3.py and follow below instructions for 1 to n)
- <b>TODO1</b>: Update the location to latent models
- <b>TODO2</b>: Update the location to dataset
- <b>TODO3</b>: Update the location to VideoID_Mapping.csv file

### Running the Code
Run the python code using the command
```
python task3.py
```
## Expected Input  
1. Pick Latent Model
2. Count of layers
3. Count of hashes per layer
4. Video ID
5. Count of 't' similar videos to retreieve 

## Expected Output  
1. Most Similar 't' videos and their scores along with their thumbnails

# Task 4
## How to Run
### Prequisites
1. HMDB51 Videos Dataset with the following folder structure  
<pre>
- hmdb51_org  
    - target_videos
        - catch
        - climb
        - ....
    - non_target_videos
        - dive
        - kiss
        - ....
</pre> 
2. 3 Latent Models from Phase 2  
3. VideoID_Mapping.csv file

### Mandatory Code Changes
File locations to be changed (Search for "TODOn" in task4.py and follow below instructions for 1 to n)
- <b>TODO1</b>: Update the location to latent model files
- <b>TODO2</b>: Update the location to target dataset videos
- <b>TODO3</b>: Update the location to VideoID_Mapping.csv file

### Running the Code
Run the python code using the command
```
python task4.py
```
## Expected Input  
1. Pick Latent Model
2. Count of layers
3. Count of hashes per layer
4. Video ID
5. Count of 't' similar videos to retreieve 
6. After results are displayed, we take feedback by marking each result as like or dislike Mark each videos as like or dislike
7. Pick among KNN or Decision Tree  
8. Enter 'k' value for KNN, if KNN picked

## Expected Output  
1. Most Similar 't' videos and their scores along with their thumbnails
2. Most Similar 't' newly ranked videos along with their thumbnails post feedback