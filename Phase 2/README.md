# Task 0a  
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
2. Updated HoG and HoF Cluster Representatives for new target_videos with the following folder structure (Can be found in submitted folder)
<pre>
- HoG  
    - pair_4_2_HoG.csv
    - ....
    - pair_128_4_HoG.csv
- HoF
    - pair_4_2_HoF.csv
    - ....
    - pair_128_4_HoF.csv
</pre>
3. hmdb51_org_stips_filtered folder containing top 400 STIPs for each video
<pre>
- hmdb51_ord_stips_filtered  
    - brush_hair
    - cartwheel
    - catch
    - ....
    - turn
    - walk
    - wave
</pre>

### Environment Setup
Use the below common to create the environment in Sol
```
mamba create -n MWDEnv3 -c conda-forge -c pytorch python=3.10 pytorch torchvision opencv scipy scikit-learn numpy=1.23.2 pandas
```
<i>Replace mamba with conda, if run locally</i>  

### Mandatory Code Changes
File locations to be changed (Search for "TODOn" in Task0a.ipynb and follow below instructions for 1 to n)
- <b>TODO1</b>: Download the HoG Clusters and update the file location
- <b>TODO2</b>: Download the HoF Clusters and update the file location
- <b>TODO3</b>: Update the locations of filtered stips folder location
- <b>TODO4</b>: Update the folder location of target_videos and non_target_videos
- <b>TODO5</b>: Update the file location to store the VideoId Mapping File
- <b>TODO6</b>: Update the folder location to store the 6 Features of all videos
- <b>TODO7</b>: Update the saved output location after running the code [Ideally Same location as TODO6]
- <b>TODO8</b>: Update the locations for End Task 0a Output that will be used in subsqeuent tasks

### Running the Code
Run all cells from top to bottom
## Expected Output
An intermediate output folder containing all 6 features for each videoId individually in videoId Folder
<pre>
- Task0  
    - target_videos
        - Even
            - videoID_3894
            - ...
            - videoId_6764
        - Odd
            - videoID_3893
            - ...
            - videoId_6765
    - non_target_videos
        - videoID_0
        - ...
        - videoId_3892
</pre>
Each videoId folder contains
<pre>
- videoID_0  
    - BOF-HOF-480.npy
    - BOF-HOG-480.npy
    - metadata.txt
    - R3D18-AvgPool-512
    - R3D18-Layer3-512
    - R3D18-Layer4-512
    - COL-HIST
        - col_hist.npy
    
</pre>
A folder with 6 files. 1 file for each for the model with the features for all videos. We have submitted the following folder in ZIP
<pre>
- Task0  
    - avgpool_data.csv
    - col_hist_data.csv
    - hof_data.csv
    - hog_data.csv
    - layer3_data.csv
    - layer4_data.csv
</pre>
The above files are used in subsequent tasks and are needed to run the tasks

# Task 0b
## How to Run
### Prequisites
1. A folder with 6 files. 1 file for each for the model with the features for all videos. We have submitted the following folder in ZIP. This is the output of Task0a
<pre>
- Task0  
    - avgpool_data.csv
    - col_hist_data.csv
    - hof_data.csv
    - hog_data.csv
    - layer3_data.csv
    - layer4_data.csv
</pre>

### Environment Setup
Follow Task 0a Environment Setup and libraries installation to run Task 0b, if not done.

### Mandatory Code Changes
File locations to be changed (Search for "TODOn" in Task0b.py and follow below instructions for 1 to n)
- <b>TODO1</b>: Update the Task0a Output Folder locations
- <b>TODO2</b>: Update the location to the videos database

### Running the Code
Run the python code using the command
```
python Task0b.py
```
## Expected Output
### Input
Model Type (pick one): ['avgpool' , 'col_hist' , 'layer3' , 'layer4' , 'hof' , 'hog']  
Query Type (n: query by video name, i: query by video id): ['n', 'i']
Similar Videos Count (to fetch top 'k' videos): Integer'

### Output
A horizontal bar chart with video names on y axis and score on x axis

Video Name and Score for each of top 'k' similar videos and playback of those 'k' videos

# Task 1
## How to Run
### Prequisites
1. A folder with 6 files. 1 file for each for the model with the features for all videos. We have submitted the following folder in ZIP. This is the output of Task0a
<pre>
- Task0  
    - avgpool_data.csv
    - col_hist_data.csv
    - hof_data.csv
    - hog_data.csv
    - layer3_data.csv
    - layer4_data.csv
</pre>
2. A VideoId_Mapping.csv file. File that maps VideoIds to VideoNames

### Environment Setup
Follow Task 0a Environment Setup and libraries installation to run Task 0b, if not done.

### Mandatory Code Changes
File locations to be changed (Search for "TODOn" in Task0b.py and follow below instructions for 1 to n)
- <b>TODO1</b>: Update the Task0a Output File locations
- <b>TODO2</b>: Update the location to VideoId_Mapping.csv file location

### Running the Code
Run the python code using the command
```
python Task1.py
```
## Expected Output
### Input
VideoId or VideoName: The video to query on
Model Type (pick one integer): [3 for 'avgpool' , 4 for 'col_hist' , 1 for 'layer3' , 2 for 'layer4' , 5 for 'hof' , 6 for 'hog']  
Similar Labels Count (to fetch top 'l' labels): Integer

### Output

Label and Score for each of top 'l' similar labels

# Task 2
## How to Run
### Prequisites
1. A folder with 6 files. 1 file for each for the model with the features for all videos. We have submitted the following folder in ZIP. This is the output of Task0a
<pre>
- Task0  
    - avgpool_data.csv
    - col_hist_data.csv
    - hof_data.csv
    - hog_data.csv
    - layer3_data.csv
    - layer4_data.csv
</pre>

### Environment Setup
Follow Task 0a Environment Setup and libraries installation to run Task 0b, if not done.

### Mandatory Code Changes
File locations to be changed (Search for "TODOn" in Task2.py and follow below instructions for 1 to n)
- <b>TODO1</b>: Change file location to each of the 6 Task0a output files invidually
- <b>TODO2</b>: Change the output directory where the factor and core matrices will be saved


### Running the Code
Run the python code using the command
```
python Task2.py
```
## Expected Output
### Input
Feature Model Type (pick one integer): [1 for 'avgpool' , 6 for 'col_hist' , 2 for 'layer3' , 3 for 'layer4' , 5 for 'hof' , 4 for 'hog']  
Top Features to save 's': Integer Input  
Pick Dimensionality Reducation Technqiue (pick one integer): [1 for 'LDA' , 2 for 'PCA' , 3 for 'SVD' , 4 for 'K-Means']  

### Output
A task_2_output.txt where all the videoId-Weight pairs in decreasing order of weights for each latent semantic should be saved. The output being large is stored in a text file.  

The factor and core factor matrices are saved in the following folder for each given input:  
<pre>
{Directory}/Latent Semantics/Task2/{feature_model_name}/{s}/
</pre>
For all reduction techniques we saved the following files: 
<pre>
{method}_reduced.npy : Stores the dimensionally reduced feature

{method}_reduced_videoId.npy : Stores the object to videoId mapping of each row in dimensionally reduced feature

Here method can have the following values: 'pca', 'svd', 'k_means', 'lda'
</pre>
Additionally for PCA and SVD we save the following factor and core matrices in the same folder 
<pre>
SVD: svd_core.npy, svd_u.npy, svd_v.npy
PCA: pca_core.npy, pca_factor.npy
</pre>

# Task 3
## How to Run
### Prequisites
1. A folder with 6 files. 1 file for each for the model with the features for all videos. We have submitted the following folder in ZIP. This is the output of Task0a
<pre>
- Task0  
    - avgpool_data.csv
    - col_hist_data.csv
    - hof_data.csv
    - hog_data.csv
    - layer3_data.csv
    - layer4_data.csv
</pre>
2. A folder containing Output of Task 2. It should have Latent Semantics cached for all videos. 
<pre>
{Directory}/Latent Semantics/Task2/{feature_model_name}/{s}/
</pre>
The above folder should have {model}_reduced_all.csv containing latent semantics for all videos

### Environment Setup
Follow Task 0a Environment Setup and libraries installation to run Task 0b, if not done.

### Mandatory Code Changes
File locations to be changed (Search for "TODOn" in Task0b.py and follow below instructions for 1 to n)
- <b>TODO1</b>: Update the location to VideoId_Mapping.csv file location
- <b>TODO2</b>: Update the directory where the Latent Semantics are stored
- <b>TODO3</b>: Update the directory where the dataset is stored
- <b>TODO4</b>: Update the location to Task 0 output, avgpool_data.csv files

### Running the Code
Run the python code using the command
```
python Task3.py
```
## Expected Output
### Input
VideoId or VideoName: The video to query on  
Path Type [0 for Feature Model, 1 for Latent Semantics]: Feature Model for no dimensionality reduction whereas pick Latent Semantics for reducing dimensions  

Feature Model Path:  
m: Top m similar videos to fetch  
Model Type (pick one): ['avgpool' ,'col_hist' ,'layer3' ,'layer4' ,'hof' ,'hog']  

Latent Semantics Path:  
m: Top m similar videos to fetch  
Pick Model: {1: 'avgpool', 2: 'layer3', 3: 'layer4', 4: 'hog', 5: 'hof', 6: 'col_hist'}  
Pick Dimensionality Reducation Technqiue: {1: 'lda', 2: 'pca', 3: 'svd', 4: 'k_means'}  
s: Top s features to pick from Original Features

### Output

VideoId, VideoName and Score for each of top 'm' similar videos

# Task 4
## How to Run
### Prequisites
1. A folder with 6 files. 1 file for each for the model with the features for all videos. We have submitted the following folder in ZIP. This is the output of Task0a
<pre>
- Task0  
    - avgpool_data.csv
    - col_hist_data.csv
    - hof_data.csv
    - hog_data.csv
    - layer3_data.csv
    - layer4_data.csv
</pre>
2. A folder containing Output of Task 2. It should have Latent Semantics cached for all videos. 
<pre>
{Directory}/Latent Semantics/Task2/{feature_model_name}/{s}/
</pre>
The above folder should have {model}_reduced_all.csv containing latent semantics for all videos

### Environment Setup
Follow Task 0a Environment Setup and libraries installation to run Task 0b, if not done.

### Mandatory Code Changes
File locations to be changed (Search for "TODOn" in Task0b.py and follow below instructions for 1 to n)
- <b>TODO1</b>: Update the directory where the dataset is stored
- <b>TODO2</b>: Update the location to VideoId_Mapping.csv file location
- <b>TODO3</b>: Update the directory where the Latent Semantics of Task2 are stored

### Running the Code
Run the python code using the command
```
python Task4.py
```
## Expected Output
### Input
Label: Enter a label from target or non-target videos
Model Type (pick one): [1 for 'avgpool' , 6 for 'col_hist' , 2 for 'layer3' , 3 for 'layer4' , 5 for 'hof' , 4 for 'hog']  
Pick Dimensionality Reducation Technqiue: {1: 'lda', 2: 'pca', 3: 'svd', 4: 'k_means'}  
s: Top s features to pick from Original Features
m: Top m similar videos to fetch  

### Output

VideoId, VideoName and Score for each of top 'm' similar videos

# Task 5
## How to Run
### Prequisites
1. A folder with 6 files. 1 file for each for the model with the features for all videos. We have submitted the following folder in ZIP. This is the output of Task0a
<pre>
- Task0  
    - avgpool_data.csv
    - col_hist_data.csv
    - hof_data.csv
    - hog_data.csv
    - layer3_data.csv
    - layer4_data.csv
</pre>  

### Environment Setup
Follow Task 0a Environment Setup and libraries installation to run Task 0b, if not done.

### Mandatory Code Changes
File locations to be changed (Search for "TODOn" in Task5.py and follow below instructions for 1 to n)
- <b>TODO1</b>: Change file location to each of the 6 Task0a output files invidually
- <b>TODO2</b>: Change the output directory where the factor and core matrices will be saved


### Running the Code
Run the python code using the command
```
python Task5.py
```
## Expected Output
### Input
Feature Model Type (pick one integer): [3 for 'avgpool' , 4 for 'col_hist' , 1 for 'layer3' , 2 for 'layer4' , 5 for 'hof' , 6 for 'hog']  
Pick Dimensionality Reducation Technqiue (pick one integer): [1 for 'PCA' , 2 for 'SVD' , 3 for 'LDA' , 4 for 'K-Means']  
Top Features to save 's': Integer Input  

### Output
The factor and core factor matrices are saved in the following folder for each given input:  
<pre>
{Directory}/Latent Semantics/Task5
</pre>
For all reduction techniques we saved the following files: 
<pre>
label_label_similarity_matrix_{method}.csv : Stores the a label-label similarity matrix

Here method can have the following values: 'col_hist', 'avgpool', 'hof', 'hog', 'layer3', 'layer4'
</pre>
Additionally for PCA and SVD we save the following factor and core matrices in the same folder 
<pre>
PCA: 
latent_semantics_PCA_{s}_components_{method}.csv 
latent_semantics_PCA_{s}_eigenvectors_{method}.csv

SVD: 
latent_semantics_SVD_{s}_components_{method}.csv 
latent_semantics_SVD_Sigma_{method}.csv 
latent_semantics_SVD_U_{method}.csv 
latent_semantics_SVD_Vt_{method}.csv  

K Means:
kmeans_weights_{s}_{method}.pkl  

LDA:
lda_weights_{s}_{method}.pkl

Here method can have the following values: 'col_hist', 'avgpool', 'hof', 'hog', 'layer3', 'layer4'
</pre>

# Task 6
## How to Run
### Prequisites
1. A folder with 6 files. 1 file for each for the model with the features for all videos. We have submitted the following folder in ZIP. This is the output of Task0a
<pre>
- Task0  
    - avgpool_data.csv
    - col_hist_data.csv
    - hof_data.csv
    - hog_data.csv
    - layer3_data.csv
    - layer4_data.csv
</pre>


### Environment Setup
Follow Task 0a Environment Setup and libraries installation to run Task 0b, if not done.

### Mandatory Code Changes
File locations to be changed (Search for "TODOn" in Task0b.py and follow below instructions for 1 to n)
- <b>TODO1</b>: Change the main directory
- <b>TODO2</b>: Change the Latent Semantics File
- <b>TODO3</b>: Change the location to database


### Running the Code
Run the python code using the command
```
python Task6.py
```
## Expected Output
### Input

Label: The label to query on  
l: Top l similar labels to fetch  
Pick Model: {3: 'avgpool', 1: 'layer3', 2: 'layer4', 5: 'hog', 4: 'hof', 4: 'col_hist'} 

Path Type [1 for Feature Model, 2 for Latent Semantics]: Feature Model for no dimensionality reduction whereas pick Latent Semantics for reducing dimensions using Task 2 or Task 5 

If 1, No further inputs

If 2, Pick one latent semantic: [1 for Task 2 semantics, 2 for Task 5 semantics]

Pick Dimensionality Reducation Technqiue: {3: 'lda', 1: 'pca', 2: 'svd', 4: 'k_means'} 
s: Top s features to pick from Original Features


### Output

Label and Score for each of l labels