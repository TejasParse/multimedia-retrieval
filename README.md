# Enhanced Video Classification and Retrieval System Using HMDB51 Feature Engineering
## Phase 1
In the first phase of this project, we focused on developing a comprehensive set of six features to represent video data effectively. These features were extracted using various methods, leveraging pre-trained deep learning models and custom algorithms for spatiotemporal analysis. The detailed breakdown is as follows:
### R3D18-Layer3-512
This feature is derived from the intermediate layers of the pre-trained R3D-18 model. By attaching a hook to the "layer3" layer, we extracted a high-dimensional tensor (256×8×14×14
256×8×14×14) that encodes spatiotemporal patterns in video frames. To make this representation compact and computationally manageable, the tensor was averaged across each 4×14×14 subtensor to produce a 512-dimensional vector.
### R3D18-Layer4-512
This feature taps into the "layer4" layer of the pre-trained R3D-18 model, where higher-level spatiotemporal abstractions are encoded. The extracted tensor (512×4×7×7) was reduced to a 512-dimensional vector by averaging each 4×7×7 subtensor.
### R3D18-AvgPool-512
Leveraging the "avgpool" layer of the R3D-18 model, this feature directly produces a 512-dimensional vector. As the final pooling layer, it provides a highly condensed representation of the video, encapsulating essential global spatiotemporal information.
### BOF-HoG-480
The Bag-of-Features Histogram of Gradients (BOF-HoG) feature is designed to capture local spatiotemporal gradients within videos. Initially, 480 cluster representatives were created using k-means clustering on randomly sampled STIPs from non-target videos across 12 (𝜎<sup>2</sup>,𝜏<sup>2</sup>) parameter pairs. For a given video, 40-dimensional HoG histograms were computed for each parameter pair by assigning the 400 highest-confidence STIPs to the nearest cluster representative. These histograms were concatenated to form a 480-dimensional feature vector. This feature emphasizes motion and shape changes over time, making it highly effective for dynamic scene analysis.
### BOF-HoF-480
The Bag-of-Features Histogram of Flows (BOF-HoF) feature extends the idea of BOF-HoG by focusing on motion flow patterns. Similar to the BOF-HoG process, 480 HoF cluster representatives were created using k-means clustering. For each video, HoF histograms were generated for the 12 (𝜎<sup>2</sup>,𝜏<sup>2</sup>)parameter pairs, capturing flow-based dynamics in a 480-dimensional feature vector. This feature is particularly adept at distinguishing motion-based activities and behaviors within videos.
### COL-HIST
The Color Histogram (COL-HIST) feature provides a spatial breakdown of color information in video frames. For each video, three key frames (first, middle, and last) were selected. Each frame was divided into 𝑟×𝑟 cells, and an 𝑛-bin color histogram was computed for each cell. The resulting histograms encode the distribution of colors within localized regions of the frame, enabling the capture of both spatial and chromatic variations across the video. This feature is particularly useful for recognizing color-based patterns and visual themes in video content.
## Phase 2
In the second phase of the project, our focus shifted to reducing the dimensionality of the extensive feature set generated in Phase 1. High-dimensional features, while rich in information, can lead to increased computational costs and potential overfitting. To address this, we implemented several dimensionality reduction techniques from scratch, including Principal Component Analysis (PCA), Singular Value Decomposition (SVD), k-Means Clustering, and Linear Discriminant Analysis (LDA).
### Principal Component Analysis (PCA)  
PCA identifies the principal components in the data—directions of maximum variance—and projects the data onto these components. This reduces dimensionality while preserving as much variance as possible. We implemented PCA from scratch by calculating the covariance matrix, performing eigenvalue decomposition, and selecting the top eigenvectors.  

### Singular Value Decomposition (SVD)  
SVD decomposes a matrix into three components: \(U\), \(Sigma\), and \(V^T\), where \(Sigma\) contains singular values. By retaining only the largest singular values and their corresponding vectors, SVD reduces dimensionality while maintaining critical data relationships.  

### k-Means Clustering  
k-Means groups data points into \(k\) clusters by iteratively minimizing the intra-cluster variance. By assigning features to cluster centroids, this algorithm serves as a compression technique, effectively reducing dimensionality while preserving cluster-level distinctions.  

### Linear Discriminant Analysis (LDA)  
LDA maximizes the separability between classes by finding a lower-dimensional space that enhances class-specific variance. Unlike PCA, which focuses on variance irrespective of labels, LDA incorporates class labels to create discriminative projections.  

## Phase 3
### Phase 3: Search Retrieval and Incorporating User Feedback  

In the third phase, we focused on building a search retrieval system and incorporating user feedback to refine the results, working with the reduced feature set from Phase 2.  

- **K-Means Clustering**  
  We began with the K-Means algorithm to cluster the labels, enabling an analysis of how similar labels compare to each other.  

- **K-NN and SVM Classifier**  
  To classify unseen videos into labels, we implemented K-Nearest Neighbors (K-NN) and Support Vector Machine (SVM) classifiers. These methods provided robust classification capabilities based on the reduced feature set.  

- **Locality-Sensitive Hashing (LSH)**  
  LSH was implemented to efficiently retrieve videos similar to a given video. This approach allowed us to quickly find matches within the feature space, significantly optimizing search performance.  

- **Relevance Feedback System**  
  Finally, we incorporated user feedback by implementing a relevance feedback system based on decision trees and K-NN. This system enabled iterative refinement of search results by learning from user interactions and improving the ranking of retrieved videos.  

This phase completed the project by integrating classification, retrieval, and feedback mechanisms, delivering a comprehensive solution for video search and classification.  

## Team Members
 - Aryan Patel
 - Tejas Ajay Parse
 - Om Patel
 - Tanishque Zaware