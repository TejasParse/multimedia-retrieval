# Video Classification and Retrieval System Using HMDB51 Feature Engineering
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
## Phase 3
## Team Members
 - Aryan Patel
 - Tejas Ajay Parse
 - Om Patel
 - Tanishque Zaware