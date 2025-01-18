import time
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

def task3(path, r=4, n=12, frames_folder='./Demo_Codes/Frames', feature_folder='./Demo_Codes/Features'):

    os.makedirs(frames_folder, exist_ok=True) # Folder for saving frames
    os.makedirs(feature_folder, exist_ok=True)  # Folder for saving features
    hist_dir = "./Demo_Codes/task3_save"
    os.makedirs(hist_dir, exist_ok=True)  # Folder for saving features
    base_name = os.path.basename(path).split('.')[0] # This is to get the video name and not path
    cam = cv2.VideoCapture(path)
    
    # Here we are counting total frames in the video
    total_frames=int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    frameno = 0
    while(True):
        ret,frame = cam.read()
        if ret:
            frameno=frameno+1  # Counter for frames
            if(frameno==1):  # If and else condition for reaching first,middle and last frame and saving them
                frame_name = os.path.join(frames_folder, f'{base_name}_frame_1.jpg')  # Save with base name
                print("First frame: "+frame_name)
                cv2.imwrite(frame_name,frame)
            elif (frameno==int(total_frames/2)):
                frame_name = os.path.join(frames_folder, f'{base_name}_frame_2.jpg')  # Save with base name
                print("Middle frame: "+frame_name)
                cv2.imwrite(frame_name,frame)
            elif (frameno==int(total_frames)-1):
                frame_name = os.path.join(frames_folder, f'{base_name}_frame_3.jpg')  # Save with base name
                print("Last frame: "+frame_name)
                cv2.imwrite(frame_name,frame)
                # Will skip rest of the frames
        else:
            break # If no more frames are found in video we exit
    
    cam.release()
    cv2.destroyAllWindows()
    
#     r = 4  # here we can take value of r and n as inputs incase if user wants to define them
#     n = 12
    histos = []
    video_feature_vector = [] # This is a list for feature vectors of concatenated histograms
    
    for frame_index in range(1, 4):  # first loop to go through first,middle and last frame
        image_name = os.path.join(frames_folder, f'{base_name}_frame_{frame_index}.jpg')
        img = cv2.imread(image_name)
        if img is None:
            print(f"Error reading {image_name}. File may not exist or is corrupted.")
            continue  # Skip to the next frame
        
        im_h, im_w, channels = img.shape  # Reading the image and extracting its height and width
        
        figure, axis = plt.subplots(r, r)  # This is for creating figure of attached cell histograms
        plt.subplots_adjust(top=1,bottom=0.5,right=1,left=0.5)
        
        # Feature vector for the current frame
        frame_feature_vector = []
        
        for i in range(1,r+1): # This loop is for going from top to bottom of image
            for j in range(1,r+1):  # This loop is for going from left to right of image
                tile = img[(im_h//r)*(i-1):(im_h//r)*i,(im_w//r)*(j-1):(im_w//r)*j]# This is cutting
                # out cells based on dimensions of the image
                rgb_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB) # Converting cells/tiles from bgr to rgb
                
                # Now we calculate histograms for red,green and blue channels of the cell
                hist_r = np.histogram(rgb_tile[:,:,0].ravel(),bins=n)[0]
                hist_g = np.histogram(rgb_tile[:,:,1].ravel(),bins=n)[0]
                hist_b = np.histogram(rgb_tile[:,:,2].ravel(),bins=n)[0]
                # Concatenate histograms for all three channels into a single vector for this tile
                tile_histogram = np.concatenate([hist_r, hist_g, hist_b])
                # Append this tile's histogram to the frame feature vector
                frame_feature_vector.extend(tile_histogram)
                
                # Now we plot the histograms for n bins for tile and for red,green blue channel
                axis[i-1,j-1].hist(rgb_tile[:,:,0].ravel(),bins=n,edgecolor='black',color='red',alpha=0.5)
                axis[i-1,j-1].hist(rgb_tile[:,:,1].ravel(),bins=n,edgecolor='black',color='green',alpha=0.5)
                axis[i-1,j-1].hist(rgb_tile[:,:,2].ravel(),bins=n,edgecolor='black',color='blue',alpha=0.5)
                axis[i-1,j-1].tick_params(axis='both',labelsize=4)
        
        # Append this frame's feature vector to the video feature vector
        video_feature_vector.extend(frame_feature_vector)
        
        # following is to save the histogram based on details like timestamp and video it belongs to
        timestamp = int(time.time())  # here I used timestamp because if the name of histograms are same than it will
        # overwrite the saved histogram so with timestamp it gives unique name to each saved file
        hist_name = f'{base_name}_histogram_frame_{frame_index}_{timestamp}.png'
        hist_path = os.path.join(hist_dir, hist_name)
        plt.savefig(hist_path)
        plt.close(figure)  # Close the figure after saving
        
        histos.append(hist_path)
        
    video_feature_vector = np.array(video_feature_vector) # Here we convert the video feature vector to np array
    # following is to save the feature vector
    feature_file = os.path.join(feature_folder,f'{base_name}_features.npy')
    np.save(feature_file,video_feature_vector)
    
    print(f"Saved feature vector for video '{base_name}' to {feature_file}")
    
    return video_feature_vector, histos


video_path = input("Enter path of video : ")
output = task3(video_path)

print("feature vector : " , output[0])
print("Output length : ", len(output[0]))
