import torch
import torchvision.models.video as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch.nn as nn
import warnings
import os
warnings.filterwarnings('ignore')

def task1(file_path,model_name):
    
    #visualise the video
    cap = cv2.VideoCapture(file_path)
    if(cap.isOpened() == False):
        print("error... cant open video")
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: break
    cap.release()
    cv2.destroyAllWindows()


    def getFrames(file_path):
        '''
        Reads the video frame by frame and returns a list of frames
        '''
        video = cv2.VideoCapture(file_path)
        frames = []
        while True:
            ret, frame = video.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        video.release()
        return frames

    def processFrames(frames,t):
        '''
        Processes and transforms the frames based on the transformation (t) provided
        '''
        processed = []
        for frame in frames:
            frame = t(frame)
            processed.append(frame)
        return torch.stack(processed)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112,112)),
        transforms.ToTensor()  
    ])

    frames = getFrames(file_path)
    pro = processFrames(frames,transform)
    pro = pro.unsqueeze(0) # Add batch dimension , shape becomes [N,D,C,H,W] , N-batch, D-Depth, C-channels, Height(H) and Width(W) of frame
    pro1 = torch.movedim(pro,1,2) # convert to shape [N,C,D,H,W] which is the required input shape in r3d 
    
    def hook(module,input,output):
        global features
        features = output

    def initialize_model(model_name):
        '''
        Add hooks to the layer based on the model name provided
        '''
        global features
        model = models.r3d_18(pretrained=True)
        model.eval()
        if model_name == "layer3":
            h1 = model.layer3.register_forward_hook(hook)
        elif model_name == "layer4":
            h1 = model.layer4.register_forward_hook(hook)
        elif model_name == "avgpool":
            h1 = model.avgpool.register_forward_hook(hook)
        # pro1 = pro[0][None,:,:,:]
        out = model(pro1)
        h1.remove()

    initialize_model(model_name)

    if(features.shape[1] == 256): 
       # if model is layer 3, average spatial dimensions and then flatten the tensor followed by a linear transformation to get 512 dimensional tensor
        torch.manual_seed(0) # setting seed so that linear layer is initialised with same weights each time 
        avg_features = torch.mean(features,dim=(3,4))
        squeezed = torch.squeeze(avg_features) #remove batch dimension
        in_tensor = torch.flatten(squeezed) #collapse into a single dimension 
        myLayer = nn.Linear(in_features=256*features.shape[2],out_features=512) # define a linear layer 
        final_tensor = myLayer(torch.squeeze(in_tensor)) #remove batch dimension with squeeze and then apply linear transformation
        print("in l3")
    elif(features.shape[1] == 512 and features.shape[2] != 1): # case of layer 4
        # average the tensor on dimension 2,3,4 to get 512 dimensional tensor
        final_tensor = torch.squeeze(torch.mean(features,dim=(2,3,4)))
        print("in l4")
    else: #case of avgpool
        #this layer will already give output as 512 dimensional tensor
        final_tensor = torch.squeeze(features)
        print("in avg")
    return final_tensor

video_path = input("Enter path of video : ")
model_name = input("Enter model name (layer3 / layer4 / avgpool) : ")
output = task1(video_path,model_name)

print(output)