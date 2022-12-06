
# import streamlit as st
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""

import pathlib
from pathlib import Path
# from nsfw_detector import predict

import torch
from model_resnet import model
# import config
from torchvision import datasets
import torchvision.transforms as transforms


# model_path = Path('model_mobilenet')/ 'saved_model.h5'
# model=predict.load_model(model_path)

##conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pred = []
lab = []

import cv2 
result=""
fres={}

from PIL import Image 
from tensorflow import keras


model_path = Path('model_resnet')/'model.pth' #"C:\\code\\image nudity detector app resnet\\model_resnet\\model.pth"
model_resnet = model.resnet_model_50()
model_resnet.load_state_dict(torch.load(model_path))


def video_nsfw(video_path):
    classes = {0:'Drawing', 1:'Hentai', 2:'Neutral', 3:'Porn', 4:'Sexy'}
    
    pred = []
    lab = []

    transform_data = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])


    # from PIL import Image
    # img=Image.open(file_path)
    # input=transform_data(img)
    # input = input.unsqueeze(0)
# c=0
    vidcap = cv2.VideoCapture(video_path) 
    # flag=''
    def getFrame(sec,flag=''): 
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
        hasFrames,frame = vidcap.read() 
        if hasFrames: 
            im=Image.fromarray(frame)#.resize((224,224))
            image = keras.preprocessing.image.img_to_array(im)
            # image /= 255

            input=transform_data(im)
            input = input.unsqueeze(0)
            predictions = model_resnet(input)
            res = torch.max(predictions, dim=1)[1].tolist()[0]
            pred.extend(torch.max(predictions, dim=1)[1].tolist())

            if res==1 or res == 3 or res == 4:
                flag='x'
                hasFrames=False

        return (hasFrames, flag)
        # return hasFrames 
    sec = 0 
    frameRate = 3#it will capture image in each 0.5 second 
    success = getFrame(sec) 
    # print(success)
    while success: 
        sec = sec + frameRate 
        sec = round(sec, 3) 
        success, fl = getFrame(sec) 
        # print(sec)
        # print(fl)
        if fl=='x':
            return "NSFW"
            # break
    else:
        return 'SAFE'


if __name__ == "__main__":
    video_path = "C:\\code\\video nudity detector app resnet\\p2.mp4"
    res=video_nsfw(video_path)
    print(res)
