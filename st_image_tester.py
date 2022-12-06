
import streamlit as st
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

correct = 0
total = 0

model_resnet = model.resnet_model_50()
model_path = Path('model_resnet')/'model.pth' #"C:\\code\\image nudity detector app resnet\\model_resnet\\model.pth"
model_resnet.load_state_dict(torch.load(model_path))

st.header('VIDEO NUDITY DETECTOR RESNET')
st.write("Please upload a VIDEO for nudity check")

video_uploaded=st.file_uploader(label= "Please upload a file",accept_multiple_files=False, type = ['wmv', '.mp4'])
if video_uploaded is not None:
    st.video(video_uploaded)
    # st.write(image_uploaded.name)

    # uploaded_image_path=os.path.join('tmp_dir',image_uploaded.name)
    # st.write(uploaded_image_path)
    fname=os.path.splitext(video_uploaded.name)
    file_ext=fname[1]
    fname_u='tmp'+file_ext
    video_uploaded.name = fname_u
    # st.write(fname_u)
    
    # saving file
    with open(video_uploaded.name, 'wb') as f:
        f.write(video_uploaded.getbuffer())
        # st.success("File Saved")


    import cv2 
    result=""
    fres={}

    from PIL import Image 
    from tensorflow import keras


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

    
    import time
    stime=time.time()
    result = video_nsfw(video_uploaded.name)
    etime = time.time()

    with st.spinner('LOADING RESULTS....'):
    # time.sleep(5)
        # st.spinner('Loading results...')
        if result == 'NSFW':
            st.error(result)
        else:
            st.success(result)
        st.info(f"Execution time: {((etime-stime))} sec")
        path = pathlib.Path(video_uploaded.name)
        path.unlink()
    # st.write(result)

# print(result)


