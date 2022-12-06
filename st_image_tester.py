
import streamlit as st
import os

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

st.header('IMAGE NUDITY DETECTOR RESNET')
st.write("Please upload an image for nudity check")

image_uploaded=st.file_uploader(label= "Please upload a file",accept_multiple_files=False, type = ['jpg', 'jpeg', 'png'])
if image_uploaded is not None:
    st.image(image_uploaded)
    # st.write(image_uploaded.name)

    # uploaded_image_path=os.path.join('tmp_dir',image_uploaded.name)
    # st.write(uploaded_image_path)
    fname=os.path.splitext(image_uploaded.name)
    file_ext=fname[1]
    fname_u='tmp'+file_ext
    # st.write(fname_u)
    
    # saving file
    with open(image_uploaded.name, 'wb') as f:
        f.write(image_uploaded.getbuffer())
        # st.success("File Saved")



    def image_nsfw_detector(file_path):
        classes = {0:'Drawing', 1:'Hentai', 2:'Neutral', 3:'Porn', 4:'Sexy'}
        
        pred = []
        lab = []

        transform_data = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])


        from PIL import Image
        img=Image.open(file_path)
        input=transform_data(img)
        input = input.unsqueeze(0)
        input = input.cuda()

        predictions = model_resnet(input)

        pred.extend(torch.max(predictions, dim=1)[1].tolist())

        if classes.get(pred[0]) == 'Porn' or classes.get(pred[0]) == 'Sexy' or classes.get(pred[0]) == 'Hentai':
            result = 'NSFW'
        else:
            result = 'SAFE'

        return result
    
    import time
    stime=time.time()
    result = image_nsfw_detector(image_uploaded.name)
    etime = time.time()

    with st.spinner('LOADING RESULTS....'):
    # time.sleep(5)
        # st.spinner('Loading results...')
        if result == 'NSFW':
            st.error(result)
        else:
            st.success(result)
        st.info(f"Execution time: {((etime-stime))} ms")
    # st.write(result)

# print(result)


