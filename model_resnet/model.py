import config
from torchvision import models
import torch.nn as nn
from torchsummary import summary
import torch

# preprocessing done in dataset_prep
def resnet_model_50():

    # resnet = models.resnet50(pretrained = True)
    resnet = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V1")


    # Already Trained Parameters will not train
    for param in resnet.parameters():
        param.requires_grad = False

    in_features = resnet.fc.in_features

    # Changing the last layer according to the classes
    fc = nn.Linear(in_features = in_features, out_features = config.N_CLASSES)
    resnet.fc = fc

    # summary(resnet.to(config.DEVICE), input_size = (3, 224, 224))
    summary(resnet.to(torch.device('cpu')), input_size = (3, 224, 224))

    return (resnet)

def resnet_model_101():

    resnet = models.resnet101(pretrained = True)

    # Already Trained Parameters will not train
    for param in resnet.parameters():
        param.requires_grad = False

    in_features = resnet.fc.in_features

    # Changing the last layer according to the classes
    fc = nn.Linear(in_features = in_features, out_features = config.N_CLASSES)
    resnet.fc = fc

    summary(resnet.to(config.DEVICE), input_size = (3, 224, 224))

    return (resnet)
