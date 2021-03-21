from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np

net18 = models.resnet18(False)
resnet18 = nn.Sequential(
    net18.conv1,
    net18.bn1,
    net18.relu,
    net18.maxpool,
    net18.layer1,
    net18.layer2,
    net18.layer3,
    net18.layer4

)


class ResNet18 (nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = resnet18

    def forward(self,x):
        x1 = resnet18[0:5](x)
        x2 = resnet18[5](x1)
        x3 = resnet18[6](x2)
        x4 = resnet18[7](x3)
        avg= nn.AdaptiveAvgPool2d((1,1))
        x1 = avg(x1)
        x2 = avg(x2)
        x3 = avg(x3)
        x4 = avg(x4)
        L = [x1, x2, x3, x4]
        L = [xx.squeeze() for xx in L]
        return L


class Representation(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet18()

    def forward(self,x1,x2):
        features1 = self.resnet(x1)
        features2= self.resnet(x2)
        features1 = [F.normalize(v) for v in features1]
        features1 = [F.normalize(v) for v in features2]
        return (torch.cat(features1,1),torch.cat(features2,1))



class Adaptation (nn.Module):
    def __init__(self,input):
        super().__init__()
        self.weight_matrix = nn.Linear(input,1024,bias=False)
        self.relu_activation = nn.ReLU()


    def forward(self,x1,x2):
        x1= self.weight_matrix(torch.from_numpy(x1).type(torch.FloatTensor))
        x1 = self.relu_activation(x1)
        x2= self.weight_matrix(torch.from_numpy(x2).type(torch.FloatTensor))
        x2 = self.relu_activation(x2)
        return x1,x2
