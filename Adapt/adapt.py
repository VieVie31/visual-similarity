import os
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
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA



data_dir = '/content/data'
dataset = TTLDataset(data_dir, transform=  transforms.ToTensor())

batch_size =256

train_size = int(0.75 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True)

valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                        shuffle=True)


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

features_left = []
features_right = []
rep = Representation()

for i,batch in enumerate(train_loader):
    left = batch['left']
    right = batch['right']
    dleft, dright = rep(left,right)
    features_left.extend(dleft.detach().numpy())
    features_right.extend(dright.detach().numpy())


pca = PCA(256).fit((features_left))
new_features_left = pca.transform(features_left)
new_features_right = pca.transform(features_right)

adaptation = Adaptation(256)
adaptation.train()
optimizer = torch.optim.Adam(adaptation.parameters())


for epoch in range(20):
  left_adapted, right_adapted = adaptation(new_features_left, new_features_right)
  sim1 = sim_matrix(left_adapted, right_adapted)
  sim2 = sim_matrix(right_adapted, left_adapted)
  loss_left2right= F.cross_entropy(sim1* 15, torch.arange(len(sim1)).long())
  loss_right2left= F.cross_entropy(sim2* 15, torch.arange(len(sim2)).long())
  loss = loss_left2right* 0.5 + loss_right2left * 0.5
  loss.backward()
  optimizer.step()
  print("Epoch " , (epoch+1))
  print("Loss : ",loss.item())
