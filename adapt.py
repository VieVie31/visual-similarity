from torch.utils.data import DataLoader, Dataset
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from PIL import Image
from os import walk
from tqdm.auto import tqdm, trange
from pathlib import Path


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Adaptation (nn.Module):
    def __init__(self,input):
        super().__init__()
        self.weight_matrix = nn.Linear(input,1024,bias=False)



    def forward(self,x):
        x = self.weight_matrix(x)
        x = torch.relu(x)

        return x


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def calc_loss(left,right,temp):
  sim1 = sim_matrix(left, right)
  sim2 = sim1.t()
  loss_left2right= F.cross_entropy(sim1* temp, torch.arange(len(sim1)).long().to(device)).to(device)
  loss_right2left= F.cross_entropy(sim2* temp, torch.arange(len(sim2)).long().to(device)).to(device)
  loss = loss_left2right* 0.5 + loss_right2left * 0.5
  return loss

def load_embeddings (path):
  data = np.load(path, allow_pickle=True).reshape(-1)[0]
  return data

def get_topk(left,right, k):
  sim = -sim_matrix(left, right)
  sorted_idx = sim.argsort(1)
  sens1 = np.array([lbl in sorted_idx[lbl][:k] for lbl in range(len(left))])
  sim = sim.t()
  sorted_idx = sim.argsort(1)
  sens2 = np.array([lbl in sorted_idx[lbl][:k] for lbl in range(len(left))])
  return ((sens1 | sens2).mean())

def train ( model, left, right,optimizer, k, temp,  device ):
  model.train()
  left_adapted = model(left)
  right_adapted = model(right)
  loss = calc_loss(left_adapted, right_adapted, temp)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss.item()

def validate(model, left_test, right_test,k, temp, device):
  model.eval()
  with torch.no_grad():
    left_adapted_test = model(left_test)
    right_adapted_test = model(right_test)
    loss = calc_loss(left_adapted_test,right_adapted_test,temp)
    recall = get_topk(left_adapted_test,right_adapted_test,k)

  return loss.item(), recall

def init (fname, spl):
  data = load_embeddings(fname)
  left = data['left']
  right = data['right']
  left = torch.FloatTensor(left).to(device)
  right = torch.FloatTensor(right).to(device)
  model = Adaptation(256)
  model.to(device)
  N = len(left)
  sample = int(spl*N)
  idx = np.random.permutation(left.shape[0])
  train_idx, test_idx = idx[:sample], idx[sample:]
  optimizer = torch.optim.Adam(model.parameters(),lr =0.0005,weight_decay=1e-5)
  left_train, left_test, right_train, right_test = left[train_idx,:], left[test_idx,:], right[train_idx,], right[test_idx,]
  return left_train, left_test, right_train, right_test,model,optimizer


def plot_scores(train_losses, valid_losses, recalls):
  plt.plot(train_losses, label='Training loss')
  plt.plot(valid_losses, label='Testing loss')
  plt.legend()
  plt.show()
  plt.figure(1)
  plt.plot(recalls,label ='Recall' )
  plt.legend()
  plt.show

def create_exp(expname, fname, train_losses, valid_losses, recalls, configs):
  left_train, left_test, right_train, right_test,model,optimizer = init(fname, configs['spl'])

  for epoch in range(configs['epochs']):
    loss_train= train(model, left_train, right_train, optimizer,configs['k'],configs['temp'], device)
    loss_valid, recall= validate(model,left_test,right_test,configs['k'],configs['temp'], device)

    train_losses.append(loss_train)
    valid_losses.append(loss_valid)
    recalls.append(recall)

  plot_scores(train_losses, valid_losses, recalls)

parser = argparse.ArgumentParser(description="Feature maps extractor")

parser.add_argument(
    "-d", "--data", required=True, type=str, metavar="DIR", help="Path to embds"
)

parser.add_argument(
    "-s",
    "--split",
    type=int,
    metavar="N",
    default=0.75,
    help="split",
)
parser.add_argument(
    "-e",
    "--epochs",
    required=True,
    type=int,
    metavar="N",
    help="split",
)
parser.add_argument(
    "-k",
    "--topk",
    default=1,
    help="Top K",
)
parser.add_argument(
    "-t",
    "--temp",
    default=15,
    help="Temperature ",
)

def check_dirct(root_dir):
    ebds_list =[]
    assert os.path.isdir(root_dir)

    _, dirs, _ = next(os.walk(root_dir))

    assert len(dirs) > 0

    for dir in dirs:
        _, _, ebds = next(os.walk(root_dir+"/"+dir))

        assert len(ebds) == 2

        file = dir+".npy"
        ebds_list.append (Path(root_dir) / dir / file)

    return ebds_list

def main():

    args = parser.parse_args()
    ebds_list = check_dirct(args.data)
    configs = {
    'epochs': args.epochs,
    'use_cuda': True,
    'k': args.topk,
    'temp' : args.temp,
    'spl' : args.split,

    'learning_rate': 0.0005,
    }
    train_losses, valid_losses, recalls = [],[],[]
    for ebds in ebds_list:
        for i in range(20):
            create_exp(str(i),ebds,train_losses, valid_losses, recalls,configs)



if __name__ == "__main__":

    main()

