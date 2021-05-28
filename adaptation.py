import os
import argparse

from os import walk
from pathlib import Path
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import EmbDataset
from tqdm.auto import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import metric
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from types import FunctionType
from typing import Dict


"""
    to run this script : give the embds path as an argument
    it saves the results : loss, recall (per run and the average/standard deviation of all runs),
    model, optimizer and epochs
    of the training of each model in the same path

"""


"""
Adaptation module
"""

class Adaptation (nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_matrix = nn.Linear(256,1024,bias=False)

    def forward(self,x):
        x = self.weight_matrix(x)
        x = torch.relu(x)
        return x


"""
training method
"""
def fit(model, loss_func, optimizer, loader, device, temp):

    model.train()
    data, target = next(iter(loader))
    size = len( data["left"])
    left, right = data["left"], data["right"]
    left, right = left.to(device), right.to(device)
    out_left, out_right = model(left), model(right)

    loss = loss_func(out_left, out_right,temp,device)

    train_loss = loss
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    return {'Train/Loss' : train_loss.item()}



"""
    validate function
"""

def validate(model, loss_func, loader, device, temp):
    model.eval()

    with torch.no_grad():
        data, target = next(iter(loader))

        size = len( data["left"])
        left, right = data["left"], data["right"]
        left, right = left.to(device), right.to(device)
        out_left, out_right = model(left), model(right)

        loss = loss_func(out_left, out_right,temp,device)
        train_loss = loss
        test_loss = loss_func(out_left, out_right, temp,device)
        metrics = {"Test/Loss": test_loss.item()}

        for k in range(0,config["topk"]+1,5):
            if k==0:
                metrics[f"Test/top {k+1}"] = metric.topk(out_left, out_right, k+1)
            else:
                metrics[f"Test/top {k}"] = metric.topk(out_left, out_right, k)
        return metrics

"""
    experiment class : initialize reset and run an xp
"""

class Experiment():
    """
    split embds into train and valid set
    """
    @staticmethod
    def split ( input_embeddings, config):
        dataset_size = len(input_embeddings) 
        indices = list(range(dataset_size))

        split = int(np.floor(config["validation_split"] * dataset_size))
        if config["shuffle_dataset"]:
            np.random.seed(config["seed"])
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(
            input_embeddings, batch_size=len(train_indices), sampler=train_sampler
        )

        valid_loader = torch.utils.data.DataLoader(
            input_embeddings, batch_size=len(val_indices), sampler=valid_sampler
        )

        return train_loader, valid_loader

    """
    initialize one experiment
    """

    def __init__(self, input_embeddings: EmbDataset, expermiement_id: str, config):
        self.config = config
        self.id = expermiement_id
        self.ebds = input_embeddings

    """
        reset train and valid loader with a new random split
    """
    def reset(self):
        train_loader, valid_loader = Experiment.split(self.ebds, self.config)
        self.config ["train_loader"] = train_loader
        self.config ["valid_loader"] = valid_loader

    """
    for a model run the experiment "n times"
    """
    def run(self,path_ebds, modelname,config):
        metrics_of_all_runs = []
        for run in trange(config["runs"]):
            r = f"run{run+1}"
            #save each run summary in the model Directory
            writer = SummaryWriter(path_ebds / Path(modelname) / Path(r))
            model = Adaptation()
            model.to(config["device"])
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            self.reset()
            metrics_per_run = {"train": dict(), "test": dict()}
            for epoch in trange(1, self.config["epochs"]+1):
                info = fit(model,
                    metric.calc_loss,
                    optimizer,
                    self.config["train_loader"],
                    self.config["device"],
                    self.config["temperature"],
                )

                for key, val in info.items():
                    if not (key in metrics_per_run["train"]):
                        metrics_per_run["train"][key] = []

                    metrics_per_run["train"][key].append(val)

                    writer.add_scalar(key, val, epoch)

                    writer.flush()

                info = validate(model,metric.calc_loss, self.config["valid_loader"],self.config["device"], self.config["temperature"])
                for key, val in info.items():
                    if not (key in metrics_per_run["test"]):
                        metrics_per_run["test"][key] = []

                    metrics_per_run["test"][key].append(val)

                    writer.add_scalar(key, val, epoch)
                    writer.flush()

                #save the model after each run  in the model Directory

            np.save(path_ebds / Path(modelname) / Path(r) / "data",metrics_per_run)
            metrics_of_all_runs.append(metrics_per_run)

        torch.save(
            {
                "epoch": self.config["epochs"],
                "state_dict": model.state_dict(),
                "optimzier": optimizer.state_dict(),
            },
            path_ebds / Path(modelname) / "model_and_optimizer.pt",
        )
        np.save(path_ebds / Path(modelname) / "all_runs", metrics_of_all_runs)
        return metrics_of_all_runs

"""
    check ebds Directory and returns a Dict of each model and its embds directory
"""

def check_dirct(root_dir, to_get):
    ebds_list = dict ()

    assert os.path.isdir(root_dir)

    _, dirs, _ = next(os.walk(root_dir))

    assert len(dirs) > 0

    for dir in dirs:
        _, _, ebds = next(os.walk(Path(root_dir) / dir ))
        assert len(ebds) >= 2

        file = dir+to_get
        ebds_list[dir] = Path(root_dir) / dir / file

    return ebds_list

parser = argparse.ArgumentParser(description="Training an adaptation model")
parser.add_argument(
    "-d",
    "--data",
    required=True,
    type=str,
    metavar="DIR",
    help="Path of the extracted embeddings",
)

if __name__ == "__main__":

    config ={
        "epochs": 5,
         "validation_split": 0.25,
         "shuffle_dataset": False,
         "device" : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
         "seed": 5,
         "temperature": 15,
         "topk": 100,
         "train_loader" :None,
         "valid_loader" : None,
         "runs" : 5
        }

    #path_ebds = "/home/hala/Documents/positive-similarity/ebds"
    args = parser.parse_args()
    path_ebds = args.data
    args = parser.parse_args()
    info = check_dirct(args.data,"_with_pca_256.npy")
    avgs = check_dirct(args.data,"avg_of_all_runs.npy")
    stds = check_dirct(args.data,"std_of_all_runs.npy")

    for modelname, ebds in info.items():

        embds_dataset= EmbDataset(ebds)
        experiment = Experiment(embds_dataset, "test", config)
        metrics_of_all_runs = experiment.run(path_ebds, modelname, config)
        avg_path = path_ebds / Path(modelname) / "avg"
        writer = SummaryWriter(avg_path)
        # constructing the avg and std values for the saved metrics
        avg = dict()
        st = dict()
        for k in metrics_of_all_runs[0].keys():
            for kk in metrics_of_all_runs[0][k].keys():
                avg[kk] = np.average(
                    [
                        metrics_of_all_runs[i][k][kk]
                        for i in range(len(metrics_of_all_runs))
                    ],
                    axis=0,
                )

                st[kk] = np.std(
                    [
                        metrics_of_all_runs[i][k][kk]
                        for i in range(len(metrics_of_all_runs))
                    ],
                    axis=0,
                )
                for i in range(len(avg[kk])):
                    writer.add_scalar("Avg_" + kk, avg[kk][i], i)
                    writer.flush()

        # save avg and std of all runs
        np.save(path_ebds / Path(modelname) / "avg_of_all_runs", avg)
        np.save(path_ebds / Path(modelname) / "std_of_all_runs", st)
