import os
import argparse

from os import walk
from pathlib import Path
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import EmbDataset
from tqdm.auto import tqdm, trange
from labml import lab, tracker, experiment
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import metric
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from tqdm.auto import tqdm, trange
from torch.utils.tensorboard import SummaryWriter



"""
Adaptation module
"""

class Adaptation (nn.Module):
    def __init__(self,input):
        super().__init__()
        self.weight_matrix = nn.Linear(input,1024,bias=False)

    def forward(self,x):
        x = self.weight_matrix(x)
        x = torch.relu(x)
        return x



"""
    Global trainer class
"""

class Trainer (nn.Module):

    def __init__(self):
        super().__init__()
        self.adapt = Adaptation(256)

    def reset(self, config):
        self.config = config
        self.__trained = False
        self.__device = config["device"] if "device" in config else "cpu"

    """
        training method
    """
    def fit(self, loss_func, optimizer, loader, device, model_log_interval):

        self.adapt.train()
        data, target = next(iter(loader))
        left, right = data["left"], data["right"]
        left, right = left.to(device), right.to(device)

        out_left, out_right = self.transform(left), self.transform(right)
        loss = loss_func(out_left, out_right, config["temperature"],device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {'loss' : loss.item()}

    """
    returns embds addapted
    """

    def transform(self, X):
        return self.adapt(X)


    """
    validate function
    """

    def validate(self, loss_func, loader, device):
        self.adapt.eval()
        with torch.no_grad():
            data, target = next(iter(loader))
            left, right = data["left"], data["right"]
            left, right = left.to(device), right.to(device)
            out_left, out_right = self.transform(left),self.transform(right)

            test_loss = loss_func(out_left, out_right, config["temperature"],device)
            metrics = {"Test/Loss": test_loss.item()}
            for k in range(1,config["topk"]+1):
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
            embds_dataset, batch_size=len(train_indices), sampler=train_sampler
        )

        valid_loader = torch.utils.data.DataLoader(
            embds_dataset, batch_size=len(val_indices), sampler=valid_sampler
        )

        return train_loader, valid_loader

    """
    initialize one experiment
    """

    def __init__(self, model: Trainer, input_embeddings: EmbDataset, expermiement_id: str, config):
        self.config = config
        self.model = model
        self.id = expermiement_id
        self.ebds = input_embeddings


    """
        reset train and valid loader with a new random split
    """
    def reset(self):
        train_loader, valid_loader = Experiment.split(self.ebds, self.config)
        self.config ["train_loader"] = train_loader
        self.config ["valid_loader"] = valid_loader
        self.model.reset(self.config)

    """
    for a model run the experiment "n times"
    """
    def run(self,path_ebds, modelname,config):
        metrics_of_all_runs = []
        for run in range(config["runs"]):
            r = f"run{run+1}"
            #save each run summary in the model Directory
            writer = SummaryWriter(path_ebds / Path(modelname) / Path(r))
            self.reset()
            metrics_per_run = {"train": dict(), "test": dict()}
            for epoch in range(1, config["epochs"]+1):
                info = model.fit(
                    metric.calc_loss,
                    config["optimizer"],
                    config["train_loader"],
                    config["device"],
                    config["train_log_interval"],
                )

                for key, val in info.items():
                    if not (key in metrics_per_run["train"]):
                        metrics_per_run["train"][key] = []
                    metrics_per_run["train"][key].append(val)

                    writer.add_scalar(key, val, epoch)

                    writer.flush()

                info = model.validate(metric.calc_loss, config["valid_loader"],config["device"])
                for key, val in info.items():
                    if not (key in metrics_per_run["test"]):
                        metrics_per_run["test"][key] = []
                    metrics_per_run["test"][key].append(val)

                    writer.add_scalar(key, val, epoch)
                    writer.flush()

                #save the model after each run  in the model Directory
                torch.save(
                    {
                        "epoch": config["epochs"],
                        "state_dict": model.state_dict(),
                        "optimzier": config["optimizer"].state_dict(),
                    },
                    path_ebds / Path(modelname) / Path(r) / "model_and_optimizer.pt",
                )

            metrics_of_all_runs.append(metrics_per_run)

        np.save(path_ebds / Path(modelname) / "all_runs", metrics_of_all_runs)
        return metrics_of_all_runs



"""
    check ebds Directory and returns a Dict of each model and its embds directory
"""

def check_dirct(root_dir):
    ebds_list = dict ()

    assert os.path.isdir(root_dir)

    _, dirs, _ = next(os.walk(root_dir))

    assert len(dirs) > 0

    for dir in dirs:
        _, _, ebds = next(os.walk(root_dir+"/"+dir))
        assert len(ebds) >= 2

        file = dir+".npy"
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
        "epochs": 20,
         "validation_split": 0.25,
         "shuffle_dataset": False,
         "device" : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
         "seed": 5,
         "train_log_interval": 10,
         "learning_rate": 0.01,
         "temperature": 15,
         "topk": 1,
         "train_loader" :None,
         "valid_loader" : None,
         "optimizer" : None,
         "runs" : 5
        }

    #path_ebds = "/home/hala/Documents/positive-similarity/ebds"
    args = parser.parse_args()
    path_ebds = args.data
    args = parser.parse_args()
    info = check_dirct(args.data)


    for modelname, ebds in info.items():

        embds_dataset= EmbDataset(ebds)

        model = Trainer()
        model.to(config["device"])
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.01, weight_decay=1e-5
            )
        config["optimizer"]= optimizer

        experiment = Experiment(Trainer(),embds_dataset, "test", config)
        metrics_of_all_runs = experiment.run(path_ebds, modelname, config)
        avg_path = path_ebds / Path(modelname) / "avg"
        writer = SummaryWriter(avg_path)

        # constructing the avg values for the saved metrics
        avg = dict()

        for k in metrics_of_all_runs[0].keys():
            for kk in metrics_of_all_runs[0][k].keys():
                avg[kk] = np.average(
                    [
                        metrics_of_all_runs[i][k][kk]
                        for i in range(len(metrics_of_all_runs))
                    ],
                    axis=0,
                )

                for i in range(len(avg[kk])):
                    writer.add_scalar("Avg_" + kk, avg[kk][i], i)
                    writer.flush()

        # save avg of all runs
        np.save(path_ebds / Path(modelname) / "avg_of_all_runs", avg)
