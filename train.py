import os
import argparse

import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataset import EmbDataset

import matplotlib.pyplot as plt
import numpy as np
import my_utils
from pathlib import Path
from tqdm.auto import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from config import config

from types import FunctionType
from typing import Dict, List, Tuple


class Adaptation(nn.Module):
    def __init__(self, in_dim=256):
        super().__init__()
        self.weight_matrix = nn.Linear(in_dim, 1024, bias=False)

    def forward(self, x):
        x = self.weight_matrix(x)
        x = torch.relu(x)
        return x

class DummyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def calc_loss(left, right, temp):
    sim1 = my_utils.sim_matrix(left, right)
    sim2 = sim1.t()

    loss_left2right = F.cross_entropy(
        sim1 * temp, torch.arange(len(sim1)).long().to(device)
    )
    loss_right2left = F.cross_entropy(
        sim2 * temp, torch.arange(len(sim2)).long().to(device)
    )
    loss = loss_left2right * 0.5 + loss_right2left * 0.5

    return loss


def training_step(model, loss_func, optimizer, train_sets : Tuple[DataLoader]): #FIXME: Dataset et pas DataLoader ?
    """
    Training the model for one epoch.
    Generally it will take two dataloaders:
        - the first one is the training loader without augmentation
        - the second one is the training loader with the augmentations
    """
    model.train()

    total_loss = 0

    for training_set in train_sets:
        left, right = training_set["left"], training_set["right"]

        out_left, out_right = model(left), model(right)

        loss = loss_func(out_left, out_right, config["temperature"])

        total_loss += loss

    total_loss /= len(train_sets)

    # debate: include this in the loop above or not ?
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    metrics = {"Train/Loss": total_loss.item()}

    aR = my_utils.AsymmetricRecall(out_left.clone().detach().cpu(), out_right.clone().detach().cpu())

    for k in config["topk"]:
        metrics[f"Train/top {k}"] = aR.eval(at=k) #my_utils.topk(out_left, out_right, k)

    return metrics


def validate(model, loss_func, valid_set):
    """
    Validating the model.
    """
    model.eval()

    with torch.no_grad():
        left, right = valid_set["left"], valid_set["right"]
        out_left, out_right = model(left), model(right)

        test_loss = loss_func(out_left, out_right, config["temperature"])

        metrics = {"Test/Loss": test_loss.item()}

        aR = my_utils.AsymmetricRecall(out_left.detach().cpu(), out_right.detach().cpu())

        for k in config["topk"]:
            metrics[f"Test/top {k}"] = aR.eval(at=k) #my_utils.topk(out_left, out_right, k)

        

        return metrics


def train(
    model: nn.Module,
    loss_func: FunctionType,
    train_set: List[Dict],
    valid_set: Dict,
    num_epochs: int,
    save_to: str,
    device: str,
) -> Dict:
    """
    This will train the model with the given optimizer for {num_epochs} epochs.
    And it will save the metrics defined in training_step and validate in the given path {save_to}.
    """
    # test before adaptation
    #before_adaptation_path = Path(save_to) / f"before_adapt"
    #before_adapt_writer = SummaryWriter(before_adaptation_path)
    #before_adapt_info = validate(DummyNetwork(), calc_loss, valid_set)


    writer = SummaryWriter(save_to)
    metrics_per_run = {"train": dict(), "test": dict()}

    for epoch in trange(1, num_epochs + 1):
        info = training_step(
            model,
            loss_func,
            optimizer,
            train_set,
        )
        for metric, val in info.items():
            if not (metric in metrics_per_run["train"]):
                metrics_per_run["train"][metric] = []
            metrics_per_run["train"][metric].append(val)

            writer.add_scalar(metric, val, epoch)
            writer.flush()

        #"""
        info = validate(model, loss_func, valid_set)
        for metric, val in info.items():
            if not (metric in metrics_per_run["test"]):
                metrics_per_run["test"][metric] = []
            metrics_per_run["test"][metric].append(val)

            writer.add_scalar(metric, val, epoch)
            writer.flush()
        #"""

        # add before adaptation metrics
        #for metric, val in before_adapt_info.items():
        #    before_adapt_writer.add_scalar(metric, val, epoch)
        #    before_adapt_writer.flush()

    np.save(run_path / "data", metrics_per_run)
    #np.save(run_path / "before_adapt_data", before_adapt_info)

    return metrics_per_run


parser = argparse.ArgumentParser(description="Training an adaptation model")
parser.add_argument(
    "-d",
    "--data",
    required=True,
    type=str,
    metavar="DIR",
    help="Path of the extracted embeddings",
)
parser.add_argument(
    "-s",
    "--save-to",
    required=True,
    type=str,
    metavar="DIR",
    help="Path to which we are going to save the calculated metrics (tensorboard)",
)
parser.add_argument(
    "-r",
    "--runs",
    type=int,
    metavar="N",
    default=1,
    help="Number of runs",
)

if __name__ == "__main__":
    args = parser.parse_args()
    print("temp ", config['temperature'])

    if not Path(args.save_to).exists():
        is_cuda = config["use_cuda"] and torch.cuda.is_available()
        if not is_cuda:
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:0")

        # load the dataset and split
        
        embds_dataset = EmbDataset(args.data, only_original=False)

        print('total ', len(embds_dataset))
        in_dim = embds_dataset[0][0]['left'].shape[0]

        data = embds_dataset.load_all_to_device(device)

        splitted = my_utils.split_train_test(
            embds_dataset, config["test_split"], device, config["seed"], config["shuffle_dataset"]
        )

        train_set, valid_set = splitted['train'], splitted['test']

        print('train set without aug size ', len(train_set[0]['left']))
        print('train set with aug size ', len(train_set[1]['left']))
        print('test  set without aug size  ', len(valid_set['left']))

        metrics_of_all_runs = []
        # optimizers_args = [{'lr': 1e-3}, {'lr': 0.0005, 'weight_decay': 1e-5},
        # {'lr': 1e-1, 'weight_decay': 1e-1}, {'lr': 1, 'weight_decay': 1e-3}]

        info = 'Embd path: ' + str(args.data) + '\n' + 'Using augmented images: ' + str(embds_dataset.augmented) + '\n'

        for run in trange(args.runs):
            run_path = Path(args.save_to) / f"run{run+1}"
            model = Adaptation(in_dim).to(device)
            # optimizer = torch.optim.Adam(model.parameters(), **optimizers_args[run])
            optimizer = torch.optim.Adam(model.parameters())

            writer = SummaryWriter(run_path)
            writer.add_text('Temperature: ', str(config['temperature']))
            writer.add_text('Optimizer', str(optimizer).replace('\n', '  \n'))
            writer.add_text('Model', str(model).replace('\n', '  \n'))
            writer.add_text('Informations: ', info.replace('\n', '  \n'))
            

            metrics_per_run = train(
                model,
                calc_loss,
                train_set,
                valid_set,
                config["epochs"],
                run_path,
                device,
            )

            metrics_of_all_runs.append(metrics_per_run)

            # save the model and the optimizer state_dics for this run
            torch.save(
                {
                    "epoch": config["epochs"],
                    "state_dict": model.state_dict(),
                    "optimzier": optimizer.state_dict(),
                },
                run_path / "model_and_optimizer.pt",
            )

        # save all runs metrics into one file
        np.save(Path(args.save_to) / "all_runs", metrics_of_all_runs)

        avg_path = Path(args.save_to) / "avg"
        writer = SummaryWriter(avg_path)

        # TODO generalize this

        if args.runs > 1:
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
            np.save(Path(args.save_to) / "avg_of_all_runs", avg)

    else:
        print("Found directory with the with the same saved name ! Will not perform training.")
