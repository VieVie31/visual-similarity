import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataset import EmbDataset

import matplotlib.pyplot as plt
import numpy as np
import utils
from pathlib import Path
from tqdm.auto import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from config import config

from types import FunctionType
from typing import Dict


class Adaptation(nn.Module):
    def __init__(self, in_dim=256):
        super().__init__()
        self.weight_matrix = nn.Linear(in_dim, 1024, bias=False)

    def forward(self, x):
        x = self.weight_matrix(x)
        x = torch.relu(x)
        return x


def calc_loss(left, right, temp):
    sim1 = utils.sim_matrix(left, right)
    sim2 = sim1.t()

    loss_left2right = F.cross_entropy(
        sim1 * temp, torch.arange(len(sim1)).long().to(device)
    ).to(device)
    loss_right2left = F.cross_entropy(
        sim2 * temp, torch.arange(len(sim2)).long().to(device)
    ).to(device)
    loss = loss_left2right * 0.5 + loss_right2left * 0.5

    return loss


def training_step(model, loss_func, optimizer, loader, device):
    """
    Training the model for one epoch.
    """
    model.train()

    data, target = next(iter(loader))

    left, right = data["left"], data["right"]
    left, right = left.to(device), right.to(device)

    out_left, out_right = model(left), model(right)

    loss = loss_func(out_left, out_right, config["temperature"])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {"Train/Loss": loss.item()}


def validate(model, loss_func, loader, device):
    """
    Validating the model.
    """
    model.eval()

    with torch.no_grad():
        data, target = next(iter(loader))
        left, right = data["left"], data["right"]
        left, right = left.to(device), right.to(device)
        out_left, out_right = model(left), model(right)

        test_loss = loss_func(out_left, out_right, config["temperature"])

        metrics = {"Test/Loss": test_loss.item()}

        for k in config["topk"]:
            metrics[f"Test/top {k}"] = utils.topk(out_left, out_right, k)

        return metrics


def train(
    model: nn.Module,
    loss_func: FunctionType,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epochs: int,
    save_to: str,
    device: str,
) -> Dict:
    """
    This will train the model with the given optimizer for {num_epochs} epochs.
    And it will save the metrics defined in training_step and validate in the given path {save_to}.
    """
    writer = SummaryWriter(save_to)
    metrics_per_run = {"train": dict(), "test": dict()}

    for epoch in trange(1, num_epochs + 1):
        info = training_step(
            model,
            loss_func,
            optimizer,
            train_loader,
            device,
        )
        for metric, val in info.items():
            if not (metric in metrics_per_run["train"]):
                metrics_per_run["train"][metric] = []
            metrics_per_run["train"][metric].append(val)

            writer.add_scalar(metric, val, epoch)
            writer.flush()

        info = validate(model, loss_func, valid_loader, device)
        for metric, val in info.items():
            if not (metric in metrics_per_run["test"]):
                metrics_per_run["test"][metric] = []
            metrics_per_run["test"][metric].append(val)

            writer.add_scalar(metric, val, epoch)
            writer.flush()

    np.save(run_path / "data", metrics_per_run)
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

    is_cuda = config["use_cuda"] and torch.cuda.is_available()
    if not is_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:0")

    # load the dataset and split
    embds_dataset = EmbDataset(args.data)
    train_loader, valid_loader = utils.split_train_test(
        embds_dataset, config["test_split"], config["seed"], config["shuffle_dataset"]
    )

    # multiple runs
    metrics_of_all_runs = []
    optimizers_args = [{'lr': 1e-3, 'weight_decay': 1e-5}, {'lr': 1e-2, 'weight_decay': 1e-2},
    {'lr': 1e-1, 'weight_decay': 1e-1}, {'lr': 1, 'weight_decay': 1e-3}]

    for run in trange(args.runs):
        run_path = Path(args.save_to) / f"run{run+1}"
        model = Adaptation()
        optimizer = torch.optim.Adam(model.parameters(), **optimizers_args[run])

        writer = SummaryWriter(run_path)
        writer.add_text('Optimizer', str(optimizer).replace('\n', '  \n'))
        writer.add_text('Model', str(model).replace('\n', '  \n'))
        

        metrics_per_run = train(
            model,
            calc_loss,
            train_loader,
            valid_loader,
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