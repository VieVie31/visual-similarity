"""
The script to use when you want to train some adaptation module on the extracted features (embeddings).

usage: train.py [-h] -d DIR -s DIR [--model {original,peterson,dummy}] [-e N] [-r N] [-t N] [--test-split N]
                [-k N [N ...]] [--gpu | --cpu]

Training an adaptation model

optional arguments:
  -h, --help            show this help message and exit
  -d DIR, --data DIR    Path of the extracted embeddings
  -s DIR, --save-to DIR
                        Path to which we are going to save the calculated metrics (tensorboard)
  --model {original,peterson,dummy}
                        Which adaptation model to use ?. Default is original.
  -e N, --epochs N      Number of epochs. Default is 150.
  -r N, --runs N        Number of runs. Default is 1.
  -t N, --temperature N
                        Temperature value. Default is 1.
  --test-split N        Splitting percentage for the validation set. Default is 0.25.
  -k N [N ...], --topk N [N ...]
                        Top k list values. Default is 1 and 3.
  --gpu                 Perform training on gpu
  --cpu                 Perform training on cpu
"""

import utils
import argparse

import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from adaptation import *
from dataset import EmbDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from tqdm.auto import trange
from types import FunctionType
from typing import Dict, Iterable, List, Tuple


def training_step(
    model: nn.Module,
    loss_func: FunctionType,
    optimizer: optim.Optimizer,
    train_sets: Iterable[Dict],
    temperature: int,
    topk_list: List[int],
    device,
):
    """
    Training the model for one epoch.
    Generally it will take two dicts:
        - the first one is the extracted embds of the original dataset
        - the second one is augmented version of these embds
    """
    model.train()

    total_loss = 0

    for training_set in train_sets:
        left, right = training_set["left"], training_set["right"]

        out_left, out_right = model(left), model(right)

        loss = loss_func(out_left, out_right, temperature, device)

        total_loss += loss

    total_loss /= len(train_sets)

    # debate: include this in the loop above or not ?
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    metrics = {"Train/Loss": total_loss.item()}

    aR = utils.AsymmetricRecall(
        out_left.clone().detach().cpu(), out_right.clone().detach().cpu()
    )

    for k in topk_list:
        metrics[f"Train/top {k}"] = aR.eval(at=k)
    return metrics

def validate(
    model: nn.Module,
    loss_func: FunctionType,
    valid_set: Dict,
    temperature: int,
    topk_list: List[int],
    device,
):
    """
    Validating the model.
    """
    model.eval()

    with torch.no_grad():
        left, right = valid_set["left"], valid_set["right"]
        out_left, out_right = model(left), model(right)

        test_loss = loss_func(out_left, out_right, temperature, device)

        metrics = {"Test/Loss": test_loss.item()}

        aR = utils.AsymmetricRecall(out_left.detach().cpu(), out_right.detach().cpu())

        for k in topk_list:
            metrics[f"Test/top {k}"] = aR.eval(at=k)

        return metrics


def train(
    model: nn.Module,
    loss_func: FunctionType,
    train_set: Iterable[Dict],
    valid_set: Dict,
    num_epochs: int,
    save_to: str,
    temperature: int,
    topk_list: Iterable[int],
    device,
) -> Dict:
    """
    This will train the model with the given optimizer for {num_epochs} epochs.
    And it will save the metrics defined in training_step and validate in the given path {save_to}.
    """
    for k in topk_list:
        assert k > 0

    writer = SummaryWriter(save_to)
    metrics_per_run = {"train": dict(), "test": dict()}

    for epoch in trange(1, num_epochs + 1):
        info = training_step(
            model, loss_func, optimizer, train_set, temperature, topk_list, device
        )
        for metric, val in info.items():
            if not (metric in metrics_per_run["train"]):
                metrics_per_run["train"][metric] = []
            metrics_per_run["train"][metric].append(val)

            writer.add_scalar(metric, val, epoch)
            writer.flush()

        info = validate(model, loss_func, valid_set, temperature, topk_list, device)
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
    "--model",
    type=str,
    default="original",
    choices=["original", "peterson", "dummy"],
    help="Which adaptation model to use ?. Default is original.",
)
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    metavar="N",
    default=150,
    help="Number of epochs. Default is 150.",
)
parser.add_argument(
    "-r",
    "--runs",
    type=int,
    metavar="N",
    default=1,
    help="Number of runs. Default is 1.",
)
parser.add_argument(
    "-t",
    "--temperature",
    type=int,
    metavar="N",
    default=1,
    help="Temperature value. Default is 1.",
)
parser.add_argument(
    "--test-split",
    type=float,
    metavar="N",
    default=0.25,
    help="Splitting percentage for the validation set. Default is 0.25.",
)
parser.add_argument(
    "-k",
    "--topk",
    nargs="+",
    type=int,
    metavar="N",
    default=[1, 3],
    help="Top k list values. Default is 1 and 3.",
)

use_cuda_parser = parser.add_mutually_exclusive_group(required=False)
use_cuda_parser.add_argument(
    "--gpu", dest="use_cuda", action="store_true", help="Perform training on gpu"
)
use_cuda_parser.add_argument(
    "--cpu", dest="use_cuda", action="store_false", help="Perform training on cpu"
)
parser.set_defaults(use_cuda=True)

if __name__ == "__main__":
    args = parser.parse_args()
    print("temp ", args.temperature)

    if not Path(args.save_to).exists():
        is_cuda = args.use_cuda and torch.cuda.is_available()
        if not is_cuda:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        # load the dataset and split

        embds_dataset = EmbDataset(args.data, only_original=False)

        print("total ", len(embds_dataset))
        in_dim = embds_dataset[0][0]["left"].shape[0]

        # data = embds_dataset.load_all_to_device(device)

        splitted = utils.split_train_test(embds_dataset, args.test_split, device)
        train_set, valid_set = splitted["train"], splitted["test"]

        print("train set without aug size ", len(train_set[0]["left"]))
        print("train set with    aug size ", len(train_set[1]["left"]))
        print("test  set without aug size  ", len(valid_set["left"]))

        metrics_of_all_runs = []
        # optimizers_args = [{'lr': 1e-3}, {'lr': 0.0005, 'weight_decay': 1e-5},
        # {'lr': 1e-1, 'weight_decay': 1e-1}, {'lr': 1, 'weight_decay': 1e-3}]

        info = (
            "Embd path: "
            + str(args.data)
            + "\n"
            + "Using augmented images: "
            + str(embds_dataset.augmented)
            + "\n"
        )

        for run in trange(args.runs):
            run_path = Path(args.save_to) / f"run{run+1}"
            model = utils.construct_model(args.model, in_dim).to(device)

            writer = SummaryWriter(run_path)
            writer.add_text("Temperature: ", str(args.temperature))
            writer.add_text("Model", str(model).replace("\n", "  \n"))
            writer.add_text("Informations: ", info.replace("\n", "  \n"))

            if args.model != "dummy":
                optimizer = torch.optim.Adam(model.parameters())
                writer.add_text("Optimizer", str(optimizer).replace("\n", "  \n"))

                metrics_per_run = train(
                    model,
                    calc_loss,
                    train_set,
                    valid_set,
                    args.epochs,
                    run_path,
                    args.temperature,
                    args.topk,
                    device,
                )
                # save the model and the optimizer state_dics for this run
                torch.save(
                    {
                        "epoch": args.epochs,
                        "state_dict": model.state_dict(),
                        "optimzier": optimizer.state_dict(),
                    },
                    run_path / "model_and_optimizer.pt",
                )
            else:
                metrics_per_run = validate(
                    model,
                    calc_loss,
                    valid_set,
                    args.temperature,
                    args.topk,
                    device,
                )
                for metric, val in metrics_per_run.items():
                    writer.add_scalar(metric, val, 1)
                    writer.flush()
                    

            np.save(run_path / "data", metrics_per_run)

            metrics_of_all_runs.append(metrics_per_run)

            splitted = utils.split_train_test(embds_dataset, args.test_split, device)
            train_set, valid_set = splitted["train"], splitted["test"]

        # save all runs metrics into one file
        np.save(Path(args.save_to) / "all_runs", metrics_of_all_runs)

        avg_path = Path(args.save_to) / "avg"
        writer = SummaryWriter(avg_path)

        # TODO generalize this

        if args.runs > 1:
            # constructing the avg values for the saved metrics
            avg = dict()

            if args.model != "dummy":
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
            else:
                for k in metrics_of_all_runs[0].keys():
                    avg[k] = np.average(
                        [
                            metrics_of_all_runs[i][k]
                            for i in range(len(metrics_of_all_runs))
                        ],
                        axis=0,
                    )
                print(avg)

            # save avg of all runs
            np.save(Path(args.save_to) / "avg_of_all_runs", avg)

    else:
        print(
            "Found directory with the with the same saved name ! Will not perform training."
        )
