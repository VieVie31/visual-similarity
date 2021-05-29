"""Util functions that are useful for training."""

import torch
import numpy as np
from dataset import EmbDataset
from typing import Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from adaptation import Adaptation, AdaptationPeterson, DummyNetwork

def construct_model(name : str, in_dim : int):
    """Construct the appropriate adaptation module from the passed string."""
    assert name in ("original", "peterson", "dummy")
    in_dim = int(in_dim)
    assert in_dim > 0

    if name == "original":
        return Adaptation(in_dim)
    if name == "peterson":
        return AdaptationPeterson(in_dim)

    return DummyNetwork()

def sim_matrix(a, b, eps=1e-8):
    """
    Calculate the similarity matrix given two lists of embeddings.
    added eps for numerical stability.
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))

    return sim_mt


def new_sim_matrix(a, b, eps=1e-8):
    """
    Calculate the similarity matrix given two lists of embeddings.
    added eps for numerical stability.
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm((a_norm + 1) ** 1.5, (b_norm + 1).transpose(0, 1) ** 1.5)
    return sim_mt


def split_train_test(dataset: EmbDataset, test_split: float, device) -> Tuple[Dict]:
    """
    Split the given dataset into train and test sets using test_split percentage.
    If augmented_dataset is True, it will assume that it is a concated dataset where the original
    datapoints are stocked first then followed by their augmented versions.
    """
    assert isinstance(dataset, EmbDataset)
    # Creating data indices for training and validation splits:
    augmented_dataset = dataset.augmented
    dataset_size = len(dataset) if not augmented_dataset else len(dataset) // 2
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))

    np.random.shuffle(indices)

    train_indices_normal, test_indices = indices[split:], indices[:split]
    embds = dataset.load_all_to_device(device)
    left, right = embds["left"].to(device), embds["right"].to(device)

    splitted = {
        "train": [
            {
                "left": left[train_indices_normal],
                "right": right[train_indices_normal],
            }
        ],
        "test": {
            "left": left[test_indices],
            "right": right[test_indices],
        },
    }

    if augmented_dataset:
        train_indices_aug = [i + len(dataset) // 2 for i in train_indices_normal]
        splitted["train"].append(
            {
                "left": left[train_indices_aug],
                "right": right[train_indices_aug],
            }
        )

    return splitted


class AsymmetricRecall:
    """
    This class is responsible for calculating the topk matches to our query.
    """
    def __init__(self, left_ebds, right_ebds):

        self.M = -cosine_similarity(left_ebds, right_ebds)
        self.sorted_idx_1 = self.M.argsort(1)
        self.sorted_idx_2 = self.M.T.argsort(1)  # TODO: make faster with arg partition ?

    def eval(self, at):
        sens1 = np.array(
            [lbl in self.sorted_idx_1[lbl][:at] for lbl in range(len(self.M))]
        )
        sens2 = np.array(
            [lbl in self.sorted_idx_2[lbl][:at] for lbl in range(len(self.M))]
        )

        return (sens1 | sens2).mean()