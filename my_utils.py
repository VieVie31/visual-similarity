from dataset import EmbDataset
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from torch.utils import data
import torchvision.transforms as transforms

from sklearn.preprocessing import normalize as l2
from sklearn.neighbors import NearestNeighbors

# def sim_matrix(a, b, eps=1e-8):
#     """
#     added eps for numerical stability
#     """
#     a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
#     a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
#     b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
#     sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))

#     return sim_mt

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm((a_norm + 1) ** 1.5, (b_norm+1).transpose(0, 1) ** 1.5)
    return sim_mt


def topk(left, right, k):
    sim = -sim_matrix(left, right)
    sorted_idx = sim.argsort(1)

    sens1 = np.array([lbl in sorted_idx[lbl][:k] for lbl in range(len(left))])
    sim = sim.t()
    sorted_idx = sim.argsort(1)
    sens2 = np.array([lbl in sorted_idx[lbl][:k] for lbl in range(len(left))])

    return (sens1 | sens2).mean()


def split_train_test(
    dataset: EmbDataset,
    test_split: float,
    device,
    seed: int = 5,
    shuffle: bool = False,
) -> Tuple[Dict]:
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

    #if shuffle:
    #    np.random.seed(seed)
    #    np.random.shuffle(indices)
    np.random.shuffle(indices) # To get different splits each time


    train_indices_normal, test_indices = indices[split:], indices[:split]
    embds = dataset.load_all_to_device(device)
    #left, right = embds["left"], embds["right"]
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
        splitted["train"].append({
                "left": left[train_indices_aug],
                "right": right[train_indices_aug],
            })

    return splitted


# used in models.py


def get_layer_index(model: nn.Module, layer_name: str):
    assert isinstance(model, nn.Module)
    assert isinstance(layer_name, str)
    assert hasattr(model, layer_name)

    return list(model.modules()).index(getattr(model, layer_name))


def load_clip_model(name, *args):
    model, _ = clip.load(name)  # clip.load('RN50x4')
    model = clip.model.build_model(model.state_dict())
    model = model.visual.float().eval()

    return model


def get_clip_transforms(name):
    _, t = clip.load(name)
    t = transforms.Compose([transforms.ToPILImage(), t])
    return t


class AsymmetricRecall:
    def __init__(self, left_ebds, right_ebds):
        from sklearn.metrics.pairwise import cosine_similarity

        self.M = -cosine_similarity(left_ebds, right_ebds)
        self.sorted_idx_1 = self.M.argsort(1)
        self.sorted_idx_2 = self.M.T.argsort(1) #TODO: make faster with arg partition ?


    def eval(self, at):
        sens1 = np.array([lbl in self.sorted_idx_1[lbl][:at] for lbl in range(len(self.M))])
        sens2 = np.array([lbl in self.sorted_idx_2[lbl][:at] for lbl in range(len(self.M))])

        return (sens1 | sens2).mean()

