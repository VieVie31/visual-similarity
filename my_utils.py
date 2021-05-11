import clip
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))

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
    dataset: Dataset, test_split: float, seed: int = 5, shuffle: bool = False
) -> Tuple[DataLoader, DataLoader]:
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(train_indices), sampler=train_sampler
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(val_indices), sampler=valid_sampler
    )

    return train_loader, valid_loader

# used in models.py

def get_layer_index(model: nn.Module, layer_name: str):
    assert isinstance(model, nn.Module)
    assert isinstance(layer_name, str)
    assert hasattr(model, layer_name)

    return list(model.modules()).index(getattr(model, layer_name))

def load_clip_model(name, *args):
    model, _ = clip.load(name) #clip.load('RN50x4')
    model = clip.model.build_model(model.state_dict())
    model = model.visual.float().eval()

    return model

def get_clip_transforms(name):
    _, t = clip.load(name)
    t = transforms.Compose([transforms.ToPILImage(), t])
    return t