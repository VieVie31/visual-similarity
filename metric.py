import torch
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def calc_loss(left, right, temp, device):
    sim1 = sim_matrix(left, right)
    sim2 = sim1.t()
    loss_left2right = F.cross_entropy(
        sim1 * temp, torch.arange(len(sim1)).long().to(device)
    ).to(device)
    loss_right2left = F.cross_entropy(
        sim2 * temp, torch.arange(len(sim2)).long().to(device)
    ).to(device)
    loss = loss_left2right * 0.5 + loss_right2left * 0.5

    return loss
