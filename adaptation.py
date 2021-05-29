"""This is where we define the different adaptation modules. The one we used in practice is Adaptation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class Adaptation(nn.Module):
    """
    Model that is responsible for adapting the extracted features.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.weight_matrix = nn.Linear(in_dim, 1024, bias=False)

    def forward(self, x):
        x = self.weight_matrix(x)
        x = torch.relu(x)
        return x

class AdaptationPeterson(torch.nn.Module):
    """
    See: https://cocosci.princeton.edu/papers/Peterson_et_al-2018-Cognitive_Science.pdf
    """
    def __init__(self, n_dim):
        super(AdaptationPeterson, self).__init__()
        self.W = torch.nn.Parameter(torch.ones(n_dim))

    def parameters(self):
        return [self.W]
    
    def forward(self, x):
        return x * torch.relu(self.W)
    
    def to(self, device):
        self.W = self.W.to(device)
        return self

class DummyNetwork(nn.Module):
    """
        Dummy network, there is no training done in this one.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def calc_loss(left, right, temp, device):
    """
    Our loss function that is used in training.
    """
    sim1 = utils.sim_matrix(left, right)
    sim2 = sim1.t()

    loss_left2right = F.cross_entropy(
        sim1 * temp, torch.arange(len(sim1)).long().to(device)
    )
    loss_right2left = F.cross_entropy(
        sim2 * temp, torch.arange(len(sim2)).long().to(device)
    )
    loss = loss_left2right * 0.5 + loss_right2left * 0.5

    return loss