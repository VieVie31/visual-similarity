import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import EmbDataset

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm, trange
from labml import lab, tracker, experiment


path = "C:\\Users\\Amine\\Desktop\\saved_features\\AdaptationProcessor\\resnet101\\resnet101.npy"
tracker.set_text("using : " + path.split("\\")[-1])

# ✨ Set the types of the stats/indicators.
# They default to scalars if not specified
tracker.set_queue("loss.train", 20, True)
tracker.set_histogram("loss.valid", True)
tracker.set_scalar("accuracy.valid", True)


config = {
    "epochs": 200,
    "validation_split": 0.25,
    "shuffle_dataset": False,
    "use_cuda": False,
    "seed": 5,
    "train_log_interval": 10,
    "learning_rate": 0.01,
    "temperature": 15,
    "topk": 1,
}

is_cuda = config["use_cuda"] and torch.cuda.is_available()
if not is_cuda:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:0")

embds_dataset = EmbDataset(path)


# Creating data indices for training and validation splits:
dataset_size = len(embds_dataset)
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


class Adaptation(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.weight_matrix = nn.Linear(in_dim, 1024, bias=False)

    def forward(self, x):
        x = self.weight_matrix(x)
        x = torch.relu(x)
        return x


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))

    return sim_mt


def calc_loss(left, right, temp):
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


def topk(left, right, k):
    sim = -sim_matrix(left, right)
    sorted_idx = sim.argsort(1)

    sens1 = np.array([lbl in sorted_idx[lbl][:k] for lbl in range(len(left))])
    sim = sim.t()
    sorted_idx = sim.argsort(1)
    sens2 = np.array([lbl in sorted_idx[lbl][:k] for lbl in range(len(left))])

    return (sens1 | sens2).mean()


def train(model, loss_func, optimizer, loader, device, model_log_interval):
    model.train()

    for i, (data, target) in enumerate(loader):
        left, right = data["left"], data["right"]
        left, right = left.to(device), right.to(device)

        out_left, out_right = model(left), model(right)

        loss = loss_func(out_left, out_right, config["temperature"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ✨ Increment the global step
        tracker.add_global_step(len(data))
        # ✨ Save stats
        tracker.save({"loss.train": loss})

        if (i + 1) % model_log_interval == 0:
            # ✨ Save model stats
            tracker.save(model=model)


def validate(model, loss_func, loader, device):
    model.eval()

    correct = 0
    with torch.no_grad():
        for data, target in loader:
            left, right = data["left"], data["right"]
            left, right = left.to(device), right.to(device)
            out_left, out_right = model(left), model(right)

            tracker.add(
                "loss.valid", loss_func(out_left, out_right, config["temperature"])
            )

    # **✨ Save stats**
    tracker.save(
        {f'top {config["topk"]}.valid': topk(out_left, out_right, config["topk"])}
    )
    tracker.save({f"top 5.valid": topk(out_left, out_right, 5)})


model = Adaptation(256)
optimizer = torch.optim.Adam(
    model.parameters(), lr=config["learning_rate"], weight_decay=1e-5
)

experiment.create(
    name="adapt",
    comment="Adaptation learning using extracted embds of " + path.split("\\")[-1],
)
experiment.configs(config)
experiment.add_pytorch_models(dict(model=model))

with experiment.start():
    for epoch in range(1, config["epochs"] + 1):
        train(
            model,
            calc_loss,
            optimizer,
            train_loader,
            device,
            config["train_log_interval"],
        )
        validate(model, calc_loss, valid_loader, device)
        tracker.new_line()

        # ✨ Save the models
        experiment.save_checkpoint()