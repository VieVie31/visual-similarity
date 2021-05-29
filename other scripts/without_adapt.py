import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import my_utils
from config import config
from pathlib import Path
from train import validate
from dataset import EmbDataset
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

# hyperparams
runs = 20
device = 'cpu'

def calc_loss(left, right, temp, device):
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

class DummyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

saved_features_path = Path("/media/olivier/KINGSTON/adaptation/saved_features/AdaptationProcessor")
save_metrics_path = Path("/media/olivier/KINGSTON/adaptation/xp_no_learning_")

_, dirs, _ = next(os.walk(saved_features_path))

features_paths = []
features_paths_with_pca = []
for d in dirs:
    features_paths.append(saved_features_path / d / str(d + '.npy'))
    features_paths_with_pca.append(saved_features_path / d / str(d + '_with_pca_256.npy'))

all_features = {'concatenated' : features_paths, 'with_pca': features_paths_with_pca}

i = 1
for name, paths in all_features.items():    
    for f, d in zip(paths, dirs):
        print('embd path ', str(f))
        print('save path', str(save_metrics_path / name / d ))

        embd_path = str(f)
        save_to = str(save_metrics_path / name / d )
        
        
        if not Path(save_to).exists():
            metrics_of_all_runs = []

            # use one dummy network for all 
            dummy_model = DummyNetwork()

            for run in trange(runs):
                # load the dataset and split
                embds_dataset = EmbDataset(embd_path, only_original=False)
                # print('total ', len(embds_dataset))
                in_dim = embds_dataset[0][0]['left'].shape[0]

                splitted = my_utils.split_train_test(
                    embds_dataset, config["test_split"], device, config["seed"], config["shuffle_dataset"]
                )

                train_set, valid_set = splitted['train'], splitted['test']
                run_path = Path(save_to) / f"run{run+1}"
                writer = SummaryWriter(run_path)
                
                metrics_per_run = validate(dummy_model, calc_loss, valid_set)
                metrics_of_all_runs.append(metrics_per_run)

            # save all runs metrics into one file
            np.save(Path(save_to) / "all_runs", metrics_of_all_runs)

            avg_path = Path(save_to) / "avg"
            writer = SummaryWriter(avg_path)

            # TODO generalize this

            print(metrics_of_all_runs[0])

            if runs > 1:
                # constructing the avg values for the saved metrics
                avg = dict()

                for k in metrics_of_all_runs[0].keys():
                        avg[k] = np.average(
                            [
                                metrics_of_all_runs[i][k]
                                for i in range(len(metrics_of_all_runs))
                            ],
                            axis=0,
                        )

                # save avg of all runs
                print(avg)
                np.save(Path(save_to) / "avg_of_all_runs", avg)

        else:
            print("Found directory with the with the same saved name ! Will not perform training.")
