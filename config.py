import torch

config = {
    "epochs": 5,
    "test_split": 0.25,
    "shuffle_dataset": False,
    "use_cuda": False,
    "seed": 5,
    # "train_log_interval": 10,
    "temperature": 15,
    "topk": list(range(1, 6)),
    "runs" : 5
}