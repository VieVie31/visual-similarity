import torch

k = [1, 3]
#k.extend(list(range(5, 101, 5)))

config = {
    "epochs": 500,
    "test_split": 0.25,
    "shuffle_dataset": True,
    "use_cuda": True,
    "seed": 5,
    # "train_log_interval": 10,
    "temperature": 15,
    "topk": k,
    "runs" : 2
}
