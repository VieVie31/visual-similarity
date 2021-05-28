from pathlib import Path
import numpy as np
import os

def calc_stds(xp_path):
    dirs = next(os.walk(Path(xp_path)))[1]
    assert len(dirs) > 0

    std_dict = dict()

    for d in dirs:
        avg_run = np.load(Path(xp_path) / d / 'avg_of_all_runs.npy', allow_pickle=True).reshape(-1)[0]
        
        argmax = np.argmax(avg_run['Test/top 1'])
        all_runs = np.load(Path(xp_path) / d / 'all_runs.npy', allow_pickle=True).reshape(-1)
        l = [run['test']['Test/top 1'][argmax] for run in all_runs]
        # if d == 'clip_rn50x4':
        #     print(l)
        std_dict[d] = np.std(l)

    return std_dict

def get_scores(xp_path):
    dirs = next(os.walk(Path(xp_path)))[1]
    assert len(dirs) > 0

    score_dict = dict()

    for d in dirs:
        if 'BiT' in d:
            avg_run = np.load(Path(xp_path) / d / 'avg_of_all_runs.npy', allow_pickle=True).reshape(-1)[0]
            score_dict[d] = avg_run['Test/top 1']

    return score_dict

def calc_std_no_learning(xp_path):
    dirs = next(os.walk(Path(xp_path)))[1]
    assert len(dirs) > 0

    std_dict = dict()

    for d in dirs:
        all_runs = np.load(Path(xp_path) / d / 'all_runs.npy', allow_pickle=True).reshape(-1)
        l = [run['Test/top 1'] for run in all_runs]
        # if d == 'clip_rn50x4':
        #     print(l)
        std_dict[d] = np.std(l)

    return std_dict

if __name__ == '__main__':
    # xp_path = 'D:/adaptation/xp_temperature_1'
    # print(xp_path)

    # std = (calc_stds(xp_path))

    # for name, val in std.items():
    #     print(name, ": ", val)

    xp_path = 'D:/adaptation/xp_no_learning_/concatenated'
    print(xp_path)

    score_dict = calc_std_no_learning(xp_path)

    for name, val in score_dict.items():
        print(name, ": ", val)