import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import argparse
import os
from os import system, walk
from pathlib import Path


"""
    to execute the script give the path to the results as an argument
"""

def check_dirct(root_dir, to_get):
    ebds_list = dict ()

    assert os.path.isdir(root_dir)

    _, dirs, _ = next(os.walk(root_dir))
    assert len(dirs) > 0
    for dir in dirs:
        _, _, ebds = next(os.walk(Path(root_dir) / dir ))

        ebds_list[dir] = Path(root_dir) / dir / to_get
    return ebds_list


def average (runs, is_adapted = True):
    avgs_all_models =[]
    std_all_models =[]
    modelsnames =[]
    at = [1,3,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    for modelname, run in runs.items():
        data= np.load(run, allow_pickle=True).reshape(-1)[0]
        av = []
        st=[]
        for k in at:
            if is_adapted:
                data1= data["test"]["Test/top "+str(k)]
            else:
                data1= data["Test/top "+str(k)]

            av.append(np.average(data1))
            st.append(np.std(data1))

        avg= np.array(av)
        st= np.array(st)
        avgs_all_models.append(av)
        std_all_models.append(st)
        modelsnames.append(modelname)

    avgs_all_models= np.array(avgs_all_models)
    std_all_models= np.array(std_all_models)

    return avgs_all_models, std_all_models, modelsnames

parser = argparse.ArgumentParser(description="Training an adaptation model")
parser.add_argument(
    "-d",
    "--data",
    required=True,
    type=str,
    metavar="DIR",

)
parser.add_argument(
    "-s",
    "--sans",
    required=True,
    type=str,
    metavar="DIR",

)
args = parser.parse_args()
models = check_dirct(args.sans,"all_runs.npy")
models_adapted = check_dirct(args.data,"all_runs.npy")

print(models)

avgs_models_adapted, stds_models_adapted, modelsnames = average(models_adapted)
avgs_models, stds_models,modelsnames = average(models, False)

# print(avgs_models)
# print(avgs_models_adapted)
# print(len(avgs_models))
"""
    for each model get the average top k for all the runs and its standard deviation
"""


"""
    plot each recall and its standard deviation
"""
linestyle_str =  ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']


# clip_RN50x4
# dino_resnet50
# resnet50_ssl
# resnet101
# resnet50_swsl



fig, ax = plt.subplots()
clrs = sns.color_palette("Set2", 6)

clrs = ['red', 'blue', 'cyan', 'yellow', 'purple', 'orange']

with sns.axes_style("darkgrid"):

    at = [1,3,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    
    plt.set_cmap('Set1')

    for i in range(len(modelsnames)):
      meanst_adapted = avgs_models_adapted[i]
      sdt_adapted = stds_models_adapted[i]
      print(len(meanst_adapted))
      plt.plot(at, meanst_adapted, label=modelsnames[i], c=clrs[i])
      plt.xticks(at, at)

      ax.set_xlim(xmin=0)
      ax.set_ylim(ymin=0)
      plt.fill_between(at, meanst_adapted-sdt_adapted, meanst_adapted+sdt_adapted, alpha=0, facecolor=clrs[i])
      meanst = avgs_models[i]
      sdt = stds_models[i]
      plt.plot(at, meanst, linestyle='--', c=clrs[i])
      plt.fill_between(at, meanst-sdt, meanst+sdt ,alpha=0.3, facecolor=clrs[i])

      plt.xticks(at, at)
      
    ax.legend()

# ax.xaxis.set_major_locator(MultipleLocator())
ax.yaxis.set_major_locator(MultipleLocator(0.1))
plt.grid(which='minor', alpha=0.2)
plt.ylim([0,1])
plt.grid(which='major', alpha=0.5)
plt.title('Asymmetric Recall at')
plt.xlabel('at')
plt.ylabel('scores')
plt.show()
