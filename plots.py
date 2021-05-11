import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import argparse
import os
from os import walk
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
        assert len(ebds) >= 2
        if to_get =="_with_pca_256.npy":
            file = dir+to_get
        else:
            file = to_get
        ebds_list[dir] = Path(root_dir) / dir / file

    return ebds_list

parser = argparse.ArgumentParser(description="Training an adaptation model")
parser.add_argument(
    "-d",
    "--data",
    required=True,
    type=str,
    metavar="DIR",
    help="Path of the results",
)
args = parser.parse_args()
avgs = check_dirct(args.data,"avg_of_all_runs.npy")
stds = check_dirct(args.data,"std_of_all_runs.npy")
avgs_all_models =[]
std_all_models =[]
modelsnames =[]

"""
    for each model get the average top k for all the runs and its standard deviation
"""
for modelname, avg in avgs.items():
    data_avg= np.load(avg, allow_pickle=True).reshape(-1)[0]
    data_std = np.load(stds[modelname], allow_pickle=True).reshape(-1)[0]
    av = []
    st=[]
    for k in range(0, 100, 5):
        if k==0:
            data1= data_avg["Test/top "+str(k+1)]
            data2 = data_std["Test/top "+str(k+1)]
        else:
            data1= data_avg["Test/top "+str(k)]
            data2 = data_std["Test/top "+str(k)]
        av.append(np.average(data1, axis =0))
        st.append(np.average(data2, axis =0))

    av = np.insert(av,0,0)
    st = np.insert(st,0,0)
    avgs_all_models.append(av)
    std_all_models.append(st)
    modelsnames.append(modelname)

"""
    plot each recall and its standard deviation
"""
fig, ax = plt.subplots()
clrs = sns.color_palette("husl", 5)
with sns.axes_style("darkgrid"):

    at = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    for i in range(2):
      meanst = avgs_all_models[i]
      sdt = std_all_models[i]
      plt.plot(at, meanst, label =modelsnames[i],c=clrs[i+2])
      ax.set_xlim(xmin=0)
      ax.set_ylim(ymin=0)
      plt.fill_between(at, meanst-sdt, meanst+sdt ,alpha=0.3, facecolor=clrs[i+2])
    ax.legend()

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
plt.grid(which='minor', alpha=0.2)
plt.grid(which='major', alpha=0.5)
plt.show()
