import os
import time

import numpy as np

from pathlib import Path

"""
time.sleep(10)
print(10)
for i in range(1, 29):
    print((29 - i) * 1000, "secondes restantes avant le départ")
    time.sleep(1000)
"""

saved_features_path = Path("/media/olivier/KINGSTON/adaptation/saved_features/AdaptationProcessor")
save_metrics_path = Path("/media/olivier/KINGSTON/adaptation/xp_temperature_1")

_, dirs, _ = next(os.walk(saved_features_path))

features_paths = []
for d in dirs:
    features_paths.append(saved_features_path / d / str(d + '_with_pca_256.npy'))

for f, d in zip(features_paths, dirs):
    os.system("python train.py -d '" + str(f) + "' -s '" + str(save_metrics_path / d) + "' -r 20")
