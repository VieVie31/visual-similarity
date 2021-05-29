import os
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

path = "../saved_features/AdaptationProcessor"
out_dim = 256

def check_dirct(root_dir):
    ebds_list = dict ()

    assert os.path.isdir(root_dir)

    _, dirs, _ = next(os.walk(root_dir))

    assert len(dirs) > 0

    for dir in dirs:
        _, _, ebds = next(os.walk(root_dir+"/"+dir))

        file = dir+".npy"
        ebds_list[dir] = Path(root_dir) / dir / file


    return ebds_list

embds_paths = check_dirct(path)

for name, embd in embds_paths.items():
    print(embd)
    
    if Path('/'.join(str(embd).replace('\\', '/').split('/')[:-1]) + '/pca.npy').exists():
        print(f"already exists, pass and do not recompute it...")
        continue

    embds = np.load(embd, allow_pickle=True).reshape(-1)[0]

    left_right = np.concatenate([embds['left'], embds['right']])
    print('old shape: ', left_right.shape)

    pca = PCA(out_dim)
    pca.fit(left_right)

    d = dict()
    d['left'] = pca.transform(embds['left'])
    d['right'] = pca.transform(embds['right'])
    d['left_name'] = embds['left_name']
    d['right_name'] = embds['right_name']

    # print(str(embd).split('.npy')[0])
    embd = str(embd).replace('\\', '/')
    
    print(embd.split('.npy')[0] + '_with_pca')
    np.save(embd.split('.npy')[0] + '_with_pca_' + str(out_dim), d)
    np.save('/'.join(embd.split('/')[:-1]) + '/pca', pca)
    
