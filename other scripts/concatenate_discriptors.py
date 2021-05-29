from pathlib import Path
import numpy as np
import os
import argparse

path = '../saved_features/AdaptationProcessor'


def load_embd(path):
    return np.load(path, allow_pickle=True).reshape(-1)[0]

def concatenate(features_path, with_pca=False, only_top_model=False):
    """
    For each xp ( aka 20 runs given some model ), calculate the std of top1.
    how?
        make a list of the top1 scores for each run ( max(top1 of all epochs) for each run )
        save std of this list
    """
    dirs = next(os.walk(Path(features_path)))[1]
    assert len(dirs) > 0
    
    if with_pca:
        first_embd = load_embd(Path(features_path) / dirs[0] / ( dirs[0] + '_with_pca_256.npy'))
    else:
        first_embd = load_embd(Path(features_path) / dirs[0] / ( dirs[0] + '.npy'))

    for i in range(1, len(dirs)):
        if with_pca:
            path = Path(features_path) / dirs[i] / ( dirs[i] + '_with_pca_256.npy')
        else:
            path = Path(features_path) / dirs[i] / ( dirs[i] + '.npy')

        # with np.load(path, allow_pickle=True).reshape(-1)[0] as e:
        e = np.load(path, allow_pickle=True).reshape(-1)[0]
        first_embd['left'] = np.concatenate((first_embd['left'], e['left']), axis=1)
        first_embd['right'] = np.concatenate((first_embd['right'], e['right']), axis=1)
       
    
    return first_embd


parser = argparse.ArgumentParser()

pca_parser = parser.add_mutually_exclusive_group(required=False)
pca_parser.add_argument("--with-pca", dest="pca", action="store_true")
pca_parser.add_argument("--no-pca", dest="pca", action="store_false")
parser.set_defaults(pca=False)

only_top_models_parser = parser.add_mutually_exclusive_group(required=False)
only_top_models_parser.add_argument("--only-top", dest="top", action="store_true")
only_top_models_parser.add_argument("--all", dest="top", action="store_false")
parser.set_defaults(top=False)

# parser.add_argument(
#     "-d", "--data", required=True, type=str, metavar="DIR", help="Path to saved features"
# )
parser.add_argument(
    "-s",
    "--save-to",
    required=True,
    type=str,
    metavar="DIR",
    help="Path to which we are going to save the concatenated features",
)

if __name__ == '__main__':
    args = parser.parse_args()

    c = concatenate(path, with_pca=args.pca, only_top_model=args.top)
    
    print(c['left'].shape)

    np.save('concatenate.npy', c)

