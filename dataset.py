"""All our custom pytorch Dataset classes, the most important one is TTLDataset."""

import os

import torch
import numpy as np

from skimage import io
from pathlib import Path
from typing import Tuple, Dict
from torch.utils.data import Dataset


class TTLDataset(Dataset):
    """
    This is our pytorch TTL dataset, it is mainly used in the extracting part.
    """
    def __init__(self, root_dir: str, transform=None, target_transform=None):
        """
        Args:
            root_dir (str): Directory with all the images. left & right
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on a target (label).
        """
        assert isinstance(root_dir, str)

        self.root_dir = Path(root_dir)
        (
            self.left_images,
            self.right_images,
        ) = self.check_directories_and_get_images_paths(self.root_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.right_images)

    def __getitem__(self, idx: int) -> Tuple[Dict, str]:

        left_img_path = self.left_images[idx]
        right_img_path = self.right_images[idx]

        left_image = io.imread(left_img_path)
        right_image = io.imread(right_img_path)

        sample = {"left": left_image, "right": right_image}

        if self.transform:
            sample["left"] = self.transform(sample["left"])
            sample["right"] = self.transform(sample["right"])

        label = left_img_path.name
        if self.target_transform:
            label = self.target_transform(label)

        # same label for both and using name rather than stem to deal with the case of 2 imgs that
        # share the same name with diff extensions
        return (
            sample,
            label,
        )

    def check_directories_and_get_images_paths(self, root_dir):
        """
        Ensure that the way the root_dir has two subdirs left & right that contains the same number
        of images.
        This will not check that each pair is named the same way. 
        """
        assert os.path.isdir(root_dir)

        _, dirs, _ = next(os.walk(root_dir))

        assert "left" in dirs
        assert "right" in dirs

        _, _, left_images = next(os.walk(root_dir / "left"))
        _, _, right_images = next(os.walk(root_dir / "right"))

        assert len(left_images) > 0
        assert len(left_images) == len(right_images)

        return [root_dir / "left" / f for f in left_images], [
            root_dir / "right" / f for f in right_images
        ]


class SimpleDataset(Dataset):
    """
    Dataset class to load images that are located in some directory.
    The assumed structure is the following:
        Dir/*.[img_exts]
    """
    def __init__(self, root_dir: str, transform=None, target_transform=None):
        """
        Args:
            root_dir (str): Directory with all the images. left & right
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert isinstance(root_dir, str)

        self.root_dir = Path(root_dir)
        self.images = self.check_directories_and_get_images_paths(self.root_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:

        img_path = self.images[idx]

        image = io.imread(img_path)
        label = img_path.name

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return (
            image,
            label,
        )

    def check_directories_and_get_images_paths(self, root_dir):
        assert os.path.isdir(root_dir)

        _, dirs, _ = next(os.walk(root_dir))

        assert len(dirs) == 0

        _, _, images = next(os.walk(root_dir))

        assert len(images) > 0

        return [root_dir / f for f in images]


class EmbDataset(Dataset):
    """
    This is was initally created to load the embeddings i.e the extracted features, to keep up with 
    the pytorch training classical style using a DataLoader.
    Nevertheless, we always load all the embeddings at once since they are light in memory.

    [Warning]: this is still doesn't support the case of Simple dataset, it assumes that the embds
    are serialized in a form of a dictionnary that contains the following keys:
        ['left', 'right', 'left_name', 'right_name']
    """
    def __init__(self, path: str, only_original=True):
        """
        Args:
            path (str): the serialized file that contains the extracted embeddings.
            only_original (bool, optional): if is true, it means that we are going to use only the
            first half of the embeddings, and that is because of the way we constructed the TTLDataset
            when it is augmented, the features of the original images are located at the first half
            then followed by their augmented versions.
            In short, if you want to load all the embds, weather it contains augmented version or not
            use only_original=False, otherwise it will load only the first half.
        """
        self.embds = self._load_embds(path, only_original)
        self.augmented = not only_original
        
        assert "left" in self.embds and "left_name" in self.embds
        assert "right" in self.embds and "right_name" in self.embds
        assert len(self.embds["left"]) == len(self.embds["right"]) and len(
            self.embds["left"]
        ) == len(self.embds["left_name"])
        assert len(self.embds["right"]) == len(self.embds["right_name"])

    def _load_embds(self, path: str, only_original : bool) -> dict:
        e = np.load(path, allow_pickle=True).reshape(-1)[0]

        if 'left_ebds' in e.keys():
            e['left'] = e['left_ebds']
            e['right'] = e['right_ebds']

        if only_original:
            length = len(e['left'])
            e['left'] = e['left'][: length // 2]
            e['left_name'] = e['left_name'][: length // 2]
            e['right'] = e['right'][: length // 2]
            e['right_name'] = e['right_name'][: length // 2]

        return e

    def load_all_to_device(self, device):
        embds, _ = self[:]
        embds['left'].to(device)
        embds['right'].to(device)

        return embds

    def __getitem__(self, i):
        left, right = self.embds["left"][i], self.embds["right"][i]
        left, right = torch.from_numpy(left).float(), torch.from_numpy(right).float()

        return ({"left": left, "right": right}, self.embds["left_name"][i])

    def __len__(self):
        return len(self.embds["left"])