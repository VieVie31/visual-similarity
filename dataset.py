import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io
from pathlib import Path
from typing import Tuple, Dict


class TTLDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images. left & right
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert isinstance(root_dir, str)

        self.root_dir = Path(root_dir)
        (
            self.left_images,
            self.right_images,
        ) = self.check_directories_and_get_images_paths(self.root_dir)
        self.transform = transform

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

        # same label for both and using name rather than stem to deal with the case of 2 imgs that
        # share the same name with diff extensions
        return (
            sample,
            left_img_path.name,
        )

    def check_directories_and_get_images_paths(self, root_dir):
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
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images. left & right
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert isinstance(root_dir, str)

        self.root_dir = Path(root_dir)
        self.images = self.check_directories_and_get_images_paths(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Dict, str]:

        img_path = self.images[idx]

        image = io.imread(img_path)

        if self.transform:
            image = self.transform(image)

        return (
            image,
            img_path.name,
        )

    def check_directories_and_get_images_paths(self, root_dir):
        assert os.path.isdir(root_dir)

        _, dirs, _ = next(os.walk(root_dir))

        assert len(dirs) == 0

        _, _, images = next(os.walk(root_dir))

        assert len(images) > 0

        return [root_dir / f for f in images]