import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from dataset import TTLDataset, SimpleDataset
from processor import Processor
from pathlib import Path
from typing import List, Tuple, Dict
from types import FunctionType
from collections.abc import Iterable
from tqdm import tqdm


class FeatureExtractor:
    """
    Extract images intermediate layer[s] output[s] using some pretrained model[s].
    These extracted features will be then given to the processor who is responsible of doing something with them.
    """

    def __init__(self, processor: Processor, models_dict: Dict) -> None:
        """
        Parameters:
        processor (Processor)      : The Processor that is going to handle the extracted features.

        models_dict (str)          : dictionnary that contains the models that we are going to use
                                     where the keys are the names of the models and the values are
                                     a dictionnary that contains the nn.Module object and the list of layers
                                     indices ( see models.py )
        """
        assert isinstance(processor, Processor)
        assert isinstance(models_dict, Dict)

        for model_name, model_d in models_dict.items():
            assert isinstance(model_name, str)
            assert isinstance(model_d, Dict)
            assert isinstance(model_d["model"], nn.Module)
            assert isinstance(model_d["layers"], Iterable)

        self.processor = processor
        self.save_path = self.processor.save_path

        self.models = [d["model"] for d in list(models_dict.values())]

        for model_name, model_d in models_dict.items():
            model, layers_indices = model_d["model"], model_d["layers"]

            for layer_index in layers_indices:
                layer_name, module = list(model.named_modules())[layer_index]
                module.register_forward_hook(
                    self.__get_activation(model_name, layer_name, processor)
                )

    @torch.no_grad()
    def extract_features_from_directory(
        self, path: str, transforms=None, batch_size: int = 1
    ):
        """
        This will extract the features of the images located at path.

        Will inspect the directory, if it there is no subdirectories, i.e : dir/*.[image_extensions]
        it will create a simple dataset, otherwise it will assume that it follows the structure of
        TTL dataset i.e:
            .dir/
                left/*.[image_extensions]
                right/*.[image_extensions]

        Parameters:
        path (str)       : path of the directory that contains the RGB images.
        batch_size (int) : batch size that will be used in the loader.
        """
        assert isinstance(path, str)

        _, dirs, _ = next(os.walk(path))
        loader = None

        if len(dirs) == 0:
            dataset = SimpleDataset(path, transform=transforms)
            loader = DataLoader(dataset, batch_size=batch_size)
        else:
            print(
                "The path you gave contains subdirectories, will assume it's a TTL like dataset."
            )
            ttl = TTLDataset(path, transform=transforms)
            loader = DataLoader(ttl, batch_size=batch_size)

        self._extract_features(loader)

    @torch.no_grad()
    def _extract_features(self, loader: DataLoader):
        """
        This is the function that calls the forward method for each model when iterating over
        the passed dataloader.
        The models already have a forward hook registered to them ( this is done in the constructor ),
        and the hook implemented above will give them to the processor with the register method.


        Parameters:
        loader (DataLoader): torch.utils.data.Dataloader
        """
        assert isinstance(loader, DataLoader)

        is_ttl = isinstance(loader.dataset, TTLDataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for model in tqdm(self.models, desc="Models loop -> "):
            # load model to gpu if it's available for faster computations
            model.to(device)

            # pass the imgs into the model so we can capture the intermediate layers outputs.
            # self.corresponding_labels will be used in get_activation method below
            # to know the label for each img in the batch

            if is_ttl:
                # special case here since we have pairs, and we need to change the save_path each time.
                original_save_path = self.save_path

                for i, data in tqdm(
                    enumerate(loader, 0), desc="Current batch -> ", leave=False
                ):
                    imgs, self.corresponding_labels = data

                    # move the imgs to gpu if it's available
                    imgs["left"].to(device)
                    imgs["right"].to(device)

                    # go
                    self.save_path = original_save_path / "left"
                    self.processor.set_path(self.save_path)
                    model(imgs["left"])

                    self.save_path = original_save_path / "right"
                    self.processor.set_path(self.save_path)
                    model(imgs["right"])

                self.save_path = original_save_path

            else:
                for i, data in tqdm(enumerate(loader, 0)):
                    imgs, self.corresponding_labels = data
                    imgs.to(device)  # move the imgs to gpu if it's available
                    model(imgs)

            # re-move the model from gpu if it was available to cpu
            model.to(torch.device("cpu"))

            self.processor.execute()

    @torch.no_grad()
    def __get_activation(self, model_name: str, layer_name: str, processor: Processor):
        """
        A sort of a wrapper for the hook function in order to use the provided arguments to save
        the features into files appropriately.

        Returns:
            the hook function that will be used in register_forward_hook method.
        """

        def hook(model, inp, output) -> None:
            """
            Hook function, this is where we are placing our processor to handle the extracted features
            """

            processor.register(
                model_name, layer_name, output.detach(), self.corresponding_labels
            )

        return hook