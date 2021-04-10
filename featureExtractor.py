import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict
from types import FunctionType
from collections.abc import Iterable
from tqdm import tqdm

from dataset import TTLDataset, SimpleDataset
from processor import Processor
from models import get_layer_index


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
                                     a dictionnary that contains the the callable function and
                                     the list of layers indices ( see models.py ).
        """
        assert isinstance(processor, Processor)
        assert isinstance(models_dict, Dict)

        for model_name, model_d in models_dict.items():
            assert isinstance(model_name, str)
            assert isinstance(model_d, Dict)
            assert isinstance(model_d["model"], FunctionType)
            assert isinstance(model_d["layers"], Iterable)

        self.processor = processor
        self.save_path = self.processor.save_path
        self.models_dict = models_dict
        self.models_names = list(models_dict.keys())

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
        dataset = None

        if len(dirs) == 0:
            dataset = SimpleDataset(path, transform=transforms)
        else:
            print(
                "The path you gave contains subdirectories, will assume it's a TTL like dataset."
            )
            dataset = TTLDataset(path, transform=transforms)
            
        loader = DataLoader(dataset, batch_size=batch_size)

        # check before extracting features
        if self.processor.__class__.__name__ == "AdaptationProcessor":
            if len(dataset) % loader.batch_size == 1:
                raise ValueError(
                    f"[len(dataset): {len(dataset)} % Batch size: {loader.batch_size} = 1].\n"
                    "This batch size is inapproriate for this processor. "
                    "It wi'll fail when trying to normalize the batch that contains only one tensor !"
                )

        # TODO: add checking for all processors that use PCA before extracting the features

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

        print(f"Extracting features with {len(self.models_names)} models !")

        t = tqdm(self.models_names)
        for model in t:
            model_name = model
            t.set_description(f"Models Loop: Loading {model_name} model...")
            t.refresh()
            
            # load model to gpu if it's available for faster computations
            model = self.__construct_model(model)
            model = model.to(device)

            t.set_description(
                f"Models Loop: {model_name} is loaded. Extracting features..."
            )
            t.refresh()

            # pass the imgs into the model so we can capture the intermediate layers outputs.
            # self.corresponding_labels will be used in get_activation method below
            # to know the label for each img in the batch

            if is_ttl:
                # special case here since we have pairs, and we need to change the save_path each time.
                self.original_save_path = self.save_path

                with tqdm(total=len(loader), desc="Batch loop") as progress_bar:
                    for i, data in enumerate(loader, 0):
                        imgs, self.corresponding_labels = data

                        # move the imgs to gpu if it's available
                        imgs["left"], imgs["right"] = (
                            imgs["left"].to(device),
                            imgs["right"].to(device),
                        )

                        # go
                        self.save_path = self.original_save_path / "left"
                        self.processor.set_path(self.save_path)
                        model(imgs["left"])

                        self.save_path = self.original_save_path / "right"
                        self.processor.set_path(self.save_path)
                        model(imgs["right"])

                        progress_bar.update(1)

                self.save_path = self.original_save_path

            else:
                for i, data in tqdm(enumerate(loader, 0)):
                    imgs, self.corresponding_labels = data
                    imgs.to(device)  # move the imgs to gpu if it's available
                    model(imgs)

            # We are done with this model, let's free the memory
            del model

            self.processor.set_path(self.original_save_path)
            self.processor.execute()

    def __construct_model(self, model_name: str):
        """
        This method serves as constructor for the model and will register the forward hook accordingly.
        What it does is:
            Get the lambda expression in self.models_dict[model_name].
            Construct the pytorch model.
            register the forward hook.

            return the model
        """
        assert model_name in self.models_names

        m_dict = self.models_dict[model_name]
        # constructing the model
        model = m_dict["model"].__call__()

        # constructing the layer indices
        layers_indices = [
            layer if isinstance(layer, int) else get_layer_index(model, layer)
            for layer in m_dict["layers"]
        ]

        for layer_index in layers_indices:
            layer_name, module = list(model.named_modules())[layer_index]
            module.register_forward_hook(
                self.__get_activation(model_name, layer_name, self.processor)
            )

        return model

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