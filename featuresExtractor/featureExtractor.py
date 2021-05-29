"""This is where we define our feature Extractor class, it is here where all the magic happens."""

import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.hooks import RemovableHandle
from typing import List, Dict
from types import FunctionType
from collections.abc import Iterable
from tqdm import tqdm

from dataset import TTLDataset, SimpleDataset
from .processor import Processor, AdaptationProcessor
from .utils import get_layer_index

from . import transforms as default_transforms


class FeatureExtractor:
    """
    Extract images intermediate layer[s] output[s] using some pretrained model[s].
    These extracted features will be then given to the processor who is responsible of doing something
    with them.
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
        self.hooks_dict = dict()

        self.printed_dataset_info = False 

    @torch.no_grad()
    def extract_features_from_directory(
        self, path: str, batch_size: int = 1, add_augmentation: bool = False
    ):
        """
        This will extract the features of the images located at path.

        Will inspect the directory, if it there is no subdirectories, i.e : dir/*.[image_extensions]
        it will create a simple dataset, otherwise it will assume that it follows the structure of
        TTL dataset i.e:
            .dir/
                left/*.[image_extensions]
                right/*.[image_extensions]

        Then it will construct the dataset with the corresponding transforms and call the forward method
        for each model.
        The models already have a forward hook registered to them ( this is done in the constructor ),
        and the hook implemented above will give them to the processor with the register method.

        Parameters:
        path (str)              : path of the directory that contains the RGB images.
        batch_size (int)        : batch size that will be used in the loader.
        add_augmentation (bool) : Add augmentation to the dataset or not ?
        """
        assert isinstance(path, str)
        batch_size = int(batch_size)
        assert batch_size >= 1
        assert isinstance(add_augmentation, bool)

        self.dataset_path = path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Extracting features with {len(self.models_names)} models !")

        t = tqdm(self.models_names)
        for model in t:
            model_name = model
            
            # we are going to need the original path in case we have a TTL dataset
            self.original_save_path = self.save_path

            # don't overwrite the old calculated features
            skip = self.__skip_if_found(model_name)
            if skip:
                print(f"[Info] Skipping {model_name}. Found old calculated embds for this model !")
                t.update(1)
                continue

            t.set_description(f"Models Loop: Loading {model_name} model...")
            t.refresh()

            # load model to gpu if it's available for faster computations
            model = self.__build_model(model)
            model = model.to(device)

            t.set_description(
                f"Models Loop: {model_name} is loaded. Extracting features..."
            )
            t.refresh()

            # pass the imgs into the model so we can capture the intermediate layers outputs.
            # self.corresponding_labels will be used in get_activation method below
            # to know the label for each img in the batch

            # construct the dataset and the loader
            dataset = self.__construct_dataset(path, model_name, add_augmentation)
            loader = DataLoader(dataset, batch_size=batch_size)

            

            # check before extracting features
            if isinstance(self.processor, AdaptationProcessor):
                if len(dataset) % loader.batch_size == 1:
                    raise ValueError(
                        f"[len(dataset): {len(dataset)} % Batch size: {loader.batch_size} = 1].\n"
                        "This batch size is inapproriate for this processor. "
                        "It wi'll fail when trying to normalize the batch that contains only one tensor !"
                    )
            # TODO: add checking for all processors that use PCA before extracting the features

            # now after the model is loaded, we'll test if we can use the provided batch size
            # we'll devide it by 2 until it fits the memory
            loader = self.__adapt_batch_size(model, loader, device)

            # we have determined a good batch size, let's register our hooks
            self.__register_hooks(model, model_name)

            is_ttl = dataset.__class__.__name__ == "TTLDataset"
            
            print("[INFO] Dataset is loaded.", flush=True)

            if is_ttl:
                with tqdm(total=len(loader), desc="Batch loop") as progress_bar:
                    for i, data in enumerate(loader, 0):
                        imgs, self.corresponding_labels = data

                        # move the left imgs to gpu if available, treat them, and then delete them
                        imgs["left"] = imgs["left"].to(device)
                        self.save_path = self.original_save_path / "left"
                        self.processor.set_path(self.save_path)
                        model(imgs["left"])
                        del imgs["left"]

                        # right imgs turn...
                        imgs["right"] = imgs["right"].to(device)
                        self.save_path = self.original_save_path / "right"
                        self.processor.set_path(self.save_path)
                        model(imgs["right"])
                        del imgs["right"]

                        progress_bar.update(1)

                self.save_path = self.original_save_path

            else:
                for i, data in tqdm(enumerate(loader, 0)):
                    imgs, self.corresponding_labels = data
                    imgs.to(device)  # move the imgs to gpu if it's available
                    model(imgs)
                    del imgs

            # We are done with this model, let's free the memory
            del model

            self.processor.set_path(self.original_save_path)
            self.processor.execute()

    def __adapt_batch_size(
        self, model: nn.Module, loader: DataLoader, device
    ) -> DataLoader:
        """
        Adapt the batch size so that both the model and the batch can fit in memory.
        """

        # changed this so it can work with ConcatDataset
        is_ttl = loader.dataset.__class__.__name__ == "TTLDataset"
        good_batch_size = False
        batch_size = loader.batch_size
        new_loader = DataLoader(loader.dataset, batch_size=loader.batch_size)

        while not good_batch_size:
            try:
                imgs, labels = next(iter(new_loader))
                imgs = imgs["left"].to(device) if is_ttl else imgs.to(device)
                model(imgs)

                return new_loader
            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_size //= 2
                    print(
                        f"\n[+] WARNING: ran out of memory, retrying with batch size = {batch_size}"
                    )
                    torch.cuda.empty_cache()

                    new_loader = DataLoader(loader.dataset, batch_size=batch_size)
                else:
                    raise e

    def __build_model(self, model_name: str):
        """
        This method serves as constructor for the model.
        What it does is:
            Get the lambda expression in self.models_dict[model_name].
            Construct the pytorch model.
            return the model
        """
        assert model_name in self.models_names

        m_dict = self.models_dict[model_name]
        # constructing the model
        model = m_dict["model"].__call__()

        return model

    def __construct_dataset(
        self, path: str, model_name: str, add_augmentation: bool = False
    ):
        """
        This method is used build the dataset and apply the corresponding transforms located at models
        dictionnary.
        """
        assert model_name in self.models_names
        assert isinstance(path, str)

        # get the corresponding transforms
        model_transforms = self.__get_transforms(model_name)

        # build the dataset
        _, dirs, _ = next(os.walk(path))

        if len(dirs) == 0:
            dataset_original = SimpleDataset(path, transform=model_transforms["data"])
        else:
            if not self.printed_dataset_info:
                print(
                    "The path you gave contains subdirectories, will assume it's a TTL like dataset."
                )
                self.printed_dataset_info = True

            dataset_original = TTLDataset(path, transform=model_transforms["data"])

        if add_augmentation:
            Corresponding_dataset_constructor = (
                SimpleDataset if len(dirs) == 0 else TTLDataset
            )
            dataset_aug = Corresponding_dataset_constructor(
                path,
                transform=model_transforms["data_aug"],
                target_transform=model_transforms["target_aug"],
            )

            dataset = ConcatDataset([dataset_original, dataset_aug])

            if isinstance(dataset_original, TTLDataset):
                dataset.__class__.__name__ = "TTLDataset"

            return dataset

        return dataset_original

    def __get_transforms(self, model_name: str) -> Dict:
        """
        Returns a dictionnary that contains the corresponding transforms of the model that will be applied
        to the dataset.
        """
        assert model_name in self.models_names
        m_dict = self.models_dict[model_name]

        data_transform = (
            m_dict["transform"]
            if "transform" in m_dict
            else default_transforms.pre_processing_transforms
        )

        """
        still need to settle on this one, do we assume that aug_transform is a complete one or just
        some transforms after the pre_processing done in m_dict["transform"] ?
        For the time being we'll assume that it is done with the second approach as it is more convenient
        to our case.
        """
        data_aug_transform = (
            m_dict["aug_transform"]
            if "aug_transform" in m_dict
            else default_transforms.data_aug_transform
        )
        data_aug_transform = transforms.Compose(
            [data_transform, default_transforms.data_aug_transform]
        )

        target_aug_transform = (
            m_dict["target_transform"]
            if "target_transform" in m_dict
            else default_transforms.target_aug_transform
        )

        return {
            "data": data_transform,
            "data_aug": data_aug_transform,
            "target_aug": target_aug_transform,
        }

    def __register_hooks(
        self, model: nn.Module, model_name: str
    ) -> List[RemovableHandle]:
        """
        This method will register the forward hooks on all the specified modules (layers)
        and will return all the removable handles.
        We'll make use of these handles to test the batch size.
        """
        assert isinstance(model, nn.Module)
        assert isinstance(model_name, str)

        handles = []
        m_dict = self.models_dict[model_name]

        # constructing the layer indices
        layers_indices = [
            layer if isinstance(layer, int) else get_layer_index(model, layer)
            for layer in m_dict["layers"]
        ]

        for layer_index in layers_indices:
            layer_name, module = list(model.named_modules())[layer_index]
            handles.append(
                module.register_forward_hook(
                    self.__get_activation(model_name, layer_name, self.processor)
                )
            )

        return handles

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
            # Processing is generally done on CPU, so let's move this tensor to cpu to save GPU memory
            processor.register(
                model_name, layer_name, output.detach().cpu(), self.corresponding_labels
            )

        return hook

    def __skip_if_found(self, model_name: str) -> bool:
        """
        This will see if we already used this model to extract the features or not.
        If there are files inside the corresponding directory, it will assume that we already did
        and will return True.
        """
        try:
            _, _, files = next(
                os.walk(self.original_save_path / self.processor.name / str(model_name))
            )

            if len(files) > 0:
                return True
        except StopIteration:
            return False

        return False