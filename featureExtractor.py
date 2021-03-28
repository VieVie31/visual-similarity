import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from dataset import TTLDataset, SimpleDataset
from processor import Processor
from pathlib import Path
from typing import List, Tuple, Dict
from types import FunctionType
from collections.abc import Iterable
from tqdm import tqdm


class FeatureExtractor:
    """
    Extract images features using a given pretrained model[s] and save them in the path provided.

    This will create a subdirectory for each model passed, and will serialize the captured features and
    put them into the corresponding dir.
    """

    def __init__(
        self, processor: Processor, models_dict: Dict, layers_indices: List[int] = [-2]
    ) -> None:
        """
        Parameters:
        processor (Processor)      : The Processor that is going to handle the extracted features.

        models_dict (str)          : dictionnary that contains the models that we are going to use
                                     where the keys are the names of the models and the values are
                                     nn.Module instances

        layers_indices (List[int]) : Which layers we are going to use to extract the features.
        """
        assert isinstance(processor, Processor)
        assert isinstance(models_dict, Dict)

        for model_name, model in models_dict.items():
            assert isinstance(model_name, str)
            assert isinstance(model, nn.Module)

        self.models = list(models_dict.values())
        self.processor = processor
        self.save_path = self.processor.save_path

        for model_name, model in models_dict.items():
            for layer_index in layers_indices:
                layer_name, module = list(model.named_modules())[layer_index]
                module.register_forward_hook(
                    self.get_activation(model_name, layer_name, processor)
                )

    def get_activation(self, model_name: str, layer_name: str, processor: Processor):
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
            save_path = self.save_path / str(model_name) / str(layer_name)
            save_path.mkdir(parents=True, exist_ok=True)
            self.processor.set_path(save_path)

            name = None
            try:
                name = str(next(self.image_iterator))
            except StopIteration:
                self.image_iterator.reset()
                name = str(next(self.image_iterator))

            processor.register(output.detach(), name)

        return hook

    def extract_features_ttl_dataset(self, dataset: TTLDataset):
        # special case here, because each datapoint is actually a pair of images
        assert len(dataset) > 0 and isinstance(dataset, TTLDataset)
        original_save_path = self.save_path

        for pair, label in tqdm(dataset):
            left, right = pair.values()

            self.save_path = original_save_path / "left"
            self._extract_features([left], ["left_" + str(label)])
            self.save_path = original_save_path / "right"
            self._extract_features([right], ["right_" + str(label)])

            self.save_path = original_save_path

    def extract_features_from_directory(self, path: str):
        """
        This will extract the features of the images located at path.
        Assuming that the structure of the directory located at path is:

            dir/*.[image_extensions]
        """
        assert isinstance(path, str)

        dataset = SimpleDataset(path, transform=transforms.ToTensor())

        for img, label in tqdm(dataset):
            self._extract_features([img], [str(label)])

    def _extract_features(self, images: Iterable, labels: Iterable = []):
        """
        This is the function that calls the forward method for each model on each image in images arg.
        The models already have a forward hook registered to them ( this is done in the constructor ),
        and the hook implemented above will save the calculated features and serialize them.


        Parameters:
        images (Iterable): images must be tensor that have a size of 3 i.e len(img.size()) == 3
                           batches won't work with this implementation !

        labels (Iterable): an iterable that contains the labels of the passed images.
        """
        assert len(images) > 0

        if len(labels) == 0:
            labels = [i + 1 for i in range(len(images))]
        else:
            assert len(images) == len(labels)
        for img in images:
            assert isinstance(img, torch.Tensor)
        self.image_iterator = np.nditer([labels])

        """
            Why are we using this iterator above ?
            
            It is necessary to register only one hook for each module, which is some how constant even if we
            used the idea of a super function to give more information to the hook so we can save the model_name
            and the layer_name.
            The problem is that we have multiple labels that change each iteration, and since we can't register
            multiple hooks ( we can but that would be stupid ), we need to pass somehow each label to the hook,
            hence the idea of using an iterator as a field of the current object ( so it can be accessed by the hook
            function ).

            And we are using numpy iterator instead of iter() in order to make use of reset() 
            in case we need to compute features with multiple layers for each model.
        """

        for model in self.models:
            for img in images:
                model(img.unsqueeze(0))

        # self.processor.execute()