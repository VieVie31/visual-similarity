""" This is where we define our post processing function that will be used after the features extraction part"""
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.decomposition import PCA
from collections.abc import Iterable


class Processor(ABC):
    """
    Our base class. Each Processor must implement the following methods.
    """

    def __init__(self, save_path: str):
        assert isinstance(save_path, str)

        self.save_path = Path(save_path)

    def set_path(self, path: Path):
        assert isinstance(path, Path)

        self.save_path = path

    @abstractmethod
    def register(
        self, model_name: str, layer_name: str, imgs_batch: torch.Tensor, names: str
    ):
        """
        Figuring out when to call execute will depend on what operations are done by the processor.
        Generally, this method should figure out when to call the execute method.

        Parameters:
        model_name (str)    : name of the current model
        layer_name (str)    : name of the layer of the current model
        imgs_batch (Tensor) : a batch of RGB imgs
        names (Iterable)    : the corresponding labels of these imgs
        """
        raise NotImplementedError("Please implement this method")

    @abstractmethod
    def execute(self):
        """
        This is where our post processing logic needs to be implemented.
        Once you registered an extracted feature, this is where to define what are you going to do with it.
        """
        raise NotImplementedError("Please implement this method")


class SaveProcessor(Processor):
    """
    This Processor is responsible for serializing and saving each feature in the given path.
    """

    def register(
        self,
        model_name: str,
        layer_name: str,
        imgs_batch: torch.Tensor,
        names: Iterable,
    ):
        assert isinstance(model_name, str)
        assert isinstance(layer_name, str)
        assert isinstance(names, Iterable)
        assert isinstance(imgs_batch, torch.Tensor)
        assert imgs_batch.size()[0] == len(names)

        self.model_name = model_name
        self.layer_name = layer_name
        self.imgs = imgs_batch
        self.names = names

        # saving will be done step by step, each time we register a feature we will save it.
        # the last one will be executed two times, but it's not that bad since we only overrite a file.
        self.execute()

    def execute(self):
        save_path = (
            self.save_path
            / str(self.__class__.__name__)
            / str(self.model_name)
            / str(self.layer_name)
        )
        save_path.mkdir(parents=True, exist_ok=True)

        for img, name in zip(self.imgs, self.names):
            # must use clone otherwise it'll save the whole batch tensor
            # see: https://discuss.pytorch.org/t/saving-tensor-with-torch-save-uses-too-much-memory/46865/2
            # and: https://pytorch.org/docs/stable/tensor_view.html
            torch.save(img.clone(), save_path / (name + ".pt"))


class PCAProcessor(Processor):
    """
    This Processor is responsible for fitting a PCA to all the extracted features, and apply the
    dimensionality reduction. The result will be saved in the save_path dir.
    """

    def __init__(self, save_path: str, out_dim: int):
        assert int(out_dim) > 0
        super().__init__(save_path)

        self.PCA = PCA(out_dim)
        self.out_dim = out_dim
        self.all_features_dict = dict()

        self.called_register = False

    def register(
        self, model_name: str, layer_name: str, imgs_batch: torch.Tensor, names: str
    ):
        """
        Register the feature into the dictionnary after squeezing it and cast it to numpy array
        """
        assert isinstance(model_name, str)
        assert isinstance(layer_name, str)
        assert len(imgs_batch.size()) == 4
        assert imgs_batch.size()[0] == len(names)

        self.model_name = model_name

        self.last_layer_name = layer_name
        self.layer_name = layer_name

        for img, name in zip(imgs_batch, names):
            self.all_features_dict[name] = img.squeeze().numpy()

    def execute(self):
        """
        Fit the PCA with the saved features, and then save them in their corresponding path
        """

        pca = self.PCA.fit(list(self.all_features_dict.values()))
        reduced_features = pca.transform(list(self.all_features_dict.values()))

        save_path = (
            self.save_path
            / str(self.__class__.__name__)
            / str(self.model_name)
            / str(self.layer_name)
        )
        save_path.mkdir(parents=True, exist_ok=True)

        for name, feature in zip(self.all_features_dict.keys(), reduced_features):
            torch.save(torch.from_numpy(feature), save_path / (name + ".pt"))