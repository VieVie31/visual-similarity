""" This is where we define our post processing function that will be used after the features extraction part"""
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.decomposition import PCA


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
    def register(self, x: torch.Tensor, name: str):
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

    def register(self, x: torch.Tensor, name: str):
        self.x = x
        self.name = name

        # saving will be done step by step, each time we register a feature we will save it.
        # the last one will be executed two times, but it's not that bad since we only overrite a file.
        self.execute()

    def execute(self):
        torch.save(self.x.squeeze(0), self.save_path / (self.name + ".pt"))


class PCAProcessor(Processor):
    """
    This Processor is responsible for fitting a PCA to all the extracted features, and apply the
    dimensionality reduction. The result will be saved in the save_path dir.
    """

    def __init__(self, save_path: str, out_dim: int):
        super().__init__(save_path)
        assert int(out_dim) > 0

        self.PCA = PCA(out_dim)
        self.all_features_dict = dict()

    def register(self, x: torch.Tensor, name: str):
        """
        Register the feature into the dictionnary after squeezing it and cast it to numpy array
        """

        self.all_features_dict[name] = x.squeeze().numpy()

    def execute(self):
        """
        Fit the PCA with the saved features, and then save them in their corresponding path
        """

        # if self.is_finished():
        reduced_features = self.PCA.fit_transform(list(self.all_features_dict.values()))

        for name, feature in zip(self.all_features_dict.keys(), reduced_features):
            x = torch.from_numpy(feature)
            torch.save(x, self.save_path / (name + ".pt"))  # x is already squeezed
