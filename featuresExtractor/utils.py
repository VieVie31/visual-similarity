"""Util functions that we use for extracting features."""

import clip
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
from .processor import SaveProcessor, PCAProcessor, AdaptationProcessor, AdaptationProcessorPCA

class ValidateProcessor(argparse.Action):
    """
    Validator for our arg parser that is used the validate the argument for the processor parameter.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        processor = values[0].lower()

        if processor not in ("save", "pca", "adapt", "adapt-pca"):
            raise ValueError(f"invalid processor {values[0]}")

        if processor == "pca" or processor == "adapt-pca":
            if len(values) > 2:
                raise ValueError(
                    f"Too much arguments ! pca must be followed only by 1 number (out dim)"
                )
            elif len(values) < 2:
                raise ValueError(f"Need out dim for the pca !")

            if processor == "pca":
                processor = PCAProcessor(namespace.save_to, int(values[1]))
            else:
                processor = AdaptationProcessorPCA(namespace.save_to, int(values[1]))

        elif processor == "adapt":
            processor = AdaptationProcessor(namespace.save_to)
        else:
            processor = SaveProcessor(namespace.save_to)

        setattr(namespace, self.dest, processor)


# used in models.py

def get_layer_index(model: nn.Module, layer_name: str):
    """
    Given a model and a layer_name (str), this will return its corresponding index in 
    the list model.modules(). That is given some nn.Module object M that is composed of other nn.Modules
    [m_1, m_2, m_3..., m_n], if we can access these sub_modules like M.sub_module_name, then this
    function will return the corresponding index of this submodule.
    
    [Warning] : This doesn't work for nested submodules !


    Example:
        resnet = torchvision.models.resnet18(pretrained=True).eval()
            ResNet(
                (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                ...
                ...
                (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
                (fc): Linear(in_features=512, out_features=1000, bias=True)
            )

        In [3]: utils.get_layer_index(resnet, 'avgpool')
        Out[3]: 66

    """
    assert isinstance(model, nn.Module)
    assert isinstance(layer_name, str)
    assert hasattr(model, layer_name)

    return list(model.modules()).index(getattr(model, layer_name))


def load_clip_model(name, *args):
    """
    Serves to load some clip model.
    """
    model, _ = clip.load(name)  # clip.load('RN50x4')
    model = clip.model.build_model(model.state_dict())
    model = model.visual.float().eval()

    return model


def get_clip_transforms(name):
    """
    Used to get the corresponding transforms for a given clip model's name.
    We added ToPILImage to get it to work with our TTLDataset since we didn't use PIL to read images.
    """
    _, t = clip.load(name)
    t = transforms.Compose([transforms.ToPILImage(), t])
    return t