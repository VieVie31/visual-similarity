import clip
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
from .processor import SaveProcessor, PCAProcessor, AdaptationProcessor, AdaptationProcessorPCA

class ValidateProcessor(argparse.Action):
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
    assert isinstance(model, nn.Module)
    assert isinstance(layer_name, str)
    assert hasattr(model, layer_name)

    return list(model.modules()).index(getattr(model, layer_name))


def load_clip_model(name, *args):
    model, _ = clip.load(name)  # clip.load('RN50x4')
    model = clip.model.build_model(model.state_dict())
    model = model.visual.float().eval()

    return model


def get_clip_transforms(name):
    _, t = clip.load(name)
    t = transforms.Compose([transforms.ToPILImage(), t])
    return t