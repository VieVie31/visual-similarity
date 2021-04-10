import torch
import torch.nn as nn
import torchvision.models
import timm

from facenet_pytorch import InceptionResnetV1

"""
    This is the file that will contain the models dictionnary that will be used to compute the feature maps.

    The use of lambda expressions is merely a trick to not load all the models at once when executing
    the script. Instead, we load just the lambda expressions (FunctionType), and then in the featureExtractor
    we'll loop over all these lambdas expressions and we'll load model by model.
    If we keep the previous version, we'll need to load all the models into memory first and then we can
    start extracting features, which is horrible if we have a lot of models (huge time is needed to 
    load all the models and we need a LOT of RAM), hence the use of this trick.

    Each value of this dictionnary must be a dict itself with 2 keys:
        model: the lambda expression that will be called to load the model.
        layers: List[int or str] ( the layers indices or names of which we want to extract
                                   the output of this model )
"""


def get_layer_index(model: nn.Module, layer_name: str):
    assert isinstance(model, nn.Module)
    assert isinstance(layer_name, str)
    assert hasattr(model, layer_name)

    return list(model.modules()).index(getattr(model, layer_name))


models_dict = dict()

# SUPERVISED

# RESNETs

resnet18 = lambda *args: torchvision.models.resnet18(pretrained=True, *args).eval()
models_dict["resnet18"] = {
    "model": resnet18,
    "layers": ["layer1", "layer2", "layer3", "layer4", -2, -3],
}

resnet34 = lambda *args: torchvision.models.resnet34(pretrained=True, *args).eval()
models_dict["resnet34"] = {
    "model": resnet34,
    "layers": ["layer1", "layer2", "layer3", "layer4"],
}

resnet50 = lambda *args: torchvision.models.resnet50(pretrained=True, *args).eval()
models_dict["resnet50"] = {
    "model": resnet50,
    "layers": ["layer1", "layer2", "layer3", "layer4"],
}

resnet101 = lambda *args: torchvision.models.resnet101(pretrained=True, *args).eval()
models_dict["resnet101"] = {
    "model": resnet101,
    "layers": ["layer1", "layer2", "layer3", "layer4"],
}

resnet152 = lambda *args: torchvision.models.resnet152(pretrained=True, *args).eval()
models_dict["resnet152"] = {
    "model": resnet152,
    "layers": ["layer1", "layer2", "layer3", "layer4"],
}

# Big Transfer
# pretrained on imagenet21k, finetuned on imagenet1k

models_dict["BiT-M-R50x1"] = {
    "model": lambda *args: timm.create_model(
        "resnetv2_50x1_bitm", pretrained=True, *args
    ).eval(),
    "layers": [-2],
}

models_dict["BiT-M-R50x3"] = {
    "model": lambda *args: timm.create_model(
        "resnetv2_50x3_bitm", pretrained=True, *args
    ).eval(),
    "layers": [-2],
}

models_dict["BiT-M-R101x1"] = {
    "model": lambda *args: timm.create_model(
        "resnetv2_101x1_bitm", pretrained=True, *args
    ).eval(),
    "layers": [-2],
}

models_dict["BiT-M-R101x3"] = {
    "model": lambda *args: timm.create_model(
        "resnetv2_101x3_bitm", pretrained=True, *args
    ).eval(),
    "layers": [-2],
}

models_dict["BiT-M-R152x2"] = {
    "model": lambda *args: timm.create_model(
        "resnetv2_152x2_bitm", pretrained=True, *args
    ).eval(),
    "layers": [-2],
}

models_dict["BiT-M-R152x4"] = {
    "model": lambda *args: timm.create_model(
        "resnetv2_152x4_bitm", pretrained=True, *args
    ).eval(),
    "layers": [-2],
}

# Big Transfer trained on imagenet-21k

models_dict["BiT-M-R50x1_in21k"] = {
    "model": lambda *args: timm.create_model(
        "resnetv2_50x1_bitm_in21k", pretrained=True, *args
    ).eval(),
    "layers": [-2],
}

models_dict["BiT-M-R50x3_in21k"] = {
    "model": lambda *args: timm.create_model(
        "resnetv2_50x3_bitm_in21k", pretrained=True, *args
    ).eval(),
    "layers": [-2],
}

models_dict["BiT-M-R101x1_in21k"] = {
    "model": lambda *args: timm.create_model(
        "resnetv2_101x1_bitm_in21k", pretrained=True, *args
    ).eval(),
    "layers": [-2],
}

models_dict["BiT-M-R101x3_in21k"] = {
    "model": lambda *args: timm.create_model(
        "resnetv2_101x3_bitm_in21k", pretrained=True, *args
    ).eval(),
    "layers": [-2],
}

models_dict["BiT-M-R152x2_in21k"] = {
    "model": lambda *args: timm.create_model(
        "resnetv2_152x2_bitm_in21k", pretrained=True, *args
    ).eval(),
    "layers": [-2],
}

models_dict["BiT-M-R152x4_in21k"] = {
    "model": lambda *args: timm.create_model(
        "resnetv2_152x4_bitm_in21k", pretrained=True, *args
    ).eval(),
    "layers": [-2],
}


models_dict["alexnet"] = {
    "model": lambda *args: torchvision.models.alexnet(pretrained=True, *args).eval(),
    "layers": [-2],
}


# other efficientnet versions : torch.hub.list('rwightman/gen-efficientnet-pytorch')
models_dict["efficientnet_b0"] = {
    "model": lambda *args: torch.hub.load(
        "rwightman/gen-efficientnet-pytorch", "efficientnet_b0", pretrained=True, *args
    ).eval(),
    "layers": [-2],
}

models_dict["facenet"] = {
    "model": lambda *args: InceptionResnetV1(pretrained="vggface2", *args).eval(),
    "layers": [-2],
}


# Vision Transformer
# all versions are available here: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
models_dict["ViT"] = {
    "model": lambda *args: timm.create_model(
        "vit_base_patch16_224", pretrained=True, *args
    ).eval(),
    "layers": [-2],
}


# SEMI-SUPERVISED

# facebook semi-supervised models: https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
models_dict["resnet50_swsl"] = {
    "model": lambda *args: torch.hub.load(
        "facebookresearch/semi-supervised-ImageNet1K-models", "resnet50_swsl", *args
    ).eval(),
    "layers": [-2],
}


# SELF-SUPERVISED

models_dict["barlow"] = {
    "model": lambda *args: torch.hub.load(
        "facebookresearch/barlowtwins:main", "resnet50", *args
    ).eval(),
    "layers": [-2],
}