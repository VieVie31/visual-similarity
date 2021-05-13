import my_utils
import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms as transforms
import timm

from facenet_pytorch import InceptionResnetV1

"""
    This is the file that will contain the models dictionnary that will be used to compute the feature maps.

    The use of lambda expressions is merely a trick to not load all the models at once when executing
    the script. Instead, we load just the lambda expressions (FunctionType), and then in the featureExtractor
    we'll loop over all these lambdas expressions and we'll load model by model.

    Each value of this dictionnary must be a dict itself with 2 or 3 keys:
        model: the lambda expression that will be called to load the model.
        layers: List[int or str] ( the layers indices or names of which we want to extract
                                   the output of this model )
        transform (optional): torchvision.transforms (the transforms that will be applied to the dataset)
                              if not present, we'll use the default one in transforms.py file.
"""
models_dict = dict()

# CLIP - OPENAI

clip_rn50x4 = lambda *args: my_utils.load_clip_model("RN50x4", *args)
models_dict["clip_rn50x4"] = {
    "model": clip_rn50x4,
    "layers": ["layer1", "layer2", "layer3", "layer4", "attnpool"],
    "transform": my_utils.get_clip_transforms("RN50x4"),
}

clip_rn50 = lambda *args: my_utils.load_clip_model("RN50", *args)
models_dict["clip_rn50"] = {
    "model": clip_rn50,
    "layers": ["layer1", "layer2", "layer3", "layer4", "attnpool"],
    "transform": my_utils.get_clip_transforms("RN50"),
}

clip_rn101 = lambda *args: my_utils.load_clip_model("RN101", *args)
models_dict["clip_rn101"] = {
    "model": clip_rn101,
    "layers": ["layer1", "layer2", "layer3", "layer4", "attnpool"],
    "transform": my_utils.get_clip_transforms("RN101"),
}

clip_vit32_b = lambda *args: my_utils.load_clip_model("ViT-B/32", *args)
models_dict["clip_vit32_b"] = {
    "model": clip_vit32_b,
    "layers": ["ln_post"],
    "transform": my_utils.get_clip_transforms("ViT-B/32"),
}


# SUPERVISED

# RESNETs

resnet18 = lambda *args: torchvision.models.resnet18(pretrained=True, *args).eval()
models_dict["resnet18"] = {
    "model": resnet18,
    "layers": ["layer1", "layer2", "layer3", "layer4"],
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

# # Big Transfer
# # pretrained on imagenet21k, finetuned on imagenet1k

#FIXME: pk les BiT-* sont avec juste des -2 et pas de layers intermédiaires ??!!

models_dict["BiT-M-R50x1"] = {
    "model": lambda *args: timm.create_model(
        "resnetv2_50x1_bitm", pretrained=True, *args
    ).eval(),
    "layers": [-2, -3],
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

# # Big Transfer trained on imagenet-21k

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



# ALEXNET
models_dict["alexnet"] = {
    "model": lambda *args: torchvision.models.alexnet(pretrained=True, *args).eval(),
    "layers": ["features"],
}



# # other efficientnet versions : torch.hub.list('rwightman/gen-efficientnet-pytorch')
models_dict["efficientnet_b0"] = {
    "model": lambda *args: torch.hub.load(
        "rwightman/gen-efficientnet-pytorch", "efficientnet_b0", pretrained=True, *args
    ).eval(),
    "layers": [-2],
}


# FACENET
models_dict["facenet"] = {
    "model": lambda *args: InceptionResnetV1(pretrained="vggface2", *args).eval(),
    "layers": ["block8"],
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
    "layers": ["layer1", "layer2", "layer3", "layer4"],
}

models_dict["resnext101_32x8d_swsl"] = {
    "model": lambda *args: torch.hub.load(
        "facebookresearch/semi-supervised-ImageNet1K-models", "resnext101_32x8d_swsl", *args
    ).eval(),
    "layers": ["layer1", "layer2", "layer3", "layer4"],
}

models_dict["resnext101_32x16d_swsl"] = {
    "model": lambda *args: torch.hub.load(
        "facebookresearch/semi-supervised-ImageNet1K-models", "resnext101_32x16d_swsl", *args
    ).eval(),
    "layers": ["layer1", "layer2", "layer3", "layer4"],
}




models_dict["resnet50_ssl"] = {
    "model": lambda *args: torch.hub.load(
        "facebookresearch/semi-supervised-ImageNet1K-models", "resnet50_ssl", *args
    ).eval(),
    "layers": ["layer1", "layer2", "layer3", "layer4"],
}

#TODO: ajouter les autres 101 32x8 et x16





# WEAKLY-SUPERVISED

# https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/
models_dict["resnext101_32x8d_wsl"] = {
    "model": lambda *args: torch.hub.load(
        "facebookresearch/WSL-Images", "resnext101_32x8d_wsl", *args
    ).eval(),
    "layers": ["layer1", "layer2", "layer3", "layer4"],
}

models_dict["resnext101_32x16d_wsl"] = {
    "model": lambda *args: torch.hub.load(
        "facebookresearch/WSL-Images", "resnext101_32x16d_wsl", *args
    ).eval(),
    "layers": ["layer1", "layer2", "layer3", "layer4"],
}

models_dict["resnext101_32x32d_wsl"] = {
    "model": lambda *args: torch.hub.load(
        "facebookresearch/WSL-Images", "resnext101_32x32d_wsl", *args
    ).eval(),
    "layers": ["layer1", "layer2", "layer3", "layer4"],
}

models_dict["resnext101_32x48d_wsl"] = {
    "model": lambda *args: torch.hub.load(
        "facebookresearch/WSL-Images", "resnext101_32x48d_wsl", *args
    ).eval(),
    "layers": ["layer1", "layer2", "layer3", "layer4"],
}



# SELF-SUPERVISED

models_dict["barlow"] = {
    "model": lambda *args: torch.hub.load(
        "facebookresearch/barlowtwins:main", "resnet50", *args
    ).eval(),
    "layers": ["layer1", "layer2", "layer3", "layer4"],
}

# Dino (SS)
models_dict["dino_resnet50"] = {
    "model": lambda *args: torch.hub.load(
        "facebookresearch/dino:main", "dino_resnet50", *args
    ).eval(),
    "layers": ["layer1", "layer2", "layer3", "layer4"],
}

models_dict["deits16"] = {
    "model": lambda *args: torch.hub.load(
        'facebookresearch/dino:main', 'dino_deits16', *args
    ).eval(),
    "layers": ["head"],
}

models_dict["deits8"] = {
    "model": lambda *args: torch.hub.load(
        'facebookresearch/dino:main', 'dino_deits8', *args
    ).eval(),
    "layers": ["head"],
}

models_dict["dino_vitb8"] = {
    "model": lambda *args: torch.hub.load(
        'facebookresearch/dino:main', 'dino_vitb8', *args
    ).eval(),
    "layers": ["head"],
}

models_dict["dino_vitb16"] = {
    "model": lambda *args: torch.hub.load(
        'facebookresearch/dino:main', 'dino_vitb16', *args
    ).eval(),
    "layers": ["head"],
}
