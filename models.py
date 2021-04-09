import torch
import torch.nn as nn
import torchvision.models
import timm

from facenet_pytorch import InceptionResnetV1

"""
    This is the file that will contain the models dictionnary that will be used to compute the feature maps.

    Each value of this dictionnary must be a dict itself with 2 keys:
        model: torch.nn.Module ( the model that we'll use to extract the features )
        layers: List[int] ( the layers indices of which we want to extract the output of this model )
"""


def get_layer_index(model: nn.Module, layer_name: str):
    assert isinstance(model, nn.Module)
    assert isinstance(layer_name, str)
    assert hasattr(model, layer_name)

    return list(model.modules()).index(getattr(model, layer_name))


models_dict = dict()

# SUPERVISED

# RESNETs

resnet18 = torchvision.models.resnet18(pretrained=True).eval()
models_dict["resnet18"] = {
    "model": resnet18,
    "layers": [
        get_layer_index(resnet18, "layer1"),
        get_layer_index(resnet18, "layer2"),
        get_layer_index(resnet18, "layer3"),
        get_layer_index(resnet18, "layer4"),
    ],
}

# resnet34 = torchvision.models.resnet34(pretrained=True).eval()
# models_dict["resnet34"] = {
#     "model": torchvision.models.resnet34(pretrained=True).eval(),
#     "layers": [
#         get_layer_index(resnet34, "layer1"),
#         get_layer_index(resnet34, "layer2"),
#         get_layer_index(resnet34, "layer3"),
#         get_layer_index(resnet34, "layer4"),
#     ],
# }

# resnet50 = torchvision.models.resnet50(pretrained=True).eval()
# models_dict["resnet50"] = {
#     "model": torchvision.models.resnet50(pretrained=True).eval(),
#     "layers": [
#         get_layer_index(resnet50, "layer1"),
#         get_layer_index(resnet50, "layer2"),
#         get_layer_index(resnet50, "layer3"),
#         get_layer_index(resnet50, "layer4"),
#     ],
# }

# models_dict["resnet101"] = {
#     "model": torchvision.models.resnet101(pretrained=True).eval(),
#     "layers": [-2],
# }

# models_dict["resnet152"] = {
#     "model": torchvision.models.resnet152(pretrained=True).eval(),
#     "layers": [-2],
# }


# Big Transfer
# pretrained on imagenet21k, finetuned on imagenet1k

# models_dict["BiT-M-R50x1"] = {
#     "model": timm.create_model("resnetv2_50x1_bitm", pretrained=True).eval(),
#     "layers": [-2],
# }

# models_dict["BiT-M-R50x3"] = {
#     "model": timm.create_model('resnetv2_50x3_bitm', pretrained=True).eval(),
#     "layers": [-2],
# }

# models_dict["BiT-M-R101x1"] = {
#     "model": timm.create_model('resnetv2_101x1_bitm', pretrained=True).eval(),
#     "layers": [-2],
# }

# models_dict["BiT-M-R101x3"] = {
#     "model": timm.create_model('resnetv2_101x3_bitm', pretrained=True).eval(),
#     "layers": [-2],
# }

# models_dict["BiT-M-R152x2"] = {
#     "model": timm.create_model('resnetv2_152x2_bitm', pretrained=True).eval(),
#     "layers": [-2],
# }

# models_dict["BiT-M-R152x4"] = {
#     "model": timm.create_model('resnetv2_152x4_bitm', pretrained=True).eval(),
#     "layers": [-2],
# }

# Big Transfer trained on imagenet-21k

# models_dict["BiT-M-R50x1_in21k"] = {
#     "model": timm.create_model('resnetv2_50x1_bitm_in21k', pretrained=True).eval(),
#     "layers": [-2],
# }

# models_dict["BiT-M-R50x3_in21k"] = {
#     "model": timm.create_model('resnetv2_50x3_bitm_in21k', pretrained=True).eval(),
#     "layers": [-2],
# }

# models_dict["BiT-M-R101x1_in21k"] = {
#     "model": timm.create_model('resnetv2_101x1_bitm_in21k', pretrained=True).eval(),
#     "layers": [-2],
# }

# models_dict["BiT-M-R101x3_in21k"] = {
#     "model": timm.create_model('resnetv2_101x3_bitm_in21k', pretrained=True).eval(),
#     "layers": [-2],
# }

# models_dict["BiT-M-R152x2_in21k"] = {
#     "model": timm.create_model('resnetv2_152x2_bitm_in21k', pretrained=True).eval(),
#     "layers": [-2],
# }

# models_dict["BiT-M-R152x4_in21k"] = {
#     "model": timm.create_model('resnetv2_152x4_bitm_in21k', pretrained=True).eval(),
#     "layers": [-2],
# }


# models_dict["alexnet"] = {
#     "model": torchvision.models.alexnet(pretrained=True).eval(),
#     "layers": [-2],
# }

# other efficientnet versions : torch.hub.list('rwightman/gen-efficientnet-pytorch')
# models_dict["efficientnet_b0"] = {
#     "model": torch.hub.load(
#         "rwightman/gen-efficientnet-pytorch", "efficientnet_b0", pretrained=True
#     ).eval(),
#     "layers": [-2],
# }

# models_dict["facenet"] = {
#     "model": InceptionResnetV1(pretrained="vggface2").eval(),
#     "layers": [-2],
# }


# SEMI-SUPERVISED

# facebook semi-supervised models: https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
# models_dict["resnet50_swsl"] = {
#     "model": torch.hub.load(
#         "facebookresearch/semi-supervised-ImageNet1K-models", "resnet50_swsl"
#     ).eval(),
#     "layers": [-2],
# }


# SELF-SUPERVISED

# models_dict["barlow"] = {
#     "model": torch.hub.load("facebookresearch/barlowtwins:main", "resnet50").eval(),
#     "layers": [-2],
# }