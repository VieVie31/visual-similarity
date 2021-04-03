import torch
import torchvision.models

"""
    This is the file that will contain the models dictionnary that will be used to compute the feature maps.

    Each value of this dictionnary must be a dict itself with 2 keys:
        model: torch.nn.Module ( the model that we'll use to extract the features )
        layers: List[int] ( the layers indices of which we want to extract the output of this model )
"""

models_dict = dict()

# SUPERVISED

models_dict["resnet18"] = {
    "model": torchvision.models.resnet18(pretrained=True).eval(),
    "layers": [-2, -3],
}

models_dict["resnet34"] = {
    "model": torchvision.models.resnet34(pretrained=True).eval(),
    "layers": [-2],
}

# models_dict["resnet50"] = {
#     "model": torchvision.models.resnet50(pretrained=True).eval(),
#     "layers": [-2, -3],
# }

# models_dict["resnet101"] = {
#     "model": torchvision.models.resnet101(pretrained=True).eval(),
#     "layers": [-2],
# }

# models_dict["resnet152"] = {
#     "model": torchvision.models.resnet152(pretrained=True).eval(),
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

# from facenet_pytorch import InceptionResnetV1
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