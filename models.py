import torch
import torchvision.models

"""
    This is the file that will contain the models dictionnary that will be used to compute the feature maps.
"""

models_dict = dict()

# SUPERVISED

models_dict["resnet18"] = torchvision.models.resnet18(pretrained=True)
# models_dict['resnet34'] = torchvision.models.resnet34(pretrained=True)
# models_dict['resnet50'] = torchvision.models.resnet50(pretrained=True)
# models_dict['resnet101'] = torchvision.models.resnet101(pretrained=True)
# models_dict['resnet152'] = torchvision.models.resnet152(pretrained=True)
# models_dict['alexnet'] = torchvision.models.alexnet(pretrained=True)

# other efficientnet versions : torch.hub.list('rwightman/gen-efficientnet-pytorch')
# models_dict['efficientnet_b0'] = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)

# from facenet_pytorch import InceptionResnetV1
# models_dict['facenet'] = InceptionResnetV1(pretrained='vggface2').eval()

# SEMI-SUPERVISED

# facebook semi-supervised models: https://github.com/facebookresearch/semi-supervised-ImageNet1K-models
# models_dict['resnet50_swsl'] = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')


# SELF-SUPERVISED

# models_dict['barlow'] = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')