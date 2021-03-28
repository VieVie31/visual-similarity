import torch
import torchvision.models

"""
    This is the file that will contain the models dictionnary that will be used to compute the feature maps.
"""

models_dict = dict()

models_dict['resnet18'] = torchvision.models.resnet18(pretrained=True)
#models_dict['alexnet'] = torchvision.models.alexnet(pretrained=True)
#models_dict['barlow'] = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')