# Here goes our pre_processing transforms that we're going to apply to our dataset

import torchvision.transforms as transforms

pre_processing_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize(size=[224, 224])]
)
