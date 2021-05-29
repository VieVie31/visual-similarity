# Here goes our pre_processing transforms that we're going to apply to our dataset
# These are the default ones in case there is no tranforms defined in the models dict

import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

pre_processing_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize(size=[224, 224]), normalize]
)

data_aug_transform = transforms.RandomHorizontalFlip(p=1)
target_aug_transform = lambda label: "augmented_" + label