# Here goes our pre_processing transforms that we're going to apply to our dataset

import torchvision.transforms as transforms

# taken from https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

pre_processing_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize(size=[224, 224]), normalize]
)

data_aug_transform = transforms.Compose(
    [pre_processing_transforms, transforms.RandomHorizontalFlip(p=1)]
)

aug_target_transform = lambda label : 'augmented_' + label