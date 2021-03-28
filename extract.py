import os
import argparse
import torchvision.transforms as transforms
from featureExtractor import FeatureExtractor
from models import models_dict
from dataset import TTLDataset
from processor import SaveProcessor, PCAProcessor


parser = argparse.ArgumentParser(description="Feature maps extractor")

parser.add_argument(
    "-d", "--data", required=True, type=str, metavar="DIR", help="Path to TTL dataset"
)
parser.add_argument(
    "-s",
    "--save-to",
    required=True,
    type=str,
    metavar="DIR",
    help="Path to which we are going to save the calculated features",
)
parser.add_argument(
    "-p",
    "--processor",
    required=True,
    type=str,
    default="save",
    help="Which processor to use, save or pca ?  (default: save)",
)
parser.add_argument(
    "-l",
    "--layers",
    type=int,
    nargs="+",
    metavar="N",
    default=[-2],
    help="layers_indices:  Which layers we are going to use to extract the features. (default: -2)",
)


def main():
    args = parser.parse_args()

    processor = None  # TODO: figure out a way to dynamically choose which Processor with Identity()

    if args.processor == "pca":
        processor = PCAProcessor(args.save_to, 256)
    else:
        if args.processor != "save":
            print("Unsupported processor, will use SaveProcessor !")

        processor = SaveProcessor(args.save_to)

    extractor = FeatureExtractor(processor, models_dict, args.layers)

    _, dirs, _ = next(os.walk(args.data))

    if len(dirs) == 0:
        extractor.extract_features_from_directory(args.data)
    else:
        print(
            "The path you gave contains subdirectories, will assume it's a TTL like dataset."
        )
        ttl = TTLDataset(args.data, transform=transforms.ToTensor())
        extractor.extract_features_ttl_dataset(ttl)

    processor.execute()  # do what he's supposed to do with the extracted features


if __name__ == "__main__":
    # example:
    # python extract.py -d ..\Desktop\TER\positive-similarity\data\ -s ..\Desktop\saved_features -p pca -l -2 -3
    main()