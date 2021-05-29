"""
Script that we can use with command line arguments to extract the feature maps of the images located
in some directory.

usage: extract.py [-h] -d DIR -s DIR [-b N] [-p [PROCESSOR ...]] [--data-aug | --no-data-aug]

Feature maps extractor

optional arguments:
  -h, --help            show this help message and exit
  -d DIR, --data DIR    Path to TTL dataset
  -s DIR, --save-to DIR
                        Path to which we are going to save the calculated features
  -b N, --batch-size N  Batch size
  -p [PROCESSOR ...], --processor [PROCESSOR ...]
                        Which processor to use ? (default: save)
  --data-aug
  --no-data-aug
"""

import argparse
from featuresExtractor import models_dict
from featuresExtractor import FeatureExtractor
from featuresExtractor import ValidateProcessor

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
    "-b",
    "--batch-size",
    type=int,
    metavar="N",
    default=1,
    help="Batch size",
)
parser.add_argument(
    "-p",
    "--processor",
    nargs="*",
    action=ValidateProcessor,
    default="save",
    help="Which processor to use ?  (default: save)",
)

augment_parser = parser.add_mutually_exclusive_group(required=False)
augment_parser.add_argument("--data-aug", dest="augment", action="store_true")
augment_parser.add_argument("--no-data-aug", dest="augment", action="store_false")
parser.set_defaults(augment=False)


def main():
    args = parser.parse_args()
    processor = args.processor

    extractor = FeatureExtractor(processor, models_dict)
    extractor.extract_features_from_directory(args.data, args.batch_size, args.augment)


if __name__ == "__main__":
    # example:
    # python extract.py -d ..\Desktop\TER\positive-similarity\data\ -s ..\Desktop\saved_features -b 64 -p adapt 256
    # python extract.py -d ../data_original -s ../saved_features -b 512 -p adapt 256 --data-aug
    
    # python extract.py -d TTLDataset_PATH -s WHERE_TO_SAVE_CALCULATED_FEATURES -b BATCH_SIZE [--data-aug | --no-data-aug] -p PROCESSOR [with args] 
    main()