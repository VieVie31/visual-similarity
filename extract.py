import os
import argparse
from pre_processing import pre_processing_transforms
from featureExtractor import FeatureExtractor
from models import models_dict
from dataset import TTLDataset
from processor import SaveProcessor, PCAProcessor, AdaptationProcessor


class ValidateProcessor(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        processor = values[0].lower()

        if processor not in ("save", "pca", "adapt"):
            raise ValueError(f"invalid processor {values[0]}")

        if processor == "pca" or processor == "adapt":
            if len(values) > 2:
                raise ValueError(
                    f"Too much arguments ! pca must be followed only by 1 number (out dim)"
                )
            elif len(values) < 2:
                raise ValueError(f"Need out dim for the pca !")

            if processor == "pca":
                processor = PCAProcessor(namespace.save_to, int(values[1]))
            else:
                processor = AdaptationProcessor(namespace.save_to, int(values[1]))
        else:
            processor = SaveProcessor(namespace.save_to)

        setattr(namespace, self.dest, processor)


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
    help="Which processor to use, save or pca ?  (default: save)",
)

# layers indices are now specified in models_dict


def main():
    args = parser.parse_args()
    processor = args.processor

    extractor = FeatureExtractor(processor, models_dict)
    extractor.extract_features_from_directory(
        args.data,
        pre_processing_transforms,
        args.batch_size,
    )


if __name__ == "__main__":
    # example:
    # python extract.py -d ..\Desktop\TER\positive-similarity\data\ -s ..\Desktop\saved_features -b 64 -p adapt 256
    main()