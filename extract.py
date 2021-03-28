import argparse
import torchvision.transforms as transforms
from featureExtractor import FeatureExtractor
from models import models_dict
from dataset import TTLDataset


parser = argparse.ArgumentParser(description='Feature maps extractor')

parser.add_argument('-d', '--data', required=True, type=str, metavar='DIR', 
                    help='Path to TTL dataset')
parser.add_argument('-s', '--save-to', required=True, type=str, metavar='DIR',
                    help='Path to which we are going to save the calculated features')
parser.add_argument('-l', '--layers', type=int, nargs='+', metavar='N', default=[-2], 
                    help='layers_indices:  Which layers we are going to use to extract the features. (default: -2)')

def main():
    args = parser.parse_args()
    
    print(args.save_to)
    ttl = TTLDataset(args.data, transform=transforms.ToTensor())
    extractor = FeatureExtractor(args.save_to, models_dict, args.layers)

    extractor.extract_features_ttl_dataset(ttl)

if __name__ == '__main__':
    # example: 
    # python extract.py -d ..\Desktop\TER\positive-similarity\data\ -s ..\Desktop\saved_features_ttl -l -2 -3
    main()