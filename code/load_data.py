import os
import sys
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dataset import Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #Data/Encoder
    parser.add_argument("--dataset",  default='peerread')
    parser.add_argument("--dataset_type", default='all')
    parser.add_argument("--emb", default='bert')
    parser.add_argument("--bert_encode_type", default='last')
    parser.add_argument("--path", default='../../../data/neuralsum/v2')
    parser.add_argument("--encoder_path", default='../../../data/word2vec')

    args = parser.parse_args()

    # load dataset
    data = Dataset(dataset=args.dataset, dataset_type=args.dataset_type, path=args.path)
    pairs = data.load_dataset()





