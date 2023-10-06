"""Train UCI dataset classifier."""
import argparse
import math
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import numpy as np
import random
import torch

from exact_pytorch import EXACTLoss, GradientNormalizer
from uci_class import UCI_DATASETS, split_crossval


def parse_arguments():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("dataset", choices=["all"] + list(UCI_DATASETS))
    parser.add_argument("--root", help="Datasets root.")
    args = parser.parse_args()
    return args


def print_info(root, name):
    dataset = UCI_DATASETS[name](root, split="train")
    print("=" * (5 + len(name)))
    print("Name", name)
    print("=" * (5 + len(name)))
    print("Size", len(dataset))
    print("Classes", len(set(dataset.y.tolist())))
    print("")


if __name__ == "__main__":
    args = parse_arguments()
    if args.dataset == "all":
        for dataset in UCI_DATASETS:
            print_info(args.root, dataset)
    else:
        print_info(args.root, args.dataset)
