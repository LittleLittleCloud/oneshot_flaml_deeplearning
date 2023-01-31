from src.dataset import *
from src.search import *
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler, SubsetRandomSampler
from src.task.task_helper import *
import json
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, type = int )
    parser.add_argument('--data', type = str )
    args = parser.parse_args()
    data = args.data
    dataset_group = DATASET_GROUP_0 + DATASET_GROUP_2
    dataset_group = filter(lambda x: data in x.__name__, dataset_group)

    device = torch.device(f'cuda:{args.device}')
    search_portfolio(dataset_group, device)
    evaluate_portfolio(dataset_group, device)
