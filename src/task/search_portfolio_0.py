from src.dataset import *
from src.search import *
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler, SubsetRandomSampler
from src.task.task_helper import *
import json
import argparse

# FashionMNist get
# Flower102 get
# Aircraft get
# DTD get
# Omniglot get


# xiaoyun4:
# python -m src.task.search_portfolio_0 --device 0 --data Food101

# xiaoyun3:
# python -m src.task.search_portfolio_0 --device 0 --data country211 KMNIST

# xiaoyun1:
# python -m src.task.search_portfolio_0 --device 0 --data GTSRB
# python -m src.task.search_portfolio_0 --device 1 --data LFWPeople
# python -m src.task.search_portfolio_0 --device 2 --data Omniglot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, type = int )
    parser.add_argument('--data', type = str, action='extend', nargs='+' )
    parser.add_argument('--subsample', type = float, default=-1 )
    args = parser.parse_args()
    datas = args.data
    SUB_SAMPLE = args.subsample
    datas = [f'load_{data}' for data in datas]
    print(datas)
    dataset_group = DATASET_GROUP_0 + DATASET_GROUP_2
    dataset_group = filter(lambda x: x.__name__ in datas, dataset_group)

    device = torch.device(f'cuda:{args.device}')
    search_portfolio(dataset_group, device)
    evaluate_portfolio(dataset_group, device)
